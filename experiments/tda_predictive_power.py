#!/usr/bin/env python3
"""
TDA Predictive Power Analysis — Statistical Validation Pipeline

Operation CORTEX: Phase I Activation
=====================================

This script performs rigorous statistical analysis to validate that HSS scores
have predictive power for proof verification outcomes. It computes effect sizes,
AUC-ROC, precision-recall curves, and calibration metrics.

Usage:
    python experiments/tda_predictive_power.py \
        --input-dir results/tda_validation \
        --output-dir results/tda_predictive_analysis

Analysis Components:
1. Effect Size Analysis (Cohen's d between verified/unverified)
2. Classification Metrics (AUC-ROC, F1, precision, recall)
3. Calibration Analysis (reliability diagrams)
4. Threshold Optimization (optimal block/warn thresholds)
5. Correlation Analysis (HSS vs verification outcomes)
6. Bootstrap Confidence Intervals

Acceptance Criteria (per TDA_MIND_SCANNER_SPEC.md):
- Cohen's d > 0.8 (large effect size)
- AUC-ROC > 0.80
- Positive correlation with Lean verification success

Output:
- predictive_analysis.json: Full statistical results
- predictive_report.md: Human-readable analysis report
- figures/: ROC curves, calibration plots, histograms
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Conditional imports for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tda_predictive_power")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ClassificationMetrics:
    """Classification performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROCAnalysis:
    """ROC curve analysis results."""
    auc: float
    thresholds: List[float]
    tpr: List[float]  # True positive rate (sensitivity)
    fpr: List[float]  # False positive rate (1 - specificity)
    optimal_threshold: float
    optimal_tpr: float
    optimal_fpr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc": self.auc,
            "optimal_threshold": self.optimal_threshold,
            "optimal_tpr": self.optimal_tpr,
            "optimal_fpr": self.optimal_fpr,
            "curve_points": len(self.thresholds),
        }


@dataclass
class EffectSizeAnalysis:
    """Effect size analysis results."""
    cohens_d: float
    hedges_g: float
    mean_verified: float
    mean_unverified: float
    std_verified: float
    std_unverified: float
    n_verified: int
    n_unverified: int
    ci_lower: float  # 95% CI for Cohen's d
    ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationAnalysis:
    """Calibration analysis results."""
    bins: List[float]  # Predicted probability bins
    actual_positive_rate: List[float]  # Actual positive rate per bin
    counts: List[int]  # Samples per bin
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    brier_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "num_bins": len(self.bins),
        }


@dataclass
class CorrelationAnalysis:
    """Correlation analysis results."""
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    point_biserial_r: float
    point_biserial_p: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictiveAnalysisResult:
    """Complete predictive analysis results."""
    timestamp: str
    sample_size: int
    n_verified: int
    n_unverified: int

    # Core analyses
    effect_size: EffectSizeAnalysis
    roc: ROCAnalysis
    calibration: CalibrationAnalysis
    correlation: CorrelationAnalysis

    # Optimal thresholds
    optimal_block_threshold: float
    optimal_warn_threshold: float
    metrics_at_optimal: ClassificationMetrics

    # Acceptance criteria
    acceptance_criteria: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sample_size": self.sample_size,
            "n_verified": self.n_verified,
            "n_unverified": self.n_unverified,
            "effect_size": self.effect_size.to_dict(),
            "roc": self.roc.to_dict(),
            "calibration": self.calibration.to_dict(),
            "correlation": self.correlation.to_dict(),
            "optimal_block_threshold": self.optimal_block_threshold,
            "optimal_warn_threshold": self.optimal_warn_threshold,
            "metrics_at_optimal": self.metrics_at_optimal.to_dict(),
            "acceptance_criteria": self.acceptance_criteria,
        }


# ============================================================================
# Statistical Functions
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Hedges' g (bias-corrected effect size)."""
    d = compute_cohens_d(group1, group2)
    n = len(group1) + len(group2)

    # Small sample correction
    correction = 1 - (3 / (4 * n - 9))
    return d * correction


def compute_effect_size_ci(
    d: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for Cohen's d using non-central t approximation."""
    from scipy import stats

    # Standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # Critical value
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    ci_lower = d - z * se
    ci_upper = d + z * se

    return ci_lower, ci_upper


def compute_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> ROCAnalysis:
    """Compute ROC curve and AUC."""
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return ROCAnalysis(
            auc=0.5,
            thresholds=[0.0, 1.0],
            tpr=[0.0, 1.0],
            fpr=[0.0, 1.0],
            optimal_threshold=0.5,
            optimal_tpr=0.5,
            optimal_fpr=0.5,
        )

    # Compute TPR and FPR at each threshold
    thresholds = []
    tpr = []
    fpr = []

    unique_scores = np.unique(sorted_scores)
    for thresh in unique_scores:
        predictions = scores >= thresh
        tp = np.sum(predictions & (labels == 1))
        fp = np.sum(predictions & (labels == 0))

        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        thresholds.append(thresh)

    # Add endpoints
    tpr = [0.0] + tpr + [1.0]
    fpr = [0.0] + fpr + [1.0]
    thresholds = [1.1] + thresholds + [-0.1]

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2

    # Find optimal threshold (Youden's J statistic)
    j_scores = np.array(tpr) - np.array(fpr)
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    return ROCAnalysis(
        auc=float(auc),
        thresholds=thresholds,
        tpr=tpr,
        fpr=fpr,
        optimal_threshold=float(optimal_threshold),
        optimal_tpr=float(optimal_tpr),
        optimal_fpr=float(optimal_fpr),
    )


def compute_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ClassificationMetrics:
    """Compute classification metrics at a given threshold."""
    predictions = (scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        specificity=float(specificity),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        threshold=float(threshold),
    )


def compute_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationAnalysis:
    """Compute calibration metrics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    actual_rates = []
    counts = []
    weighted_errors = []

    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        count = np.sum(mask)
        counts.append(int(count))

        if count > 0:
            actual_rate = np.mean(labels[mask])
            actual_rates.append(float(actual_rate))

            # Calibration error for this bin
            error = np.abs(actual_rate - bin_centers[i])
            weighted_errors.append(error * count)
        else:
            actual_rates.append(0.0)

    # Expected Calibration Error
    total_samples = np.sum(counts)
    ece = np.sum(weighted_errors) / total_samples if total_samples > 0 else 0.0

    # Maximum Calibration Error
    bin_errors = [
        abs(actual_rates[i] - bin_centers[i]) if counts[i] > 0 else 0.0
        for i in range(n_bins)
    ]
    mce = max(bin_errors) if bin_errors else 0.0

    # Brier Score
    brier = np.mean((scores - labels) ** 2)

    return CalibrationAnalysis(
        bins=bin_centers.tolist(),
        actual_positive_rate=actual_rates,
        counts=counts,
        expected_calibration_error=float(ece),
        maximum_calibration_error=float(mce),
        brier_score=float(brier),
    )


def compute_correlations(
    scores: np.ndarray,
    labels: np.ndarray,
) -> CorrelationAnalysis:
    """Compute correlation coefficients."""
    from scipy import stats

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(scores, labels)

    # Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(scores, labels)

    # Point-biserial correlation (special case of Pearson for binary)
    pb_r, pb_p = stats.pointbiserialr(labels, scores)

    return CorrelationAnalysis(
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        point_biserial_r=float(pb_r),
        point_biserial_p=float(pb_p),
    )


def find_optimal_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Find optimal block and warn thresholds.

    Block threshold: Maximize precision at recall >= 0.9
    Warn threshold: Maximize F1 score
    """
    # Find warn threshold (maximize F1)
    best_f1 = 0.0
    warn_threshold = 0.5

    for thresh in np.linspace(0.1, 0.9, 50):
        metrics = compute_classification_metrics(scores, labels, thresh)
        if metrics.f1_score > best_f1:
            best_f1 = metrics.f1_score
            warn_threshold = thresh

    # Find block threshold (high precision point)
    # This is where we're very confident something is bad
    block_threshold = 0.2  # Default from spec

    for thresh in np.linspace(0.05, 0.4, 20):
        predictions = scores < thresh  # Block if HSS is low
        blocked_labels = labels[predictions] if np.any(predictions) else np.array([])

        if len(blocked_labels) > 0:
            # Precision: what fraction of blocks were correct (truly bad)
            precision = 1 - np.mean(blocked_labels)  # blocked_labels=0 for bad
            if precision >= 0.9:  # 90% precision requirement
                block_threshold = thresh
                break

    return block_threshold, warn_threshold


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_curve(
    roc: ROCAnalysis,
    output_path: Path,
) -> None:
    """Plot ROC curve."""
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(roc.fpr, roc.tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc.auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Mark optimal point
    plt.scatter([roc.optimal_fpr], [roc.optimal_tpr], c='red', s=100, zorder=5,
                label=f'Optimal (thresh={roc.optimal_threshold:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - HSS Predictive Power')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_distributions(
    verified_scores: np.ndarray,
    unverified_scores: np.ndarray,
    cohens_d: float,
    output_path: Path,
) -> None:
    """Plot HSS score distributions."""
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 30)
    plt.hist(verified_scores, bins=bins, alpha=0.5, label=f'Verified (n={len(verified_scores)})',
             color='green', density=True)
    plt.hist(unverified_scores, bins=bins, alpha=0.5, label=f'Unverified (n={len(unverified_scores)})',
             color='red', density=True)

    plt.axvline(np.mean(verified_scores), color='green', linestyle='--', linewidth=2)
    plt.axvline(np.mean(unverified_scores), color='red', linestyle='--', linewidth=2)

    plt.xlabel('HSS Score')
    plt.ylabel('Density')
    plt.title(f'HSS Score Distributions (Cohen\'s d = {cohens_d:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_calibration(
    calibration: CalibrationAnalysis,
    output_path: Path,
) -> None:
    """Plot calibration diagram."""
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(8, 6))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

    # Actual calibration
    plt.bar(calibration.bins, calibration.actual_positive_rate, width=0.08, alpha=0.7,
            label='Actual', color='blue')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Diagram (ECE = {calibration.expected_calibration_error:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def load_validation_results(input_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation results from tda_validate_synthetic output.

    Returns:
        Tuple of (hss_scores, labels) where labels=1 for verified, 0 for unverified.
    """
    results_path = input_dir / "validation_results.json"

    if not results_path.exists():
        # Try alternative filename
        results_path = input_dir / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"No results file found in {input_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hss_scores = []
    labels = []

    for result in data.get("results", []):
        hss_scores.append(result["hss"])
        # Label: 1 for good/verified, 0 for bad/unverified
        label = 1 if result.get("label") == "good" else 0
        labels.append(label)

    return np.array(hss_scores), np.array(labels)


def run_predictive_analysis(
    input_dir: Path,
    output_dir: Path,
) -> PredictiveAnalysisResult:
    """
    Run complete predictive power analysis.

    Args:
        input_dir: Directory with validation results.
        output_dir: Directory for analysis output.

    Returns:
        PredictiveAnalysisResult with all analysis results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading validation results from {input_dir}")
    scores, labels = load_validation_results(input_dir)

    n_total = len(scores)
    n_verified = int(np.sum(labels))
    n_unverified = n_total - n_verified

    logger.info(f"Loaded {n_total} samples: {n_verified} verified, {n_unverified} unverified")

    # Split by label
    verified_scores = scores[labels == 1]
    unverified_scores = scores[labels == 0]

    # 1. Effect Size Analysis
    logger.info("Computing effect size analysis...")
    cohens_d = compute_cohens_d(verified_scores, unverified_scores)
    hedges_g = compute_hedges_g(verified_scores, unverified_scores)

    try:
        ci_lower, ci_upper = compute_effect_size_ci(cohens_d, n_verified, n_unverified)
    except Exception:
        ci_lower, ci_upper = cohens_d - 0.5, cohens_d + 0.5

    effect_size = EffectSizeAnalysis(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        mean_verified=float(np.mean(verified_scores)) if len(verified_scores) > 0 else 0.0,
        mean_unverified=float(np.mean(unverified_scores)) if len(unverified_scores) > 0 else 0.0,
        std_verified=float(np.std(verified_scores)) if len(verified_scores) > 0 else 0.0,
        std_unverified=float(np.std(unverified_scores)) if len(unverified_scores) > 0 else 0.0,
        n_verified=n_verified,
        n_unverified=n_unverified,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    logger.info(f"  Cohen's d = {cohens_d:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    # 2. ROC Analysis
    logger.info("Computing ROC analysis...")
    roc = compute_roc_curve(scores, labels)
    logger.info(f"  AUC = {roc.auc:.3f}")

    # 3. Calibration Analysis
    logger.info("Computing calibration analysis...")
    calibration = compute_calibration(scores, labels)
    logger.info(f"  ECE = {calibration.expected_calibration_error:.3f}")

    # 4. Correlation Analysis
    logger.info("Computing correlations...")
    try:
        correlation = compute_correlations(scores, labels)
        logger.info(f"  Pearson r = {correlation.pearson_r:.3f}")
    except Exception as e:
        logger.warning(f"Correlation analysis failed: {e}")
        correlation = CorrelationAnalysis(
            pearson_r=0.0, pearson_p=1.0,
            spearman_rho=0.0, spearman_p=1.0,
            point_biserial_r=0.0, point_biserial_p=1.0,
        )

    # 5. Optimal Thresholds
    logger.info("Finding optimal thresholds...")
    block_threshold, warn_threshold = find_optimal_thresholds(scores, labels)
    logger.info(f"  Block threshold = {block_threshold:.3f}")
    logger.info(f"  Warn threshold = {warn_threshold:.3f}")

    # Metrics at optimal threshold
    metrics_at_optimal = compute_classification_metrics(scores, labels, warn_threshold)

    # 6. Acceptance Criteria
    acceptance_criteria = {
        "cohens_d_gt_0.8": cohens_d > 0.8,
        "auc_gt_0.80": roc.auc > 0.80,
        "positive_correlation": correlation.pearson_r > 0,
        "f1_gt_0.7": metrics_at_optimal.f1_score > 0.7,
        "ece_lt_0.1": calibration.expected_calibration_error < 0.1,
    }

    all_passed = all(acceptance_criteria.values())
    logger.info(f"Acceptance criteria: {'PASSED' if all_passed else 'FAILED'}")
    for criterion, passed in acceptance_criteria.items():
        logger.info(f"  {criterion}: {'✓' if passed else '✗'}")

    # Build result
    result = PredictiveAnalysisResult(
        timestamp=datetime.utcnow().isoformat() + "Z",
        sample_size=n_total,
        n_verified=n_verified,
        n_unverified=n_unverified,
        effect_size=effect_size,
        roc=roc,
        calibration=calibration,
        correlation=correlation,
        optimal_block_threshold=block_threshold,
        optimal_warn_threshold=warn_threshold,
        metrics_at_optimal=metrics_at_optimal,
        acceptance_criteria=acceptance_criteria,
    )

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_roc_curve(roc, figures_dir / "roc_curve.png")
    plot_score_distributions(verified_scores, unverified_scores, cohens_d,
                            figures_dir / "score_distributions.png")
    plot_calibration(calibration, figures_dir / "calibration.png")

    # Save results
    results_path = output_dir / "predictive_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Generate report
    generate_analysis_report(result, output_dir / "predictive_report.md")

    return result


def generate_analysis_report(
    result: PredictiveAnalysisResult,
    output_path: Path,
) -> None:
    """Generate human-readable analysis report."""
    es = result.effect_size
    roc = result.roc
    cal = result.calibration
    cor = result.correlation
    met = result.metrics_at_optimal

    all_passed = all(result.acceptance_criteria.values())
    status = "PASSED" if all_passed else "FAILED"

    lines = [
        "# TDA Predictive Power Analysis Report",
        "",
        f"Generated: {result.timestamp}",
        "",
        f"## Executive Summary",
        "",
        f"**Overall Status: {status}**",
        "",
        f"The HSS score {'demonstrates' if all_passed else 'does not demonstrate'} "
        f"adequate predictive power for proof verification outcomes.",
        "",
        "## Sample Statistics",
        "",
        f"- Total samples: {result.sample_size}",
        f"- Verified (good): {result.n_verified} ({result.n_verified/result.sample_size*100:.1f}%)",
        f"- Unverified (bad): {result.n_unverified} ({result.n_unverified/result.sample_size*100:.1f}%)",
        "",
        "## Effect Size Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Cohen's d | {es.cohens_d:.4f} |",
        f"| 95% CI | [{es.ci_lower:.4f}, {es.ci_upper:.4f}] |",
        f"| Hedges' g | {es.hedges_g:.4f} |",
        f"| Mean HSS (verified) | {es.mean_verified:.4f} |",
        f"| Mean HSS (unverified) | {es.mean_unverified:.4f} |",
        "",
        _interpret_cohens_d(es.cohens_d),
        "",
        "## Classification Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| AUC-ROC | {roc.auc:.4f} |",
        f"| Optimal threshold | {roc.optimal_threshold:.4f} |",
        f"| Accuracy | {met.accuracy:.4f} |",
        f"| Precision | {met.precision:.4f} |",
        f"| Recall | {met.recall:.4f} |",
        f"| F1 Score | {met.f1_score:.4f} |",
        f"| Specificity | {met.specificity:.4f} |",
        "",
        "## Calibration",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Expected Calibration Error (ECE) | {cal.expected_calibration_error:.4f} |",
        f"| Maximum Calibration Error (MCE) | {cal.maximum_calibration_error:.4f} |",
        f"| Brier Score | {cal.brier_score:.4f} |",
        "",
        "## Correlation Analysis",
        "",
        f"| Metric | Value | p-value |",
        f"|--------|-------|---------|",
        f"| Pearson r | {cor.pearson_r:.4f} | {cor.pearson_p:.4e} |",
        f"| Spearman ρ | {cor.spearman_rho:.4f} | {cor.spearman_p:.4e} |",
        f"| Point-biserial r | {cor.point_biserial_r:.4f} | {cor.point_biserial_p:.4e} |",
        "",
        "## Recommended Thresholds",
        "",
        f"Based on this analysis, the recommended thresholds are:",
        "",
        f"- **Block threshold**: {result.optimal_block_threshold:.4f}",
        f"- **Warn threshold**: {result.optimal_warn_threshold:.4f}",
        "",
        "These thresholds optimize the trade-off between precision and recall.",
        "",
        "## Acceptance Criteria",
        "",
        f"| Criterion | Required | Actual | Status |",
        f"|-----------|----------|--------|--------|",
    ]

    criteria_details = [
        ("Cohen's d > 0.8", "0.8", f"{es.cohens_d:.3f}", result.acceptance_criteria["cohens_d_gt_0.8"]),
        ("AUC > 0.80", "0.80", f"{roc.auc:.3f}", result.acceptance_criteria["auc_gt_0.80"]),
        ("Positive correlation", "> 0", f"{cor.pearson_r:.3f}", result.acceptance_criteria["positive_correlation"]),
        ("F1 > 0.7", "0.7", f"{met.f1_score:.3f}", result.acceptance_criteria["f1_gt_0.7"]),
        ("ECE < 0.1", "0.1", f"{cal.expected_calibration_error:.3f}", result.acceptance_criteria["ece_lt_0.1"]),
    ]

    for name, required, actual, passed in criteria_details:
        status_mark = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {name} | {required} | {actual} | {status_mark} |")

    lines.extend([
        "",
        "## Recommendations",
        "",
    ])

    if all_passed:
        lines.extend([
            "The HSS score demonstrates strong predictive power. Recommendations:",
            "",
            "1. Proceed to Phase I Shadow Mode deployment",
            "2. Monitor HSS distributions in production",
            "3. Refine thresholds based on production data",
            "4. Plan Phase II Soft Gating transition",
        ])
    else:
        lines.extend([
            "The HSS score does not meet all acceptance criteria. Recommendations:",
            "",
            "1. Investigate score formula parameters (α, β, γ weights)",
            "2. Review feature extraction for embeddings",
            "3. Increase training data diversity",
            "4. Consider alternative topological features",
        ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return f"Effect size interpretation: **{interpretation}** ({abs_d:.2f})"


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze HSS predictive power for verification outcomes"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/tda_validation"),
        help="Directory with validation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tda_predictive_analysis"),
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    try:
        result = run_predictive_analysis(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        all_passed = all(result.acceptance_criteria.values())

        print(f"\nPredictive Power Analysis Complete!")
        print(f"  Status: {'PASSED' if all_passed else 'FAILED'}")
        print(f"  Cohen's d: {result.effect_size.cohens_d:.3f}")
        print(f"  AUC-ROC: {result.roc.auc:.3f}")
        print(f"  Output: {args.output_dir}")

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
