#!/usr/bin/env python3
"""
TDA Soft Gate Analysis — Phase II Validation Pipeline

Operation CORTEX: Phase II Soft Gating
======================================

This script performs comprehensive analysis of CORTEX soft gating effects on
MathLedger's learning and planning systems. It validates that topology-based
modulation improves or maintains system performance while detecting structural
anomalies.

Usage:
    python experiments/tda_softgate_analysis.py \
        --baseline-results results/u2_baseline_2000.jsonl \
        --softgate-results results/u2_rfl_softgate_2000.jsonl \
        --output-dir results/tda_softgate_analysis

Analysis Components:
1. Delta Success Rate Analysis
2. Learning Rate Stability Analysis
3. Planner Divergence Statistics
4. Topological Degradation Correlation
5. Per-Slice HSS Distributions
6. Before/After ROC Curves

Output:
- softgate_analysis.json: Complete analysis results
- softgate_report.md: Human-readable analysis report
- figures/: Visualization artifacts
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

# Conditional imports
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tda_softgate_analysis")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CycleRecord:
    """A single cycle telemetry record."""
    cycle: int
    slice_name: str
    mode: str
    success: bool
    hss: Optional[float] = None
    sns: Optional[float] = None
    pcs: Optional[float] = None
    drs: Optional[float] = None
    tda_signal: Optional[str] = None
    eta_base: Optional[float] = None
    eta_eff: Optional[float] = None
    hss_class: Optional[str] = None
    reweighting_applied: bool = False
    computation_ms: Optional[float] = None


@dataclass
class SuccessRateAnalysis:
    """Success rate comparison analysis."""
    baseline_success_rate: float
    softgate_success_rate: float
    delta_success_rate: float
    delta_pct: float
    baseline_n: int
    softgate_n: int
    p_value: Optional[float] = None
    significant: bool = False
    effect_direction: str = "neutral"  # "improved", "degraded", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningRateStabilityAnalysis:
    """Learning rate stability analysis under soft gating."""
    eta_eff_mean: float
    eta_eff_std: float
    eta_eff_min: float
    eta_eff_max: float
    modulation_factor_mean: float
    learning_skipped_count: int
    learning_skipped_pct: float
    hss_class_distribution: Dict[str, int]
    stability_score: float  # Higher = more stable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerDivergenceAnalysis:
    """Planner divergence statistics under HSS reweighting."""
    reweighting_applied_count: int
    reweighting_applied_pct: float
    mean_score_delta: float
    std_score_delta: float
    selection_changed_count: int
    selection_changed_pct: float
    divergence_correlation: float  # Correlation between HSS and selection change

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TopologicalCorrelationAnalysis:
    """Correlation between topological degradation and performance."""
    hss_success_correlation: float
    hss_success_p_value: float
    sns_success_correlation: float
    pcs_success_correlation: float
    drs_failure_correlation: float
    degradation_performance_r2: float
    significant_correlations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SliceHSSDistribution:
    """HSS distribution for a single slice."""
    slice_name: str
    count: int
    hss_mean: float
    hss_std: float
    hss_min: float
    hss_max: float
    hss_p25: float
    hss_p50: float
    hss_p75: float
    signal_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROCComparison:
    """Before/After ROC curve comparison."""
    baseline_auc: float
    softgate_auc: float
    delta_auc: float
    baseline_optimal_threshold: float
    softgate_optimal_threshold: float
    improvement_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SoftGateAnalysisResult:
    """Complete soft gate analysis result."""
    timestamp: str
    baseline_path: str
    softgate_path: str
    baseline_cycles: int
    softgate_cycles: int

    # Core analyses
    success_rate: SuccessRateAnalysis
    learning_stability: LearningRateStabilityAnalysis
    planner_divergence: PlannerDivergenceAnalysis
    topological_correlation: TopologicalCorrelationAnalysis
    slice_distributions: List[SliceHSSDistribution]
    roc_comparison: ROCComparison

    # Summary metrics
    overall_health_score: float  # [0, 1] composite health metric
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "baseline_path": self.baseline_path,
            "softgate_path": self.softgate_path,
            "baseline_cycles": self.baseline_cycles,
            "softgate_cycles": self.softgate_cycles,
            "success_rate": self.success_rate.to_dict(),
            "learning_stability": self.learning_stability.to_dict(),
            "planner_divergence": self.planner_divergence.to_dict(),
            "topological_correlation": self.topological_correlation.to_dict(),
            "slice_distributions": [s.to_dict() for s in self.slice_distributions],
            "roc_comparison": self.roc_comparison.to_dict(),
            "overall_health_score": self.overall_health_score,
            "recommendation": self.recommendation,
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_cycle_records(path: Path) -> List[CycleRecord]:
    """Load cycle records from JSONL file."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                record = CycleRecord(
                    cycle=data.get("cycle", 0),
                    slice_name=data.get("slice_name", data.get("slice", "unknown")),
                    mode=data.get("mode", "unknown"),
                    success=data.get("success", False),
                    hss=data.get("tda_hss", data.get("hss")),
                    sns=data.get("tda_sns", data.get("sns")),
                    pcs=data.get("tda_pcs", data.get("pcs")),
                    drs=data.get("tda_drs", data.get("drs")),
                    tda_signal=data.get("tda_signal"),
                    eta_base=data.get("eta_base"),
                    eta_eff=data.get("eta_eff"),
                    hss_class=data.get("hss_class"),
                    reweighting_applied=data.get("tda_reweighting_applied", False),
                    computation_ms=data.get("tda_computation_ms"),
                )
                records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing record: {e}")
                continue

    return records


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_success_rate(
    baseline: List[CycleRecord],
    softgate: List[CycleRecord],
) -> SuccessRateAnalysis:
    """Compare success rates between baseline and soft-gated runs."""
    baseline_successes = sum(1 for r in baseline if r.success)
    softgate_successes = sum(1 for r in softgate if r.success)

    baseline_rate = baseline_successes / len(baseline) if baseline else 0.0
    softgate_rate = softgate_successes / len(softgate) if softgate else 0.0

    delta = softgate_rate - baseline_rate
    delta_pct = (delta / baseline_rate * 100) if baseline_rate > 0 else 0.0

    # Statistical significance (Fisher's exact test or chi-squared)
    p_value = None
    significant = False
    if HAS_SCIPY and len(baseline) > 0 and len(softgate) > 0:
        try:
            # Create contingency table
            table = [
                [baseline_successes, len(baseline) - baseline_successes],
                [softgate_successes, len(softgate) - softgate_successes],
            ]
            _, p_value = stats.fisher_exact(table)
            significant = p_value < 0.05
        except Exception:
            pass

    # Determine effect direction
    if delta > 0.01:
        effect_direction = "improved"
    elif delta < -0.01:
        effect_direction = "degraded"
    else:
        effect_direction = "neutral"

    return SuccessRateAnalysis(
        baseline_success_rate=baseline_rate,
        softgate_success_rate=softgate_rate,
        delta_success_rate=delta,
        delta_pct=delta_pct,
        baseline_n=len(baseline),
        softgate_n=len(softgate),
        p_value=p_value,
        significant=significant,
        effect_direction=effect_direction,
    )


def analyze_learning_stability(
    softgate: List[CycleRecord],
) -> LearningRateStabilityAnalysis:
    """Analyze learning rate stability under soft gating."""
    eta_effs = [r.eta_eff for r in softgate if r.eta_eff is not None]
    eta_bases = [r.eta_base for r in softgate if r.eta_base is not None]

    # HSS class distribution
    hss_classes = {}
    for r in softgate:
        if r.hss_class:
            hss_classes[r.hss_class] = hss_classes.get(r.hss_class, 0) + 1

    # Learning skipped count (eta_eff = 0 or SOFT_BLOCK class)
    learning_skipped = sum(
        1 for r in softgate
        if r.hss_class == "SOFT_BLOCK" or (r.eta_eff is not None and r.eta_eff == 0)
    )

    # Compute modulation factors
    modulation_factors = []
    for r in softgate:
        if r.eta_base and r.eta_eff and r.eta_base > 0:
            modulation_factors.append(r.eta_eff / r.eta_base)

    # Stability score: higher when eta_eff has low variance
    if eta_effs:
        cv = np.std(eta_effs) / np.mean(eta_effs) if np.mean(eta_effs) > 0 else 1.0
        stability_score = max(0.0, 1.0 - cv)
    else:
        stability_score = 0.0

    return LearningRateStabilityAnalysis(
        eta_eff_mean=float(np.mean(eta_effs)) if eta_effs else 0.0,
        eta_eff_std=float(np.std(eta_effs)) if eta_effs else 0.0,
        eta_eff_min=float(np.min(eta_effs)) if eta_effs else 0.0,
        eta_eff_max=float(np.max(eta_effs)) if eta_effs else 0.0,
        modulation_factor_mean=float(np.mean(modulation_factors)) if modulation_factors else 1.0,
        learning_skipped_count=learning_skipped,
        learning_skipped_pct=learning_skipped / len(softgate) * 100 if softgate else 0.0,
        hss_class_distribution=hss_classes,
        stability_score=stability_score,
    )


def analyze_planner_divergence(
    softgate: List[CycleRecord],
) -> PlannerDivergenceAnalysis:
    """Analyze planner divergence under HSS reweighting."""
    reweighting_applied = sum(1 for r in softgate if r.reweighting_applied)
    reweighting_pct = reweighting_applied / len(softgate) * 100 if softgate else 0.0

    # For this analysis, we estimate selection changes based on HSS variance
    # In production, this would track actual selection changes
    hss_values = [r.hss for r in softgate if r.hss is not None]

    # Estimate selection change probability based on HSS variance
    if hss_values and len(hss_values) > 1:
        hss_std = np.std(hss_values)
        # Higher HSS variance = more likely selections changed
        selection_change_estimate = min(1.0, hss_std * 2)
        selection_changed = int(len(softgate) * selection_change_estimate * 0.1)
    else:
        selection_changed = 0

    # Divergence correlation (HSS vs implicit selection change indicator)
    divergence_correlation = 0.0
    if HAS_SCIPY and hss_values:
        try:
            # Proxy: correlate HSS with success (divergence from poor choices)
            successes = [1.0 if r.success else 0.0 for r in softgate if r.hss is not None]
            if len(successes) == len(hss_values):
                divergence_correlation, _ = stats.pearsonr(hss_values, successes)
        except Exception:
            pass

    return PlannerDivergenceAnalysis(
        reweighting_applied_count=reweighting_applied,
        reweighting_applied_pct=reweighting_pct,
        mean_score_delta=0.0,  # Would need actual score tracking
        std_score_delta=0.0,
        selection_changed_count=selection_changed,
        selection_changed_pct=selection_changed / len(softgate) * 100 if softgate else 0.0,
        divergence_correlation=float(divergence_correlation),
    )


def analyze_topological_correlation(
    records: List[CycleRecord],
) -> TopologicalCorrelationAnalysis:
    """Analyze correlation between topological features and performance."""
    # Extract arrays
    hss = [r.hss for r in records if r.hss is not None]
    sns = [r.sns for r in records if r.sns is not None]
    pcs = [r.pcs for r in records if r.pcs is not None]
    drs = [r.drs for r in records if r.drs is not None]
    success = [1.0 if r.success else 0.0 for r in records if r.hss is not None]

    correlations = {
        "hss_success": (0.0, 1.0),
        "sns_success": (0.0, 1.0),
        "pcs_success": (0.0, 1.0),
        "drs_failure": (0.0, 1.0),
    }

    significant = []

    if HAS_SCIPY:
        try:
            if len(hss) == len(success) and len(hss) > 2:
                r, p = stats.pearsonr(hss, success)
                correlations["hss_success"] = (r, p)
                if p < 0.05:
                    significant.append("hss_success")

            if len(sns) == len(success) and len(sns) > 2:
                r, p = stats.pearsonr(sns, success[:len(sns)])
                correlations["sns_success"] = (r, p)
                if p < 0.05:
                    significant.append("sns_success")

            if len(pcs) == len(success) and len(pcs) > 2:
                r, p = stats.pearsonr(pcs, success[:len(pcs)])
                correlations["pcs_success"] = (r, p)
                if p < 0.05:
                    significant.append("pcs_success")

            if len(drs) > 2:
                failures = [1.0 if not r.success else 0.0 for r in records if r.drs is not None]
                r, p = stats.pearsonr(drs, failures[:len(drs)])
                correlations["drs_failure"] = (r, p)
                if p < 0.05:
                    significant.append("drs_failure")
        except Exception as e:
            logger.warning(f"Correlation analysis error: {e}")

    # R² approximation: use HSS correlation squared
    r2 = correlations["hss_success"][0] ** 2

    return TopologicalCorrelationAnalysis(
        hss_success_correlation=correlations["hss_success"][0],
        hss_success_p_value=correlations["hss_success"][1],
        sns_success_correlation=correlations["sns_success"][0],
        pcs_success_correlation=correlations["pcs_success"][0],
        drs_failure_correlation=correlations["drs_failure"][0],
        degradation_performance_r2=r2,
        significant_correlations=significant,
    )


def analyze_slice_distributions(
    records: List[CycleRecord],
) -> List[SliceHSSDistribution]:
    """Analyze HSS distributions per slice."""
    # Group by slice
    by_slice: Dict[str, List[CycleRecord]] = {}
    for r in records:
        if r.slice_name not in by_slice:
            by_slice[r.slice_name] = []
        by_slice[r.slice_name].append(r)

    distributions = []
    for slice_name, slice_records in by_slice.items():
        hss_values = [r.hss for r in slice_records if r.hss is not None]

        if not hss_values:
            continue

        signal_dist = {}
        for r in slice_records:
            if r.tda_signal:
                signal_dist[r.tda_signal] = signal_dist.get(r.tda_signal, 0) + 1

        distributions.append(SliceHSSDistribution(
            slice_name=slice_name,
            count=len(hss_values),
            hss_mean=float(np.mean(hss_values)),
            hss_std=float(np.std(hss_values)),
            hss_min=float(np.min(hss_values)),
            hss_max=float(np.max(hss_values)),
            hss_p25=float(np.percentile(hss_values, 25)),
            hss_p50=float(np.percentile(hss_values, 50)),
            hss_p75=float(np.percentile(hss_values, 75)),
            signal_distribution=signal_dist,
        ))

    return distributions


def compute_roc_comparison(
    baseline: List[CycleRecord],
    softgate: List[CycleRecord],
) -> ROCComparison:
    """Compute before/after ROC curves."""
    def compute_auc(records: List[CycleRecord]) -> Tuple[float, float]:
        """Compute AUC and optimal threshold."""
        hss = [r.hss for r in records if r.hss is not None]
        success = [1.0 if r.success else 0.0 for r in records if r.hss is not None]

        if len(hss) < 10:
            return 0.5, 0.5

        # Sort by HSS descending
        pairs = sorted(zip(hss, success), key=lambda x: -x[0])

        # Compute ROC points
        n_pos = sum(success)
        n_neg = len(success) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5, 0.5

        tpr, fpr = [], []
        for thresh in np.linspace(0, 1, 50):
            tp = sum(1 for h, s in pairs if h >= thresh and s == 1.0)
            fp = sum(1 for h, s in pairs if h >= thresh and s == 0.0)
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        # AUC via trapezoidal rule
        auc = 0.0
        for i in range(len(fpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2
        auc = abs(auc)

        # Optimal threshold (Youden's J)
        j_scores = [t - f for t, f in zip(tpr, fpr)]
        optimal_idx = np.argmax(j_scores)
        optimal_thresh = np.linspace(0, 1, 50)[optimal_idx]

        return auc, optimal_thresh

    baseline_auc, baseline_thresh = compute_auc(baseline)
    softgate_auc, softgate_thresh = compute_auc(softgate)

    delta = softgate_auc - baseline_auc
    improvement = (delta / baseline_auc * 100) if baseline_auc > 0 else 0.0

    return ROCComparison(
        baseline_auc=baseline_auc,
        softgate_auc=softgate_auc,
        delta_auc=delta,
        baseline_optimal_threshold=baseline_thresh,
        softgate_optimal_threshold=softgate_thresh,
        improvement_pct=improvement,
    )


def compute_health_score(
    success: SuccessRateAnalysis,
    stability: LearningRateStabilityAnalysis,
    correlation: TopologicalCorrelationAnalysis,
    roc: ROCComparison,
) -> float:
    """
    Compute overall health score for soft gating.

    Components (weighted):
    - Success rate delta >= 0: 0.30
    - Learning stability > 0.7: 0.25
    - HSS-success correlation > 0: 0.25
    - AUC improvement: 0.20
    """
    score = 0.0

    # Success rate (no degradation)
    if success.delta_success_rate >= 0:
        score += 0.30
    elif success.delta_success_rate > -0.05:
        score += 0.15  # Small degradation

    # Learning stability
    if stability.stability_score >= 0.7:
        score += 0.25
    elif stability.stability_score >= 0.5:
        score += 0.15

    # HSS-success correlation
    if correlation.hss_success_correlation > 0:
        score += 0.25 * min(1.0, correlation.hss_success_correlation * 2)

    # AUC improvement
    if roc.delta_auc >= 0:
        score += 0.20 * min(1.0, roc.delta_auc * 10 + 0.5)

    return min(1.0, score)


# ============================================================================
# Visualization
# ============================================================================

def plot_success_rate_comparison(
    baseline: List[CycleRecord],
    softgate: List[CycleRecord],
    output_path: Path,
) -> None:
    """Plot success rate comparison."""
    if not HAS_MATPLOTLIB:
        return

    # Compute rolling success rates
    window = 100

    def rolling_success(records: List[CycleRecord]) -> List[float]:
        rates = []
        for i in range(len(records)):
            start = max(0, i - window + 1)
            window_records = records[start:i + 1]
            rate = sum(1 for r in window_records if r.success) / len(window_records)
            rates.append(rate)
        return rates

    baseline_rolling = rolling_success(baseline)
    softgate_rolling = rolling_success(softgate)

    plt.figure(figsize=(12, 6))
    plt.plot(baseline_rolling, label="Baseline", alpha=0.7)
    plt.plot(softgate_rolling, label="Soft-Gated", alpha=0.7)
    plt.xlabel("Cycle")
    plt.ylabel("Rolling Success Rate (window=100)")
    plt.title("Success Rate: Baseline vs Soft-Gated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hss_distribution(
    softgate: List[CycleRecord],
    output_path: Path,
) -> None:
    """Plot HSS distribution with class annotations."""
    if not HAS_MATPLOTLIB:
        return

    hss_values = [r.hss for r in softgate if r.hss is not None]

    if not hss_values:
        return

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(hss_values, bins=30, alpha=0.7, color="blue", edgecolor="black")

    # Threshold lines
    plt.axvline(0.5, color="orange", linestyle="--", linewidth=2, label="θ_warn (0.5)")
    plt.axvline(0.2, color="red", linestyle="--", linewidth=2, label="θ_block (0.2)")

    plt.xlabel("HSS Score")
    plt.ylabel("Count")
    plt.title("HSS Distribution Under Soft Gating")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_learning_rate_over_time(
    softgate: List[CycleRecord],
    output_path: Path,
) -> None:
    """Plot effective learning rate over time."""
    if not HAS_MATPLOTLIB:
        return

    cycles = [r.cycle for r in softgate if r.eta_eff is not None]
    eta_effs = [r.eta_eff for r in softgate if r.eta_eff is not None]
    hss_values = [r.hss for r in softgate if r.eta_eff is not None and r.hss is not None]

    if not cycles:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Eta_eff over time
    ax1.plot(cycles, eta_effs, alpha=0.7, color="blue")
    ax1.axhline(0.1, color="gray", linestyle="--", label="η_base")
    ax1.set_ylabel("η_eff")
    ax1.set_title("Effective Learning Rate Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # HSS over time
    if len(hss_values) == len(cycles):
        ax2.plot(cycles, hss_values, alpha=0.7, color="green")
        ax2.axhline(0.5, color="orange", linestyle="--", label="θ_warn")
        ax2.axhline(0.2, color="red", linestyle="--", label="θ_block")
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("HSS")
        ax2.set_title("HSS Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_comparison(
    baseline: List[CycleRecord],
    softgate: List[CycleRecord],
    output_path: Path,
) -> None:
    """Plot ROC curve comparison."""
    if not HAS_MATPLOTLIB:
        return

    def compute_roc_curve(records: List[CycleRecord]) -> Tuple[List[float], List[float]]:
        hss = [r.hss for r in records if r.hss is not None]
        success = [1.0 if r.success else 0.0 for r in records if r.hss is not None]

        if len(hss) < 10:
            return [0, 1], [0, 1]

        n_pos = sum(success)
        n_neg = len(success) - n_pos

        if n_pos == 0 or n_neg == 0:
            return [0, 1], [0, 1]

        tpr, fpr = [], []
        for thresh in np.linspace(0, 1, 50):
            tp = sum(1 for h, s in zip(hss, success) if h >= thresh and s == 1.0)
            fp = sum(1 for h, s in zip(hss, success) if h >= thresh and s == 0.0)
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        return fpr, tpr

    baseline_fpr, baseline_tpr = compute_roc_curve(baseline)
    softgate_fpr, softgate_tpr = compute_roc_curve(softgate)

    plt.figure(figsize=(8, 6))
    plt.plot(baseline_fpr, baseline_tpr, label="Baseline", alpha=0.7)
    plt.plot(softgate_fpr, softgate_tpr, label="Soft-Gated", alpha=0.7)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Baseline vs Soft-Gated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(result: SoftGateAnalysisResult, output_path: Path) -> None:
    """Generate human-readable analysis report."""
    sr = result.success_rate
    ls = result.learning_stability
    pd = result.planner_divergence
    tc = result.topological_correlation
    roc = result.roc_comparison

    lines = [
        "# CORTEX Soft Gate Analysis Report",
        "",
        f"**Generated**: {result.timestamp}",
        f"**Overall Health Score**: {result.overall_health_score:.2f}/1.00",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"{result.recommendation}",
        "",
        "## Data Summary",
        "",
        f"| Metric | Baseline | Soft-Gated |",
        f"|--------|----------|------------|",
        f"| Cycles | {result.baseline_cycles} | {result.softgate_cycles} |",
        f"| Success Rate | {sr.baseline_success_rate:.4f} | {sr.softgate_success_rate:.4f} |",
        "",
        "## 1. Success Rate Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline Success Rate | {sr.baseline_success_rate:.4f} |",
        f"| Soft-Gated Success Rate | {sr.softgate_success_rate:.4f} |",
        f"| Delta | {sr.delta_success_rate:+.4f} ({sr.delta_pct:+.2f}%) |",
        f"| P-value | {sr.p_value:.4e if sr.p_value else 'N/A'} |",
        f"| Significant | {'Yes' if sr.significant else 'No'} |",
        f"| Effect Direction | {sr.effect_direction.upper()} |",
        "",
        "## 2. Learning Rate Stability",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| η_eff Mean | {ls.eta_eff_mean:.4f} |",
        f"| η_eff Std | {ls.eta_eff_std:.4f} |",
        f"| η_eff Range | [{ls.eta_eff_min:.4f}, {ls.eta_eff_max:.4f}] |",
        f"| Modulation Factor Mean | {ls.modulation_factor_mean:.4f} |",
        f"| Learning Skipped | {ls.learning_skipped_count} ({ls.learning_skipped_pct:.2f}%) |",
        f"| Stability Score | {ls.stability_score:.4f} |",
        "",
        "**HSS Class Distribution**:",
        "",
    ]

    for cls, count in ls.hss_class_distribution.items():
        lines.append(f"- {cls}: {count}")

    lines.extend([
        "",
        "## 3. Planner Divergence",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Reweighting Applied | {pd.reweighting_applied_count} ({pd.reweighting_applied_pct:.2f}%) |",
        f"| Selection Changed (est.) | {pd.selection_changed_count} ({pd.selection_changed_pct:.2f}%) |",
        f"| Divergence Correlation | {pd.divergence_correlation:.4f} |",
        "",
        "## 4. Topological Correlation",
        "",
        f"| Correlation | Value | P-value |",
        f"|-------------|-------|---------|",
        f"| HSS ↔ Success | {tc.hss_success_correlation:.4f} | {tc.hss_success_p_value:.4e} |",
        f"| SNS ↔ Success | {tc.sns_success_correlation:.4f} | - |",
        f"| PCS ↔ Success | {tc.pcs_success_correlation:.4f} | - |",
        f"| DRS ↔ Failure | {tc.drs_failure_correlation:.4f} | - |",
        f"| R² (HSS→Performance) | {tc.degradation_performance_r2:.4f} | - |",
        "",
        f"**Significant Correlations**: {', '.join(tc.significant_correlations) or 'None'}",
        "",
        "## 5. Per-Slice HSS Distributions",
        "",
        f"| Slice | N | Mean | Std | Min | Max |",
        f"|-------|---|------|-----|-----|-----|",
    ])

    for sd in result.slice_distributions:
        lines.append(
            f"| {sd.slice_name} | {sd.count} | {sd.hss_mean:.3f} | "
            f"{sd.hss_std:.3f} | {sd.hss_min:.3f} | {sd.hss_max:.3f} |"
        )

    lines.extend([
        "",
        "## 6. ROC Comparison",
        "",
        f"| Metric | Baseline | Soft-Gated |",
        f"|--------|----------|------------|",
        f"| AUC | {roc.baseline_auc:.4f} | {roc.softgate_auc:.4f} |",
        f"| Optimal Threshold | {roc.baseline_optimal_threshold:.4f} | {roc.softgate_optimal_threshold:.4f} |",
        f"| AUC Improvement | - | {roc.improvement_pct:+.2f}% |",
        "",
        "## Phase III Readiness",
        "",
    ])

    if result.overall_health_score >= 0.7:
        lines.extend([
            "**READY FOR PHASE III HARD GATING**",
            "",
            "The soft gating analysis indicates that CORTEX is functioning correctly:",
            f"- Success rate: {sr.effect_direction}",
            f"- HSS-success correlation: {tc.hss_success_correlation:.3f}",
            f"- Stability score: {ls.stability_score:.3f}",
        ])
    else:
        lines.extend([
            "**NOT READY FOR PHASE III**",
            "",
            "Additional tuning required before hard gating:",
        ])
        if sr.effect_direction == "degraded":
            lines.append("- Address success rate degradation")
        if ls.stability_score < 0.5:
            lines.append("- Improve learning rate stability")
        if tc.hss_success_correlation <= 0:
            lines.append("- Investigate negative/zero HSS-success correlation")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_softgate_analysis(
    baseline_path: Path,
    softgate_path: Path,
    output_dir: Path,
) -> SoftGateAnalysisResult:
    """Run complete soft gate analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load data
    logger.info(f"Loading baseline data from {baseline_path}")
    baseline = load_cycle_records(baseline_path)
    logger.info(f"  Loaded {len(baseline)} baseline cycles")

    logger.info(f"Loading soft-gated data from {softgate_path}")
    softgate = load_cycle_records(softgate_path)
    logger.info(f"  Loaded {len(softgate)} soft-gated cycles")

    # Run analyses
    logger.info("Analyzing success rates...")
    success_analysis = analyze_success_rate(baseline, softgate)

    logger.info("Analyzing learning stability...")
    learning_analysis = analyze_learning_stability(softgate)

    logger.info("Analyzing planner divergence...")
    planner_analysis = analyze_planner_divergence(softgate)

    logger.info("Analyzing topological correlations...")
    correlation_analysis = analyze_topological_correlation(softgate)

    logger.info("Analyzing per-slice distributions...")
    slice_distributions = analyze_slice_distributions(softgate)

    logger.info("Computing ROC comparison...")
    roc_comparison = compute_roc_comparison(baseline, softgate)

    # Compute health score
    health_score = compute_health_score(
        success_analysis, learning_analysis, correlation_analysis, roc_comparison
    )

    # Generate recommendation
    if health_score >= 0.7:
        recommendation = (
            "CORTEX soft gating is operating correctly. The system maintains or improves "
            "performance while successfully integrating topological signals. "
            "Recommend proceeding to Phase III Hard Gating trials."
        )
    elif health_score >= 0.5:
        recommendation = (
            "CORTEX soft gating shows mixed results. Some improvements observed but "
            "stability concerns remain. Recommend additional tuning before Phase III."
        )
    else:
        recommendation = (
            "CORTEX soft gating requires attention. Performance degradation or "
            "correlation issues detected. Recommend reverting to Shadow Mode for "
            "further investigation."
        )

    # Build result
    result = SoftGateAnalysisResult(
        timestamp=datetime.utcnow().isoformat() + "Z",
        baseline_path=str(baseline_path),
        softgate_path=str(softgate_path),
        baseline_cycles=len(baseline),
        softgate_cycles=len(softgate),
        success_rate=success_analysis,
        learning_stability=learning_analysis,
        planner_divergence=planner_analysis,
        topological_correlation=correlation_analysis,
        slice_distributions=slice_distributions,
        roc_comparison=roc_comparison,
        overall_health_score=health_score,
        recommendation=recommendation,
    )

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_success_rate_comparison(baseline, softgate, figures_dir / "success_rate_comparison.png")
    plot_hss_distribution(softgate, figures_dir / "hss_distribution.png")
    plot_learning_rate_over_time(softgate, figures_dir / "learning_rate_over_time.png")
    plot_roc_comparison(baseline, softgate, figures_dir / "roc_comparison.png")

    # Save results
    results_path = output_dir / "softgate_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Generate report
    generate_report(result, output_dir / "softgate_report.md")

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze CORTEX soft gating effects"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        required=True,
        help="Path to baseline results JSONL",
    )
    parser.add_argument(
        "--softgate-results",
        type=Path,
        required=True,
        help="Path to soft-gated results JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tda_softgate_analysis"),
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    try:
        result = run_softgate_analysis(
            baseline_path=args.baseline_results,
            softgate_path=args.softgate_results,
            output_dir=args.output_dir,
        )

        print(f"\nSoft Gate Analysis Complete!")
        print(f"  Health Score: {result.overall_health_score:.2f}/1.00")
        print(f"  Success Rate Delta: {result.success_rate.delta_success_rate:+.4f}")
        print(f"  HSS-Success Correlation: {result.topological_correlation.hss_success_correlation:.3f}")
        print(f"  Output: {args.output_dir}")

        return 0 if result.overall_health_score >= 0.5 else 1

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
