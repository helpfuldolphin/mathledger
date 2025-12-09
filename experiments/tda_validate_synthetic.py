#!/usr/bin/env python3
"""
TDA Mind Scanner - Synthetic Validation Framework (Phase 0).

This script provides the scientific evidence behind CORTEX:
- Runs all synthetic DAG generators at scale (100+ DAGs per class)
- Computes SNS/PCS/DRS/HSS for good/bad/marginal proofs
- Calculates separation metrics: Cohen's d, AUC, confusion matrices
- Emits calibration JSON, summary report, and visualizations

Usage:
    python experiments/tda_validate_synthetic.py --output-dir results/tda_validation
    python experiments/tda_validate_synthetic.py --num-per-class 200 --seed 42

References:
    - docs/TDA_MIND_SCANNER_SPEC.md
    - Phase 0 validation requirements
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tda_validate_synthetic")


@dataclass
class ValidationResult:
    """Result for a single DAG evaluation."""
    label: str  # "good", "bad", "marginal"
    dag_id: str
    sns: float
    pcs: float
    drs: float
    hss: float
    betti_0: int
    betti_1: int
    num_nodes: int
    num_edges: int
    num_simplices: int
    signal: str  # "BLOCK", "WARN", "OK"
    f_size: float
    f_topo: float


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix."""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class ValidationSummary:
    """Summary statistics from validation run."""
    num_good: int
    num_bad: int
    num_marginal: int
    good_hss_mean: float
    good_hss_std: float
    bad_hss_mean: float
    bad_hss_std: float
    marginal_hss_mean: float
    marginal_hss_std: float
    cohens_d: float
    auc: float
    optimal_threshold: float
    confusion_at_optimal: ConfusionMatrix
    confusion_at_default: ConfusionMatrix  # θ_warn = 0.5
    confusion_at_block: ConfusionMatrix    # θ_block = 0.2
    # Per-score statistics
    sns_separation: float
    pcs_separation: float
    drs_separation: float


# ============================================================================
# Synthetic DAG Generators
# ============================================================================

def create_good_dag_v1(rng: random.Random) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Good proof DAG v1: Multi-level derivation with cross-links.

    Characteristics:
    - 3 axioms → 5 lemmas → 3 theorems → goal
    - Cross-links create non-trivial topology
    - Connected structure
    """
    import networkx as nx

    G = nx.DiGraph()

    # Axioms
    axioms = [f"axiom_{i}" for i in range(3)]
    for ax in axioms:
        G.add_node(ax)

    # Lemmas derived from axioms
    lemmas = []
    for i in range(5):
        lemma = f"lemma_{i}"
        G.add_node(lemma)
        lemmas.append(lemma)
        # Each lemma depends on 1-2 axioms
        num_parents = rng.randint(1, 2)
        parents = rng.sample(axioms, k=min(num_parents, len(axioms)))
        for p in parents:
            G.add_edge(p, lemma)

    # Theorems depend on multiple lemmas (creating cross-links)
    theorems = []
    for i in range(3):
        thm = f"theorem_{i}"
        G.add_node(thm)
        theorems.append(thm)
        num_parents = rng.randint(2, 3)
        parents = rng.sample(lemmas, k=min(num_parents, len(lemmas)))
        for p in parents:
            G.add_edge(p, thm)

    # Goal depends on all theorems
    G.add_node("goal")
    for thm in theorems:
        G.add_edge(thm, "goal")

    embeddings = _generate_embeddings(list(G.nodes()), rng)
    return G, embeddings


def create_good_dag_v2(rng: random.Random) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Good proof DAG v2: Diamond pattern with reuse.

    Characteristics:
    - Shared intermediate lemmas
    - Multiple paths to goal
    - Dense connectivity
    """
    import networkx as nx

    G = nx.DiGraph()

    # Create diamond pattern
    G.add_node("base_1")
    G.add_node("base_2")
    G.add_node("shared_lemma")
    G.add_node("branch_a")
    G.add_node("branch_b")
    G.add_node("merge")
    G.add_node("goal")

    # Edges
    G.add_edge("base_1", "shared_lemma")
    G.add_edge("base_2", "shared_lemma")
    G.add_edge("shared_lemma", "branch_a")
    G.add_edge("shared_lemma", "branch_b")
    G.add_edge("base_1", "branch_a")
    G.add_edge("base_2", "branch_b")
    G.add_edge("branch_a", "merge")
    G.add_edge("branch_b", "merge")
    G.add_edge("merge", "goal")

    # Add extra cross-links for richness
    if rng.random() > 0.3:
        G.add_edge("base_1", "merge")
    if rng.random() > 0.3:
        G.add_edge("base_2", "branch_a")

    embeddings = _generate_embeddings(list(G.nodes()), rng)
    return G, embeddings


def create_bad_dag_v1(rng: random.Random) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Bad proof DAG v1: Simple linear chain (trivial).

    Characteristics:
    - Linear path with no branching
    - Minimal structure
    - Tree topology (β_1 = 0)
    """
    import networkx as nx

    G = nx.DiGraph()
    length = rng.randint(2, 4)
    nodes = [f"step_{i}" for i in range(length)]
    for node in nodes:
        G.add_node(node)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    embeddings = _generate_embeddings(list(G.nodes()), rng)
    return G, embeddings


def create_bad_dag_v2(rng: random.Random) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Bad proof DAG v2: Disconnected components.

    Characteristics:
    - Multiple isolated components
    - No coherent derivation path
    - β_0 > 1
    """
    import networkx as nx

    G = nx.DiGraph()

    # Component 1: small chain
    G.add_edge("a1", "a2")

    # Component 2: single edge
    G.add_edge("b1", "b2")

    # Component 3: isolated node
    G.add_node("isolated")

    embeddings = _generate_embeddings(list(G.nodes()), rng)
    return G, embeddings


def create_marginal_dag(rng: random.Random) -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Marginal proof DAG: Tree structure without cycles.

    Characteristics:
    - Connected but no cross-links
    - Tree topology
    - Some depth but β_1 = 0
    """
    import networkx as nx

    G = nx.DiGraph()

    G.add_node("root")
    num_children = rng.randint(2, 4)
    for i in range(num_children):
        child = f"child_{i}"
        G.add_node(child)
        G.add_edge("root", child)

        num_grandchildren = rng.randint(1, 3)
        for j in range(num_grandchildren):
            gc = f"gc_{i}_{j}"
            G.add_node(gc)
            G.add_edge(child, gc)

    embeddings = _generate_embeddings(list(G.nodes()), rng)
    return G, embeddings


def _generate_embeddings(nodes: List[str], rng: random.Random) -> Dict[str, np.ndarray]:
    """Generate 19-dimensional embeddings matching features.py format."""
    embeddings = {}
    for i, node in enumerate(nodes):
        base = np.zeros(19, dtype=np.float32)
        base[0] = len(node)  # length
        base[1] = 2  # word count
        base[2] = len(node.replace("_", ""))  # char count
        base[10] = float(i) / max(len(nodes), 1)  # depth proxy
        base[11] = rng.uniform(0.1, 0.5)  # operator density
        # Add controlled noise
        noise = np.array([rng.gauss(0, 0.1) for _ in range(19)], dtype=np.float32)
        base += noise
        embeddings[node] = base
    return embeddings


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_dag(
    dag: Any,
    embeddings: Dict[str, np.ndarray],
    ref_profile: Optional[Any] = None,
) -> ValidationResult:
    """Evaluate a single DAG using TDA metrics."""
    from backend.tda.proof_complex import build_combinatorial_complex
    from backend.tda.metric_complex import build_metric_complex
    from backend.tda.scores import (
        compute_structural_nontriviality_detailed,
        compute_persistence_coherence,
        compute_deviation_from_reference,
        compute_hallucination_stability_score,
        classify_hss,
    )

    # Build complexes
    comb_complex = build_combinatorial_complex(dag, max_clique_size=4)
    tda_result = build_metric_complex(embeddings, max_dim=1)

    # Compute scores
    sns_details = compute_structural_nontriviality_detailed(comb_complex, ref_profile)
    sns = sns_details["sns"]
    pcs = compute_persistence_coherence(tda_result, ref_profile)
    drs = compute_deviation_from_reference(tda_result, ref_profile)
    hss = compute_hallucination_stability_score(sns, pcs, drs)

    # Get Betti numbers
    betti = comb_complex.compute_betti_numbers(max_dim=1)

    # Classify
    signal = classify_hss(hss)

    return ValidationResult(
        label="unknown",
        dag_id="",
        sns=sns,
        pcs=pcs,
        drs=drs,
        hss=hss,
        betti_0=betti.get(0, 0),
        betti_1=betti.get(1, 0),
        num_nodes=comb_complex.num_vertices,
        num_edges=comb_complex.num_edges,
        num_simplices=comb_complex.num_simplices,
        signal=signal,
        f_size=sns_details["f_size"],
        f_topo=sns_details["f_topo"],
    )


def generate_and_evaluate(
    num_per_class: int,
    seed: int,
) -> List[ValidationResult]:
    """Generate and evaluate DAGs for all classes."""
    rng = random.Random(seed)
    np.random.seed(seed)

    results: List[ValidationResult] = []

    # Good DAGs (alternate between variants)
    logger.info(f"Generating {num_per_class} good DAGs...")
    for i in range(num_per_class):
        if i % 2 == 0:
            dag, emb = create_good_dag_v1(rng)
        else:
            dag, emb = create_good_dag_v2(rng)
        result = evaluate_dag(dag, emb)
        result.label = "good"
        result.dag_id = f"good_{i:04d}"
        results.append(result)

    # Bad DAGs (alternate between variants)
    logger.info(f"Generating {num_per_class} bad DAGs...")
    for i in range(num_per_class):
        if i % 2 == 0:
            dag, emb = create_bad_dag_v1(rng)
        else:
            dag, emb = create_bad_dag_v2(rng)
        result = evaluate_dag(dag, emb)
        result.label = "bad"
        result.dag_id = f"bad_{i:04d}"
        results.append(result)

    # Marginal DAGs
    logger.info(f"Generating {num_per_class // 2} marginal DAGs...")
    for i in range(num_per_class // 2):
        dag, emb = create_marginal_dag(rng)
        result = evaluate_dag(dag, emb)
        result.label = "marginal"
        result.dag_id = f"marginal_{i:04d}"
        results.append(result)

    return results


# ============================================================================
# Statistics
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_auc(scores: List[Tuple[float, int]]) -> float:
    """
    Compute AUC via trapezoidal rule.

    Args:
        scores: List of (score, label) where label=1 is positive (good)
    """
    if not scores:
        return 0.5

    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: -x[0])

    n_pos = sum(1 for _, l in sorted_scores if l == 1)
    n_neg = sum(1 for _, l in sorted_scores if l == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs
    tp = 0
    fp = 0
    prev_score = None
    auc = 0.0

    for score, label in sorted_scores:
        if prev_score is not None and score != prev_score:
            # Add rectangle
            auc += tp * (fp / n_neg) / n_pos

        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    # Final addition
    auc += tp * (fp / n_neg) / n_pos

    # Compute via Mann-Whitney U statistic (more accurate)
    pos_scores = [s for s, l in scores if l == 1]
    neg_scores = [s for s, l in scores if l == 0]

    concordant = 0
    ties = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1
            elif ps == ns:
                ties += 1

    return (concordant + 0.5 * ties) / (n_pos * n_neg)


def find_optimal_threshold(
    good_hss: np.ndarray,
    bad_hss: np.ndarray,
) -> Tuple[float, ConfusionMatrix]:
    """Find threshold that maximizes F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0
    best_cm = ConfusionMatrix()

    # Try thresholds from 0.1 to 0.9
    for threshold in np.arange(0.1, 0.95, 0.05):
        cm = compute_confusion_matrix(good_hss, bad_hss, threshold)
        if cm.f1 > best_f1:
            best_f1 = cm.f1
            best_threshold = threshold
            best_cm = cm

    return best_threshold, best_cm


def compute_confusion_matrix(
    good_hss: np.ndarray,
    bad_hss: np.ndarray,
    threshold: float,
) -> ConfusionMatrix:
    """
    Compute confusion matrix at given threshold.

    Positive = good (HSS >= threshold)
    Negative = bad (HSS < threshold)
    """
    cm = ConfusionMatrix()

    # Good DAGs: should have HSS >= threshold (positive)
    cm.true_positives = int(np.sum(good_hss >= threshold))
    cm.false_negatives = int(np.sum(good_hss < threshold))

    # Bad DAGs: should have HSS < threshold (negative)
    cm.true_negatives = int(np.sum(bad_hss < threshold))
    cm.false_positives = int(np.sum(bad_hss >= threshold))

    return cm


def compute_summary(results: List[ValidationResult]) -> ValidationSummary:
    """Compute comprehensive summary statistics."""
    good = [r for r in results if r.label == "good"]
    bad = [r for r in results if r.label == "bad"]
    marginal = [r for r in results if r.label == "marginal"]

    good_hss = np.array([r.hss for r in good])
    bad_hss = np.array([r.hss for r in bad])
    marginal_hss = np.array([r.hss for r in marginal]) if marginal else np.array([])

    # Basic stats
    good_mean = float(np.mean(good_hss)) if len(good_hss) > 0 else 0.0
    good_std = float(np.std(good_hss)) if len(good_hss) > 0 else 0.0
    bad_mean = float(np.mean(bad_hss)) if len(bad_hss) > 0 else 0.0
    bad_std = float(np.std(bad_hss)) if len(bad_hss) > 0 else 0.0
    marginal_mean = float(np.mean(marginal_hss)) if len(marginal_hss) > 0 else 0.0
    marginal_std = float(np.std(marginal_hss)) if len(marginal_hss) > 0 else 0.0

    # Cohen's d
    cohens_d = compute_cohens_d(good_hss, bad_hss) if len(good_hss) > 0 and len(bad_hss) > 0 else 0.0

    # AUC
    auc_data = [(r.hss, 1) for r in good] + [(r.hss, 0) for r in bad]
    auc = compute_auc(auc_data)

    # Optimal threshold
    optimal_threshold, confusion_optimal = find_optimal_threshold(good_hss, bad_hss)

    # Confusion at default thresholds
    confusion_default = compute_confusion_matrix(good_hss, bad_hss, 0.5)
    confusion_block = compute_confusion_matrix(good_hss, bad_hss, 0.2)

    # Per-score separation (Cohen's d for each)
    good_sns = np.array([r.sns for r in good])
    bad_sns = np.array([r.sns for r in bad])
    sns_sep = compute_cohens_d(good_sns, bad_sns) if len(good_sns) > 0 and len(bad_sns) > 0 else 0.0

    good_pcs = np.array([r.pcs for r in good])
    bad_pcs = np.array([r.pcs for r in bad])
    pcs_sep = compute_cohens_d(good_pcs, bad_pcs) if len(good_pcs) > 0 and len(bad_pcs) > 0 else 0.0

    good_drs = np.array([r.drs for r in good])
    bad_drs = np.array([r.drs for r in bad])
    # DRS is inverted (lower is better for good)
    drs_sep = compute_cohens_d(bad_drs, good_drs) if len(good_drs) > 0 and len(bad_drs) > 0 else 0.0

    return ValidationSummary(
        num_good=len(good),
        num_bad=len(bad),
        num_marginal=len(marginal),
        good_hss_mean=good_mean,
        good_hss_std=good_std,
        bad_hss_mean=bad_mean,
        bad_hss_std=bad_std,
        marginal_hss_mean=marginal_mean,
        marginal_hss_std=marginal_std,
        cohens_d=cohens_d,
        auc=auc,
        optimal_threshold=optimal_threshold,
        confusion_at_optimal=confusion_optimal,
        confusion_at_default=confusion_default,
        confusion_at_block=confusion_block,
        sns_separation=sns_sep,
        pcs_separation=pcs_sep,
        drs_separation=drs_sep,
    )


# ============================================================================
# Visualization
# ============================================================================

def generate_visualizations(
    results: List[ValidationResult],
    summary: ValidationSummary,
    output_dir: Path,
) -> None:
    """Generate PNG visualizations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
        return

    good = [r for r in results if r.label == "good"]
    bad = [r for r in results if r.label == "bad"]
    marginal = [r for r in results if r.label == "marginal"]

    # Figure 1: HSS Distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 30)
    ax.hist([r.hss for r in good], bins=bins, alpha=0.7, label=f"Good (n={len(good)})", color="green")
    ax.hist([r.hss for r in bad], bins=bins, alpha=0.7, label=f"Bad (n={len(bad)})", color="red")
    if marginal:
        ax.hist([r.hss for r in marginal], bins=bins, alpha=0.7, label=f"Marginal (n={len(marginal)})", color="orange")

    ax.axvline(x=0.2, color="darkred", linestyle="--", label="θ_block=0.2")
    ax.axvline(x=0.5, color="darkorange", linestyle="--", label="θ_warn=0.5")
    ax.axvline(x=summary.optimal_threshold, color="blue", linestyle=":", label=f"Optimal={summary.optimal_threshold:.2f}")

    ax.set_xlabel("Hallucination Stability Score (HSS)")
    ax.set_ylabel("Count")
    ax.set_title(f"HSS Distribution by DAG Class\nCohen's d={summary.cohens_d:.2f}, AUC={summary.auc:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hss_distribution.png", dpi=150)
    plt.close()

    # Figure 2: Score Components
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (score_name, good_vals, bad_vals, sep) in zip(axes, [
        ("SNS", [r.sns for r in good], [r.sns for r in bad], summary.sns_separation),
        ("PCS", [r.pcs for r in good], [r.pcs for r in bad], summary.pcs_separation),
        ("DRS", [r.drs for r in good], [r.drs for r in bad], summary.drs_separation),
    ]):
        ax.boxplot([good_vals, bad_vals], labels=["Good", "Bad"])
        ax.set_ylabel(score_name)
        ax.set_title(f"{score_name} (d={sep:.2f})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Score Component Separation")
    plt.tight_layout()
    plt.savefig(output_dir / "score_components.png", dpi=150)
    plt.close()

    # Figure 3: Betti Numbers
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (betti_name, good_vals, bad_vals) in zip(axes, [
        ("β₀ (Components)", [r.betti_0 for r in good], [r.betti_0 for r in bad]),
        ("β₁ (Cycles)", [r.betti_1 for r in good], [r.betti_1 for r in bad]),
    ]):
        ax.boxplot([good_vals, bad_vals], labels=["Good", "Bad"])
        ax.set_ylabel(betti_name)
        ax.set_title(betti_name)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Betti Number Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "betti_numbers.png", dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


# ============================================================================
# Output
# ============================================================================

def generate_report(
    results: List[ValidationResult],
    summary: ValidationSummary,
    output_dir: Path,
) -> None:
    """Generate markdown summary report."""
    report = f"""# TDA Mind Scanner Validation Report

## Summary

| Metric | Value |
|--------|-------|
| Good DAGs | {summary.num_good} |
| Bad DAGs | {summary.num_bad} |
| Marginal DAGs | {summary.num_marginal} |
| **Cohen's d** | **{summary.cohens_d:.3f}** |
| **AUC** | **{summary.auc:.3f}** |
| Optimal Threshold | {summary.optimal_threshold:.3f} |

## HSS Distribution

| Class | Mean | Std |
|-------|------|-----|
| Good | {summary.good_hss_mean:.3f} | {summary.good_hss_std:.3f} |
| Bad | {summary.bad_hss_mean:.3f} | {summary.bad_hss_std:.3f} |
| Marginal | {summary.marginal_hss_mean:.3f} | {summary.marginal_hss_std:.3f} |

## Score Component Separation (Cohen's d)

| Component | Cohen's d |
|-----------|-----------|
| SNS | {summary.sns_separation:.3f} |
| PCS | {summary.pcs_separation:.3f} |
| DRS | {summary.drs_separation:.3f} |

## Confusion Matrix at Optimal Threshold ({summary.optimal_threshold:.2f})

|  | Predicted Good | Predicted Bad |
|--|----------------|---------------|
| Actual Good | {summary.confusion_at_optimal.true_positives} | {summary.confusion_at_optimal.false_negatives} |
| Actual Bad | {summary.confusion_at_optimal.false_positives} | {summary.confusion_at_optimal.true_negatives} |

- Accuracy: {summary.confusion_at_optimal.accuracy:.3f}
- Precision: {summary.confusion_at_optimal.precision:.3f}
- Recall: {summary.confusion_at_optimal.recall:.3f}
- F1: {summary.confusion_at_optimal.f1:.3f}

## Confusion Matrix at Default Threshold (0.5)

|  | Predicted Good | Predicted Bad |
|--|----------------|---------------|
| Actual Good | {summary.confusion_at_default.true_positives} | {summary.confusion_at_default.false_negatives} |
| Actual Bad | {summary.confusion_at_default.false_positives} | {summary.confusion_at_default.true_negatives} |

- Accuracy: {summary.confusion_at_default.accuracy:.3f}
- F1: {summary.confusion_at_default.f1:.3f}

## Interpretation

"""
    # Add interpretation
    if summary.cohens_d > 1.5:
        report += "**STRONG SEPARATION**: Cohen's d > 1.5 indicates excellent discrimination.\n"
    elif summary.cohens_d > 1.0:
        report += "**GOOD SEPARATION**: Cohen's d > 1.0 indicates strong discrimination.\n"
    elif summary.cohens_d > 0.5:
        report += "**MODERATE SEPARATION**: Cohen's d > 0.5 indicates acceptable discrimination.\n"
    else:
        report += "**WEAK SEPARATION**: Cohen's d < 0.5 indicates poor discrimination. Review score formulas.\n"

    if summary.auc > 0.9:
        report += "**EXCELLENT AUC**: AUC > 0.9 indicates very high classification accuracy.\n"
    elif summary.auc > 0.8:
        report += "**GOOD AUC**: AUC > 0.8 indicates good classification accuracy.\n"
    elif summary.auc > 0.7:
        report += "**ACCEPTABLE AUC**: AUC > 0.7 indicates moderate classification accuracy.\n"
    else:
        report += "**POOR AUC**: AUC < 0.7 indicates classification needs improvement.\n"

    report += f"""
## Recommendations

- θ_block (BLOCK threshold): {min(0.3, summary.bad_hss_mean + 2 * summary.bad_hss_std):.2f}
- θ_warn (WARN threshold): {summary.optimal_threshold:.2f}
- N_ref (reference node count): {int(np.median([r.num_nodes for r in results if r.label == "good"]))}

## Generated

- Date: {Path(__file__).stat().st_mtime if Path(__file__).exists() else "N/A"}
- Seed: See calibration.json
- Spec Version: TDA_MIND_SCANNER_SPEC v0.1
"""

    with open(output_dir / "summary_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'summary_report.md'}")


def save_results(
    results: List[ValidationResult],
    summary: ValidationSummary,
    output_dir: Path,
    seed: int,
) -> None:
    """Save all results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration JSON
    good_results = [r for r in results if r.label == "good"]
    calibration = {
        "version": "1.0",
        "spec": "TDA_MIND_SCANNER_SPEC v0.1",
        "seed": seed,
        "n_ref": int(np.median([r.num_nodes for r in good_results])) if good_results else 50,
        "lifetime_threshold": 0.05,
        "deviation_max": 0.5,
        "recommended_thresholds": {
            "block": float(max(0.1, summary.bad_hss_mean + summary.bad_hss_std)),
            "warn": float(summary.optimal_threshold),
        },
        "validation_metrics": {
            "cohens_d": summary.cohens_d,
            "auc": summary.auc,
            "f1_at_optimal": summary.confusion_at_optimal.f1,
        },
        "sample_sizes": {
            "good": summary.num_good,
            "bad": summary.num_bad,
            "marginal": summary.num_marginal,
        },
    }

    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Full results
    results_data = {
        "results": [asdict(r) for r in results],
        "summary": {
            "num_good": summary.num_good,
            "num_bad": summary.num_bad,
            "num_marginal": summary.num_marginal,
            "good_hss": {"mean": summary.good_hss_mean, "std": summary.good_hss_std},
            "bad_hss": {"mean": summary.bad_hss_mean, "std": summary.bad_hss_std},
            "marginal_hss": {"mean": summary.marginal_hss_mean, "std": summary.marginal_hss_std},
            "cohens_d": summary.cohens_d,
            "auc": summary.auc,
            "optimal_threshold": summary.optimal_threshold,
            "confusion_at_optimal": summary.confusion_at_optimal.to_dict(),
            "confusion_at_default": summary.confusion_at_default.to_dict(),
            "confusion_at_block": summary.confusion_at_block.to_dict(),
            "sns_separation": summary.sns_separation,
            "pcs_separation": summary.pcs_separation,
            "drs_separation": summary.drs_separation,
        },
    }

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TDA Mind Scanner Synthetic Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=100,
        help="Number of DAGs per class (good, bad)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "tda_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TDA Mind Scanner - Synthetic Validation Framework")
    logger.info("=" * 60)
    logger.info(f"DAGs per class: {args.num_per_class}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output_dir}")

    # Generate and evaluate
    results = generate_and_evaluate(args.num_per_class, args.seed)
    logger.info(f"Generated {len(results)} total DAGs")

    # Compute summary
    summary = compute_summary(results)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Good HSS:     {summary.good_hss_mean:.3f} ± {summary.good_hss_std:.3f}")
    print(f"Bad HSS:      {summary.bad_hss_mean:.3f} ± {summary.bad_hss_std:.3f}")
    print(f"Cohen's d:    {summary.cohens_d:.3f}")
    print(f"AUC:          {summary.auc:.3f}")
    print(f"Optimal θ:    {summary.optimal_threshold:.3f}")
    print(f"F1 @ optimal: {summary.confusion_at_optimal.f1:.3f}")
    print("=" * 60)

    # Save outputs
    save_results(results, summary, args.output_dir, args.seed)
    generate_report(results, summary, args.output_dir)

    if not args.skip_viz:
        generate_visualizations(results, summary, args.output_dir)

    # Determine pass/fail
    if summary.cohens_d > 0.5 and summary.auc > 0.7:
        logger.info("VALIDATION PASSED: Adequate separation achieved")
        return 0
    else:
        logger.warning("VALIDATION WARNING: Separation may be insufficient")
        return 1


if __name__ == "__main__":
    sys.exit(main())
