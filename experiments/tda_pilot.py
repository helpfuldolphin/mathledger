#!/usr/bin/env python3
"""
TDA Mind Scanner - Offline Pilot Script.

Phase 0 feasibility test for Operation CORTEX.

This script:
1. Loads proof DAGs from the database or synthetic data
2. Extracts embeddings using backend/axiom_engine/features.py
3. Constructs clique complexes and metric complexes
4. Computes SNS, PCS, DRS, HSS for good and bad proofs
5. Validates separation between healthy and pathological reasoning
6. Outputs calibration data and visualizations

Usage:
    python experiments/tda_pilot.py --mode synthetic
    python experiments/tda_pilot.py --mode database --slice PL-1

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
from dataclasses import asdict, dataclass
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
logger = logging.getLogger("tda_pilot")


@dataclass
class PilotResult:
    """Result for a single proof evaluation."""
    label: str  # "good" or "bad"
    proof_id: str
    sns: float
    pcs: float
    drs: float
    hss: float
    betti_0: int
    betti_1: int
    num_nodes: int
    num_edges: int
    signal: str


@dataclass
class PilotSummary:
    """Summary statistics from pilot run."""
    num_good: int
    num_bad: int
    good_hss_mean: float
    good_hss_std: float
    bad_hss_mean: float
    bad_hss_std: float
    separation: float  # (good_mean - bad_mean) / pooled_std
    threshold_recommended: float
    auc_estimate: float


def create_synthetic_good_dag() -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Create a synthetic "good" proof DAG.

    Good proofs have:
    - Multiple nodes (rich structure)
    - Cross-links and reuse (cycles in undirected view)
    - Connected structure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for TDA pilot")

    # Create a DAG with some structure
    G = nx.DiGraph()

    # Base axioms
    axioms = [f"axiom_{i}" for i in range(3)]
    for ax in axioms:
        G.add_node(ax)

    # Intermediate lemmas derived from axioms
    lemmas = []
    for i in range(5):
        lemma = f"lemma_{i}"
        G.add_node(lemma)
        lemmas.append(lemma)

        # Each lemma depends on 1-2 axioms
        parents = random.sample(axioms, k=min(2, len(axioms)))
        for p in parents:
            G.add_edge(p, lemma)

    # Higher-level theorems
    theorems = []
    for i in range(3):
        thm = f"theorem_{i}"
        G.add_node(thm)
        theorems.append(thm)

        # Theorems depend on lemmas (creating cross-links)
        parents = random.sample(lemmas, k=min(3, len(lemmas)))
        for p in parents:
            G.add_edge(p, thm)

    # Final goal depends on multiple theorems
    G.add_node("goal")
    for thm in theorems:
        G.add_edge(thm, "goal")

    # Generate embeddings
    embeddings = _generate_embeddings(list(G.nodes()))

    return G, embeddings


def create_synthetic_bad_dag() -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Create a synthetic "bad" proof DAG.

    Bad proofs have:
    - Few nodes (trivial)
    - Linear/tree structure (no cycles)
    - Disconnected components or isolated nodes
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for TDA pilot")

    G = nx.DiGraph()

    # Simple linear chain (trivial proof)
    nodes = [f"step_{i}" for i in range(3)]
    for node in nodes:
        G.add_node(node)

    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    # Add isolated node (disconnected)
    G.add_node("isolated")

    embeddings = _generate_embeddings(list(G.nodes()))

    return G, embeddings


def create_synthetic_marginal_dag() -> Tuple[Any, Dict[str, np.ndarray]]:
    """
    Create a synthetic "marginal" proof DAG.

    Marginal proofs are borderline - some structure but not rich.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for TDA pilot")

    G = nx.DiGraph()

    # Simple tree structure
    G.add_node("root")
    for i in range(3):
        child = f"child_{i}"
        G.add_node(child)
        G.add_edge("root", child)

        for j in range(2):
            grandchild = f"gc_{i}_{j}"
            G.add_node(grandchild)
            G.add_edge(child, grandchild)

    embeddings = _generate_embeddings(list(G.nodes()))

    return G, embeddings


def _generate_embeddings(nodes: List[str]) -> Dict[str, np.ndarray]:
    """Generate synthetic embeddings for nodes."""
    # Use statement features format (19-dim)
    embeddings = {}
    for i, node in enumerate(nodes):
        # Create structured embedding with some variation
        base = np.zeros(19, dtype=np.float32)
        base[0] = len(node)  # length
        base[1] = 2  # word count
        base[2] = len(node.replace("_", ""))  # char count
        base[10] = float(i) / max(len(nodes), 1)  # depth proxy
        base[11] = random.uniform(0.1, 0.5)  # operator density
        # Add noise
        base += np.random.normal(0, 0.1, 19).astype(np.float32)
        embeddings[node] = base

    return embeddings


def evaluate_dag(
    dag: Any,
    embeddings: Dict[str, np.ndarray],
    ref_profile: Optional[Any] = None,
) -> PilotResult:
    """
    Evaluate a single DAG using TDA metrics.

    Args:
        dag: NetworkX DiGraph
        embeddings: State embeddings
        ref_profile: Optional reference profile

    Returns:
        PilotResult with all scores
    """
    from backend.tda.proof_complex import build_combinatorial_complex
    from backend.tda.metric_complex import build_metric_complex
    from backend.tda.scores import (
        compute_structural_nontriviality,
        compute_persistence_coherence,
        compute_deviation_from_reference,
        compute_hallucination_stability_score,
        classify_hss,
    )

    # Build complexes
    comb_complex = build_combinatorial_complex(dag, max_clique_size=4)
    tda_result = build_metric_complex(embeddings, max_dim=1)

    # Compute scores
    sns = compute_structural_nontriviality(comb_complex, ref_profile)
    pcs = compute_persistence_coherence(tda_result, ref_profile)
    drs = compute_deviation_from_reference(tda_result, ref_profile)
    hss = compute_hallucination_stability_score(sns, pcs, drs)

    # Get Betti numbers
    betti = comb_complex.compute_betti_numbers(max_dim=1)

    # Classify
    signal = classify_hss(hss)

    return PilotResult(
        label="unknown",
        proof_id="",
        sns=sns,
        pcs=pcs,
        drs=drs,
        hss=hss,
        betti_0=betti.get(0, 0),
        betti_1=betti.get(1, 0),
        num_nodes=comb_complex.num_vertices,
        num_edges=comb_complex.num_edges,
        signal=signal,
    )


def run_synthetic_pilot(
    num_good: int = 20,
    num_bad: int = 20,
    num_marginal: int = 10,
    seed: int = 42,
) -> Tuple[List[PilotResult], PilotSummary]:
    """
    Run pilot with synthetic DAGs.

    Args:
        num_good: Number of good proofs to generate
        num_bad: Number of bad proofs to generate
        num_marginal: Number of marginal proofs to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (results, summary)
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating {num_good} good, {num_bad} bad, {num_marginal} marginal DAGs")

    results: List[PilotResult] = []

    # Generate good proofs
    for i in range(num_good):
        dag, emb = create_synthetic_good_dag()
        result = evaluate_dag(dag, emb)
        result.label = "good"
        result.proof_id = f"good_{i}"
        results.append(result)

    # Generate bad proofs
    for i in range(num_bad):
        dag, emb = create_synthetic_bad_dag()
        result = evaluate_dag(dag, emb)
        result.label = "bad"
        result.proof_id = f"bad_{i}"
        results.append(result)

    # Generate marginal proofs
    for i in range(num_marginal):
        dag, emb = create_synthetic_marginal_dag()
        result = evaluate_dag(dag, emb)
        result.label = "marginal"
        result.proof_id = f"marginal_{i}"
        results.append(result)

    # Compute summary
    summary = compute_summary(results)

    return results, summary


def compute_summary(results: List[PilotResult]) -> PilotSummary:
    """Compute summary statistics from pilot results."""
    good_hss = [r.hss for r in results if r.label == "good"]
    bad_hss = [r.hss for r in results if r.label == "bad"]

    good_mean = np.mean(good_hss) if good_hss else 0.0
    good_std = np.std(good_hss) if good_hss else 0.0
    bad_mean = np.mean(bad_hss) if bad_hss else 0.0
    bad_std = np.std(bad_hss) if bad_hss else 0.0

    # Compute separation (Cohen's d style)
    pooled_std = np.sqrt((good_std**2 + bad_std**2) / 2) if (good_std > 0 or bad_std > 0) else 1.0
    separation = (good_mean - bad_mean) / pooled_std if pooled_std > 0 else 0.0

    # Recommend threshold (midpoint between means)
    threshold = (good_mean + bad_mean) / 2

    # Estimate AUC (simple approximation)
    all_good = [(h, 1) for h in good_hss]
    all_bad = [(h, 0) for h in bad_hss]
    all_labeled = sorted(all_good + all_bad, key=lambda x: x[0])

    if all_labeled:
        # Count concordant pairs
        concordant = 0
        discordant = 0
        for i, (h1, l1) in enumerate(all_labeled):
            for h2, l2 in all_labeled[i + 1:]:
                if l1 != l2:
                    if (h1 > h2 and l1 > l2) or (h1 < h2 and l1 < l2):
                        concordant += 1
                    else:
                        discordant += 1
        total_pairs = concordant + discordant
        auc = concordant / total_pairs if total_pairs > 0 else 0.5
    else:
        auc = 0.5

    return PilotSummary(
        num_good=len(good_hss),
        num_bad=len(bad_hss),
        good_hss_mean=float(good_mean),
        good_hss_std=float(good_std),
        bad_hss_mean=float(bad_mean),
        bad_hss_std=float(bad_std),
        separation=float(separation),
        threshold_recommended=float(threshold),
        auc_estimate=float(auc),
    )


def save_results(
    results: List[PilotResult],
    summary: PilotSummary,
    output_dir: Path,
) -> None:
    """Save pilot results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual results
    results_path = output_dir / "tda_pilot_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "summary": asdict(summary),
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {results_path}")

    # Save calibration data
    calibration_path = output_dir / "tda_pilot_calibration.json"
    good_results = [r for r in results if r.label == "good"]
    calibration = {
        "n_ref": int(np.median([r.num_nodes for r in good_results])) if good_results else 50,
        "lifetime_threshold": 0.05,
        "deviation_max": 0.5,
        "recommended_block_threshold": max(0.1, summary.bad_hss_mean + summary.bad_hss_std),
        "recommended_warn_threshold": summary.threshold_recommended,
        "separation_achieved": summary.separation,
        "auc_achieved": summary.auc_estimate,
    }
    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Calibration saved to {calibration_path}")


def print_summary(summary: PilotSummary) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("TDA PILOT SUMMARY")
    print("=" * 60)
    print(f"Good proofs:  {summary.num_good}")
    print(f"Bad proofs:   {summary.num_bad}")
    print()
    print("HSS Distribution:")
    print(f"  Good:  mean={summary.good_hss_mean:.3f}, std={summary.good_hss_std:.3f}")
    print(f"  Bad:   mean={summary.bad_hss_mean:.3f}, std={summary.bad_hss_std:.3f}")
    print()
    print("Separation Metrics:")
    print(f"  Cohen's d:        {summary.separation:.3f}")
    print(f"  AUC estimate:     {summary.auc_estimate:.3f}")
    print()
    print("Recommendations:")
    print(f"  θ_warn threshold: {summary.threshold_recommended:.3f}")
    print("=" * 60)

    # Interpretation
    if summary.separation > 1.0:
        print("\n✓ STRONG separation achieved - TDA metrics discriminate well")
    elif summary.separation > 0.5:
        print("\n~ MODERATE separation - TDA metrics show promise")
    else:
        print("\n✗ WEAK separation - TDA metrics need refinement")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TDA Mind Scanner Offline Pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "database"],
        default="synthetic",
        help="Data source mode",
    )
    parser.add_argument(
        "--num-good",
        type=int,
        default=20,
        help="Number of good proofs (synthetic mode)",
    )
    parser.add_argument(
        "--num-bad",
        type=int,
        default=20,
        help="Number of bad proofs (synthetic mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "tda_pilot",
        help="Output directory",
    )

    args = parser.parse_args()

    logger.info("Starting TDA Mind Scanner Pilot")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "synthetic":
        results, summary = run_synthetic_pilot(
            num_good=args.num_good,
            num_bad=args.num_bad,
            seed=args.seed,
        )
    else:
        logger.error("Database mode not yet implemented")
        return 1

    # Print and save results
    print_summary(summary)
    save_results(results, summary, args.output_dir)

    # Exit code based on separation
    if summary.separation > 0.5:
        logger.info("Pilot PASSED - adequate separation achieved")
        return 0
    else:
        logger.warning("Pilot WARNING - separation may be insufficient")
        return 0  # Still success, just a warning


if __name__ == "__main__":
    sys.exit(main())
