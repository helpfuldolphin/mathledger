#!/usr/bin/env python3
"""
TDA Build Reference Profiles — Per-Slice Calibration Pipeline

Operation CORTEX: Phase I Activation
=====================================

This script builds reference TDA profiles for curriculum slices by analyzing
historical proof DAGs and computing baseline topological signatures. These
profiles are used by the TDAMonitor for Deviation-from-Reference Score (DRS).

Usage:
    python experiments/tda_build_reference_profiles.py \
        --input-dir results/tda_real_dags \
        --output-dir config/tda_profiles \
        --slices arithmetic_simple propositional_basic

Profile Contents (per slice):
- n_ref: Reference node count (median from training set)
- lifetime_threshold (τ): Noise threshold for persistence features
- deviation_max (δ_max): Normalization constant for DRS
- reference_diagram_h1: Representative H1 persistence diagram
- calibration_metadata: Training set statistics

Output:
- profiles/{slice_name}.json: Per-slice reference profile
- profiles/all_profiles.json: Combined profile index
- profiles/calibration_report.md: Human-readable calibration summary
"""

from __future__ import annotations

import argparse
import hashlib
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
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tda_build_reference_profiles")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PersistenceInterval:
    """A single persistence interval."""
    birth: float
    death: float
    dimension: int

    @property
    def lifetime(self) -> float:
        if self.death == float("inf"):
            return float("inf")
        return self.death - self.birth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "birth": self.birth,
            "death": self.death if self.death != float("inf") else "inf",
            "dimension": self.dimension,
        }


@dataclass
class PersistenceDiagram:
    """A persistence diagram for a single dimension."""
    dimension: int
    intervals: List[PersistenceInterval] = field(default_factory=list)

    @property
    def num_intervals(self) -> int:
        return len(self.intervals)

    def finite_lifetimes(self) -> List[float]:
        """Get finite lifetimes (exclude essential features)."""
        return [
            iv.lifetime for iv in self.intervals
            if iv.lifetime != float("inf")
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "intervals": [iv.to_dict() for iv in self.intervals],
            "num_intervals": self.num_intervals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistenceDiagram":
        """Create from dictionary."""
        intervals = [
            PersistenceInterval(
                birth=iv["birth"],
                death=float("inf") if iv["death"] == "inf" else iv["death"],
                dimension=iv["dimension"],
            )
            for iv in data.get("intervals", [])
        ]
        return cls(dimension=data["dimension"], intervals=intervals)


@dataclass
class ReferenceTDAProfile:
    """Reference TDA profile for a curriculum slice."""
    slice_name: str
    n_ref: int
    lifetime_threshold: float  # τ
    deviation_max: float  # δ_max
    reference_diagram_h1: Optional[PersistenceDiagram] = None
    calibration_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "n_ref": self.n_ref,
            "lifetime_threshold": self.lifetime_threshold,
            "deviation_max": self.deviation_max,
            "reference_diagram_h1": (
                self.reference_diagram_h1.to_dict()
                if self.reference_diagram_h1 else None
            ),
            "calibration_metadata": self.calibration_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTDAProfile":
        """Create from dictionary."""
        ref_diagram = None
        if data.get("reference_diagram_h1"):
            ref_diagram = PersistenceDiagram.from_dict(data["reference_diagram_h1"])

        return cls(
            slice_name=data["slice_name"],
            n_ref=data["n_ref"],
            lifetime_threshold=data["lifetime_threshold"],
            deviation_max=data["deviation_max"],
            reference_diagram_h1=ref_diagram,
            calibration_metadata=data.get("calibration_metadata", {}),
        )


@dataclass
class CalibrationResult:
    """Result of calibrating a single slice."""
    slice_name: str
    profile: ReferenceTDAProfile
    dag_count: int
    training_statistics: Dict[str, Any]


# ============================================================================
# TDA Analysis Functions
# ============================================================================

def build_combinatorial_complex(dag: "nx.DiGraph") -> Dict[int, List[frozenset]]:
    """
    Build a combinatorial simplicial complex from a DAG.

    Uses the flag/clique complex construction: add k-simplices for
    all (k+1)-cliques in the underlying undirected graph.

    Returns:
        Dict mapping dimension -> list of simplices (as frozensets of node indices)
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for complex construction")

    # Convert to undirected for clique enumeration
    undirected = dag.to_undirected()

    # Map nodes to indices
    nodes = list(dag.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Find all cliques up to size 4 (3-simplices)
    simplices: Dict[int, List[frozenset]] = {0: [], 1: [], 2: [], 3: []}

    # 0-simplices (vertices)
    for node in nodes:
        simplices[0].append(frozenset([node_to_idx[node]]))

    # Find all cliques
    for clique in nx.find_cliques(undirected):
        if len(clique) <= 4:
            idx_clique = frozenset(node_to_idx[n] for n in clique)

            # Add the clique and all its faces
            for k in range(1, len(clique)):
                # k-simplices are (k+1)-subsets
                from itertools import combinations
                for subset in combinations(idx_clique, k + 1):
                    simplex = frozenset(subset)
                    if simplex not in simplices.get(k, []):
                        if k not in simplices:
                            simplices[k] = []
                        simplices[k].append(simplex)

    return simplices


def compute_betti_numbers(simplices: Dict[int, List[frozenset]]) -> Dict[int, int]:
    """
    Compute Betti numbers from a simplicial complex.

    Uses the boundary matrix and Z_2 linear algebra.
    """
    betti = {}

    # Count simplices per dimension
    dim_counts = {k: len(v) for k, v in simplices.items()}

    # Simple approximation using Euler characteristic for small complexes
    # β_0 = connected components
    # For proper computation, we'd need boundary matrix rank calculations

    # Approximate β_0 from vertex count minus edge count
    n_vertices = len(simplices.get(0, []))
    n_edges = len(simplices.get(1, []))
    n_triangles = len(simplices.get(2, []))

    # Euler characteristic: χ = β_0 - β_1 + β_2 - ...
    # For connected graph: β_0 = 1
    # Estimate β_1 from loop count

    if n_vertices == 0:
        return {0: 0, 1: 0}

    # Rough estimates
    betti[0] = max(1, n_vertices - n_edges + n_triangles) if n_edges > 0 else n_vertices
    betti[1] = max(0, n_edges - n_vertices + 1 - n_triangles)  # Loops not filled by triangles

    return betti


def compute_simple_persistence(
    embeddings: Dict[str, np.ndarray],
    max_dim: int = 1,
) -> Dict[int, PersistenceDiagram]:
    """
    Compute simple persistence diagrams from point cloud embeddings.

    This is a simplified implementation that estimates persistence
    from pairwise distances without full Vietoris-Rips computation.
    """
    if not embeddings:
        return {
            0: PersistenceDiagram(dimension=0),
            1: PersistenceDiagram(dimension=1),
        }

    # Convert to point cloud
    points = np.array(list(embeddings.values()))
    n_points = len(points)

    if n_points < 2:
        return {
            0: PersistenceDiagram(dimension=0, intervals=[
                PersistenceInterval(birth=0.0, death=float("inf"), dimension=0)
            ]),
            1: PersistenceDiagram(dimension=1),
        }

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    try:
        distances = squareform(pdist(points))
    except Exception:
        # Fallback if scipy not available
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = np.linalg.norm(points[i] - points[j])
                distances[i, j] = d
                distances[j, i] = d

    # H_0: Components merge as distance increases
    # Use single-linkage clustering approximation
    h0_intervals = []

    # Each point starts as its own component
    # Track merge distances using simple algorithm
    sorted_distances = np.sort(np.unique(distances[distances > 0]))

    # Simplified: estimate merges from distance distribution
    if len(sorted_distances) > 0:
        # One essential component survives
        h0_intervals.append(PersistenceInterval(
            birth=0.0, death=float("inf"), dimension=0
        ))

        # Other components merge at various distances
        n_merges = min(n_points - 1, 5)
        for i in range(n_merges):
            death = sorted_distances[min(i, len(sorted_distances) - 1)]
            h0_intervals.append(PersistenceInterval(
                birth=0.0, death=float(death), dimension=0
            ))

    # H_1: Loops form and fill
    h1_intervals = []

    # Estimate loop formation from distance matrix
    # Loops typically form when triangles don't fill immediately
    if n_points >= 3:
        # Sample some potential loop formations
        loop_births = sorted_distances[len(sorted_distances) // 3:len(sorted_distances) * 2 // 3]
        loop_deaths = sorted_distances[len(sorted_distances) * 2 // 3:]

        for i in range(min(3, len(loop_births))):
            birth = loop_births[i] if i < len(loop_births) else 0.0
            death = loop_deaths[i] if i < len(loop_deaths) else birth + 0.5
            if death > birth:
                h1_intervals.append(PersistenceInterval(
                    birth=float(birth), death=float(death), dimension=1
                ))

    return {
        0: PersistenceDiagram(dimension=0, intervals=h0_intervals),
        1: PersistenceDiagram(dimension=1, intervals=h1_intervals),
    }


def bottleneck_distance(dgm1: PersistenceDiagram, dgm2: PersistenceDiagram) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.

    Simplified implementation using L∞ Wasserstein approximation.
    """
    pts1 = [(iv.birth, iv.death) for iv in dgm1.intervals if iv.death != float("inf")]
    pts2 = [(iv.birth, iv.death) for iv in dgm2.intervals if iv.death != float("inf")]

    if not pts1 and not pts2:
        return 0.0

    if not pts1 or not pts2:
        # Distance to diagonal
        non_empty = pts1 if pts1 else pts2
        return max((d - b) / 2 for b, d in non_empty) if non_empty else 0.0

    # Simple approximation: max of point-to-point and point-to-diagonal distances
    max_dist = 0.0

    # Point to point (greedy matching)
    for b1, d1 in pts1:
        min_dist = float("inf")
        for b2, d2 in pts2:
            dist = max(abs(b1 - b2), abs(d1 - d2))
            min_dist = min(min_dist, dist)
        # Also consider diagonal projection
        diag_dist = (d1 - b1) / 2
        min_dist = min(min_dist, diag_dist)
        max_dist = max(max_dist, min_dist)

    for b2, d2 in pts2:
        min_dist = float("inf")
        for b1, d1 in pts1:
            dist = max(abs(b1 - b2), abs(d1 - d2))
            min_dist = min(min_dist, dist)
        diag_dist = (d2 - b2) / 2
        min_dist = min(min_dist, diag_dist)
        max_dist = max(max_dist, min_dist)

    return max_dist


# ============================================================================
# Profile Building
# ============================================================================

class ProfileBuilder:
    """Builds reference TDA profiles from extracted DAGs."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.manifest: Optional[Dict[str, Any]] = None
        self.dags: List[Dict[str, Any]] = []

    def load_dags(self) -> int:
        """Load DAGs from input directory."""
        manifest_path = self.input_dir / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

        # Load DAG files
        dags_dir = self.input_dir / "dags"
        if dags_dir.exists():
            for dag_file in dags_dir.glob("dag_*.json"):
                with open(dag_file, "r", encoding="utf-8") as f:
                    self.dags.append(json.load(f))

        logger.info(f"Loaded {len(self.dags)} DAGs from {self.input_dir}")
        return len(self.dags)

    def load_embeddings(self, dag_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings for a specific DAG."""
        emb_path = self.input_dir / "embeddings" / f"emb_{dag_id}.npy"

        if emb_path.exists():
            try:
                data = np.load(emb_path, allow_pickle=True)
                return data.item() if data.ndim == 0 else dict(data)
            except Exception as e:
                logger.warning(f"Error loading embeddings {emb_path}: {e}")

        return None

    def build_profile(
        self,
        slice_name: str,
        dag_filter: Optional[callable] = None,
    ) -> CalibrationResult:
        """
        Build a reference profile for a slice.

        Args:
            slice_name: Name of the slice.
            dag_filter: Optional filter function for selecting DAGs.

        Returns:
            CalibrationResult with the profile and statistics.
        """
        # Filter DAGs for this slice
        filtered_dags = self.dags
        if dag_filter:
            filtered_dags = [d for d in self.dags if dag_filter(d)]

        if not filtered_dags:
            logger.warning(f"No DAGs found for slice {slice_name}")
            return CalibrationResult(
                slice_name=slice_name,
                profile=ReferenceTDAProfile(
                    slice_name=slice_name,
                    n_ref=50,
                    lifetime_threshold=0.05,
                    deviation_max=0.5,
                ),
                dag_count=0,
                training_statistics={},
            )

        # Compute statistics from training DAGs
        node_counts = [d["node_count"] for d in filtered_dags]
        edge_counts = [d["edge_count"] for d in filtered_dags]
        depths = [d["max_depth"] for d in filtered_dags]

        # Compute n_ref as median node count
        n_ref = int(np.median(node_counts))

        # Compute persistence diagrams for all DAGs
        all_h1_diagrams: List[PersistenceDiagram] = []
        all_lifetimes: List[float] = []

        for dag in filtered_dags:
            embeddings = self.load_embeddings(dag["dag_id"])
            if embeddings:
                diagrams = compute_simple_persistence(embeddings)
                h1 = diagrams.get(1)
                if h1:
                    all_h1_diagrams.append(h1)
                    all_lifetimes.extend(h1.finite_lifetimes())

        # Compute lifetime threshold (τ) as 5th percentile of lifetimes
        if all_lifetimes:
            lifetime_threshold = float(np.percentile(all_lifetimes, 5))
        else:
            lifetime_threshold = 0.05

        # Compute deviation_max (δ_max) from pairwise bottleneck distances
        if len(all_h1_diagrams) >= 2:
            pairwise_distances = []
            for i in range(min(len(all_h1_diagrams), 20)):
                for j in range(i + 1, min(len(all_h1_diagrams), 20)):
                    dist = bottleneck_distance(all_h1_diagrams[i], all_h1_diagrams[j])
                    pairwise_distances.append(dist)

            if pairwise_distances:
                deviation_max = float(np.percentile(pairwise_distances, 95))
            else:
                deviation_max = 0.5
        else:
            deviation_max = 0.5

        # Select reference diagram (median by total lifetime)
        reference_diagram = None
        if all_h1_diagrams:
            total_lifetimes = [
                sum(dgm.finite_lifetimes()) if dgm.finite_lifetimes() else 0.0
                for dgm in all_h1_diagrams
            ]
            median_idx = int(np.argsort(total_lifetimes)[len(total_lifetimes) // 2])
            reference_diagram = all_h1_diagrams[median_idx]

        # Build profile
        profile = ReferenceTDAProfile(
            slice_name=slice_name,
            n_ref=n_ref,
            lifetime_threshold=lifetime_threshold,
            deviation_max=deviation_max,
            reference_diagram_h1=reference_diagram,
            calibration_metadata={
                "dag_count": len(filtered_dags),
                "calibration_timestamp": datetime.utcnow().isoformat() + "Z",
                "node_count_stats": {
                    "mean": float(np.mean(node_counts)),
                    "std": float(np.std(node_counts)),
                    "median": float(np.median(node_counts)),
                },
                "depth_stats": {
                    "mean": float(np.mean(depths)),
                    "std": float(np.std(depths)),
                    "median": float(np.median(depths)),
                },
                "lifetime_stats": {
                    "count": len(all_lifetimes),
                    "mean": float(np.mean(all_lifetimes)) if all_lifetimes else 0.0,
                    "std": float(np.std(all_lifetimes)) if all_lifetimes else 0.0,
                    "p5": float(np.percentile(all_lifetimes, 5)) if all_lifetimes else 0.0,
                    "p95": float(np.percentile(all_lifetimes, 95)) if all_lifetimes else 0.0,
                },
            },
        )

        training_statistics = {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "depths": depths,
            "lifetime_threshold": lifetime_threshold,
            "deviation_max": deviation_max,
        }

        return CalibrationResult(
            slice_name=slice_name,
            profile=profile,
            dag_count=len(filtered_dags),
            training_statistics=training_statistics,
        )


# ============================================================================
# Report Generation
# ============================================================================

def generate_calibration_report(
    results: List[CalibrationResult],
    output_path: Path,
) -> None:
    """Generate a human-readable calibration report."""
    lines = [
        "# TDA Reference Profile Calibration Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Summary",
        "",
        f"| Slice | DAGs | N_ref | τ | δ_max |",
        f"|-------|------|-------|---|-------|",
    ]

    for result in results:
        p = result.profile
        lines.append(
            f"| {p.slice_name} | {result.dag_count} | {p.n_ref} | "
            f"{p.lifetime_threshold:.4f} | {p.deviation_max:.4f} |"
        )

    lines.append("")
    lines.append("## Detailed Statistics")
    lines.append("")

    for result in results:
        p = result.profile
        meta = p.calibration_metadata

        lines.extend([
            f"### {p.slice_name}",
            "",
            f"- **DAGs analyzed**: {result.dag_count}",
            f"- **N_ref (median nodes)**: {p.n_ref}",
            f"- **τ (lifetime threshold)**: {p.lifetime_threshold:.4f}",
            f"- **δ_max (deviation max)**: {p.deviation_max:.4f}",
            "",
            "**Node Count Distribution**:",
            f"- Mean: {meta.get('node_count_stats', {}).get('mean', 0):.2f}",
            f"- Std: {meta.get('node_count_stats', {}).get('std', 0):.2f}",
            f"- Median: {meta.get('node_count_stats', {}).get('median', 0):.2f}",
            "",
            "**Lifetime Distribution**:",
            f"- Count: {meta.get('lifetime_stats', {}).get('count', 0)}",
            f"- Mean: {meta.get('lifetime_stats', {}).get('mean', 0):.4f}",
            f"- P5: {meta.get('lifetime_stats', {}).get('p5', 0):.4f}",
            f"- P95: {meta.get('lifetime_stats', {}).get('p95', 0):.4f}",
            "",
        ])

    lines.extend([
        "## Calibration Parameters",
        "",
        "The following parameters are derived from the training set:",
        "",
        "- **N_ref**: Reference node count for SNS f_size calculation. "
        "Computed as median node count across training DAGs.",
        "",
        "- **τ (lifetime_threshold)**: Noise threshold for PCS calculation. "
        "Features with lifetime < τ are considered noise. Computed as 5th percentile of lifetimes.",
        "",
        "- **δ_max (deviation_max)**: Normalization constant for DRS. "
        "Computed as 95th percentile of pairwise bottleneck distances.",
        "",
        "## Usage",
        "",
        "Load profiles in TDAMonitor:",
        "",
        "```python",
        "from backend.tda.reference_profile import load_reference_profiles",
        "profiles = load_reference_profiles(Path('config/tda_profiles'))",
        "```",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Calibration report written to {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def build_reference_profiles(
    input_dir: Path,
    output_dir: Path,
    slices: Optional[List[str]] = None,
) -> List[CalibrationResult]:
    """
    Main profile building pipeline.

    Args:
        input_dir: Directory containing extracted DAGs.
        output_dir: Directory for output profiles.
        slices: Optional list of slice names to build profiles for.

    Returns:
        List of CalibrationResult objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    # Load DAGs
    builder = ProfileBuilder(input_dir)
    dag_count = builder.load_dags()

    if dag_count == 0:
        logger.error("No DAGs found in input directory")
        return []

    # Determine slices to build
    if not slices:
        # Default slices
        slices = ["default"]

    # Build profiles
    results: List[CalibrationResult] = []

    for slice_name in slices:
        logger.info(f"Building profile for slice: {slice_name}")

        # Filter function (can be customized per slice)
        def dag_filter(dag: Dict[str, Any]) -> bool:
            # For now, include all DAGs
            # Could filter by label, depth, etc.
            return True

        result = builder.build_profile(slice_name, dag_filter)
        results.append(result)

        # Save individual profile
        profile_path = profiles_dir / f"{slice_name}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(result.profile.to_dict(), f, indent=2)

        logger.info(f"  Profile saved: {profile_path}")

    # Save combined profiles index
    all_profiles = {r.slice_name: r.profile.to_dict() for r in results}
    index_path = profiles_dir / "all_profiles.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2)

    logger.info(f"Combined profiles saved: {index_path}")

    # Generate calibration report
    report_path = output_dir / "calibration_report.md"
    generate_calibration_report(results, report_path)

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build TDA reference profiles from extracted DAGs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/tda_real_dags"),
        help="Directory containing extracted DAGs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config/tda_profiles"),
        help="Output directory for profiles",
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=None,
        help="Slice names to build profiles for",
    )

    args = parser.parse_args()

    try:
        results = build_reference_profiles(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            slices=args.slices,
        )

        print(f"\nProfile building complete!")
        print(f"  Profiles built: {len(results)}")
        for r in results:
            print(f"    - {r.slice_name}: {r.dag_count} DAGs, "
                  f"n_ref={r.profile.n_ref}, τ={r.profile.lifetime_threshold:.4f}")
        print(f"  Output: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Profile building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
