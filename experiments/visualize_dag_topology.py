#!/usr/bin/env python
"""
PHASE II — DAG Topology Explorer & Extended Visualization Suite

Provides topology-aware visualizations of derivation DAGs for Group D analysis.

This module is an EXPLORATORY TOOL for Phase II analysis. It does NOT make
uplift claims; it exposes structural topology of derivation graphs across
baseline vs RFL runs.

ABSOLUTE SAFEGUARDS:
  - Visualizations must NOT produce uplift claims.
  - Must not alter log contents or metrics.
  - Must not alter experiment outputs.
  - No policy interpretation. No inference. Visualization only.

Primary Capabilities:
  - DAG adjacency matrix heatmaps
  - Layered depth-level graphs
  - Branching factor distributions
  - Longest path visualization
  - Side-by-side baseline vs RFL comparisons
  - DAG evolution timelines

Output Directory: artifacts/phase_ii/topology/
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

# Use non-interactive backend for headless environments (CI/servers)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

OUTPUT_DIR = "artifacts/phase_ii/topology"

# Fixed seed for determinism - NO RANDOMNESS in plotting
DETERMINISTIC_SEED = 42

# Color scheme
COLOR_BASELINE = "#666666"
COLOR_RFL = "#000000"
COLOR_ACCENT = "#4a90d9"
COLOR_SUCCESS = "#2d8a4e"
COLOR_ABSTENTION = "#c44536"

# Heatmap colormap (white to dark blue)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "dag_heatmap", ["#ffffff", "#e6f0ff", "#99c2ff", "#3385ff", "#0052cc"]
)

# -----------------------------------------------------------------------------
# Extended DAG Analyzer
# -----------------------------------------------------------------------------


class DAGTopologyAnalyzer:
    """
    Extended analyzer for DAG topology metrics.
    
    Computes structural properties beyond simple chain depth:
      - Adjacency matrix
      - Branching factors (in-degree, out-degree)
      - Depth level distribution
      - Longest paths
      - Connected components
    
    PHASE II ONLY - No policy interpretation.
    """
    
    def __init__(self, derivations: List[Dict[str, Any]]):
        """
        Initialize from derivation records.
        
        Args:
            derivations: List of dicts with 'hash' and 'premises' keys.
        """
        self._derivations = derivations
        self._dag: Dict[str, List[str]] = {}  # hash -> premises (parents)
        self._reverse_dag: Dict[str, List[str]] = {}  # hash -> children
        self._depth_cache: Dict[str, int] = {}
        self._node_order: List[str] = []
        
        self._build_dag()
    
    def _build_dag(self) -> None:
        """Build DAG and reverse DAG from derivations."""
        for d in self._derivations:
            h = d.get("hash")
            if not h:
                continue
            
            premises = d.get("premises", [])
            self._dag[h] = premises
            self._node_order.append(h)
            
            # Build reverse graph (children)
            if h not in self._reverse_dag:
                self._reverse_dag[h] = []
            for p in premises:
                if p not in self._reverse_dag:
                    self._reverse_dag[p] = []
                self._reverse_dag[p].append(h)
    
    @property
    def dag(self) -> Dict[str, List[str]]:
        """Get the DAG (hash -> premises)."""
        return self._dag
    
    @property
    def node_count(self) -> int:
        """Total number of nodes in the DAG."""
        return len(self._dag)
    
    @property
    def edge_count(self) -> int:
        """Total number of edges in the DAG."""
        return sum(len(premises) for premises in self._dag.values())
    
    def get_depth(self, h: str) -> int:
        """
        Compute maximum depth for a node (memoized).
        
        Depth = 1 + max(depth of premises)
        Axioms (no premises) have depth 1.
        """
        if h in self._depth_cache:
            return self._depth_cache[h]
        
        if h not in self._dag:
            self._depth_cache[h] = 1
            return 1
        
        premises = self._dag[h]
        if not premises:
            self._depth_cache[h] = 1
            return 1
        
        max_premise_depth = max((self.get_depth(p) for p in premises), default=0)
        depth = 1 + max_premise_depth
        self._depth_cache[h] = depth
        return depth
    
    def get_all_depths(self) -> Dict[str, int]:
        """Compute depths for all nodes."""
        return {h: self.get_depth(h) for h in self._dag}
    
    def get_depth_distribution(self) -> Dict[int, int]:
        """Get distribution of node depths."""
        depths = self.get_all_depths()
        dist: Dict[int, int] = defaultdict(int)
        for d in depths.values():
            dist[d] += 1
        return dict(sorted(dist.items()))
    
    def get_max_depth(self) -> int:
        """Get maximum depth in the DAG."""
        if not self._dag:
            return 0
        return max(self.get_depth(h) for h in self._dag)
    
    def get_in_degree(self, h: str) -> int:
        """Get in-degree (number of premises) for a node."""
        return len(self._dag.get(h, []))
    
    def get_out_degree(self, h: str) -> int:
        """Get out-degree (number of children) for a node."""
        return len(self._reverse_dag.get(h, []))
    
    def get_branching_factors(self) -> Dict[str, Dict[str, int]]:
        """
        Compute branching factors for all nodes.
        
        Returns:
            Dict mapping hash -> {'in_degree': int, 'out_degree': int}
        """
        return {
            h: {
                "in_degree": self.get_in_degree(h),
                "out_degree": self.get_out_degree(h),
            }
            for h in self._dag
        }
    
    def get_branching_distribution(self) -> Dict[str, Dict[int, int]]:
        """
        Get distribution of branching factors.
        
        Returns:
            {'in_degree': {degree: count}, 'out_degree': {degree: count}}
        """
        factors = self.get_branching_factors()
        
        in_dist: Dict[int, int] = defaultdict(int)
        out_dist: Dict[int, int] = defaultdict(int)
        
        for f in factors.values():
            in_dist[f["in_degree"]] += 1
            out_dist[f["out_degree"]] += 1
        
        return {
            "in_degree": dict(sorted(in_dist.items())),
            "out_degree": dict(sorted(out_dist.items())),
        }
    
    def get_longest_path(self, h: str) -> List[str]:
        """
        Get the longest path ending at node h.
        
        Returns list of hashes from root to h.
        """
        if h not in self._dag:
            return [h] if h else []
        
        premises = self._dag[h]
        if not premises:
            return [h]
        
        # Find premise with maximum depth
        max_depth = -1
        max_premise = None
        for p in premises:
            d = self.get_depth(p)
            if d > max_depth:
                max_depth = d
                max_premise = p
        
        if max_premise is None:
            return [h]
        
        return self.get_longest_path(max_premise) + [h]
    
    def get_nodes_at_depth(self, depth: int) -> List[str]:
        """Get all nodes at a specific depth level."""
        return [h for h in self._dag if self.get_depth(h) == depth]
    
    def get_adjacency_matrix(self, max_nodes: int = 100) -> Tuple[np.ndarray, List[str]]:
        """
        Build adjacency matrix for visualization.
        
        Args:
            max_nodes: Maximum number of nodes to include (for performance).
        
        Returns:
            Tuple of (adjacency_matrix, node_labels)
        """
        # Select nodes (first N by order, or sample if too many)
        nodes = self._node_order[:max_nodes]
        n = len(nodes)
        
        if n == 0:
            return np.zeros((0, 0)), []
        
        node_idx = {h: i for i, h in enumerate(nodes)}
        matrix = np.zeros((n, n), dtype=np.float32)
        
        for h in nodes:
            i = node_idx[h]
            for p in self._dag.get(h, []):
                if p in node_idx:
                    j = node_idx[p]
                    matrix[i, j] = 1.0
        
        return matrix, nodes
    
    def get_dag_footprint(self) -> Dict[str, Any]:
        """Get summary statistics of the DAG."""
        depths = self.get_all_depths()
        depth_values = list(depths.values()) if depths else [0]
        
        branching = self.get_branching_distribution()
        
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_depth": max(depth_values),
            "mean_depth": float(np.mean(depth_values)),
            "median_depth": float(np.median(depth_values)),
            "depth_distribution": self.get_depth_distribution(),
            "branching_distribution": branching,
        }


# -----------------------------------------------------------------------------
# Data Loading Utilities
# -----------------------------------------------------------------------------


def load_experiment_log(log_path: str | Path) -> List[Dict[str, Any]]:
    """Load experiment log from JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_derivations_from_log(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract derivation records with hash/premises from experiment log.
    
    Handles multiple log formats:
      - Records with explicit 'derivations' array
      - Records with 'derivation.candidate_hash' + inferred structure
      - Synthetic derivation generation from cycle sequence
    """
    derivations: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    
    for i, rec in enumerate(records):
        # Try explicit derivations array
        if "derivations" in rec:
            for d in rec["derivations"]:
                if d.get("hash") and d["hash"] not in seen_hashes:
                    derivations.append(d)
                    seen_hashes.add(d["hash"])
            continue
        
        # Try derivation block with candidate_hash
        deriv = rec.get("derivation", {})
        candidate_hash = deriv.get("candidate_hash")
        
        if candidate_hash and candidate_hash not in seen_hashes:
            # Infer premises from previous cycle's candidates
            premises = []
            if i > 0:
                prev_deriv = records[i - 1].get("derivation", {})
                prev_hash = prev_deriv.get("candidate_hash")
                if prev_hash and prev_hash != candidate_hash:
                    premises = [prev_hash]
            
            derivations.append({
                "hash": candidate_hash,
                "premises": premises,
                "cycle": rec.get("cycle", i),
            })
            seen_hashes.add(candidate_hash)
    
    return derivations


def extract_cycle_metrics(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Extract per-cycle metrics from experiment log.
    
    Returns dict of metric_name -> [values per cycle].
    """
    metrics: Dict[str, List[Any]] = {
        "cycle": [],
        "candidates": [],
        "verified": [],
        "abstained": [],
        "abstention": [],
        "success": [],
        "depth": [],
    }
    
    for rec in records:
        metrics["cycle"].append(rec.get("cycle", len(metrics["cycle"])))
        
        deriv = rec.get("derivation", {})
        metrics["candidates"].append(deriv.get("candidates", 0))
        metrics["verified"].append(deriv.get("verified", 0))
        metrics["abstained"].append(deriv.get("abstained", 0))
        metrics["depth"].append(deriv.get("depth", deriv.get("verified", 1)))
        
        # Abstention detection
        abstention = rec.get("abstention", False)
        if not abstention:
            abstention = deriv.get("abstained", 0) > 0
        metrics["abstention"].append(1 if abstention else 0)
        
        # Success detection
        success = rec.get("success", False)
        if not success:
            success = rec.get("status") == "verified"
        metrics["success"].append(1 if success else 0)
    
    return metrics


# -----------------------------------------------------------------------------
# Visualization Setup
# -----------------------------------------------------------------------------


def setup_style() -> None:
    """Apply consistent matplotlib style for topology plots."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('default')
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.2,
    })


def ensure_output_dir(path: str | Path = OUTPUT_DIR) -> Path:
    """Ensure output directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Core Topology Visualizations
# -----------------------------------------------------------------------------


def plot_adjacency_matrix_heatmap(
    analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "DAG Adjacency Matrix",
    max_nodes: int = 100,
) -> Path:
    """
    Generate adjacency matrix heatmap for DAG visualization.
    
    PHASE II ONLY - Visualization, no inference.
    
    Args:
        analyzer: DAGTopologyAnalyzer instance.
        out_path: Output path for image.
        title: Plot title.
        max_nodes: Maximum nodes to display.
    
    Returns:
        Path to saved figure.
    """
    setup_style()
    
    matrix, nodes = analyzer.get_adjacency_matrix(max_nodes)
    
    if matrix.size == 0:
        # Empty DAG - create placeholder
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Empty DAG", ha="center", va="center", fontsize=14)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(matrix, cmap=HEATMAP_CMAP, aspect="auto")
        
        ax.set_xlabel("Premise Index")
        ax.set_ylabel("Derivation Index")
        ax.set_title(f"{title}\n({len(nodes)} nodes, {analyzer.edge_count} edges)")
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Edge Present")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_depth_level_layers(
    analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "DAG Depth Levels",
    max_nodes_per_level: int = 20,
) -> Path:
    """
    Plot DAG as layered graph by depth level.
    
    PHASE II ONLY - Visualization, no inference.
    
    Args:
        analyzer: DAGTopologyAnalyzer instance.
        out_path: Output path for image.
        title: Plot title.
        max_nodes_per_level: Max nodes to show per depth level.
    
    Returns:
        Path to saved figure.
    """
    setup_style()
    
    max_depth = analyzer.get_max_depth()
    
    if max_depth == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Empty DAG", ha="center", va="center", fontsize=14)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(12, max(6, max_depth * 0.8)))
        
        # Collect nodes by depth
        level_nodes: Dict[int, List[str]] = defaultdict(list)
        for h in analyzer.dag:
            d = analyzer.get_depth(h)
            level_nodes[d].append(h)
        
        # Plot each level
        y_positions: Dict[str, float] = {}
        x_positions: Dict[str, float] = {}
        
        for depth in range(1, max_depth + 1):
            nodes = level_nodes[depth][:max_nodes_per_level]
            n_nodes = len(nodes)
            
            if n_nodes == 0:
                continue
            
            # Spread nodes horizontally
            x_spread = np.linspace(0, 1, n_nodes + 2)[1:-1] if n_nodes > 1 else [0.5]
            
            for i, h in enumerate(nodes):
                x = x_spread[i] if i < len(x_spread) else 0.5
                y = depth
                x_positions[h] = x
                y_positions[h] = y
                
                # Draw node
                ax.scatter(x, y, s=100, c=COLOR_ACCENT, zorder=3, alpha=0.8)
        
        # Draw edges
        for h in x_positions:
            for p in analyzer.dag.get(h, []):
                if p in x_positions:
                    ax.plot(
                        [x_positions[p], x_positions[h]],
                        [y_positions[p], y_positions[h]],
                        color=COLOR_BASELINE,
                        alpha=0.3,
                        linewidth=0.5,
                        zorder=1,
                    )
        
        ax.set_xlabel("Node Spread")
        ax.set_ylabel("Depth Level")
        ax.set_title(f"{title}\n(max depth: {max_depth})")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.5, max_depth + 0.5)
        ax.invert_yaxis()  # Depth 1 at top
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_branching_factor_distribution(
    analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "Branching Factor Distribution",
) -> Path:
    """
    Plot distribution of in-degree and out-degree (branching factors).
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    dist = analyzer.get_branching_distribution()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # In-degree distribution
    ax1 = axes[0]
    in_degrees = list(dist["in_degree"].keys())
    in_counts = list(dist["in_degree"].values())
    
    if in_degrees:
        ax1.bar(in_degrees, in_counts, color=COLOR_BASELINE, alpha=0.8, edgecolor="black")
    ax1.set_xlabel("In-Degree (# Premises)")
    ax1.set_ylabel("Node Count")
    ax1.set_title("In-Degree Distribution")
    
    # Out-degree distribution
    ax2 = axes[1]
    out_degrees = list(dist["out_degree"].keys())
    out_counts = list(dist["out_degree"].values())
    
    if out_degrees:
        ax2.bar(out_degrees, out_counts, color=COLOR_RFL, alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Out-Degree (# Children)")
    ax2.set_ylabel("Node Count")
    ax2.set_title("Out-Degree Distribution")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_longest_path_visualization(
    analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "Longest Derivation Path",
) -> Path:
    """
    Visualize the longest path in the DAG.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    # Find node with maximum depth
    max_depth = 0
    max_node = None
    for h in analyzer.dag:
        d = analyzer.get_depth(h)
        if d > max_depth:
            max_depth = d
            max_node = h
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if max_node is None:
        ax.text(0.5, 0.5, "Empty DAG", ha="center", va="center", fontsize=14)
    else:
        path = analyzer.get_longest_path(max_node)
        
        # Plot path as horizontal chain
        n = len(path)
        x_pos = np.linspace(0, 1, n)
        y_pos = [0.5] * n
        
        # Draw edges
        for i in range(n - 1):
            ax.plot(
                [x_pos[i], x_pos[i + 1]],
                [y_pos[i], y_pos[i + 1]],
                color=COLOR_BASELINE,
                linewidth=2,
                zorder=1,
            )
        
        # Draw nodes
        ax.scatter(x_pos, y_pos, s=200, c=COLOR_ACCENT, zorder=2, edgecolors="black")
        
        # Label depths
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            ax.annotate(
                f"d={i + 1}",
                (x, y - 0.15),
                ha="center",
                fontsize=9,
            )
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    
    ax.set_title(f"{title}\n(length: {max_depth})")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_depth_distribution_histogram(
    analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "Node Depth Distribution",
) -> Path:
    """
    Plot histogram of node depths.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    dist = analyzer.get_depth_distribution()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if dist:
        depths = list(dist.keys())
        counts = list(dist.values())
        ax.bar(depths, counts, color=COLOR_ACCENT, alpha=0.8, edgecolor="black")
        
        # Add statistics
        all_depths = []
        for d, c in dist.items():
            all_depths.extend([d] * c)
        
        mean_d = np.mean(all_depths)
        median_d = np.median(all_depths)
        
        stats_text = f"μ={mean_d:.2f}, median={median_d:.1f}, max={max(depths)}"
        ax.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Node Count")
    ax.set_title(title)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


# -----------------------------------------------------------------------------
# Side-by-Side Comparison Visualizations
# -----------------------------------------------------------------------------


def plot_node_count_comparison(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Node Count Over Cycles",
) -> Path:
    """
    Compare cumulative node count over cycles (baseline vs RFL).
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cumulative verified counts as proxy for node growth
    baseline_cum = np.cumsum(baseline_metrics["verified"])
    rfl_cum = np.cumsum(rfl_metrics["verified"])
    
    ax.plot(
        baseline_metrics["cycle"],
        baseline_cum,
        label="Baseline",
        color=COLOR_BASELINE,
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        rfl_metrics["cycle"],
        rfl_cum,
        label="RFL",
        color=COLOR_RFL,
        linewidth=2,
    )
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative Verified Derivations")
    ax.set_title(title)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_depth_distribution_comparison(
    baseline_analyzer: DAGTopologyAnalyzer,
    rfl_analyzer: DAGTopologyAnalyzer,
    out_path: str | Path,
    title: str = "Depth Distribution Comparison",
) -> Path:
    """
    Compare depth distributions between baseline and RFL DAGs.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    baseline_dist = baseline_analyzer.get_depth_distribution()
    rfl_dist = rfl_analyzer.get_depth_distribution()
    
    # Get all depths
    all_depths = set(baseline_dist.keys()) | set(rfl_dist.keys())
    if not all_depths:
        all_depths = {1}
    
    depths = sorted(all_depths)
    baseline_counts = [baseline_dist.get(d, 0) for d in depths]
    rfl_counts = [rfl_dist.get(d, 0) for d in depths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(depths))
    width = 0.35
    
    ax.bar(x - width / 2, baseline_counts, width, label="Baseline", color=COLOR_BASELINE, alpha=0.8)
    ax.bar(x + width / 2, rfl_counts, width, label="RFL", color=COLOR_RFL, alpha=0.8)
    
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Node Count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_chain_depth_trend(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Chain Depth Trend Over Cycles",
    window: int = 20,
) -> Path:
    """
    Compare rolling chain depth trend (baseline vs RFL).
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Rolling mean of depth
    baseline_depth = np.array(baseline_metrics["depth"], dtype=float)
    rfl_depth = np.array(rfl_metrics["depth"], dtype=float)
    
    if len(baseline_depth) >= window:
        baseline_smooth = np.convolve(baseline_depth, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(baseline_depth)),
            baseline_smooth,
            label=f"Baseline ({window}-cycle avg)",
            color=COLOR_BASELINE,
            linewidth=2,
            linestyle="--",
        )
    
    if len(rfl_depth) >= window:
        rfl_smooth = np.convolve(rfl_depth, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(rfl_depth)),
            rfl_smooth,
            label=f"RFL ({window}-cycle avg)",
            color=COLOR_RFL,
            linewidth=2,
        )
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Chain Depth")
    ax.set_title(title)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def generate_dag_evolution_timeline(
    records: List[Dict[str, Any]],
    out_dir: str | Path,
    prefix: str = "dag_frame",
    frame_interval: int = 100,
) -> List[Path]:
    """
    Generate DAG evolution timeline as series of PNG frames.
    
    Creates snapshots of DAG state at regular intervals.
    
    PHASE II ONLY - Visualization, no inference.
    
    Args:
        records: Experiment log records.
        out_dir: Output directory for frames.
        prefix: Filename prefix for frames.
        frame_interval: Cycles between frames.
    
    Returns:
        List of paths to generated frames.
    """
    setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    frames: List[Path] = []
    
    for i in range(0, len(records), frame_interval):
        # Build DAG from records up to this point
        partial_records = records[: i + 1]
        derivations = extract_derivations_from_log(partial_records)
        
        if not derivations:
            continue
        
        analyzer = DAGTopologyAnalyzer(derivations)
        footprint = analyzer.get_dag_footprint()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot depth distribution at this point
        dist = analyzer.get_depth_distribution()
        if dist:
            depths = list(dist.keys())
            counts = list(dist.values())
            ax.bar(depths, counts, color=COLOR_ACCENT, alpha=0.8)
        
        cycle = partial_records[-1].get("cycle", i)
        ax.set_xlabel("Depth Level")
        ax.set_ylabel("Node Count")
        ax.set_title(f"DAG State at Cycle {cycle}\n(nodes: {footprint['node_count']}, max_depth: {footprint['max_depth']})")
        
        frame_path = out_dir / f"{prefix}_{i:05d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        frames.append(frame_path)
    
    return frames


# -----------------------------------------------------------------------------
# Extended Visualizations (10 Additional Types)
# -----------------------------------------------------------------------------


def plot_abstention_vs_depth_scatter(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Abstention vs Depth Scatter",
) -> Path:
    """
    Scatter plot of abstention rate vs chain depth per cycle.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths = np.array(metrics["depth"])
    abstentions = np.array(metrics["abstention"])
    
    # Color by abstention
    colors = [COLOR_ABSTENTION if a else COLOR_SUCCESS for a in abstentions]
    
    ax.scatter(depths, abstentions, c=colors, alpha=0.6, s=30)
    
    ax.set_xlabel("Chain Depth")
    ax.set_ylabel("Abstention (0/1)")
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLOR_SUCCESS, label="Success"),
        mpatches.Patch(color=COLOR_ABSTENTION, label="Abstention"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_candidate_pool_entropy(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Candidate Pool Entropy Per Cycle",
) -> Path:
    """
    Plot entropy-like metric of candidate pool diversity per cycle.
    
    Uses candidate count as proxy for pool diversity.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cycles = metrics["cycle"]
    candidates = np.array(metrics["candidates"], dtype=float)
    
    # Compute "entropy" as log(candidates + 1)
    entropy = np.log(candidates + 1)
    
    ax.plot(cycles, entropy, color=COLOR_ACCENT, linewidth=1.5, alpha=0.8)
    ax.fill_between(cycles, 0, entropy, color=COLOR_ACCENT, alpha=0.2)
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Pool Entropy (log(candidates + 1))")
    ax.set_title(title)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_mp_round_yield_vs_depth(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "MP-Round Yield vs Depth",
) -> Path:
    """
    Plot derivation yield (verified/candidates) vs depth.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths = np.array(metrics["depth"])
    candidates = np.array(metrics["candidates"], dtype=float)
    verified = np.array(metrics["verified"], dtype=float)
    
    # Compute yield (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_rate = np.where(candidates > 0, verified / candidates, 0)
    
    ax.scatter(depths, yield_rate, c=COLOR_ACCENT, alpha=0.6, s=30)
    
    ax.set_xlabel("Chain Depth")
    ax.set_ylabel("Yield (verified / candidates)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_success_concentration_diagram(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Success Concentration Diagram",
    bins: int = 10,
) -> Path:
    """
    Plot success rate concentration across depth bins.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths = np.array(metrics["depth"])
    successes = np.array(metrics["success"])
    
    if len(depths) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        # Bin by depth
        depth_min, depth_max = depths.min(), depths.max()
        if depth_min == depth_max:
            depth_max = depth_min + 1
        
        bin_edges = np.linspace(depth_min, depth_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        success_rates = []
        for i in range(bins):
            mask = (depths >= bin_edges[i]) & (depths < bin_edges[i + 1])
            if mask.sum() > 0:
                rate = successes[mask].mean()
            else:
                rate = 0
            success_rates.append(rate)
        
        ax.bar(bin_centers, success_rates, width=(depth_max - depth_min) / bins * 0.8,
               color=COLOR_SUCCESS, alpha=0.8, edgecolor="black")
    
    ax.set_xlabel("Depth Bin")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_cumulative_abstention_curve(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Cumulative Abstention Curve",
) -> Path:
    """
    Plot cumulative abstention count over cycles.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_cum = np.cumsum(baseline_metrics["abstention"])
    rfl_cum = np.cumsum(rfl_metrics["abstention"])
    
    ax.plot(baseline_metrics["cycle"], baseline_cum, label="Baseline",
            color=COLOR_BASELINE, linewidth=2, linestyle="--")
    ax.plot(rfl_metrics["cycle"], rfl_cum, label="RFL",
            color=COLOR_RFL, linewidth=2)
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative Abstentions")
    ax.set_title(title)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_verification_rate_heatmap(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Verification Rate by Cycle Blocks",
    block_size: int = 50,
) -> Path:
    """
    Heatmap of verification rates across cycle blocks and depth.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    n_cycles = len(metrics["cycle"])
    n_blocks = (n_cycles + block_size - 1) // block_size
    
    depths = np.array(metrics["depth"])
    verified = np.array(metrics["verified"])
    candidates = np.array(metrics["candidates"], dtype=float)
    
    max_depth = int(depths.max()) if len(depths) > 0 else 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Build heatmap matrix
    heatmap = np.zeros((max_depth, n_blocks))
    counts = np.zeros((max_depth, n_blocks))
    
    for i in range(n_cycles):
        block = i // block_size
        d = int(depths[i]) - 1  # 0-indexed
        if d >= 0 and d < max_depth and block < n_blocks:
            rate = verified[i] / candidates[i] if candidates[i] > 0 else 0
            heatmap[d, block] += rate
            counts[d, block] += 1
    
    # Average
    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap = np.where(counts > 0, heatmap / counts, 0)
    
    im = ax.imshow(heatmap, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xlabel("Cycle Block")
    ax.set_ylabel("Depth Level")
    ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Verification Rate")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_depth_velocity(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Depth Velocity (Rate of Depth Change)",
    window: int = 10,
) -> Path:
    """
    Plot rate of depth change (velocity) over cycles.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths = np.array(metrics["depth"], dtype=float)
    
    if len(depths) > 1:
        # Compute velocity as diff of rolling mean
        if len(depths) >= window:
            smooth = np.convolve(depths, np.ones(window) / window, mode="valid")
            velocity = np.diff(smooth)
            x = np.arange(window, window + len(velocity))
            
            ax.plot(x, velocity, color=COLOR_ACCENT, linewidth=1.5)
            ax.axhline(0, color=COLOR_BASELINE, linestyle="--", alpha=0.5)
            ax.fill_between(x, 0, velocity, where=(velocity > 0),
                           color=COLOR_SUCCESS, alpha=0.3)
            ax.fill_between(x, 0, velocity, where=(velocity < 0),
                           color=COLOR_ABSTENTION, alpha=0.3)
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Depth Velocity (Δdepth/cycle)")
    ax.set_title(title)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_branching_over_cycles(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Branching Factor Over Cycles",
) -> Path:
    """
    Plot rolling average of branching factor (candidates/verified).
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    candidates = np.array(metrics["candidates"], dtype=float)
    verified = np.array(metrics["verified"], dtype=float)
    
    # Branching factor as candidates per verified
    with np.errstate(divide="ignore", invalid="ignore"):
        branching = np.where(verified > 0, candidates / verified, candidates)
    
    ax.plot(metrics["cycle"], branching, color=COLOR_ACCENT, alpha=0.3, linewidth=0.5)
    
    # Rolling mean
    window = 20
    if len(branching) >= window:
        smooth = np.convolve(branching, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(branching)), smooth,
               color=COLOR_RFL, linewidth=2, label=f"{window}-cycle avg")
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Branching Factor")
    ax.set_title(title)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_depth_vs_candidates_2d(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Depth vs Candidates 2D Density",
) -> Path:
    """
    2D histogram of depth vs candidate count.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    depths = np.array(metrics["depth"])
    candidates = np.array(metrics["candidates"])
    
    if len(depths) > 0:
        h = ax.hist2d(depths, candidates, bins=20, cmap=HEATMAP_CMAP)
        fig.colorbar(h[3], ax=ax, label="Count")
    
    ax.set_xlabel("Depth")
    ax.set_ylabel("Candidates")
    ax.set_title(title)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_success_failure_timeline(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Success/Failure Timeline",
) -> Path:
    """
    Timeline visualization of successes and failures.
    
    PHASE II ONLY - Visualization, no inference.
    """
    setup_style()
    
    metrics = extract_cycle_metrics(records)
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    cycles = metrics["cycle"]
    successes = metrics["success"]
    
    for i, (c, s) in enumerate(zip(cycles, successes)):
        color = COLOR_SUCCESS if s else COLOR_ABSTENTION
        ax.axvline(c, color=color, alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel("Cycle")
    ax.set_yticks([])
    ax.set_title(title)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_SUCCESS, alpha=0.5, label="Success"),
        mpatches.Patch(color=COLOR_ABSTENTION, alpha=0.5, label="Failure/Abstention"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


# -----------------------------------------------------------------------------
# Batch Generation
# -----------------------------------------------------------------------------


def generate_all_topology_visualizations(
    baseline_log: str | Path,
    rfl_log: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
) -> Dict[str, Path]:
    """
    Generate all topology visualizations for baseline/RFL log pair.
    
    PHASE II ONLY - Visualization, no inference.
    
    Args:
        baseline_log: Path to baseline JSONL log.
        rfl_log: Path to RFL JSONL log.
        output_dir: Output directory.
    
    Returns:
        Dict mapping visualization names to output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)
    
    baseline_derivations = extract_derivations_from_log(baseline_records)
    rfl_derivations = extract_derivations_from_log(rfl_records)
    
    baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
    rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
    
    results: Dict[str, Path] = {}
    
    # Core topology visualizations
    results["adjacency_baseline"] = plot_adjacency_matrix_heatmap(
        baseline_analyzer, output_dir / "adjacency_matrix_baseline.png"
    )
    results["adjacency_rfl"] = plot_adjacency_matrix_heatmap(
        rfl_analyzer, output_dir / "adjacency_matrix_rfl.png"
    )
    results["depth_layers_baseline"] = plot_depth_level_layers(
        baseline_analyzer, output_dir / "depth_layers_baseline.png"
    )
    results["depth_layers_rfl"] = plot_depth_level_layers(
        rfl_analyzer, output_dir / "depth_layers_rfl.png"
    )
    results["branching_baseline"] = plot_branching_factor_distribution(
        baseline_analyzer, output_dir / "branching_distribution_baseline.png"
    )
    results["branching_rfl"] = plot_branching_factor_distribution(
        rfl_analyzer, output_dir / "branching_distribution_rfl.png"
    )
    results["longest_path_baseline"] = plot_longest_path_visualization(
        baseline_analyzer, output_dir / "longest_path_baseline.png"
    )
    results["longest_path_rfl"] = plot_longest_path_visualization(
        rfl_analyzer, output_dir / "longest_path_rfl.png"
    )
    results["depth_dist_baseline"] = plot_depth_distribution_histogram(
        baseline_analyzer, output_dir / "depth_distribution_baseline.png"
    )
    results["depth_dist_rfl"] = plot_depth_distribution_histogram(
        rfl_analyzer, output_dir / "depth_distribution_rfl.png"
    )
    
    # Comparison visualizations
    results["node_count_comparison"] = plot_node_count_comparison(
        baseline_records, rfl_records, output_dir / "node_count_comparison.png"
    )
    results["depth_dist_comparison"] = plot_depth_distribution_comparison(
        baseline_analyzer, rfl_analyzer, output_dir / "depth_distribution_comparison.png"
    )
    results["depth_trend"] = plot_chain_depth_trend(
        baseline_records, rfl_records, output_dir / "chain_depth_trend.png"
    )
    
    # Extended visualizations
    results["abstention_vs_depth_baseline"] = plot_abstention_vs_depth_scatter(
        baseline_records, output_dir / "abstention_vs_depth_baseline.png"
    )
    results["abstention_vs_depth_rfl"] = plot_abstention_vs_depth_scatter(
        rfl_records, output_dir / "abstention_vs_depth_rfl.png"
    )
    results["entropy_baseline"] = plot_candidate_pool_entropy(
        baseline_records, output_dir / "candidate_entropy_baseline.png"
    )
    results["entropy_rfl"] = plot_candidate_pool_entropy(
        rfl_records, output_dir / "candidate_entropy_rfl.png"
    )
    results["yield_baseline"] = plot_mp_round_yield_vs_depth(
        baseline_records, output_dir / "mp_yield_baseline.png"
    )
    results["yield_rfl"] = plot_mp_round_yield_vs_depth(
        rfl_records, output_dir / "mp_yield_rfl.png"
    )
    results["success_concentration_baseline"] = plot_success_concentration_diagram(
        baseline_records, output_dir / "success_concentration_baseline.png"
    )
    results["success_concentration_rfl"] = plot_success_concentration_diagram(
        rfl_records, output_dir / "success_concentration_rfl.png"
    )
    results["cumulative_abstention"] = plot_cumulative_abstention_curve(
        baseline_records, rfl_records, output_dir / "cumulative_abstention.png"
    )
    results["verification_heatmap_baseline"] = plot_verification_rate_heatmap(
        baseline_records, output_dir / "verification_heatmap_baseline.png"
    )
    results["verification_heatmap_rfl"] = plot_verification_rate_heatmap(
        rfl_records, output_dir / "verification_heatmap_rfl.png"
    )
    results["depth_velocity_baseline"] = plot_depth_velocity(
        baseline_records, output_dir / "depth_velocity_baseline.png"
    )
    results["depth_velocity_rfl"] = plot_depth_velocity(
        rfl_records, output_dir / "depth_velocity_rfl.png"
    )
    results["branching_cycles_baseline"] = plot_branching_over_cycles(
        baseline_records, output_dir / "branching_over_cycles_baseline.png"
    )
    results["branching_cycles_rfl"] = plot_branching_over_cycles(
        rfl_records, output_dir / "branching_over_cycles_rfl.png"
    )
    results["depth_candidates_2d_baseline"] = plot_depth_vs_candidates_2d(
        baseline_records, output_dir / "depth_candidates_2d_baseline.png"
    )
    results["depth_candidates_2d_rfl"] = plot_depth_vs_candidates_2d(
        rfl_records, output_dir / "depth_candidates_2d_rfl.png"
    )
    results["timeline_baseline"] = plot_success_failure_timeline(
        baseline_records, output_dir / "success_timeline_baseline.png"
    )
    results["timeline_rfl"] = plot_success_failure_timeline(
        rfl_records, output_dir / "success_timeline_rfl.png"
    )
    
    # DAG evolution frames
    frames_dir = output_dir / "evolution_frames"
    results["evolution_frames_baseline"] = frames_dir / "baseline"
    results["evolution_frames_rfl"] = frames_dir / "rfl"
    
    generate_dag_evolution_timeline(
        baseline_records, frames_dir / "baseline", "baseline", frame_interval=100
    )
    generate_dag_evolution_timeline(
        rfl_records, frames_dir / "rfl", "rfl", frame_interval=100
    )
    
    # Generate manifest
    manifest = {
        "phase": "II",
        "type": "dag_topology_visualizations",
        "outputs": {k: str(v) for k, v in results.items()},
        "baseline_log": str(baseline_log),
        "rfl_log": str(rfl_log),
        "baseline_footprint": baseline_analyzer.get_dag_footprint(),
        "rfl_footprint": rfl_analyzer.get_dag_footprint(),
    }
    
    manifest_path = output_dir / "topology_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    results["manifest"] = manifest_path
    
    return results


# -----------------------------------------------------------------------------
# Curriculum → DAG Expectation Mapping
# -----------------------------------------------------------------------------


class ExpectedDAGProfile:
    """
    Expected DAG behavior profile for a curriculum slice.
    
    Each slice type has characteristic DAG topology expectations:
      - goal_hit: shallow, narrow DAGs (quick hits)
      - sparse: medium depth, medium branching
      - tree: deeper branching, significant chain depth
      - dependency: multiple branching subtrees
    
    PHASE II ONLY - Descriptive analytics, no uplift claims.
    """
    
    def __init__(
        self,
        slice_type: str,
        expected_depth_range: Tuple[float, float],
        expected_branching_range: Tuple[float, float],
        expected_success_rate_range: Tuple[float, float],
        description: str,
    ):
        """
        Initialize expected profile.
        
        Args:
            slice_type: Category (goal_hit, sparse, tree, dependency, unknown)
            expected_depth_range: (min, max) expected mean depth
            expected_branching_range: (min, max) expected mean out-degree
            expected_success_rate_range: (min, max) expected success rate
            description: Human-readable description of expected behavior
        """
        self.slice_type = slice_type
        self.expected_depth_range = expected_depth_range
        self.expected_branching_range = expected_branching_range
        self.expected_success_rate_range = expected_success_rate_range
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "slice_type": self.slice_type,
            "expected_depth_range": list(self.expected_depth_range),
            "expected_branching_range": list(self.expected_branching_range),
            "expected_success_rate_range": list(self.expected_success_rate_range),
            "description": self.description,
        }
    
    def check_depth_drift(self, observed_mean_depth: float) -> Dict[str, Any]:
        """
        Check if observed depth is within expected range.
        
        Returns dict with drift status and magnitude.
        """
        min_d, max_d = self.expected_depth_range
        if observed_mean_depth < min_d:
            drift = min_d - observed_mean_depth
            return {"status": "below", "drift": drift, "expected": self.expected_depth_range}
        elif observed_mean_depth > max_d:
            drift = observed_mean_depth - max_d
            return {"status": "above", "drift": drift, "expected": self.expected_depth_range}
        else:
            return {"status": "within", "drift": 0.0, "expected": self.expected_depth_range}
    
    def check_branching_drift(self, observed_mean_branching: float) -> Dict[str, Any]:
        """Check if observed branching is within expected range."""
        min_b, max_b = self.expected_branching_range
        if observed_mean_branching < min_b:
            drift = min_b - observed_mean_branching
            return {"status": "below", "drift": drift, "expected": self.expected_branching_range}
        elif observed_mean_branching > max_b:
            drift = observed_mean_branching - max_b
            return {"status": "above", "drift": drift, "expected": self.expected_branching_range}
        else:
            return {"status": "within", "drift": 0.0, "expected": self.expected_branching_range}
    
    def check_success_drift(self, observed_success_rate: float) -> Dict[str, Any]:
        """Check if observed success rate is within expected range."""
        min_s, max_s = self.expected_success_rate_range
        if observed_success_rate < min_s:
            drift = min_s - observed_success_rate
            return {"status": "below", "drift": drift, "expected": self.expected_success_rate_range}
        elif observed_success_rate > max_s:
            drift = observed_success_rate - max_s
            return {"status": "above", "drift": drift, "expected": self.expected_success_rate_range}
        else:
            return {"status": "within", "drift": 0.0, "expected": self.expected_success_rate_range}


# Slice type keyword mappings for inference
_SLICE_TYPE_KEYWORDS = {
    "goal_hit": ["goal", "hit", "easy", "debug", "simple"],
    "sparse": ["sparse", "medium", "proto", "uplift_proto"],
    "tree": ["tree", "branching", "deep", "hard"],
    "dependency": ["dependency", "dep", "chain", "organism", "fo"],
}


def infer_expected_dag_profile(slice_name: str) -> ExpectedDAGProfile:
    """
    Infer expected DAG profile from slice name.
    
    Maps slice naming conventions to expected DAG topology:
      - goal_hit slices: shallow (1-2), narrow (0-1), high success (0.7-1.0)
      - sparse slices: medium depth (2-4), medium branching (1-2), moderate success (0.3-0.7)
      - tree slices: deep (3-8), high branching (2-5), variable success (0.2-0.6)
      - dependency slices: medium depth (2-5), multi-branching (1-3), moderate success (0.3-0.7)
    
    PHASE II ONLY - Descriptive analytics, no uplift claims.
    
    Args:
        slice_name: Name of the curriculum slice (e.g., "slice_uplift_tree")
    
    Returns:
        ExpectedDAGProfile with expected behavior ranges.
    """
    slice_lower = slice_name.lower()
    
    # Check for goal_hit patterns
    if any(kw in slice_lower for kw in _SLICE_TYPE_KEYWORDS["goal_hit"]):
        return ExpectedDAGProfile(
            slice_type="goal_hit",
            expected_depth_range=(1.0, 3.0),
            expected_branching_range=(0.0, 1.5),
            expected_success_rate_range=(0.6, 1.0),
            description="Shallow, narrow DAGs optimized for quick goal hits",
        )
    
    # Check for tree patterns
    if any(kw in slice_lower for kw in _SLICE_TYPE_KEYWORDS["tree"]):
        return ExpectedDAGProfile(
            slice_type="tree",
            expected_depth_range=(3.0, 10.0),
            expected_branching_range=(1.5, 5.0),
            expected_success_rate_range=(0.1, 0.6),
            description="Deep DAGs with significant branching, chain depth exploration",
        )
    
    # Check for dependency patterns (First Organism, etc.)
    if any(kw in slice_lower for kw in _SLICE_TYPE_KEYWORDS["dependency"]):
        return ExpectedDAGProfile(
            slice_type="dependency",
            expected_depth_range=(2.0, 6.0),
            expected_branching_range=(1.0, 3.0),
            expected_success_rate_range=(0.2, 0.7),
            description="Multiple branching subtrees with dependency chains",
        )
    
    # Check for sparse/proto patterns
    if any(kw in slice_lower for kw in _SLICE_TYPE_KEYWORDS["sparse"]):
        return ExpectedDAGProfile(
            slice_type="sparse",
            expected_depth_range=(2.0, 5.0),
            expected_branching_range=(0.5, 2.5),
            expected_success_rate_range=(0.3, 0.7),
            description="Medium depth and branching for uplift measurement",
        )
    
    # Default: unknown slice type with wide ranges
    return ExpectedDAGProfile(
        slice_type="unknown",
        expected_depth_range=(1.0, 10.0),
        expected_branching_range=(0.0, 5.0),
        expected_success_rate_range=(0.0, 1.0),
        description="Unknown slice type - wide expected ranges",
    )


def load_curriculum_slice_params(slice_name: str) -> Optional[Dict[str, Any]]:
    """
    Load slice parameters from curriculum.yaml if available.
    
    Args:
        slice_name: Name of the slice to look up.
    
    Returns:
        Dict of slice params or None if not found.
    """
    import yaml
    
    curriculum_path = Path("config/curriculum.yaml")
    if not curriculum_path.exists():
        return None
    
    try:
        with open(curriculum_path, "r") as f:
            curriculum = yaml.safe_load(f)
        
        # Navigate to slices
        systems = curriculum.get("systems", {})
        for system_name, system_data in systems.items():
            slices = system_data.get("slices", [])
            for slice_def in slices:
                if slice_def.get("name") == slice_name:
                    return slice_def.get("params", {})
        
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Behavior Drift Computation
# -----------------------------------------------------------------------------


def compute_behavior_drift(
    baseline_analyzer: DAGTopologyAnalyzer,
    rfl_analyzer: DAGTopologyAnalyzer,
    baseline_metrics: Dict[str, List[Any]],
    rfl_metrics: Dict[str, List[Any]],
    slice_name: str,
) -> Dict[str, Any]:
    """
    Compute behavior drift between baseline and RFL, and against expectations.
    
    Generates comprehensive drift report:
      - Depth drift (observed vs expected)
      - Branching drift
      - Success distribution drift
      - Baseline vs RFL delta
    
    PHASE II ONLY - Descriptive analytics, no uplift claims.
    
    Args:
        baseline_analyzer: Analyzer for baseline DAG.
        rfl_analyzer: Analyzer for RFL DAG.
        baseline_metrics: Extracted metrics from baseline log.
        rfl_metrics: Extracted metrics from RFL log.
        slice_name: Name of the slice for expectation lookup.
    
    Returns:
        Drift report dictionary.
    """
    # Get expected profile
    expected = infer_expected_dag_profile(slice_name)
    
    # Compute observed statistics
    baseline_footprint = baseline_analyzer.get_dag_footprint()
    rfl_footprint = rfl_analyzer.get_dag_footprint()
    
    baseline_depths = np.array(baseline_metrics.get("depth", [1]))
    rfl_depths = np.array(rfl_metrics.get("depth", [1]))
    
    baseline_success = np.array(baseline_metrics.get("success", [0]))
    rfl_success = np.array(rfl_metrics.get("success", [0]))
    
    # Compute mean branching from footprint
    baseline_mean_branching = (
        baseline_footprint["edge_count"] / max(baseline_footprint["node_count"], 1)
    )
    rfl_mean_branching = (
        rfl_footprint["edge_count"] / max(rfl_footprint["node_count"], 1)
    )
    
    # Observed means
    baseline_mean_depth = float(np.mean(baseline_depths)) if len(baseline_depths) > 0 else 1.0
    rfl_mean_depth = float(np.mean(rfl_depths)) if len(rfl_depths) > 0 else 1.0
    
    baseline_success_rate = float(np.mean(baseline_success)) if len(baseline_success) > 0 else 0.0
    rfl_success_rate = float(np.mean(rfl_success)) if len(rfl_success) > 0 else 0.0
    
    # Drift against expectations
    baseline_depth_drift = expected.check_depth_drift(baseline_mean_depth)
    rfl_depth_drift = expected.check_depth_drift(rfl_mean_depth)
    
    baseline_branching_drift = expected.check_branching_drift(baseline_mean_branching)
    rfl_branching_drift = expected.check_branching_drift(rfl_mean_branching)
    
    baseline_success_drift = expected.check_success_drift(baseline_success_rate)
    rfl_success_drift = expected.check_success_drift(rfl_success_rate)
    
    # Delta between baseline and RFL
    depth_delta = rfl_mean_depth - baseline_mean_depth
    branching_delta = rfl_mean_branching - baseline_mean_branching
    success_delta = rfl_success_rate - baseline_success_rate
    
    # Success distribution comparison (histogram bins)
    def compute_success_distribution(successes: np.ndarray, window: int = 50) -> List[float]:
        """Compute windowed success rate distribution."""
        if len(successes) < window:
            return [float(np.mean(successes))] if len(successes) > 0 else [0.0]
        
        rates = []
        for i in range(0, len(successes) - window + 1, window):
            chunk = successes[i:i + window]
            rates.append(float(np.mean(chunk)))
        return rates
    
    baseline_success_dist = compute_success_distribution(baseline_success)
    rfl_success_dist = compute_success_distribution(rfl_success)
    
    # Compute distribution drift (KL-divergence-like)
    def distribution_drift(dist1: List[float], dist2: List[float]) -> float:
        """Compute simple distribution distance."""
        if not dist1 or not dist2:
            return 0.0
        
        # Align lengths
        min_len = min(len(dist1), len(dist2))
        d1 = np.array(dist1[:min_len])
        d2 = np.array(dist2[:min_len])
        
        # Mean absolute difference
        return float(np.mean(np.abs(d1 - d2)))
    
    success_dist_drift = distribution_drift(baseline_success_dist, rfl_success_dist)
    
    # Build drift report
    drift_report = {
        "phase": "II",
        "type": "behavior_drift_report",
        "slice_name": slice_name,
        "expected_profile": expected.to_dict(),
        "observed": {
            "baseline": {
                "mean_depth": baseline_mean_depth,
                "mean_branching": baseline_mean_branching,
                "success_rate": baseline_success_rate,
                "node_count": baseline_footprint["node_count"],
                "edge_count": baseline_footprint["edge_count"],
            },
            "rfl": {
                "mean_depth": rfl_mean_depth,
                "mean_branching": rfl_mean_branching,
                "success_rate": rfl_success_rate,
                "node_count": rfl_footprint["node_count"],
                "edge_count": rfl_footprint["edge_count"],
            },
        },
        "drift_vs_expected": {
            "baseline": {
                "depth": baseline_depth_drift,
                "branching": baseline_branching_drift,
                "success": baseline_success_drift,
            },
            "rfl": {
                "depth": rfl_depth_drift,
                "branching": rfl_branching_drift,
                "success": rfl_success_drift,
            },
        },
        "delta_baseline_vs_rfl": {
            "depth_delta": depth_delta,
            "branching_delta": branching_delta,
            "success_delta": success_delta,
            "success_distribution_drift": success_dist_drift,
        },
        "success_distributions": {
            "baseline": baseline_success_dist,
            "rfl": rfl_success_dist,
        },
        "narrative_flags": [],
    }
    
    # Add narrative flags for notable drift
    if abs(depth_delta) > 1.0:
        drift_report["narrative_flags"].append(
            f"Notable depth delta: {depth_delta:+.2f} (RFL vs baseline)"
        )
    
    if baseline_depth_drift["status"] != "within":
        drift_report["narrative_flags"].append(
            f"Baseline depth drift: {baseline_depth_drift['status']} expected range"
        )
    
    if rfl_depth_drift["status"] != "within":
        drift_report["narrative_flags"].append(
            f"RFL depth drift: {rfl_depth_drift['status']} expected range"
        )
    
    if success_dist_drift > 0.15:
        drift_report["narrative_flags"].append(
            f"Success distribution drift: {success_dist_drift:.3f} (notable)"
        )
    
    return drift_report


def write_behavior_drift_report(
    drift_report: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Write behavior drift report to JSON file.
    
    Args:
        drift_report: Drift report dictionary.
        output_path: Output file path.
    
    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)
    
    return output_path


# -----------------------------------------------------------------------------
# Advanced Differential Visualizations
# -----------------------------------------------------------------------------


def plot_differential_depth_histogram(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Differential Depth Histogram (RFL - Baseline)",
    bins: int = 20,
) -> Path:
    """
    Plot differential histogram showing depth distribution difference.
    
    Shows where RFL has more/fewer occurrences at each depth level.
    Positive bars = RFL has more; Negative bars = baseline has more.
    
    PHASE II ONLY - Visualization, no uplift claims.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    baseline_depths = np.array(baseline_metrics["depth"])
    rfl_depths = np.array(rfl_metrics["depth"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(baseline_depths) > 0 and len(rfl_depths) > 0:
        # Determine common bin edges
        all_depths = np.concatenate([baseline_depths, rfl_depths])
        min_d, max_d = all_depths.min(), all_depths.max()
        bin_edges = np.linspace(min_d, max_d, bins + 1)
        
        # Compute histograms
        baseline_hist, _ = np.histogram(baseline_depths, bins=bin_edges, density=True)
        rfl_hist, _ = np.histogram(rfl_depths, bins=bin_edges, density=True)
        
        # Differential
        diff_hist = rfl_hist - baseline_hist
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot with color coding
        colors = [COLOR_SUCCESS if d > 0 else COLOR_ABSTENTION for d in diff_hist]
        ax.bar(bin_centers, diff_hist, width=(max_d - min_d) / bins * 0.8,
               color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        ax.axhline(0, color=COLOR_BASELINE, linestyle="-", linewidth=1)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=COLOR_SUCCESS, label="RFL > Baseline"),
            mpatches.Patch(color=COLOR_ABSTENTION, label="Baseline > RFL"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    
    ax.set_xlabel("Depth")
    ax.set_ylabel("Density Difference (RFL - Baseline)")
    ax.set_title(title)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_branching_factor_delta_series(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Branching Factor Delta Over Cycles",
    window: int = 20,
) -> Path:
    """
    Plot delta of branching factors (RFL - baseline) over cycles.
    
    Shows how branching factor difference evolves over time.
    
    PHASE II ONLY - Visualization, no uplift claims.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    # Compute branching factor per cycle (candidates / max(verified, 1))
    def compute_branching_series(metrics: Dict[str, List[Any]]) -> np.ndarray:
        candidates = np.array(metrics["candidates"], dtype=float)
        verified = np.array(metrics["verified"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            branching = np.where(verified > 0, candidates / verified, candidates)
        return branching
    
    baseline_branching = compute_branching_series(baseline_metrics)
    rfl_branching = compute_branching_series(rfl_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Align lengths
    min_len = min(len(baseline_branching), len(rfl_branching))
    if min_len > 0:
        baseline_b = baseline_branching[:min_len]
        rfl_b = rfl_branching[:min_len]
        
        # Compute delta
        delta = rfl_b - baseline_b
        
        # Plot raw delta (faint)
        cycles = np.arange(min_len)
        ax.plot(cycles, delta, color=COLOR_ACCENT, alpha=0.2, linewidth=0.5)
        
        # Plot rolling mean delta
        if min_len >= window:
            delta_smooth = np.convolve(delta, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window - 1, min_len),
                delta_smooth,
                color=COLOR_RFL,
                linewidth=2,
                label=f"{window}-cycle rolling mean",
            )
        
        ax.axhline(0, color=COLOR_BASELINE, linestyle="--", alpha=0.5)
        
        # Fill positive/negative regions
        ax.fill_between(cycles, 0, delta, where=(delta > 0),
                       color=COLOR_SUCCESS, alpha=0.1)
        ax.fill_between(cycles, 0, delta, where=(delta < 0),
                       color=COLOR_ABSTENTION, alpha=0.1)
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Branching Factor Delta (RFL - Baseline)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_chain_depth_drift_visual(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    slice_name: str = "unknown",
    title: str = "Chain Depth Drift Analysis",
) -> Path:
    """
    Visualize chain depth drift against expected profile.
    
    Shows:
      - Expected depth range (shaded band)
      - Baseline observed depths (rolling mean)
      - RFL observed depths (rolling mean)
      - Drift zones highlighted
    
    PHASE II ONLY - Visualization, no uplift claims.
    """
    setup_style()
    
    expected = infer_expected_dag_profile(slice_name)
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    baseline_depths = np.array(baseline_metrics["depth"], dtype=float)
    rfl_depths = np.array(rfl_metrics["depth"], dtype=float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    window = 20
    
    # Plot expected range band
    expected_min, expected_max = expected.expected_depth_range
    max_cycles = max(len(baseline_depths), len(rfl_depths))
    
    if max_cycles > 0:
        ax.fill_between(
            [0, max_cycles],
            [expected_min, expected_min],
            [expected_max, expected_max],
            color=COLOR_ACCENT,
            alpha=0.15,
            label=f"Expected range ({expected.slice_type})",
        )
        
        ax.axhline(expected_min, color=COLOR_ACCENT, linestyle=":", alpha=0.5)
        ax.axhline(expected_max, color=COLOR_ACCENT, linestyle=":", alpha=0.5)
    
    # Plot baseline rolling mean
    if len(baseline_depths) >= window:
        baseline_smooth = np.convolve(baseline_depths, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(baseline_depths)),
            baseline_smooth,
            color=COLOR_BASELINE,
            linewidth=2,
            linestyle="--",
            label="Baseline (rolling mean)",
        )
    
    # Plot RFL rolling mean
    if len(rfl_depths) >= window:
        rfl_smooth = np.convolve(rfl_depths, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(rfl_depths)),
            rfl_smooth,
            color=COLOR_RFL,
            linewidth=2,
            label="RFL (rolling mean)",
        )
    
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Chain Depth")
    ax.set_title(f"{title}\n(Slice: {slice_name}, Type: {expected.slice_type})")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(expected_max * 1.5, np.max(baseline_depths) * 1.2 if len(baseline_depths) > 0 else 5))
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_success_drift_waterfall(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    out_path: str | Path,
    title: str = "Success Rate Drift Waterfall",
    n_segments: int = 10,
) -> Path:
    """
    Waterfall chart showing success rate changes across run segments.
    
    PHASE II ONLY - Visualization, no uplift claims.
    """
    setup_style()
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    baseline_success = np.array(baseline_metrics["success"], dtype=float)
    rfl_success = np.array(rfl_metrics["success"], dtype=float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def segment_means(data: np.ndarray, n: int) -> List[float]:
        if len(data) == 0:
            return [0.0] * n
        segment_size = max(1, len(data) // n)
        means = []
        for i in range(n):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(data))
            if start < len(data):
                means.append(float(np.mean(data[start:end])))
            else:
                means.append(0.0)
        return means
    
    baseline_segments = segment_means(baseline_success, n_segments)
    rfl_segments = segment_means(rfl_success, n_segments)
    
    x = np.arange(n_segments)
    width = 0.35
    
    ax.bar(x - width / 2, baseline_segments, width, label="Baseline",
           color=COLOR_BASELINE, alpha=0.8)
    ax.bar(x + width / 2, rfl_segments, width, label="RFL",
           color=COLOR_RFL, alpha=0.8)
    
    # Add delta annotations
    for i in range(n_segments):
        delta = rfl_segments[i] - baseline_segments[i]
        color = COLOR_SUCCESS if delta > 0 else COLOR_ABSTENTION
        ax.annotate(
            f"{delta:+.2f}",
            (i, max(baseline_segments[i], rfl_segments[i]) + 0.02),
            ha="center",
            fontsize=8,
            color=color,
        )
    
    ax.set_xlabel("Run Segment")
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}" for i in range(n_segments)])
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


def plot_curriculum_behavior_comparison(
    baseline_analyzer: DAGTopologyAnalyzer,
    rfl_analyzer: DAGTopologyAnalyzer,
    baseline_metrics: Dict[str, List[Any]],
    rfl_metrics: Dict[str, List[Any]],
    slice_name: str,
    out_path: str | Path,
    title: str = "Curriculum-Behavior Comparison",
) -> Path:
    """
    Multi-panel visualization comparing expected vs observed behavior.
    
    PHASE II ONLY - Visualization, no uplift claims.
    """
    setup_style()
    
    expected = infer_expected_dag_profile(slice_name)
    
    # Compute observed values
    baseline_footprint = baseline_analyzer.get_dag_footprint()
    rfl_footprint = rfl_analyzer.get_dag_footprint()
    
    baseline_depths = np.array(baseline_metrics.get("depth", [1]))
    rfl_depths = np.array(rfl_metrics.get("depth", [1]))
    baseline_success = np.array(baseline_metrics.get("success", [0]))
    rfl_success = np.array(rfl_metrics.get("success", [0]))
    
    baseline_mean_depth = float(np.mean(baseline_depths)) if len(baseline_depths) > 0 else 1.0
    rfl_mean_depth = float(np.mean(rfl_depths)) if len(rfl_depths) > 0 else 1.0
    
    baseline_branching = baseline_footprint["edge_count"] / max(baseline_footprint["node_count"], 1)
    rfl_branching = rfl_footprint["edge_count"] / max(rfl_footprint["node_count"], 1)
    
    baseline_success_rate = float(np.mean(baseline_success)) if len(baseline_success) > 0 else 0.0
    rfl_success_rate = float(np.mean(rfl_success)) if len(rfl_success) > 0 else 0.0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("Depth", expected.expected_depth_range, baseline_mean_depth, rfl_mean_depth),
        ("Branching", expected.expected_branching_range, baseline_branching, rfl_branching),
        ("Success Rate", expected.expected_success_rate_range, baseline_success_rate, rfl_success_rate),
    ]
    
    for ax, (name, expected_range, baseline_val, rfl_val) in zip(axes, metrics):
        # Expected range bar
        ax.barh(
            ["Expected"],
            [expected_range[1] - expected_range[0]],
            left=[expected_range[0]],
            color=COLOR_ACCENT,
            alpha=0.3,
            edgecolor=COLOR_ACCENT,
            linewidth=2,
        )
        
        # Observed values as markers
        ax.scatter([baseline_val], ["Baseline"], s=200, c=COLOR_BASELINE, marker="o", zorder=5)
        ax.scatter([rfl_val], ["RFL"], s=200, c=COLOR_RFL, marker="s", zorder=5)
        
        # Annotate values
        ax.annotate(f"{baseline_val:.2f}", (baseline_val, "Baseline"),
                   xytext=(5, 0), textcoords="offset points", fontsize=10)
        ax.annotate(f"{rfl_val:.2f}", (rfl_val, "RFL"),
                   xytext=(5, 0), textcoords="offset points", fontsize=10)
        
        ax.set_xlabel(name)
        ax.set_title(f"{name} vs Expected")
        ax.axvline(expected_range[0], color=COLOR_ACCENT, linestyle=":", alpha=0.5)
        ax.axvline(expected_range[1], color=COLOR_ACCENT, linestyle=":", alpha=0.5)
    
    fig.suptitle(f"{title}\n(Slice: {slice_name}, Type: {expected.slice_type})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return out_path


# -----------------------------------------------------------------------------
# Extended Batch Generation with Drift Analysis
# -----------------------------------------------------------------------------


def generate_all_topology_with_drift(
    baseline_log: str | Path,
    rfl_log: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    slice_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generate all topology visualizations including drift analysis.
    
    Extends generate_all_topology_visualizations with:
      - Behavior drift report (JSON)
      - Differential visualizations
      - Curriculum-behavior comparison
    
    PHASE II ONLY - Visualization + descriptive analytics, no uplift claims.
    
    Args:
        baseline_log: Path to baseline JSONL log.
        rfl_log: Path to RFL JSONL log.
        output_dir: Output directory.
        slice_name: Optional slice name (auto-detected from logs if not provided).
    
    Returns:
        Dict mapping visualization names to output paths.
    """
    # First generate standard visualizations
    results = generate_all_topology_visualizations(baseline_log, rfl_log, output_dir)
    
    output_dir = Path(output_dir)
    
    # Load data for drift analysis
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)
    
    baseline_derivations = extract_derivations_from_log(baseline_records)
    rfl_derivations = extract_derivations_from_log(rfl_records)
    
    baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
    rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
    
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    # Auto-detect slice name if not provided
    if slice_name is None:
        if baseline_records:
            slice_name = baseline_records[0].get("slice_name", "unknown")
        else:
            slice_name = "unknown"
    
    # Compute and write drift report
    drift_report = compute_behavior_drift(
        baseline_analyzer, rfl_analyzer,
        baseline_metrics, rfl_metrics,
        slice_name,
    )
    
    drift_path = write_behavior_drift_report(
        drift_report,
        output_dir / "behavior_drift.json",
    )
    results["behavior_drift"] = drift_path
    
    # Generate advanced differential visualizations
    results["differential_depth_histogram"] = plot_differential_depth_histogram(
        baseline_records, rfl_records,
        output_dir / "differential_depth_histogram.png",
    )
    
    results["branching_delta_series"] = plot_branching_factor_delta_series(
        baseline_records, rfl_records,
        output_dir / "branching_factor_delta_series.png",
    )
    
    results["chain_depth_drift"] = plot_chain_depth_drift_visual(
        baseline_records, rfl_records,
        output_dir / "chain_depth_drift.png",
        slice_name=slice_name,
    )
    
    results["success_drift_waterfall"] = plot_success_drift_waterfall(
        baseline_records, rfl_records,
        output_dir / "success_drift_waterfall.png",
    )
    
    results["curriculum_behavior_comparison"] = plot_curriculum_behavior_comparison(
        baseline_analyzer, rfl_analyzer,
        baseline_metrics, rfl_metrics,
        slice_name,
        output_dir / "curriculum_behavior_comparison.png",
    )
    
    # Update manifest with drift info
    manifest_path = results.get("manifest")
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["drift_analysis"] = {
            "slice_name": slice_name,
            "expected_profile": drift_report["expected_profile"],
            "narrative_flags": drift_report["narrative_flags"],
            "delta_baseline_vs_rfl": drift_report["delta_baseline_vs_rfl"],
        }
        manifest["outputs"].update({k: str(v) for k, v in results.items() if k not in manifest["outputs"]})
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    return results


# -----------------------------------------------------------------------------
# Curriculum ↔ Topology Diagnostic Layer
# -----------------------------------------------------------------------------


# Warning thresholds (configurable)
WARNING_DEPTH_DELTA_THRESHOLD = 1.0
WARNING_BRANCHING_DELTA_THRESHOLD = 0.5
WARNING_BRANCHING_COLLAPSE_EPSILON = 0.1
WARNING_DEPTH_SATURATION_MARGIN = 0.9  # 90% of theoretical max


def _get_theoretical_max_depth(slice_name: str) -> Optional[int]:
    """
    Get theoretical max depth from curriculum for a slice.
    
    Attempts to load from config/curriculum.yaml if available.
    
    Args:
        slice_name: Name of the slice.
    
    Returns:
        depth_max from curriculum params, or None if not found.
    """
    params = load_curriculum_slice_params(slice_name)
    if params:
        return params.get("depth_max")
    return None


def compute_warning_flags(
    baseline_footprint: Dict[str, Any],
    rfl_footprint: Dict[str, Any],
    depth_delta: float,
    branching_delta: float,
    slice_name: str,
) -> Dict[str, Any]:
    """
    Compute warning flags for curriculum designers.
    
    These are WARNINGS ONLY — no exit codes, no CI gating.
    Signals for curriculum authors, not promotion logic.
    
    PHASE II ONLY - Diagnostic flags, no uplift claims.
    
    Args:
        baseline_footprint: Baseline DAG footprint dict.
        rfl_footprint: RFL DAG footprint dict.
        depth_delta: RFL max_depth - baseline max_depth.
        branching_delta: RFL mean_branching - baseline mean_branching.
        slice_name: Slice name for curriculum lookup.
    
    Returns:
        Dict with warning flags and explanations.
    """
    warnings: Dict[str, Any] = {
        "depth_saturation_warning": False,
        "branching_collapse_warning": False,
        "topology_change_warning": False,
        "explanations": [],
    }
    
    # Get theoretical max depth from curriculum
    theoretical_max = _get_theoretical_max_depth(slice_name)
    
    # 1. Depth saturation warning
    # True if max_depth ≈ theoretical max (from curriculum, if available)
    baseline_max_depth = baseline_footprint.get("max_depth", 0)
    rfl_max_depth = rfl_footprint.get("max_depth", 0)
    
    if theoretical_max is not None and theoretical_max > 0:
        baseline_saturation = baseline_max_depth / theoretical_max
        rfl_saturation = rfl_max_depth / theoretical_max
        
        if baseline_saturation >= WARNING_DEPTH_SATURATION_MARGIN:
            warnings["depth_saturation_warning"] = True
            warnings["explanations"].append(
                f"Baseline depth saturated: {baseline_max_depth}/{theoretical_max} "
                f"({baseline_saturation:.1%} of theoretical max)"
            )
        
        if rfl_saturation >= WARNING_DEPTH_SATURATION_MARGIN:
            warnings["depth_saturation_warning"] = True
            warnings["explanations"].append(
                f"RFL depth saturated: {rfl_max_depth}/{theoretical_max} "
                f"({rfl_saturation:.1%} of theoretical max)"
            )
    
    # 2. Branching collapse warning
    # True if mean_branching < ε for both baseline and RFL
    baseline_mean_branching = (
        baseline_footprint.get("edge_count", 0) / 
        max(baseline_footprint.get("node_count", 1), 1)
    )
    rfl_mean_branching = (
        rfl_footprint.get("edge_count", 0) / 
        max(rfl_footprint.get("node_count", 1), 1)
    )
    
    if (baseline_mean_branching < WARNING_BRANCHING_COLLAPSE_EPSILON and
        rfl_mean_branching < WARNING_BRANCHING_COLLAPSE_EPSILON):
        warnings["branching_collapse_warning"] = True
        warnings["explanations"].append(
            f"Branching collapse detected: baseline={baseline_mean_branching:.3f}, "
            f"rfl={rfl_mean_branching:.3f} (both < {WARNING_BRANCHING_COLLAPSE_EPSILON})"
        )
    
    # 3. Topology change warning
    # True if |depth_delta| > 1.0 or |branching_delta| > 0.5
    if abs(depth_delta) > WARNING_DEPTH_DELTA_THRESHOLD:
        warnings["topology_change_warning"] = True
        warnings["explanations"].append(
            f"Significant depth delta: {depth_delta:+.2f} "
            f"(threshold: ±{WARNING_DEPTH_DELTA_THRESHOLD})"
        )
    
    if abs(branching_delta) > WARNING_BRANCHING_DELTA_THRESHOLD:
        warnings["topology_change_warning"] = True
        warnings["explanations"].append(
            f"Significant branching delta: {branching_delta:+.3f} "
            f"(threshold: ±{WARNING_BRANCHING_DELTA_THRESHOLD})"
        )
    
    # Add warning thresholds to output for transparency
    warnings["thresholds"] = {
        "depth_delta": WARNING_DEPTH_DELTA_THRESHOLD,
        "branching_delta": WARNING_BRANCHING_DELTA_THRESHOLD,
        "branching_collapse_epsilon": WARNING_BRANCHING_COLLAPSE_EPSILON,
        "depth_saturation_margin": WARNING_DEPTH_SATURATION_MARGIN,
    }
    
    if theoretical_max is not None:
        warnings["theoretical_max_depth"] = theoretical_max
    
    return warnings


# -----------------------------------------------------------------------------
# Depth Evolution Time-Series
# -----------------------------------------------------------------------------


def compute_depth_timeseries(
    records: List[Dict[str, Any]],
    downsample_factor: int = 1,
) -> List[int]:
    """
    Extract depth maxima time-series from experiment records.
    
    Traces max depth per cycle for topology evolution analysis.
    
    PHASE II ONLY - Structural observation, no uplift claims.
    
    Args:
        records: Experiment log records.
        downsample_factor: Deterministic downsampling (1 = no downsampling).
    
    Returns:
        List of max depth values per (downsampled) cycle.
    """
    metrics = extract_cycle_metrics(records)
    depths = metrics.get("depth", [])
    
    if not depths:
        return []
    
    # Deterministic downsampling: take every Nth sample starting at index 0
    if downsample_factor > 1:
        depths = depths[::downsample_factor]
    
    return [int(d) for d in depths]


def compute_expected_depth_band(
    slice_name: str,
    length: int,
) -> Dict[str, List[float]]:
    """
    Generate expected depth band from curriculum parameters.
    
    Creates constant bands based on curriculum depth_max.
    
    Args:
        slice_name: Slice name for curriculum lookup.
        length: Length of the series to generate.
    
    Returns:
        Dict with 'expected_min' and 'expected_max' arrays.
    """
    expected_profile = infer_expected_dag_profile(slice_name)
    min_depth, max_depth = expected_profile.expected_depth_range
    
    # Also check curriculum for theoretical max
    theoretical_max = _get_theoretical_max_depth(slice_name)
    if theoretical_max is not None:
        max_depth = float(theoretical_max)
    
    return {
        "expected_min": [min_depth] * length,
        "expected_max": [max_depth] * length,
    }


def compute_depth_evolution_contract(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
    slice_name: str,
    downsample_factor: int = 1,
) -> Dict[str, Any]:
    """
    Compute depth evolution time-series contract.
    
    Provides time-aligned depth curves for baseline vs RFL with
    curriculum-guided expected range bands.
    
    PHASE II ONLY - Structural observation, no uplift claims.
    
    Args:
        baseline_records: Baseline experiment log records.
        rfl_records: RFL experiment log records.
        slice_name: Slice name for curriculum lookup.
        downsample_factor: Deterministic downsampling factor.
    
    Returns:
        Depth timeseries contract dict.
    """
    baseline_curve = compute_depth_timeseries(baseline_records, downsample_factor)
    rfl_curve = compute_depth_timeseries(rfl_records, downsample_factor)
    
    # Align lengths for comparison (use shorter length)
    min_len = min(len(baseline_curve), len(rfl_curve)) if baseline_curve and rfl_curve else 0
    
    if min_len > 0:
        baseline_curve = baseline_curve[:min_len]
        rfl_curve = rfl_curve[:min_len]
    
    # Generate expected band
    band = compute_expected_depth_band(slice_name, min_len if min_len > 0 else 1)
    
    return {
        "baseline": baseline_curve,
        "rfl": rfl_curve,
        "expected_min": band["expected_min"][:min_len] if min_len > 0 else band["expected_min"],
        "expected_max": band["expected_max"][:min_len] if min_len > 0 else band["expected_max"],
        "length": min_len,
        "downsample_factor": downsample_factor,
    }


# -----------------------------------------------------------------------------
# Structural Stability Score (Advisory Only)
# -----------------------------------------------------------------------------


def compute_cycle_to_cycle_variation(series: List[float]) -> float:
    """
    Compute mean absolute cycle-to-cycle variation.
    
    Args:
        series: Time series of values.
    
    Returns:
        Mean absolute difference between consecutive cycles.
    """
    if len(series) < 2:
        return 0.0
    
    diffs = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    return float(np.mean(diffs))


def compute_stability_coefficient(series: List[float]) -> float:
    """
    Compute stability coefficient (inverse of coefficient of variation).
    
    Higher values indicate more stable series.
    Range: 0.0 (highly variable) to 1.0 (perfectly stable)
    
    Args:
        series: Time series of values.
    
    Returns:
        Stability coefficient in [0, 1].
    """
    if len(series) < 2:
        return 1.0  # Single point is maximally stable
    
    arr = np.array(series, dtype=float)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    if mean_val == 0:
        return 1.0 if std_val == 0 else 0.0
    
    cv = std_val / abs(mean_val)  # Coefficient of variation
    
    # Convert to stability: 1 / (1 + cv) maps [0, inf) -> (0, 1]
    stability = 1.0 / (1.0 + cv)
    
    return float(stability)


def compute_structural_stability_score(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute structural stability score (ADVISORY ONLY).
    
    Measures topology stability across cycles. Does NOT affect exit codes.
    This is a diagnostic metric for curriculum designers.
    
    PHASE II ONLY - Advisory signal, no uplift claims.
    
    Components:
      - depth_stability_score: Stability of depth evolution
      - branching_stability_score: Stability of branching factor
      - cycle_variation: Mean cycle-to-cycle changes
    
    Args:
        baseline_records: Baseline experiment log records.
        rfl_records: RFL experiment log records.
    
    Returns:
        Stability score dict with components.
    """
    baseline_metrics = extract_cycle_metrics(baseline_records)
    rfl_metrics = extract_cycle_metrics(rfl_records)
    
    baseline_depths = [float(d) for d in baseline_metrics.get("depth", [1])]
    rfl_depths = [float(d) for d in rfl_metrics.get("depth", [1])]
    
    # Compute branching per cycle
    def branching_series(metrics: Dict[str, List[Any]]) -> List[float]:
        candidates = metrics.get("candidates", [])
        verified = metrics.get("verified", [])
        if not candidates or not verified:
            return [0.0]
        result = []
        for c, v in zip(candidates, verified):
            c, v = float(c), float(v)
            result.append(c / v if v > 0 else c)
        return result
    
    baseline_branching = branching_series(baseline_metrics)
    rfl_branching = branching_series(rfl_metrics)
    
    # Compute stability components
    baseline_depth_stability = compute_stability_coefficient(baseline_depths)
    rfl_depth_stability = compute_stability_coefficient(rfl_depths)
    
    baseline_branch_stability = compute_stability_coefficient(baseline_branching)
    rfl_branch_stability = compute_stability_coefficient(rfl_branching)
    
    # Cycle-to-cycle variation
    baseline_depth_variation = compute_cycle_to_cycle_variation(baseline_depths)
    rfl_depth_variation = compute_cycle_to_cycle_variation(rfl_depths)
    
    baseline_branch_variation = compute_cycle_to_cycle_variation(baseline_branching)
    rfl_branch_variation = compute_cycle_to_cycle_variation(rfl_branching)
    
    # Combined stability score (average of components)
    # Higher = more stable topology
    combined_baseline = (baseline_depth_stability + baseline_branch_stability) / 2
    combined_rfl = (rfl_depth_stability + rfl_branch_stability) / 2
    
    # Overall score is average of both modes
    overall_score = (combined_baseline + combined_rfl) / 2
    
    return {
        "score": round(overall_score, 4),
        "components": {
            "baseline": {
                "depth_stability": round(baseline_depth_stability, 4),
                "branching_stability": round(baseline_branch_stability, 4),
                "depth_variation": round(baseline_depth_variation, 4),
                "branching_variation": round(baseline_branch_variation, 4),
            },
            "rfl": {
                "depth_stability": round(rfl_depth_stability, 4),
                "branching_stability": round(rfl_branch_stability, 4),
                "depth_variation": round(rfl_depth_variation, 4),
                "branching_variation": round(rfl_branch_variation, 4),
            },
        },
        "interpretation": "Advisory metric only. Higher score indicates more stable topology.",
    }


# -----------------------------------------------------------------------------
# Ledger Snapshot Integration
# -----------------------------------------------------------------------------


def _canonicalize_timestamp() -> str:
    """
    Generate deterministic canonicalized timestamp.
    
    Uses UTC, ISO 8601 format, truncated to seconds.
    
    Returns:
        Timestamp string.
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    # Truncate to seconds for determinism across runs within same second
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_topology_hash(data: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of topology data.
    
    Uses canonical JSON + SHA256.
    
    Args:
        data: Dict to hash.
    
    Returns:
        Hex digest of hash.
    """
    import hashlib
    
    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def to_ledger_topology_entry(
    baseline_log: str | Path,
    rfl_log: str | Path,
    slice_name: Optional[str] = None,
    include_timeseries: bool = False,
) -> Dict[str, Any]:
    """
    Generate ledger-grade topology snapshot entry.
    
    Creates a hashable, deterministic record suitable for audit ledgers.
    Contains structural metrics, warning flags, and stability scores.
    
    PHASE II ONLY - Structural observation, no uplift claims.
    
    Contract fields:
      - topology_hash: Deterministic SHA256 of structural metrics
      - structural_metrics: Footprints and deltas
      - warning_flags: Advisory flags (depth_saturation, branching_collapse, topology_change)
      - stability: Structural stability score
      - timestamp: Canonicalized UTC timestamp
      - slice_name: Curriculum slice identifier
    
    Args:
        baseline_log: Path to baseline JSONL log.
        rfl_log: Path to RFL JSONL log.
        slice_name: Optional slice name (auto-detected if not provided).
        include_timeseries: Whether to include full depth timeseries.
    
    Returns:
        Ledger entry dict.
    
    Raises:
        FileNotFoundError: If logs don't exist.
        ValueError: If logs are empty.
    """
    baseline_log = Path(baseline_log)
    rfl_log = Path(rfl_log)
    
    # Validate inputs
    if not baseline_log.exists():
        raise FileNotFoundError(f"Baseline log not found: {baseline_log}")
    if not rfl_log.exists():
        raise FileNotFoundError(f"RFL log not found: {rfl_log}")
    
    # Load data
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)
    
    if not baseline_records:
        raise ValueError(f"Baseline log is empty: {baseline_log}")
    if not rfl_records:
        raise ValueError(f"RFL log is empty: {rfl_log}")
    
    # Auto-detect slice name
    if slice_name is None:
        slice_name = baseline_records[0].get("slice_name")
        if slice_name is None:
            slice_name = rfl_records[0].get("slice_name", "unknown")
    
    # Build analyzers
    baseline_derivations = extract_derivations_from_log(baseline_records)
    rfl_derivations = extract_derivations_from_log(rfl_records)
    
    baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
    rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
    
    # Get footprints
    baseline_footprint = baseline_analyzer.get_dag_footprint()
    rfl_footprint = rfl_analyzer.get_dag_footprint()
    
    # Compute mean branching
    baseline_mean_branching = (
        baseline_footprint["edge_count"] / 
        max(baseline_footprint["node_count"], 1)
    )
    rfl_mean_branching = (
        rfl_footprint["edge_count"] / 
        max(rfl_footprint["node_count"], 1)
    )
    
    # Compute deltas
    depth_delta = rfl_footprint["max_depth"] - baseline_footprint["max_depth"]
    node_delta = rfl_footprint["node_count"] - baseline_footprint["node_count"]
    edge_delta = rfl_footprint["edge_count"] - baseline_footprint["edge_count"]
    branching_delta = rfl_mean_branching - baseline_mean_branching
    
    # Structural metrics (hashable core)
    structural_metrics = {
        "baseline_footprint": {
            "node_count": baseline_footprint["node_count"],
            "edge_count": baseline_footprint["edge_count"],
            "max_depth": baseline_footprint["max_depth"],
            "mean_depth": round(baseline_footprint["mean_depth"], 4),
            "mean_branching": round(baseline_mean_branching, 4),
        },
        "rfl_footprint": {
            "node_count": rfl_footprint["node_count"],
            "edge_count": rfl_footprint["edge_count"],
            "max_depth": rfl_footprint["max_depth"],
            "mean_depth": round(rfl_footprint["mean_depth"], 4),
            "mean_branching": round(rfl_mean_branching, 4),
        },
        "deltas": {
            "depth_delta": depth_delta,
            "node_delta": node_delta,
            "edge_delta": edge_delta,
            "branching_delta": round(branching_delta, 4),
        },
    }
    
    # Warning flags
    warning_flags = compute_warning_flags(
        baseline_footprint, rfl_footprint,
        float(depth_delta), branching_delta,
        slice_name,
    )
    
    # Extract just the boolean flags for ledger
    flags_compact = {
        "depth_saturation": warning_flags["depth_saturation_warning"],
        "branching_collapse": warning_flags["branching_collapse_warning"],
        "topology_change": warning_flags["topology_change_warning"],
    }
    
    # Stability score
    stability = compute_structural_stability_score(baseline_records, rfl_records)
    
    # Compute topology hash (from structural metrics only, not timestamp)
    topology_hash = _compute_topology_hash(structural_metrics)
    
    # Build ledger entry
    entry: Dict[str, Any] = {
        "phase": "II",
        "type": "topology_ledger_entry",
        "version": "1.2",
        "slice_name": slice_name,
        "topology_hash": topology_hash,
        "timestamp": _canonicalize_timestamp(),
        "structural_metrics": structural_metrics,
        "warning_flags": flags_compact,
        "stability": {
            "score": stability["score"],
            "components": stability["components"],
        },
    }
    
    # Optionally include depth timeseries
    if include_timeseries:
        timeseries = compute_depth_evolution_contract(
            baseline_records, rfl_records, slice_name
        )
        entry["depth_timeseries"] = timeseries
    
    return entry


def write_ledger_topology_entry(
    entry: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """
    Write ledger topology entry to JSON file.
    
    Uses deterministic formatting (sorted keys).
    
    Args:
        entry: Ledger entry dict.
        out_path: Output file path.
    
    Returns:
        Path to written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, sort_keys=True)
    
    return out_path


# -----------------------------------------------------------------------------
# Phase III: Topology Ledger Analytics & Director Light
# -----------------------------------------------------------------------------

# Thresholds for analytics and director status
ANALYTICS_LOW_STABILITY_THRESHOLD = 0.5
DIRECTOR_WARNING_FLAG_THRESHOLD = 2
DIRECTOR_LOW_STABILITY_RUN_THRESHOLD = 2
DIRECTOR_TOPOLOGY_CHANGE_THRESHOLD = 3


def analyze_topology_ledger_entries(
    entries: Sequence[Dict[str, Any]],
    low_stability_threshold: float = ANALYTICS_LOW_STABILITY_THRESHOLD,
) -> Dict[str, Any]:
    """
    Analyze a collection of topology ledger entries.
    
    Produces aggregated analytics over multiple ledger snapshots for
    trend analysis and Director reporting.
    
    PHASE III — Analytics layer, no policy interpretation.
    
    Args:
        entries: Sequence of ledger entries from to_ledger_topology_entry().
        low_stability_threshold: Threshold for flagging low-stability runs.
    
    Returns:
        Analytics dict with:
          - schema_version: Analytics schema version
          - entry_count: Number of entries analyzed
          - average_stability_score: Mean stability across entries
          - max_depth_over_time: List of max depths per entry
          - frequency_of_warning_flags: Count per warning type
          - runs_with_low_stability: List of entries with score < threshold
    """
    if not entries:
        return {
            "schema_version": "1.0",
            "entry_count": 0,
            "average_stability_score": 0.0,
            "max_depth_over_time": [],
            "frequency_of_warning_flags": {
                "depth_saturation": 0,
                "branching_collapse": 0,
                "topology_change": 0,
            },
            "runs_with_low_stability": [],
            "low_stability_threshold": low_stability_threshold,
        }
    
    # Collect stability scores
    stability_scores = []
    for entry in entries:
        stability = entry.get("stability", {})
        score = stability.get("score", 0.0)
        stability_scores.append(score)
    
    # Compute average stability
    avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.0
    
    # Collect max depths over time
    max_depths = []
    for entry in entries:
        metrics = entry.get("structural_metrics", {})
        # Take max of baseline and rfl max_depth
        baseline_depth = metrics.get("baseline_footprint", {}).get("max_depth", 0)
        rfl_depth = metrics.get("rfl_footprint", {}).get("max_depth", 0)
        max_depths.append(max(baseline_depth, rfl_depth))
    
    # Count warning flags
    flag_counts = {
        "depth_saturation": 0,
        "branching_collapse": 0,
        "topology_change": 0,
    }
    for entry in entries:
        flags = entry.get("warning_flags", {})
        if flags.get("depth_saturation"):
            flag_counts["depth_saturation"] += 1
        if flags.get("branching_collapse"):
            flag_counts["branching_collapse"] += 1
        if flags.get("topology_change"):
            flag_counts["topology_change"] += 1
    
    # Identify low-stability runs
    low_stability_runs = []
    for i, (entry, score) in enumerate(zip(entries, stability_scores)):
        if score < low_stability_threshold:
            run_id = entry.get("slice_name", f"run_{i}")
            timestamp = entry.get("timestamp", "unknown")
            low_stability_runs.append({
                "run_id": run_id,
                "timestamp": timestamp,
                "stability_score": round(score, 4),
            })
    
    return {
        "schema_version": "1.0",
        "entry_count": len(entries),
        "average_stability_score": round(avg_stability, 4),
        "max_depth_over_time": max_depths,
        "frequency_of_warning_flags": flag_counts,
        "runs_with_low_stability": low_stability_runs,
        "low_stability_threshold": low_stability_threshold,
    }


def map_topology_to_director_status(
    analytics: Dict[str, Any],
    warning_flag_threshold: int = DIRECTOR_WARNING_FLAG_THRESHOLD,
    low_stability_run_threshold: int = DIRECTOR_LOW_STABILITY_RUN_THRESHOLD,
    topology_change_threshold: int = DIRECTOR_TOPOLOGY_CHANGE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Map topology analytics to Director console status light.
    
    Produces a status light (GREEN/YELLOW/RED) with neutral rationale.
    This is an ADVISORY signal — no policy implications, no success/failure language.
    
    PHASE III — Director-facing summary, neutral language only.
    
    Status logic:
      - GREEN: High average stability (≥0.7), few warning flags, no low-stability runs
      - YELLOW: Moderate warning flags OR some low-stability runs
      - RED: Frequent topology_change warnings OR many low-stability runs
    
    Args:
        analytics: Output from analyze_topology_ledger_entries().
        warning_flag_threshold: Total warning count for YELLOW.
        low_stability_run_threshold: Low-stability run count for RED.
        topology_change_threshold: Topology change count for RED.
    
    Returns:
        Director status dict:
          - status_light: GREEN | YELLOW | RED
          - rationale: Neutral explanation
          - thresholds: Applied thresholds for transparency
    """
    avg_stability = analytics.get("average_stability_score", 0.0)
    flag_counts = analytics.get("frequency_of_warning_flags", {})
    low_stability_runs = analytics.get("runs_with_low_stability", [])
    entry_count = analytics.get("entry_count", 0)
    
    # Count total warning flags
    total_flags = sum(flag_counts.values())
    topology_changes = flag_counts.get("topology_change", 0)
    low_stability_count = len(low_stability_runs)
    
    # Determine status light
    status_light = "GREEN"
    rationale_parts = []
    
    # RED conditions
    if topology_changes >= topology_change_threshold:
        status_light = "RED"
        rationale_parts.append(
            f"topology_change observed {topology_changes} times (threshold: {topology_change_threshold})"
        )
    
    if low_stability_count >= low_stability_run_threshold:
        status_light = "RED"
        rationale_parts.append(
            f"{low_stability_count} runs with low stability (threshold: {low_stability_run_threshold})"
        )
    
    # YELLOW conditions (if not already RED)
    if status_light != "RED":
        if total_flags >= warning_flag_threshold:
            status_light = "YELLOW"
            rationale_parts.append(
                f"{total_flags} warning flags observed (threshold: {warning_flag_threshold})"
            )
        
        if low_stability_count > 0 and low_stability_count < low_stability_run_threshold:
            status_light = "YELLOW"
            rationale_parts.append(
                f"{low_stability_count} run(s) with low stability"
            )
        
        if avg_stability < 0.7 and entry_count > 0:
            status_light = "YELLOW"
            rationale_parts.append(
                f"average stability {avg_stability:.3f} (below 0.7)"
            )
    
    # GREEN conditions (if no issues)
    if status_light == "GREEN":
        if entry_count == 0:
            rationale_parts.append("no ledger entries to analyze")
        else:
            rationale_parts.append(
                f"topology metrics within expected ranges across {entry_count} entries"
            )
    
    # Build neutral rationale (no value-laden words)
    rationale = "; ".join(rationale_parts)
    
    return {
        "status_light": status_light,
        "rationale": rationale,
        "thresholds": {
            "warning_flag_threshold": warning_flag_threshold,
            "low_stability_run_threshold": low_stability_run_threshold,
            "topology_change_threshold": topology_change_threshold,
            "stability_yellow_threshold": 0.7,
        },
        "metrics_summary": {
            "entry_count": entry_count,
            "average_stability": avg_stability,
            "total_warning_flags": total_flags,
            "low_stability_runs": low_stability_count,
            "topology_changes": topology_changes,
        },
    }


def summarize_topology_for_global_health(
    analytics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize topology analytics for global health integration.
    
    Produces a compact summary suitable for embedding in broader
    system health reports.
    
    PHASE III — Global health signal, advisory only.
    
    Status logic:
      - OK: No warning flags, average stability ≥ 0.7
      - WARN: Some warning flags OR moderate stability
      - BLOCK: Frequent topology changes OR very low stability
    
    Args:
        analytics: Output from analyze_topology_ledger_entries().
    
    Returns:
        Global health summary:
          - topology_ok: Boolean for quick checks
          - average_stability_score: Stability metric
          - warning_run_count: Count of runs with warnings
          - status: OK | WARN | BLOCK
    """
    avg_stability = analytics.get("average_stability_score", 0.0)
    flag_counts = analytics.get("frequency_of_warning_flags", {})
    low_stability_runs = analytics.get("runs_with_low_stability", [])
    entry_count = analytics.get("entry_count", 0)
    
    total_flags = sum(flag_counts.values())
    topology_changes = flag_counts.get("topology_change", 0)
    low_stability_count = len(low_stability_runs)
    
    # Count runs with any warning
    warning_run_count = 0
    for flag_type, count in flag_counts.items():
        warning_run_count += count
    # Also count low stability runs
    warning_run_count = max(warning_run_count, low_stability_count)
    
    # Determine status
    status = "OK"
    topology_ok = True
    
    # BLOCK conditions
    if topology_changes >= 3 or low_stability_count >= 3 or avg_stability < 0.3:
        status = "BLOCK"
        topology_ok = False
    # WARN conditions
    elif total_flags > 0 or low_stability_count > 0 or avg_stability < 0.7:
        status = "WARN"
        topology_ok = False
    
    # Handle empty case
    if entry_count == 0:
        status = "OK"
        topology_ok = True
        warning_run_count = 0
    
    return {
        "topology_ok": topology_ok,
        "average_stability_score": round(avg_stability, 4),
        "warning_run_count": warning_run_count,
        "status": status,
        "entry_count": entry_count,
    }


# -----------------------------------------------------------------------------
# Phase IV: Topology-Guided Curriculum & Policy Advisor
# -----------------------------------------------------------------------------

# Thresholds for slice-level analysis
SLICE_DEPTH_TREND_THRESHOLD = 0.1  # 10% change for trend detection
SLICE_ATTENTION_THRESHOLD = 1  # Warning flags for ATTENTION
SLICE_STRESSED_THRESHOLD = 2  # Warning flags for STRESSED


def build_slice_topology_curriculum_view(
    analytics: Dict[str, Any],
    manifest_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build slice-level topology curriculum view.
    
    Analyzes topology patterns per slice to inform curriculum design.
    Provides depth trends, branching behavior, and slice health status.
    
    PHASE IV — Curriculum guidance, advisory only.
    
    Args:
        analytics: Output from analyze_topology_ledger_entries().
        manifest_timeline: Optional timeline manifest (for future use).
    
    Returns:
        Slice topology view dict with per-slice analysis:
          - slice_name: {
              - typical_max_depth: Typical maximum depth observed
              - depth_trend: "SHALLOWING" | "STABLE" | "DEEPENING"
              - branching_behavior: Qualitative descriptor
              - slice_topology_status: "OK" | "ATTENTION" | "STRESSED"
            }
    """
    entry_count = analytics.get("entry_count", 0)
    max_depths = analytics.get("max_depth_over_time", [])
    flag_counts = analytics.get("frequency_of_warning_flags", {})
    low_stability_runs = analytics.get("runs_with_low_stability", [])
    
    # Compute typical max depth (median of max depths)
    typical_max_depth = 0
    if max_depths:
        sorted_depths = sorted(max_depths)
        mid = len(sorted_depths) // 2
        typical_max_depth = sorted_depths[mid] if len(sorted_depths) % 2 == 1 else (
            (sorted_depths[mid - 1] + sorted_depths[mid]) / 2
        )
        typical_max_depth = int(typical_max_depth)
    
    # Determine depth trend
    depth_trend = "STABLE"
    if len(max_depths) >= 3:
        # Compare first third vs last third
        first_third = max_depths[:len(max_depths) // 3]
        last_third = max_depths[-len(max_depths) // 3:]
        
        avg_first = sum(first_third) / len(first_third) if first_third else 0
        avg_last = sum(last_third) / len(last_third) if last_third else 0
        
        if avg_last > avg_first * (1 + SLICE_DEPTH_TREND_THRESHOLD):
            depth_trend = "DEEPENING"
        elif avg_last < avg_first * (1 - SLICE_DEPTH_TREND_THRESHOLD):
            depth_trend = "SHALLOWING"
    
    # Determine branching behavior from warning flags
    branching_behavior = "normal"
    if flag_counts.get("branching_collapse", 0) > 0:
        branching_behavior = "low_edge_density"
    elif flag_counts.get("topology_change", 0) > 0:
        branching_behavior = "variable_structure"
    
    # Determine slice topology status
    total_flags = sum(flag_counts.values())
    low_stability_count = len(low_stability_runs)
    
    slice_topology_status = "OK"
    if total_flags >= SLICE_STRESSED_THRESHOLD or low_stability_count >= 2:
        slice_topology_status = "STRESSED"
    elif total_flags >= SLICE_ATTENTION_THRESHOLD or low_stability_count >= 1:
        slice_topology_status = "ATTENTION"
    
    # Build per-slice view (for now, aggregate view; can be extended for multi-slice)
    slice_view = {
        "typical_max_depth": typical_max_depth,
        "depth_trend": depth_trend,
        "branching_behavior": branching_behavior,
        "slice_topology_status": slice_topology_status,
        "depth_history": max_depths,
        "warning_summary": {
            "total_flags": total_flags,
            "depth_saturation": flag_counts.get("depth_saturation", 0),
            "branching_collapse": flag_counts.get("branching_collapse", 0),
            "topology_change": flag_counts.get("topology_change", 0),
        },
    }
    
    return {
        "slice_view": slice_view,
        "entry_count": entry_count,
    }


def derive_topology_policy_recommendations(
    slice_topology_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive topology-aware policy recommendations.
    
    Analyzes slice topology patterns to suggest policy adjustments:
    - Which slices need policy adjustment
    - Which slices are suitable for deeper exploration
    - Neutral recommendation notes
    
    PHASE IV — Policy guidance, advisory only.
    
    Args:
        slice_topology_view: Output from build_slice_topology_curriculum_view().
    
    Returns:
        Policy recommendations dict:
          - slices_needing_policy_adjustment: List of slice identifiers
          - slices_suitable_for_deeper_slices: List of slice identifiers
          - policy_recommendation_notes: Neutral descriptive notes
    """
    slice_view = slice_topology_view.get("slice_view", {})
    
    depth_trend = slice_view.get("depth_trend", "STABLE")
    slice_status = slice_view.get("slice_topology_status", "OK")
    branching_behavior = slice_view.get("branching_behavior", "normal")
    warning_summary = slice_view.get("warning_summary", {})
    
    slices_needing_policy_adjustment = []
    slices_suitable_for_deeper_slices = []
    recommendation_notes = []
    
    # Determine if policy adjustment needed
    needs_adjustment = False
    adjustment_reasons = []
    
    if slice_status == "STRESSED":
        needs_adjustment = True
        adjustment_reasons.append("repeated topology stress indicators")
    
    if warning_summary.get("topology_change", 0) >= 2:
        needs_adjustment = True
        adjustment_reasons.append("frequent topology change flags")
    
    if branching_behavior == "low_edge_density" and warning_summary.get("branching_collapse", 0) > 0:
        needs_adjustment = True
        adjustment_reasons.append("persistent low edge density")
    
    if needs_adjustment:
        slices_needing_policy_adjustment.append("current_slice")  # Can be extended with actual slice names
        recommendation_notes.append(
            f"Policy adjustment suggested: {'; '.join(adjustment_reasons)}"
        )
    
    # Determine if suitable for deeper exploration
    suitable_for_deeper = False
    deeper_reasons = []
    
    if depth_trend == "STABLE" and slice_status == "OK":
        suitable_for_deeper = True
        deeper_reasons.append("stable depth trend with no topology stress")
    
    if depth_trend == "SHALLOWING" and slice_status in ["OK", "ATTENTION"]:
        suitable_for_deeper = True
        deeper_reasons.append("shallowing trend suggests capacity for deeper exploration")
    
    if suitable_for_deeper:
        slices_suitable_for_deeper_slices.append("current_slice")  # Can be extended
        recommendation_notes.append(
            f"Suitable for deeper slices: {'; '.join(deeper_reasons)}"
        )
    
    # Add neutral notes about RFL/curriculum direction
    if depth_trend == "DEEPENING" and slice_status == "STRESSED":
        recommendation_notes.append(
            "Deepening trend with stress indicators: consider stabilizing depth before further expansion"
        )
    elif depth_trend == "STABLE" and slice_status == "OK":
        recommendation_notes.append(
            "Stable topology with no stress: current curriculum parameters appear aligned"
        )
    
    return {
        "slices_needing_policy_adjustment": slices_needing_policy_adjustment,
        "slices_suitable_for_deeper_slices": slices_suitable_for_deeper_slices,
        "policy_recommendation_notes": recommendation_notes,
        "analysis_basis": {
            "depth_trend": depth_trend,
            "slice_status": slice_status,
            "branching_behavior": branching_behavior,
        },
    }


def build_topology_director_panel(
    analytics: Dict[str, Any],
    slice_view: Optional[Dict[str, Any]] = None,
    policy_advice: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build extended Director topology panel.
    
    Combines analytics, slice view, and policy advice into a comprehensive
    Director-facing dashboard with status light, metrics, and neutral headlines.
    
    PHASE IV — Director dashboard, advisory only.
    
    Args:
        analytics: Output from analyze_topology_ledger_entries().
        slice_view: Optional output from build_slice_topology_curriculum_view().
        policy_advice: Optional output from derive_topology_policy_recommendations().
    
    Returns:
        Director panel dict:
          - status_light: GREEN | YELLOW | RED
          - average_stability_score: Stability metric
          - topology_ok: Boolean health indicator
          - slices_stressed: Count of stressed slices
          - headline: Neutral summary headline
          - hints: List of neutral observation hints
    """
    from experiments.visualize_dag_topology import (
        map_topology_to_director_status,
        summarize_topology_for_global_health,
    )
    
    # Get director status and global health
    director_status = map_topology_to_director_status(analytics)
    global_health = summarize_topology_for_global_health(analytics)
    
    status_light = director_status.get("status_light", "GREEN")
    avg_stability = analytics.get("average_stability_score", 0.0)
    topology_ok = global_health.get("topology_ok", True)
    
    # Count stressed slices
    slices_stressed = 0
    if slice_view:
        slice_data = slice_view.get("slice_view", {})
        if slice_data.get("slice_topology_status") == "STRESSED":
            slices_stressed = 1
    
    # Build headline
    entry_count = analytics.get("entry_count", 0)
    flag_counts = analytics.get("frequency_of_warning_flags", {})
    topology_changes = flag_counts.get("topology_change", 0)
    
    headline_parts = []
    if entry_count > 0:
        if topology_ok and avg_stability >= 0.7:
            headline_parts.append(f"Topology metrics stable across {entry_count} run(s)")
        elif topology_changes > 0:
            headline_parts.append(f"{topology_changes} topology change observation(s) across {entry_count} run(s)")
        else:
            headline_parts.append(f"Topology analysis for {entry_count} run(s)")
    else:
        headline_parts.append("No topology data available")
    
    headline = "; ".join(headline_parts)
    
    # Build hints (neutral observations)
    hints = []
    
    # Depth frontier hint
    max_depths = analytics.get("max_depth_over_time", [])
    if len(max_depths) >= 3:
        recent_depths = max_depths[-3:]
        if len(set(recent_depths)) == 1:
            hints.append(f"Depth frontier stable across last {len(recent_depths)} runs")
        elif max(recent_depths) - min(recent_depths) <= 1:
            hints.append("Depth frontier showing minimal variation")
    
    # Topology change hint
    if topology_changes >= 2:
        hints.append(f"{topology_changes} slices show repeated topology_change flags")
    elif topology_changes == 1:
        hints.append("One slice shows topology_change flag")
    
    # Stability hint
    if avg_stability >= 0.8:
        hints.append("High average stability observed")
    elif avg_stability < 0.5:
        hints.append("Low average stability observed")
    
    # Policy adjustment hint
    if policy_advice:
        adjustment_slices = policy_advice.get("slices_needing_policy_adjustment", [])
        if adjustment_slices:
            hints.append(f"{len(adjustment_slices)} slice(s) may benefit from policy adjustment")
        
        deeper_slices = policy_advice.get("slices_suitable_for_deeper_slices", [])
        if deeper_slices:
            hints.append(f"{len(deeper_slices)} slice(s) suitable for deeper exploration")
    
    return {
        "status_light": status_light,
        "average_stability_score": round(avg_stability, 4),
        "topology_ok": topology_ok,
        "slices_stressed": slices_stressed,
        "headline": headline,
        "hints": hints,
        "metrics": {
            "entry_count": entry_count,
            "total_warning_flags": sum(flag_counts.values()),
            "low_stability_runs": len(analytics.get("runs_with_low_stability", [])),
        },
    }


# -----------------------------------------------------------------------------
# Topology Risk Envelope & Curriculum Progression Predictor
# -----------------------------------------------------------------------------

# Thresholds for risk envelope
RISK_VOLATILITY_HIGH_THRESHOLD = 1.5  # Standard deviation threshold
RISK_DEPTH_INCREASE_THRESHOLD = 0.15  # 15% increase for risk
RISK_STRESSED_THRESHOLD = 2.0  # Combined volatility + depth increase


def build_topology_risk_envelope(
    ledger_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute topology risk envelope from ledger history.
    
    Analyzes depth volatility and trends to establish risk bands for
    long-horizon topology control.
    
    PHASE IV — Risk assessment, advisory only.
    
    Args:
        ledger_history: Sequence of ledger entries from to_ledger_topology_entry().
    
    Returns:
        Risk envelope dict:
          - max_depth_band: [min, max] depth range observed
          - branching_volatility: Standard deviation of depth deltas
          - risk_band: "OK" | "ATTENTION" | "STRESSED"
          - envelope_summary: Neutral descriptive summary
    """
    if not ledger_history:
        return {
            "max_depth_band": [0, 0],
            "branching_volatility": 0.0,
            "risk_band": "OK",
            "envelope_summary": "No ledger history available",
            "depth_history": [],
            "depth_deltas": [],
        }
    
    # Extract max depths from each ledger entry
    max_depths = []
    for entry in ledger_history:
        metrics = entry.get("structural_metrics", {})
        baseline_depth = metrics.get("baseline_footprint", {}).get("max_depth", 0)
        rfl_depth = metrics.get("rfl_footprint", {}).get("max_depth", 0)
        max_depths.append(max(baseline_depth, rfl_depth))
    
    if not max_depths:
        return {
            "max_depth_band": [0, 0],
            "branching_volatility": 0.0,
            "risk_band": "OK",
            "envelope_summary": "No depth data in ledger history",
            "depth_history": [],
            "depth_deltas": [],
        }
    
    # Compute max depth band
    max_depth_band = [min(max_depths), max(max_depths)]
    
    # Compute depth deltas (cycle-to-cycle changes)
    depth_deltas = []
    for i in range(1, len(max_depths)):
        delta = max_depths[i] - max_depths[i - 1]
        depth_deltas.append(float(delta))
    
    # Compute volatility (standard deviation of depth deltas)
    branching_volatility = 0.0
    if len(depth_deltas) > 1:
        branching_volatility = float(np.std(depth_deltas))
    elif len(depth_deltas) == 1:
        branching_volatility = abs(depth_deltas[0])
    
    # Determine risk band
    risk_band = "OK"
    
    # Check for depth increase trend
    depth_increasing = False
    if len(max_depths) >= 3:
        first_third = max_depths[:len(max_depths) // 3]
        last_third = max_depths[-len(max_depths) // 3:]
        avg_first = sum(first_third) / len(first_third) if first_third else 0
        avg_last = sum(last_third) / len(last_third) if last_third else 0
        
        if avg_first > 0:
            depth_increase_ratio = (avg_last - avg_first) / avg_first
            depth_increasing = depth_increase_ratio >= RISK_DEPTH_INCREASE_THRESHOLD
    
    # STRESSED: High volatility AND depth increasing
    if branching_volatility >= RISK_VOLATILITY_HIGH_THRESHOLD and depth_increasing:
        risk_band = "STRESSED"
    # ATTENTION: High volatility OR depth increasing
    elif branching_volatility >= RISK_VOLATILITY_HIGH_THRESHOLD or depth_increasing:
        risk_band = "ATTENTION"
    
    # Build envelope summary
    summary_parts = []
    summary_parts.append(f"Depth range: {max_depth_band[0]} to {max_depth_band[1]}")
    summary_parts.append(f"Volatility: {branching_volatility:.2f}")
    if depth_increasing:
        summary_parts.append("Depth trend: increasing")
    else:
        summary_parts.append("Depth trend: stable or decreasing")
    
    envelope_summary = "; ".join(summary_parts)
    
    return {
        "max_depth_band": max_depth_band,
        "branching_volatility": round(branching_volatility, 4),
        "risk_band": risk_band,
        "envelope_summary": envelope_summary,
        "depth_history": max_depths,
        "depth_deltas": [round(d, 4) for d in depth_deltas],
        "depth_increasing": depth_increasing,
    }


def predict_curriculum_progression_from_topology(
    slice_views: Dict[str, Any],
    risk_envelope: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict curriculum progression readiness from topology analysis.
    
    Combines slice-level topology views with risk envelope to determine
    which slices are ready for next depth level and which need stabilization.
    
    PHASE IV — Progression prediction, advisory only.
    
    Args:
        slice_views: Output from build_slice_topology_curriculum_view().
        risk_envelope: Output from build_topology_risk_envelope().
    
    Returns:
        Progression prediction dict:
          - slices_ready_for_next_depth: List of slice identifiers
          - slices_needing_stabilization: List of slice identifiers
          - readiness_status: "READY" | "STABILIZE" | "HOLD"
          - notes: Neutral descriptive notes
    """
    slice_view = slice_views.get("slice_view", {})
    risk_band = risk_envelope.get("risk_band", "OK")
    depth_trend = slice_view.get("depth_trend", "STABLE")
    slice_status = slice_view.get("slice_topology_status", "OK")
    branching_volatility = risk_envelope.get("branching_volatility", 0.0)
    
    slices_ready_for_next_depth = []
    slices_needing_stabilization = []
    notes = []
    
    # Determine readiness
    readiness_status = "HOLD"
    
    # Ready conditions: stable/OK slice with low risk
    if (
        depth_trend in ["STABLE", "SHALLOWING"]
        and slice_status == "OK"
        and risk_band == "OK"
        and branching_volatility < RISK_VOLATILITY_HIGH_THRESHOLD
    ):
        slices_ready_for_next_depth.append("current_slice")  # Can be extended with actual slice names
        readiness_status = "READY"
        notes.append(
            "Topology metrics indicate readiness for next depth level: stable trend, no stress indicators, low volatility"
        )
    
    # Stabilization needed conditions
    needs_stabilization = False
    stabilization_reasons = []
    
    if risk_band == "STRESSED":
        needs_stabilization = True
        stabilization_reasons.append("high risk envelope")
    
    if slice_status == "STRESSED":
        needs_stabilization = True
        stabilization_reasons.append("slice topology stress")
    
    if depth_trend == "DEEPENING" and slice_status in ["ATTENTION", "STRESSED"]:
        needs_stabilization = True
        stabilization_reasons.append("deepening trend with stress indicators")
    
    if branching_volatility >= RISK_VOLATILITY_HIGH_THRESHOLD:
        needs_stabilization = True
        stabilization_reasons.append("high depth volatility")
    
    if needs_stabilization:
        slices_needing_stabilization.append("current_slice")  # Can be extended
        readiness_status = "STABILIZE"
        notes.append(
            f"Stabilization recommended: {'; '.join(stabilization_reasons)}"
        )
    
    # HOLD status (default)
    if readiness_status == "HOLD":
        notes.append(
            "Current topology metrics suggest maintaining current depth level"
        )
    
    return {
        "slices_ready_for_next_depth": slices_ready_for_next_depth,
        "slices_needing_stabilization": slices_needing_stabilization,
        "readiness_status": readiness_status,
        "notes": notes,
        "analysis_basis": {
            "risk_band": risk_band,
            "depth_trend": depth_trend,
            "slice_status": slice_status,
            "branching_volatility": round(branching_volatility, 4),
        },
    }


# -----------------------------------------------------------------------------
# Phase V: Topological Pressure Field & Curriculum Promotion Gate
# -----------------------------------------------------------------------------

# Pressure field weights and thresholds
PRESSURE_DEPTH_WEIGHT = 0.4
PRESSURE_BRANCHING_WEIGHT = 0.3
PRESSURE_RISK_WEIGHT = 0.3
PRESSURE_HIGH_THRESHOLD = 0.7
PRESSURE_MEDIUM_THRESHOLD = 0.4


def _normalize_depth_trend(depth_trend: str) -> float:
    """
    Normalize depth trend to [0, 1] scale.
    
    DEEPENING = 1.0 (highest pressure)
    STABLE = 0.5 (medium pressure)
    SHALLOWING = 0.0 (lowest pressure)
    """
    trend_map = {
        "DEEPENING": 1.0,
        "STABLE": 0.5,
        "SHALLOWING": 0.0,
    }
    return trend_map.get(depth_trend, 0.5)


def _normalize_branching_volatility(volatility: float, max_volatility: float = 3.0) -> float:
    """
    Normalize branching volatility to [0, 1] scale.
    
    Clips at max_volatility for normalization.
    """
    if max_volatility <= 0:
        return 0.0
    normalized = min(volatility / max_volatility, 1.0)
    return float(normalized)


def _normalize_risk_band(risk_band: str) -> float:
    """
    Normalize risk band to [0, 1] scale.
    
    STRESSED = 1.0 (highest pressure)
    ATTENTION = 0.5 (medium pressure)
    OK = 0.0 (lowest pressure)
    """
    risk_map = {
        "STRESSED": 1.0,
        "ATTENTION": 0.5,
        "OK": 0.0,
    }
    return risk_map.get(risk_band, 0.0)


def build_topological_pressure_field(
    depth_trend: str,
    branching_volatility: float,
    risk_envelope: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topological pressure field from depth trend, volatility, and risk envelope.
    
    Computes a composite pressure metric by normalizing and weighting components.
    Provides pressure band classification and neutral notes.
    
    PHASE V — Pressure assessment, advisory only.
    
    Args:
        depth_trend: Depth trend from slice view ("DEEPENING" | "STABLE" | "SHALLOWING").
        branching_volatility: Branching volatility from risk envelope.
        risk_envelope: Risk envelope dict from build_topology_risk_envelope().
    
    Returns:
        Pressure field dict:
          - slice_pressure: Composite pressure score [0, 1]
          - pressure_components: Normalized component scores
          - pressure_band: "LOW" | "MEDIUM" | "HIGH"
          - neutral_notes: List of neutral descriptive notes
    """
    # Extract risk band from envelope
    risk_band = risk_envelope.get("risk_band", "OK")
    
    # Normalize components to [0, 1]
    depth_pressure = _normalize_depth_trend(depth_trend)
    branching_pressure = _normalize_branching_volatility(branching_volatility)
    risk_pressure = _normalize_risk_band(risk_band)
    
    # Compute weighted composite pressure
    slice_pressure = (
        depth_pressure * PRESSURE_DEPTH_WEIGHT +
        branching_pressure * PRESSURE_BRANCHING_WEIGHT +
        risk_pressure * PRESSURE_RISK_WEIGHT
    )
    
    # Determine pressure band
    if slice_pressure >= PRESSURE_HIGH_THRESHOLD:
        pressure_band = "HIGH"
    elif slice_pressure >= PRESSURE_MEDIUM_THRESHOLD:
        pressure_band = "MEDIUM"
    else:
        pressure_band = "LOW"
    
    # Build neutral notes
    notes = []
    notes.append(f"Composite pressure: {slice_pressure:.3f} ({pressure_band} band)")
    
    if depth_pressure > 0.7:
        notes.append("Depth trend contributing to elevated pressure")
    elif depth_pressure < 0.3:
        notes.append("Depth trend contributing to reduced pressure")
    
    if branching_pressure > 0.7:
        notes.append("Branching volatility contributing to elevated pressure")
    elif branching_pressure < 0.3:
        notes.append("Branching volatility contributing to reduced pressure")
    
    if risk_pressure > 0.7:
        notes.append("Risk envelope contributing to elevated pressure")
    elif risk_pressure < 0.3:
        notes.append("Risk envelope contributing to reduced pressure")
    
    return {
        "slice_pressure": round(slice_pressure, 4),
        "pressure_components": {
            "depth": round(depth_pressure, 4),
            "branching": round(branching_pressure, 4),
            "risk": round(risk_pressure, 4),
        },
        "pressure_band": pressure_band,
        "neutral_notes": notes,
        "weights": {
            "depth": PRESSURE_DEPTH_WEIGHT,
            "branching": PRESSURE_BRANCHING_WEIGHT,
            "risk": PRESSURE_RISK_WEIGHT,
        },
    }


def topology_curriculum_promotion_gate(
    slice_topology_view: Dict[str, Any],
    pressure_field: Dict[str, Any],
    progression_predictor: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Topology × Curriculum Promotion Gate (advisory).
    
    Combines slice topology view, pressure field, and progression predictor
    to determine promotion readiness. This is an ADVISORY gate only.
    
    PHASE V — Promotion gate, advisory only.
    
    Args:
        slice_topology_view: Output from build_slice_topology_curriculum_view().
        pressure_field: Output from build_topological_pressure_field().
        progression_predictor: Output from predict_curriculum_progression_from_topology().
    
    Returns:
        Promotion gate dict:
          - promotion_status: "OK" | "ATTENTION" | "BLOCK"
          - explanations: List of neutral explanations
          - slices_at_risk: List of slice identifiers at risk
    """
    slice_view = slice_topology_view.get("slice_view", {})
    slice_status = slice_view.get("slice_topology_status", "OK")
    pressure_band = pressure_field.get("pressure_band", "LOW")
    slice_pressure = pressure_field.get("slice_pressure", 0.0)
    readiness_status = progression_predictor.get("readiness_status", "HOLD")
    
    # Determine promotion status
    promotion_status = "OK"
    explanations = []
    slices_at_risk = []
    
    # BLOCK conditions
    if slice_status == "STRESSED":
        promotion_status = "BLOCK"
        explanations.append("Slice topology status: STRESSED")
        slices_at_risk.append("current_slice")
    
    if pressure_band == "HIGH" and slice_pressure >= 0.8:
        promotion_status = "BLOCK"
        explanations.append("High topological pressure detected")
        if "current_slice" not in slices_at_risk:
            slices_at_risk.append("current_slice")
    
    if readiness_status == "STABILIZE":
        promotion_status = "BLOCK"
        explanations.append("Progression predictor indicates stabilization needed")
        if "current_slice" not in slices_at_risk:
            slices_at_risk.append("current_slice")
    
    # ATTENTION conditions (if not already BLOCK)
    if promotion_status != "BLOCK":
        if slice_status == "ATTENTION":
            promotion_status = "ATTENTION"
            explanations.append("Slice topology status: ATTENTION")
        
        if pressure_band == "MEDIUM" or (pressure_band == "HIGH" and slice_pressure < 0.8):
            promotion_status = "ATTENTION"
            explanations.append(f"Moderate topological pressure: {pressure_band} band")
        
        if readiness_status == "HOLD":
            promotion_status = "ATTENTION"
            explanations.append("Progression predictor indicates HOLD status")
    
    # OK status (default if no issues)
    if promotion_status == "OK":
        explanations.append("Topology metrics within acceptable ranges for promotion")
    
    return {
        "promotion_status": promotion_status,
        "explanations": explanations,
        "slices_at_risk": slices_at_risk,
        "gate_components": {
            "slice_status": slice_status,
            "pressure_band": pressure_band,
            "readiness_status": readiness_status,
        },
    }


def build_topology_console_tile(
    analytics: Dict[str, Any],
    pressure_field: Optional[Dict[str, Any]] = None,
    promotion_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build topology console tile with pressure hotspots.
    
    Extended director panel that includes pressure field information
    and promotion gate status.
    
    PHASE V — Console tile, advisory only.
    
    Args:
        analytics: Output from analyze_topology_ledger_entries().
        pressure_field: Optional output from build_topological_pressure_field().
        promotion_gate: Optional output from topology_curriculum_promotion_gate().
    
    Returns:
        Console tile dict:
          - status_light: GREEN | YELLOW | RED
          - promotion_gate: OK | ATTENTION | BLOCK
          - pressure_hotspots: List of pressure observations
          - headline: Neutral summary headline
    """
    from experiments.visualize_dag_topology import (
        map_topology_to_director_status,
        summarize_topology_for_global_health,
    )
    
    # Get director status and global health
    director_status = map_topology_to_director_status(analytics)
    global_health = summarize_topology_for_global_health(analytics)
    
    status_light = director_status.get("status_light", "GREEN")
    topology_ok = global_health.get("topology_ok", True)
    
    # Get promotion gate status
    promotion_gate_status = "OK"
    if promotion_gate:
        promotion_gate_status = promotion_gate.get("promotion_status", "OK")
    
    # Build pressure hotspots
    pressure_hotspots = []
    
    if pressure_field:
        pressure_band = pressure_field.get("pressure_band", "LOW")
        slice_pressure = pressure_field.get("slice_pressure", 0.0)
        
        if pressure_band == "HIGH":
            pressure_hotspots.append(
                f"High pressure detected: {slice_pressure:.3f} ({pressure_band} band)"
            )
        elif pressure_band == "MEDIUM":
            pressure_hotspots.append(
                f"Moderate pressure: {slice_pressure:.3f} ({pressure_band} band)"
            )
        
        # Component hotspots
        components = pressure_field.get("pressure_components", {})
        if components.get("depth", 0) > 0.7:
            pressure_hotspots.append("Depth trend contributing to pressure")
        if components.get("branching", 0) > 0.7:
            pressure_hotspots.append("Branching volatility contributing to pressure")
        if components.get("risk", 0) > 0.7:
            pressure_hotspots.append("Risk envelope contributing to pressure")
    
    # Build headline
    entry_count = analytics.get("entry_count", 0)
    headline_parts = []
    
    if entry_count > 0:
        if promotion_gate_status == "BLOCK":
            headline_parts.append(f"Promotion gate: BLOCK across {entry_count} run(s)")
        elif promotion_gate_status == "ATTENTION":
            headline_parts.append(f"Promotion gate: ATTENTION across {entry_count} run(s)")
        else:
            headline_parts.append(f"Topology console: {entry_count} run(s) analyzed")
    else:
        headline_parts.append("No topology data available")
    
    headline = "; ".join(headline_parts)
    
    return {
        "status_light": status_light,
        "promotion_gate": promotion_gate_status,
        "pressure_hotspots": pressure_hotspots,
        "headline": headline,
        "topology_ok": topology_ok,
        "metrics": {
            "entry_count": entry_count,
            "average_stability": analytics.get("average_stability_score", 0.0),
        },
    }


def summarize_slice_topology(
    baseline_log: str | Path,
    rfl_log: str | Path,
    out_path: str | Path,
    slice_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a topology summary for a slice (JSON export).
    
    Provides curriculum designers with a quick diagnostic view of
    DAG topology characteristics for a baseline/RFL pair.
    
    This is the Curriculum ↔ Topology Bridge — summarizes and flags,
    does NOT change curriculum or DAG construction.
    
    PHASE II ONLY - Diagnostic summary, no uplift claims.
    
    Args:
        baseline_log: Path to baseline experiment JSONL log.
        rfl_log: Path to RFL experiment JSONL log.
        out_path: Output path for summary JSON.
        slice_name: Optional slice name (auto-detected if not provided).
    
    Returns:
        Summary dictionary (also written to out_path).
    
    Raises:
        FileNotFoundError: If log files don't exist.
        ValueError: If logs are empty or unreadable.
    """
    baseline_log = Path(baseline_log)
    rfl_log = Path(rfl_log)
    out_path = Path(out_path)
    
    # Validate inputs
    if not baseline_log.exists():
        raise FileNotFoundError(f"Baseline log not found: {baseline_log}")
    if not rfl_log.exists():
        raise FileNotFoundError(f"RFL log not found: {rfl_log}")
    
    # Load data
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)
    
    if not baseline_records:
        raise ValueError(f"Baseline log is empty: {baseline_log}")
    if not rfl_records:
        raise ValueError(f"RFL log is empty: {rfl_log}")
    
    # Auto-detect slice name
    if slice_name is None:
        slice_name = baseline_records[0].get("slice_name")
        if slice_name is None:
            slice_name = rfl_records[0].get("slice_name", "unknown")
    
    # Extract derivations and build analyzers
    baseline_derivations = extract_derivations_from_log(baseline_records)
    rfl_derivations = extract_derivations_from_log(rfl_records)
    
    baseline_analyzer = DAGTopologyAnalyzer(baseline_derivations)
    rfl_analyzer = DAGTopologyAnalyzer(rfl_derivations)
    
    # Get footprints
    baseline_footprint = baseline_analyzer.get_dag_footprint()
    rfl_footprint = rfl_analyzer.get_dag_footprint()
    
    # Compute mean branching
    baseline_mean_branching = (
        baseline_footprint["edge_count"] / 
        max(baseline_footprint["node_count"], 1)
    )
    rfl_mean_branching = (
        rfl_footprint["edge_count"] / 
        max(rfl_footprint["node_count"], 1)
    )
    
    # Compute deltas
    depth_delta = rfl_footprint["max_depth"] - baseline_footprint["max_depth"]
    node_delta = rfl_footprint["node_count"] - baseline_footprint["node_count"]
    edge_delta = rfl_footprint["edge_count"] - baseline_footprint["edge_count"]
    branching_delta = rfl_mean_branching - baseline_mean_branching
    
    # Compute warning flags
    warning_flags = compute_warning_flags(
        baseline_footprint, rfl_footprint,
        depth_delta, branching_delta,
        slice_name,
    )
    
    # Build summary (deterministic structure)
    summary: Dict[str, Any] = {
        "phase": "II",
        "type": "slice_topology_summary",
        "slice_name": slice_name,
        "baseline_footprint": {
            "node_count": baseline_footprint["node_count"],
            "edge_count": baseline_footprint["edge_count"],
            "max_depth": baseline_footprint["max_depth"],
            "mean_depth": baseline_footprint["mean_depth"],
            "mean_branching": baseline_mean_branching,
        },
        "rfl_footprint": {
            "node_count": rfl_footprint["node_count"],
            "edge_count": rfl_footprint["edge_count"],
            "max_depth": rfl_footprint["max_depth"],
            "mean_depth": rfl_footprint["mean_depth"],
            "mean_branching": rfl_mean_branching,
        },
        "deltas": {
            "depth_delta": depth_delta,
            "node_delta": node_delta,
            "edge_delta": edge_delta,
            "branching_delta": branching_delta,
        },
        "warnings": warning_flags,
        "source_logs": {
            "baseline": str(baseline_log),
            "rfl": str(rfl_log),
        },
    }
    
    # Add stability score (advisory only)
    stability = compute_structural_stability_score(baseline_records, rfl_records)
    summary["stability"] = {
        "score": stability["score"],
        "components": stability["components"],
    }
    
    # Add depth timeseries
    timeseries = compute_depth_evolution_contract(
        baseline_records, rfl_records, slice_name
    )
    summary["depth_timeseries"] = timeseries
    
    # Write output (deterministic JSON formatting)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    return summary


def print_topology_summary(summary: Dict[str, Any]) -> None:
    """
    Print a human-readable topology summary to stdout.
    
    Uses purely geometric language — no value judgments (good/bad).
    Warning flags are ADVISORY signals for curriculum designers.
    
    Args:
        summary: Summary dict from summarize_slice_topology().
    """
    print(f"\n{'='*60}")
    print(f"STRUCTURAL HEALTH REPORT: {summary['slice_name']}")
    print(f"{'='*60}")
    
    bf = summary["baseline_footprint"]
    rf = summary["rfl_footprint"]
    deltas = summary["deltas"]
    
    print(f"\n{'Metric':<20} {'Baseline':>12} {'RFL':>12} {'Delta':>12}")
    print(f"{'-'*56}")
    print(f"{'Node Count':<20} {bf['node_count']:>12} {rf['node_count']:>12} {deltas['node_delta']:>+12}")
    print(f"{'Edge Count':<20} {bf['edge_count']:>12} {rf['edge_count']:>12} {deltas['edge_delta']:>+12}")
    print(f"{'Max Depth':<20} {bf['max_depth']:>12} {rf['max_depth']:>12} {deltas['depth_delta']:>+12.2f}")
    print(f"{'Mean Depth':<20} {bf['mean_depth']:>12.2f} {rf['mean_depth']:>12.2f} {'-':>12}")
    print(f"{'Mean Branching':<20} {bf['mean_branching']:>12.3f} {rf['mean_branching']:>12.3f} {deltas['branching_delta']:>+12.3f}")
    
    # Show theoretical max if available
    warnings = summary.get("warnings", {})
    theoretical_max = warnings.get("theoretical_max_depth")
    if theoretical_max is not None:
        print(f"\n{'Curriculum Depth Max':<20} {theoretical_max:>12}")
    
    # Show stability score (advisory metric)
    stability = summary.get("stability", {})
    if stability:
        print(f"\nSTABILITY (advisory metric):")
        print(f"{'-'*56}")
        score = stability.get("score", 0)
        print(f"  Overall Score: {score:.4f}")
        
        components = stability.get("components", {})
        if components:
            bc = components.get("baseline", {})
            rc = components.get("rfl", {})
            print(f"  Baseline — depth: {bc.get('depth_stability', 0):.3f}, branching: {bc.get('branching_stability', 0):.3f}")
            print(f"  RFL      — depth: {rc.get('depth_stability', 0):.3f}, branching: {rc.get('branching_stability', 0):.3f}")
    
    # Show depth timeseries length if available
    timeseries = summary.get("depth_timeseries", {})
    if timeseries:
        ts_len = timeseries.get("length", 0)
        print(f"\nDEPTH TIMESERIES: {ts_len} cycles")
    
    # Display advisory flags (purely geometric, no value judgments)
    if any([
        warnings.get("depth_saturation_warning"),
        warnings.get("branching_collapse_warning"),
        warnings.get("topology_change_warning"),
    ]):
        print(f"\nADVISORY FLAGS (structural observations):")
        print(f"{'-'*56}")
        if warnings.get("depth_saturation_warning"):
            print(f"  [!] DEPTH_SATURATION — depth near curriculum limit")
        if warnings.get("branching_collapse_warning"):
            print(f"  [!] BRANCHING_COLLAPSE — low edge density observed")
        if warnings.get("topology_change_warning"):
            print(f"  [!] TOPOLOGY_CHANGE — structural delta exceeds threshold")
        
        for explanation in warnings.get("explanations", []):
            print(f"      {explanation}")
    else:
        print(f"\n[i] No advisory flags triggered")
    
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """
    Command-line entry point.
    
    EXIT CODE CONTRACT:
      - Always returns 0 on successful execution
      - Warning flags do NOT alter exit codes
      - Warnings are ADVISORY only — signals for curriculum designers
      - Returns non-zero only on actual errors (missing files, invalid data)
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate DAG topology visualizations with drift analysis (PHASE II)."
    )
    parser.add_argument(
        "--baseline", type=str, required=True,
        help="Path to baseline experiment JSONL log."
    )
    parser.add_argument(
        "--rfl", type=str, required=True,
        help="Path to RFL experiment JSONL log."
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory for visualizations (default: {OUTPUT_DIR})."
    )
    parser.add_argument(
        "--slice-name", "--slice", type=str, default=None,
        dest="slice_name",
        help="Slice name for expectation mapping (auto-detected if not provided)."
    )
    parser.add_argument(
        "--with-drift", action="store_true", default=True,
        help="Include behavior drift analysis (default: True)."
    )
    parser.add_argument(
        "--summary-out", type=str, default=None,
        help="Output path for summary JSON only (skips plot generation)."
    )
    parser.add_argument(
        "--ledger-out", type=str, default=None,
        help="Output path for ledger-grade topology entry JSON."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress stdout output (only write files)."
    )
    
    args = parser.parse_args()
    
    # Validate log files exist
    baseline_path = Path(args.baseline)
    rfl_path = Path(args.rfl)
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline log not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not rfl_path.exists():
        print(f"ERROR: RFL log not found: {args.rfl}", file=sys.stderr)
        sys.exit(1)
    
    # Summary-only mode: just produce JSON, no plots
    # WARNING FLAGS DO NOT AFFECT EXIT CODE — they are advisory only
    if args.summary_out:
        if not args.quiet:
            print(f"Generating topology summary...")
            print(f"  Baseline: {args.baseline}")
            print(f"  RFL:      {args.rfl}")
            print(f"  Output:   {args.summary_out}")
        
        try:
            summary = summarize_slice_topology(
                baseline_log=args.baseline,
                rfl_log=args.rfl,
                out_path=args.summary_out,
                slice_name=args.slice_name,
            )
            
            if not args.quiet:
                print_topology_summary(summary)
                print(f"Summary written to: {args.summary_out}")
            
            # Explicit exit 0 — warnings do NOT alter exit code
            sys.exit(0)
        
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        
        return
    
    # Ledger entry mode: produce ledger-grade topology snapshot
    # WARNING FLAGS DO NOT AFFECT EXIT CODE — they are advisory only
    if args.ledger_out:
        if not args.quiet:
            print(f"Generating ledger topology entry...")
            print(f"  Baseline: {args.baseline}")
            print(f"  RFL:      {args.rfl}")
            print(f"  Output:   {args.ledger_out}")
        
        try:
            entry = to_ledger_topology_entry(
                baseline_log=args.baseline,
                rfl_log=args.rfl,
                slice_name=args.slice_name,
                include_timeseries=True,
            )
            
            write_ledger_topology_entry(entry, args.ledger_out)
            
            if not args.quiet:
                print(f"\nLedger Entry Created:")
                print(f"  Topology Hash: {entry['topology_hash'][:16]}...")
                print(f"  Slice:         {entry['slice_name']}")
                print(f"  Stability:     {entry['stability']['score']:.4f}")
                print(f"  Warnings:      {sum(entry['warning_flags'].values())} flags")
                print(f"\nWritten to: {args.ledger_out}")
            
            # Explicit exit 0 — warnings do NOT alter exit code
            sys.exit(0)
        
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        
        return
    
    # Full visualization mode
    if not args.quiet:
        print("Generating DAG topology visualizations...")
        print(f"  Baseline: {args.baseline}")
        print(f"  RFL:      {args.rfl}")
        print(f"  Output:   {args.output_dir}")
        if args.slice_name:
            print(f"  Slice:    {args.slice_name}")
    
    if args.with_drift:
        results = generate_all_topology_with_drift(
            baseline_log=args.baseline,
            rfl_log=args.rfl,
            output_dir=args.output_dir,
            slice_name=args.slice_name,
        )
    else:
        results = generate_all_topology_visualizations(
            baseline_log=args.baseline,
            rfl_log=args.rfl,
            output_dir=args.output_dir,
        )
    
    if not args.quiet:
        print(f"\nGenerated {len(results)} artifacts:")
        for name, path in results.items():
            print(f"  - {name}: {path}")
    
    # Explicit exit 0 — warnings do NOT alter exit code
    sys.exit(0)


if __name__ == "__main__":
    main()

