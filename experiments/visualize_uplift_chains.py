#!/usr/bin/env python
"""
PHASE II — Chain-Depth Cartographer & Slice Visualizer

Visualizes derivation structure—chain depth and goal reachability—for uplift slices.

This module is an EXPLORATORY TOOL for Phase II analysis. It does NOT make
uplift claims; it exposes structural behavior of derivation chains across
baseline vs RFL runs.

Primary Use Cases:
  - Inspect chain-depth distributions per cycle
  - Compare longest-chain evolution (baseline vs RFL)
  - Track goal coverage trajectories (A/B/C goals over cycles)

Relevant Slices:
  - slice_uplift_tree: Tree-structured derivation experiments
  - slice_uplift_dependency: Dependency-chain experiments
  - slice_uplift_proto: Prototype uplift slice
  - slice_medium: Wide slice for RFL uplift experiments

SAFEGUARDS:
  - No modification of core metrics or governance.
  - No success definitions changed.
  - No Phase I data touched.
  - These plots are NOT formal evidence—they are inspection tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use non-interactive backend for headless environments (CI/servers)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Import ChainAnalyzer for depth computation
from experiments.derivation_chain_analysis import ChainAnalyzer

# Import shared plotting utilities
from experiments.plotting import setup_style, ensure_output_dir

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

OUTPUT_DIR = "artifacts/phase_ii/uplift_visualizations"

# PRNG Contract (Agent A2 — runtime-ops-2):
#   This visualization module does not use randomness.
#   If randomness is needed in the future, use DeterministicPRNG from rfl.prng.
#   The legacy DETERMINISTIC_SEED is preserved for documentation only.
DETERMINISTIC_SEED = 42  # Legacy - do not use np.random.seed()

# Color scheme for baseline vs RFL comparison
COLOR_BASELINE = "#666666"  # Gray
COLOR_RFL = "#000000"       # Black
COLOR_ACCENT = "#4a90d9"    # Blue accent for highlights

# -----------------------------------------------------------------------------
# Data Loading Utilities
# -----------------------------------------------------------------------------


def load_experiment_log(log_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load experiment log from JSONL file.

    Args:
        log_path: Path to the JSONL experiment log file.

    Returns:
        List of cycle records.
    """
    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_chain_depths(records: List[Dict[str, Any]]) -> List[int]:
    """
    Extract chain depths from experiment records.

    If records contain full derivation data with premises, computes actual depth.
    Otherwise, uses 'depth_max' from derivation metadata or defaults to cycle index proxy.

    Args:
        records: List of experiment cycle records.

    Returns:
        List of chain depths, one per cycle.
    """
    depths: List[int] = []

    for rec in records:
        derivation = rec.get("derivation", {})

        # Try to get explicit depth if available
        if "depth" in derivation:
            depths.append(int(derivation["depth"]))
        elif "depth_max" in derivation:
            depths.append(int(derivation["depth_max"]))
        elif "chain_length" in derivation:
            depths.append(int(derivation["chain_length"]))
        else:
            # Use verified count as a proxy for derivation activity
            verified = derivation.get("verified", 0)
            # Minimum depth is 1 if any derivation occurred
            depths.append(max(1, verified))

    return depths


def extract_goals_reached(records: List[Dict[str, Any]]) -> Dict[str, List[bool]]:
    """
    Extract goal coverage per cycle.

    Goals are tracked as A/B/C if available, or derived from success status.

    Args:
        records: List of experiment cycle records.

    Returns:
        Dictionary mapping goal names to list of boolean achieved states per cycle.
    """
    goals: Dict[str, List[bool]] = {
        "goal_A": [],
        "goal_B": [],
        "goal_C": [],
    }

    for rec in records:
        # Try explicit goal fields
        if "goals" in rec:
            goal_data = rec["goals"]
            goals["goal_A"].append(bool(goal_data.get("A", False)))
            goals["goal_B"].append(bool(goal_data.get("B", False)))
            goals["goal_C"].append(bool(goal_data.get("C", False)))
        else:
            # Derive from success/status fields
            success = rec.get("success", False)
            status = rec.get("status", "")
            proof_found = rec.get("proof_found", False)

            # Map success indicators to goals
            # Goal A: Any proof attempt made
            # Goal B: Proof verified
            # Goal C: Full success (no abstention)
            derivation = rec.get("derivation", {})
            candidates = derivation.get("candidates", 0)
            verified = derivation.get("verified", 0)
            abstained = derivation.get("abstained", 0)

            goal_a = candidates > 0
            goal_b = verified > 0 or proof_found
            goal_c = success or status == "verified"

            goals["goal_A"].append(goal_a)
            goals["goal_B"].append(goal_b)
            goals["goal_C"].append(goal_c)

    return goals


def compute_cumulative_coverage(achieved: List[bool]) -> List[float]:
    """
    Compute cumulative coverage rate over cycles.

    Args:
        achieved: Boolean list of goal achievement per cycle.

    Returns:
        List of cumulative coverage rates (0.0 to 1.0).
    """
    if not achieved:
        return []

    cumulative: List[float] = []
    total_achieved = 0

    for i, hit in enumerate(achieved):
        if hit:
            total_achieved += 1
        cumulative.append(total_achieved / (i + 1))

    return cumulative


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------


def plot_chain_depth_histogram(
    log_path: str | Path,
    out_path: str | Path,
    title: str = "Chain Depth Distribution",
    mode: str = "auto",
    slice_name: Optional[str] = None,
    bins: int = 20,
) -> Path:
    """
    Generate a chain-depth histogram showing distribution of chain lengths per cycle.

    PHASE II ONLY - Exploratory visualization, not formal evidence.

    Args:
        log_path: Path to experiment JSONL log.
        out_path: Output path for the histogram image.
        title: Plot title.
        mode: 'baseline', 'rfl', or 'auto' (detect from log).
        slice_name: Optional slice name to include in title.
        bins: Number of histogram bins.

    Returns:
        Path to saved figure.
    """
    setup_style()
    # PRNG Contract (Agent A2): No global np.random.seed() - use DeterministicPRNG if needed

    # Load data
    records = load_experiment_log(log_path)
    depths = extract_chain_depths(records)

    if not depths:
        raise ValueError(f"No chain depth data found in {log_path}")

    # Detect mode if auto
    if mode == "auto":
        mode = records[0].get("mode", "unknown") if records else "unknown"

    # Detect slice name if not provided
    if slice_name is None and records:
        slice_name = records[0].get("slice_name", None)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose color based on mode
    color = COLOR_RFL if mode == "rfl" else COLOR_BASELINE

    # Plot histogram
    ax.hist(depths, bins=bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Labels and title
    full_title = title
    if slice_name:
        full_title = f"{title}\n({slice_name}, mode={mode})"
    else:
        full_title = f"{title} (mode={mode})"

    ax.set_xlabel("Chain Depth")
    ax.set_ylabel("Frequency (cycles)")
    ax.set_title(full_title)

    # Add statistics annotation
    mean_depth = np.mean(depths)
    max_depth = np.max(depths)
    median_depth = np.median(depths)

    stats_text = f"μ={mean_depth:.2f}, max={max_depth}, median={median_depth:.1f}"
    ax.annotate(
        stats_text,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


def plot_longest_chain_comparison(
    baseline_log: str | Path,
    rfl_log: str | Path,
    out_path: str | Path,
    title: str = "Longest Chain Comparison: Baseline vs RFL",
    window: int = 10,
) -> Path:
    """
    Generate a comparison plot of longest chain evolution across cycles.

    Shows how the maximum chain depth evolves over time for baseline vs RFL runs.

    PHASE II ONLY - Exploratory visualization, not formal evidence.

    Args:
        baseline_log: Path to baseline experiment JSONL log.
        rfl_log: Path to RFL experiment JSONL log.
        out_path: Output path for the comparison image.
        title: Plot title.
        window: Rolling window size for smoothing.

    Returns:
        Path to saved figure.
    """
    setup_style()

    # Load data
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)

    baseline_depths = extract_chain_depths(baseline_records)
    rfl_depths = extract_chain_depths(rfl_records)

    if not baseline_depths or not rfl_depths:
        raise ValueError("Both baseline and RFL logs must contain chain depth data")

    # Compute running maximum
    def running_max(depths: List[int]) -> List[int]:
        result = []
        current_max = 0
        for d in depths:
            current_max = max(current_max, d)
            result.append(current_max)
        return result

    baseline_max = running_max(baseline_depths)
    rfl_max = running_max(rfl_depths)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot running maximum
    cycles_baseline = np.arange(len(baseline_max))
    cycles_rfl = np.arange(len(rfl_max))

    ax.plot(
        cycles_baseline,
        baseline_max,
        label="Baseline (RFL off)",
        color=COLOR_BASELINE,
        linewidth=2.0,
        linestyle="--",
    )

    ax.plot(
        cycles_rfl,
        rfl_max,
        label="RFL Enabled",
        color=COLOR_RFL,
        linewidth=2.0,
        linestyle="-",
    )

    # Add rolling mean overlay for smoothed view
    if len(baseline_depths) >= window:
        baseline_smooth = np.convolve(
            baseline_depths, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            np.arange(window - 1, len(baseline_depths)),
            baseline_smooth,
            color=COLOR_BASELINE,
            alpha=0.4,
            linewidth=1.0,
            label=f"Baseline ({window}-cycle avg)",
        )

    if len(rfl_depths) >= window:
        rfl_smooth = np.convolve(rfl_depths, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(rfl_depths)),
            rfl_smooth,
            color=COLOR_RFL,
            alpha=0.4,
            linewidth=1.0,
            label=f"RFL ({window}-cycle avg)",
        )

    # Detect slice name for title
    slice_name = None
    if baseline_records:
        slice_name = baseline_records[0].get("slice_name")

    full_title = title
    if slice_name:
        full_title = f"{title}\n({slice_name})"

    ax.set_xlabel("Cycle Index")
    ax.set_ylabel("Chain Depth")
    ax.set_title(full_title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


def plot_goal_coverage_trajectories(
    baseline_log: str | Path,
    rfl_log: str | Path,
    out_path: str | Path,
    title: str = "Goal Coverage Trajectories",
) -> Path:
    """
    Generate goal coverage trajectory plots showing per-goal A/B/C progress over cycles.

    Compares cumulative goal achievement between baseline and RFL runs.

    PHASE II ONLY - Exploratory visualization, not formal evidence.

    Args:
        baseline_log: Path to baseline experiment JSONL log.
        rfl_log: Path to RFL experiment JSONL log.
        out_path: Output path for the trajectory image.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    setup_style()

    # Load data
    baseline_records = load_experiment_log(baseline_log)
    rfl_records = load_experiment_log(rfl_log)

    baseline_goals = extract_goals_reached(baseline_records)
    rfl_goals = extract_goals_reached(rfl_records)

    # Create figure with subplots for each goal
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    goal_labels = ["Goal A\n(Attempt)", "Goal B\n(Verified)", "Goal C\n(Success)"]
    goal_keys = ["goal_A", "goal_B", "goal_C"]

    for ax, label, key in zip(axes, goal_labels, goal_keys):
        # Compute cumulative coverage
        baseline_coverage = compute_cumulative_coverage(baseline_goals[key])
        rfl_coverage = compute_cumulative_coverage(rfl_goals[key])

        # Plot trajectories
        if baseline_coverage:
            ax.plot(
                np.arange(len(baseline_coverage)),
                baseline_coverage,
                label="Baseline",
                color=COLOR_BASELINE,
                linewidth=2.0,
                linestyle="--",
            )

        if rfl_coverage:
            ax.plot(
                np.arange(len(rfl_coverage)),
                rfl_coverage,
                label="RFL",
                color=COLOR_RFL,
                linewidth=2.0,
                linestyle="-",
            )

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Cumulative Rate")
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)

    # Detect slice name for title
    slice_name = None
    if baseline_records:
        slice_name = baseline_records[0].get("slice_name")

    full_title = title
    if slice_name:
        full_title = f"{title} ({slice_name})"

    fig.suptitle(full_title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


# -----------------------------------------------------------------------------
# Batch Visualization
# -----------------------------------------------------------------------------


def generate_all_visualizations(
    baseline_log: str | Path,
    rfl_log: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    slice_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generate all visualizations for a baseline/RFL log pair.

    Creates:
      - chain_depth_histogram_baseline.png
      - chain_depth_histogram_rfl.png
      - longest_chain_comparison.png
      - goal_coverage_trajectories.png

    PHASE II ONLY - Exploratory visualization, not formal evidence.

    Args:
        baseline_log: Path to baseline experiment JSONL log.
        rfl_log: Path to RFL experiment JSONL log.
        output_dir: Directory to save visualizations.
        slice_name: Optional slice name for labeling.

    Returns:
        Dictionary mapping visualization names to output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Path] = {}

    # Generate individual histograms
    results["histogram_baseline"] = plot_chain_depth_histogram(
        baseline_log,
        output_dir / "chain_depth_histogram_baseline.png",
        title="Chain Depth Distribution",
        mode="baseline",
        slice_name=slice_name,
    )

    results["histogram_rfl"] = plot_chain_depth_histogram(
        rfl_log,
        output_dir / "chain_depth_histogram_rfl.png",
        title="Chain Depth Distribution",
        mode="rfl",
        slice_name=slice_name,
    )

    # Generate comparison plots
    results["longest_chain"] = plot_longest_chain_comparison(
        baseline_log,
        rfl_log,
        output_dir / "longest_chain_comparison.png",
    )

    results["goal_coverage"] = plot_goal_coverage_trajectories(
        baseline_log,
        rfl_log,
        output_dir / "goal_coverage_trajectories.png",
    )

    # Generate summary JSON
    summary = {
        "phase": "II",
        "type": "chain_depth_visualizations",
        "slice_name": slice_name,
        "outputs": {k: str(v) for k, v in results.items()},
        "baseline_log": str(baseline_log),
        "rfl_log": str(rfl_log),
    }

    summary_path = output_dir / "visualization_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results["manifest"] = summary_path

    return results


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chain-depth visualizations for uplift experiments (PHASE II)."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline experiment JSONL log.",
    )
    parser.add_argument(
        "--rfl",
        type=str,
        required=True,
        help="Path to RFL experiment JSONL log.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for visualizations (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--slice-name",
        type=str,
        default=None,
        help="Optional slice name for labeling plots.",
    )

    args = parser.parse_args()

    print(f"Generating visualizations...")
    print(f"  Baseline: {args.baseline}")
    print(f"  RFL:      {args.rfl}")
    print(f"  Output:   {args.output_dir}")

    results = generate_all_visualizations(
        baseline_log=args.baseline,
        rfl_log=args.rfl,
        output_dir=args.output_dir,
        slice_name=args.slice_name,
    )

    print(f"\nGenerated {len(results)} artifacts:")
    for name, path in results.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()

