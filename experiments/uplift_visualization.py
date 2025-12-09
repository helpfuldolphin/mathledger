#!/usr/bin/env python
"""
PHASE II — Uplift Visualization Pipeline

Descriptive only — not admissible as uplift evidence.

This module provides deterministic visualization functions for rendering
uplift experiment logs into governance-grade curves. All outputs are
purely descriptive and do not constitute statistical inference or
uplift claims.

Design Principles:
- Deterministic: All plots are reproducible with identical checksums.
- No randomness: Uses fixed seeds, sorted data, and deterministic rendering.
- PHASE II labeling: All outputs are marked as descriptive only.
- Fixed styles: Consistent matplotlib rcParams for publication quality.

Usage:
    uv run python experiments/uplift_visualization.py --baseline path/to/baseline.jsonl --rfl path/to/rfl.jsonl --out-dir artifacts/uplift_figs

Author: metrics-engineer-4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Force non-interactive backend for determinism
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

PHASE_II_LABEL = "PHASE II — Descriptive only, not admissible as uplift evidence"
PHASE_II_WATERMARK = "PHASE II — NOT EVIDENCE"

# PRNG Contract (Agent A2 — runtime-ops-2):
#   This visualization module does not use randomness.
#   If randomness is needed in the future, use DeterministicPRNG from rfl.prng.
#   The legacy DETERMINISTIC_SEED is preserved for documentation only.
DETERMINISTIC_SEED = 42  # Legacy - do not use np.random.seed()

# Default output directory
DEFAULT_OUT_DIR = "artifacts/uplift_figs"

# Fixed matplotlib style configuration for determinism
_STYLE_CONFIG: Dict[str, Any] = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'savefig.format': 'png',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    # Determinism: disable date-based auto-formatting
    'axes.formatter.use_mathtext': True,
}

# Color palette (fixed, monochrome-friendly)
COLOR_BASELINE = '#666666'
COLOR_RFL = '#1a1a1a'
COLOR_DELTA = '#333333'
COLOR_GRID = '#cccccc'


# -----------------------------------------------------------------------------
# Style Setup
# -----------------------------------------------------------------------------

def setup_deterministic_style() -> None:
    """
    Apply fixed matplotlib style settings for deterministic, reproducible figures.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function sets rcParams to ensure identical outputs across runs,
    regardless of system defaults or user configuration.
    """
    # Reset to defaults first for clean slate
    plt.rcdefaults()
    
    # Apply fixed configuration
    plt.rcParams.update(_STYLE_CONFIG)

    # PRNG Contract (Agent A2): No global np.random.seed() - use DeterministicPRNG if needed
    # Legacy np.random.seed(DETERMINISTIC_SEED) removed to prevent global state pollution


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL file.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        filepath: Path to the JSONL file.
    
    Returns:
        List of dictionaries, one per line.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_cycle_success(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract cycle index and success status from experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    
    Handles multiple data formats:
    - {'cycle': N, 'success': bool}
    - {'cycle': N, 'derivation': {'verified': int}}
    - {'cycle': N, 'gates_passed': bool}
    
    Args:
        records: List of experiment records.
    
    Returns:
        DataFrame with columns: ['cycle', 'success', 'verified_count', 'abstained_count']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
        # Determine success
        success = False
        verified_count = 0
        abstained_count = 0
        
        if 'success' in rec:
            success = bool(rec['success'])
        elif 'derivation' in rec and isinstance(rec['derivation'], dict):
            verified_count = rec['derivation'].get('verified', 0)
            abstained_count = rec['derivation'].get('abstained', 0)
            success = verified_count > 0
        elif 'gates_passed' in rec:
            success = bool(rec['gates_passed'])
        
        rows.append({
            'cycle': int(cycle),
            'success': success,
            'verified_count': int(verified_count),
            'abstained_count': int(abstained_count),
        })
    
    df = pd.DataFrame(rows)
    # Sort by cycle for determinism
    df = df.sort_values('cycle').reset_index(drop=True)
    return df


def extract_metric_values(
    records: List[Dict[str, Any]],
    metric_path: str = 'derivation.verified'
) -> pd.DataFrame:
    """
    Extract a metric value trajectory from experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        records: List of experiment records.
        metric_path: Dot-separated path to the metric (e.g., 'derivation.verified').
    
    Returns:
        DataFrame with columns: ['cycle', 'value']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
        # Navigate to metric value
        value = rec
        for key in metric_path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = 0
                break
        
        if not isinstance(value, (int, float)):
            value = 0
        
        rows.append({'cycle': int(cycle), 'value': float(value)})
    
    df = pd.DataFrame(rows)
    df = df.sort_values('cycle').reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Core Plotting Functions
# -----------------------------------------------------------------------------

def plot_success_rate_curve(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    window: int = 20,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot success rate curves for baseline vs RFL experiments.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function generates a line plot comparing the rolling success rate
    between baseline and RFL experiment runs. The output is purely descriptive
    and does not constitute evidence of uplift.
    
    Args:
        baseline_path: Path to baseline JSONL file.
        rfl_path: Path to RFL JSONL file.
        out_path: Output path for the PNG file.
        window: Rolling window size for smoothing.
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with plot information and checksum.
    """
    setup_deterministic_style()
    
    # Load data
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_cycle_success(baseline_records)
    df_rfl = extract_cycle_success(rfl_records)
    
    # Compute rolling success rate
    df_baseline['rolling_success'] = (
        df_baseline['success'].astype(float)
        .rolling(window=window, min_periods=1)
        .mean()
    )
    df_rfl['rolling_success'] = (
        df_rfl['success'].astype(float)
        .rolling(window=window, min_periods=1)
        .mean()
    )
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot baseline
    ax.plot(
        df_baseline['cycle'],
        df_baseline['rolling_success'],
        color=COLOR_BASELINE,
        linestyle='--',
        linewidth=2,
        label=f'Baseline (n={len(df_baseline)})'
    )
    
    # Plot RFL
    ax.plot(
        df_rfl['cycle'],
        df_rfl['rolling_success'],
        color=COLOR_RFL,
        linestyle='-',
        linewidth=2,
        label=f'RFL (n={len(df_rfl)})'
    )
    
    # Labels and title
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel(f'Success Rate (rolling window={window})')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Success Rate Curve — Baseline vs RFL')
    
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add PHASE II watermark
    ax.text(
        0.99, 0.01, PHASE_II_WATERMARK,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.5,
        ha='right',
        va='bottom',
        style='italic'
    )
    
    # Save figure
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Compute checksum
    checksum = _compute_file_checksum(out_path)
    
    metadata = {
        'plot_type': 'success_rate_curve',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'window_size': window,
        'baseline_cycles': len(df_baseline),
        'rfl_cycles': len(df_rfl),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    
    return metadata


def plot_metric_value_trajectory(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    metric_path: str = 'derivation.verified',
    window: int = 10,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot metric value trajectories for baseline vs RFL experiments.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function generates a line plot showing how a specific metric evolves
    over cycles for both baseline and RFL runs. The output is purely descriptive.
    
    Args:
        baseline_path: Path to baseline JSONL file.
        rfl_path: Path to RFL JSONL file.
        out_path: Output path for the PNG file.
        metric_path: Dot-separated path to the metric in each record.
        window: Rolling window size for smoothing.
        title: Optional custom title.
        ylabel: Optional custom y-axis label.
    
    Returns:
        Metadata dictionary with plot information and checksum.
    """
    setup_deterministic_style()
    
    # Load data
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_metric_values(baseline_records, metric_path)
    df_rfl = extract_metric_values(rfl_records, metric_path)
    
    # Compute rolling mean
    df_baseline['rolling_value'] = (
        df_baseline['value']
        .rolling(window=window, min_periods=1)
        .mean()
    )
    df_rfl['rolling_value'] = (
        df_rfl['value']
        .rolling(window=window, min_periods=1)
        .mean()
    )
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot raw values (light)
    ax.scatter(
        df_baseline['cycle'],
        df_baseline['value'],
        color=COLOR_BASELINE,
        alpha=0.2,
        s=10,
        label=None
    )
    ax.scatter(
        df_rfl['cycle'],
        df_rfl['value'],
        color=COLOR_RFL,
        alpha=0.2,
        s=10,
        label=None
    )
    
    # Plot rolling mean (bold)
    ax.plot(
        df_baseline['cycle'],
        df_baseline['rolling_value'],
        color=COLOR_BASELINE,
        linestyle='--',
        linewidth=2,
        label=f'Baseline (rolling mean)'
    )
    ax.plot(
        df_rfl['cycle'],
        df_rfl['rolling_value'],
        color=COLOR_RFL,
        linestyle='-',
        linewidth=2,
        label=f'RFL (rolling mean)'
    )
    
    # Labels and title
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel(ylabel or f'{metric_path} (window={window})')
    ax.set_title(title or f'Metric Trajectory: {metric_path}')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add PHASE II watermark
    ax.text(
        0.99, 0.01, PHASE_II_WATERMARK,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.5,
        ha='right',
        va='bottom',
        style='italic'
    )
    
    # Save figure
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Compute checksum
    checksum = _compute_file_checksum(out_path)
    
    metadata = {
        'plot_type': 'metric_value_trajectory',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'metric_path': metric_path,
        'window_size': window,
        'baseline_cycles': len(df_baseline),
        'rfl_cycles': len(df_rfl),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    
    return metadata


def plot_delta_p_point_estimates(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    bin_size: int = 50,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot delta-p (success rate difference) point estimates across cycle bins.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function computes the difference in success rate between RFL and baseline
    for each bin of cycles, plotting them as point estimates with error bars
    representing the raw standard error (descriptive, not inferential).
    
    Args:
        baseline_path: Path to baseline JSONL file.
        rfl_path: Path to RFL JSONL file.
        out_path: Output path for the PNG file.
        bin_size: Number of cycles per bin.
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with plot information and checksum.
    """
    setup_deterministic_style()
    
    # Load data
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_cycle_success(baseline_records)
    df_rfl = extract_cycle_success(rfl_records)
    
    # Determine cycle range
    max_cycle = max(
        df_baseline['cycle'].max() if len(df_baseline) > 0 else 0,
        df_rfl['cycle'].max() if len(df_rfl) > 0 else 0
    )
    
    # Compute binned statistics
    bin_centers = []
    delta_ps = []
    delta_errs = []
    
    for bin_start in range(0, int(max_cycle) + 1, bin_size):
        bin_end = bin_start + bin_size
        bin_center = bin_start + bin_size / 2
        
        # Filter to bin
        mask_b = (df_baseline['cycle'] >= bin_start) & (df_baseline['cycle'] < bin_end)
        mask_r = (df_rfl['cycle'] >= bin_start) & (df_rfl['cycle'] < bin_end)
        
        n_b = mask_b.sum()
        n_r = mask_r.sum()
        
        if n_b == 0 or n_r == 0:
            continue
        
        # Compute success rates
        p_b = df_baseline.loc[mask_b, 'success'].astype(float).mean()
        p_r = df_rfl.loc[mask_r, 'success'].astype(float).mean()
        
        # Delta p
        delta_p = p_r - p_b
        
        # Standard error (descriptive, not inferential)
        # SE = sqrt(p*(1-p)/n) for each, combined via sqrt(se_b^2 + se_r^2)
        se_b = np.sqrt(p_b * (1 - p_b) / n_b) if n_b > 1 else 0
        se_r = np.sqrt(p_r * (1 - p_r) / n_r) if n_r > 1 else 0
        se_delta = np.sqrt(se_b**2 + se_r**2)
        
        bin_centers.append(bin_center)
        delta_ps.append(delta_p)
        delta_errs.append(se_delta)
    
    # Create figure
    fig, ax = plt.subplots()
    
    if bin_centers:
        ax.errorbar(
            bin_centers,
            delta_ps,
            yerr=delta_errs,
            fmt='o',
            color=COLOR_DELTA,
            capsize=4,
            capthick=1.5,
            markersize=8,
            label='Δp (RFL - Baseline)'
        )
        
        # Add zero reference line
        ax.axhline(y=0, color=COLOR_GRID, linestyle='-', linewidth=1, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel(f'Cycle Bin Center (bin size={bin_size})')
    ax.set_ylabel('Δp = p(RFL) − p(Baseline)')
    ax.set_title(title or 'Delta-p Point Estimates — RFL vs Baseline')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add PHASE II watermark
    ax.text(
        0.99, 0.01, PHASE_II_WATERMARK,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.5,
        ha='right',
        va='bottom',
        style='italic'
    )
    
    # Save figure
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Compute checksum
    checksum = _compute_file_checksum(out_path)
    
    metadata = {
        'plot_type': 'delta_p_point_estimates',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'bin_size': bin_size,
        'num_bins': len(bin_centers),
        'delta_p_values': delta_ps,
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    
    return metadata


def plot_slice_comparison_grid(
    slice_data: List[Dict[str, str]],
    out_path: str,
    window: int = 20,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot a grid comparing success rates across multiple slices.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function creates a multi-panel figure showing success rate curves
    for multiple experiment slices (e.g., different curriculum levels or
    problem types).
    
    Args:
        slice_data: List of dicts with keys 'name', 'baseline_path', 'rfl_path'.
        out_path: Output path for the PNG file.
        window: Rolling window size for smoothing.
        title: Optional overall title.
    
    Returns:
        Metadata dictionary with plot information and checksum.
    """
    setup_deterministic_style()
    
    n_slices = len(slice_data)
    if n_slices == 0:
        raise ValueError("slice_data must contain at least one slice")
    
    # Determine grid dimensions
    n_cols = min(2, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False
    )
    
    slice_metadata = []
    
    for idx, slice_info in enumerate(sorted(slice_data, key=lambda x: x['name'])):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        name = slice_info['name']
        baseline_path = slice_info['baseline_path']
        rfl_path = slice_info['rfl_path']
        
        # Load data
        try:
            baseline_records = load_jsonl(baseline_path)
            rfl_records = load_jsonl(rfl_path)
            
            df_baseline = extract_cycle_success(baseline_records)
            df_rfl = extract_cycle_success(rfl_records)
            
            # Compute rolling success rate
            df_baseline['rolling_success'] = (
                df_baseline['success'].astype(float)
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_rfl['rolling_success'] = (
                df_rfl['success'].astype(float)
                .rolling(window=window, min_periods=1)
                .mean()
            )
            
            # Plot
            ax.plot(
                df_baseline['cycle'],
                df_baseline['rolling_success'],
                color=COLOR_BASELINE,
                linestyle='--',
                linewidth=1.5,
                label='Baseline'
            )
            ax.plot(
                df_rfl['cycle'],
                df_rfl['rolling_success'],
                color=COLOR_RFL,
                linestyle='-',
                linewidth=1.5,
                label='RFL'
            )
            
            slice_metadata.append({
                'name': name,
                'baseline_cycles': len(df_baseline),
                'rfl_cycles': len(df_rfl),
                'status': 'ok'
            })
            
        except Exception as e:
            ax.text(
                0.5, 0.5, f'Error loading data:\n{str(e)[:50]}',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=9, color='red'
            )
            slice_metadata.append({
                'name': name,
                'status': 'error',
                'error': str(e)
            })
        
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Success Rate')
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_slices, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Overall title
    fig.suptitle(
        title or 'Slice Comparison Grid — Success Rates',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    # Add PHASE II watermark to figure
    fig.text(
        0.99, 0.01, PHASE_II_WATERMARK,
        fontsize=8,
        alpha=0.5,
        ha='right',
        va='bottom',
        style='italic'
    )
    
    plt.tight_layout()
    
    # Save figure
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Compute checksum
    checksum = _compute_file_checksum(out_path)
    
    metadata = {
        'plot_type': 'slice_comparison_grid',
        'output_path': out_path,
        'window_size': window,
        'num_slices': n_slices,
        'slices': slice_metadata,
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    
    return metadata


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _ensure_dir(filepath: str) -> None:
    """Ensure the directory for a file path exists."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _compute_file_checksum(filepath: str) -> str:
    """
    Compute SHA-256 checksum of a file.
    
    Descriptive only — not admissible as uplift evidence.
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_metadata(metadata: Dict[str, Any], out_path: str) -> None:
    """
    Write metadata to a JSON file.
    
    Descriptive only — not admissible as uplift evidence.
    """
    _ensure_dir(out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def main() -> int:
    """
    CLI entry point for the Uplift Visualization Pipeline.
    
    Descriptive only — not admissible as uplift evidence.
    """
    parser = argparse.ArgumentParser(
        description='PHASE II — Uplift Visualization Pipeline. '
                    'Descriptive only — not admissible as uplift evidence.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots for a single comparison
  uv run python experiments/uplift_visualization.py \\
      --baseline results/fo_baseline.jsonl \\
      --rfl results/fo_rfl.jsonl \\
      --out-dir artifacts/uplift_figs

  # Generate specific plot type
  uv run python experiments/uplift_visualization.py \\
      --baseline results/fo_baseline.jsonl \\
      --rfl results/fo_rfl.jsonl \\
      --out-dir artifacts/uplift_figs \\
      --plot-type success_rate

  # Generate slice comparison grid
  uv run python experiments/uplift_visualization.py \\
      --slice-json slices.json \\
      --out-dir artifacts/uplift_figs \\
      --plot-type slice_grid
"""
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        help='Path to baseline JSONL file'
    )
    parser.add_argument(
        '--rfl',
        type=str,
        help='Path to RFL JSONL file'
    )
    parser.add_argument(
        '--slice-json',
        type=str,
        help='Path to JSON file defining multiple slices for grid comparison'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f'Output directory for figures (default: {DEFAULT_OUT_DIR})'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['all', 'success_rate', 'metric_trajectory', 'delta_p', 'slice_grid'],
        default='all',
        help='Type of plot to generate (default: all)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        help='Rolling window size for smoothing (default: 20)'
    )
    parser.add_argument(
        '--bin-size',
        type=int,
        default=50,
        help='Bin size for delta-p estimates (default: 50)'
    )
    parser.add_argument(
        '--metric-path',
        type=str,
        default='derivation.verified',
        help='Metric path for trajectory plot (default: derivation.verified)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.plot_type == 'slice_grid':
        if not args.slice_json:
            parser.error('--slice-json required for slice_grid plot type')
    else:
        if not args.baseline or not args.rfl:
            parser.error('--baseline and --rfl required for non-slice_grid plot types')
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_metadata = []
    
    print(f"PHASE II — Uplift Visualization Pipeline")
    print(f"Descriptive only — not admissible as uplift evidence")
    print(f"=" * 60)
    
    # Generate requested plots
    if args.plot_type in ('all', 'success_rate'):
        print("\nGenerating success rate curve...")
        out_path = os.path.join(args.out_dir, 'success_rate_curve.png')
        meta = plot_success_rate_curve(
            args.baseline, args.rfl, out_path,
            window=args.window
        )
        all_metadata.append(meta)
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type in ('all', 'metric_trajectory'):
        print("\nGenerating metric trajectory plot...")
        out_path = os.path.join(args.out_dir, 'metric_trajectory.png')
        meta = plot_metric_value_trajectory(
            args.baseline, args.rfl, out_path,
            metric_path=args.metric_path,
            window=args.window
        )
        all_metadata.append(meta)
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type in ('all', 'delta_p'):
        print("\nGenerating delta-p point estimates...")
        out_path = os.path.join(args.out_dir, 'delta_p_estimates.png')
        meta = plot_delta_p_point_estimates(
            args.baseline, args.rfl, out_path,
            bin_size=args.bin_size
        )
        all_metadata.append(meta)
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type == 'slice_grid':
        print("\nGenerating slice comparison grid...")
        # Load slice definitions
        with open(args.slice_json, 'r', encoding='utf-8') as f:
            slice_data = json.load(f)
        
        out_path = os.path.join(args.out_dir, 'slice_comparison_grid.png')
        meta = plot_slice_comparison_grid(
            slice_data, out_path,
            window=args.window
        )
        all_metadata.append(meta)
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    # Write combined metadata
    metadata_path = os.path.join(args.out_dir, 'visualization_metadata.json')
    combined_meta = {
        'pipeline': 'uplift_visualization',
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
        'plots': all_metadata,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(combined_meta, metadata_path)
    
    print(f"\n{'=' * 60}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Total plots generated: {len(all_metadata)}")
    print(f"\n{PHASE_II_LABEL}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

