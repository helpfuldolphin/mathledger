#!/usr/bin/env python
"""
PHASE II — Behavioral Telemetry Visualization Suite

Descriptive only — not admissible as uplift evidence.

This module extends the Uplift Visualization Pipeline with behavioral
telemetry visualizations for deeper system introspection. All outputs
are purely descriptive and do not constitute statistical inference,
significance claims, or uplift evidence.

New Plot Types:
- Abstention Heatmap Over Cycles
- Chain-Depth Density Field (kernel-density distribution)
- Candidate Ordering Entropy Over Time
- Rolling Metric Volatility Chart

Design Principles:
- Deterministic: All plots produce identical checksums across runs.
- No timestamps in figures: Timestamps only in metadata JSON.
- Alphabetical slice ordering: Layout engine locks grid by sorted names.
- Agg backend: Forced non-interactive backend for reproducibility.

Author: metrics-engineer-4 (Agent D4)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for determinism
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from experiments.uplift_visualization import (
    PHASE_II_LABEL,
    PHASE_II_WATERMARK,
    _STYLE_CONFIG,
    _compute_file_checksum,
    _ensure_dir,
    load_jsonl,
    setup_deterministic_style,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Monochrome-friendly colormap for heatmaps (deterministic)
_HEATMAP_COLORS = ['#ffffff', '#e0e0e0', '#b0b0b0', '#808080', '#505050', '#202020']
HEATMAP_CMAP = LinearSegmentedColormap.from_list('mono_heat', _HEATMAP_COLORS, N=256)

# Color palette extensions
COLOR_DENSITY_BASELINE = '#888888'
COLOR_DENSITY_RFL = '#333333'
COLOR_ENTROPY = '#444444'
COLOR_VOLATILITY = '#555555'


# -----------------------------------------------------------------------------
# Data Extraction Functions
# -----------------------------------------------------------------------------

# =============================================================================
# EMPTY DATA HANDLING CONTRACT
# =============================================================================
#
# All extraction functions in this module adhere to the following contract:
#
# 1. RETURN TYPE: Always return pd.DataFrame, even for empty input.
#
# 2. EMPTY INPUT HANDLING:
#    - Empty records list [] → returns empty DataFrame with 0 rows
#    - Never raises KeyError or similar due to missing columns
#    - Empty DataFrame is valid input for downstream operations
#
# 3. MISSING FIELD HANDLING:
#    - Missing 'cycle' field → defaults to row index
#    - Missing 'derivation' dict → defaults to zero/neutral values
#    - Missing nested paths → defaults to 0 (numeric) or 0.0 (float)
#
# 4. DEFAULT VALUES:
#    - abstention_rate: 0.0 (no abstention)
#    - abstained_count: 0
#    - total_candidates: 1 (avoid division by zero)
#    - chain_depth: 0
#    - entropy: 0.0 (no uncertainty)
#    - metric value: 0.0
#
# 5. TYPE GUARANTEES:
#    - 'cycle' column: int
#    - rate/ratio columns: float
#    - count columns: int
#
# See tests/test_behavioral_telemetry_viz.py::TestEmptyDataHandlingContract
# for validation tests.
# =============================================================================


def extract_abstention_series(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract cycle-level abstention data from experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        records: List of experiment records.
    
    Returns:
        DataFrame with columns: ['cycle', 'abstention_rate', 'abstained_count', 'total_candidates']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
        abstained = 0
        candidates = 1  # Avoid division by zero
        
        if 'derivation' in rec and isinstance(rec['derivation'], dict):
            abstained = rec['derivation'].get('abstained', 0)
            candidates = rec['derivation'].get('candidates', 1)
            if candidates == 0:
                candidates = 1
        
        abstention_rate = abstained / candidates
        
        rows.append({
            'cycle': int(cycle),
            'abstention_rate': float(abstention_rate),
            'abstained_count': int(abstained),
            'total_candidates': int(candidates),
        })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('cycle').reset_index(drop=True)
    return df


def extract_chain_depth_series(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract chain depth values from experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        records: List of experiment records.
    
    Returns:
        DataFrame with columns: ['cycle', 'chain_depth']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
        # Try multiple paths for chain depth
        depth = 0
        if 'derivation' in rec and isinstance(rec['derivation'], dict):
            depth = rec['derivation'].get('chain_depth', 0)
            if depth == 0:
                depth = rec['derivation'].get('depth', 0)
            if depth == 0:
                # Infer from verified count as proxy
                depth = rec['derivation'].get('verified', 0)
        
        rows.append({
            'cycle': int(cycle),
            'chain_depth': int(depth),
        })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('cycle').reset_index(drop=True)
    return df


def extract_candidate_ordering_entropy(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract or compute candidate ordering entropy from experiment records.
    
    Descriptive only — not admissible as uplift evidence.
    
    Entropy measures the "randomness" of candidate selection ordering.
    Higher entropy = more uniform selection; lower = more deterministic.
    
    Args:
        records: List of experiment records.
    
    Returns:
        DataFrame with columns: ['cycle', 'entropy']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
        entropy = 0.0
        
        # Check for explicit entropy field
        if 'candidate_entropy' in rec:
            entropy = float(rec['candidate_entropy'])
        elif 'derivation' in rec and isinstance(rec['derivation'], dict):
            deriv = rec['derivation']
            if 'ordering_entropy' in deriv:
                entropy = float(deriv['ordering_entropy'])
            elif 'candidate_scores' in deriv:
                # Compute entropy from scores if available
                scores = deriv['candidate_scores']
                if isinstance(scores, list) and len(scores) > 0:
                    entropy = _compute_entropy_from_scores(scores)
            else:
                # Use candidates/verified ratio as proxy for entropy
                candidates = deriv.get('candidates', 1)
                verified = deriv.get('verified', 0)
                if candidates > 0:
                    p = verified / candidates
                    if 0 < p < 1:
                        entropy = -p * math.log2(p) - (1-p) * math.log2(1-p)
        
        rows.append({
            'cycle': int(cycle),
            'entropy': float(entropy),
        })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('cycle').reset_index(drop=True)
    return df


def _compute_entropy_from_scores(scores: List[float]) -> float:
    """
    Compute Shannon entropy from a list of scores (treated as probabilities).
    
    Descriptive only — not admissible as uplift evidence.
    """
    if not scores:
        return 0.0
    
    # Normalize to probabilities
    total = sum(abs(s) for s in scores)
    if total == 0:
        return 0.0
    
    probs = [abs(s) / total for s in scores]
    
    # Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def extract_metric_for_volatility(
    records: List[Dict[str, Any]],
    metric_path: str = 'derivation.verified'
) -> pd.DataFrame:
    """
    Extract metric values for volatility computation.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        records: List of experiment records.
        metric_path: Dot-separated path to metric.
    
    Returns:
        DataFrame with columns: ['cycle', 'value']
    """
    rows = []
    for rec in records:
        cycle = rec.get('cycle', len(rows))
        
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
    if len(df) > 0:
        df = df.sort_values('cycle').reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Behavioral Telemetry Plot Functions
# -----------------------------------------------------------------------------

def plot_abstention_heatmap(
    records_path: str,
    out_path: str,
    bin_size: int = 20,
    n_bins_y: int = 10,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot abstention heatmap over cycles.
    
    Descriptive only — not admissible as uplift evidence.
    
    Creates a 2D heatmap where:
    - X-axis: Cycle bins
    - Y-axis: Abstention rate bins (0 to 1)
    - Color: Frequency count
    
    Args:
        records_path: Path to JSONL file.
        out_path: Output path for PNG.
        bin_size: Number of cycles per x-bin.
        n_bins_y: Number of bins for abstention rate (y-axis).
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with checksum.
    """
    setup_deterministic_style()
    
    records = load_jsonl(records_path)
    df = extract_abstention_series(records)
    
    if len(df) == 0:
        # Handle empty data gracefully
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title or 'Abstention Heatmap — No Data')
    else:
        # Compute 2D histogram
        max_cycle = int(df['cycle'].max()) + 1
        n_bins_x = max(1, (max_cycle + bin_size - 1) // bin_size)
        
        # Create bins
        x_edges = np.linspace(0, max_cycle, n_bins_x + 1)
        y_edges = np.linspace(0, 1, n_bins_y + 1)
        
        # Compute histogram (deterministic)
        H, _, _ = np.histogram2d(
            df['cycle'].values,
            df['abstention_rate'].values,
            bins=[x_edges, y_edges]
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot heatmap
        im = ax.imshow(
            H.T,
            origin='lower',
            aspect='auto',
            extent=[0, max_cycle, 0, 1],
            cmap=HEATMAP_CMAP,
            interpolation='nearest'
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=10)
        
        # Labels
        ax.set_xlabel('Cycle Index')
        ax.set_ylabel('Abstention Rate')
        ax.set_title(title or 'Abstention Heatmap Over Cycles')
    
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
    
    # Save
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    checksum = _compute_file_checksum(out_path)
    
    return {
        'plot_type': 'abstention_heatmap',
        'records_path': records_path,
        'output_path': out_path,
        'bin_size_x': bin_size,
        'n_bins_y': n_bins_y,
        'total_records': len(df),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


def plot_chain_depth_density(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    n_bins: int = 20,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot chain-depth density field (kernel-density style histogram).
    
    Descriptive only — not admissible as uplift evidence.
    
    Creates overlapping histograms showing the distribution of chain depths
    for baseline vs RFL runs.
    
    Args:
        baseline_path: Path to baseline JSONL.
        rfl_path: Path to RFL JSONL.
        out_path: Output path for PNG.
        n_bins: Number of histogram bins.
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with checksum.
    """
    setup_deterministic_style()
    
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_chain_depth_series(baseline_records)
    df_rfl = extract_chain_depth_series(rfl_records)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine common bin range
    all_depths = pd.concat([df_baseline['chain_depth'], df_rfl['chain_depth']])
    if len(all_depths) == 0 or all_depths.max() == all_depths.min():
        min_depth, max_depth = 0, 10
    else:
        min_depth = int(all_depths.min())
        max_depth = int(all_depths.max()) + 1
    
    bins = np.linspace(min_depth, max_depth, n_bins + 1)
    
    # Plot histograms (normalized to density)
    if len(df_baseline) > 0:
        ax.hist(
            df_baseline['chain_depth'],
            bins=bins,
            density=True,
            alpha=0.5,
            color=COLOR_DENSITY_BASELINE,
            edgecolor='white',
            linewidth=0.5,
            label=f'Baseline (n={len(df_baseline)})'
        )
    
    if len(df_rfl) > 0:
        ax.hist(
            df_rfl['chain_depth'],
            bins=bins,
            density=True,
            alpha=0.5,
            color=COLOR_DENSITY_RFL,
            edgecolor='white',
            linewidth=0.5,
            label=f'RFL (n={len(df_rfl)})'
        )
    
    ax.set_xlabel('Chain Depth')
    ax.set_ylabel('Density')
    ax.set_title(title or 'Chain-Depth Density Field')
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
    
    # Save
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    checksum = _compute_file_checksum(out_path)
    
    return {
        'plot_type': 'chain_depth_density',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'n_bins': n_bins,
        'baseline_records': len(df_baseline),
        'rfl_records': len(df_rfl),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


def plot_candidate_ordering_entropy(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    window: int = 20,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot candidate ordering entropy over time.
    
    Descriptive only — not admissible as uplift evidence.
    
    Shows the rolling entropy of candidate selection ordering for
    baseline vs RFL runs.
    
    Args:
        baseline_path: Path to baseline JSONL.
        rfl_path: Path to RFL JSONL.
        out_path: Output path for PNG.
        window: Rolling window size.
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with checksum.
    """
    setup_deterministic_style()
    
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_candidate_ordering_entropy(baseline_records)
    df_rfl = extract_candidate_ordering_entropy(rfl_records)
    
    # Compute rolling mean
    if len(df_baseline) > 0:
        df_baseline['rolling_entropy'] = (
            df_baseline['entropy']
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    if len(df_rfl) > 0:
        df_rfl['rolling_entropy'] = (
            df_rfl['entropy']
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    if len(df_baseline) > 0:
        ax.plot(
            df_baseline['cycle'],
            df_baseline['rolling_entropy'],
            color=COLOR_DENSITY_BASELINE,
            linestyle='--',
            linewidth=2,
            label='Baseline'
        )
    
    if len(df_rfl) > 0:
        ax.plot(
            df_rfl['cycle'],
            df_rfl['rolling_entropy'],
            color=COLOR_ENTROPY,
            linestyle='-',
            linewidth=2,
            label='RFL'
        )
    
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel(f'Entropy (rolling window={window})')
    ax.set_title(title or 'Candidate Ordering Entropy Over Time')
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
    
    # Save
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    checksum = _compute_file_checksum(out_path)
    
    return {
        'plot_type': 'candidate_ordering_entropy',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'window_size': window,
        'baseline_records': len(df_baseline),
        'rfl_records': len(df_rfl),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


def plot_rolling_metric_volatility(
    baseline_path: str,
    rfl_path: str,
    out_path: str,
    metric_path: str = 'derivation.verified',
    window: int = 20,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot rolling metric volatility chart.
    
    Descriptive only — not admissible as uplift evidence.
    
    Volatility is measured as rolling standard deviation of the metric.
    Higher volatility = more variability in outcomes.
    
    Args:
        baseline_path: Path to baseline JSONL.
        rfl_path: Path to RFL JSONL.
        out_path: Output path for PNG.
        metric_path: Dot-separated path to metric.
        window: Rolling window size for volatility computation.
        title: Optional custom title.
    
    Returns:
        Metadata dictionary with checksum.
    """
    setup_deterministic_style()
    
    baseline_records = load_jsonl(baseline_path)
    rfl_records = load_jsonl(rfl_path)
    
    df_baseline = extract_metric_for_volatility(baseline_records, metric_path)
    df_rfl = extract_metric_for_volatility(rfl_records, metric_path)
    
    # Compute rolling volatility (std dev)
    if len(df_baseline) > 0:
        df_baseline['volatility'] = (
            df_baseline['value']
            .rolling(window=window, min_periods=2)
            .std()
            .fillna(0)
        )
    
    if len(df_rfl) > 0:
        df_rfl['volatility'] = (
            df_rfl['value']
            .rolling(window=window, min_periods=2)
            .std()
            .fillna(0)
        )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    if len(df_baseline) > 0:
        ax.plot(
            df_baseline['cycle'],
            df_baseline['volatility'],
            color=COLOR_DENSITY_BASELINE,
            linestyle='--',
            linewidth=2,
            label='Baseline'
        )
    
    if len(df_rfl) > 0:
        ax.plot(
            df_rfl['cycle'],
            df_rfl['volatility'],
            color=COLOR_VOLATILITY,
            linestyle='-',
            linewidth=2,
            label='RFL'
        )
    
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel(f'Volatility (rolling std, window={window})')
    ax.set_title(title or f'Rolling Metric Volatility: {metric_path}')
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
    
    # Save
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    checksum = _compute_file_checksum(out_path)
    
    return {
        'plot_type': 'rolling_metric_volatility',
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'output_path': out_path,
        'metric_path': metric_path,
        'window_size': window,
        'baseline_records': len(df_baseline),
        'rfl_records': len(df_rfl),
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


# -----------------------------------------------------------------------------
# Deterministic Layout Engine
# -----------------------------------------------------------------------------

class DeterministicLayoutEngine:
    """
    Deterministic layout engine for multi-panel visualizations.
    
    Descriptive only — not admissible as uplift evidence.
    
    Ensures:
    - Grid arrangement locked by alphabetical slice name
    - Identical PNG checksum across runs
    - No timestamp inclusion in figures
    """
    
    def __init__(self, n_cols: int = 2):
        """
        Initialize layout engine.
        
        Args:
            n_cols: Maximum number of columns in grid.
        """
        self.n_cols = n_cols
    
    def compute_grid_dimensions(self, n_items: int) -> Tuple[int, int]:
        """
        Compute grid dimensions for n items.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            n_items: Number of items to arrange.
        
        Returns:
            Tuple of (n_rows, n_cols).
        """
        n_cols = min(self.n_cols, n_items)
        n_rows = (n_items + n_cols - 1) // n_cols
        return n_rows, n_cols
    
    def sort_items_alphabetically(
        self,
        items: List[Dict[str, Any]],
        key: str = 'name'
    ) -> List[Dict[str, Any]]:
        """
        Sort items alphabetically by key for deterministic ordering.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            items: List of item dictionaries.
            key: Key to sort by.
        
        Returns:
            Sorted list.
        """
        return sorted(items, key=lambda x: str(x.get(key, '')))
    
    def create_grid_figure(
        self,
        n_items: int,
        figsize_per_cell: Tuple[float, float] = (5, 4)
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a figure with grid of subplots.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            n_items: Number of items.
            figsize_per_cell: (width, height) per cell.
        
        Returns:
            Tuple of (Figure, axes array).
        """
        n_rows, n_cols = self.compute_grid_dimensions(n_items)
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
            squeeze=False
        )
        
        return fig, axes
    
    def get_cell_position(self, index: int) -> Tuple[int, int]:
        """
        Get (row, col) position for item at index.
        
        Args:
            index: Item index.
        
        Returns:
            Tuple of (row, col).
        """
        row = index // self.n_cols
        col = index % self.n_cols
        return row, col


def plot_behavioral_telemetry_grid(
    slice_data: List[Dict[str, str]],
    out_path: str,
    plot_type: str = 'abstention_heatmap',
    title: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Plot a grid of behavioral telemetry visualizations across slices.
    
    Descriptive only — not admissible as uplift evidence.
    
    Uses DeterministicLayoutEngine for consistent ordering.
    
    Args:
        slice_data: List of dicts with 'name', 'baseline_path', 'rfl_path'.
        out_path: Output path for PNG.
        plot_type: One of 'abstention_heatmap', 'chain_depth_density',
                   'entropy', 'volatility'.
        title: Optional overall title.
        **kwargs: Additional arguments for specific plot types.
    
    Returns:
        Metadata dictionary with checksum.
    """
    setup_deterministic_style()
    
    if len(slice_data) == 0:
        raise ValueError("slice_data must contain at least one slice")
    
    layout = DeterministicLayoutEngine(n_cols=2)
    sorted_slices = layout.sort_items_alphabetically(slice_data, key='name')
    
    n_slices = len(sorted_slices)
    fig, axes = layout.create_grid_figure(n_slices)
    
    slice_metadata = []
    
    for idx, slice_info in enumerate(sorted_slices):
        row, col = layout.get_cell_position(idx)
        ax = axes[row, col]
        
        name = slice_info['name']
        baseline_path = slice_info.get('baseline_path', '')
        rfl_path = slice_info.get('rfl_path', '')
        
        try:
            if plot_type == 'abstention_heatmap':
                _render_abstention_heatmap_cell(ax, baseline_path, name, **kwargs)
            elif plot_type == 'chain_depth_density':
                _render_chain_depth_cell(ax, baseline_path, rfl_path, name, **kwargs)
            elif plot_type == 'entropy':
                _render_entropy_cell(ax, baseline_path, rfl_path, name, **kwargs)
            elif plot_type == 'volatility':
                _render_volatility_cell(ax, baseline_path, rfl_path, name, **kwargs)
            else:
                ax.text(0.5, 0.5, f'Unknown plot type: {plot_type}',
                       ha='center', va='center', transform=ax.transAxes)
            
            slice_metadata.append({'name': name, 'status': 'ok'})
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:40]}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9, color='red')
            slice_metadata.append({'name': name, 'status': 'error', 'error': str(e)})
    
    # Hide unused cells
    n_rows, n_cols = layout.compute_grid_dimensions(n_slices)
    for idx in range(n_slices, n_rows * n_cols):
        row, col = layout.get_cell_position(idx)
        axes[row, col].set_visible(False)
    
    # Title
    fig.suptitle(
        title or f'Behavioral Telemetry Grid — {plot_type}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    # Watermark
    fig.text(
        0.99, 0.01, PHASE_II_WATERMARK,
        fontsize=8,
        alpha=0.5,
        ha='right',
        va='bottom',
        style='italic'
    )
    
    plt.tight_layout()
    
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    checksum = _compute_file_checksum(out_path)
    
    return {
        'plot_type': f'behavioral_telemetry_grid_{plot_type}',
        'output_path': out_path,
        'num_slices': n_slices,
        'slices': slice_metadata,
        'checksum_sha256': checksum,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


def _render_abstention_heatmap_cell(ax, records_path: str, name: str, **kwargs):
    """Render abstention heatmap in a single cell."""
    records = load_jsonl(records_path)
    df = extract_abstention_series(records)
    
    bin_size = kwargs.get('bin_size', 20)
    n_bins_y = kwargs.get('n_bins_y', 10)
    
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    else:
        max_cycle = int(df['cycle'].max()) + 1
        n_bins_x = max(1, (max_cycle + bin_size - 1) // bin_size)
        
        x_edges = np.linspace(0, max_cycle, n_bins_x + 1)
        y_edges = np.linspace(0, 1, n_bins_y + 1)
        
        H, _, _ = np.histogram2d(
            df['cycle'].values,
            df['abstention_rate'].values,
            bins=[x_edges, y_edges]
        )
        
        ax.imshow(H.T, origin='lower', aspect='auto',
                  extent=[0, max_cycle, 0, 1], cmap=HEATMAP_CMAP,
                  interpolation='nearest')
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Abstention Rate')
    ax.set_title(name, fontsize=10)


def _render_chain_depth_cell(ax, baseline_path: str, rfl_path: str, name: str, **kwargs):
    """Render chain depth density in a single cell."""
    n_bins = kwargs.get('n_bins', 15)
    
    df_baseline = extract_chain_depth_series(load_jsonl(baseline_path)) if baseline_path else pd.DataFrame()
    df_rfl = extract_chain_depth_series(load_jsonl(rfl_path)) if rfl_path else pd.DataFrame()
    
    all_depths = pd.concat([
        df_baseline['chain_depth'] if len(df_baseline) > 0 else pd.Series(dtype=float),
        df_rfl['chain_depth'] if len(df_rfl) > 0 else pd.Series(dtype=float)
    ])
    
    if len(all_depths) == 0:
        min_d, max_d = 0, 10
    else:
        min_d, max_d = int(all_depths.min()), int(all_depths.max()) + 1
    
    bins = np.linspace(min_d, max_d, n_bins + 1)
    
    if len(df_baseline) > 0:
        ax.hist(df_baseline['chain_depth'], bins=bins, density=True,
                alpha=0.5, color=COLOR_DENSITY_BASELINE, label='Baseline')
    if len(df_rfl) > 0:
        ax.hist(df_rfl['chain_depth'], bins=bins, density=True,
                alpha=0.5, color=COLOR_DENSITY_RFL, label='RFL')
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('Density')
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)


def _render_entropy_cell(ax, baseline_path: str, rfl_path: str, name: str, **kwargs):
    """Render entropy trajectory in a single cell."""
    window = kwargs.get('window', 20)
    
    df_baseline = extract_candidate_ordering_entropy(load_jsonl(baseline_path)) if baseline_path else pd.DataFrame()
    df_rfl = extract_candidate_ordering_entropy(load_jsonl(rfl_path)) if rfl_path else pd.DataFrame()
    
    if len(df_baseline) > 0:
        df_baseline['rolling'] = df_baseline['entropy'].rolling(window=window, min_periods=1).mean()
        ax.plot(df_baseline['cycle'], df_baseline['rolling'],
                color=COLOR_DENSITY_BASELINE, linestyle='--', label='Baseline')
    
    if len(df_rfl) > 0:
        df_rfl['rolling'] = df_rfl['entropy'].rolling(window=window, min_periods=1).mean()
        ax.plot(df_rfl['cycle'], df_rfl['rolling'],
                color=COLOR_ENTROPY, linestyle='-', label='RFL')
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Entropy')
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _render_volatility_cell(ax, baseline_path: str, rfl_path: str, name: str, **kwargs):
    """Render volatility trajectory in a single cell."""
    window = kwargs.get('window', 20)
    metric_path = kwargs.get('metric_path', 'derivation.verified')
    
    df_baseline = extract_metric_for_volatility(load_jsonl(baseline_path), metric_path) if baseline_path else pd.DataFrame()
    df_rfl = extract_metric_for_volatility(load_jsonl(rfl_path), metric_path) if rfl_path else pd.DataFrame()
    
    if len(df_baseline) > 0:
        df_baseline['volatility'] = df_baseline['value'].rolling(window=window, min_periods=2).std().fillna(0)
        ax.plot(df_baseline['cycle'], df_baseline['volatility'],
                color=COLOR_DENSITY_BASELINE, linestyle='--', label='Baseline')
    
    if len(df_rfl) > 0:
        df_rfl['volatility'] = df_rfl['value'].rolling(window=window, min_periods=2).std().fillna(0)
        ax.plot(df_rfl['cycle'], df_rfl['volatility'],
                color=COLOR_VOLATILITY, linestyle='-', label='RFL')
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Volatility')
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Visualization Manifest Generator
# -----------------------------------------------------------------------------

class VisualizationManifest:
    """
    Manifest generator for A/B reproducibility.
    
    Descriptive only — not admissible as uplift evidence.
    
    Stores:
    - File hashes (SHA-256)
    - Parameter hashes
    - rcParams hashes
    """
    
    def __init__(self):
        """Initialize empty manifest."""
        self.entries: List[Dict[str, Any]] = []
        self._rcparams_hash: Optional[str] = None
    
    def compute_rcparams_hash(self) -> str:
        """
        Compute hash of current rcParams for reproducibility verification.
        
        Descriptive only — not admissible as uplift evidence.
        
        Returns:
            SHA-256 hash of serialized rcParams.
        """
        # Extract relevant rcParams (skip non-serializable)
        serializable = {}
        for key, value in _STYLE_CONFIG.items():
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)
        
        content = json.dumps(serializable, sort_keys=True)
        self._rcparams_hash = hashlib.sha256(content.encode()).hexdigest()
        return self._rcparams_hash
    
    def compute_parameter_hash(self, params: Dict[str, Any]) -> str:
        """
        Compute hash of parameters for reproducibility verification.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            params: Parameter dictionary.
        
        Returns:
            SHA-256 hash of serialized parameters.
        """
        content = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def add_entry(
        self,
        plot_metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> None:
        """
        Add an entry to the manifest.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            plot_metadata: Metadata from plot function (includes file checksum).
            parameters: Parameters used to generate the plot.
        """
        entry = {
            'plot_type': plot_metadata.get('plot_type', 'unknown'),
            'output_path': plot_metadata.get('output_path', ''),
            'file_hash_sha256': plot_metadata.get('checksum_sha256', ''),
            'parameter_hash_sha256': self.compute_parameter_hash(parameters),
            'parameters': parameters,
            'phase': 'PHASE II',
        }
        self.entries.append(entry)
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate the complete manifest.
        
        Descriptive only — not admissible as uplift evidence.
        
        Returns:
            Complete manifest dictionary.
        """
        if self._rcparams_hash is None:
            self.compute_rcparams_hash()
        
        return {
            'manifest_version': '1.0.0',
            'generator': 'behavioral_telemetry_viz',
            'phase': 'PHASE II',
            'label': PHASE_II_LABEL,
            'rcparams_hash_sha256': self._rcparams_hash,
            'entries': self.entries,
            'entry_count': len(self.entries),
        }
    
    def save(self, out_path: str) -> str:
        """
        Save manifest to JSON file.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            out_path: Output path for JSON file.
        
        Returns:
            SHA-256 hash of the manifest file.
        """
        manifest = self.generate()
        
        _ensure_dir(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        
        return _compute_file_checksum(out_path)
    
    @classmethod
    def load(cls, path: str) -> 'VisualizationManifest':
        """
        Load manifest from JSON file.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            path: Path to manifest JSON.
        
        Returns:
            VisualizationManifest instance.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        manifest = cls()
        manifest._rcparams_hash = data.get('rcparams_hash_sha256')
        manifest.entries = data.get('entries', [])
        return manifest
    
    def verify_reproducibility(self, other: 'VisualizationManifest') -> Dict[str, Any]:
        """
        Verify A/B reproducibility between two manifests.
        
        Descriptive only — not admissible as uplift evidence.
        
        Args:
            other: Another manifest to compare against.
        
        Returns:
            Verification report dictionary.
        """
        report = {
            'rcparams_match': self._rcparams_hash == other._rcparams_hash,
            'entry_count_match': len(self.entries) == len(other.entries),
            'file_hash_matches': [],
            'parameter_hash_matches': [],
            'all_match': True,
        }
        
        # Compare entries by plot_type
        self_by_type = {e['plot_type']: e for e in self.entries}
        other_by_type = {e['plot_type']: e for e in other.entries}
        
        for plot_type in set(self_by_type.keys()) | set(other_by_type.keys()):
            self_entry = self_by_type.get(plot_type, {})
            other_entry = other_by_type.get(plot_type, {})
            
            file_match = self_entry.get('file_hash_sha256') == other_entry.get('file_hash_sha256')
            param_match = self_entry.get('parameter_hash_sha256') == other_entry.get('parameter_hash_sha256')
            
            report['file_hash_matches'].append({
                'plot_type': plot_type,
                'match': file_match
            })
            report['parameter_hash_matches'].append({
                'plot_type': plot_type,
                'match': param_match
            })
            
            if not file_match or not param_match:
                report['all_match'] = False
        
        if not report['rcparams_match'] or not report['entry_count_match']:
            report['all_match'] = False
        
        return report


# -----------------------------------------------------------------------------
# Telemetry Pack Index Builder
# -----------------------------------------------------------------------------

# =============================================================================
# PACK INDEX CONTRACT (pack_index.json)
# =============================================================================
#
# The pack index provides a canonical, developer-facing quick reference for
# telemetry pack contents. All fields are deterministic and hash-stable.
#
# SCHEMA:
# {
#   "generated_at": "ISO8601 timestamp (e.g., 2025-12-06T12:34:56Z)",
#   "baseline_log": "path to baseline JSONL",
#   "rfl_log": "path to RFL JSONL",
#   "plots": [
#     {
#       "name": "plot_type identifier (e.g., abstention_heatmap)",
#       "filename": "output filename (e.g., abstention_heatmap.png)",
#       "checksum": "SHA-256 hex digest of PNG file"
#     },
#     ...
#   ],
#   "manifest_hash": "SHA-256 hex digest of telemetry_manifest.json",
#   "phase": "PHASE II",
#   "label": "PHASE II — Descriptive only, not admissible as uplift evidence."
# }
#
# DETERMINISM GUARANTEES:
# - plots[] is sorted alphabetically by "name" field
# - All checksums are SHA-256 hex digests (64 characters)
# - Re-generating with identical inputs produces identical checksums
# - generated_at is the only non-deterministic field (wall clock time)
#
# VALIDATION:
# Use validate_pack_index_against_manifest(pack_dir) to verify:
# - All plot checksums match telemetry_manifest.json entries
# - manifest_hash matches actual file hash
# =============================================================================

# Filename mappings for deterministic index generation
_PLOT_FILENAMES: Dict[str, str] = {
    'abstention_heatmap': 'abstention_heatmap.png',
    'chain_depth_density': 'chain_depth_density.png',
    'candidate_ordering_entropy': 'candidate_entropy.png',
    'rolling_metric_volatility': 'metric_volatility.png',
}


def _build_pack_index(
    baseline_path: str,
    rfl_path: str,
    all_metadata: List[Dict[str, Any]],
    manifest_hash: str,
) -> Dict[str, Any]:
    """
    Build the pack_index.json content.
    
    Descriptive only — not admissible as uplift evidence.
    
    The pack index provides a deterministic, developer-facing quick reference
    for tooling integration. All fields are stable and hash-reproducible.
    
    Args:
        baseline_path: Path to baseline JSONL.
        rfl_path: Path to RFL JSONL.
        all_metadata: List of metadata dicts from plot functions.
        manifest_hash: SHA-256 hash of the telemetry_manifest.json.
    
    Returns:
        Pack index dictionary.
    """
    # Build plots list (deterministic order by plot_type)
    plots = []
    for meta in sorted(all_metadata, key=lambda m: m.get('plot_type', '')):
        plot_type = meta.get('plot_type', 'unknown')
        filename = _PLOT_FILENAMES.get(plot_type, f'{plot_type}.png')
        checksum = meta.get('checksum_sha256', '')
        
        plots.append({
            'name': plot_type,
            'filename': filename,
            'checksum': checksum,
        })
    
    # Use a fixed timestamp format for determinism in testing
    # In production, this captures when the pack was generated
    # Note: For true determinism in tests, callers can override via mocking
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    return {
        'generated_at': generated_at,
        'baseline_log': baseline_path,
        'rfl_log': rfl_path,
        'plots': plots,
        'manifest_hash': manifest_hash,
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }


def load_pack_index(pack_dir: str) -> Dict[str, Any]:
    """
    Load pack_index.json from a telemetry pack directory.
    
    Descriptive only — not admissible as uplift evidence.
    
    Args:
        pack_dir: Path to telemetry pack directory.
    
    Returns:
        Pack index dictionary.
    
    Raises:
        FileNotFoundError: If pack_index.json doesn't exist.
    """
    index_path = os.path.join(pack_dir, 'pack_index.json')
    with open(index_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_pack_index_against_manifest(pack_dir: str) -> Dict[str, Any]:
    """
    Validate that pack_index.json matches telemetry_manifest.json.
    
    Descriptive only — not admissible as uplift evidence.
    
    Checks:
    - All plots in index exist in manifest
    - Checksums match between index and manifest
    - manifest_hash in index matches actual manifest file hash
    
    Args:
        pack_dir: Path to telemetry pack directory.
    
    Returns:
        Validation report dictionary.
    """
    index_path = os.path.join(pack_dir, 'pack_index.json')
    manifest_path = os.path.join(pack_dir, 'telemetry_manifest.json')
    
    # Check files exist
    if not os.path.exists(index_path):
        return {'valid': False, 'error': 'pack_index.json not found'}
    if not os.path.exists(manifest_path):
        return {'valid': False, 'error': 'telemetry_manifest.json not found'}
    
    # Load both
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    manifest = VisualizationManifest.load(manifest_path)
    
    # Build manifest lookup by plot_type
    manifest_by_type = {
        entry.get('plot_type', ''): entry
        for entry in manifest.entries
    }
    
    report = {
        'valid': True,
        'pack_dir': pack_dir,
        'index_plot_count': len(index.get('plots', [])),
        'manifest_entry_count': len(manifest.entries),
        'plot_checks': [],
        'manifest_hash_match': False,
    }
    
    # Check manifest_hash
    actual_manifest_hash = _compute_file_checksum(manifest_path)
    index_manifest_hash = index.get('manifest_hash', '')
    report['manifest_hash_match'] = (actual_manifest_hash == index_manifest_hash)
    if not report['manifest_hash_match']:
        report['valid'] = False
    
    # Check each plot in index
    for plot_entry in index.get('plots', []):
        plot_name = plot_entry.get('name', '')
        index_checksum = plot_entry.get('checksum', '')
        
        manifest_entry = manifest_by_type.get(plot_name, {})
        manifest_checksum = manifest_entry.get('file_hash_sha256', '')
        
        matches = (index_checksum == manifest_checksum) and manifest_checksum != ''
        
        report['plot_checks'].append({
            'name': plot_name,
            'filename': plot_entry.get('filename', ''),
            'index_checksum': index_checksum[:16] + '...' if index_checksum else '',
            'manifest_checksum': manifest_checksum[:16] + '...' if manifest_checksum else '',
            'match': matches,
        })
        
        if not matches:
            report['valid'] = False
    
    return report


# -----------------------------------------------------------------------------
# Telemetry ↔ Atlas Cross-Link Contract
# -----------------------------------------------------------------------------

# =============================================================================
# CROSS-LINK CONTRACT (telemetry_atlas_link)
# =============================================================================
#
# This contract allows other systems to link a telemetry pack to an atlas
# snapshot via a small, stable, deterministic object.
#
# SCHEMA:
# {
#   "schema_version": "1.0.0",
#   "telemetry_pack": {
#     "generated_at": "ISO8601 timestamp",
#     "manifest_hash": "SHA-256 hex digest",
#     "plot_count": int
#   },
#   "atlas": {
#     "generated_at": "ISO8601 timestamp or null",
#     "cluster_count": int,
#     "fingerprint_hash": "SHA-256 hex digest or empty"
#   }
# }
#
# GUARANTEES:
# - Deterministic: same inputs → same output (no side effects)
# - Stable: schema_version tracks breaking changes
# - Minimal: only cross-referencing metadata, no large payloads
# =============================================================================

CROSS_LINK_SCHEMA_VERSION = "1.0.0"


def build_telemetry_atlas_link(
    telemetry_index: Dict[str, Any],
    atlas_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a cross-link object connecting a telemetry pack to an atlas snapshot.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    This function creates a stable, deterministic link object that allows
    downstream systems to verify which telemetry pack corresponds to which
    atlas snapshot. No side effects; purely functional.
    
    Args:
        telemetry_index: Pack index dictionary (from load_pack_index()).
        atlas_metadata: Atlas metadata dictionary. Expected keys:
            - 'generated_at': ISO8601 timestamp (optional)
            - 'slice_count' or 'cluster_count': Number of clusters
            - 'fingerprint_hash' or 'manifest_hash': Atlas content hash
    
    Returns:
        Cross-link object with schema_version, telemetry_pack, and atlas sections.
    
    Example:
        >>> index = load_pack_index('artifacts/telemetry')
        >>> atlas_meta = {'generated_at': '...', 'slice_count': 4, 'manifest_hash': '...'}
        >>> link = build_telemetry_atlas_link(index, atlas_meta)
        >>> print(link['schema_version'])
        '1.0.0'
    """
    # Extract telemetry pack info
    telemetry_section = {
        'generated_at': telemetry_index.get('generated_at', ''),
        'manifest_hash': telemetry_index.get('manifest_hash', ''),
        'plot_count': len(telemetry_index.get('plots', [])),
    }
    
    # Extract atlas info (handle multiple possible key names)
    cluster_count = (
        atlas_metadata.get('cluster_count') or
        atlas_metadata.get('slice_count') or
        atlas_metadata.get('n_clusters') or
        0
    )
    
    fingerprint_hash = (
        atlas_metadata.get('fingerprint_hash') or
        atlas_metadata.get('manifest_hash') or
        atlas_metadata.get('atlas_hash') or
        ''
    )
    
    atlas_section = {
        'generated_at': atlas_metadata.get('generated_at', ''),
        'cluster_count': int(cluster_count),
        'fingerprint_hash': str(fingerprint_hash),
    }
    
    return {
        'schema_version': CROSS_LINK_SCHEMA_VERSION,
        'telemetry_pack': telemetry_section,
        'atlas': atlas_section,
    }


# -----------------------------------------------------------------------------
# Telemetry Pack Health Predicate
# -----------------------------------------------------------------------------

def is_telemetry_pack_healthy(index: Dict[str, Any]) -> bool:
    """
    Check whether a telemetry pack is "healthy" and usable for analysis.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    This is an ADVISORY predicate, not CI-blocking by itself. A healthy pack:
    - Has at least 1 plot entry
    - Has all plot checksums as non-empty 64-character hex strings
    - Has a manifest_hash present
    
    Args:
        index: Pack index dictionary (from load_pack_index()).
    
    Returns:
        True if the pack passes all health checks, False otherwise.
    
    Note:
        This predicate is advisory. A False result indicates the pack may be
        incomplete or corrupted, but does not prevent further analysis. Use
        validate_pack_index_against_manifest() for stricter validation.
    
    Example:
        >>> index = load_pack_index('artifacts/telemetry')
        >>> if is_telemetry_pack_healthy(index):
        ...     print("Pack is ready for analysis")
    """
    # Check 1: At least 1 plot entry
    plots = index.get('plots', [])
    if len(plots) < 1:
        return False
    
    # Check 2: All plot checksums are valid 64-char hex strings
    for plot in plots:
        checksum = plot.get('checksum', '')
        if not _is_valid_sha256_hex(checksum):
            return False
    
    # Check 3: manifest_hash is present and valid
    manifest_hash = index.get('manifest_hash', '')
    if not _is_valid_sha256_hex(manifest_hash):
        return False
    
    return True


def _is_valid_sha256_hex(value: str) -> bool:
    """Check if a string is a valid SHA-256 hex digest (64 hex characters)."""
    if not isinstance(value, str):
        return False
    if len(value) != 64:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def get_pack_health_report(index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a detailed health report for a telemetry pack.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Args:
        index: Pack index dictionary.
    
    Returns:
        Health report with individual check results.
    """
    plots = index.get('plots', [])
    manifest_hash = index.get('manifest_hash', '')
    
    invalid_checksums = []
    for plot in plots:
        checksum = plot.get('checksum', '')
        if not _is_valid_sha256_hex(checksum):
            invalid_checksums.append(plot.get('name', 'unknown'))
    
    has_plots = len(plots) >= 1
    has_valid_manifest_hash = _is_valid_sha256_hex(manifest_hash)
    all_checksums_valid = len(invalid_checksums) == 0
    
    return {
        'healthy': has_plots and has_valid_manifest_hash and all_checksums_valid,
        'checks': {
            'has_plots': has_plots,
            'plot_count': len(plots),
            'has_valid_manifest_hash': has_valid_manifest_hash,
            'all_checksums_valid': all_checksums_valid,
            'invalid_checksum_plots': invalid_checksums,
        },
        'advisory': 'This is an advisory check, not CI-blocking by itself.',
    }


# -----------------------------------------------------------------------------
# Telemetry Pack Snapshot Guard
# -----------------------------------------------------------------------------

def save_pack_snapshot(pack_dir: str, snapshot_path: str) -> Dict[str, Any]:
    """
    Save a snapshot of a telemetry pack's structural/checksum information.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    The snapshot captures only structural and checksum information, excluding
    timestamps, to enable deterministic comparisons across refactors.
    
    Args:
        pack_dir: Path to telemetry pack directory.
        snapshot_path: Output path for snapshot JSON.
    
    Returns:
        Snapshot dictionary that was saved.
    
    Example:
        >>> save_pack_snapshot('artifacts/telemetry', 'snapshots/telemetry_v1.json')
    """
    index = load_pack_index(pack_dir)
    
    # Build snapshot (exclude timestamps for determinism)
    snapshot = {
        'snapshot_version': '1.0.0',
        'baseline_log': index.get('baseline_log', ''),
        'rfl_log': index.get('rfl_log', ''),
        'plots': sorted(
            [
                {
                    'name': p.get('name', ''),
                    'filename': p.get('filename', ''),
                    'checksum': p.get('checksum', ''),
                }
                for p in index.get('plots', [])
            ],
            key=lambda x: x['name']
        ),
        'plot_count': len(index.get('plots', [])),
        'phase': index.get('phase', 'PHASE II'),
    }
    
    # Save snapshot
    _ensure_dir(snapshot_path)
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    
    return snapshot


def check_pack_snapshot(pack_dir: str, snapshot_path: str) -> Dict[str, Any]:
    """
    Check a telemetry pack against a saved snapshot.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Compares the current pack's structural/checksum information against
    a previously saved snapshot to detect accidental changes.
    
    Args:
        pack_dir: Path to telemetry pack directory.
        snapshot_path: Path to snapshot JSON to compare against.
    
    Returns:
        Comparison report with 'match' boolean and 'differences' list.
    
    Example:
        >>> result = check_pack_snapshot('artifacts/telemetry', 'snapshots/telemetry_v1.json')
        >>> if not result['match']:
        ...     print("Pack has drifted:", result['differences'])
    """
    # Load current pack snapshot (in-memory, not saved)
    index = load_pack_index(pack_dir)
    current = {
        'baseline_log': index.get('baseline_log', ''),
        'rfl_log': index.get('rfl_log', ''),
        'plots': sorted(
            [
                {
                    'name': p.get('name', ''),
                    'filename': p.get('filename', ''),
                    'checksum': p.get('checksum', ''),
                }
                for p in index.get('plots', [])
            ],
            key=lambda x: x['name']
        ),
        'plot_count': len(index.get('plots', [])),
    }
    
    # Load saved snapshot
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        saved = json.load(f)
    
    # Compare
    differences = []
    
    # Check plot count
    if current['plot_count'] != saved.get('plot_count', 0):
        differences.append({
            'field': 'plot_count',
            'expected': saved.get('plot_count', 0),
            'actual': current['plot_count'],
        })
    
    # Check each plot
    saved_plots = {p['name']: p for p in saved.get('plots', [])}
    current_plots = {p['name']: p for p in current['plots']}
    
    all_plot_names = set(saved_plots.keys()) | set(current_plots.keys())
    
    for name in sorted(all_plot_names):
        saved_plot = saved_plots.get(name)
        current_plot = current_plots.get(name)
        
        if saved_plot is None:
            differences.append({
                'field': f'plot:{name}',
                'issue': 'added',
                'expected': None,
                'actual': current_plot,
            })
        elif current_plot is None:
            differences.append({
                'field': f'plot:{name}',
                'issue': 'removed',
                'expected': saved_plot,
                'actual': None,
            })
        elif saved_plot['checksum'] != current_plot['checksum']:
            differences.append({
                'field': f'plot:{name}:checksum',
                'issue': 'changed',
                'expected': saved_plot['checksum'][:16] + '...',
                'actual': current_plot['checksum'][:16] + '...',
            })
    
    return {
        'match': len(differences) == 0,
        'differences': differences,
        'snapshot_path': snapshot_path,
        'pack_dir': pack_dir,
    }


# -----------------------------------------------------------------------------
# Phase III: Telemetry Governance & Drift Detection
# -----------------------------------------------------------------------------

# =============================================================================
# GOVERNANCE SNAPSHOT CONTRACT
# =============================================================================
#
# A governance snapshot provides a unified view of a telemetry pack's health,
# validity, and drift status. It is ADVISORY only and does not block pipelines.
#
# SCHEMA:
# {
#   "schema_version": "1.0.0",
#   "plot_count": int,
#   "has_valid_checksums": bool,
#   "manifest_hash_present": bool,
#   "snapshot_match": bool | None,
#   "governance_status": "OK" | "WARN" | "BROKEN"
# }
#
# GOVERNANCE STATUS:
# - "OK": All checks pass, no drift
# - "WARN": Minor issues (snapshot drift, non-critical missing fields)
# - "BROKEN": Critical issues (invalid checksums, no plots)
# =============================================================================

GOVERNANCE_SCHEMA_VERSION = "1.0.0"


def build_telemetry_governance_snapshot(
    index: Dict[str, Any],
    health_report: Dict[str, Any],
    snapshot_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a governance snapshot for a telemetry pack.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Combines index, health report, and optional snapshot comparison into
    a unified governance view. This is ADVISORY only and does not block
    any pipelines.
    
    Args:
        index: Pack index dictionary (from load_pack_index()).
        health_report: Health report (from get_pack_health_report()).
        snapshot_result: Optional snapshot comparison (from check_pack_snapshot()).
    
    Returns:
        Governance snapshot dictionary.
    
    Example:
        >>> index = load_pack_index('artifacts/telemetry')
        >>> report = get_pack_health_report(index)
        >>> snapshot = check_pack_snapshot('artifacts/telemetry', 'snapshots/v1.json')
        >>> gov = build_telemetry_governance_snapshot(index, report, snapshot)
        >>> print(gov['governance_status'])
    """
    # Extract health checks
    checks = health_report.get('checks', {})
    has_valid_checksums = checks.get('all_checksums_valid', False)
    manifest_hash_present = checks.get('has_valid_manifest_hash', False)
    has_plots = checks.get('has_plots', False)
    plot_count = checks.get('plot_count', 0)
    
    # Snapshot match (None if no snapshot provided)
    snapshot_match: Optional[bool] = None
    if snapshot_result is not None:
        snapshot_match = snapshot_result.get('match', False)
    
    # Determine governance status
    governance_status = _compute_governance_status(
        has_valid_checksums=has_valid_checksums,
        manifest_hash_present=manifest_hash_present,
        has_plots=has_plots,
        snapshot_match=snapshot_match,
    )
    
    return {
        'schema_version': GOVERNANCE_SCHEMA_VERSION,
        'plot_count': plot_count,
        'has_valid_checksums': has_valid_checksums,
        'manifest_hash_present': manifest_hash_present,
        'snapshot_match': snapshot_match,
        'governance_status': governance_status,
    }


def _compute_governance_status(
    has_valid_checksums: bool,
    manifest_hash_present: bool,
    has_plots: bool,
    snapshot_match: Optional[bool],
) -> str:
    """
    Compute governance status from individual checks.
    
    Returns:
        "OK" | "WARN" | "BROKEN"
    """
    # BROKEN: Critical issues that make the pack unusable
    if not has_plots:
        return "BROKEN"
    if not has_valid_checksums:
        return "BROKEN"
    
    # WARN: Non-critical issues
    if not manifest_hash_present:
        return "WARN"
    if snapshot_match is False:  # Explicit False (not None)
        return "WARN"
    
    # OK: All checks pass
    return "OK"


# =============================================================================
# DRIFT HISTORY CONTRACT
# =============================================================================
#
# Drift history provides a longitudinal view over multiple telemetry pack runs.
#
# SCHEMA:
# {
#   "total_runs": int,
#   "runs_with_broken_packs": int,
#   "runs_with_snapshot_drift": int,
#   "stable_runs": int,
#   "status": "OK" | "ATTENTION" | "UNSTABLE"
# }
#
# STATUS:
# - "OK": All runs stable, no broken packs
# - "ATTENTION": Some drift or warnings detected
# - "UNSTABLE": Multiple broken packs or widespread drift
# =============================================================================


def build_telemetry_drift_history(
    governance_snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build drift history from multiple governance snapshots.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Provides a longitudinal view over many telemetry pack runs to detect
    patterns of drift or instability.
    
    Args:
        governance_snapshots: Sequence of governance snapshot dictionaries.
    
    Returns:
        Drift history dictionary.
    
    Example:
        >>> snapshots = [gov1, gov2, gov3]  # From multiple runs
        >>> history = build_telemetry_drift_history(snapshots)
        >>> if history['status'] == 'UNSTABLE':
        ...     print(f"Warning: {history['runs_with_broken_packs']} broken packs")
    """
    total_runs = len(governance_snapshots)
    
    if total_runs == 0:
        return {
            'total_runs': 0,
            'runs_with_broken_packs': 0,
            'runs_with_snapshot_drift': 0,
            'stable_runs': 0,
            'status': 'OK',
        }
    
    runs_with_broken_packs = 0
    runs_with_snapshot_drift = 0
    stable_runs = 0
    
    for snapshot in governance_snapshots:
        status = snapshot.get('governance_status', 'BROKEN')
        snapshot_match = snapshot.get('snapshot_match')
        
        if status == 'BROKEN':
            runs_with_broken_packs += 1
        elif snapshot_match is False:  # Explicit False
            runs_with_snapshot_drift += 1
        elif status == 'OK':
            stable_runs += 1
        # WARN without drift also counts toward stable_runs
        elif status == 'WARN' and snapshot_match is not False:
            stable_runs += 1
    
    # Determine overall status
    history_status = _compute_history_status(
        total_runs=total_runs,
        runs_with_broken_packs=runs_with_broken_packs,
        runs_with_snapshot_drift=runs_with_snapshot_drift,
    )
    
    return {
        'total_runs': total_runs,
        'runs_with_broken_packs': runs_with_broken_packs,
        'runs_with_snapshot_drift': runs_with_snapshot_drift,
        'stable_runs': stable_runs,
        'status': history_status,
    }


def _compute_history_status(
    total_runs: int,
    runs_with_broken_packs: int,
    runs_with_snapshot_drift: int,
) -> str:
    """
    Compute history status from run counts.
    
    Returns:
        "OK" | "ATTENTION" | "UNSTABLE"
    """
    if total_runs == 0:
        return "OK"
    
    # UNSTABLE: Multiple broken packs or > 50% drift
    if runs_with_broken_packs >= 2:
        return "UNSTABLE"
    if runs_with_snapshot_drift > total_runs // 2:
        return "UNSTABLE"
    
    # ATTENTION: Any broken pack or any drift
    if runs_with_broken_packs > 0:
        return "ATTENTION"
    if runs_with_snapshot_drift > 0:
        return "ATTENTION"
    
    # OK: All stable
    return "OK"


# =============================================================================
# GLOBAL HEALTH TELEMETRY SIGNAL
# =============================================================================
#
# The global health signal summarizes drift history into a simple status
# for integration with global health monitoring systems.
#
# SCHEMA:
# {
#   "telemetry_ok": bool,
#   "broken_pack_count": int,
#   "snapshot_drift_count": int,
#   "status": "OK" | "WARN" | "BLOCK"
# }
#
# STATUS:
# - "OK": All telemetry is healthy
# - "WARN": Some issues detected, investigation recommended
# - "BLOCK": Critical issues that should halt dependent processes
# =============================================================================


def summarize_telemetry_for_global_health(
    history: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize telemetry drift history for global health monitoring.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Produces a simple status signal for integration with global health
    monitoring systems.
    
    Args:
        history: Drift history dictionary (from build_telemetry_drift_history()).
    
    Returns:
        Global health summary dictionary.
    
    Example:
        >>> history = build_telemetry_drift_history(governance_snapshots)
        >>> signal = summarize_telemetry_for_global_health(history)
        >>> if signal['status'] == 'BLOCK':
        ...     raise RuntimeError("Telemetry packs are broken!")
    """
    broken_pack_count = history.get('runs_with_broken_packs', 0)
    snapshot_drift_count = history.get('runs_with_snapshot_drift', 0)
    history_status = history.get('status', 'OK')
    
    # Determine global status
    if history_status == 'UNSTABLE':
        global_status = 'BLOCK'
    elif history_status == 'ATTENTION':
        global_status = 'WARN'
    else:
        global_status = 'OK'
    
    # telemetry_ok is True only if status is OK
    telemetry_ok = (global_status == 'OK')
    
    return {
        'telemetry_ok': telemetry_ok,
        'broken_pack_count': broken_pack_count,
        'snapshot_drift_count': snapshot_drift_count,
        'status': global_status,
    }


def build_full_telemetry_governance_report(
    pack_dir: str,
    snapshot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete governance report for a single telemetry pack.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Convenience function that combines loading, health check, snapshot check,
    and governance snapshot into a single call.
    
    Args:
        pack_dir: Path to telemetry pack directory.
        snapshot_path: Optional path to snapshot JSON for drift detection.
    
    Returns:
        Complete governance report.
    
    Example:
        >>> report = build_full_telemetry_governance_report(
        ...     'artifacts/telemetry',
        ...     snapshot_path='snapshots/v1.json'
        ... )
        >>> print(report['governance']['governance_status'])
    """
    # Load pack index
    index = load_pack_index(pack_dir)
    
    # Get health report
    health_report = get_pack_health_report(index)
    
    # Get snapshot result if path provided
    snapshot_result = None
    if snapshot_path is not None:
        try:
            snapshot_result = check_pack_snapshot(pack_dir, snapshot_path)
        except FileNotFoundError:
            snapshot_result = {'match': None, 'error': 'snapshot not found'}
    
    # Build governance snapshot
    governance = build_telemetry_governance_snapshot(
        index, health_report, snapshot_result
    )
    
    return {
        'pack_dir': pack_dir,
        'index': index,
        'health': health_report,
        'snapshot': snapshot_result,
        'governance': governance,
    }


# =============================================================================
# Phase IV: Telemetry as First-Class Release Signal & Structural Coupler
# =============================================================================

# =============================================================================
# RELEASE READINESS EVALUATION
# =============================================================================
#
# Evaluates telemetry history to determine if a release should proceed.
#
# SCHEMA:
# {
#   "release_ok": bool,
#   "status": "OK" | "WARN" | "BLOCK",
#   "blocking_reasons": list[str]
# }
#
# RULES:
# - BLOCK if telemetry status == BLOCK or history status == UNSTABLE
# - WARN if telemetry status == WARN or any drift
# - OK otherwise
# =============================================================================


def evaluate_telemetry_for_release(
    history: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate telemetry history for release readiness.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Determines if telemetry is healthy enough to proceed with a release.
    This is a FIRST-CLASS RELEASE GATE that should be checked before
    deploying to production.
    
    Args:
        history: Drift history dictionary (from build_telemetry_drift_history()).
    
    Returns:
        Release evaluation dictionary with release_ok, status, and blocking_reasons.
    
    Example:
        >>> history = build_telemetry_drift_history(governance_snapshots)
        >>> eval_result = evaluate_telemetry_for_release(history)
        >>> if not eval_result['release_ok']:
        ...     print(f"Release blocked: {eval_result['blocking_reasons']}")
    """
    # Get global health summary
    health_summary = summarize_telemetry_for_global_health(history)
    
    telemetry_status = health_summary.get('status', 'OK')
    history_status = history.get('status', 'OK')
    broken_pack_count = history.get('runs_with_broken_packs', 0)
    snapshot_drift_count = history.get('runs_with_snapshot_drift', 0)
    
    blocking_reasons = []
    release_ok = True
    status = 'OK'
    
    # BLOCK conditions
    if telemetry_status == 'BLOCK':
        release_ok = False
        status = 'BLOCK'
        blocking_reasons.append('Telemetry global health status is BLOCK')
    
    if history_status == 'UNSTABLE':
        release_ok = False
        status = 'BLOCK'
        blocking_reasons.append('Telemetry history status is UNSTABLE')
    
    # WARN conditions (only if not already BLOCK)
    if status != 'BLOCK':
        if telemetry_status == 'WARN':
            status = 'WARN'
            blocking_reasons.append('Telemetry global health status is WARN')
        
        if snapshot_drift_count > 0:
            status = 'WARN'
            blocking_reasons.append(f'{snapshot_drift_count} runs with snapshot drift detected')
        
        if broken_pack_count > 0:
            status = 'WARN'
            blocking_reasons.append(f'{broken_pack_count} runs with broken packs detected')
    
    # If we have blocking reasons, release is not OK
    if blocking_reasons:
        release_ok = False
    
    return {
        'release_ok': release_ok,
        'status': status,
        'blocking_reasons': blocking_reasons,
    }


# =============================================================================
# TELEMETRY ↔ TOPOLOGY/CURRICULUM STRUCTURAL COUPLING
# =============================================================================
#
# Analyzes the coupling between telemetry, topology analytics, and curriculum
# timeline to detect misalignments.
#
# SCHEMA:
# {
#   "runs_missing_telemetry_for_topology_events": int,
#   "runs_with_topology_warnings_but_no_telemetry_drift": int,
#   "coupling_status": "ALIGNED" | "PARTIAL" | "MISALIGNED"
# }
#
# COUPLING STATUS:
# - "ALIGNED": Telemetry, topology, and curriculum are in sync
# - "PARTIAL": Some misalignments detected but not critical
# - "MISALIGNED": Significant structural misalignments
# =============================================================================


def build_telemetry_structural_coupling_view(
    telemetry_history: Dict[str, Any],
    topology_analytics: Dict[str, Any],
    curriculum_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structural coupling view between telemetry, topology, and curriculum.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Analyzes how well telemetry aligns with topology events and curriculum
    changes to detect structural misalignments that might indicate systemic
    issues.
    
    Args:
        telemetry_history: Drift history dictionary.
        topology_analytics: Topology analytics dictionary. Expected keys:
            - 'runs': List of run dictionaries with 'run_id', 'topology_events', 'warnings'
        curriculum_timeline: Curriculum timeline dictionary. Expected keys:
            - 'runs': List of run dictionaries with 'run_id', 'curriculum_changes'
    
    Returns:
        Coupling view dictionary.
    
    Example:
        >>> coupling = build_telemetry_structural_coupling_view(
        ...     telemetry_history,
        ...     topology_analytics,
        ...     curriculum_timeline
        ... )
        >>> if coupling['coupling_status'] == 'MISALIGNED':
        ...     print("Warning: Structural misalignment detected")
    """
    # Extract run information
    topology_runs = topology_analytics.get('runs', [])
    curriculum_runs = curriculum_timeline.get('runs', [])
    telemetry_total_runs = telemetry_history.get('total_runs', 0)
    
    # Build run ID sets
    topology_run_ids = {run.get('run_id', '') for run in topology_runs}
    curriculum_run_ids = {run.get('run_id', '') for run in curriculum_runs}
    all_run_ids = topology_run_ids | curriculum_run_ids
    
    # Count runs missing telemetry for topology events
    # A run is "missing telemetry" if it has topology events but telemetry_total_runs
    # is insufficient to cover all topology runs with events
    topology_runs_with_events = sum(
        1 for run in topology_runs
        if run.get('topology_events', [])
    )
    # If we have topology events but fewer telemetry runs than topology runs with events
    runs_missing_telemetry = max(0, topology_runs_with_events - telemetry_total_runs)
    
    # Count runs with topology warnings but no telemetry drift
    runs_with_warnings_no_drift = 0
    telemetry_drift_runs = telemetry_history.get('runs_with_snapshot_drift', 0)
    telemetry_broken_runs = telemetry_history.get('runs_with_broken_packs', 0)
    has_telemetry_issues = (telemetry_drift_runs > 0) or (telemetry_broken_runs > 0)
    
    for run in topology_runs:
        warnings = run.get('warnings', [])
        if warnings and not has_telemetry_issues:
            runs_with_warnings_no_drift += 1
    
    # Determine coupling status
    coupling_status = _compute_coupling_status(
        runs_missing_telemetry=runs_missing_telemetry,
        runs_with_warnings_no_drift=runs_with_warnings_no_drift,
        telemetry_total_runs=telemetry_total_runs,
        topology_run_count=len(topology_runs),
    )
    
    return {
        'runs_missing_telemetry_for_topology_events': runs_missing_telemetry,
        'runs_with_topology_warnings_but_no_telemetry_drift': runs_with_warnings_no_drift,
        'coupling_status': coupling_status,
    }


def _compute_coupling_status(
    runs_missing_telemetry: int,
    runs_with_warnings_no_drift: int,
    telemetry_total_runs: int,
    topology_run_count: int,
) -> str:
    """
    Compute coupling status from metrics.
    
    Returns:
        "ALIGNED" | "PARTIAL" | "MISALIGNED"
    """
    # MISALIGNED: Significant structural issues
    if runs_missing_telemetry > topology_run_count // 2:
        return "MISALIGNED"
    if runs_with_warnings_no_drift > telemetry_total_runs // 2:
        return "MISALIGNED"
    
    # PARTIAL: Some misalignments
    if runs_missing_telemetry > 0:
        return "PARTIAL"
    if runs_with_warnings_no_drift > 0:
        return "PARTIAL"
    
    # ALIGNED: All good
    return "ALIGNED"


# =============================================================================
# TELEMETRY DIRECTOR PANEL
# =============================================================================
#
# A unified dashboard view combining telemetry health, release readiness,
# and structural coupling into a single status panel.
#
# SCHEMA:
# {
#   "status_light": "GREEN" | "YELLOW" | "RED",
#   "telemetry_ok": bool,
#   "history_status": str,
#   "coupling_status": str,
#   "headline": str
# }
#
# STATUS LIGHT:
# - GREEN: All systems go
# - YELLOW: Warnings detected, proceed with caution
# - RED: Blocking issues, do not proceed
# =============================================================================


def build_telemetry_director_panel(
    telemetry_health: Dict[str, Any],
    release_eval: Dict[str, Any],
    coupling_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director panel combining all telemetry signals.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Provides a unified view of telemetry health, release readiness, and
    structural coupling for executive decision-making.
    
    Args:
        telemetry_health: Global health summary (from summarize_telemetry_for_global_health()).
        release_eval: Release evaluation (from evaluate_telemetry_for_release()).
        coupling_view: Coupling view (from build_telemetry_structural_coupling_view()).
    
    Returns:
        Director panel dictionary.
    
    Example:
        >>> panel = build_telemetry_director_panel(health, release, coupling)
        >>> print(f"Status: {panel['status_light']}")
        >>> print(f"Headline: {panel['headline']}")
    """
    telemetry_ok = telemetry_health.get('telemetry_ok', False)
    release_ok = release_eval.get('release_ok', False)
    release_status = release_eval.get('status', 'OK')
    history_status = telemetry_health.get('status', 'OK')
    coupling_status = coupling_view.get('coupling_status', 'ALIGNED')
    
    # Determine status light (check status first, then release_ok)
    if release_status == 'BLOCK':
        status_light = 'RED'
    elif release_status == 'WARN' or coupling_status != 'ALIGNED':
        status_light = 'YELLOW'
    elif not release_ok:
        status_light = 'RED'
    else:
        status_light = 'GREEN'
    
    # Build headline
    headline = _build_telemetry_headline(
        telemetry_ok=telemetry_ok,
        release_ok=release_ok,
        release_status=release_status,
        coupling_status=coupling_status,
    )
    
    return {
        'status_light': status_light,
        'telemetry_ok': telemetry_ok,
        'history_status': history_status,
        'coupling_status': coupling_status,
        'headline': headline,
    }


def _build_telemetry_headline(
    telemetry_ok: bool,
    release_ok: bool,
    release_status: str,
    coupling_status: str,
) -> str:
    """
    Build a neutral headline summarizing telemetry readiness and coupling.
    
    Returns:
        Short neutral sentence.
    """
    if not release_ok:
        if release_status == 'BLOCK':
            return "Telemetry indicates release should be blocked due to critical issues."
        else:
            return "Telemetry indicates release warnings; proceed with caution."
    
    if coupling_status == 'MISALIGNED':
        return "Telemetry is healthy but structural misalignment detected with topology/curriculum."
    elif coupling_status == 'PARTIAL':
        return "Telemetry is healthy with partial structural alignment."
    else:
        return "Telemetry is healthy and structurally aligned; release readiness confirmed."


# =============================================================================
# Telemetry Drift Envelope & Uplift Safety Adapter
# =============================================================================

# =============================================================================
# DRIFT ENVELOPE
# =============================================================================
#
# Analyzes checksum drift patterns over time to determine drift severity.
#
# SCHEMA:
# {
#   "drift_band": "LOW" | "MEDIUM" | "HIGH",
#   "plots_with_repeated_drift": list[str],
#   "neutral_notes": list[str]
# }
#
# DRIFT BAND:
# - "LOW": Minimal or no drift detected
# - "MEDIUM": Some drift detected, may need attention
# - "HIGH": Significant repeated drift, blocking concern
# =============================================================================


def build_telemetry_drift_envelope(
    history: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a drift envelope from telemetry history.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Analyzes snapshot comparisons and checksum deltas over time to determine
    drift severity and identify plots with repeated drift patterns.
    
    Args:
        history: Drift history dictionary (from build_telemetry_drift_history()).
    
    Returns:
        Drift envelope dictionary.
    
    Example:
        >>> history = build_telemetry_drift_history(governance_snapshots)
        >>> envelope = build_telemetry_drift_envelope(history)
        >>> if envelope['drift_band'] == 'HIGH':
        ...     print(f"High drift in: {envelope['plots_with_repeated_drift']}")
    """
    total_runs = history.get('total_runs', 0)
    snapshot_drift_count = history.get('runs_with_snapshot_drift', 0)
    broken_pack_count = history.get('runs_with_broken_packs', 0)
    
    # Calculate drift ratio
    if total_runs == 0:
        drift_ratio = 0.0
    else:
        drift_ratio = (snapshot_drift_count + broken_pack_count) / total_runs
    
    # Determine drift band
    if drift_ratio >= 0.5:  # >50% of runs have drift
        drift_band = 'HIGH'
    elif drift_ratio >= 0.2:  # 20-50% have drift
        drift_band = 'MEDIUM'
    else:
        drift_band = 'LOW'
    
    # Identify plots with repeated drift
    # For now, we use a heuristic: if drift is HIGH, assume core plots are affected
    plots_with_repeated_drift = []
    if drift_band == 'HIGH':
        plots_with_repeated_drift = [
            'abstention_heatmap',
            'chain_depth_density',
            'candidate_ordering_entropy',
            'rolling_metric_volatility',
        ]
    elif drift_band == 'MEDIUM' and snapshot_drift_count > 0:
        # In MEDIUM drift, at least some plots are affected
        plots_with_repeated_drift = ['candidate_ordering_entropy', 'rolling_metric_volatility']
    
    # Build neutral notes
    neutral_notes = []
    if total_runs == 0:
        neutral_notes.append("No telemetry history available for drift analysis.")
    elif drift_band == 'LOW':
        neutral_notes.append("Telemetry shows minimal drift; checksums are stable.")
    elif drift_band == 'MEDIUM':
        neutral_notes.append(f"Telemetry shows moderate drift in {snapshot_drift_count} of {total_runs} runs.")
    else:
        neutral_notes.append(f"Telemetry shows significant drift in {snapshot_drift_count + broken_pack_count} of {total_runs} runs.")
    
    if broken_pack_count > 0:
        neutral_notes.append(f"{broken_pack_count} runs had broken packs, indicating structural issues.")
    
    return {
        'drift_band': drift_band,
        'plots_with_repeated_drift': plots_with_repeated_drift,
        'neutral_notes': neutral_notes,
    }


# =============================================================================
# UPLIFT SAFETY ADAPTER
# =============================================================================
#
# Adapts telemetry drift and health signals for uplift safety gating.
#
# SCHEMA:
# {
#   "telemetry_ok_for_uplift": bool,
#   "status": "OK" | "ATTENTION" | "BLOCK",
#   "blocking_reasons": list[str],
#   "advisory_notes": list[str]
# }
#
# STATUS:
# - "OK": Telemetry is safe for uplift analysis
# - "ATTENTION": Some concerns, proceed with caution
# - "BLOCK": Do not use telemetry for uplift analysis
# =============================================================================


def summarize_telemetry_for_uplift_safety(
    drift_envelope: Dict[str, Any],
    pack_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize telemetry for uplift safety gating.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Adapts telemetry drift and health signals to determine if telemetry
    is safe to use for uplift analysis. This is a CRITICAL GATE for
    preventing invalid uplift claims.
    
    Args:
        drift_envelope: Drift envelope (from build_telemetry_drift_envelope()).
        pack_health: Pack health summary (from summarize_telemetry_for_global_health()).
    
    Returns:
        Uplift safety summary dictionary.
    
    Example:
        >>> envelope = build_telemetry_drift_envelope(history)
        >>> health = summarize_telemetry_for_global_health(history)
        >>> safety = summarize_telemetry_for_uplift_safety(envelope, health)
        >>> if not safety['telemetry_ok_for_uplift']:
        ...     raise ValueError("Telemetry not safe for uplift analysis")
    """
    drift_band = drift_envelope.get('drift_band', 'LOW')
    telemetry_ok = pack_health.get('telemetry_ok', False)
    telemetry_status = pack_health.get('status', 'OK')
    
    blocking_reasons = []
    advisory_notes = []
    telemetry_ok_for_uplift = True
    status = 'OK'
    
    # BLOCK conditions
    if drift_band == 'HIGH':
        telemetry_ok_for_uplift = False
        status = 'BLOCK'
        blocking_reasons.append('Telemetry drift band is HIGH; checksums are unstable')
        plots_with_drift = drift_envelope.get('plots_with_repeated_drift', [])
        if plots_with_drift:
            blocking_reasons.append(f'Repeated drift detected in: {", ".join(plots_with_drift)}')
    
    if not telemetry_ok or telemetry_status == 'BLOCK':
        telemetry_ok_for_uplift = False
        status = 'BLOCK'
        blocking_reasons.append('Telemetry pack health is BLOCK; packs are broken or invalid')
    
    # ATTENTION conditions (only if not already BLOCK)
    # Note: ATTENTION does not block uplift, only warns
    if status != 'BLOCK':
        if drift_band == 'MEDIUM':
            status = 'ATTENTION'
            advisory_notes.append('Telemetry drift band is MEDIUM; proceed with caution')
        
        if telemetry_status == 'WARN':
            status = 'ATTENTION'
            advisory_notes.append('Telemetry pack health is WARN; some issues detected')
            # WARN status doesn't block, so telemetry_ok_for_uplift stays True
    
    # Add neutral notes from drift envelope
    advisory_notes.extend(drift_envelope.get('neutral_notes', []))
    
    return {
        'telemetry_ok_for_uplift': telemetry_ok_for_uplift,
        'status': status,
        'blocking_reasons': blocking_reasons,
        'advisory_notes': advisory_notes,
    }


# =============================================================================
# Phase VI: Telemetry Safety Envelope v2 + Uplift Hardening Shield
# =============================================================================

# =============================================================================
# TELEMETRY–TOPOLOGY–SEMANTIC FUSION TILE
# =============================================================================
#
# Fuses telemetry envelope with topology and semantic structures to detect
# incoherence vectors and compute fusion risk.
#
# SCHEMA:
# {
#   "fusion_risk_score": float (0.0-1.0),
#   "incoherence_vectors": list[str],
#   "fusion_band": "LOW" | "MEDIUM" | "HIGH",
#   "neutral_notes": list[str]
# }
#
# FUSION BAND:
# - "LOW": fusion_risk_score < 0.3
# - "MEDIUM": 0.3 <= fusion_risk_score < 0.7
# - "HIGH": fusion_risk_score >= 0.7
# =============================================================================


def build_telemetry_topology_semantic_fusion(
    telemetry_envelope: Dict[str, Any],
    topology_struct: Dict[str, Any],
    semantic_struct: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a fusion tile combining telemetry, topology, and semantic structures.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Analyzes the coherence between telemetry drift patterns, topology events,
    and semantic changes to detect structural misalignments that might indicate
    systemic issues or invalid analysis conditions.
    
    Args:
        telemetry_envelope: Drift envelope (from build_telemetry_drift_envelope()).
        topology_struct: Topology structure dictionary. Expected keys:
            - 'events': List of topology events
            - 'warnings': List of topology warnings
            - 'stability_score': float (0.0-1.0)
        semantic_struct: Semantic structure dictionary. Expected keys:
            - 'changes': List of semantic changes
            - 'coherence_score': float (0.0-1.0)
            - 'drift_indicators': List of drift indicators
    
    Returns:
        Fusion tile dictionary.
    
    Example:
        >>> envelope = build_telemetry_drift_envelope(history)
        >>> fusion = build_telemetry_topology_semantic_fusion(
        ...     envelope, topology_struct, semantic_struct
        ... )
        >>> if fusion['fusion_band'] == 'HIGH':
        ...     print(f"High fusion risk: {fusion['incoherence_vectors']}")
    """
    # Extract drift band from telemetry envelope
    drift_band = telemetry_envelope.get('drift_band', 'LOW')
    drift_band_scores = {'LOW': 0.1, 'MEDIUM': 0.5, 'HIGH': 0.9}
    telemetry_risk = drift_band_scores.get(drift_band, 0.1)
    
    # Extract topology stability
    topology_stability = topology_struct.get('stability_score', 1.0)
    topology_risk = 1.0 - topology_stability  # Invert: low stability = high risk
    
    # Extract semantic coherence
    semantic_coherence = semantic_struct.get('coherence_score', 1.0)
    semantic_risk = 1.0 - semantic_coherence  # Invert: low coherence = high risk
    
    # Compute fusion risk score (weighted average)
    # Telemetry gets 40% weight, topology 30%, semantic 30%
    fusion_risk_score = (
        0.4 * telemetry_risk +
        0.3 * topology_risk +
        0.3 * semantic_risk
    )
    
    # Clamp to [0.0, 1.0]
    fusion_risk_score = max(0.0, min(1.0, fusion_risk_score))
    
    # Determine fusion band
    if fusion_risk_score >= 0.7:
        fusion_band = 'HIGH'
    elif fusion_risk_score >= 0.3:
        fusion_band = 'MEDIUM'
    else:
        fusion_band = 'LOW'
    
    # Identify incoherence vectors
    incoherence_vectors = []
    
    # Telemetry incoherence
    if drift_band == 'HIGH':
        incoherence_vectors.append('telemetry:high_drift_band')
    elif drift_band == 'MEDIUM':
        incoherence_vectors.append('telemetry:moderate_drift_band')
    
    # Topology incoherence
    topology_warnings = topology_struct.get('warnings', [])
    if topology_warnings:
        incoherence_vectors.append(f'topology:{len(topology_warnings)}_warnings')
    
    if topology_stability < 0.5:
        incoherence_vectors.append('topology:low_stability')
    
    # Semantic incoherence
    semantic_changes = semantic_struct.get('changes', [])
    if semantic_changes:
        incoherence_vectors.append(f'semantic:{len(semantic_changes)}_changes')
    
    if semantic_coherence < 0.5:
        incoherence_vectors.append('semantic:low_coherence')
    
    # Cross-domain incoherence
    if drift_band != 'LOW' and topology_stability < 0.7:
        incoherence_vectors.append('cross_domain:telemetry_topology_mismatch')
    
    if drift_band != 'LOW' and semantic_coherence < 0.7:
        incoherence_vectors.append('cross_domain:telemetry_semantic_mismatch')
    
    # Build neutral notes
    neutral_notes = []
    neutral_notes.append(f"Fusion risk score: {fusion_risk_score:.2f} (band: {fusion_band})")
    
    if incoherence_vectors:
        neutral_notes.append(f"Detected {len(incoherence_vectors)} incoherence vector(s)")
    else:
        neutral_notes.append("No incoherence vectors detected; systems appear aligned")
    
    if drift_band != 'LOW':
        neutral_notes.append(f"Telemetry drift band: {drift_band}")
    
    if topology_stability < 0.7:
        neutral_notes.append(f"Topology stability: {topology_stability:.2f}")
    
    if semantic_coherence < 0.7:
        neutral_notes.append(f"Semantic coherence: {semantic_coherence:.2f}")
    
    return {
        'fusion_risk_score': round(fusion_risk_score, 3),
        'incoherence_vectors': sorted(incoherence_vectors),
        'fusion_band': fusion_band,
        'neutral_notes': neutral_notes,
    }


# =============================================================================
# TELEMETRY-DRIVEN UPLIFT PHASE GATE
# =============================================================================
#
# Combines fusion outputs with Phase IV uplift-safety to determine if uplift
# analysis should proceed.
#
# SCHEMA:
# {
#   "uplift_gate_status": "OK" | "ATTENTION" | "BLOCK",
#   "drivers": list[str],
#   "recommended_hold_slices": list[str],
#   "headline": str
# }
#
# GATE STATUS:
# - "OK": Uplift analysis can proceed
# - "ATTENTION": Proceed with caution, review drivers
# - "BLOCK": Do not proceed with uplift analysis
# =============================================================================


def build_telemetry_driven_uplift_phase_gate(
    fusion_tile: Dict[str, Any],
    uplift_safety: Dict[str, Any],
    coupling_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a telemetry-driven uplift phase gate.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Combines fusion tile outputs with Phase IV uplift-safety signals to
    determine if uplift analysis should proceed. This is a CRITICAL GATE
    for protecting uplift analysis integrity.
    
    Args:
        fusion_tile: Fusion tile (from build_telemetry_topology_semantic_fusion()).
        uplift_safety: Uplift safety summary (from summarize_telemetry_for_uplift_safety()).
        coupling_view: Optional coupling view (from build_telemetry_structural_coupling_view()).
    
    Returns:
        Uplift phase gate dictionary.
    
    Example:
        >>> fusion = build_telemetry_topology_semantic_fusion(...)
        >>> safety = summarize_telemetry_for_uplift_safety(...)
        >>> gate = build_telemetry_driven_uplift_phase_gate(fusion, safety)
        >>> if gate['uplift_gate_status'] == 'BLOCK':
        ...     raise ValueError("Uplift analysis blocked by telemetry gate")
    """
    fusion_band = fusion_tile.get('fusion_band', 'LOW')
    fusion_risk = fusion_tile.get('fusion_risk_score', 0.0)
    incoherence_vectors = fusion_tile.get('incoherence_vectors', [])
    
    uplift_status = uplift_safety.get('status', 'OK')
    uplift_blocking = uplift_safety.get('blocking_reasons', [])
    
    coupling_status = coupling_view.get('coupling_status', 'ALIGNED') if coupling_view else 'ALIGNED'
    
    # Determine gate status
    gate_status = 'OK'
    drivers = []
    recommended_hold_slices = []
    
    # BLOCK conditions
    if uplift_status == 'BLOCK':
        gate_status = 'BLOCK'
        drivers.extend(uplift_blocking)
    
    if fusion_band == 'HIGH':
        gate_status = 'BLOCK'
        drivers.append(f'Fusion risk band is HIGH (score: {fusion_risk:.2f})')
        drivers.append(f'Incoherence vectors: {len(incoherence_vectors)} detected')
    
    if coupling_status == 'MISALIGNED':
        gate_status = 'BLOCK'
        drivers.append('Structural coupling status is MISALIGNED')
    
    # ATTENTION conditions (only if not already BLOCK)
    if gate_status != 'BLOCK':
        if uplift_status == 'ATTENTION':
            gate_status = 'ATTENTION'
            drivers.extend(uplift_safety.get('advisory_notes', []))
        
        if fusion_band == 'MEDIUM':
            gate_status = 'ATTENTION'
            drivers.append(f'Fusion risk band is MEDIUM (score: {fusion_risk:.2f})')
        
        if coupling_status == 'PARTIAL':
            gate_status = 'ATTENTION'
            drivers.append('Structural coupling status is PARTIAL')
    
    # Recommend holding slices with high drift
    if fusion_band == 'HIGH':
        # In a real implementation, this would identify specific slices
        # For now, use a placeholder
        recommended_hold_slices = ['slices_with_high_drift']
    
    if incoherence_vectors:
        # Identify slices affected by incoherence
        telemetry_incoherence = [v for v in incoherence_vectors if v.startswith('telemetry:')]
        if telemetry_incoherence:
            recommended_hold_slices.append('slices_with_telemetry_incoherence')
    
    # Build headline
    headline = _build_uplift_gate_headline(
        gate_status=gate_status,
        fusion_band=fusion_band,
        uplift_status=uplift_status,
        coupling_status=coupling_status,
    )
    
    return {
        'uplift_gate_status': gate_status,
        'drivers': drivers,
        'recommended_hold_slices': recommended_hold_slices,
        'headline': headline,
    }


def _build_uplift_gate_headline(
    gate_status: str,
    fusion_band: str,
    uplift_status: str,
    coupling_status: str,
) -> str:
    """
    Build a neutral headline for the uplift phase gate.
    
    Returns:
        Short neutral sentence.
    """
    if gate_status == 'BLOCK':
        if fusion_band == 'HIGH':
            return "Uplift analysis blocked: High fusion risk detected across telemetry, topology, and semantic domains."
        elif uplift_status == 'BLOCK':
            return "Uplift analysis blocked: Telemetry safety checks indicate blocking conditions."
        else:
            return "Uplift analysis blocked: Structural coupling misalignment detected."
    
    elif gate_status == 'ATTENTION':
        return "Uplift analysis may proceed with attention: Some telemetry concerns detected; review drivers before proceeding."
    
    else:
        return "Uplift analysis gate: OK. Telemetry, topology, and semantic structures are aligned and stable."


# =============================================================================
# TELEMETRY DIRECTOR TILE v2
# =============================================================================
#
# Enhanced director tile combining fusion band, uplift gate status, and
# structural coupling state.
#
# SCHEMA:
# {
#   "status_light": "GREEN" | "YELLOW" | "RED",
#   "fusion_band": str,
#   "uplift_gate_status": str,
#   "structural_coupling_state": str,
#   "telemetry_ok": bool,
#   "headline": str
# }
# =============================================================================


def build_telemetry_director_tile_v2(
    fusion_tile: Dict[str, Any],
    uplift_gate: Dict[str, Any],
    coupling_view: Dict[str, Any],
    telemetry_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build enhanced director tile v2 with fusion and uplift gate information.
    
    PHASE II — Descriptive only, not admissible as uplift evidence.
    
    Provides a unified executive view combining fusion analysis, uplift gate
    status, structural coupling, and telemetry health.
    
    Args:
        fusion_tile: Fusion tile (from build_telemetry_topology_semantic_fusion()).
        uplift_gate: Uplift phase gate (from build_telemetry_driven_uplift_phase_gate()).
        coupling_view: Coupling view (from build_telemetry_structural_coupling_view()).
        telemetry_health: Telemetry health (from summarize_telemetry_for_global_health()).
    
    Returns:
        Director tile v2 dictionary.
    
    Example:
        >>> tile = build_telemetry_director_tile_v2(fusion, gate, coupling, health)
        >>> print(f"Status: {tile['status_light']}")
        >>> print(f"Headline: {tile['headline']}")
    """
    fusion_band = fusion_tile.get('fusion_band', 'LOW')
    uplift_gate_status = uplift_gate.get('uplift_gate_status', 'OK')
    coupling_status = coupling_view.get('coupling_status', 'ALIGNED')
    telemetry_ok = telemetry_health.get('telemetry_ok', False)
    
    # Determine status light
    if uplift_gate_status == 'BLOCK' or fusion_band == 'HIGH':
        status_light = 'RED'
    elif uplift_gate_status == 'ATTENTION' or fusion_band == 'MEDIUM' or coupling_status != 'ALIGNED':
        status_light = 'YELLOW'
    else:
        status_light = 'GREEN'
    
    # Build headline
    headline = _build_director_tile_v2_headline(
        status_light=status_light,
        fusion_band=fusion_band,
        uplift_gate_status=uplift_gate_status,
        coupling_status=coupling_status,
        telemetry_ok=telemetry_ok,
    )
    
    return {
        'status_light': status_light,
        'fusion_band': fusion_band,
        'uplift_gate_status': uplift_gate_status,
        'structural_coupling_state': coupling_status,
        'telemetry_ok': telemetry_ok,
        'headline': headline,
    }


def _build_director_tile_v2_headline(
    status_light: str,
    fusion_band: str,
    uplift_gate_status: str,
    coupling_status: str,
    telemetry_ok: bool,
) -> str:
    """
    Build a neutral headline for director tile v2.
    
    Returns:
        Short neutral sentence.
    """
    if status_light == 'RED':
        if fusion_band == 'HIGH':
            return "Telemetry director panel: RED. High fusion risk detected; uplift analysis blocked."
        elif uplift_gate_status == 'BLOCK':
            return "Telemetry director panel: RED. Uplift phase gate blocked; review drivers."
        else:
            return "Telemetry director panel: RED. Critical telemetry issues detected."
    
    elif status_light == 'YELLOW':
        if fusion_band == 'MEDIUM':
            return "Telemetry director panel: YELLOW. Moderate fusion risk; proceed with caution."
        elif uplift_gate_status == 'ATTENTION':
            return "Telemetry director panel: YELLOW. Uplift gate attention; review before proceeding."
        elif coupling_status != 'ALIGNED':
            return "Telemetry director panel: YELLOW. Structural coupling partial or misaligned."
        else:
            return "Telemetry director panel: YELLOW. Some telemetry concerns detected."
    
    else:
        return "Telemetry director panel: GREEN. All systems aligned; telemetry healthy and stable."


# -----------------------------------------------------------------------------
# Telemetry Pack Generator
# -----------------------------------------------------------------------------

def generate_telemetry_pack(
    baseline_path: str,
    rfl_path: str,
    out_dir: str,
    window: int = 20,
    bin_size: int = 20,
    metric_path: str = 'derivation.verified',
) -> Dict[str, Any]:
    """
    Generate a complete developer telemetry pack.
    
    Descriptive only — not admissible as uplift evidence.
    
    This function generates all behavioral telemetry plots into a single
    directory along with a reproducibility manifest. The pack includes:
    
    - abstention_heatmap.png: 2D heatmap of abstention rate vs cycles
    - chain_depth_density.png: Overlapping histograms of chain depths
    - candidate_entropy.png: Rolling entropy trajectory
    - metric_volatility.png: Rolling standard deviation chart
    - telemetry_manifest.json: Reproducibility manifest with hashes
    
    Args:
        baseline_path: Path to baseline JSONL file.
        rfl_path: Path to RFL JSONL file.
        out_dir: Output directory for the pack.
        window: Rolling window size for smoothing.
        bin_size: Bin size for heatmap x-axis.
        metric_path: Metric path for volatility chart.
    
    Returns:
        Pack metadata dictionary with all plot checksums and manifest hash.
    
    Example:
        >>> pack_meta = generate_telemetry_pack(
        ...     'results/fo_baseline.jsonl',
        ...     'results/fo_rfl.jsonl',
        ...     'artifacts/telemetry_pack'
        ... )
        >>> print(pack_meta['manifest_hash'][:16])
    """
    os.makedirs(out_dir, exist_ok=True)
    
    manifest = VisualizationManifest()
    all_metadata = []
    
    # Common parameters for manifest
    common_params = {
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'window': window,
        'bin_size': bin_size,
        'metric_path': metric_path,
    }
    
    # 1. Abstention Heatmap (baseline only)
    heatmap_path = os.path.join(out_dir, 'abstention_heatmap.png')
    heatmap_meta = plot_abstention_heatmap(
        baseline_path, heatmap_path,
        bin_size=bin_size
    )
    all_metadata.append(heatmap_meta)
    manifest.add_entry(heatmap_meta, {
        'records_path': baseline_path,
        'bin_size': bin_size,
    })
    
    # 2. Chain Depth Density
    density_path = os.path.join(out_dir, 'chain_depth_density.png')
    density_meta = plot_chain_depth_density(
        baseline_path, rfl_path, density_path
    )
    all_metadata.append(density_meta)
    manifest.add_entry(density_meta, {
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
    })
    
    # 3. Candidate Ordering Entropy
    entropy_path = os.path.join(out_dir, 'candidate_entropy.png')
    entropy_meta = plot_candidate_ordering_entropy(
        baseline_path, rfl_path, entropy_path,
        window=window
    )
    all_metadata.append(entropy_meta)
    manifest.add_entry(entropy_meta, {
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'window': window,
    })
    
    # 4. Rolling Metric Volatility
    volatility_path = os.path.join(out_dir, 'metric_volatility.png')
    volatility_meta = plot_rolling_metric_volatility(
        baseline_path, rfl_path, volatility_path,
        metric_path=metric_path,
        window=window
    )
    all_metadata.append(volatility_meta)
    manifest.add_entry(volatility_meta, {
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'metric_path': metric_path,
        'window': window,
    })
    
    # Save manifest
    manifest_path = os.path.join(out_dir, 'telemetry_manifest.json')
    manifest_hash = manifest.save(manifest_path)
    
    # Build pack index (developer-facing quick reference)
    # This is a stable, deterministic index for tooling integration
    pack_index = _build_pack_index(
        baseline_path=baseline_path,
        rfl_path=rfl_path,
        all_metadata=all_metadata,
        manifest_hash=manifest_hash,
    )
    
    # Save pack index
    index_path = os.path.join(out_dir, 'pack_index.json')
    _ensure_dir(index_path)
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(pack_index, f, indent=2, sort_keys=True)
    
    # Build pack metadata
    pack_metadata = {
        'pack_type': 'telemetry_pack',
        'output_dir': out_dir,
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'parameters': common_params,
        'plots': [m['plot_type'] for m in all_metadata],
        'plot_checksums': {m['plot_type']: m['checksum_sha256'] for m in all_metadata},
        'manifest_path': manifest_path,
        'manifest_hash': manifest_hash,
        'index_path': index_path,
        'plot_count': len(all_metadata),
        'phase': 'PHASE II',
        'label': PHASE_II_LABEL,
    }
    
    # Save pack summary
    summary_path = os.path.join(out_dir, 'pack_summary.json')
    _ensure_dir(summary_path)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(pack_metadata, f, indent=2, sort_keys=True)
    
    return pack_metadata


def verify_telemetry_pack(pack_dir: str) -> Dict[str, Any]:
    """
    Verify integrity of an existing telemetry pack.
    
    Descriptive only — not admissible as uplift evidence.
    
    Checks that all files exist and checksums match the manifest.
    
    Args:
        pack_dir: Path to telemetry pack directory.
    
    Returns:
        Verification report dictionary.
    """
    manifest_path = os.path.join(pack_dir, 'telemetry_manifest.json')
    
    if not os.path.exists(manifest_path):
        return {
            'valid': False,
            'error': 'telemetry_manifest.json not found',
        }
    
    manifest = VisualizationManifest.load(manifest_path)
    
    report = {
        'valid': True,
        'pack_dir': pack_dir,
        'entry_count': len(manifest.entries),
        'file_checks': [],
    }
    
    for entry in manifest.entries:
        output_path = entry.get('output_path', '')
        expected_hash = entry.get('file_hash_sha256', '')
        
        if not os.path.exists(output_path):
            report['file_checks'].append({
                'path': output_path,
                'status': 'missing',
            })
            report['valid'] = False
            continue
        
        actual_hash = _compute_file_checksum(output_path)
        matches = actual_hash == expected_hash
        
        report['file_checks'].append({
            'path': output_path,
            'status': 'ok' if matches else 'checksum_mismatch',
            'expected': expected_hash[:16] + '...',
            'actual': actual_hash[:16] + '...',
        })
        
        if not matches:
            report['valid'] = False
    
    return report


# -----------------------------------------------------------------------------
# CLI Extension
# -----------------------------------------------------------------------------

def main() -> int:
    """
    CLI entry point for Behavioral Telemetry Visualization Suite.
    
    Descriptive only — not admissible as uplift evidence.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PHASE II — Behavioral Telemetry Visualization Suite. '
                    'Descriptive only — not admissible as uplift evidence.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --pack           Generate complete developer telemetry pack (recommended)
  --plot-type X    Generate specific plot type only
  --verify         Verify an existing telemetry pack

Examples:
  # Generate full telemetry pack (recommended)
  uv run python experiments/behavioral_telemetry_viz.py \\
      --pack --baseline results/fo_baseline.jsonl --rfl results/fo_rfl.jsonl \\
      --out-dir artifacts/telemetry_pack

  # Verify existing pack
  uv run python experiments/behavioral_telemetry_viz.py \\
      --verify --out-dir artifacts/telemetry_pack

  # Snapshot pack for drift detection
  uv run python experiments/behavioral_telemetry_viz.py \\
      --snapshot-pack snapshots/telemetry_v1.json --out-dir artifacts/telemetry_pack

  # Check pack against snapshot
  uv run python experiments/behavioral_telemetry_viz.py \\
      --check-pack snapshots/telemetry_v1.json --out-dir artifacts/telemetry_pack

  # Check pack health
  uv run python experiments/behavioral_telemetry_viz.py \\
      --health --out-dir artifacts/telemetry_pack

  # Generate single plot type
  uv run python experiments/behavioral_telemetry_viz.py \\
      --plot-type entropy --baseline ... --rfl ... --out-dir ...

See docs/TELEMETRY_PLAYBOOK.md for full documentation.
"""
    )
    
    parser.add_argument('--baseline', type=str, help='Path to baseline JSONL')
    parser.add_argument('--rfl', type=str, help='Path to RFL JSONL')
    parser.add_argument('--out-dir', type=str, default='artifacts/behavioral_telemetry',
                       help='Output directory')
    parser.add_argument('--pack', action='store_true',
                       help='Generate complete telemetry pack (all plots + manifest)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify integrity of existing telemetry pack')
    parser.add_argument('--health', action='store_true',
                       help='Check health of existing telemetry pack (advisory)')
    parser.add_argument('--snapshot-pack', type=str, metavar='PATH',
                       help='Save pack snapshot to PATH for drift detection')
    parser.add_argument('--check-pack', type=str, metavar='PATH',
                       help='Check pack against snapshot at PATH (exit 0=match, 1=drift)')
    parser.add_argument('--plot-type', type=str,
                       choices=['all', 'abstention_heatmap', 'chain_depth_density',
                                'entropy', 'volatility'],
                       default='all', help='Plot type to generate')
    parser.add_argument('--window', type=int, default=20, help='Rolling window size')
    parser.add_argument('--bin-size', type=int, default=20, help='Bin size for heatmap')
    parser.add_argument('--metric-path', type=str, default='derivation.verified',
                       help='Metric path for volatility chart')
    parser.add_argument('--generate-manifest', action='store_true',
                       help='Generate reproducibility manifest (implied by --pack)')
    
    args = parser.parse_args()
    
    print("PHASE II — Behavioral Telemetry Visualization Suite")
    print("Descriptive only — not admissible as uplift evidence")
    print("=" * 60)
    
    # Verify mode
    if args.verify:
        print(f"\nVerifying telemetry pack: {args.out_dir}")
        report = verify_telemetry_pack(args.out_dir)
        
        if report['valid']:
            print("✓ Pack verified successfully")
            print(f"  Entries: {report['entry_count']}")
            for check in report['file_checks']:
                print(f"  ✓ {check['path']}: {check['status']}")
        else:
            print("✗ Pack verification FAILED")
            if 'error' in report:
                print(f"  Error: {report['error']}")
            for check in report.get('file_checks', []):
                status = '✓' if check['status'] == 'ok' else '✗'
                print(f"  {status} {check['path']}: {check['status']}")
            return 1
        
        return 0
    
    # Health check mode (advisory)
    if args.health:
        print(f"\nChecking telemetry pack health: {args.out_dir}")
        try:
            index = load_pack_index(args.out_dir)
            report = get_pack_health_report(index)
            
            if report['healthy']:
                print("✓ Pack is healthy (advisory)")
                print(f"  Plot count: {report['checks']['plot_count']}")
                print(f"  Manifest hash valid: {report['checks']['has_valid_manifest_hash']}")
                print(f"  All checksums valid: {report['checks']['all_checksums_valid']}")
            else:
                print("⚠ Pack health check FAILED (advisory)")
                print(f"  Plot count: {report['checks']['plot_count']}")
                print(f"  Has plots: {report['checks']['has_plots']}")
                print(f"  Manifest hash valid: {report['checks']['has_valid_manifest_hash']}")
                print(f"  All checksums valid: {report['checks']['all_checksums_valid']}")
                if report['checks']['invalid_checksum_plots']:
                    print(f"  Invalid checksum plots: {report['checks']['invalid_checksum_plots']}")
            
            print(f"\n{report['advisory']}")
            return 0 if report['healthy'] else 1
            
        except FileNotFoundError as e:
            print(f"✗ Pack not found: {e}")
            return 1
    
    # Snapshot pack mode
    if args.snapshot_pack:
        print(f"\nSaving pack snapshot: {args.out_dir} -> {args.snapshot_pack}")
        try:
            snapshot = save_pack_snapshot(args.out_dir, args.snapshot_pack)
            print("✓ Snapshot saved successfully")
            print(f"  Plots captured: {snapshot['plot_count']}")
            for plot in snapshot['plots']:
                print(f"    - {plot['name']}: {plot['checksum'][:16]}...")
            print(f"  Output: {args.snapshot_pack}")
            return 0
        except FileNotFoundError as e:
            print(f"✗ Pack not found: {e}")
            return 1
    
    # Check pack against snapshot mode
    if args.check_pack:
        print(f"\nChecking pack against snapshot: {args.out_dir} vs {args.check_pack}")
        try:
            result = check_pack_snapshot(args.out_dir, args.check_pack)
            
            if result['match']:
                print("✓ Pack matches snapshot")
                return 0
            else:
                print("✗ Pack has DRIFTED from snapshot")
                print(f"  Differences ({len(result['differences'])}):")
                for diff in result['differences']:
                    field = diff['field']
                    issue = diff.get('issue', 'changed')
                    expected = diff.get('expected', '')
                    actual = diff.get('actual', '')
                    print(f"    - {field}: {issue}")
                    if expected or actual:
                        print(f"      expected: {expected}")
                        print(f"      actual:   {actual}")
                return 1
                
        except FileNotFoundError as e:
            print(f"✗ File not found: {e}")
            return 1
    
    # Pack mode (recommended)
    if args.pack:
        if not args.baseline or not args.rfl:
            print("ERROR: --baseline and --rfl required for --pack mode")
            return 1
        
        print(f"\nGenerating telemetry pack...")
        print(f"  Baseline: {args.baseline}")
        print(f"  RFL: {args.rfl}")
        print(f"  Output: {args.out_dir}")
        print()
        
        pack_meta = generate_telemetry_pack(
            args.baseline, args.rfl, args.out_dir,
            window=args.window,
            bin_size=args.bin_size,
            metric_path=args.metric_path,
        )
        
        print("Generated plots:")
        for plot_type, checksum in pack_meta['plot_checksums'].items():
            print(f"  ✓ {plot_type}: {checksum[:16]}...")
        
        print(f"\nManifest: {pack_meta['manifest_path']}")
        print(f"  Hash: {pack_meta['manifest_hash'][:16]}...")
        
        print(f"\n{'=' * 60}")
        print(f"Telemetry pack complete: {args.out_dir}")
        print(f"Total plots: {pack_meta['plot_count']}")
        print(f"\n{PHASE_II_LABEL}")
        
        return 0
    
    # Individual plot mode (legacy)
    os.makedirs(args.out_dir, exist_ok=True)
    
    manifest = VisualizationManifest()
    all_metadata = []
    
    if args.plot_type in ('all', 'abstention_heatmap') and args.baseline:
        print("\nGenerating abstention heatmap...")
        out_path = os.path.join(args.out_dir, 'abstention_heatmap.png')
        meta = plot_abstention_heatmap(args.baseline, out_path, bin_size=args.bin_size)
        all_metadata.append(meta)
        manifest.add_entry(meta, {'records_path': args.baseline, 'bin_size': args.bin_size})
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type in ('all', 'chain_depth_density') and args.baseline and args.rfl:
        print("\nGenerating chain-depth density field...")
        out_path = os.path.join(args.out_dir, 'chain_depth_density.png')
        meta = plot_chain_depth_density(args.baseline, args.rfl, out_path)
        all_metadata.append(meta)
        manifest.add_entry(meta, {'baseline_path': args.baseline, 'rfl_path': args.rfl})
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type in ('all', 'entropy') and args.baseline and args.rfl:
        print("\nGenerating candidate ordering entropy...")
        out_path = os.path.join(args.out_dir, 'candidate_entropy.png')
        meta = plot_candidate_ordering_entropy(
            args.baseline, args.rfl, out_path, window=args.window
        )
        all_metadata.append(meta)
        manifest.add_entry(meta, {'baseline_path': args.baseline, 'rfl_path': args.rfl,
                                  'window': args.window})
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.plot_type in ('all', 'volatility') and args.baseline and args.rfl:
        print("\nGenerating rolling metric volatility...")
        out_path = os.path.join(args.out_dir, 'metric_volatility.png')
        meta = plot_rolling_metric_volatility(
            args.baseline, args.rfl, out_path,
            metric_path=args.metric_path,
            window=args.window
        )
        all_metadata.append(meta)
        manifest.add_entry(meta, {'baseline_path': args.baseline, 'rfl_path': args.rfl,
                                  'window': args.window, 'metric_path': args.metric_path})
        print(f"  Output: {out_path}")
        print(f"  Checksum: {meta['checksum_sha256'][:16]}...")
    
    if args.generate_manifest or args.pack:
        manifest_path = os.path.join(args.out_dir, 'telemetry_manifest.json')
        manifest_hash = manifest.save(manifest_path)
        print(f"\nManifest saved: {manifest_path}")
        print(f"  Manifest hash: {manifest_hash[:16]}...")
    
    print(f"\n{'=' * 60}")
    print(f"Total plots generated: {len(all_metadata)}")
    print(f"\n{PHASE_II_LABEL}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

