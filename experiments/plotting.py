"""
Plotting Toolkit for MathLedger Experiments.

Provides consistent styling and helper functions for generating
investor-grade and paper-grade figures.

Design Principles:
- Minimalist: Black & white friendly, high contrast.
- Type-safe: Uses typed data structures (ExperimentResult).
- Reproducible: Figures are generated from saved JSON artifacts.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
import pandas as pd
from rfl.experiment import ExperimentResult

# -----------------------------------------------------------------------------
# Style Configuration
# -----------------------------------------------------------------------------

def setup_style() -> None:
    """
    Apply consistent matplotlib style settings for publication-quality figures.
    
    Sets high-DPI, monochrome-friendly matplotlib rcParams with readable fonts
    and clear axis labeling for Dyno Chart and other visualizations.
    """
    # Use a clean, minimal style base
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        # Fallback if style not available
        plt.style.use('default')
    
    # High-DPI, monochrome-friendly overrides
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'grid.alpha': 0.2,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'savefig.format': 'png',
        # Monochrome-friendly: ensure good contrast
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

# -----------------------------------------------------------------------------
# IO / Utils
# -----------------------------------------------------------------------------

def ensure_output_dir(path: str = "artifacts/figures") -> None:
    """Ensure the output directory exists."""
    os.makedirs(path, exist_ok=True)

def save_figure(name: str, fig: Optional[plt.Figure] = None) -> str:
    """
    Save the current (or passed) figure to artifacts/figures.
    Saves both PNG and PDF formats when possible.
    Returns the absolute path of the PNG file.
    """
    ensure_output_dir()
    
    # Save PNG
    png_filename = f"artifacts/figures/{name}.png"
    if fig:
        fig.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {png_filename}")
    
    # Save PDF if possible
    try:
        pdf_filename = f"artifacts/figures/{name}.pdf"
        if fig:
            fig.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
        print(f"Saved figure: {pdf_filename}")
    except Exception as e:
        # PDF backend may not be available, that's okay
        print(f"Note: Could not save PDF format: {e}")
    
    if fig:
        plt.close(fig)
    else:
        plt.close()
    
    return png_filename

# -----------------------------------------------------------------------------
# Plotting Primitives
# -----------------------------------------------------------------------------

def plot_abstention_dynamics_from_results(
    results: List[ExperimentResult],
    window_size: int = 5,
    title: str = "Abstention Rate Dynamics"
) -> plt.Figure:
    """
    Figure 1: Evolution of Abstention Rate over Runs (Epochs).
    
    Narrative:
        Shows the system 'learning' when to abstain. We expect this to 
        stabilize or converge to an optimal H_t.
    """
    runs = [r.run_id for r in results]
    # Assuming run_id is sortable or we use index
    x = np.arange(len(runs)) + 1
    y = [r.abstention_rate for r in results]
    
    fig, ax = plt.subplots()
    
    # Raw data (light)
    ax.plot(x, y, 'o', color='#cccccc', alpha=0.5, label='Raw Run')
    
    # Moving average (dark)
    if len(y) >= window_size:
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        x_smooth = x[window_size-1:]
        ax.plot(x_smooth, y_smooth, '-', color='black', linewidth=2, label=f'{window_size}-Run Avg')
    
    ax.set_xlabel('Experiment Run (Epoch)')
    ax.set_ylabel('Abstention Rate')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    
    # Annotate phases if possible (heuristic based on curriculum)
    # simplistic assumption: first 20% warmup
    warmup_end = int(len(runs) * 0.2)
    if warmup_end > 0:
        ax.axvline(warmup_end, linestyle='--', color='gray', alpha=0.5)
        ax.text(warmup_end + 0.5, 0.05, 'Warmup', rotation=90, verticalalignment='bottom', color='gray')
        
    return fig

def plot_throughput_vs_depth(
    results: List[ExperimentResult],
    title: str = "System Throughput vs. Reasoning Depth"
) -> plt.Figure:
    """
    Figure 2: Throughput (Proofs/hr) as a function of Mean Depth.
    
    Narrative:
        Demonstrates efficiency. Ideally, throughput degrades gracefully (linearly)
        rather than exponentially as depth increases.
    """
    depths = [r.mean_depth for r in results]
    throughputs = [r.throughput_proofs_per_hour for r in results]
    
    fig, ax = plt.subplots()
    
    ax.scatter(depths, throughputs, c='black', alpha=0.7, s=30)
    
    ax.set_xlabel('Mean Reasoning Depth')
    ax.set_ylabel('Throughput (Proofs / Hour)')
    ax.set_title(title)
    
    return fig

def plot_capability_frontier(
    results: List[ExperimentResult],
    metric: str = "success_rate",
    title: str = "Capability Frontier"
) -> plt.Figure:
    """
    Figure 3: Success Rate vs Complexity (Depth * Breadth approx).
    
    Narrative:
        Defines the operational envelope of the system.
    """
    # Complexity proxy: max_depth * log(max_breadth) or similar.
    # Here we use max_depth for simplicity as per prompt request for "simple capability".
    
    # Group by depth to get mean success rate
    from collections import defaultdict
    depth_groups = defaultdict(list)
    
    for r in results:
        # Bin by integer max_depth
        d = int(r.max_depth)
        if d > 0:
            depth_groups[d].append(r.success_rate)
            
    x = sorted(depth_groups.keys())
    y = [np.mean(depth_groups[d]) for d in x]
    y_err = [np.std(depth_groups[d]) for d in x]
    
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y, yerr=y_err, fmt='-o', color='black', capsize=5)
    
    ax.set_xlabel('Max Reasoning Depth')
    ax.set_ylabel('Success Rate')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    
    return fig

def plot_rfl_comparison_bar(
    rfl_on_results: List[ExperimentResult],
    rfl_off_results: List[ExperimentResult],
    title: str = "Impact of Reflexive Feedback (RFL)"
) -> plt.Figure:
    """
    Figure 4: Comparative Bar Chart (RFL On vs Off).
    
    Narrative:
        Head-to-head comparison showing improvements in efficiency or accuracy.
    """
    # Metrics to compare
    metrics = ['Success Rate', 'Abstention Rate', 'Throughput']
    
    # Helper to average
    def avg(res, attr):
        vals = [getattr(r, attr) for r in res]
        return np.mean(vals) if vals else 0
        
    # Pre-calc means
    on_means = [
        avg(rfl_on_results, 'success_rate'),
        avg(rfl_on_results, 'abstention_rate'),
        # Normalize throughput relative to OFF for same scale? 
        # Or just use twin axis. Let's keep it simple: just rates first.
    ]
    
    off_means = [
        avg(rfl_off_results, 'success_rate'),
        avg(rfl_off_results, 'abstention_rate'),
    ]
    
    # We'll just plot the rates (0-1 scale)
    labels = ['Success Rate', 'Abstention Rate']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    
    ax.bar(x - width/2, [on_means[0], on_means[1]], width, label='RFL ON', color='black')
    ax.bar(x + width/2, [off_means[0], off_means[1]], width, label='RFL OFF', color='gray')
    
    ax.set_ylabel('Rate (0-1)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    return fig

def plot_knowledge_growth(
    results: List[ExperimentResult],
    title: str = "Knowledge Base Growth"
) -> plt.Figure:
    """
    Figure 5: Cumulative Unique Statements Proved.
    
    Narrative:
        Shows the 'Data Moat' accumulating over time.
    """
    runs = np.arange(len(results)) + 1
    
    # Simulation of diminishing returns (discovery curve)
    # Total gathered so far
    cumulative_list = []
    current_total = 0
    
    for i, r in enumerate(results):
        current_total += r.distinct_statements
        cumulative_list.append(current_total)
    
    cumulative = np.array(cumulative_list)
    
    fig, ax = plt.subplots()
    
    ax.plot(runs, cumulative, '-', color='black', linewidth=2)
    ax.fill_between(runs, cumulative, color='#eeeeee', alpha=0.5)
    
    ax.set_xlabel('Experiment Run')
    ax.set_ylabel('Cumulative Unique Statements')
    ax.set_title(title)
    
    return fig

# -----------------------------------------------------------------------------
# Dyno Chart: Wide Slice Abstention Dynamics
# -----------------------------------------------------------------------------

def plot_abstention_dynamics(
    df_baseline: pd.DataFrame,
    df_rfl: pd.DataFrame,
    window: int = 100,
    slice_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot Baseline vs RFL abstention rate over cycles (Dyno Chart).
    
    Args:
        df_baseline: DataFrame with columns ['cycle', 'is_abstention', 'run_type']
        df_rfl: DataFrame with columns ['cycle', 'is_abstention', 'run_type']
        window: Rolling window size for computing rolling mean
        slice_name: Optional slice name to include in title (e.g., "Wide Slice", "Default Slice")
    
    Returns:
        matplotlib Figure with Baseline vs RFL comparison
    """
    fig, ax = plt.subplots()
    
    # Calculate rolling mean abstention rate for baseline
    if len(df_baseline) > 0:
        df_baseline_sorted = df_baseline.sort_values('cycle')
        baseline_rolling = df_baseline_sorted['is_abstention'].rolling(window=window, min_periods=1).mean()
        ax.plot(
            df_baseline_sorted['cycle'],
            baseline_rolling,
            label='Baseline (RFL off)',
            color='#666666',
            linewidth=2.0,
            linestyle='--'
        )
    
    # Calculate rolling mean abstention rate for RFL
    if len(df_rfl) > 0:
        df_rfl_sorted = df_rfl.sort_values('cycle')
        rfl_rolling = df_rfl_sorted['is_abstention'].rolling(window=window, min_periods=1).mean()
        ax.plot(
            df_rfl_sorted['cycle'],
            rfl_rolling,
            label='RFL Enabled',
            color='black',
            linewidth=2.0,
            linestyle='-'
        )
    
    # Build title with optional slice name
    if slice_name:
        title = f'RFL Uplift: Abstention Rate vs Cycles ({slice_name})'
    else:
        title = 'RFL Uplift: Abstention Rate vs Cycles'
    
    # Labeling and narrative
    ax.set_xlabel('Cycle Index', fontsize=12)
    ax.set_ylabel(f'Rolling Abstention Rate (window={window})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set y-axis to 0-1 range
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    return fig

def make_dyno_chart(
    baseline_path: str,
    rfl_path: str,
    window: int = 100,
    return_metadata: bool = False
) -> plt.Figure:
    """
    Create Dyno Chart comparing Baseline vs RFL abstention dynamics.
    
    Args:
        baseline_path: Path to baseline JSONL file (e.g., results/fo_baseline_wide.jsonl)
        rfl_path: Path to RFL JSONL file (e.g., results/fo_rfl_wide.jsonl)
        window: Rolling window size for computing rolling mean
        return_metadata: If True, returns tuple (fig, metadata_dict) instead of just fig
    
    Returns:
        matplotlib Figure with the Dyno Chart, or (Figure, metadata_dict) if return_metadata=True
    """
    import json
    
    # Load JSONL files
    baseline_records = []
    rfl_records = []
    
    # Read baseline file
    with open(baseline_path, 'r') as f:
        for line in f:
            if line.strip():
                baseline_records.append(json.loads(line))
    
    # Read RFL file
    with open(rfl_path, 'r') as f:
        for line in f:
            if line.strip():
                rfl_records.append(json.loads(line))
    
    # Convert to DataFrames
    df_baseline_raw = pd.DataFrame(baseline_records)
    df_rfl_raw = pd.DataFrame(rfl_records)
    
    # Normalize is_abstention column
    # Try multiple ways to determine abstention:
    # 1. Direct 'abstention' field (boolean)
    # 2. 'status' == "abstain"
    # 3. 'method' == "lean-disabled" or similar
    # 4. 'derivation.abstained' > 0
    
    def normalize_abstention(row):
        # Check direct abstention field
        if 'abstention' in row and pd.notna(row['abstention']):
            return bool(row['abstention'])
        
        # Check status field
        if 'status' in row and pd.notna(row['status']):
            if row['status'] == 'abstain':
                return True
            if row['status'] == 'verified':
                return False
        
        # Check method field
        if 'method' in row and pd.notna(row['method']):
            method = str(row['method']).lower()
            if 'lean-disabled' in method or 'abstain' in method:
                return True
        
        # Check derivation.abstained
        if isinstance(row.get('derivation'), dict):
            derivation = row['derivation']
            if 'abstained' in derivation:
                return int(derivation.get('abstained', 0)) > 0
        
        # Default to False if we can't determine
        return False
    
    # Apply normalization
    df_baseline_raw['is_abstention'] = df_baseline_raw.apply(normalize_abstention, axis=1)
    df_rfl_raw['is_abstention'] = df_rfl_raw.apply(normalize_abstention, axis=1)
    
    # Ensure cycle column exists (use index if not present)
    if 'cycle' not in df_baseline_raw.columns:
        df_baseline_raw['cycle'] = df_baseline_raw.index
    if 'cycle' not in df_rfl_raw.columns:
        df_rfl_raw['cycle'] = df_rfl_raw.index
    
    # Add run_type column
    df_baseline_raw['run_type'] = 'Baseline'
    df_rfl_raw['run_type'] = 'RFL'
    
    # Select only required columns
    df_baseline = df_baseline_raw[['cycle', 'is_abstention', 'run_type']].copy()
    df_rfl = df_rfl_raw[['cycle', 'is_abstention', 'run_type']].copy()
    
    # Sort by cycle
    df_baseline = df_baseline.sort_values('cycle')
    df_rfl = df_rfl.sort_values('cycle')
    
    # Detect slice name from logs (for title)
    slice_name = None
    sample_baseline = baseline_records[0] if baseline_records else {}
    sample_rfl = rfl_records[0] if rfl_records else {}
    
    # Try to get slice_name from either log
    if 'slice_name' in sample_baseline:
        slice_name = sample_baseline['slice_name']
    elif 'slice_name' in sample_rfl:
        slice_name = sample_rfl['slice_name']
    
    # Map slice_name to display name
    if slice_name == 'slice_medium':
        slice_display = 'Wide Slice'
    elif slice_name:
        slice_display = slice_name.replace('-', ' ').title()
    else:
        # Infer from filename
        if 'wide' in baseline_path.lower():
            slice_display = 'Wide Slice'
        else:
            slice_display = 'Default Slice'
    
    # Collect metadata for provenance manifest
    metadata = {
        'baseline_path': baseline_path,
        'rfl_path': rfl_path,
        'baseline_cycles': len(df_baseline),
        'rfl_cycles': len(df_rfl),
        'window_size': window,
        'slice_name': slice_name if slice_name else 'inferred',
        'slice_display': slice_display,
        'baseline_cycle_range': [int(df_baseline['cycle'].min()), int(df_baseline['cycle'].max())] if len(df_baseline) > 0 else None,
        'rfl_cycle_range': [int(df_rfl['cycle'].min()), int(df_rfl['cycle'].max())] if len(df_rfl) > 0 else None,
    }
    
    # Create the plot
    setup_style()
    fig = plot_abstention_dynamics(df_baseline, df_rfl, window=window, slice_name=slice_display)
    
    if return_metadata:
        return fig, metadata
    return fig

if __name__ == "__main__":
    # Quick test
    setup_style()
    print("Style setup complete.")