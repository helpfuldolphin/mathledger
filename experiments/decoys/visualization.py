# PHASE II — NOT USED IN PHASE I
"""
Decoy Landscape Visualization

Provides visualization tools for analyzing the difficulty-confusability
landscape of decoy formulas. Generates scatterplots showing how formulas
are distributed across the difficulty/confusability space.

Usage:
    from experiments.decoys.visualization import plot_decoy_landscape

    # Generate and save plot
    plot_decoy_landscape("slice_uplift_goal")
    
    # Custom output path
    plot_decoy_landscape("slice_uplift_goal", output_path="my_plot.png")

Output:
    PNG files are written to artifacts/phase_ii/decoys/ by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .confusability import ConfusabilityMap, ConfusabilityMapReport, get_all_confusability_maps


# Default output directory
DEFAULT_OUTPUT_DIR = Path("artifacts/phase_ii/decoys")

# Role colors for the plot
ROLE_COLORS = {
    "target": "#2ecc71",      # Green
    "decoy_near": "#e74c3c",  # Red
    "decoy_far": "#3498db",   # Blue
    "bridge": "#9b59b6",      # Purple
}

ROLE_MARKERS = {
    "target": "^",      # Triangle up
    "decoy_near": "o",  # Circle
    "decoy_far": "s",   # Square
    "bridge": "d",      # Diamond
}


def _ensure_output_dir(output_dir: Path) -> None:
    """Ensure the output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_decoy_landscape(
    slice_name: str,
    config_path: str = "config/curriculum_uplift_phase2.yaml",
    output_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
) -> str:
    """
    Generate a scatterplot of the decoy difficulty-confusability landscape.
    
    Args:
        slice_name: Name of the slice to visualize
        config_path: Path to curriculum YAML file
        output_path: Custom output path (default: artifacts/phase_ii/decoys/<slice>.png)
        show: Whether to display the plot interactively
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for output
        
    Returns:
        Path to the generated PNG file
        
    Scatterplot:
        - x-axis: Difficulty score (0-1)
        - y-axis: Confusability score (0-1)
        - Color: Role (target/decoy_near/decoy_far/bridge)
        - Marker: Role-specific shape
        - Labels: Formula names
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for server use
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
    
    # Generate confusability map
    cmap = ConfusabilityMap(slice_name, config_path)
    report = cmap.generate_report()
    
    # Prepare data
    data_by_role: Dict[str, List[Tuple[float, float, str]]] = {
        role: [] for role in ROLE_COLORS.keys()
    }
    
    for f in report.formulas:
        if f.role in data_by_role:
            data_by_role[f.role].append((f.difficulty, f.confusability, f.name))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each role
    for role, data in data_by_role.items():
        if not data:
            continue
        
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        names = [d[2] for d in data]
        
        ax.scatter(
            xs, ys,
            c=ROLE_COLORS[role],
            marker=ROLE_MARKERS[role],
            s=100,
            label=role.replace('_', ' ').title(),
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5,
        )
        
        # Add labels for each point
        for x, y, name in zip(xs, ys, names):
            # Truncate long names
            display_name = name[:12] + "..." if len(name) > 15 else name
            ax.annotate(
                display_name,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.7,
            )
    
    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Quadrant labels
    ax.text(0.25, 0.75, 'Low Diff\nHigh Conf', ha='center', va='center', 
            fontsize=8, alpha=0.3, style='italic')
    ax.text(0.75, 0.75, 'High Diff\nHigh Conf', ha='center', va='center', 
            fontsize=8, alpha=0.3, style='italic')
    ax.text(0.25, 0.25, 'Low Diff\nLow Conf', ha='center', va='center', 
            fontsize=8, alpha=0.3, style='italic')
    ax.text(0.75, 0.25, 'High Diff\nLow Conf', ha='center', va='center', 
            fontsize=8, alpha=0.3, style='italic')
    
    # Labels and title
    ax.set_xlabel('Difficulty Score', fontsize=11)
    ax.set_ylabel('Confusability Score', fontsize=11)
    ax.set_title(f'Decoy Landscape: {slice_name}', fontsize=13, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(
        loc='upper left',
        framealpha=0.9,
        fontsize=9,
    )
    
    # Statistics annotation
    stats_text = (
        f"Near-Far Gap: {report.near_far_gap:.3f}\n"
        f"Coverage: {report.coverage_score:.3f}\n"
        f"Formulas: {len(report.formulas)}"
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        _ensure_output_dir(DEFAULT_OUTPUT_DIR)
        output_path = str(DEFAULT_OUTPUT_DIR / f"{slice_name}_landscape.png")
    else:
        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    
    plt.close(fig)
    
    return output_path


def plot_all_landscapes(
    config_path: str = "config/curriculum_uplift_phase2.yaml",
    output_dir: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Generate landscape plots for all uplift slices.
    
    Args:
        config_path: Path to curriculum YAML file
        output_dir: Custom output directory (default: artifacts/phase_ii/decoys/)
        **kwargs: Additional arguments passed to plot_decoy_landscape
        
    Returns:
        List of paths to generated PNG files
    """
    reports = get_all_confusability_maps(config_path)
    
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = DEFAULT_OUTPUT_DIR
    
    _ensure_output_dir(out_path)
    
    generated = []
    for slice_name in reports.keys():
        output_file = str(out_path / f"{slice_name}_landscape.png")
        try:
            path = plot_decoy_landscape(
                slice_name,
                config_path=config_path,
                output_path=output_file,
                **kwargs,
            )
            generated.append(path)
        except Exception as e:
            print(f"Warning: Failed to generate plot for {slice_name}: {e}")
    
    return generated


def plot_comparison_landscape(
    slice_names: List[str],
    config_path: str = "config/curriculum_uplift_phase2.yaml",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
) -> str:
    """
    Generate a comparison plot showing multiple slices side by side.
    
    Args:
        slice_names: List of slice names to compare
        config_path: Path to curriculum YAML file
        output_path: Custom output path
        figsize: Figure size
        dpi: Output resolution
        
    Returns:
        Path to generated PNG file
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")
    
    n_slices = len(slice_names)
    if n_slices == 0:
        raise ValueError("No slices to compare")
    
    # Determine grid layout
    if n_slices <= 2:
        rows, cols = 1, n_slices
    elif n_slices <= 4:
        rows, cols = 2, 2
    else:
        cols = 3
        rows = (n_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_slices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, slice_name in enumerate(slice_names):
        ax = axes[idx]
        
        try:
            cmap = ConfusabilityMap(slice_name, config_path)
            report = cmap.generate_report()
            
            # Plot data
            for f in report.formulas:
                if f.role in ROLE_COLORS:
                    ax.scatter(
                        f.difficulty, f.confusability,
                        c=ROLE_COLORS[f.role],
                        marker=ROLE_MARKERS[f.role],
                        s=60,
                        alpha=0.8,
                    )
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(slice_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Difficulty', fontsize=8)
            ax.set_ylabel('Confusability', fontsize=8)
            ax.grid(True, alpha=0.2)
            
            # Stats
            ax.text(
                0.98, 0.02,
                f"Gap: {report.near_far_gap:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                ha='right', va='bottom',
            )
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax.set_title(f"{slice_name} (error)", fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_slices, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend to figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=ROLE_MARKERS[role], color='w', 
               markerfacecolor=ROLE_COLORS[role], markersize=10, 
               label=role.replace('_', ' ').title())
        for role in ROLE_COLORS.keys()
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.98),
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    if output_path is None:
        _ensure_output_dir(DEFAULT_OUTPUT_DIR)
        output_path = str(DEFAULT_OUTPUT_DIR / "comparison_landscape.png")
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path


def generate_landscape_report(
    config_path: str = "config/curriculum_uplift_phase2.yaml",
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate a complete landscape report with all visualizations.
    
    Args:
        config_path: Path to curriculum YAML file
        output_dir: Output directory for files
        
    Returns:
        Dictionary mapping slice names to their generated file paths
    """
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = DEFAULT_OUTPUT_DIR
    
    _ensure_output_dir(out_path)
    
    result = {}
    
    # Individual landscapes
    reports = get_all_confusability_maps(config_path)
    slice_names = list(reports.keys())
    
    for slice_name in slice_names:
        try:
            path = plot_decoy_landscape(
                slice_name,
                config_path=config_path,
                output_path=str(out_path / f"{slice_name}_landscape.png"),
            )
            result[slice_name] = path
        except Exception as e:
            result[slice_name] = f"Error: {e}"
    
    # Comparison landscape
    if len(slice_names) >= 2:
        try:
            comparison_path = plot_comparison_landscape(
                slice_names,
                config_path=config_path,
                output_path=str(out_path / "comparison_landscape.png"),
            )
            result["_comparison"] = comparison_path
        except Exception as e:
            result["_comparison"] = f"Error: {e}"
    
    return result

