"""
Evidence Curve Visualizer for RFL Results

Generates empirical evidence plots:
- Coverage over runs with bootstrap CI
- Uplift over runs with bootstrap CI
- Convergence analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/server environments
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .experiment import ExperimentResult
from .coverage import CoverageMetrics
from .bootstrap_stats import BootstrapResult


class RFLVisualizer:
    """Generates evidence curves for RFL experiment results."""

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize

        # Style configuration
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'coverage': '#2E86AB',
            'uplift': '#A23B72',
            'threshold': '#F18F01',
            'ci': '#C73E1D',
            'success': '#06A77D'
        }

    def generate_full_report(
        self,
        run_results: List[ExperimentResult],
        coverage_metrics: List[CoverageMetrics],
        coverage_ci: BootstrapResult,
        uplift_ci: BootstrapResult,
        output_path: str,
        coverage_threshold: float = 0.92,
        uplift_threshold: float = 1.0
    ) -> None:
        """
        Generate comprehensive RFL evidence report with multiple panels.

        Args:
            run_results: List of experiment results
            coverage_metrics: Per-run coverage metrics
            coverage_ci: Bootstrap CI for coverage
            uplift_ci: Bootstrap CI for uplift
            output_path: Path to save figure
            coverage_threshold: Coverage acceptance threshold
            uplift_threshold: Uplift acceptance threshold
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('RFL Reflexive Metabolism Evidence Report', fontsize=16, fontweight='bold')

        # Extract successful runs only
        successful_runs = [r for r in run_results if r.status == "success"]
        run_indices = np.arange(1, len(successful_runs) + 1)

        # Panel 1: Coverage over runs
        self._plot_coverage_over_runs(
            axes[0, 0],
            coverage_metrics,
            coverage_ci,
            coverage_threshold
        )

        # Panel 2: Throughput over runs (for uplift visualization)
        self._plot_throughput_over_runs(
            axes[0, 1],
            successful_runs,
            uplift_ci
        )

        # Panel 3: Success rate over runs
        self._plot_success_rate_over_runs(
            axes[0, 2],
            successful_runs
        )

        # Panel 4: Novelty rate over runs
        self._plot_novelty_over_runs(
            axes[1, 0],
            coverage_metrics
        )

        # Panel 5: Mean depth over runs
        self._plot_depth_over_runs(
            axes[1, 1],
            successful_runs
        )

        # Panel 6: Bootstrap CI summary
        self._plot_ci_summary(
            axes[1, 2],
            coverage_ci,
            uplift_ci,
            coverage_threshold,
            uplift_threshold
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evidence curves saved to {output_path}")

    def _plot_coverage_over_runs(
        self,
        ax: plt.Axes,
        coverage_metrics: List[CoverageMetrics],
        coverage_ci: BootstrapResult,
        threshold: float
    ) -> None:
        """Plot coverage rate over runs with bootstrap CI."""
        run_indices = np.arange(1, len(coverage_metrics) + 1)
        coverage_rates = [m.coverage_rate for m in coverage_metrics]

        # Plot coverage trajectory
        ax.plot(
            run_indices,
            coverage_rates,
            'o-',
            color=self.colors['coverage'],
            linewidth=2,
            markersize=4,
            label='Coverage rate',
            alpha=0.7
        )

        # Add moving average (window=5)
        if len(coverage_rates) >= 5:
            window = 5
            moving_avg = np.convolve(
                coverage_rates,
                np.ones(window) / window,
                mode='valid'
            )
            ma_indices = run_indices[window-1:]
            ax.plot(
                ma_indices,
                moving_avg,
                '--',
                color=self.colors['coverage'],
                linewidth=2,
                label=f'MA({window})',
                alpha=0.9
            )

        # Bootstrap CI horizontal band
        ax.axhline(
            coverage_ci.point_estimate,
            color=self.colors['ci'],
            linestyle='-',
            linewidth=2,
            label=f'Mean: {coverage_ci.point_estimate:.4f}'
        )
        ax.axhspan(
            coverage_ci.ci_lower,
            coverage_ci.ci_upper,
            alpha=0.2,
            color=self.colors['ci'],
            label=f'95% CI: [{coverage_ci.ci_lower:.4f}, {coverage_ci.ci_upper:.4f}]'
        )

        # Threshold line
        ax.axhline(
            threshold,
            color=self.colors['threshold'],
            linestyle='--',
            linewidth=2,
            label=f'Threshold: {threshold}'
        )

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Coverage Rate')
        ax.set_title('Coverage Over Runs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    def _plot_throughput_over_runs(
        self,
        ax: plt.Axes,
        run_results: List[ExperimentResult],
        uplift_ci: BootstrapResult
    ) -> None:
        """Plot throughput over runs with uplift visualization."""
        run_indices = np.arange(1, len(run_results) + 1)
        throughput = [r.throughput_proofs_per_hour for r in run_results]

        # Split into baseline (first half) and treatment (second half)
        n = len(run_results)
        baseline_idx = run_indices[:n//2]
        treatment_idx = run_indices[n//2:]
        baseline_thr = throughput[:n//2]
        treatment_thr = throughput[n//2:]

        # Plot trajectories
        ax.plot(
            baseline_idx,
            baseline_thr,
            'o-',
            color='gray',
            linewidth=2,
            markersize=4,
            label='Baseline (early runs)',
            alpha=0.7
        )
        ax.plot(
            treatment_idx,
            treatment_thr,
            'o-',
            color=self.colors['uplift'],
            linewidth=2,
            markersize=4,
            label='Treatment (later runs)',
            alpha=0.7
        )

        # Mean lines
        ax.axhline(
            np.mean(baseline_thr),
            color='gray',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label=f'Baseline mean: {np.mean(baseline_thr):.1f}'
        )
        ax.axhline(
            np.mean(treatment_thr),
            color=self.colors['uplift'],
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label=f'Treatment mean: {np.mean(treatment_thr):.1f}'
        )

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Proofs per Hour')
        ax.set_title(f'Throughput Over Runs (Uplift: {uplift_ci.point_estimate:.2f}x)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_success_rate_over_runs(
        self,
        ax: plt.Axes,
        run_results: List[ExperimentResult]
    ) -> None:
        """Plot success rate over runs."""
        run_indices = np.arange(1, len(run_results) + 1)
        success_rates = [r.success_rate for r in run_results]

        ax.plot(
            run_indices,
            success_rates,
            'o-',
            color=self.colors['success'],
            linewidth=2,
            markersize=4,
            alpha=0.7
        )

        # Mean line
        mean_success = np.mean(success_rates)
        ax.axhline(
            mean_success,
            color=self.colors['success'],
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_success:.2%}'
        )

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Success Rate')
        ax.set_title('Verification Success Rate Over Runs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    def _plot_novelty_over_runs(
        self,
        ax: plt.Axes,
        coverage_metrics: List[CoverageMetrics]
    ) -> None:
        """Plot novelty rate over runs."""
        run_indices = np.arange(1, len(coverage_metrics) + 1)
        novelty_rates = [m.novelty_rate for m in coverage_metrics]

        ax.plot(
            run_indices,
            novelty_rates,
            'o-',
            color='#F39C12',
            linewidth=2,
            markersize=4,
            alpha=0.7
        )

        # Mean line
        mean_novelty = np.mean(novelty_rates)
        ax.axhline(
            mean_novelty,
            color='#F39C12',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_novelty:.2%}'
        )

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Novelty Rate')
        ax.set_title('Statement Novelty Over Runs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    def _plot_depth_over_runs(
        self,
        ax: plt.Axes,
        run_results: List[ExperimentResult]
    ) -> None:
        """Plot mean depth over runs."""
        run_indices = np.arange(1, len(run_results) + 1)
        mean_depths = [r.mean_depth for r in run_results]

        ax.plot(
            run_indices,
            mean_depths,
            'o-',
            color='#8E44AD',
            linewidth=2,
            markersize=4,
            alpha=0.7
        )

        # Mean line
        overall_mean = np.mean(mean_depths)
        ax.axhline(
            overall_mean,
            color='#8E44AD',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {overall_mean:.2f}'
        )

        ax.set_xlabel('Run Number')
        ax.set_ylabel('Mean Proof Depth')
        ax.set_title('Mean Proof Depth Over Runs')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_ci_summary(
        self,
        ax: plt.Axes,
        coverage_ci: BootstrapResult,
        uplift_ci: BootstrapResult,
        coverage_threshold: float,
        uplift_threshold: float
    ) -> None:
        """Plot bootstrap CI summary with acceptance criteria."""
        # Data
        metrics = ['Coverage', 'Uplift']
        point_estimates = [coverage_ci.point_estimate, uplift_ci.point_estimate]
        ci_lowers = [coverage_ci.ci_lower, uplift_ci.ci_lower]
        ci_uppers = [coverage_ci.ci_upper, uplift_ci.ci_upper]
        thresholds = [coverage_threshold, uplift_threshold]

        y_positions = np.arange(len(metrics))

        # Plot error bars (CIs)
        for i, (metric, pe, lo, hi, thresh) in enumerate(
            zip(metrics, point_estimates, ci_lowers, ci_uppers, thresholds)
        ):
            # CI bar
            ax.barh(
                i,
                hi - lo,
                left=lo,
                height=0.3,
                color=self.colors['ci'],
                alpha=0.3,
                label='95% CI' if i == 0 else None
            )

            # Point estimate
            ax.plot(
                pe,
                i,
                'o',
                color=self.colors['ci'],
                markersize=10,
                label='Point estimate' if i == 0 else None
            )

            # Threshold line
            ax.axvline(
                thresh,
                color=self.colors['threshold'],
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label='Threshold' if i == 0 else None
            )

            # Pass/fail indicator
            passed = lo >= thresh if metric == 'Coverage' else lo > thresh
            status = '✓ PASS' if passed else '✗ FAIL'
            color = 'green' if passed else 'red'

            ax.text(
                pe,
                i + 0.15,
                f'{pe:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
            ax.text(
                hi + 0.05,
                i,
                status,
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold',
                color=color
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Metric Value')
        ax.set_title('Bootstrap CI Summary')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')


def generate_quick_summary(
    results_json_path: str,
    output_path: str
) -> None:
    """
    Generate evidence curves from saved JSON results.

    Args:
        results_json_path: Path to rfl_results.json
        output_path: Path to save figure
    """
    import json
    from .experiment import ExperimentResult
    from .coverage import CoverageMetrics
    from .bootstrap_stats import BootstrapResult

    # Load results
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Reconstruct objects
    run_results = [
        ExperimentResult(**r) for r in data['runs']
    ]

    coverage_metrics = [
        CoverageMetrics(**m) for m in data['coverage']['per_run']
    ]

    coverage_ci = BootstrapResult(**data['coverage']['bootstrap_ci'])
    uplift_ci = BootstrapResult(**data['uplift']['bootstrap_ci'])

    # Generate visualization
    visualizer = RFLVisualizer()
    visualizer.generate_full_report(
        run_results,
        coverage_metrics,
        coverage_ci,
        uplift_ci,
        output_path,
        coverage_threshold=data['config']['coverage_threshold'],
        uplift_threshold=data['config']['uplift_threshold']
    )


if __name__ == "__main__":
    # Example: Generate curves from saved results
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m backend.rfl.visualizer <results.json> <output.png>")
        sys.exit(1)

    generate_quick_summary(sys.argv[1], sys.argv[2])
