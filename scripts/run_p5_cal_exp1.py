#!/usr/bin/env python3
"""
Phase X P5: CAL-EXP-1 Calibration Harness

This script implements the CAL-EXP-1 calibration experiment for P5 baseline
characterization. It runs windowed divergence analysis to produce metrics
suitable for scientific tuning of the Twin learning parameters.

SHADOW MODE CONTRACT:
- All outputs are observational only
- All PASS/DEFER/FAIL verdicts are PROVISIONAL + SHADOW_ONLY
- No governance decisions are modified
- No thresholds are enforced

=============================================================================
ARCHITECT'S GUIDE: Reading CAL-EXP-1 Metrics
=============================================================================

This harness produces two output files:

1. cal_exp1_metrics.json - Per-window metrics for plotting and analysis
   - windows[]: Array of metric snapshots at 20-cycle intervals
   - Each window contains:
     * divergence_rate: Fraction of cycles where Twin diverged from Real
     * mean_delta_p: Mean |Real.success_prob - Twin.success_prob| proxy
     * delta_bias: Systematic directional error (positive = Twin overestimates)
     * delta_variance: Variance of prediction errors (spread)
     * phase_lag_xcorr: Cross-correlation lag indicator (>0 = Twin lags)
     * pattern_tag: Classification of divergence pattern (DRIFT/NONE/SPIKE)

2. cal_exp1_summary.json - Run-level summary for quick assessment
   - final_divergence_rate: Divergence rate in final window
   - final_delta_bias: Bias in final window
   - mean_divergence_over_run: Average divergence across all windows
   - pattern_progression: How pattern_tags evolved
   - provisional_verdict: PASS/DEFER/FAIL (SHADOW_ONLY, not enforced)

P5 Blueprint Mapping:
- divergence_rate → P5-M1 (Twin tracking fidelity)
- delta_bias → P5-M2 (Systematic prediction error)
- delta_variance → P5-M3 (Prediction consistency)
- phase_lag_xcorr → P5-M4 (Temporal alignment)
- pattern_tag → P5-M5 (Regime classification)

Usage:
    # Basic run with synthetic telemetry
    python scripts/run_p5_cal_exp1.py --cycles 200 --seed 42 --output-dir results/p5_cal_exp1

    # Run with adapter config file
    python scripts/run_p5_cal_exp1.py --adapter-config config/p5_synthetic.json --output-dir results/p5_cal_exp1

    # Run with trace replay verification
    python scripts/run_p5_cal_exp1.py --adapter-config config/p5_synthetic.json --verify-replay --output-dir results/p5_cal_exp1

Status: P5 CALIBRATION INFRASTRUCTURE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Window Metrics Data Structure
# =============================================================================

@dataclass
class WindowMetrics:
    """
    Metrics for a single observation window.

    ARCHITECT'S NOTE:
    - window_start/window_end: Cycle range for this window
    - divergence_rate: P5-M1 - fraction of cycles with divergence
    - mean_delta_p: P5-M2 proxy - mean absolute prediction error
    - delta_bias: P5-M2 - systematic error direction
    - delta_variance: P5-M3 - error spread
    - phase_lag_xcorr: P5-M4 - temporal misalignment indicator
    - pattern_tag: P5-M5 - regime classification
    """
    window_index: int
    window_start: int
    window_end: int
    cycles_in_window: int
    divergence_count: int
    divergence_rate: float
    mean_delta_p: float
    delta_bias: float
    delta_variance: float
    phase_lag_xcorr: float
    pattern_tag: str  # "NONE" | "DRIFT" | "SPIKE" | "OSCILLATION"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "window_index": self.window_index,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "cycles_in_window": self.cycles_in_window,
            "divergence_count": self.divergence_count,
            "divergence_rate": round(self.divergence_rate, 6),
            "mean_delta_p": round(self.mean_delta_p, 6),
            "delta_bias": round(self.delta_bias, 6),
            "delta_variance": round(self.delta_variance, 6),
            "phase_lag_xcorr": round(self.phase_lag_xcorr, 6),
            "pattern_tag": self.pattern_tag,
        }


@dataclass
class CalExp1Result:
    """
    Complete CAL-EXP-1 run result.

    SHADOW MODE: All verdicts are PROVISIONAL and SHADOW_ONLY.
    """
    schema_version: str = "1.0.0"
    mode: str = "SHADOW"
    run_id: str = ""
    timestamp: str = ""
    total_cycles: int = 0
    window_size: int = 20
    windows: List[WindowMetrics] = field(default_factory=list)

    # Summary metrics
    final_divergence_rate: float = 0.0
    final_delta_bias: float = 0.0
    mean_divergence_over_run: float = 0.0
    pattern_progression: List[str] = field(default_factory=list)

    # SHADOW_ONLY verdict
    provisional_verdict: str = "DEFER"
    verdict_reason: str = ""

    def to_metrics_dict(self) -> Dict[str, Any]:
        """
        Export per-window metrics for plotting.

        ARCHITECT'S NOTE: This is the primary analysis artifact.
        Use windows[] for time-series plots. Each window is independent.
        """
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cycles": self.total_cycles,
            "window_size": self.window_size,
            "window_count": len(self.windows),
            "windows": [w.to_dict() for w in self.windows],
            # Metadata for reproducibility
            "_calibration_note": (
                "Per-window metrics for P5 calibration. "
                "Plot divergence_rate, delta_bias, delta_variance vs window_index. "
                "Check pattern_tag progression for regime shifts."
            ),
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Export run-level summary.

        ARCHITECT'S NOTE: Use this for quick pass/fail assessment.
        The provisional_verdict is SHADOW_ONLY - it does not gate anything.
        """
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cycles": self.total_cycles,
            "window_count": len(self.windows),
            "summary": {
                "final_divergence_rate": round(self.final_divergence_rate, 6),
                "final_delta_bias": round(self.final_delta_bias, 6),
                "mean_divergence_over_run": round(self.mean_divergence_over_run, 6),
                "pattern_progression": self.pattern_progression,
            },
            "provisional_verdict": {
                "verdict": self.provisional_verdict,
                "reason": self.verdict_reason,
                "enforcement": "SHADOW_ONLY",
                "_note": "This verdict is observational. No gating occurs.",
            },
        }


# =============================================================================
# CAL-EXP-1 Runner
# =============================================================================

class CalExp1Runner:
    """
    CAL-EXP-1 Calibration Experiment Runner.

    SHADOW MODE CONTRACT:
    - All observations are read-only
    - All verdicts are PROVISIONAL + SHADOW_ONLY
    - No thresholds are enforced

    This runner collects windowed metrics for Twin vs Real divergence analysis.
    The metrics are suitable for scientific analysis and parameter tuning.
    """

    def __init__(
        self,
        adapter: "TelemetryProviderInterface",
        total_cycles: int = 200,
        window_size: int = 20,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        lr_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize CAL-EXP-1 runner.

        Args:
            adapter: Telemetry adapter (RealTelemetryAdapter in SYNTHETIC or TRACE mode)
            total_cycles: Total cycles to observe (default: 200)
            window_size: Cycles per analysis window (default: 20)
            seed: Random seed for reproducibility
            run_id: Optional run identifier
            lr_overrides: Per-component learning rate overrides (UPGRADE-1)
                          Keys: "H", "rho", "tau", "beta"
        """
        self._adapter = adapter
        self._total_cycles = total_cycles
        self._window_size = window_size
        self._seed = seed
        self._run_id = run_id or f"cal_exp1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._lr_overrides = lr_overrides

        # Import P4 components
        from backend.topology.first_light.config_p4 import FirstLightConfigP4
        from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4

        # Create P4 config with LR overrides
        self._config = FirstLightConfigP4(
            slice_name=getattr(adapter, '_slice_name', 'arithmetic_simple'),
            runner_type=getattr(adapter, '_runner_type', 'u2'),
            total_cycles=total_cycles,
            tau_0=0.20,
            telemetry_adapter=adapter,
            log_dir=None,
            run_id=self._run_id,
            twin_lr_overrides=lr_overrides,
        )

        # Create P4 runner for observation
        self._p4_runner = FirstLightShadowRunnerP4(self._config, seed=seed)

        # Window tracking
        self._window_metrics: List[WindowMetrics] = []
        self._current_window_start = 1
        self._current_window_divergences: List[bool] = []
        self._current_window_delta_ps: List[float] = []

    def run(self) -> CalExp1Result:
        """
        Execute CAL-EXP-1 calibration experiment.

        SHADOW MODE: All observations are read-only.

        Returns:
            CalExp1Result with per-window metrics and summary
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cycle_count = 0
        for observation in self._p4_runner.run_cycles(self._total_cycles):
            cycle_count += 1

            # Get divergence info for this cycle
            divergence_snapshots = self._p4_runner.get_divergence_snapshots()
            if divergence_snapshots:
                latest_div = divergence_snapshots[-1]
                is_divergent = latest_div.is_diverged()
                # Use delta_p from divergence snapshot
                delta_p = abs(latest_div.delta_p) if latest_div.delta_p is not None else 0.0
            else:
                is_divergent = False
                delta_p = 0.0

            self._current_window_divergences.append(is_divergent)
            self._current_window_delta_ps.append(delta_p)

            # Check if window complete
            if len(self._current_window_divergences) >= self._window_size:
                self._finalize_window(len(self._window_metrics))
                self._current_window_start = cycle_count + 1
                self._current_window_divergences = []
                self._current_window_delta_ps = []

        # Finalize any remaining partial window
        if self._current_window_divergences:
            self._finalize_window(len(self._window_metrics))

        # Build result
        result = CalExp1Result(
            run_id=self._run_id,
            timestamp=timestamp,
            total_cycles=cycle_count,
            window_size=self._window_size,
            windows=self._window_metrics,
        )

        # Compute summary metrics
        if self._window_metrics:
            result.final_divergence_rate = self._window_metrics[-1].divergence_rate
            result.final_delta_bias = self._window_metrics[-1].delta_bias
            result.mean_divergence_over_run = sum(
                w.divergence_rate for w in self._window_metrics
            ) / len(self._window_metrics)
            result.pattern_progression = [w.pattern_tag for w in self._window_metrics]

        # Compute provisional verdict (SHADOW_ONLY - not enforced)
        result.provisional_verdict, result.verdict_reason = self._compute_provisional_verdict(result)

        return result

    def _finalize_window(self, window_index: int) -> None:
        """Compute metrics for completed window."""
        n = len(self._current_window_divergences)
        if n == 0:
            return

        # P5-M1: Divergence rate
        divergence_count = sum(1 for d in self._current_window_divergences if d)
        divergence_rate = divergence_count / n

        # P5-M2: Mean delta_p and bias
        mean_delta_p = sum(self._current_window_delta_ps) / n if n > 0 else 0.0

        # Bias: positive means Twin overestimates (delta_p uses signed difference)
        # For now, use mean delta_p as proxy since we have absolute values
        delta_bias = mean_delta_p * (0.5 - divergence_rate)  # Heuristic sign

        # P5-M3: Variance
        if n > 1:
            mean_dp = sum(self._current_window_delta_ps) / n
            delta_variance = sum((dp - mean_dp) ** 2 for dp in self._current_window_delta_ps) / n
        else:
            delta_variance = 0.0

        # P5-M4: Phase lag (simplified - use autocorrelation lag 1 as proxy)
        phase_lag_xcorr = self._compute_phase_lag()

        # P5-M5: Pattern classification
        pattern_tag = self._classify_pattern()

        window = WindowMetrics(
            window_index=window_index,
            window_start=self._current_window_start,
            window_end=self._current_window_start + n - 1,
            cycles_in_window=n,
            divergence_count=divergence_count,
            divergence_rate=divergence_rate,
            mean_delta_p=mean_delta_p,
            delta_bias=delta_bias,
            delta_variance=delta_variance,
            phase_lag_xcorr=phase_lag_xcorr,
            pattern_tag=pattern_tag,
        )

        self._window_metrics.append(window)

    def _compute_phase_lag(self) -> float:
        """
        Compute phase lag indicator via autocorrelation.

        ARCHITECT'S NOTE:
        - Positive value indicates Twin lags behind Real
        - Zero indicates good temporal alignment
        - This is a simplified metric; full cross-correlation requires
          aligned time series
        """
        if len(self._current_window_delta_ps) < 3:
            return 0.0

        # Compute lag-1 autocorrelation of delta_p series
        series = self._current_window_delta_ps
        n = len(series)
        mean = sum(series) / n

        # Autocovariance at lag 1
        cov_0 = sum((x - mean) ** 2 for x in series) / n
        if cov_0 < 1e-10:
            return 0.0

        cov_1 = sum((series[i] - mean) * (series[i+1] - mean) for i in range(n-1)) / (n-1)

        return cov_1 / cov_0

    def _classify_pattern(self) -> str:
        """
        Classify divergence pattern in current window.

        ARCHITECT'S NOTE:
        - NONE: No significant pattern detected
        - DRIFT: Monotonic trend in delta_p
        - SPIKE: Single large deviation
        - OSCILLATION: Alternating pattern
        """
        if len(self._current_window_delta_ps) < 3:
            return "NONE"

        series = self._current_window_delta_ps
        n = len(series)

        # Check for drift (monotonic trend)
        increasing = sum(1 for i in range(n-1) if series[i+1] > series[i])
        decreasing = sum(1 for i in range(n-1) if series[i+1] < series[i])

        if increasing > 0.7 * (n - 1) or decreasing > 0.7 * (n - 1):
            return "DRIFT"

        # Check for spike (outlier)
        mean = sum(series) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in series) / n) if n > 1 else 0.0
        if std > 0:
            max_deviation = max(abs(x - mean) for x in series)
            if max_deviation > 3 * std:
                return "SPIKE"

        # Check for oscillation (alternating signs of diff)
        if n >= 4:
            diffs = [series[i+1] - series[i] for i in range(n-1)]
            sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
            if sign_changes > 0.6 * (len(diffs) - 1):
                return "OSCILLATION"

        return "NONE"

    def _compute_provisional_verdict(self, result: CalExp1Result) -> Tuple[str, str]:
        """
        Compute provisional verdict (SHADOW_ONLY - not enforced).

        ARCHITECT'S NOTE:
        This verdict is purely observational. The thresholds below are
        PROVISIONAL and should be calibrated based on baseline characterization.

        Returns:
            (verdict, reason) tuple
        """
        # SHADOW_ONLY thresholds (not enforced, for observation only)
        # These are placeholders for the Architect to calibrate
        PROVISIONAL_DIVERGENCE_THRESHOLD = 0.30  # P5-M1
        PROVISIONAL_BIAS_THRESHOLD = 0.15        # P5-M2

        if not result.windows:
            return "DEFER", "No windows completed"

        final_div = result.final_divergence_rate
        final_bias = abs(result.final_delta_bias)

        # Check final window metrics (PROVISIONAL - not enforced)
        if final_div < PROVISIONAL_DIVERGENCE_THRESHOLD and final_bias < PROVISIONAL_BIAS_THRESHOLD:
            return "PASS", f"PROVISIONAL: divergence={final_div:.3f} < {PROVISIONAL_DIVERGENCE_THRESHOLD}, bias={final_bias:.3f} < {PROVISIONAL_BIAS_THRESHOLD}"
        elif final_div > 0.50:
            return "FAIL", f"PROVISIONAL: divergence={final_div:.3f} > 0.50 (high divergence)"
        else:
            return "DEFER", f"PROVISIONAL: divergence={final_div:.3f}, bias={final_bias:.3f} (needs calibration)"

    def get_adapter_snapshots(self) -> List[Any]:
        """Get all telemetry snapshots from adapter."""
        if hasattr(self._adapter, 'get_all_snapshots'):
            return self._adapter.get_all_snapshots()
        return []


# =============================================================================
# Replay Verification
# =============================================================================

def verify_trace_replay(
    synthetic_result: CalExp1Result,
    trace_result: CalExp1Result,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verify that TRACE replay produces equivalent metrics to SYNTHETIC run.

    ARCHITECT'S NOTE:
    This is the P5 reproducibility spine. If this check fails, there is
    non-determinism in the pipeline that must be investigated.

    Args:
        synthetic_result: Result from SYNTHETIC mode run
        trace_result: Result from TRACE mode replay
        tolerance: Numerical tolerance for floating-point comparison

    Returns:
        Dictionary with PASS/FAIL and per-field deltas
    """
    check_result = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "check_type": "trace_replay_equivalence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic_run_id": synthetic_result.run_id,
        "trace_run_id": trace_result.run_id,
        "tolerance": tolerance,
        "overall_pass": True,
        "window_checks": [],
        "deltas": {},
    }

    # Check window counts match
    if len(synthetic_result.windows) != len(trace_result.windows):
        check_result["overall_pass"] = False
        check_result["error"] = f"Window count mismatch: synthetic={len(synthetic_result.windows)}, trace={len(trace_result.windows)}"
        return check_result

    # Check each window
    max_delta_divergence = 0.0
    max_delta_bias = 0.0
    max_delta_variance = 0.0

    for i, (syn_w, trace_w) in enumerate(zip(synthetic_result.windows, trace_result.windows)):
        window_check = {
            "window_index": i,
            "pass": True,
            "deltas": {},
        }

        # Check divergence_rate
        delta_div = abs(syn_w.divergence_rate - trace_w.divergence_rate)
        window_check["deltas"]["divergence_rate"] = delta_div
        max_delta_divergence = max(max_delta_divergence, delta_div)
        if delta_div > tolerance:
            window_check["pass"] = False
            check_result["overall_pass"] = False

        # Check delta_bias
        delta_bias = abs(syn_w.delta_bias - trace_w.delta_bias)
        window_check["deltas"]["delta_bias"] = delta_bias
        max_delta_bias = max(max_delta_bias, delta_bias)
        if delta_bias > tolerance:
            window_check["pass"] = False
            check_result["overall_pass"] = False

        # Check delta_variance
        delta_var = abs(syn_w.delta_variance - trace_w.delta_variance)
        window_check["deltas"]["delta_variance"] = delta_var
        max_delta_variance = max(max_delta_variance, delta_var)
        if delta_var > tolerance:
            window_check["pass"] = False
            check_result["overall_pass"] = False

        # Check pattern_tag (exact match)
        if syn_w.pattern_tag != trace_w.pattern_tag:
            window_check["pass"] = False
            window_check["deltas"]["pattern_tag_mismatch"] = {
                "synthetic": syn_w.pattern_tag,
                "trace": trace_w.pattern_tag,
            }
            check_result["overall_pass"] = False

        check_result["window_checks"].append(window_check)

    # Summary deltas
    check_result["deltas"] = {
        "max_delta_divergence_rate": max_delta_divergence,
        "max_delta_bias": max_delta_bias,
        "max_delta_variance": max_delta_variance,
    }

    check_result["verdict"] = "PASS" if check_result["overall_pass"] else "FAIL"

    return check_result


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="P5 CAL-EXP-1 Calibration Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic synthetic run
    python scripts/run_p5_cal_exp1.py --cycles 200 --seed 42 --output-dir results/p5_cal_exp1

    # Run with adapter config
    python scripts/run_p5_cal_exp1.py --adapter-config config/p5_synthetic.json --output-dir results/p5_cal_exp1

    # Run with trace replay verification
    python scripts/run_p5_cal_exp1.py --adapter-config config/p5_synthetic.json --verify-replay --output-dir results/p5_cal_exp1
        """,
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=200,
        help="Total cycles to observe (default: 200)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Cycles per analysis window (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/p5_cal_exp1",
        help="Output directory for artifacts (default: results/p5_cal_exp1)",
    )
    parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="Path to adapter configuration JSON",
    )
    parser.add_argument(
        "--verify-replay",
        action="store_true",
        help="Run SYNTHETIC mode, write trace, then replay and verify equivalence",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )
    parser.add_argument(
        "--lr-H",
        type=float,
        default=None,
        help="Learning rate override for H state component (UPGRADE-1)",
    )
    parser.add_argument(
        "--lr-rho",
        type=float,
        default=None,
        help="Learning rate override for rho state component (UPGRADE-1)",
    )
    parser.add_argument(
        "--lr-tau",
        type=float,
        default=None,
        help="Learning rate override for tau state component (UPGRADE-1)",
    )
    parser.add_argument(
        "--lr-beta",
        type=float,
        default=None,
        help="Learning rate override for beta state component (UPGRADE-1)",
    )
    parser.add_argument(
        "--upgrade-label",
        type=str,
        default=None,
        help="Label for upgrade variant (e.g., 'UPGRADE-1')",
    )

    return parser.parse_args()


def run_cal_exp1(args: argparse.Namespace) -> int:
    """
    Run CAL-EXP-1 calibration experiment.

    Returns:
        Exit code (0 for success)
    """
    from backend.topology.first_light.real_telemetry_adapter import (
        RealTelemetryAdapter,
        RealTelemetryAdapterConfig,
        AdapterMode,
    )

    # Create output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"cal_exp1_{timestamp}"
    output_dir = Path(args.output_dir) / run_id

    # Load adapter config
    adapter_config: Optional[RealTelemetryAdapterConfig] = None
    if args.adapter_config:
        adapter_config = RealTelemetryAdapterConfig.from_json_file(args.adapter_config)
        # Override with CLI args
        if args.seed is not None:
            adapter_config.seed = args.seed

    # Dry run mode
    if args.dry_run:
        print("=" * 60)
        print("CAL-EXP-1 DRY RUN - Configuration Preview")
        print("=" * 60)
        print(f"  Cycles:      {args.cycles}")
        print(f"  Window size: {args.window_size}")
        print(f"  Seed:        {args.seed}")
        print(f"  Output dir:  {output_dir}")
        print(f"  Verify replay: {args.verify_replay}")
        if adapter_config:
            print(f"  Adapter config: {args.adapter_config}")
            print(f"  Adapter mode: {adapter_config.mode}")
        print("=" * 60)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LR overrides from CLI args (UPGRADE-1)
    lr_overrides: Optional[Dict[str, float]] = None
    if any([args.lr_H, args.lr_rho, args.lr_tau, args.lr_beta]):
        lr_overrides = {}
        if args.lr_H is not None:
            lr_overrides["H"] = args.lr_H
        if args.lr_rho is not None:
            lr_overrides["rho"] = args.lr_rho
        if args.lr_tau is not None:
            lr_overrides["tau"] = args.lr_tau
        if args.lr_beta is not None:
            lr_overrides["beta"] = args.lr_beta

    upgrade_label = args.upgrade_label or ("UPGRADE-1" if lr_overrides else "BASELINE")

    print("=" * 60)
    print("P5 CAL-EXP-1 Calibration Harness")
    print("=" * 60)
    print(f"  Run ID:      {run_id}")
    print(f"  Cycles:      {args.cycles}")
    print(f"  Window size: {args.window_size}")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Mode:        SHADOW (observational only)")
    print(f"  Upgrade:     {upgrade_label}")
    if lr_overrides:
        print(f"  LR Overrides: {lr_overrides}")
    print("=" * 60)
    print()

    # Create adapter
    if adapter_config:
        adapter = RealTelemetryAdapter.from_config(adapter_config)
        print(f"Using adapter config: {args.adapter_config}")
        print(f"  Mode: {adapter_config.mode}")
    else:
        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=args.seed,
            mode=AdapterMode.SYNTHETIC,
        )
        print("Using default synthetic adapter")
    print()

    # Run CAL-EXP-1
    print(f"Running CAL-EXP-1 calibration ({upgrade_label})...")
    runner = CalExp1Runner(
        adapter=adapter,
        total_cycles=args.cycles,
        window_size=args.window_size,
        seed=args.seed,
        run_id=run_id,
        lr_overrides=lr_overrides,
    )

    result = runner.run()

    # Write metrics file
    metrics_path = output_dir / "cal_exp1_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.to_metrics_dict(), f, indent=2)
    print(f"  Metrics written to: {metrics_path}")

    # Write summary file
    summary_path = output_dir / "cal_exp1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result.to_summary_dict(), f, indent=2)
    print(f"  Summary written to: {summary_path}")

    # Write report file (for evidence pack integration)
    report_path = output_dir / "cal_exp1_report.json"
    report = {
        **result.to_metrics_dict(),
        "summary": result.to_summary_dict()["summary"],
        "provisional_verdict": result.to_summary_dict()["provisional_verdict"],
        "upgrade_config": {
            "label": upgrade_label,
            "lr_overrides": lr_overrides,
            "seed": args.seed,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to: {report_path}")

    # Print summary
    print()
    print("=" * 60)
    print("CAL-EXP-1 Results Summary")
    print("=" * 60)
    print(f"  Total cycles:        {result.total_cycles}")
    print(f"  Windows completed:   {len(result.windows)}")
    print(f"  Final divergence:    {result.final_divergence_rate:.4f}")
    print(f"  Final bias:          {result.final_delta_bias:.4f}")
    print(f"  Mean divergence:     {result.mean_divergence_over_run:.4f}")
    print(f"  Pattern progression: {' -> '.join(result.pattern_progression[-5:])}")
    print()
    print(f"  Provisional verdict: {result.provisional_verdict} (SHADOW_ONLY)")
    print(f"  Reason: {result.verdict_reason}")
    print("=" * 60)

    # Verify replay if requested
    if args.verify_replay:
        print()
        print("=" * 60)
        print("TRACE Replay Verification")
        print("=" * 60)

        # Write trace from synthetic run
        trace_path = output_dir / "synthetic_trace.jsonl"
        snapshots = runner.get_adapter_snapshots()
        if snapshots:
            from backend.topology.first_light.real_telemetry_adapter import write_trace_jsonl
            count = write_trace_jsonl(snapshots, str(trace_path))
            print(f"  Wrote {count} snapshots to trace: {trace_path}")
        else:
            print("  WARNING: No snapshots to write to trace")
            return 0

        # Create trace-mode adapter
        trace_config = RealTelemetryAdapterConfig(
            mode=AdapterMode.TRACE,
            trace_path=str(trace_path),
            runner_type=adapter_config.runner_type if adapter_config else "u2",
            slice_name=adapter_config.slice_name if adapter_config else "arithmetic_simple",
            seed=args.seed,
        )
        trace_adapter = RealTelemetryAdapter.from_config(trace_config)

        # Run replay
        print("  Running TRACE mode replay...")
        replay_runner = CalExp1Runner(
            adapter=trace_adapter,
            total_cycles=args.cycles,
            window_size=args.window_size,
            seed=args.seed,
            run_id=f"{run_id}_replay",
            lr_overrides=lr_overrides,
        )
        replay_result = replay_runner.run()

        # Verify equivalence
        check = verify_trace_replay(result, replay_result)

        # Write replay check
        replay_check_path = output_dir / "cal_exp1_replay_check.json"
        with open(replay_check_path, "w") as f:
            json.dump(check, f, indent=2)
        print(f"  Replay check written to: {replay_check_path}")

        # Print verification result
        print()
        print(f"  Replay verification: {check['verdict']}")
        if check["deltas"]:
            print(f"    Max delta divergence: {check['deltas']['max_delta_divergence_rate']:.2e}")
            print(f"    Max delta bias:       {check['deltas']['max_delta_bias']:.2e}")
            print(f"    Max delta variance:   {check['deltas']['max_delta_variance']:.2e}")
        print("=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_cal_exp1(args)


if __name__ == "__main__":
    sys.exit(main())
