#!/usr/bin/env python3
"""
CAL-EXP-1 Runtime Stability Metrics Harness (200-cycle snippet)

Logs runtime profile stability metrics during CAL-EXP-1 execution for P5 calibration
drift detection. This harness integrates with the existing CAL-EXP-1 warm-start
harness and emits runtime stability metrics alongside divergence metrics.

SHADOW MODE: All metrics are observational only; no governance decisions are made.

Usage:
    python scripts/first_light_cal_exp1_runtime_stability.py \
        --cycles 200 \
        --seed 42 \
        --output-dir results/cal_exp1_runtime_stability
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from experiments.u2.runtime import (
    build_runtime_health_snapshot,
    evaluate_runtime_profile,
    load_runtime_profile,
    reset_feature_flags,
    set_feature_flag,
    SYNTHETIC_FEATURE_FLAGS,
)
from scripts.first_light_cal_exp1_warm_start import (
    build_adapter,
    parse_args as parse_cal_exp1_args,
    run_cal_exp1,
)


def log_runtime_stability_metrics(
    cycle: int,
    profile_name: str,
    output_dir: Path,
) -> Dict[str, any]:
    """
    Log runtime stability metrics for a given cycle.

    SHADOW MODE: This function is purely observational and does not gate execution.

    Args:
        cycle: Current cycle number
        profile_name: Runtime profile name to evaluate against
        output_dir: Directory to write metrics JSONL

    Returns:
        Dict with runtime stability metrics for this cycle
    """
    # Build runtime health snapshot
    health_snapshot = build_runtime_health_snapshot(config_path=None)

    # Load profile and evaluate
    profile = load_runtime_profile(profile_name)
    profile_eval = evaluate_runtime_profile(
        profile=profile,
        snapshot=health_snapshot,
        policy_result={"policy_ok": True, "violations": []},
    )

    # Extract key metrics
    metrics = {
        "cycle": cycle,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "profile_ok": profile_eval.get("profile_ok", False),
        "status": profile_eval.get("status", "UNKNOWN"),
        "violations_count": len(profile_eval.get("violations", [])),
        "active_flags": health_snapshot.get("active_flags", {}),
        "flag_stabilities": health_snapshot.get("flag_stabilities", {}),
        "config_valid": health_snapshot.get("config_valid", False),
    }

    # Write to JSONL file
    metrics_file = output_dir / "runtime_stability_metrics.jsonl"
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    return metrics


def run_cal_exp1_with_runtime_stability(
    cycles: int = 200,
    seed: int = 42,
    profile_name: str = "prod-hardened",
    output_dir: Optional[Path] = None,
    **cal_exp1_kwargs,
) -> Path:
    """
    Run CAL-EXP-1 with runtime stability metrics logging.

    This function wraps the standard CAL-EXP-1 harness and adds runtime profile
    stability metric logging at each cycle.

    SHADOW MODE: All metrics are observational only.

    Args:
        cycles: Number of cycles to run (default: 200)
        seed: Random seed (default: 42)
        profile_name: Runtime profile to evaluate against (default: prod-hardened)
        output_dir: Output directory for metrics (default: auto-generated)
        **cal_exp1_kwargs: Additional arguments passed to CAL-EXP-1 harness

    Returns:
        Path to the runtime stability metrics summary JSON file
    """
    if output_dir is None:
        output_dir = Path(f"results/cal_exp1_runtime_stability_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset feature flags to known state
    reset_feature_flags()

    # Log initial runtime stability
    initial_metrics = log_runtime_stability_metrics(0, profile_name, output_dir)

    # Run CAL-EXP-1 harness (this will execute cycles internally)
    # Note: In a real implementation, we would need to hook into the runner's
    # cycle callback to log metrics at each cycle. For this snippet, we log
    # at start and end, and provide a structure for cycle-by-cycle logging.
    cal_exp1_output_dir = output_dir / "cal_exp1"
    
    # Create args namespace for CAL-EXP-1 harness
    cal_exp1_args = argparse.Namespace(
        adapter=cal_exp1_kwargs.get("adapter", "real"),
        cycles=cycles,
        learning_rate=cal_exp1_kwargs.get("learning_rate", 0.1),
        seed=seed,
        output_dir=str(cal_exp1_output_dir),
        adapter_config=cal_exp1_kwargs.get("adapter_config"),
        runner_type=cal_exp1_kwargs.get("runner_type", "u2"),
        slice=cal_exp1_kwargs.get("slice", "arithmetic_simple"),
    )
    cal_exp1_report_path = run_cal_exp1(cal_exp1_args)

    # Log final runtime stability
    final_metrics = log_runtime_stability_metrics(cycles, profile_name, output_dir)

    # Build summary
    summary = {
        "schema_version": "1.0.0",
        "experiment": "CAL-EXP-1",
        "profile": profile_name,
        "cycles": cycles,
        "seed": seed,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "cal_exp1_report_path": str(cal_exp1_report_path),
        "runtime_stability_metrics_path": str(output_dir / "runtime_stability_metrics.jsonl"),
        "mode": "SHADOW",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write summary
    summary_path = output_dir / "runtime_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Runtime stability metrics written to: {summary_path}")
    return summary_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CAL-EXP-1 Runtime Stability Metrics Harness (200-cycle snippet)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=200,
        help="Number of cycles to run (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="prod-hardened",
        help="Runtime profile name to evaluate against (default: prod-hardened)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for metrics (default: auto-generated)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        choices=("real", "mock"),
        default="real",
        help="Telemetry adapter type (default: real)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Twin learning rate override (default: 0.1)",
    )
    parser.add_argument(
        "--runner-type",
        type=str,
        choices=("u2", "rfl"),
        default="u2",
        help="Runner type hint (default: u2)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="arithmetic_simple",
        help="Slice identifier (default: arithmetic_simple)",
    )

    args = parser.parse_args()

    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_cal_exp1_with_runtime_stability(
            cycles=args.cycles,
            seed=args.seed,
            profile_name=args.profile,
            output_dir=output_dir,
            adapter=args.adapter,
            learning_rate=args.learning_rate,
            runner_type=args.runner_type,
            slice=args.slice,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

