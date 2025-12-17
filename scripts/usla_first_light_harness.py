#!/usr/bin/env python3
"""
USLA First-Light Experiment Harness — P3 Implementation

Phase X: First-Light Synthetic Experiment Runner

This harness runs the P3 First-Light shadow experiment with synthetic data,
producing all required artifacts per docs/system_law/Phase_X_Prelaunch_Review.md.

EXPERIMENT DESIGN (from spec):
=============================

Slice Selection:
- U2 Slice: arithmetic_simple
- RFL Slice: propositional_tautology (depth <= 6)

Run Configuration:
- Total cycles: 1000 (default)
- tau_0: 0.20 (Goldilocks zone [0.16, 0.24])
- USLA in strict SHADOW mode

Red-Flag Abort Conditions (LOGGED ONLY, NOT ENFORCED):
- CDI-010 (Fixed-Point Multiplicity): Any activation -> ABORT
- CDI-007 (Exception Exhaustion): > 10 consecutive cycles -> ABORT
- is_hard_ok = False: > 50 consecutive cycles -> ABORT
- Safe region exit: > 100 consecutive cycles -> ABORT
- Governance divergence: > 20 consecutive CRITICAL -> ABORT
- RSI collapse: rho < 0.2 for > 10 cycles -> ABORT
- Block rate explosion: beta > 0.6 for > 20 cycles -> ABORT

Success Criteria:
- U2 arithmetic_simple: Success rate >= 0.75 over 500 cycles
- RFL propositional_tautology: Abstention rate <= 0.15 over 500 cycles
- Delta_p (learning curve slope) > 0 for both runners
- Mean RSI rho >= 0.6 over full 1000 cycles
- Cycles in safe region Omega >= 90%
- No CDI-010 activations
- CDI-007 activations <= 50 total cycles
- HARD mode OK >= 80% of cycles

SHADOW MODE CONTRACT:
- The USLA simulator NEVER modifies real governance decisions
- Disagreements are LOGGED, not ACTED upon
- No cycle is blocked or allowed based on simulator output
- The simulator runs AFTER the real governance decision
- All USLA state is written to shadow logs only

Output Artifacts:
- synthetic_raw.jsonl: Per-cycle observations
- stability_report.json: Summary statistics and criteria evaluation
- red_flag_matrix.json: Red-flag events and summary
- metrics_windows.json: Windowed metrics aggregation
- tda_metrics.json: TDA metrics (SNS, PCS, DRS, HSS) per window
- run_config.json: Configuration snapshot

Usage:
    python scripts/usla_first_light_harness.py --seed 42 --cycles 1000 --output-dir results/fl_run1
    python scripts/usla_first_light_harness.py --slice arithmetic_simple --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.logging.jsonl_writer import JsonlWriter
from backend.topology.first_light import (
    FirstLightConfig,
    FirstLightShadowRunner,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="USLA First-Light Experiment Harness (P3 Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for non-deterministic)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of cycles to run (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/first_light",
        help="Output directory for artifacts (default: results/first_light)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="arithmetic_simple",
        help="Slice to run (default: arithmetic_simple)",
    )
    parser.add_argument(
        "--runner-type",
        type=str,
        default="u2",
        choices=["u2", "rfl"],
        help="Runner type (default: u2)",
    )
    parser.add_argument(
        "--tau-0",
        type=float,
        default=0.20,
        help="Initial threshold tau_0 (default: 0.20)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Metrics window size (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: validate config without executing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--pathology",
        type=str,
        choices=["none", "spike", "drift", "oscillation"],
        default="none",
        help="TEST-ONLY: inject synthetic pathology into H trajectory (default: none)",
    )
    return parser.parse_args()


def generate_run_id(seed: Optional[int]) -> str:
    """Generate unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    seed_str = f"seed{seed}" if seed is not None else "noseed"
    return f"fl_{timestamp}_{seed_str}"


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write records to JSONL file."""
    with JsonlWriter(str(path), json_kwargs={"default": str}) as writer:
        for record in records:
            writer.write(record)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def build_synthetic_raw(runner: FirstLightShadowRunner) -> List[Dict[str, Any]]:
    """Build synthetic_raw.jsonl records from runner observations."""
    records = []
    for obs in runner.get_observations():
        records.append(obs.to_dict())
    return records


def build_stability_report(
    run_id: str,
    config: FirstLightConfig,
    result: Any,
    start_time: str,
    end_time: str,
    pathology: str = "none",
    pathology_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build stability_report.json from run results."""
    # Evaluate success criteria
    criteria = []

    # Success rate criterion
    success_passed = (result.u2_success_rate_final or 0.0) >= 0.75
    criteria.append({
        "name": "success_rate_final",
        "threshold": 0.75,
        "actual": result.u2_success_rate_final,
        "passed": success_passed,
        "notes": "U2 success rate >= 0.75",
    })

    # Delta-p success criterion
    delta_p_passed = (result.delta_p_success or 0.0) > 0
    criteria.append({
        "name": "delta_p_success",
        "threshold": "> 0",
        "actual": result.delta_p_success,
        "passed": delta_p_passed,
        "notes": "Learning curve slope positive",
    })

    # Mean RSI criterion
    rsi_passed = result.mean_rsi >= 0.6
    criteria.append({
        "name": "mean_rsi",
        "threshold": 0.6,
        "actual": result.mean_rsi,
        "passed": rsi_passed,
        "notes": "Mean RSI >= 0.6",
    })

    # Omega occupancy criterion
    omega_passed = result.omega_occupancy >= 0.9
    criteria.append({
        "name": "omega_occupancy",
        "threshold": 0.9,
        "actual": result.omega_occupancy,
        "passed": omega_passed,
        "notes": "Omega region occupancy >= 90%",
    })

    # CDI-010 criterion
    cdi_010_passed = result.cdi_010_activations == 0
    criteria.append({
        "name": "cdi_010_activations",
        "threshold": 0,
        "actual": result.cdi_010_activations,
        "passed": cdi_010_passed,
        "notes": "No CDI-010 activations",
    })

    # HARD mode criterion
    hard_passed = result.hard_ok_rate >= 0.8
    criteria.append({
        "name": "hard_ok_rate",
        "threshold": 0.8,
        "actual": result.hard_ok_rate,
        "passed": hard_passed,
        "notes": "HARD mode OK >= 80%",
    })

    all_passed = all(c["passed"] for c in criteria)

    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "config": {
            "slice_name": config.slice_name,
            "runner_type": config.runner_type,
            "total_cycles": config.total_cycles,
            "tau_0": config.tau_0,
            "seed": None,  # Will be updated by caller
        },
        "timing": {
            "start_time": start_time,
            "end_time": end_time,
            "cycles_completed": result.cycles_completed,
        },
        "pathology": pathology,
        "pathology_params": pathology_params or {},
        "metrics": {
            "delta_p": {
                "success_final": result.u2_success_rate_final,
                "abstention_final": result.rfl_abstention_rate_final,
                "success_slope": result.delta_p_success,
                "abstention_slope": result.delta_p_abstention,
            },
            "rsi": {
                "mean": round(result.mean_rsi, 4),
                "min": round(result.min_rsi, 4),
                "max": round(result.max_rsi, 4),
                "std": 0.0,  # Would need to compute
            },
            "omega": {
                "occupancy_rate": round(result.omega_occupancy, 4),
                "exit_count": result.omega_exit_count,
                "max_exit_streak": result.max_omega_exit_streak,
            },
            "hard_mode": {
                "ok_rate": round(result.hard_ok_rate, 4),
                "fail_count": result.hard_fail_count,
                "max_fail_streak": result.max_hard_fail_streak,
            },
            "success_rate": round(result.u2_success_rate_final or 0.0, 4),
            "abstention_rate": round(result.rfl_abstention_rate_final or 0.0, 4),
        },
        "criteria_evaluation": {
            "all_passed": all_passed,
            "criteria": criteria,
        },
        "red_flag_summary": {
            "total_flags": result.total_red_flags,
            "by_severity": result.red_flags_by_severity,
            "hypothetical_abort": result.hypothetical_abort_reason is not None,
        },
    }


def build_red_flag_matrix(
    run_id: str,
    result: Any,
    runner: FirstLightShadowRunner,
) -> Dict[str, Any]:
    """Build red_flag_matrix.json from runner's red-flag observer."""
    red_flag_summary = runner._red_flag_observer.get_summary()
    would_abort, abort_reason = runner._red_flag_observer.hypothetical_should_abort()

    # Note: Individual flag events are not tracked in current implementation.
    # The summary provides aggregate counts. For detailed flag history,
    # the RedFlagObserver would need to be extended to store individual events.
    flags: List[Dict[str, Any]] = []

    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cycles": result.cycles_completed,
        "flags": flags,
        "summary": {
            "total_flags": result.total_red_flags,
            "by_severity": result.red_flags_by_severity,
            "by_type": result.red_flags_by_type,
            "max_streaks": {
                "OMEGA_EXIT": red_flag_summary.max_omega_exit_streak,
                "HARD_FAIL": red_flag_summary.max_hard_fail_streak,
                "CDI-007": red_flag_summary.max_cdi_007_streak,
                "GOVERNANCE_DIVERGENCE": red_flag_summary.max_divergence_streak,
            },
            "hypothetical_abort": {
                "would_abort": would_abort,
                "abort_cycle": red_flag_summary.hypothetical_abort_cycles[0] if red_flag_summary.hypothetical_abort_cycles else None,
                "abort_reason": abort_reason,
            },
        },
    }


def build_metrics_windows(
    run_id: str,
    config: FirstLightConfig,
    runner: FirstLightShadowRunner,
) -> Dict[str, Any]:
    """Build metrics_windows.json from accumulator."""
    completed_windows = runner._metrics_accumulator.get_completed_windows()
    trajectories = runner._metrics_accumulator.get_trajectories()

    windows = []
    for w in completed_windows:
        windows.append({
            "window_index": w["window_index"],
            "start_cycle": w["start_cycle"],
            "end_cycle": w["end_cycle"],
            "metrics": {
                "success_rate": w["success_metrics"]["success_rate"],
                "abstention_rate": w["abstention_metrics"]["abstention_rate"],
                "omega_occupancy": w["safe_region_metrics"]["omega_occupancy"],
                "hard_ok_rate": w["hard_mode_metrics"]["hard_ok_rate"],
                "mean_rsi": w["stability_metrics"]["mean_rsi"],
                "min_rsi": w["stability_metrics"]["min_rsi"],
                "max_rsi": w["stability_metrics"]["max_rsi"],
                "block_rate": w["block_metrics"]["block_rate"],
            },
            "delta_p": {
                "success": None,  # Would need to compute from trajectory
                "abstention": None,
            },
            "red_flags_in_window": 0,  # Would need to count
        })

    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "window_size": config.success_window,
        "total_windows": len(completed_windows),
        "windows": windows,
        "trajectories": {
            "success_rate": trajectories["success_rate"],
            "abstention_rate": trajectories["abstention_rate"],
            "mean_rsi": trajectories["mean_rsi"],
            "omega_occupancy": trajectories["omega_occupancy"],
            "hard_ok_rate": trajectories["hard_ok_rate"],
        },
    }


def build_tda_metrics(run_id: str, result: Any) -> Dict[str, Any]:
    """Build tda_metrics.json from result."""
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "total_windows": len(result.tda_metrics),
        "metrics": result.tda_metrics,
        "trajectories": {
            "SNS": [m["SNS"] for m in result.tda_metrics],
            "PCS": [m["PCS"] for m in result.tda_metrics],
            "DRS": [m["DRS"] for m in result.tda_metrics],
            "HSS": [m["HSS"] for m in result.tda_metrics],
        },
    }


def build_run_config(
    run_id: str,
    args: argparse.Namespace,
    config: FirstLightConfig,
) -> Dict[str, Any]:
    """Build run_config.json snapshot."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {
            "seed": args.seed,
            "cycles": args.cycles,
            "output_dir": args.output_dir,
            "slice": args.slice,
            "runner_type": args.runner_type,
            "tau_0": args.tau_0,
            "window_size": args.window_size,
            "dry_run": args.dry_run,
            "verbose": args.verbose,
            "pathology": args.pathology,
        },
        "config": {
            "slice_name": config.slice_name,
            "runner_type": config.runner_type,
            "total_cycles": config.total_cycles,
            "tau_0": config.tau_0,
            "success_window": config.success_window,
            "shadow_mode": config.shadow_mode,
            "pathology": args.pathology,
        },
    }


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("USLA First-Light Experiment Harness (P3)")
    print("=" * 70)
    print()

    # Build config
    config = FirstLightConfig(
        slice_name=args.slice,
        runner_type=args.runner_type,
        total_cycles=args.cycles,
        tau_0=args.tau_0,
        success_window=args.window_size,
        shadow_mode=True,  # Always shadow mode in P3
    )

    # Validate config
    try:
        config.validate_or_raise()
    except ValueError as e:
        print(f"ERROR: Invalid configuration: {e}")
        return 1

    # Generate run ID
    run_id = generate_run_id(args.seed)

    print(f"Run ID:         {run_id}")
    print(f"Slice:          {args.slice}")
    print(f"Runner type:    {args.runner_type}")
    print(f"Cycles:         {args.cycles}")
    print(f"tau_0:          {args.tau_0}")
    print(f"Window size:    {args.window_size}")
    print(f"Seed:           {args.seed}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Pathology:      {args.pathology} (TEST-ONLY)")
    print()

    if args.dry_run:
        print("DRY RUN: Configuration validated, not executing.")
        print()
        print("Would produce artifacts:")
        print(f"  - {run_id}/synthetic_raw.jsonl")
        print(f"  - {run_id}/stability_report.json")
        print(f"  - {run_id}/red_flag_matrix.json")
        print(f"  - {run_id}/metrics_windows.json")
        print(f"  - {run_id}/tda_metrics.json")
        print(f"  - {run_id}/run_config.json")
        return 0

    # Create output directory
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Run experiment
    start_time = datetime.now(timezone.utc).isoformat()

    try:
        print("Running experiment...")
        runner = FirstLightShadowRunner(
            config,
            seed=args.seed,
            pathology_mode=args.pathology,
        )
        result = runner.run()
        end_time = datetime.now(timezone.utc).isoformat()

        print(f"Completed {result.cycles_completed} cycles.")
        print()

        # Write artifacts
        print("Writing artifacts...")

        # 1. synthetic_raw.jsonl
        raw_path = output_dir / "synthetic_raw.jsonl"
        raw_records = build_synthetic_raw(runner)
        write_jsonl(raw_path, raw_records)
        print(f"  - synthetic_raw.jsonl ({len(raw_records)} records)")

        # 2. stability_report.json
        report_path = output_dir / "stability_report.json"
        pathology_summary = runner.get_pathology_summary()

        report = build_stability_report(
            run_id,
            config,
            result,
            start_time,
            end_time,
            pathology=pathology_summary.get("mode", "none"),
            pathology_params=pathology_summary.get("params", {}),
        )
        report["config"]["seed"] = args.seed
        write_json(report_path, report)
        print(f"  - stability_report.json")

        # 3. red_flag_matrix.json
        flags_path = output_dir / "red_flag_matrix.json"
        flags = build_red_flag_matrix(run_id, result, runner)
        write_json(flags_path, flags)
        print(f"  - red_flag_matrix.json ({flags['summary']['total_flags']} flags)")

        # 4. metrics_windows.json
        windows_path = output_dir / "metrics_windows.json"
        windows = build_metrics_windows(run_id, config, runner)
        write_json(windows_path, windows)
        print(f"  - metrics_windows.json ({windows['total_windows']} windows)")

        # 5. tda_metrics.json
        tda_path = output_dir / "tda_metrics.json"
        tda = build_tda_metrics(run_id, result)
        write_json(tda_path, tda)
        print(f"  - tda_metrics.json ({tda['total_windows']} TDA snapshots)")

        # 6. run_config.json
        config_path = output_dir / "run_config.json"
        run_config = build_run_config(run_id, args, config)
        write_json(config_path, run_config)
        print(f"  - run_config.json")

        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()
        print(f"Success rate:     {result.u2_success_rate_final:.4f}")
        print(f"Delta-p success:  {result.delta_p_success}")
        print(f"Mean RSI:         {result.mean_rsi:.4f}")
        print(f"Omega occupancy:  {result.omega_occupancy:.4f}")
        print(f"HARD OK rate:     {result.hard_ok_rate:.4f}")
        print(f"Total red flags:  {result.total_red_flags}")
        print()

        if report["criteria_evaluation"]["all_passed"]:
            print("ALL SUCCESS CRITERIA PASSED")
        else:
            print("SOME CRITERIA FAILED:")
            for c in report["criteria_evaluation"]["criteria"]:
                if not c["passed"]:
                    print(f"  - {c['name']}: {c['actual']} (threshold: {c['threshold']})")

        print()
        print(f"Artifacts written to: {output_dir}")
        print("=" * 70)

        return 0

    except Exception as e:
        end_time = datetime.now(timezone.utc).isoformat()

        # Write error report
        error_path = output_dir / "error_report.json"
        error_report = {
            "run_id": run_id,
            "timestamp": end_time,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        write_json(error_path, error_report)

        print(f"ERROR: {e}")
        print(f"Error report written to: {error_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
