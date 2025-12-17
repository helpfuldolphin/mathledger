#!/usr/bin/env python
"""
P5 Budget Calibration CLI

Runs the 3-phase budget calibration experiment and emits:
- budget_calibration_log.jsonl: All CalibrationLogEntry records
- budget_calibration_summary.json: FP/FN rates and enablement recommendation

Usage:
    python scripts/run_budget_calibration.py --output-dir results/calibration
    python scripts/run_budget_calibration.py --phase 1 --cycles 100 --seed 42
    python scripts/run_budget_calibration.py --all --output-dir exports/budget

SHADOW MODE: All outputs are advisory only. No enablement or gating.

Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.topology.first_light.budget_calibration import (
    CalibrationHarness,
    CalibrationPhase,
    CalibrationLogEntry,
    CalibrationResult,
    FPFNReport,
)


def write_jsonl(entries: List[CalibrationLogEntry], output_path: Path) -> int:
    """
    Write calibration entries to JSONL file.

    Args:
        entries: List of CalibrationLogEntry objects
        output_path: Path to output file

    Returns:
        Number of entries written
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json_line = json.dumps(entry.to_dict(), separators=(',', ':'))
            f.write(json_line + '\n')
    return len(entries)


def write_summary(result: CalibrationResult, output_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Write calibration summary to JSON file.

    Args:
        result: CalibrationResult from harness
        output_path: Path to output file
        metadata: Additional metadata to include
    """
    summary = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shadow_mode": True,
        "metadata": metadata,
        "experiment": result.to_dict(),
        "compact_summary": build_compact_summary(result),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def build_compact_summary(result: CalibrationResult) -> Dict[str, Any]:
    """
    Build compact summary for evidence pack attachment.

    This is the structure attached under evidence["governance"]["budget_risk"]["calibration"].

    Args:
        result: CalibrationResult from harness

    Returns:
        Compact summary dict
    """
    summary = {
        "schema_version": "1.0.0",
        "experiment_id": result.experiment_id,
        "overall_pass": result.overall_pass,
        "enablement_recommendation": result.enablement_recommendation,
        "phases": {},
    }

    # Add phase summaries
    if result.phase_1_report:
        summary["phases"]["phase_1"] = {
            "cycles": result.phase_1_report.total_cycles,
            "fp_rate": round(result.phase_1_report.fp_rate, 4),
            "fn_rate": round(result.phase_1_report.fn_rate, 4),
            "meets_criteria": result.phase_1_report.meets_phase_criteria,
        }

    if result.phase_2_report:
        summary["phases"]["phase_2"] = {
            "cycles": result.phase_2_report.total_cycles,
            "fp_rate": round(result.phase_2_report.fp_rate, 4),
            "fn_rate": round(result.phase_2_report.fn_rate, 4),
            "meets_criteria": result.phase_2_report.meets_phase_criteria,
        }

    if result.phase_3_report:
        summary["phases"]["phase_3"] = {
            "cycles": result.phase_3_report.total_cycles,
            "fp_rate": round(result.phase_3_report.fp_rate, 4),
            "fn_rate": round(result.phase_3_report.fn_rate, 4),
            "meets_criteria": result.phase_3_report.meets_phase_criteria,
        }

    return summary


def run_phase_1_only(
    harness: CalibrationHarness,
    cycles: int,
    output_dir: Path,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Run Phase 1 only and generate outputs.

    Args:
        harness: CalibrationHarness instance
        cycles: Number of cycles
        output_dir: Output directory
        verbose: Print progress

    Returns:
        CalibrationResult with Phase 1 only
    """
    if verbose:
        print(f"Running Phase 1 (Synthetic Baseline) with {cycles} cycles...")

    report = harness.run_phase_1(cycles=cycles)

    if verbose:
        print(f"  FP rate: {report.fp_rate:.2%}")
        print(f"  FN rate: {report.fn_rate:.2%}")
        print(f"  Meets criteria: {report.meets_phase_criteria}")

    # Create partial result
    result = CalibrationResult(
        experiment_id=harness._generate_experiment_id(),
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        phase_1_report=report,
        overall_pass=report.meets_phase_criteria,
        enablement_recommendation="PHASE_1_ONLY" if report.meets_phase_criteria else "PHASE_1_BASELINE_FAILED",
    )

    return result


def run_phase_2_only(
    harness: CalibrationHarness,
    cycles: int,
    output_dir: Path,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Run Phase 2 only with synthetic data.

    Args:
        harness: CalibrationHarness instance
        cycles: Number of cycles
        output_dir: Output directory
        verbose: Print progress

    Returns:
        CalibrationResult with Phase 2 only
    """
    if verbose:
        print(f"Running Phase 2 (Controlled Load) with {cycles} cycles (synthetic)...")

    data = harness.generate_synthetic_phase_2_data(cycles=cycles)
    report = harness.run_phase_2(data)

    if verbose:
        print(f"  FP rate: {report.fp_rate:.2%}")
        print(f"  FN rate: {report.fn_rate:.2%}")
        print(f"  Meets criteria: {report.meets_phase_criteria}")

    result = CalibrationResult(
        experiment_id=harness._generate_experiment_id(),
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        phase_2_report=report,
        overall_pass=report.meets_phase_criteria,
        enablement_recommendation="PHASE_2_ONLY" if report.meets_phase_criteria else "PHASE_2_REMEDIATION_NEEDED",
    )

    return result


def run_phase_3_only(
    harness: CalibrationHarness,
    cycles: int,
    output_dir: Path,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Run Phase 3 only with synthetic stress data.

    Args:
        harness: CalibrationHarness instance
        cycles: Number of cycles
        output_dir: Output directory
        verbose: Print progress

    Returns:
        CalibrationResult with Phase 3 only
    """
    if verbose:
        print(f"Running Phase 3 (Stress Load) with {cycles} cycles (synthetic)...")

    data = harness.generate_synthetic_phase_3_data(cycles=cycles)
    report = harness.run_phase_3(data)

    if verbose:
        print(f"  FP rate: {report.fp_rate:.2%}")
        print(f"  FN rate: {report.fn_rate:.2%}")
        print(f"  Meets criteria: {report.meets_phase_criteria}")

    result = CalibrationResult(
        experiment_id=harness._generate_experiment_id(),
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        phase_3_report=report,
        overall_pass=report.meets_phase_criteria,
        enablement_recommendation="PHASE_3_ONLY" if report.meets_phase_criteria else "PHASE_3_REMEDIATION_NEEDED",
    )

    return result


def run_all_phases(
    harness: CalibrationHarness,
    phase_1_cycles: int,
    phase_2_cycles: int,
    phase_3_cycles: int,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Run all three phases.

    Args:
        harness: CalibrationHarness instance
        phase_1_cycles: Phase 1 cycle count
        phase_2_cycles: Phase 2 cycle count
        phase_3_cycles: Phase 3 cycle count
        verbose: Print progress

    Returns:
        CalibrationResult with all phases
    """
    start_time = datetime.now(timezone.utc).isoformat()

    # Phase 1
    if verbose:
        print(f"Phase 1 (Synthetic Baseline): {phase_1_cycles} cycles...")
    report_1 = harness.run_phase_1(cycles=phase_1_cycles)
    if verbose:
        print(f"  FP: {report_1.fp_rate:.2%}, FN: {report_1.fn_rate:.2%}, Pass: {report_1.meets_phase_criteria}")

    # Phase 2
    if verbose:
        print(f"Phase 2 (Controlled Load): {phase_2_cycles} cycles...")
    data_2 = harness.generate_synthetic_phase_2_data(cycles=phase_2_cycles)
    report_2 = harness.run_phase_2(data_2)
    if verbose:
        print(f"  FP: {report_2.fp_rate:.2%}, FN: {report_2.fn_rate:.2%}, Pass: {report_2.meets_phase_criteria}")

    # Phase 3
    if verbose:
        print(f"Phase 3 (Stress Load): {phase_3_cycles} cycles...")
    data_3 = harness.generate_synthetic_phase_3_data(cycles=phase_3_cycles)
    report_3 = harness.run_phase_3(data_3)
    if verbose:
        print(f"  FP: {report_3.fp_rate:.2%}, FN: {report_3.fn_rate:.2%}, Pass: {report_3.meets_phase_criteria}")

    end_time = datetime.now(timezone.utc).isoformat()

    # Determine overall result
    overall_pass = (
        report_1.meets_phase_criteria and
        report_2.meets_phase_criteria and
        report_3.meets_phase_criteria
    )

    if overall_pass:
        recommendation = "PROCEED_TO_STAGE_2"
    elif report_1.meets_phase_criteria and report_2.meets_phase_criteria:
        recommendation = "PHASE_3_REMEDIATION_NEEDED"
    elif report_1.meets_phase_criteria:
        recommendation = "PHASE_2_REMEDIATION_NEEDED"
    else:
        recommendation = "PHASE_1_BASELINE_FAILED"

    result = CalibrationResult(
        experiment_id=harness._generate_experiment_id(),
        start_time=start_time,
        end_time=end_time,
        phase_1_report=report_1,
        phase_2_report=report_2,
        phase_3_report=report_3,
        overall_pass=overall_pass,
        enablement_recommendation=recommendation,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="P5 Budget Calibration CLI - SHADOW MODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 only (synthetic baseline)
  python scripts/run_budget_calibration.py --phase 1 --cycles 500

  # Run all phases with default cycles
  python scripts/run_budget_calibration.py --all

  # Run all phases with custom cycles and output directory
  python scripts/run_budget_calibration.py --all --phase-1-cycles 500 --phase-2-cycles 1000 --phase-3-cycles 500 --output-dir results/calibration

  # Run with specific seed for reproducibility
  python scripts/run_budget_calibration.py --all --seed 42 --output-dir exports/budget
        """,
    )

    # Phase selection
    phase_group = parser.add_mutually_exclusive_group(required=True)
    phase_group.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        help='Run specific phase only (1, 2, or 3)',
    )
    phase_group.add_argument(
        '--all',
        action='store_true',
        help='Run all three phases',
    )

    # Cycle configuration
    parser.add_argument(
        '--cycles',
        type=int,
        default=500,
        help='Number of cycles for single-phase run (default: 500)',
    )
    parser.add_argument(
        '--phase-1-cycles',
        type=int,
        default=500,
        help='Phase 1 cycles for --all (default: 500)',
    )
    parser.add_argument(
        '--phase-2-cycles',
        type=int,
        default=1000,
        help='Phase 2 cycles for --all (default: 1000)',
    )
    parser.add_argument(
        '--phase-3-cycles',
        type=int,
        default=500,
        help='Phase 3 cycles for --all (default: 500)',
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/budget_calibration',
        help='Output directory (default: results/budget_calibration)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )

    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress information',
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output except errors',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize harness
    harness = CalibrationHarness(seed=args.seed)

    verbose = args.verbose and not args.quiet

    if verbose:
        print("=" * 60)
        print("P5 Budget Calibration Experiment")
        print("SHADOW MODE: Advisory only, no enablement")
        print("=" * 60)
        print(f"Seed: {args.seed}")
        print(f"Output: {output_dir}")
        print()

    # Run requested phase(s)
    if args.all:
        result = run_all_phases(
            harness,
            phase_1_cycles=args.phase_1_cycles,
            phase_2_cycles=args.phase_2_cycles,
            phase_3_cycles=args.phase_3_cycles,
            verbose=verbose,
        )
    elif args.phase == 1:
        result = run_phase_1_only(harness, args.cycles, output_dir, verbose)
    elif args.phase == 2:
        result = run_phase_2_only(harness, args.cycles, output_dir, verbose)
    else:
        result = run_phase_3_only(harness, args.cycles, output_dir, verbose)

    # Write outputs
    log_path = output_dir / "budget_calibration_log.jsonl"
    summary_path = output_dir / "budget_calibration_summary.json"

    entries = harness.get_all_entries()
    entries_written = write_jsonl(entries, log_path)

    metadata = {
        "seed": args.seed,
        "cli_args": vars(args),
        "phases_run": "all" if args.all else f"phase_{args.phase}",
    }
    write_summary(result, summary_path, metadata)

    if verbose:
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Entries written: {entries_written}")
        print(f"Log file: {log_path}")
        print(f"Summary file: {summary_path}")
        print()
        print(f"Overall pass: {result.overall_pass}")
        print(f"Recommendation: {result.enablement_recommendation}")

    if not args.quiet:
        # Print compact summary to stdout
        compact = build_compact_summary(result)
        print(json.dumps(compact, indent=2))

    # Exit code based on result
    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
