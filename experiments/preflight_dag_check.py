#!/usr/bin/env python3
"""
PHASE II — NOT RUN IN PHASE I

Pre-Flight DAG Check CLI Tool.

This tool executes all CHECK-* and DRIFT-* validations per docs/DAG_PRE_FLIGHT_AUDIT.md
before a U2 audit is allowed to run.

Usage:
    uv run python experiments/preflight_dag_check.py \
        --baseline results/uplift_u2_baseline.jsonl \
        --rfl results/uplift_u2_rfl.jsonl \
        --axiom-manifest config/axioms_pl.yaml \
        --slice-config config/slice_uplift_tree.yaml \
        --scope FULL \
        --out results/preflight_report.json

Exit Codes:
    0: All checks PASS
    1: WARN conditions (audit may proceed with flag)
    2: FAIL conditions (audit blocked)

Author: CLAUDE G — DAG Pre-Flight Auditor Engineer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root for imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Announce compliance
print("PHASE II — NOT USED IN PHASE I: Loading Pre-Flight DAG Check CLI.", file=sys.stderr)

from backend.dag.preflight_check import (
    CheckStatus,
    DriftEligibilityResult,
    PreflightAuditor,
    PreflightConfig,
    PreflightReport,
    Severity,
    build_dag_posture_snapshot,
    compare_dag_postures,
    compare_posture_files,
    get_exit_code,
)


def format_check_result(check_id: str, check_data: dict) -> str:
    """Format a single check result for human-readable output."""
    status = check_data["status"]
    name = check_data["name"]
    severity = check_data["severity"]

    # Status indicator
    if status == "PASS":
        indicator = "[PASS]"
    elif status == "WARN":
        indicator = "[WARN]"
    else:
        indicator = "[FAIL]"

    line = f"  {indicator} {check_id}: {name} ({severity})"

    # Add relevant details
    details = check_data.get("details", {})
    if status != "PASS":
        # Show key metrics on failure
        for key in ["cycles_found", "self_loops_found", "collisions_found",
                    "dangling_found", "violations_found", "exceeded_count"]:
            if key in details and details[key]:
                line += f" - {key}: {details[key]}"

    return line


def format_drift_result(drift_data: dict) -> str:
    """Format drift check result for human-readable output."""
    drift_id = drift_data["id"]
    status = drift_data["status"]
    name = drift_data["name"]
    metric = drift_data["metric_value"]
    threshold = drift_data["threshold"]

    if status == "PASS":
        indicator = "[PASS]"
    elif status == "WARN":
        indicator = "[WARN]"
    else:
        indicator = "[FAIL]"

    return f"  {indicator} {drift_id}: {name} (value={metric}, threshold={threshold})"


def print_report_summary(report: PreflightReport) -> None:
    """Print a human-readable summary of the pre-flight report."""
    print("\n" + "=" * 70)
    print("PRE-FLIGHT DAG AUDIT REPORT")
    print("=" * 70)
    print(f"Version: {report.preflight_version}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Label: {report.label}")

    print("\nINPUTS:")
    for key, value in report.inputs.items():
        if value:
            print(f"  {key}: {value}")

    print("\nHEALTH CHECKS:")
    report_dict = report.to_dict()
    for check_id in sorted(report_dict["checks"].keys()):
        check_data = report_dict["checks"][check_id]
        print(format_check_result(check_id, check_data))

    if report.drift_eligibility:
        print("\nDRIFT ELIGIBILITY:")
        drift_dict = report.drift_eligibility.to_dict()
        for drift_check in drift_dict["drift_checks"]:
            print(format_drift_result(drift_check))

        print(f"\n  Eligible for audit: {'YES' if drift_dict['eligible'] else 'NO'}")
        for reason in drift_dict["reasons"]:
            print(f"    - {reason}")

    print("\nSUMMARY:")
    summary = report.summary
    print(f"  Overall Status: {summary['overall_status']}")
    print(f"  Critical Failures: {summary['critical_failures']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Audit Eligible: {'YES' if summary['audit_eligible'] else 'NO'}")

    if summary.get("parse_errors"):
        baseline_errors = summary["parse_errors"].get("baseline", [])
        rfl_errors = summary["parse_errors"].get("rfl", [])
        if baseline_errors:
            print(f"\n  Baseline Parse Errors: {len(baseline_errors)}")
        if rfl_errors:
            print(f"  RFL Parse Errors: {len(rfl_errors)}")

    print("=" * 70)


def print_posture_summary(posture: dict) -> None:
    """Print a human-readable summary of the posture snapshot."""
    print("\n" + "=" * 70)
    print("DAG POSTURE SNAPSHOT")
    print("=" * 70)
    print(f"Schema Version: {posture.get('schema_version', 'unknown')}")
    print(f"Has Cycles: {'YES' if posture.get('has_cycles') else 'NO'}")
    print(f"Max Depth: {posture.get('max_depth', 0)}")
    print(f"Vertex Count: {posture.get('vertex_count', 0)}")
    print(f"Edge Count: {posture.get('edge_count', 0)}")
    print(f"Drift Eligible: {'YES' if posture.get('drift_eligible', True) else 'NO'}")
    if posture.get("drift_ineligibility_reason"):
        print(f"  Reason: {posture['drift_ineligibility_reason']}")
    print("=" * 70)


def print_posture_comparison(comparison: dict) -> None:
    """Print a human-readable summary of the posture comparison."""
    print("\n" + "=" * 70)
    print("DAG POSTURE COMPARISON (DRIFT RADAR)")
    print("=" * 70)
    print(f"Schema Compatible: {'YES' if comparison.get('schema_compatible', True) else 'NO'}")
    print(f"\nDELTAS:")
    print(f"  Depth Delta: {comparison.get('depth_delta', 0):+d}")
    print(f"  Vertex Count Delta: {comparison.get('vertex_count_delta', 0):+d}")
    print(f"  Edge Count Delta: {comparison.get('edge_count_delta', 0):+d}")
    print(f"\nSTATUS CHANGES:")
    print(f"  Drift Eligibility: {comparison.get('drift_eligibility_change', 'UNKNOWN')}")
    print(f"  Cycle Status Changed: {'YES' if comparison.get('cycle_status_changed') else 'NO'}")
    if comparison.get("cycle_transition"):
        print(f"    Transition: {comparison['cycle_transition']}")
    if comparison.get("eligibility_reasons"):
        reasons = comparison["eligibility_reasons"]
        if reasons.get("old"):
            print(f"    Old Reason: {reasons['old']}")
        if reasons.get("new"):
            print(f"    New Reason: {reasons['new']}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-Flight DAG Health and Drift Eligibility Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes (full preflight mode):
  0  All checks PASS - audit may proceed
  1  WARN conditions - audit may proceed with caveats
  2  FAIL conditions - audit blocked

Exit Codes (posture-only/compare-with modes):
  0  Always (informational output only)

Examples:
  # Basic check with baseline only
  python experiments/preflight_dag_check.py --baseline results/baseline.jsonl

  # Full check with baseline and RFL comparison
  python experiments/preflight_dag_check.py \\
      --baseline results/baseline.jsonl \\
      --rfl results/rfl.jsonl \\
      --axiom-manifest config/axioms_pl.yaml

  # Output JSON report
  python experiments/preflight_dag_check.py \\
      --baseline results/baseline.jsonl \\
      --out results/preflight_report.json

  # Posture-only mode (emit posture snapshot)
  python experiments/preflight_dag_check.py \\
      --baseline results/baseline.jsonl \\
      --posture-only \\
      --out results/posture.json

  # Compare with previous posture snapshot
  python experiments/preflight_dag_check.py \\
      --baseline results/baseline.jsonl \\
      --posture-only \\
      --compare-with results/previous_posture.json
        """,
    )

    # Input files
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline JSONL log file",
    )
    parser.add_argument(
        "--rfl",
        type=Path,
        help="Path to RFL (treatment) JSONL log file",
    )
    parser.add_argument(
        "--axiom-manifest",
        type=Path,
        help="Path to axiom manifest (JSON/YAML)",
    )
    parser.add_argument(
        "--slice-config",
        type=Path,
        help="Path to slice configuration (for depth bounds)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Path to threshold configuration file (YAML/JSON)",
    )

    # Configuration overrides
    parser.add_argument(
        "--scope",
        choices=["FULL", "BOUNDED", "EXPERIMENT"],
        default="FULL",
        help="Scope of DAG to check (default: FULL)",
    )
    parser.add_argument(
        "--dangling-tolerance",
        type=int,
        default=0,
        help="Tolerance for dangling parent references (default: 0)",
    )
    parser.add_argument(
        "--depth-tolerance",
        type=int,
        default=2,
        help="Tolerance for depth bound violations (default: 2)",
    )
    parser.add_argument(
        "--max-vertex-divergence",
        type=float,
        default=0.5,
        help="Maximum vertex divergence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-edge-divergence",
        type=float,
        default=0.6,
        help="Maximum edge divergence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--max-depth-difference",
        type=int,
        default=3,
        help="Maximum depth difference threshold (default: 3)",
    )
    parser.add_argument(
        "--cycle-tolerance",
        type=int,
        default=10,
        help="Tolerance for cycle count difference (default: 10)",
    )

    # Output options
    parser.add_argument(
        "--out", "-o",
        type=Path,
        help="Output JSON report file path",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress human-readable output, only output JSON",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit with code 2 on warnings (stricter mode)",
    )

    # Posture mode options (TASK 3)
    parser.add_argument(
        "--posture-only",
        action="store_true",
        help="Emit only the DAG posture snapshot (exit code always 0)",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        metavar="SNAPSHOT.json",
        help="Compare current posture with a previous snapshot file (requires --posture-only)",
    )

    args = parser.parse_args()

    # Handle --compare-with requiring --posture-only
    if args.compare_with and not args.posture_only:
        print("ERROR: --compare-with requires --posture-only", file=sys.stderr)
        return 2

    # Validate inputs
    if not args.baseline.exists():
        print(f"ERROR: Baseline file not found: {args.baseline}", file=sys.stderr)
        return 2

    if args.rfl and not args.rfl.exists():
        print(f"ERROR: RFL file not found: {args.rfl}", file=sys.stderr)
        return 2

    # Build configuration
    config = PreflightConfig(
        scope=args.scope,
        dangling_tolerance=args.dangling_tolerance,
        depth_tolerance=args.depth_tolerance,
        max_vertex_divergence=args.max_vertex_divergence,
        max_edge_divergence=args.max_edge_divergence,
        max_depth_difference=args.max_depth_difference,
        cycle_tolerance=args.cycle_tolerance,
    )

    # Load threshold config if provided
    if args.thresholds and args.thresholds.exists():
        try:
            content = args.thresholds.read_text(encoding='utf-8')
            try:
                threshold_data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    threshold_data = yaml.safe_load(content)
                except ImportError:
                    threshold_data = {}
            config = PreflightConfig.from_dict(threshold_data)
        except Exception as e:
            print(f"WARNING: Could not load thresholds config: {e}", file=sys.stderr)

    # Run pre-flight audit
    auditor = PreflightAuditor(config=config)

    try:
        report = auditor.run_full_preflight(
            baseline_path=args.baseline,
            rfl_path=args.rfl,
            axiom_manifest_path=args.axiom_manifest,
            slice_config_path=args.slice_config,
        )
    except Exception as e:
        print(f"ERROR: Pre-flight audit failed: {e}", file=sys.stderr)
        return 2

    # =========================================================================
    # POSTURE-ONLY MODE (TASK 3)
    # =========================================================================
    if args.posture_only:
        # Build posture snapshot from report
        posture = build_dag_posture_snapshot(report)

        # Handle comparison with previous snapshot
        comparison = None
        if args.compare_with:
            if not args.compare_with.exists():
                print(f"ERROR: Comparison file not found: {args.compare_with}", file=sys.stderr)
                return 2
            try:
                with open(args.compare_with, 'r', encoding='utf-8') as f:
                    old_posture = json.load(f)
                comparison = compare_dag_postures(old_posture, posture)
            except json.JSONDecodeError as e:
                print(f"ERROR: Invalid JSON in comparison file: {e}", file=sys.stderr)
                return 2
            except Exception as e:
                print(f"ERROR: Could not load comparison file: {e}", file=sys.stderr)
                return 2

        # Output posture (and optional comparison)
        if not args.quiet:
            print_posture_summary(posture)
            if comparison:
                print_posture_comparison(comparison)

        # Determine output content
        if comparison:
            output_data = {
                "posture": posture,
                "comparison": comparison,
            }
        else:
            output_data = posture

        if args.out:
            try:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, sort_keys=True)
                if not args.quiet:
                    print(f"\nPosture written to: {args.out}")
            except Exception as e:
                print(f"ERROR: Could not write posture: {e}", file=sys.stderr)
                return 2

        if args.quiet and not args.out:
            # Output JSON to stdout
            print(json.dumps(output_data, indent=2, sort_keys=True))

        # Posture-only mode always returns 0 (informational)
        return 0

    # =========================================================================
    # FULL PREFLIGHT MODE (existing behavior)
    # =========================================================================

    # Output results
    if not args.quiet:
        print_report_summary(report)

    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2)
            if not args.quiet:
                print(f"\nReport written to: {args.out}")
        except Exception as e:
            print(f"ERROR: Could not write report: {e}", file=sys.stderr)
            return 2

    if args.quiet and not args.out:
        # Output JSON to stdout
        print(json.dumps(report.to_dict(), indent=2))

    # Determine exit code
    exit_code = get_exit_code(report)

    if args.fail_on_warn and exit_code == 1:
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
