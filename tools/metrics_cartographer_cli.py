#!/usr/bin/env python3
"""
Metrics Cartographer CLI

Unified command-line interface for canonical metrics aggregation, validation,
and reporting across the MathLedger ecosystem.

Usage:
    python metrics_cartographer_cli.py collect    # Collect and aggregate metrics
    python metrics_cartographer_cli.py validate   # Validate latest metrics
    python metrics_cartographer_cli.py report     # Generate ASCII report
    python metrics_cartographer_cli.py full       # Collect + Validate + Report (default)
"""

import sys
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from metrics_cartographer import MetricsAggregator
from metrics_validator import MetricsValidator
from metrics_reporter import ASCIIReporter


def collect(session_id: str = None) -> int:
    """Collect and aggregate metrics from all sources"""
    if not session_id:
        session_id = None

    aggregator = MetricsAggregator()

    try:
        print("Collecting from data sources:")
        for collector in aggregator.collectors:
            print(f"  - {getattr(collector, 'name', collector.__class__.__name__)}")
        print()

        canonical = aggregator.aggregate(session_id)
        print(f"Session ID: {canonical.session_id}")
        print(f"Timestamp: {canonical.timestamp}")
        print()

        print("[OK] Aggregation complete")
        print()

        # Validate schema
        is_valid = aggregator.validate_against_schema(canonical.to_dict())
        if is_valid:
            print("[OK] Schema validation: PASS")
        else:
            print("[WARN] Schema validation: FAIL")
        print()

        # Export
        output_paths = aggregator.export(canonical)
        for path in output_paths:
            print(f"[OK] Exported to: {path}")
        print()

        # Summary
        total_entries = sum(len(v) if isinstance(v, dict) else 1 for v in canonical.metrics.values())
        variance_ok = canonical.variance['within_tolerance'] if canonical.variance else True
        epsilon = canonical.variance['epsilon_tolerance'] if canonical.variance else 0.01

        if variance_ok:
            print(f"[PASS] Metrics Canonicalized entries={total_entries} variance<=epsilon={epsilon}")
        else:
            cv = canonical.variance['coefficient_of_variation'] if canonical.variance else 0.0
            print(f"[WARN] Metrics Canonicalized entries={total_entries} variance={cv:.4f} > epsilon={epsilon}")

        if canonical.notes:
            print()
            print("Warnings:")
            for note in canonical.notes:
                print(f"  - {note}")

        trends = canonical.metrics.get("trends", {})
        if trends:
            print()
            print("Trend synopsis:")
            for key in ["proofs_per_sec", "proof_success_rate", "p95_latency_ms"]:
                series = trends.get(key, {})
                if series:
                    latest = series.get("latest", 0.0)
                    delta = series.get("delta_from_previous", 0.0)
                    trend_flag = series.get("trend", "flat")
                    print(f"  - {key}: latest={latest:.4f} delta={delta:.4f} trend={trend_flag}")

        return 0 if variance_ok else 1

    except Exception as e:
        print(f"[ERROR] Collection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def validate(epsilon: float = 0.01) -> int:
    """Validate latest metrics"""
    project_root = Path(__file__).parent.parent
    schema_path = project_root / "artifacts" / "metrics" / "schema_v1.json"
    metrics_path = project_root / "artifacts" / "metrics" / "latest.json"

    print("=" * 70)
    print("METRICS VALIDATION")
    print("=" * 70)
    print()

    if not schema_path.exists():
        print(f"[ERROR] Schema not found: {schema_path}")
        return 1

    if not metrics_path.exists():
        print(f"[ERROR] Metrics not found: {metrics_path}")
        print("Run 'metrics_cartographer_cli.py collect' first")
        return 1

    try:
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

        validator = MetricsValidator(schema_path)
        result = validator.validate(metrics, epsilon=epsilon)

        # Structure validation
        if result.is_valid:
            print("[PASS] Structure validation: OK")
        else:
            print("[FAIL] Structure validation: ERRORS")
            for error in result.errors:
                print(f"  ERROR: {error}")
        print()

        # Warnings
        if result.warnings:
            print(f"[WARN] {len(result.warnings)} warning(s):")
            for warning in result.warnings:
                print(f"  WARN: {warning}")
            print()

        # Variance check
        if result.variance_check:
            vc = result.variance_check
            print("Variance Check:")
            print(f"  CV: {vc.coefficient_of_variation:.6f}")
            print(f"  Epsilon: {vc.epsilon_tolerance:.6f}")
            print(f"  Status: {'PASS' if vc.within_tolerance else 'FAIL'}")
            print()

        # Final verdict
        if result.is_valid and (not result.variance_check or result.variance_check.within_tolerance):
            print("[PASS] Validation complete")
            return 0
        else:
            print("[FAIL] Validation failed")
            return 1

    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def report() -> int:
    """Generate ASCII report"""
    project_root = Path(__file__).parent.parent
    metrics_path = project_root / "artifacts" / "metrics" / "latest.json"

    if not metrics_path.exists():
        print(f"[ERROR] Metrics not found: {metrics_path}")
        print("Run 'metrics_cartographer_cli.py collect' first")
        return 1

    try:
        reporter = ASCIIReporter(metrics_path)
        report_text = reporter.generate_full_report()

        print()
        print(report_text)
        print()

        # Save to file
        output_file = project_root / "artifacts" / "metrics" / "latest_report.txt"
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(f"Report saved to: {output_file}")
        return 0

    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def full(session_id: str = None, epsilon: float = 0.01) -> int:
    """Run full pipeline: collect + validate + report"""
    print("=" * 70)
    print("METRICS CARTOGRAPHER - FULL PIPELINE")
    print("=" * 70)
    print()

    # Step 1: Collect
    print("Step 1/3: Collecting metrics...")
    print()
    ret = collect(session_id)
    if ret != 0:
        print()
        print("[WARN] Collection completed with warnings")
    print()

    # Step 2: Validate
    print("Step 2/3: Validating metrics...")
    print()
    ret = validate(epsilon)
    if ret != 0:
        print()
        print("[WARN] Validation found issues")
    print()

    # Step 3: Report
    print("Step 3/3: Generating report...")
    print()
    ret = report()
    if ret != 0:
        print()
        print("[ERROR] Report generation failed")
        return 1
    print()

    print("=" * 70)
    print("[COMPLETE] Metrics cartography pipeline finished")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Metrics Cartographer - Canonical metrics aggregation for MathLedger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metrics_cartographer_cli.py collect
  python metrics_cartographer_cli.py validate --epsilon 0.05
  python metrics_cartographer_cli.py report
  python metrics_cartographer_cli.py full --session my-session-id
        """
    )

    parser.add_argument(
        'command',
        nargs='?',
        default='full',
        choices=['collect', 'validate', 'report', 'full'],
        help='Command to run (default: full)'
    )
    parser.add_argument(
        '--session',
        type=str,
        help='Session ID for metrics collection'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='Variance tolerance threshold (default: 0.01)'
    )

    args = parser.parse_args()

    if args.command == 'collect':
        return collect(args.session)
    elif args.command == 'validate':
        return validate(args.epsilon)
    elif args.command == 'report':
        return report()
    elif args.command == 'full':
        return full(args.session, args.epsilon)


if __name__ == "__main__":
    sys.exit(main())
