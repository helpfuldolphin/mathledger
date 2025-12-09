"""
Metric Conformance Report CLI

Generates human-readable Markdown reports from conformance snapshots.

Usage:
    python -m backend.metrics.conformance_report --snapshots-dir path --output report.md
    python -m backend.metrics.conformance_report --snapshot path/to/snapshot.json
    python -m backend.metrics.conformance_report --compare baseline.json candidate.json

Examples:
    # Generate report from all snapshots in directory
    python -m backend.metrics.conformance_report --snapshots-dir ./conformance --output report.md

    # Compare two specific snapshots
    python -m backend.metrics.conformance_report --compare baseline.json candidate.json --output diff.md

    # Check promotion gate
    python -m backend.metrics.conformance_report --gate baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.metrics.metric_conformance_snapshot import (
    ConformanceSnapshot,
    load_conformance_snapshot,
    compare_conformance_snapshots,
    can_promote_metric,
    render_conformance_report,
)


def find_snapshots(directory: Path) -> List[Path]:
    """Find all snapshot JSON files in a directory, sorted by modification time."""
    if not directory.exists():
        return []

    snapshot_files = list(directory.glob("*.json"))
    # Sort by modification time (oldest first)
    snapshot_files.sort(key=lambda p: p.stat().st_mtime)
    return snapshot_files


def load_snapshots(paths: List[Path]) -> List[ConformanceSnapshot]:
    """Load multiple snapshots from paths."""
    snapshots = []
    for path in paths:
        try:
            snapshot = load_conformance_snapshot(path)
            snapshots.append(snapshot)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
    return snapshots


def cmd_report(args: argparse.Namespace) -> int:
    """Generate conformance report."""
    snapshots: List[ConformanceSnapshot] = []

    if args.snapshots_dir:
        directory = Path(args.snapshots_dir)
        if not directory.exists():
            print(f"Error: Directory not found: {directory}", file=sys.stderr)
            return 1

        paths = find_snapshots(directory)
        if not paths:
            print(f"Warning: No snapshot files found in {directory}", file=sys.stderr)

        snapshots = load_snapshots(paths)

    elif args.snapshot:
        path = Path(args.snapshot)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            return 1

        snapshots = load_snapshots([path])

    # Generate report
    title = args.title or "Metric Conformance Report"
    report = render_conformance_report(snapshots, title=title)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {output_path}")
    else:
        print(report)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two snapshots."""
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"Error: Baseline not found: {baseline_path}", file=sys.stderr)
        return 1
    if not candidate_path.exists():
        print(f"Error: Candidate not found: {candidate_path}", file=sys.stderr)
        return 1

    baseline = load_conformance_snapshot(baseline_path)
    candidate = load_conformance_snapshot(candidate_path)

    comparison = compare_conformance_snapshots(baseline, candidate)

    # Generate comparison report
    report = render_conformance_report([baseline, candidate], title="Conformance Comparison")

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Comparison written to: {output_path}")
    else:
        print(report)

    # Return non-zero if regression detected
    if comparison.is_regression:
        print(f"\nRegression detected: {comparison.regression_severity}", file=sys.stderr)
        return 2

    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Check promotion gate."""
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"Error: Baseline not found: {baseline_path}", file=sys.stderr)
        return 1
    if not candidate_path.exists():
        print(f"Error: Candidate not found: {candidate_path}", file=sys.stderr)
        return 1

    baseline = load_conformance_snapshot(baseline_path)
    candidate = load_conformance_snapshot(candidate_path)

    can_promote, reason = can_promote_metric(
        baseline,
        candidate,
        allow_minor_regression=args.allow_minor,
    )

    print(f"Promotion {'ALLOWED' if can_promote else 'BLOCKED'}: {reason}")

    if args.json:
        comparison = compare_conformance_snapshots(baseline, candidate)
        output = {
            "can_promote": can_promote,
            "reason": reason,
            "is_regression": comparison.is_regression,
            "regression_severity": comparison.regression_severity,
            "tests_lost": len(comparison.tests_lost),
            "tests_gained": len(comparison.tests_gained),
        }
        print(json.dumps(output, indent=2))

    return 0 if can_promote else 3


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Metric Conformance Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate conformance report")
    report_group = report_parser.add_mutually_exclusive_group(required=True)
    report_group.add_argument(
        "--snapshots-dir",
        metavar="DIR",
        help="Directory containing snapshot JSON files",
    )
    report_group.add_argument(
        "--snapshot",
        metavar="FILE",
        help="Single snapshot JSON file",
    )
    report_parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output file (default: stdout)",
    )
    report_parser.add_argument(
        "--title",
        metavar="TITLE",
        help="Report title",
    )
    report_parser.set_defaults(func=cmd_report)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two snapshots")
    compare_parser.add_argument(
        "baseline",
        metavar="BASELINE",
        help="Baseline snapshot JSON file",
    )
    compare_parser.add_argument(
        "candidate",
        metavar="CANDIDATE",
        help="Candidate snapshot JSON file",
    )
    compare_parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output file (default: stdout)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Gate command
    gate_parser = subparsers.add_parser("gate", help="Check promotion gate")
    gate_parser.add_argument(
        "baseline",
        metavar="BASELINE",
        help="Baseline snapshot JSON file",
    )
    gate_parser.add_argument(
        "candidate",
        metavar="CANDIDATE",
        help="Candidate snapshot JSON file",
    )
    gate_parser.add_argument(
        "--allow-minor",
        action="store_true",
        help="Allow minor (L3) regressions",
    )
    gate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    gate_parser.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
