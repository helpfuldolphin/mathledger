#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
PRNG Governance Check — CI Gate for PRNG Compliance.

This script runs the full PRNG governance pipeline and returns an exit code
suitable for CI gates:

    Exit 0: OK (governance passed)
    Exit 1: WARN (governance issues but non-blocking)
    Exit 2: BLOCK (governance violations must be fixed)

Usage:
    python scripts/prng_governance_check.py
    python scripts/prng_governance_check.py --manifest artifacts/manifest.json
    python scripts/prng_governance_check.py --manifest manifest.json --replay replay.json
    python scripts/prng_governance_check.py --evidence-run

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rfl.prng.governance import (
    run_full_prng_governance,
    evaluate_prng_for_ci,
    build_prng_remediation_suggestions,
    GlobalHealthSummary,
)
from scripts.seed_namespace_linter import lint_namespaces


def load_json_file(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRNG Governance Check — CI Gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - OK (governance passed)
    1 - WARN (governance issues but non-blocking)
    2 - BLOCK (governance violations must be fixed)

Examples:
    python scripts/prng_governance_check.py
    python scripts/prng_governance_check.py --manifest manifest.json
    python scripts/prng_governance_check.py --manifest m1.json --replay m2.json
    python scripts/prng_governance_check.py --evidence-run
        """,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to primary manifest.json",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="Path to replay/comparison manifest.json",
    )
    parser.add_argument(
        "--namespace-report",
        type=Path,
        help="Path to namespace linter JSON report (will run linter if not provided)",
    )
    parser.add_argument(
        "--evidence-run",
        action="store_true",
        help="Mark this as an evidence-gathering run (stricter rules)",
    )
    parser.add_argument(
        "--test-context",
        action="store_true",
        help="Mark this as running in test context",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--show-suggestions",
        action="store_true",
        help="Show remediation suggestions",
    )

    args = parser.parse_args()

    # Load manifests if provided
    manifest = None
    replay_manifest = None

    if args.manifest:
        manifest = load_json_file(args.manifest)

    if args.replay:
        replay_manifest = load_json_file(args.replay)

    # Get namespace report
    namespace_report = None
    if args.namespace_report:
        namespace_report = load_json_file(args.namespace_report)
    else:
        # Run namespace linter
        if args.verbose:
            print("Running namespace linter...", file=sys.stderr)
        lint_result = lint_namespaces()
        namespace_report = lint_result.to_dict()

    # Run full governance pipeline
    result = run_full_prng_governance(
        manifest=manifest,
        replay_manifest=replay_manifest,
        namespace_report=namespace_report,
        is_evidence_run=args.evidence_run,
        is_test_context=args.test_context,
    )

    # Extract health summary
    health_dict = result["health"]
    health = GlobalHealthSummary(
        prng_policy_ok=health_dict["prng_policy_ok"],
        has_namespace_collisions=health_dict["has_namespace_collisions"],
        has_schedule_drift=health_dict["has_schedule_drift"],
        has_hardcoded_seeds=health_dict["has_hardcoded_seeds"],
        status=health_dict["status"],
        violation_count=health_dict["violation_count"],
        summary_message=health_dict["summary_message"],
    )

    # Evaluate for CI
    exit_code = evaluate_prng_for_ci(health)

    # Print single-line summary
    status_icon = {
        "OK": "✅",
        "WARN": "⚠️",
        "BLOCK": "❌",
    }.get(health.status, "?")
    print(f"{status_icon} {health.summary_message}")

    # Show suggestions if requested
    if args.show_suggestions and result["policy_eval"]["violations"]:
        # Re-run governance to get proper objects for suggestions
        from rfl.prng.governance import (
            build_prng_governance_snapshot,
            evaluate_prng_policy,
            build_prng_remediation_suggestions,
        )
        
        snapshot = build_prng_governance_snapshot(
            manifest=manifest,
            replay_manifest=replay_manifest,
            namespace_report=namespace_report,
            is_evidence_run=args.evidence_run,
            is_test_context=args.test_context,
        )
        policy_eval = evaluate_prng_policy(snapshot)

        suggestions = build_prng_remediation_suggestions(snapshot, policy_eval)
        if suggestions:
            print("\nRemediation suggestions:", file=sys.stderr)
            for i, suggestion in enumerate(suggestions, 1):
                print(f"\n{i}. [{suggestion['rule_id']}] {suggestion['impact']}", file=sys.stderr)
                print(f"   Action: {suggestion['suggested_action']}", file=sys.stderr)
                if suggestion.get("files_involved"):
                    print(f"   Files: {', '.join(suggestion['files_involved'][:5])}", file=sys.stderr)

    # Verbose output
    if args.verbose:
        print(f"\nStatus: {health.status}", file=sys.stderr)
        print(f"Violations: {health.violation_count}", file=sys.stderr)
        if health.has_schedule_drift:
            print("  ⚠ Schedule drift detected", file=sys.stderr)
        if health.has_namespace_collisions:
            print("  ⚠ Namespace collisions detected", file=sys.stderr)
        if health.has_hardcoded_seeds:
            print("  ⚠ Hard-coded seeds detected", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

