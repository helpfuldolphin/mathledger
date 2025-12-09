#!/usr/bin/env python3
"""
security_posture_check.py - Unified Security Posture CLI

PHASE II -- NOT RUN IN PHASE I

Single command to emit the unified security posture JSON by:
- Reading existing outputs (replay_incident_report.json, seed_drift_analysis.json,
  last_mile_verification.json) if present, OR
- Running the component tools if requested

Exit codes:
- 0: is_security_ok == True
- 1: is_security_ok == False (WARN level)
- 2: is_security_ok == False (NO_GO level)
- 3: Error loading inputs

Usage:
    # Read from existing reports
    python security_posture_check.py --run-id my-run

    # Specify custom input paths
    python security_posture_check.py \
        --replay-report path/to/replay.json \
        --seed-report path/to/seed.json \
        --lastmile-report path/to/lastmile.json

    # Output to specific file
    python security_posture_check.py --output security_posture.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from security_posture import (
    build_security_posture,
    summarize_security_for_governance,
    SecurityLevel,
)


def load_json_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists, return None otherwise."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
        return None


def find_report_file(base_dir: Path, patterns: list) -> Optional[Path]:
    """Find first matching report file from patterns."""
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            # Return most recently modified
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def discover_reports(run_id: str, base_dir: Path) -> tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Discover component reports for a run.

    Searches in common locations:
    - logs/uplift/<run_id>/
    - logs/security/<run_id>/
    - Current directory

    Returns (replay_incident, seed_analysis, lastmile_report)
    """
    search_dirs = [
        base_dir / "logs" / "uplift" / run_id,
        base_dir / "logs" / "security" / run_id,
        base_dir / "logs" / "security",
        base_dir / "logs" / "uplift",
        base_dir,
    ]

    replay_patterns = [
        "replay_incident_report.json",
        "*replay*incident*.json",
        "*replay*report*.json",
    ]

    seed_patterns = [
        "seed_drift_analysis.json",
        "*seed*drift*.json",
        "*seed*analysis*.json",
    ]

    lastmile_patterns = [
        "last_mile_verification.json",
        "*lastmile*.json",
        "*last_mile*.json",
    ]

    replay_report = None
    seed_report = None
    lastmile_report = None

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        if replay_report is None:
            path = find_report_file(search_dir, replay_patterns)
            if path:
                replay_report = load_json_file(path)

        if seed_report is None:
            path = find_report_file(search_dir, seed_patterns)
            if path:
                seed_report = load_json_file(path)

        if lastmile_report is None:
            path = find_report_file(search_dir, lastmile_patterns)
            if path:
                lastmile_report = load_json_file(path)

    return replay_report, seed_report, lastmile_report


def print_posture_summary(posture: Dict[str, Any], verbose: bool = False):
    """Print formatted posture summary to console."""
    print("=" * 60)
    print("        UNIFIED SECURITY POSTURE SUMMARY")
    print("=" * 60)
    print()

    # Status indicator
    security_level = posture.get("security_level", "UNKNOWN")
    is_ok = posture.get("is_security_ok", False)

    if security_level == "OK":
        status_icon = "[OK]"
    elif security_level == "WARN":
        status_icon = "[!!]"
    else:
        status_icon = "[XX]"

    print(f"  Security Level: {security_level} {status_icon}")
    print(f"  Is Security OK: {'Yes' if is_ok else 'No'}")
    print()

    # Component status
    print("  Components:")
    print(f"    Replay:    {posture.get('replay_status', 'N/A'):<15} Severity: {posture.get('replay_severity', 'N/A')}")
    print(f"    Seed:      {posture.get('seed_classification', 'N/A'):<15} Confidence: {posture.get('seed_confidence', 'N/A')}")
    print(f"    Last-Mile: {posture.get('lastmile_status', 'N/A'):<15} ({posture.get('lastmile_passed', 0)}/{posture.get('lastmile_total', 20)} passed)")
    print()

    # Blocking reasons
    blocking = posture.get("blocking_reasons", [])
    if blocking:
        print("  Blocking Reasons:")
        for reason in blocking:
            print(f"    - {reason}")
        print()

    # Flags
    print("  Flags:")
    print(f"    has_seed_drift:              {posture.get('has_seed_drift', False)}")
    print(f"    has_substrate_nondeterminism: {posture.get('has_substrate_nondeterminism', False)}")
    print()

    if verbose:
        # Component availability
        components = posture.get("components_available", {})
        print("  Components Available:")
        for comp, available in components.items():
            status = "Yes" if available else "No"
            print(f"    {comp}: {status}")
        print()

        # Governance summary
        gov_summary = summarize_security_for_governance(posture)
        print("  Governance Summary:")
        for key, value in gov_summary.items():
            print(f"    {key}: {value}")
        print()

    print("-" * 60)
    print(f"  Run ID: {posture.get('run_id', 'unknown')}")
    print(f"  Generated: {posture.get('generated_at', 'unknown')}")
    print(f"  Schema: v{posture.get('schema_version', 'unknown')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Security Posture Check - Single command security status",
        epilog="Combines replay, seed, and last-mile analysis into unified posture."
    )

    # Input sources
    parser.add_argument(
        "--run-id",
        help="Run ID to discover reports for"
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="Path to replay_incident_report.json"
    )
    parser.add_argument(
        "--seed-report",
        type=Path,
        help="Path to seed_drift_analysis.json"
    )
    parser.add_argument(
        "--lastmile-report",
        type=Path,
        help="Path to last_mile_verification.json"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for report discovery"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("security_posture.json"),
        help="Output path for posture JSON"
    )
    parser.add_argument(
        "--governance-output",
        type=Path,
        help="Output path for governance summary JSON"
    )

    # Display options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON to stdout"
    )

    args = parser.parse_args()

    try:
        # Load component reports
        replay_report = None
        seed_report = None
        lastmile_report = None

        # Explicit paths take precedence
        if args.replay_report:
            replay_report = load_json_file(args.replay_report)
        if args.seed_report:
            seed_report = load_json_file(args.seed_report)
        if args.lastmile_report:
            lastmile_report = load_json_file(args.lastmile_report)

        # Discover reports if not explicitly provided
        if args.run_id and not all([replay_report, seed_report, lastmile_report]):
            discovered = discover_reports(args.run_id, args.base_dir)
            if replay_report is None:
                replay_report = discovered[0]
            if seed_report is None:
                seed_report = discovered[1]
            if lastmile_report is None:
                lastmile_report = discovered[2]

        # Build unified posture
        posture = build_security_posture(
            replay_incident=replay_report,
            seed_analysis=seed_report,
            lastmile_report=lastmile_report,
            run_id=args.run_id
        )

        # Save posture JSON
        with open(args.output, 'w') as f:
            json.dump(posture, f, indent=2)

        # Save governance summary if requested
        if args.governance_output:
            gov_summary = summarize_security_for_governance(posture)
            with open(args.governance_output, 'w') as f:
                json.dump(gov_summary, f, indent=2)

        # Console output
        if not args.quiet:
            if args.json_only:
                print(json.dumps(posture, indent=2))
            else:
                print_posture_summary(posture, verbose=args.verbose)
                print(f"\nPosture saved to: {args.output}")
                if args.governance_output:
                    print(f"Governance summary saved to: {args.governance_output}")

        # Exit code based on security status
        is_ok = posture.get("is_security_ok", False)
        security_level = posture.get("security_level", "NO_GO")

        if is_ok and security_level == "OK":
            sys.exit(0)
        elif is_ok and security_level == "WARN":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()