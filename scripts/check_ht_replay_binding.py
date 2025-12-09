#!/usr/bin/env python3
"""
CLI Tool for HT-Series Replay Invariant Verification

This tool verifies the INV-REPLAY-HT-* invariants defined in
H_T_SERIES_GOVERNANCE_CHARTER.md v1.1.0 Section 10.

Implements Triangle Contract v1.0.0 with governance summary support.

Usage:
    python scripts/check_ht_replay_binding.py \
        --primary PATH/TO/primary_ht_series.json \
        --replay PATH/TO/replay_ht_series.json \
        --receipt PATH/TO/replay_receipt.json \
        [--manifest PATH/TO/manifest.yaml] \
        [--output PATH/TO/report.json] \
        [--verbose]
        [--summary-only]

STATUS: PHASE II - NOT RUN IN PHASE I
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ht.ht_replay_verifier import (
    verify_all_replay_invariants,
    verify_mdap_ht_replay_triangle,
    generate_verification_report,
    summarize_ht_replay_for_governance,
    InvariantStatus,
    FailureSeverity,
    TRIANGLE_CONTRACT_VERSION,
    HtReplayStatus,
)


def load_json_file(path: Path) -> dict:
    """Load and parse a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_manifest(path: Path) -> Optional[dict]:
    """Load manifest from YAML or JSON file."""
    if not path.exists():
        return None

    suffix = path.suffix.lower()

    if suffix in ('.yaml', '.yml'):
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not installed. Cannot load YAML manifest.", file=sys.stderr)
            return None
    else:
        return load_json_file(path)


def print_result_summary(results: list, verbose: bool = False) -> None:
    """Print a human-readable summary of results to stderr."""
    passed = [r for r in results if r.status == InvariantStatus.PASS]
    failed = [r for r in results if r.status == InvariantStatus.FAIL]

    print("\n" + "=" * 60, file=sys.stderr)
    print("HT-SERIES REPLAY INVARIANT VERIFICATION", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Summary line
    if not failed:
        print(f"\n[PASS] All {len(results)} invariants passed", file=sys.stderr)
    else:
        critical = [f for f in failed if f.severity == FailureSeverity.CRITICAL]
        high = [f for f in failed if f.severity == FailureSeverity.HIGH]

        print(f"\n[FAIL] {len(failed)}/{len(results)} invariants failed", file=sys.stderr)
        if critical:
            print(f"  CRITICAL: {len(critical)} (Run INVALID)", file=sys.stderr)
        if high:
            print(f"  HIGH: {len(high)} (Manual investigation required)", file=sys.stderr)

    print("\n" + "-" * 60, file=sys.stderr)

    # Per-invariant results
    for result in results:
        status_icon = "[OK]" if result.status == InvariantStatus.PASS else "[XX]"
        severity_str = f" ({result.severity.value})" if result.severity else ""
        print(f"  {status_icon} {result.invariant_id}{severity_str}", file=sys.stderr)

        if verbose or result.status == InvariantStatus.FAIL:
            print(f"      {result.message}", file=sys.stderr)
            if result.expected and result.actual:
                print(f"      Expected: {result.expected[:32]}...", file=sys.stderr)
                print(f"      Actual:   {result.actual[:32]}...", file=sys.stderr)
            if result.cycle is not None:
                print(f"      Divergence at cycle: {result.cycle}", file=sys.stderr)

    print("-" * 60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=f"Verify HT-Series Replay Invariants (Contract v{TRIANGLE_CONTRACT_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python scripts/check_ht_replay_binding.py \\
      --primary artifacts/primary_ht_series.json \\
      --replay artifacts/replay_ht_series.json \\
      --receipt artifacts/replay_receipt.json

  # With manifest for MDAP binding verification
  python scripts/check_ht_replay_binding.py \\
      --primary artifacts/primary_ht_series.json \\
      --replay artifacts/replay_ht_series.json \\
      --receipt artifacts/replay_receipt.json \\
      --manifest manifests/uplift_u2.yaml

  # Save report to file
  python scripts/check_ht_replay_binding.py \\
      --primary artifacts/primary_ht_series.json \\
      --replay artifacts/replay_ht_series.json \\
      --receipt artifacts/replay_receipt.json \\
      --output reports/replay_verification.json

  # Governance summary only (for MAAS/dashboard integration)
  python scripts/check_ht_replay_binding.py \\
      --primary artifacts/primary_ht_series.json \\
      --replay artifacts/replay_ht_series.json \\
      --receipt artifacts/replay_receipt.json \\
      --summary-only

Exit codes:
  0 - All invariants passed (status: OK)
  1 - Critical invariant(s) failed (status: FAIL)
  2 - High severity invariant(s) failed only (status: WARN)
  3 - Error loading input files or invalid arguments

With --summary-only:
  0 - ht_replay_status == OK
  1 - ht_replay_status == FAIL
  2 - ht_replay_status == WARN or tool error
        """
    )

    parser.add_argument(
        '--primary', '-p',
        type=Path,
        required=True,
        help='Path to primary ht_series.json file'
    )

    parser.add_argument(
        '--replay', '-r',
        type=Path,
        required=True,
        help='Path to replay ht_series.json file'
    )

    parser.add_argument(
        '--receipt', '-R',
        type=Path,
        required=True,
        help='Path to replay_receipt.json file'
    )

    parser.add_argument(
        '--manifest', '-m',
        type=Path,
        default=None,
        help='Path to manifest file (YAML or JSON) for MDAP binding verification'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Path to write JSON verification report (default: stdout)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output for all invariants, not just failures'
    )

    parser.add_argument(
        '--triangle', '-t',
        action='store_true',
        help='Run full MDAP-Ht-Replay triangle verification (requires manifest)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress human-readable output, only emit JSON report'
    )

    parser.add_argument(
        '--summary-only', '-s',
        action='store_true',
        help='Output only governance summary JSON (for MAAS/dashboard integration)'
    )

    args = parser.parse_args()

    # Validate input files exist
    for name, path in [('primary', args.primary), ('replay', args.replay), ('receipt', args.receipt)]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            return 3

    # Load input files
    try:
        primary_ht_series = load_json_file(args.primary)
        replay_ht_series = load_json_file(args.replay)
        replay_receipt = load_json_file(args.receipt)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        return 3

    # Load manifest if provided
    manifest = None
    if args.manifest:
        if not args.manifest.exists():
            print(f"Warning: Manifest file not found: {args.manifest}", file=sys.stderr)
        else:
            try:
                manifest = load_manifest(args.manifest)
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}", file=sys.stderr)

    # Check if triangle verification requested but no manifest
    if args.triangle and not manifest:
        print("Error: --triangle requires --manifest", file=sys.stderr)
        return 3

    # Run verification
    if args.triangle and manifest:
        triangle_result = verify_mdap_ht_replay_triangle(
            manifest,
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            args.primary,
            args.replay
        )
        results = triangle_result.results
    else:
        results = verify_all_replay_invariants(
            primary_ht_series,
            replay_ht_series,
            replay_receipt,
            manifest,
            args.primary,
            args.replay
        )

    # Extract experiment_id and replay_id from data
    experiment_id = (
        primary_ht_series.get("meta", {}).get("experiment_id") or
        replay_receipt.get("primary_run", {}).get("experiment_id") or
        "unknown"
    )
    replay_id = (
        replay_receipt.get("replay_run", {}).get("replay_id") or
        f"{experiment_id}_replay"
    )

    # Generate report with series hashes (Triangle Contract v1.0.0)
    report = generate_verification_report(
        results,
        experiment_id,
        replay_id,
        primary_ht_series=primary_ht_series,
        replay_ht_series=replay_ht_series
    )

    # Add triangle info if applicable
    if args.triangle and manifest:
        report["triangle"] = {
            "mdap_vertex_valid": triangle_result.mdap_vertex_valid,
            "primary_ht_vertex_valid": triangle_result.primary_ht_vertex_valid,
            "replay_ht_vertex_valid": triangle_result.replay_ht_vertex_valid,
            "triangle_valid": triangle_result.triangle_valid
        }

    # Handle --summary-only mode
    if args.summary_only:
        summary = summarize_ht_replay_for_governance(report)
        summary_json = json.dumps(summary, indent=2, ensure_ascii=False)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary_json)
        else:
            print(summary_json)

        # Exit codes for summary-only mode
        status = summary.get("ht_replay_status", "FAIL")
        if status == HtReplayStatus.OK.value:
            return 0
        elif status == HtReplayStatus.FAIL.value:
            return 1
        else:  # WARN
            return 2

    # Print human-readable summary (normal mode)
    if not args.quiet:
        print_result_summary(results, args.verbose)

    # Output JSON report
    report_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_json)
        if not args.quiet:
            print(f"\nReport written to: {args.output}", file=sys.stderr)
    else:
        print(report_json)

    # Determine exit code (normal mode)
    failed = [r for r in results if r.status == InvariantStatus.FAIL]
    if not failed:
        return 0

    critical = [f for f in failed if f.severity == FailureSeverity.CRITICAL]
    if critical:
        return 1

    return 2


if __name__ == '__main__':
    sys.exit(main())
