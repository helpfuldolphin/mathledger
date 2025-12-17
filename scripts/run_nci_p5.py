#!/usr/bin/env python3
"""NCI P5 Runner Script.

Runs NCI P5 evaluation against repository documentation and produces
JSON artifacts for CI and GGFL consumption.

SHADOW MODE CONTRACT:
- All outputs are advisory only
- Recommendations do NOT gate any operations
- Exit code is always 0 (non-gating)

Usage:
    # Basic run (DOC_ONLY mode)
    uv run python scripts/run_nci_p5.py

    # With telemetry schema (TELEMETRY_CHECKED mode)
    uv run python scripts/run_nci_p5.py --telemetry-schema config/telemetry_schema.json

    # With both (FULLY_BOUND mode)
    uv run python scripts/run_nci_p5.py \
        --telemetry-schema config/telemetry_schema.json \
        --slice-registry config/slice_registry.json

    # Custom output directory
    uv run python scripts/run_nci_p5.py --output-dir results/nci_p5

    # Quiet mode (CI-friendly)
    uv run python scripts/run_nci_p5.py --quiet

Outputs:
    - nci_p5_result.json: Full evaluation result
    - nci_p5_signal.json: Compact GGFL signal
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    """Run NCI P5 evaluation."""
    parser = argparse.ArgumentParser(
        description="Run NCI P5 evaluation on repository documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory (default: current directory)",
    )

    parser.add_argument(
        "--telemetry-schema",
        type=Path,
        default=None,
        help="Path to telemetry schema JSON (enables TELEMETRY_CHECKED mode)",
    )

    parser.add_argument(
        "--slice-registry",
        type=Path,
        default=None,
        help="Path to slice registry JSON (requires --telemetry-schema for FULLY_BOUND)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nci_p5"),
        help="Output directory for JSON artifacts (default: results/nci_p5)",
    )

    parser.add_argument(
        "--mock-global-nci",
        type=float,
        default=0.85,
        help="Mock global NCI value for panel (default: 0.85)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (minimal output, CI-friendly)",
    )

    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit with code 1 if SLO status is BREACH (optional gating)",
    )

    args = parser.parse_args()

    # Import here to avoid import errors if run without dependencies
    try:
        from backend.health.nci_governance_adapter import (
            run_nci_p5_with_artifacts,
            build_nci_p5_compact_signal,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import NCI adapter: {e}", file=sys.stderr)
        print("Make sure you're running from the repository root with uv.", file=sys.stderr)
        return 1

    # Validate repo root
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        print(f"ERROR: Repository root not found: {repo_root}", file=sys.stderr)
        return 1

    # Run NCI P5 evaluation
    if not args.quiet:
        print(f"NCI P5 Evaluation")
        print(f"=" * 60)
        print(f"Repository: {repo_root}")
        print(f"Telemetry Schema: {args.telemetry_schema or 'None (DOC_ONLY mode)'}")
        print(f"Slice Registry: {args.slice_registry or 'None'}")
        print()

    try:
        result = run_nci_p5_with_artifacts(
            repo_root=repo_root,
            telemetry_schema_path=args.telemetry_schema,
            slice_registry_path=args.slice_registry,
            mock_global_nci=args.mock_global_nci,
        )
    except Exception as e:
        print(f"ERROR: NCI evaluation failed: {e}", file=sys.stderr)
        return 1

    # Build compact signal
    compact_signal = build_nci_p5_compact_signal(result)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write full result
    result_path = output_dir / "nci_p5_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    # Write compact signal
    signal_path = output_dir / "nci_p5_signal.json"
    with open(signal_path, "w", encoding="utf-8") as f:
        json.dump(compact_signal, f, indent=2, sort_keys=True)

    # Print summary
    mode = result.get("mode", "DOC_ONLY")
    slo_status = result.get("slo_evaluation", {}).get("status", "OK")
    recommendation = result.get("governance_signal", {}).get("recommendation", "NONE")
    tcl_aligned = result.get("tcl_result", {}).get("aligned", True)
    sic_aligned = result.get("sic_result", {}).get("aligned", True)
    doc_count = result.get("artifact_metadata", {}).get("doc_count", 0)
    warning_count = len(result.get("warnings", []))

    if not args.quiet:
        print(f"Results")
        print(f"-" * 60)
        print(f"Mode:           {mode}")
        print(f"Documents:      {doc_count}")
        print(f"SLO Status:     {slo_status}")
        print(f"Recommendation: {recommendation}")
        print(f"TCL Aligned:    {tcl_aligned}")
        print(f"SIC Aligned:    {sic_aligned}")
        print(f"Warnings:       {warning_count}")
        print()
        print(f"Artifacts")
        print(f"-" * 60)
        print(f"Full result:    {result_path}")
        print(f"Compact signal: {signal_path}")
        print()

        # Print warnings if any
        if warning_count > 0:
            print(f"Warnings (first 5)")
            print(f"-" * 60)
            for w in result.get("warnings", [])[:5]:
                print(f"  [{w.get('warning_type')}] {w.get('message', '')[:70]}")
            if warning_count > 5:
                print(f"  ... and {warning_count - 5} more")
            print()

        print(f"SHADOW MODE: All outputs are advisory only.")
    else:
        # Quiet mode: single-line summary
        # Use ASCII-safe characters for Windows console compatibility
        status_icon = "[OK]" if slo_status == "OK" else "[WARN]" if slo_status == "WARN" else "[FAIL]"
        print(f"{status_icon} NCI P5: {mode} | SLO={slo_status} | TCL={tcl_aligned} | SIC={sic_aligned} | Warnings={warning_count}")

    # Non-gating exit (unless --fail-on-breach)
    if args.fail_on_breach and slo_status == "BREACH":
        if not args.quiet:
            print("NOTE: Exiting with code 1 due to --fail-on-breach")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
