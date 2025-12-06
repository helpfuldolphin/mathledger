#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Curriculum Introspection CLI for Phase II uplift experiments.

This CLI provides read-only inspection of curriculum configuration,
including slice definitions, success metrics, and validation checks.

Usage:
    python -m experiments.curriculum_inspect --list-slices
    python -m experiments.curriculum_inspect --show-slice arithmetic_simple
    python -m experiments.curriculum_inspect --show-metrics
    python -m experiments.curriculum_inspect --check-nondegenerate

Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
- Output is read-only; no writing to YAML.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    CurriculumLoadError,
    UnknownMetricKindError,
    UpliftSlice,
)
from experiments.slice_success_metrics import METRIC_KINDS


def print_header(title: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)
    print()


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print(f"--- {title} ---")
    print()


def format_slice_summary(slice_obj: UpliftSlice) -> str:
    """Format a one-line summary of a slice."""
    depth_range = f"{slice_obj.depth_min}-{slice_obj.depth_max}"
    metric_kind = slice_obj.success_metric.kind if slice_obj.success_metric else "N/A"
    return (
        f"  {slice_obj.name:<25} | atoms={slice_obj.atoms:<3} | "
        f"depth={depth_range:<7} | metric={metric_kind}"
    )


def cmd_list_slices(loader: CurriculumLoaderV2) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    List all slices with their key parameters.
    """
    print_header("Phase II Curriculum Slices")
    print("PHASE II — NOT USED IN PHASE I")
    print()
    print(f"Config version: {loader.version}")
    print(f"Total slices: {len(loader.slices)}")
    print()

    print("  {:<25} | {:<10} | {:<12} | {}".format(
        "Slice Name", "Atoms", "Depth Range", "Metric Kind"
    ))
    print("  " + "-" * 70)

    for name in loader.get_slice_names():
        slice_obj = loader.get_slice(name)
        print(format_slice_summary(slice_obj))

    print()
    return 0


def cmd_show_slice(loader: CurriculumLoaderV2, slice_name: str) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Show detailed information about a specific slice.
    """
    try:
        slice_obj = loader.get_slice(slice_name)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print_header(f"Slice: {slice_name}")
    print("PHASE II — NOT USED IN PHASE I")
    print()

    # Basic info
    print(f"Name:        {slice_obj.name}")
    print(f"Description: {slice_obj.description or '(none)'}")
    print(f"Verifier:    {slice_obj.verifier}")
    if slice_obj.prereg_hash:
        print(f"Prereg Hash: {slice_obj.prereg_hash}")

    # Parameters
    print_section("Parameters")
    print(f"  atoms:        {slice_obj.atoms}")
    print(f"  depth_min:    {slice_obj.depth_min}")
    print(f"  depth_max:    {slice_obj.depth_max}")
    print(f"  total_max:    {slice_obj.total_max}")
    print(f"  formula_pool: {slice_obj.formula_pool}")

    # Items
    if slice_obj.items:
        print_section("Items")
        print(f"  Count: {len(slice_obj.items)}")
        # Show first few items
        max_show = 5
        for i, item in enumerate(slice_obj.items[:max_show]):
            print(f"    [{i}] {item}")
        if len(slice_obj.items) > max_show:
            print(f"    ... and {len(slice_obj.items) - max_show} more")

    # Success Metric
    print_section("Success Metric")
    if slice_obj.success_metric:
        metric = slice_obj.success_metric
        print(f"  Kind: {metric.kind}")
        print(f"  Description: {metric.get_description()}")
        print(f"  Required Params: {', '.join(metric.get_required_params()) or '(none)'}")

        if metric.thresholds:
            print()
            print("  Thresholds:")
            for key, value in metric.thresholds.items():
                print(f"    {key}: {value}")

        if metric.target_hashes:
            print()
            print(f"  Target Hashes: {len(metric.target_hashes)} hash(es)")
            for h in sorted(list(metric.target_hashes)[:3]):
                print(f"    - {h}")
            if len(metric.target_hashes) > 3:
                print(f"    ... and {len(metric.target_hashes) - 3} more")
    else:
        print("  (No success metric configured)")

    # Metadata
    if slice_obj.metadata:
        print_section("Metadata")
        for key, value in slice_obj.metadata.items():
            print(f"  {key}: {value}")

    print()
    return 0


def cmd_show_metrics(loader: CurriculumLoaderV2) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Show all metric kinds, their parameters, and which slices use them.
    """
    print_header("Success Metric Kinds")
    print("PHASE II — NOT USED IN PHASE I")
    print()

    # Get usage info
    usage = loader.get_metric_kinds_in_use()

    # List all registered metric kinds
    print("Registered Metric Kinds:")
    print("-" * 50)
    print()

    for kind, (required, optional, description) in sorted(METRIC_KINDS.items()):
        used_by = usage.get(kind, [])
        status = "IN USE" if used_by else "not used"

        print(f"  {kind} ({status})")
        print(f"    Description: {description}")
        print(f"    Required Params: {', '.join(required) or '(none)'}")
        if optional:
            print(f"    Optional Params: {', '.join(optional)}")
        if used_by:
            print(f"    Used by: {', '.join(used_by)}")
        print()

    # Summary
    print_section("Usage Summary")
    used_count = len([k for k in METRIC_KINDS if k in usage])
    print(f"  Metric kinds in use: {used_count}/{len(METRIC_KINDS)}")
    print(f"  Slices with metrics: {sum(len(v) for v in usage.values())}/{len(loader.slices)}")

    print()
    return 0


def cmd_check_nondegenerate(loader: CurriculumLoaderV2) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Check slices for non-degenerate defaults (sanity checks).
    """
    print_header("Non-Degenerate Defaults Check")
    print("PHASE II — NOT USED IN PHASE I")
    print()
    print("This check validates that slice parameters are plausible.")
    print("Warnings are advisory only, not fatal errors.")
    print()

    warnings = loader.verify_non_degenerate_defaults()

    if not warnings:
        print("✓ All slices pass non-degenerate checks.")
        print()
        return 0

    # Group by severity
    errors = [w for w in warnings if w.severity == "error"]
    warns = [w for w in warnings if w.severity == "warning"]

    if errors:
        print_section("Errors")
        for w in errors:
            print(f"  ✗ {w}")

    if warns:
        print_section("Warnings")
        for w in warns:
            print(f"  ⚠ {w}")

    print()
    print(f"Summary: {len(errors)} error(s), {len(warns)} warning(s)")
    print()

    # Return non-zero only if there are errors
    return 1 if errors else 0


def main() -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="PHASE II Curriculum Introspection CLI (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.curriculum_inspect --list-slices
  python -m experiments.curriculum_inspect --show-slice arithmetic_simple
  python -m experiments.curriculum_inspect --show-metrics
  python -m experiments.curriculum_inspect --check-nondegenerate

Note: This tool is read-only and does not modify any configuration files.
All output is labeled "PHASE II — NOT USED IN PHASE I".
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to curriculum YAML file (default: auto-detect Phase II config)",
    )
    parser.add_argument(
        "--list-slices",
        action="store_true",
        help="List all slice names with atoms and depth ranges",
    )
    parser.add_argument(
        "--show-slice",
        type=str,
        metavar="NAME",
        help="Pretty-print UpliftSlice + SuccessMetricSpec for a specific slice",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="List all metric kinds, required params, and which slices use them",
    )
    parser.add_argument(
        "--check-nondegenerate",
        action="store_true",
        help="Check slices for non-degenerate defaults (advisory warnings)",
    )

    args = parser.parse_args()

    # Check that at least one action is specified
    if not any([args.list_slices, args.show_slice, args.show_metrics, args.check_nondegenerate]):
        parser.print_help()
        return 1

    # Load curriculum
    try:
        if args.config:
            loader = CurriculumLoaderV2.from_yaml_file(args.config)
        else:
            loader = CurriculumLoaderV2.from_default_phase2_config()
    except CurriculumLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except UnknownMetricKindError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Execute requested command(s)
    exit_code = 0

    if args.list_slices:
        exit_code = max(exit_code, cmd_list_slices(loader))

    if args.show_slice:
        exit_code = max(exit_code, cmd_show_slice(loader, args.show_slice))

    if args.show_metrics:
        exit_code = max(exit_code, cmd_show_metrics(loader))

    if args.check_nondegenerate:
        exit_code = max(exit_code, cmd_check_nondegenerate(loader))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
