#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
# File: experiments/curriculum_inspector.py
"""
Curriculum Inspector CLI for Phase II Uplift Slices.

This module provides command-line tools for inspecting, validating, and
debugging Phase II curriculum configurations. It wraps CurriculumLoaderV2
with a user-friendly CLI interface.

Usage:
------
    # List all slices
    python -m experiments.curriculum_inspector --list-slices

    # Describe a specific slice
    python -m experiments.curriculum_inspector --describe slice_uplift_goal

    # Validate all slices
    python -m experiments.curriculum_inspector --validate

    # Dump config hash for preregistration
    python -m experiments.curriculum_inspector --dump-config-hash slice_uplift_goal

    # Export as JSON
    python -m experiments.curriculum_inspector --json

Output Formats:
---------------
- Human-readable text (default)
- JSON (--json flag or --output-format json)

Reference Documents:
--------------------
- docs/PHASE2_RFL_UPLIFT_PLAN.md
- experiments/slice_success_metrics.py
- config/curriculum_uplift_phase2.yaml
"""

import argparse
import json
import sys
from typing import Dict, Any, List, Optional

from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    FormulaPoolIntegrityResult,
    SuccessMetricValidationResult,
)


def format_validation_result_text(result: Dict[str, Any]) -> str:
    """
    PHASE II — NOT USED IN PHASE I

    Format validation result as human-readable text.

    Args:
        result: Validation result dictionary from CurriculumLoaderV2.validate_all()

    Returns:
        Multi-line string with formatted validation report.
    """
    lines = [
        "=" * 60,
        "CURRICULUM VALIDATION REPORT",
        "=" * 60,
        "",
        f"Version: {result['version']}",
        f"Total Slices: {result['slice_count']}",
        f"Overall Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}",
        "",
    ]

    # Monotonicity warnings
    if result['monotonicity_warnings']:
        lines.append("Monotonicity Warnings:")
        for warning in result['monotonicity_warnings']:
            lines.append(f"  ⚠ {warning}")
        lines.append("")
    else:
        lines.append("Monotonicity: ✓ No violations")
        lines.append("")

    # Per-slice results
    lines.append("-" * 60)
    lines.append("SLICE DETAILS")
    lines.append("-" * 60)

    for slice_name, slice_result in result['slices'].items():
        status = "✓" if (slice_result['success_metric_valid'] and
                         slice_result['formula_pool_valid']) else "✗"
        lines.append(f"\n{status} {slice_name}")
        lines.append(f"  Config Hash: {slice_result['config_hash'][:16]}...")
        lines.append(f"  Success Metric: {'✓' if slice_result['success_metric_valid'] else '✗'}")
        lines.append(f"  Formula Pool:   {'✓' if slice_result['formula_pool_valid'] else '✗'}")

        if slice_result['issues']:
            lines.append("  Issues:")
            for issue in slice_result['issues']:
                lines.append(f"    - {issue}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_slice_list_text(slices: List[str], loader: CurriculumLoaderV2) -> str:
    """
    PHASE II — NOT USED IN PHASE I

    Format slice list as human-readable text with summary info.

    Args:
        slices: List of slice names.
        loader: CurriculumLoaderV2 instance for getting slice details.

    Returns:
        Formatted string listing all slices.
    """
    lines = [
        "=" * 60,
        "PHASE II UPLIFT SLICES",
        "=" * 60,
        f"Version: {loader.get_version()}",
        f"Total: {len(slices)} slices",
        "",
        "-" * 60,
        f"{'Slice Name':<30} {'Atoms':<6} {'Depth':<8} {'Metric':<18}",
        "-" * 60,
    ]

    for slice_name in slices:
        params = loader.get_parameters(slice_name)
        metric = loader.get_success_metric_config(slice_name)
        depth_range = f"{params['depth_min']}-{params['depth_max']}"
        lines.append(
            f"{slice_name:<30} {params['atoms']:<6} {depth_range:<8} {metric['kind']:<18}"
        )

    lines.append("-" * 60)
    return "\n".join(lines)


def format_config_hash_text(slice_name: str, config_hash: str) -> str:
    """
    PHASE II — NOT USED IN PHASE I

    Format config hash for display.

    Args:
        slice_name: Name of the slice.
        config_hash: SHA256 hash of the slice configuration.

    Returns:
        Formatted string with hash info.
    """
    return (
        f"Slice: {slice_name}\n"
        f"Config Hash (SHA256): {config_hash}\n"
        f"Short: {config_hash[:16]}"
    )


def run_inspector(args: argparse.Namespace) -> int:
    """
    PHASE II — NOT USED IN PHASE I

    Main entry point for curriculum inspector.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for errors, 2 for validation failures).
    """
    try:
        loader = CurriculumLoaderV2(filepath=args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid config file: {e}", file=sys.stderr)
        return 1

    output_json = args.json or args.output_format == 'json'

    # --list-slices
    if args.list_slices:
        slices = loader.list_slices()
        if output_json:
            output = {
                'version': loader.get_version(),
                'slices': slices,
                'slice_count': len(slices),
            }
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print(format_slice_list_text(slices, loader))
        return 0

    # --describe <slice_name>
    if args.describe:
        try:
            if output_json:
                config = loader.get_slice_config(args.describe)
                output = {
                    'slice_name': args.describe,
                    'config': config,
                    'config_hash': loader.hash_slice_config(args.describe),
                }
                print(json.dumps(output, indent=2, sort_keys=True))
            else:
                print(loader.describe_slice(args.describe))
        except KeyError:
            print(f"Error: Slice not found: {args.describe}", file=sys.stderr)
            return 1
        return 0

    # --dump-config-hash <slice_name>
    if args.dump_config_hash:
        try:
            config_hash = loader.hash_slice_config(args.dump_config_hash)
            if output_json:
                output = {
                    'slice_name': args.dump_config_hash,
                    'config_hash': config_hash,
                    'algorithm': 'sha256',
                }
                print(json.dumps(output, indent=2, sort_keys=True))
            else:
                print(format_config_hash_text(args.dump_config_hash, config_hash))
        except KeyError:
            print(f"Error: Slice not found: {args.dump_config_hash}", file=sys.stderr)
            return 1
        return 0

    # --validate
    if args.validate:
        result = loader.validate_all()
        if output_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(format_validation_result_text(result))
        return 0 if result['valid'] else 2

    # --validate-slice <slice_name>
    if args.validate_slice:
        try:
            metric_result = loader.validate_success_metric(args.validate_slice)
            pool_result = loader.validate_formula_pool_integrity(args.validate_slice)

            if output_json:
                output = {
                    'slice_name': args.validate_slice,
                    'success_metric': {
                        'valid': metric_result.valid,
                        'kind': metric_result.metric_kind,
                        'missing_params': sorted(metric_result.missing_params),
                        'unknown_params': sorted(metric_result.unknown_params),
                        'param_values': metric_result.param_values,
                    },
                    'formula_pool': {
                        'valid': pool_result.valid,
                        'duplicate_count': len(pool_result.duplicate_formulas),
                        'duplicates': pool_result.duplicate_formulas,
                        'normalization_error_count': len(pool_result.normalization_errors),
                        'hash_collision_count': len(pool_result.hash_collisions),
                    },
                    'overall_valid': metric_result.valid and pool_result.valid,
                }
                print(json.dumps(output, indent=2, sort_keys=True))
            else:
                lines = [
                    f"Validation: {args.validate_slice}",
                    "=" * 40,
                    "",
                    f"Success Metric: {'✓ VALID' if metric_result.valid else '✗ INVALID'}",
                    f"  Kind: {metric_result.metric_kind}",
                    f"  Params: {metric_result.param_values}",
                ]
                if metric_result.missing_params:
                    lines.append(f"  Missing: {sorted(metric_result.missing_params)}")
                if metric_result.unknown_params:
                    lines.append(f"  Unknown: {sorted(metric_result.unknown_params)}")

                lines.extend([
                    "",
                    f"Formula Pool: {'✓ VALID' if pool_result.valid else '✗ INVALID'}",
                    f"  Entries: {len(pool_result.normalized_hashes)}",
                    f"  Duplicates: {len(pool_result.duplicate_formulas)}",
                    f"  Normalization Errors: {len(pool_result.normalization_errors)}",
                    f"  Hash Collisions: {len(pool_result.hash_collisions)}",
                ])

                print("\n".join(lines))

            return 0 if (metric_result.valid and pool_result.valid) else 2

        except KeyError:
            print(f"Error: Slice not found: {args.validate_slice}", file=sys.stderr)
            return 1

    # --dump-all-hashes
    if args.dump_all_hashes:
        hashes = {}
        for slice_name in loader.list_slices():
            hashes[slice_name] = loader.hash_slice_config(slice_name)

        if output_json:
            output = {
                'version': loader.get_version(),
                'config_file': args.config,
                'hashes': hashes,
            }
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print("Config Hashes (SHA256)")
            print("=" * 60)
            for slice_name, h in hashes.items():
                print(f"{slice_name}: {h}")
        return 0

    # --export-json (full curriculum export)
    if args.export_json:
        print(loader.to_json(indent=2))
        return 0

    # Default: show help
    print("Use --help for usage information.", file=sys.stderr)
    return 1


def main() -> int:
    """
    PHASE II — NOT USED IN PHASE I

    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Curriculum Inspector CLI for Phase II Uplift Slices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all slices
  python -m experiments.curriculum_inspector --list-slices

  # Describe a slice
  python -m experiments.curriculum_inspector --describe slice_uplift_goal

  # Validate all slices
  python -m experiments.curriculum_inspector --validate

  # Validate a specific slice
  python -m experiments.curriculum_inspector --validate-slice slice_uplift_sparse

  # Get config hash for preregistration
  python -m experiments.curriculum_inspector --dump-config-hash slice_uplift_goal

  # Export all hashes
  python -m experiments.curriculum_inspector --dump-all-hashes

  # Export as JSON
  python -m experiments.curriculum_inspector --validate --json
""",
    )

    # Config file
    parser.add_argument(
        '--config', '-c',
        default='config/curriculum_uplift_phase2.yaml',
        help='Path to curriculum YAML file (default: config/curriculum_uplift_phase2.yaml)',
    )

    # Output format
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output in JSON format',
    )
    parser.add_argument(
        '--output-format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)',
    )

    # Actions
    parser.add_argument(
        '--list-slices', '-l',
        action='store_true',
        help='List all slice names',
    )
    parser.add_argument(
        '--describe', '-d',
        metavar='SLICE',
        help='Describe a specific slice',
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate all slices',
    )
    parser.add_argument(
        '--validate-slice',
        metavar='SLICE',
        help='Validate a specific slice',
    )
    parser.add_argument(
        '--dump-config-hash',
        metavar='SLICE',
        help='Dump config hash for a slice',
    )
    parser.add_argument(
        '--dump-all-hashes',
        action='store_true',
        help='Dump config hashes for all slices',
    )
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export full curriculum as JSON',
    )

    args = parser.parse_args()
    return run_inspector(args)


if __name__ == '__main__':
    sys.exit(main())

