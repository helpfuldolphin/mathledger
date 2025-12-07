"""
Curriculum CLI

Command-line interface for curriculum management, validation, and stability checking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from curriculum.gates import CurriculumSystem, load
from curriculum.stability_envelope import (
    compute_fingerprint,
    compute_fingerprint_diff,
    validate_curriculum_invariants,
    evaluate_curriculum_stability,
)


def load_fingerprint_from_file(filepath: str) -> Dict[str, Any]:
    """Load a curriculum fingerprint from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_fingerprint_to_file(fingerprint: Dict[str, Any], filepath: str) -> None:
    """Save a curriculum fingerprint to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)


def cmd_validate_invariants(args: argparse.Namespace) -> int:
    """Validate curriculum invariants for a system."""
    try:
        system = load(args.system)
        report = validate_curriculum_invariants(system)
        
        print(f"Curriculum Invariant Validation: {args.system}")
        print(f"Valid: {report.valid}")
        
        if report.errors:
            print("\nErrors:")
            for error in report.errors:
                print(f"  ❌ {error}")
        
        if report.warnings:
            print("\nWarnings:")
            for warning in report.warnings:
                print(f"  ⚠️  {warning}")
        
        if report.valid:
            print("\n✅ All invariants validated successfully")
            return 0
        else:
            print(f"\n❌ Validation failed with {len(report.errors)} error(s)")
            return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def cmd_stability_envelope(args: argparse.Namespace) -> int:
    """
    Check curriculum stability envelope.
    
    Compares current curriculum against a baseline fingerprint.
    """
    try:
        # Load current curriculum
        system = load(args.system)
        current_fp = compute_fingerprint(system)
        
        # Load baseline fingerprint if provided
        if args.baseline:
            baseline_fp = load_fingerprint_from_file(args.baseline)
        else:
            # If no baseline, use current as both (no-op check)
            baseline_fp = current_fp
        
        # Validate invariants
        report = validate_curriculum_invariants(system)
        
        # Evaluate stability
        stability = evaluate_curriculum_stability(
            baseline_fp,
            current_fp,
            report,
            max_slice_changes=args.max_slice_changes,
            max_gate_change_pct=args.max_gate_change_pct,
        )
        
        print(f"Curriculum Stability Envelope: {args.system}")
        print(f"Allow Promotion: {stability.allow_promotion}")
        print(f"Reason: {stability.reason}")
        print(f"Fingerprint Changes: {stability.fingerprint_changes}")
        
        if stability.gate_threshold_changes:
            print("\nGate Threshold Changes:")
            for key, pct in stability.gate_threshold_changes.items():
                print(f"  {key}: {pct:.1f}%")
        
        if stability.removed_slices:
            print(f"\nRemoved Slices: {', '.join(stability.removed_slices)}")
        
        if stability.renamed_slices:
            print("\nRenamed Slices:")
            for old, new in stability.renamed_slices:
                print(f"  {old} -> {new}")
        
        if stability.invariant_regressions:
            print("\nInvariant Regressions:")
            for regression in stability.invariant_regressions:
                print(f"  {regression}")
        
        # Save current fingerprint if requested
        if args.save_fingerprint:
            save_fingerprint_to_file(current_fp, args.save_fingerprint)
            print(f"\n📝 Fingerprint saved to {args.save_fingerprint}")
        
        if stability.allow_promotion:
            print("\n✅ Curriculum is stable for promotion")
            return 0
        else:
            print("\n❌ Curriculum is NOT stable for promotion")
            return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def cmd_diff_fingerprint(args: argparse.Namespace) -> int:
    """Compute the diff between two curriculum fingerprints."""
    try:
        fp_a = load_fingerprint_from_file(args.fingerprint_a)
        fp_b = load_fingerprint_from_file(args.fingerprint_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        print(f"Fingerprint Diff: {args.fingerprint_a} -> {args.fingerprint_b}")
        
        if not diff.has_changes:
            print("No changes detected")
            return 0
        
        if diff.invariant_diffs:
            print("\nInvariant Changes:")
            for key, (old, new) in diff.invariant_diffs.items():
                print(f"  {key}: {old} -> {new}")
        
        if diff.added_slices:
            print(f"\nAdded Slices: {', '.join(diff.added_slices)}")
        
        if diff.removed_slices:
            print(f"\nRemoved Slices: {', '.join(diff.removed_slices)}")
        
        if diff.renamed_slices:
            print("\nRenamed Slices:")
            for old, new in diff.renamed_slices:
                print(f"  {old} -> {new}")
        
        if diff.changed_slices:
            print(f"\nChanged Slices: {', '.join(diff.changed_slices)}")
        
        if diff.param_diffs:
            print("\nParameter Diffs:")
            for slice_name, params in diff.param_diffs.items():
                print(f"  {slice_name}:")
                for key, (old, new) in params.items():
                    print(f"    {key}: {old} -> {new}")
        
        if diff.gate_diffs:
            print("\nGate Diffs:")
            for slice_name, gates in diff.gate_diffs.items():
                print(f"  {slice_name}:")
                for key, (old, new) in gates.items():
                    print(f"    {key}: {old} -> {new}")
        
        # Output as JSON if requested
        if args.json:
            json_output = diff.to_dict()
            print("\nJSON Output:")
            print(json.dumps(json_output, indent=2))
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Curriculum management and validation CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # validate-invariants command
    validate_parser = subparsers.add_parser(
        "validate-invariants",
        help="Validate curriculum invariants"
    )
    validate_parser.add_argument(
        "--system",
        type=str,
        default="pl",
        help="Curriculum system slug (default: pl)"
    )
    
    # stability-envelope command
    stability_parser = subparsers.add_parser(
        "stability-envelope",
        help="Check curriculum stability envelope"
    )
    stability_parser.add_argument(
        "--system",
        type=str,
        default="pl",
        help="Curriculum system slug (default: pl)"
    )
    stability_parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline fingerprint JSON file"
    )
    stability_parser.add_argument(
        "--save-fingerprint",
        type=str,
        help="Path to save current fingerprint JSON"
    )
    stability_parser.add_argument(
        "--max-slice-changes",
        type=int,
        default=3,
        help="Maximum number of slices that can change (default: 3)"
    )
    stability_parser.add_argument(
        "--max-gate-change-pct",
        type=float,
        default=10.0,
        help="Maximum percentage change for gate thresholds (default: 10.0)"
    )
    
    # diff-fingerprint command
    diff_parser = subparsers.add_parser(
        "diff-fingerprint",
        help="Compute diff between two curriculum fingerprints"
    )
    diff_parser.add_argument(
        "fingerprint_a",
        type=str,
        help="Path to first fingerprint JSON file"
    )
    diff_parser.add_argument(
        "fingerprint_b",
        type=str,
        help="Path to second fingerprint JSON file"
    )
    diff_parser.add_argument(
        "--json",
        action="store_true",
        help="Output diff as JSON"
    )
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "validate-invariants":
        return cmd_validate_invariants(args)
    elif args.command == "stability-envelope":
        return cmd_stability_envelope(args)
    elif args.command == "diff-fingerprint":
        return cmd_diff_fingerprint(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
