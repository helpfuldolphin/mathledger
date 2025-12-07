"""
PHASE II — NOT USED IN PHASE I

U2 Experiment CLI

Provides command-line interface for U2 experiment orchestration and validation.

Commands:
- promotion-precheck: Check if runs are ready for promotion

Exit codes:
- 0: PASS - All checks passed
- 1: WARN - Non-critical issues detected
- 2: BLOCK - Critical issues detected, promotion blocked
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .evidence_fusion import fuse_evidence_summaries, FusedEvidenceSummary


def promotion_precheck(manifest_paths: List[str]) -> int:
    """
    Run promotion pre-check on multiple run manifests.
    
    Args:
        manifest_paths: List of paths to run manifest JSON files
        
    Returns:
        Exit code: 0=PASS, 1=WARN, 2=BLOCK
    """
    print("=" * 80)
    print("PHASE II U2 PROMOTION PRE-CHECK")
    print("=" * 80)
    print()
    
    # Load manifests
    run_summaries = []
    for path_str in manifest_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"❌ ERROR: Manifest not found: {path}")
            return 2
        
        try:
            with open(path, "r") as f:
                summary = json.load(f)
            run_summaries.append(summary)
            print(f"✅ Loaded: {path}")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {path}: {e}")
            return 2
        except Exception as e:
            print(f"❌ ERROR: Failed to load {path}: {e}")
            return 2
    
    print()
    print(f"Total runs: {len(run_summaries)}")
    print()
    
    # Fuse evidence
    fused = fuse_evidence_summaries(run_summaries)
    
    # Print results
    print("-" * 80)
    print("VALIDATION RESULTS")
    print("-" * 80)
    print()
    
    # Determinism violations
    if fused.determinism_violations:
        print(f"❌ DETERMINISM VIOLATIONS: {len(fused.determinism_violations)}")
        for v in fused.determinism_violations:
            print(f"   - {v.description}")
            print(f"     Runs: {v.run_ids}")
            print(f"     Trace hashes: {v.trace_hashes}")
        print()
    else:
        print("✅ No determinism violations")
        print()
    
    # Missing artifacts
    if fused.missing_artifacts:
        print(f"❌ MISSING ARTIFACTS: {len(fused.missing_artifacts)}")
        for a in fused.missing_artifacts:
            print(f"   - {a.artifact_type}: {a.expected_path}")
            print(f"     Run: {a.run_id}")
        print()
    else:
        print("✅ No missing artifacts")
        print()
    
    # Conflicting slice names
    if fused.conflicting_slice_names:
        print(f"❌ CONFLICTING SLICE NAMES: {len(fused.conflicting_slice_names)}")
        for c in fused.conflicting_slice_names:
            print(f"   - Slice names: {c.slice_names}")
            print(f"     Runs: {c.run_ids}")
        print()
    else:
        print("✅ No conflicting slice names")
        print()
    
    # Run ordering anomalies
    if fused.run_ordering_anomalies:
        print(f"⚠️  RUN ORDERING ANOMALIES: {len(fused.run_ordering_anomalies)}")
        for a in fused.run_ordering_anomalies:
            print(f"   - {a.description}")
            print(f"     Details: {a.details}")
        print()
    else:
        print("✅ No run ordering anomalies")
        print()
    
    # RFL policy completeness
    if fused.rfl_policy_complete:
        print("✅ RFL policy inputs complete")
        print()
    else:
        print("⚠️  RFL policy inputs incomplete")
        print()
    
    # Validation errors
    if "validation_errors" in fused.metadata:
        print(f"❌ VALIDATION ERRORS:")
        for error in fused.metadata["validation_errors"]:
            print(f"   - {error}")
        print()
    
    # Final status
    print("-" * 80)
    print(f"FINAL STATUS: {fused.pass_status}")
    print("-" * 80)
    print()
    
    if fused.pass_status == "PASS":
        print("✅ All checks passed. Runs are ready for promotion.")
        return 0
    elif fused.pass_status == "WARN":
        print("⚠️  Non-critical issues detected. Review recommended.")
        return 1
    else:  # BLOCK
        print("❌ Critical issues detected. Promotion blocked.")
        return 2


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Experiment CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # promotion-precheck command
    precheck_parser = subparsers.add_parser(
        "promotion-precheck",
        help="Check if runs are ready for promotion",
    )
    precheck_parser.add_argument(
        "manifests",
        nargs="+",
        help="Paths to run manifest JSON files",
    )
    
    args = parser.parse_args()
    
    if args.command == "promotion-precheck":
        exit_code = promotion_precheck(args.manifests)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
