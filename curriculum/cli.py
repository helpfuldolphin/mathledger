#!/usr/bin/env python3
"""
Curriculum CLI Tool

Provides command-line interface for:
- Listing slices (--list-slices)
- Showing slice details (--show-slice NAME)
- Showing success metrics (--show-metrics)
- Generating fingerprints (--fingerprint)
- Checking drift (--check-against FILE)
- Building drift history (--drift-history FILE1 FILE2 ...)
"""

import argparse
import json
import sys
from typing import List, Optional

from curriculum.phase2_loader import (
    CurriculumLoaderV2,
    CurriculumFingerprint,
    compute_curriculum_diff
)
from curriculum.drift_radar import (
    build_curriculum_drift_history,
    classify_curriculum_drift_event,
    evaluate_curriculum_for_promotion,
    summarize_curriculum_for_global_health
)


def list_slices(config_path: Optional[str] = None) -> int:
    """List all slices in curriculum."""
    try:
        curriculum = CurriculumLoaderV2.load(config_path)
        print(f"Schema: {curriculum.schema_version}")
        print(f"Slices ({len(curriculum.slices)}):")
        for name in curriculum.list_slices():
            print(f"  - {name}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def show_slice(name: str, config_path: Optional[str] = None) -> int:
    """Show detailed information about a slice."""
    try:
        curriculum = CurriculumLoaderV2.load(config_path)
        slice_info = curriculum.show_slice(name)
        print(json.dumps(slice_info, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def show_metrics(config_path: Optional[str] = None) -> int:
    """Show success metrics for all slices."""
    try:
        curriculum = CurriculumLoaderV2.load(config_path)
        metrics = curriculum.show_metrics()
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def generate_fingerprint(
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    run_id: Optional[str] = None
) -> int:
    """Generate curriculum fingerprint."""
    try:
        curriculum = CurriculumLoaderV2.load(config_path)
        fingerprint = CurriculumFingerprint.generate(curriculum, run_id)
        
        if output_path:
            fingerprint.save(output_path)
            print(f"Fingerprint saved to {output_path}")
        else:
            print(json.dumps(fingerprint.to_dict(), indent=2))
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def check_against(reference_path: str, config_path: Optional[str] = None) -> int:
    """Check current curriculum against reference fingerprint."""
    try:
        # Load reference fingerprint
        reference = CurriculumFingerprint.load_from_file(reference_path)
        
        # Generate current fingerprint
        curriculum = CurriculumLoaderV2.load(config_path)
        current = CurriculumFingerprint.generate(curriculum)
        
        # Compute diff
        diff = compute_curriculum_diff(reference, current)
        
        # Classify drift
        classification = classify_curriculum_drift_event(diff)
        
        # Print results
        print(f"Drift Check Results")
        print(f"===================")
        print(f"Reference: {reference_path}")
        print(f"Severity: {classification['severity']}")
        print(f"Blocking: {classification['blocking']}")
        
        if classification['reasons']:
            print(f"\nReasons:")
            for reason in classification['reasons']:
                print(f"  - {reason}")
        
        if diff.get("slices_added"):
            print(f"\nSlices Added: {', '.join(diff['slices_added'])}")
        if diff.get("slices_removed"):
            print(f"\nSlices Removed: {', '.join(diff['slices_removed'])}")
        if diff.get("slices_modified"):
            print(f"\nSlices Modified: {', '.join(diff['slices_modified'])}")
        
        print(f"\nReference SHA256: {reference.sha256}")
        print(f"Current SHA256:   {current.sha256}")
        
        # Exit code for CI
        if classification['blocking']:
            print("\n❌ BLOCKING drift detected", file=sys.stderr)
            return 1
        elif classification['severity'] == "NONE":
            print("\n✅ No drift detected")
            return 0
        else:
            print("\n⚠️  Non-blocking drift detected")
            return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def drift_history(fingerprint_paths: List[str]) -> int:
    """Build and display drift history from multiple fingerprints."""
    try:
        history = build_curriculum_drift_history(fingerprint_paths)
        
        print(f"Curriculum Drift History")
        print(f"========================")
        print(f"Total Fingerprints: {len(history['fingerprints'])}")
        print(f"Current Schema: {history['schema_version']}")
        print(f"Drift Events: {history['drift_events_count']}")
        
        if history['slice_count_series']:
            print(f"\nSlice Count Series:")
            for ts, count in history['slice_count_series']:
                print(f"  {ts}: {count} slices")
        
        # Promotion evaluation
        promotion = evaluate_curriculum_for_promotion(history)
        print(f"\nPromotion Gate:")
        print(f"  OK: {promotion['promotion_ok']}")
        if promotion['blocking_reasons']:
            print(f"  Blocking Reasons:")
            for reason in promotion['blocking_reasons']:
                print(f"    - {reason}")
        
        # Global health
        health = summarize_curriculum_for_global_health(history)
        print(f"\nGlobal Health:")
        print(f"  Status: {health['status']}")
        print(f"  Current Slices: {health['current_slice_count']}")
        print(f"  Recent Drift Events: {health['recent_drift_events']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Curriculum introspection and drift detection tool"
    )
    
    parser.add_argument(
        "--config",
        help="Path to curriculum YAML file (default: config/curriculum_uplift_phase2.yaml)"
    )
    
    # Operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list-slices",
        action="store_true",
        help="List all slices"
    )
    group.add_argument(
        "--show-slice",
        metavar="NAME",
        help="Show detailed information about a slice"
    )
    group.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show success metrics for all slices"
    )
    group.add_argument(
        "--fingerprint",
        action="store_true",
        help="Generate curriculum fingerprint"
    )
    group.add_argument(
        "--check-against",
        metavar="FILE",
        help="Check current curriculum against reference fingerprint"
    )
    group.add_argument(
        "--drift-history",
        nargs="+",
        metavar="FILE",
        help="Build drift history from multiple fingerprints"
    )
    
    # Additional options
    parser.add_argument(
        "--output",
        help="Output path for fingerprint (with --fingerprint)"
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier for fingerprint timestamp (with --fingerprint)"
    )
    
    args = parser.parse_args()
    
    if args.list_slices:
        return list_slices(args.config)
    elif args.show_slice:
        return show_slice(args.show_slice, args.config)
    elif args.show_metrics:
        return show_metrics(args.config)
    elif args.fingerprint:
        return generate_fingerprint(args.config, args.output, args.run_id)
    elif args.check_against:
        return check_against(args.check_against, args.config)
    elif args.drift_history:
        return drift_history(args.drift_history)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
