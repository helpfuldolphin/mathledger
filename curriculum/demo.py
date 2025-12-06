#!/usr/bin/env python3
"""
Demonstration of Phase III Curriculum Drift Radar & Promotion Guard

This script demonstrates all major features:
1. Loading Phase II curriculum
2. Generating fingerprints
3. Detecting drift
4. Building drift history
5. Promotion gate evaluation
6. Global health summary
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from curriculum import (
    CurriculumLoaderV2,
    CurriculumFingerprint,
    compute_curriculum_diff,
    build_curriculum_drift_history,
    classify_curriculum_drift_event,
    evaluate_curriculum_for_promotion,
    summarize_curriculum_for_global_health,
)


def demo_section(title: str):
    """Print a demo section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70 + '\n')


def demo_loading():
    """Demonstrate curriculum loading."""
    demo_section("1. Loading Phase II Curriculum")
    
    curriculum = CurriculumLoaderV2.load()
    
    print(f"Schema Version: {curriculum.schema_version}")
    print(f"Slice Count: {len(curriculum.slices)}")
    print(f"\nSlices:")
    for name in curriculum.list_slices():
        print(f"  - {name}")
    
    print(f"\nSuccess Metrics:")
    metrics = curriculum.show_metrics()
    for name, metric in metrics.items():
        print(f"  {name}: {metric['kind']}")
    
    return curriculum


def demo_fingerprint(curriculum):
    """Demonstrate fingerprint generation."""
    demo_section("2. Generating Curriculum Fingerprint")
    
    fingerprint = CurriculumFingerprint.generate(curriculum, run_id="demo-run-001")
    
    print(f"Schema Version: {fingerprint.schema_version}")
    print(f"Slice Count: {fingerprint.slice_count}")
    print(f"Timestamp: {fingerprint.timestamp}")
    print(f"SHA256: {fingerprint.sha256}")
    print(f"\nPer-Slice Hashes:")
    for name, hash_val in fingerprint.slice_fingerprints.items():
        print(f"  {name}: {hash_val[:16]}...")
    
    return fingerprint


def demo_drift_detection(curriculum):
    """Demonstrate drift detection."""
    demo_section("3. Detecting Curriculum Drift")
    
    # Generate baseline fingerprint
    baseline = CurriculumFingerprint.generate(curriculum, run_id="baseline")
    print("Generated baseline fingerprint")
    
    # Simulate a modified curriculum by creating a new fingerprint
    # (In real use, this would be from a modified config)
    current = CurriculumFingerprint.generate(curriculum, run_id="current")
    print("Generated current fingerprint")
    
    # Compute diff
    diff = compute_curriculum_diff(baseline, current)
    
    print(f"\nDrift Analysis:")
    print(f"  Changed: {diff['changed']}")
    print(f"  Schema Version Changed: {diff['schema_version_changed']}")
    print(f"  Slices Added: {diff['slices_added']}")
    print(f"  Slices Removed: {diff['slices_removed']}")
    print(f"  Slices Modified: {diff['slices_modified']}")
    
    # Classify drift
    classification = classify_curriculum_drift_event(diff)
    print(f"\nDrift Classification:")
    print(f"  Severity: {classification['severity']}")
    print(f"  Blocking: {classification['blocking']}")
    if classification['reasons']:
        print(f"  Reasons:")
        for reason in classification['reasons']:
            print(f"    - {reason}")
    
    return baseline, current


def demo_drift_history():
    """Demonstrate drift history ledger."""
    demo_section("4. Building Drift History Ledger")
    
    # Create temporary fingerprint files
    curriculum = CurriculumLoaderV2.load()
    temp_files = []
    
    try:
        # Create 3 fingerprints
        for i in range(3):
            fp = CurriculumFingerprint.generate(curriculum, run_id=f"run-{i+1:03d}")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(fp.to_dict(), f)
                temp_files.append(f.name)
        
        print(f"Generated {len(temp_files)} fingerprints")
        
        # Build history
        history = build_curriculum_drift_history(temp_files)
        
        print(f"\nHistory Summary:")
        print(f"  Total Fingerprints: {len(history['fingerprints'])}")
        print(f"  Schema Version: {history['schema_version']}")
        print(f"  Drift Events: {history['drift_events_count']}")
        
        print(f"\nSlice Count Series:")
        for ts, count in history['slice_count_series']:
            print(f"  {ts}: {count} slices")
        
        return history
        
    finally:
        # Cleanup
        for path in temp_files:
            try:
                os.unlink(path)
            except:
                pass


def demo_promotion_gate(history):
    """Demonstrate promotion gate evaluation."""
    demo_section("5. Promotion Gate Evaluation")
    
    promotion = evaluate_curriculum_for_promotion(history)
    
    print(f"Promotion Decision:")
    print(f"  OK to Promote: {promotion['promotion_ok']}")
    print(f"  Last Drift Severity: {promotion['last_drift_severity']}")
    
    if promotion['blocking_reasons']:
        print(f"  Blocking Reasons:")
        for reason in promotion['blocking_reasons']:
            print(f"    - {reason}")
    else:
        print(f"  No blocking issues detected")
    
    return promotion


def demo_global_health(history):
    """Demonstrate global health summary."""
    demo_section("6. Global Health Summary")
    
    health = summarize_curriculum_for_global_health(history)
    
    print(f"Health Status: {health['status']}")
    print(f"Curriculum OK: {health['curriculum_ok']}")
    print(f"Current Slice Count: {health['current_slice_count']}")
    print(f"Recent Drift Events: {health['recent_drift_events']}")
    
    print(f"\nDetails:")
    for key, value in health['details'].items():
        print(f"  {key}: {value}")
    
    return health


def demo_cli_showcase():
    """Show CLI command examples."""
    demo_section("7. CLI Command Examples")
    
    print("List slices:")
    print("  $ python3 curriculum/cli.py --list-slices\n")
    
    print("Show slice details:")
    print("  $ python3 curriculum/cli.py --show-slice slice_uplift_goal\n")
    
    print("Generate fingerprint:")
    print("  $ python3 curriculum/cli.py --fingerprint --run-id baseline\n")
    
    print("Check for drift:")
    print("  $ python3 curriculum/cli.py --check-against baseline.json\n")
    
    print("Build drift history:")
    print("  $ python3 curriculum/cli.py --drift-history fp1.json fp2.json fp3.json\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  Phase III Curriculum Drift Radar & Promotion Guard Demo")
    print("="*70)
    
    try:
        # 1. Load curriculum
        curriculum = demo_loading()
        
        # 2. Generate fingerprint
        fingerprint = demo_fingerprint(curriculum)
        
        # 3. Detect drift
        baseline, current = demo_drift_detection(curriculum)
        
        # 4. Build drift history
        history = demo_drift_history()
        
        # 5. Evaluate promotion
        promotion = demo_promotion_gate(history)
        
        # 6. Global health summary
        health = demo_global_health(history)
        
        # 7. CLI examples
        demo_cli_showcase()
        
        # Summary
        demo_section("Summary")
        print("✅ All Phase III features demonstrated successfully!")
        print("\nKey Results:")
        print(f"  - Loaded {len(curriculum.slices)} slices")
        print(f"  - Generated fingerprint: {fingerprint.sha256[:16]}...")
        print(f"  - Drift history: {history['drift_events_count']} events")
        print(f"  - Promotion status: {'✅ OK' if promotion['promotion_ok'] else '❌ BLOCKED'}")
        print(f"  - Health status: {health['status']}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
