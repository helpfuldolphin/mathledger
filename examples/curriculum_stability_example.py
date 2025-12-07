#!/usr/bin/env python
"""
Example: Using the Curriculum Stability Envelope

This script demonstrates how to use the curriculum stability envelope
to validate curriculum changes and check for stability violations.
"""

from curriculum import (
    compute_fingerprint,
    compute_fingerprint_diff,
    validate_curriculum_invariants,
    evaluate_curriculum_stability,
)
from curriculum.gates import load


def main():
    print("=" * 70)
    print("Curriculum Stability Envelope - Example")
    print("=" * 70)
    
    # 1. Load a curriculum system
    print("\n1. Loading curriculum system...")
    system = load("pl")
    print(f"   Loaded system: {system.slug}")
    print(f"   Description: {system.description}")
    print(f"   Active slice: {system.active_name}")
    print(f"   Total slices: {len(system.slices)}")
    
    # 2. Compute fingerprint
    print("\n2. Computing curriculum fingerprint...")
    fingerprint = compute_fingerprint(system)
    print(f"   Version: {fingerprint['version']}")
    print(f"   Monotonic axes: {fingerprint['monotonic_axes']}")
    print(f"   Slices: {[s['name'] for s in fingerprint['slices'][:3]]}...")
    
    # 3. Validate invariants
    print("\n3. Validating curriculum invariants...")
    report = validate_curriculum_invariants(system)
    if report.valid:
        print("   ✅ All invariants validated successfully")
        if report.warnings:
            print(f"   ⚠️  {len(report.warnings)} warning(s) detected:")
            for warning in report.warnings[:3]:
                print(f"      - {warning[:80]}...")
    else:
        print("   ❌ Validation failed:")
        for error in report.errors:
            print(f"      - {error}")
    
    # 4. Self-comparison (no changes)
    print("\n4. Checking stability (self-comparison)...")
    stability = evaluate_curriculum_stability(
        fingerprint,
        fingerprint,
        report,
        max_slice_changes=3,
        max_gate_change_pct=10.0,
    )
    
    print(f"   Allow promotion: {stability.allow_promotion}")
    print(f"   Reason: {stability.reason}")
    print(f"   Fingerprint changes: {stability.fingerprint_changes}")
    
    # 5. Demonstrate diff detection (simulate a change)
    print("\n5. Demonstrating diff detection...")
    # Create a modified fingerprint (simulate a parameter change)
    import copy
    modified_fp = copy.deepcopy(fingerprint)
    if len(modified_fp['slices']) > 0:
        modified_fp['slices'][0]['params']['atoms'] = 999
    
    diff = compute_fingerprint_diff(fingerprint, modified_fp)
    
    if diff.has_changes:
        print("   Changes detected:")
        print(f"   - Changed slices: {diff.changed_slices}")
        if diff.param_diffs:
            print(f"   - Parameter diffs: {list(diff.param_diffs.keys())}")
    else:
        print("   No changes detected")
    
    # 6. Check stability with changes
    print("\n6. Checking stability with simulated changes...")
    stability_with_changes = evaluate_curriculum_stability(
        fingerprint,
        modified_fp,
        report,
        max_slice_changes=3,
        max_gate_change_pct=10.0,
    )
    
    print(f"   Allow promotion: {stability_with_changes.allow_promotion}")
    print(f"   Reason: {stability_with_changes.reason}")
    print(f"   Fingerprint changes: {stability_with_changes.fingerprint_changes}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
