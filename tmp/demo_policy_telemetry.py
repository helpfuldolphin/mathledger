#!/usr/bin/env python3
"""
Demonstration of Phase III Policy Learning Radar & Governance Hooks

This script shows how to use the new policy telemetry and monitoring APIs.
"""

from rfl.policy_telemetry import (
    PolicyStateSnapshot,
    build_policy_telemetry_snapshot,
    compare_policy_snapshots,
    build_policy_drift_radar,
    summarize_policy_for_governance,
    summarize_policy_for_global_health,
)


def main():
    print("=" * 80)
    print("Policy Learning Radar & Governance Hooks - Demonstration")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Scenario: Track policy evolution across 3 curriculum slices
    # ========================================================================
    
    print("1. Creating initial policy snapshots for 3 curriculum slices...")
    print("-" * 80)
    
    # Slice 1: Baseline (early learning)
    slice1_before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="baseline",
        update_count=10,
        weights={"len": 0.2, "depth": 0.1, "success": 0.5},
        clamped=False,
        clamp_count=0,
    )
    
    slice1_after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="baseline",
        update_count=20,
        weights={"len": 0.25, "depth": 0.15, "success": 0.7},  # Small drift
        clamped=False,
        clamp_count=0,
    )
    
    # Slice 2: Medium complexity (moderate drift)
    slice2_before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="medium",
        update_count=15,
        weights={"len": -0.3, "depth": 0.5, "success": 1.0},
        clamped=False,
        clamp_count=0,
    )
    
    slice2_after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="medium",
        update_count=25,
        weights={"len": -0.8, "depth": 0.9, "success": 1.5},  # Medium drift
        clamped=False,
        clamp_count=1,
    )
    
    # Slice 3: Advanced (large drift - potential issue)
    slice3_before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="advanced",
        update_count=20,
        weights={"len": 0.5, "depth": 0.8, "success": 2.0},
        clamped=False,
        clamp_count=0,
    )
    
    slice3_after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="advanced",
        update_count=30,
        weights={"len": -1.5, "depth": 2.5, "success": 0.5},  # Large drift!
        clamped=True,
        clamp_count=5,
    )
    
    # ========================================================================
    # Extract telemetry from final snapshots
    # ========================================================================
    
    print("\n2. Extracting telemetry from final policy states...")
    print("-" * 80)
    
    for snapshot in [slice1_after, slice2_after, slice3_after]:
        telemetry = build_policy_telemetry_snapshot(snapshot)
        print(f"\nSlice: {telemetry['slice_name']}")
        print(f"  Updates: {telemetry['update_count']}")
        print(f"  L1 norm: {telemetry['weight_norm_l1']:.3f}")
        print(f"  L2 norm: {telemetry['weight_norm_l2']:.3f}")
        print(f"  Nonzero weights: {telemetry['nonzero_weights']}")
        print(f"  Clamped: {telemetry['clamped']} (count: {telemetry['clamp_count']})")
        print(f"  Top positive features: {telemetry['top_k_positive_features']}")
        print(f"  Top negative features: {telemetry['top_k_negative_features']}")
    
    # ========================================================================
    # Compare before/after snapshots to detect drift
    # ========================================================================
    
    print("\n\n3. Comparing policy evolution to detect drift...")
    print("-" * 80)
    
    comparisons = []
    for before, after in [(slice1_before, slice1_after),
                           (slice2_before, slice2_after),
                           (slice3_before, slice3_after)]:
        comparison = compare_policy_snapshots(before, after)
        comparisons.append(comparison)
        
        print(f"\nSlice: {comparison['slice_name']}")
        print(f"  L2 distance: {comparison['l2_distance']:.3f}")
        print(f"  L1 distance: {comparison['l1_distance']:.3f}")
        print(f"  Sign flips: {comparison['num_sign_flips']}")
        if comparison['top_features_changed']:
            print(f"  Top changes:")
            for feat, before_val, after_val, delta in comparison['top_features_changed']:
                print(f"    {feat}: {before_val:.3f} → {after_val:.3f} (Δ={delta:+.3f})")
    
    # ========================================================================
    # Build drift radar across all slices
    # ========================================================================
    
    print("\n\n4. Building policy drift radar...")
    print("-" * 80)
    
    radar = build_policy_drift_radar(comparisons)
    
    print(f"\nSlices analyzed: {radar['num_slices_analyzed']}")
    print(f"Max L2 distance: {radar['max_l2_distance']:.3f}")
    print(f"Slices with large drift: {radar['slices_with_large_drift']}")
    
    print("\nPer-slice drift summary:")
    for slice_data in radar['slices']:
        drift_flag = "⚠️  LARGE DRIFT" if slice_data['slice_name'] in radar['slices_with_large_drift'] else "✓ OK"
        print(f"  {slice_data['slice_name']:12s}: L2={slice_data['l2_distance']:.3f}  {drift_flag}")
    
    # ========================================================================
    # Generate governance summary
    # ========================================================================
    
    print("\n\n5. Governance summary...")
    print("-" * 80)
    
    gov_summary = summarize_policy_for_governance(radar)
    
    print(f"\nHas large drift: {gov_summary['has_large_drift']}")
    print(f"Affected slices: {gov_summary['slices_with_large_drift']}")
    print(f"Status: {gov_summary['status']}")
    
    status_emoji = {
        "OK": "✅",
        "ATTENTION": "⚠️",
        "VOLATILE": "🔥"
    }
    print(f"\n{status_emoji[gov_summary['status']]} Governance Status: {gov_summary['status']}")
    
    # ========================================================================
    # Generate global health summary
    # ========================================================================
    
    print("\n\n6. Global health summary...")
    print("-" * 80)
    
    health_summary = summarize_policy_for_global_health(radar)
    
    print(f"\nPolicy surface OK: {health_summary['policy_surface_ok']}")
    print(f"Max L2 distance: {health_summary['max_l2_distance']:.3f}")
    print(f"Slices analyzed: {health_summary['num_slices_analyzed']}")
    print(f"Health status: {health_summary['status']}")
    
    health_emoji = {
        "OK": "✅",
        "WARN": "⚠️",
        "HOT": "🔥"
    }
    print(f"\n{health_emoji[health_summary['status']]} Health Status: {health_summary['status']}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n✓ Telemetry extracted for {len(comparisons)} slices")
    print(f"✓ Drift detected in {len(radar['slices_with_large_drift'])} slices")
    print(f"✓ Governance status: {gov_summary['status']}")
    print(f"✓ Health status: {health_summary['status']}")
    
    if gov_summary['status'] in ["ATTENTION", "VOLATILE"]:
        print(f"\n⚠️  ACTION REQUIRED: Review policy updates for slices: {gov_summary['slices_with_large_drift']}")
    else:
        print("\n✅ All systems nominal")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
