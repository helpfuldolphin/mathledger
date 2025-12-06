"""
Tests for Policy Learning Radar & Governance Hooks
==================================================

Validates Phase III implementation:
- Policy telemetry snapshot extraction
- Policy drift detection across slices
- Governance and global health summaries

All tests verify determinism and correctness of pure functions.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from rfl.policy_telemetry import (
    PolicyStateSnapshot,
    build_policy_telemetry_snapshot,
    compare_policy_snapshots,
    build_policy_drift_radar,
    summarize_policy_for_governance,
    summarize_policy_for_global_health,
)


# ============================================================================
# Test: PolicyStateSnapshot Validation
# ============================================================================

def test_policy_state_snapshot_creation():
    """Test basic snapshot creation and validation."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="baseline",
        update_count=5,
        weights={"len": 0.5, "depth": -0.3, "success": 1.0},
        clamped=False,
        clamp_count=0,
    )
    
    assert snapshot.schema_version == "v1"
    assert snapshot.slice_name == "baseline"
    assert snapshot.update_count == 5
    assert len(snapshot.weights) == 3
    assert snapshot.clamped is False
    assert snapshot.clamp_count == 0


def test_policy_state_snapshot_negative_update_count():
    """Test that negative update_count raises ValueError."""
    with pytest.raises(ValueError, match="update_count must be non-negative"):
        PolicyStateSnapshot(
            schema_version="v1",
            slice_name="test",
            update_count=-1,
            weights={},
        )


def test_policy_state_snapshot_negative_clamp_count():
    """Test that negative clamp_count raises ValueError."""
    with pytest.raises(ValueError, match="clamp_count must be non-negative"):
        PolicyStateSnapshot(
            schema_version="v1",
            slice_name="test",
            update_count=0,
            weights={},
            clamp_count=-1,
        )


def test_policy_state_snapshot_invalid_weights_type():
    """Test that non-dict weights raises TypeError."""
    with pytest.raises(TypeError, match="weights must be a dictionary"):
        PolicyStateSnapshot(
            schema_version="v1",
            slice_name="test",
            update_count=0,
            weights=[0.5, 0.3],  # type: ignore
        )


# ============================================================================
# Test: build_policy_telemetry_snapshot
# ============================================================================

def test_build_telemetry_snapshot_basic():
    """Test basic telemetry extraction."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="baseline",
        update_count=10,
        weights={"len": -0.5, "depth": 0.3, "success": 1.2},
        clamped=False,
        clamp_count=0,
    )
    
    telemetry = build_policy_telemetry_snapshot(snapshot)
    
    assert telemetry["schema_version"] == "v1"
    assert telemetry["slice_name"] == "baseline"
    assert telemetry["update_count"] == 10
    assert telemetry["clamped"] is False
    assert telemetry["clamp_count"] == 0
    
    # Check norms (manual calculation)
    # L1 = |−0.5| + |0.3| + |1.2| = 0.5 + 0.3 + 1.2 = 2.0
    assert abs(telemetry["weight_norm_l1"] - 2.0) < 1e-9
    
    # L2 = sqrt(0.25 + 0.09 + 1.44) = sqrt(1.78) ≈ 1.334
    expected_l2 = np.sqrt(0.5**2 + 0.3**2 + 1.2**2)
    assert abs(telemetry["weight_norm_l2"] - expected_l2) < 1e-9
    
    # All weights are nonzero
    assert telemetry["nonzero_weights"] == 3


def test_build_telemetry_snapshot_top_k_features():
    """Test extraction of top-K positive and negative features."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={
            "feature_a": 2.0,
            "feature_b": -1.5,
            "feature_c": 0.5,
            "feature_d": -0.8,
            "feature_e": 1.0,
        },
    )
    
    telemetry = build_policy_telemetry_snapshot(snapshot)
    
    # Top 3 positive: feature_a (2.0), feature_e (1.0), feature_c (0.5)
    top_positive = telemetry["top_k_positive_features"]
    assert len(top_positive) == 3
    assert top_positive[0] == ("feature_a", 2.0)
    assert top_positive[1] == ("feature_e", 1.0)
    assert top_positive[2] == ("feature_c", 0.5)
    
    # Top 3 negative (most negative first): feature_b (-1.5), feature_d (-0.8)
    top_negative = telemetry["top_k_negative_features"]
    assert len(top_negative) == 2
    assert top_negative[0] == ("feature_b", -1.5)
    assert top_negative[1] == ("feature_d", -0.8)


def test_build_telemetry_snapshot_sparse_weights():
    """Test telemetry with sparse (many zero) weights."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="sparse",
        update_count=3,
        weights={
            "active_1": 0.5,
            "zero_1": 0.0,
            "zero_2": 0.0,
            "active_2": -0.3,
            "zero_3": 0.0,
        },
    )
    
    telemetry = build_policy_telemetry_snapshot(snapshot)
    
    # Only 2 nonzero weights
    assert telemetry["nonzero_weights"] == 2
    
    # L1 norm = 0.5 + 0.3 = 0.8
    assert abs(telemetry["weight_norm_l1"] - 0.8) < 1e-9


def test_build_telemetry_snapshot_all_zeros():
    """Test telemetry with all-zero weights."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="zeros",
        update_count=0,
        weights={"a": 0.0, "b": 0.0, "c": 0.0},
    )
    
    telemetry = build_policy_telemetry_snapshot(snapshot)
    
    assert telemetry["weight_norm_l1"] == 0.0
    assert telemetry["weight_norm_l2"] == 0.0
    assert telemetry["nonzero_weights"] == 0
    assert len(telemetry["top_k_positive_features"]) == 0
    assert len(telemetry["top_k_negative_features"]) == 0


def test_build_telemetry_snapshot_clamping():
    """Test telemetry with clamping information."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="clamped",
        update_count=20,
        weights={"a": 0.5},
        clamped=True,
        clamp_count=3,
    )
    
    telemetry = build_policy_telemetry_snapshot(snapshot)
    
    assert telemetry["clamped"] is True
    assert telemetry["clamp_count"] == 3


# ============================================================================
# Test: compare_policy_snapshots
# ============================================================================

def test_compare_policy_snapshots_no_change():
    """Test comparison when policy hasn't changed."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3},
    )
    
    comparison = compare_policy_snapshots(before, after)
    
    assert comparison["slice_name"] == "test"
    assert comparison["l2_distance"] == 0.0
    assert comparison["l1_distance"] == 0.0
    assert comparison["num_sign_flips"] == 0
    assert len(comparison["top_features_changed"]) == 0


def test_compare_policy_snapshots_small_change():
    """Test comparison with small weight changes."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3, "success": 1.0},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=6,
        weights={"len": 0.6, "depth": 0.3, "success": 1.1},
    )
    
    comparison = compare_policy_snapshots(before, after)
    
    assert comparison["slice_name"] == "test"
    
    # L1 distance = |0.1| + |0.0| + |0.1| = 0.2
    assert abs(comparison["l1_distance"] - 0.2) < 1e-9
    
    # L2 distance = sqrt(0.01 + 0.0 + 0.01) = sqrt(0.02) ≈ 0.141
    expected_l2 = np.sqrt(0.1**2 + 0.0**2 + 0.1**2)
    assert abs(comparison["l2_distance"] - expected_l2) < 1e-9
    
    # No sign flips
    assert comparison["num_sign_flips"] == 0
    
    # Top changes: len and success both changed by 0.1
    top_changed = comparison["top_features_changed"]
    assert len(top_changed) <= 3
    # Check that len and success are in the list (order may vary)
    feature_names = [item[0] for item in top_changed]
    assert "len" in feature_names
    assert "success" in feature_names


def test_compare_policy_snapshots_sign_flip():
    """Test detection of sign flips."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": -0.3, "success": 1.0},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=6,
        weights={"len": -0.5, "depth": 0.3, "success": 1.0},
    )
    
    comparison = compare_policy_snapshots(before, after)
    
    # Two sign flips: len (+ to -) and depth (- to +)
    assert comparison["num_sign_flips"] == 2


def test_compare_policy_snapshots_new_features():
    """Test comparison when new features appear."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=6,
        weights={"len": 0.5, "depth": 0.3, "success": 1.0},
    )
    
    comparison = compare_policy_snapshots(before, after)
    
    # New feature "success" appears with value 1.0
    # L2 distance = sqrt(0.0 + 0.0 + 1.0) = 1.0
    assert abs(comparison["l2_distance"] - 1.0) < 1e-9
    
    # No sign flips (success went from 0 to positive, not counted as flip)
    assert comparison["num_sign_flips"] == 0


def test_compare_policy_snapshots_large_drift():
    """Test comparison with large drift."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="drift_test",
        update_count=10,
        weights={"len": 0.5, "depth": 0.3, "success": 1.0},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="drift_test",
        update_count=15,
        weights={"len": -1.5, "depth": 2.0, "success": 0.2},
    )
    
    comparison = compare_policy_snapshots(before, after)
    
    # Large L2 distance expected
    assert comparison["l2_distance"] > 1.0
    
    # len flipped sign
    assert comparison["num_sign_flips"] >= 1


# ============================================================================
# Test: build_policy_drift_radar
# ============================================================================

def test_build_drift_radar_empty():
    """Test drift radar with no comparisons."""
    radar = build_policy_drift_radar([])
    
    assert radar["slices"] == []
    assert radar["slices_with_large_drift"] == []
    assert radar["max_l2_distance"] == 0.0
    assert radar["num_slices_analyzed"] == 0


def test_build_drift_radar_single_slice_no_drift():
    """Test drift radar with single slice, no drift."""
    comparison = {
        "slice_name": "slice_a",
        "l2_distance": 0.1,
        "l1_distance": 0.15,
        "num_sign_flips": 0,
        "top_features_changed": [],
    }
    
    radar = build_policy_drift_radar([comparison])
    
    assert radar["num_slices_analyzed"] == 1
    assert radar["max_l2_distance"] == 0.1
    assert len(radar["slices_with_large_drift"]) == 0
    assert len(radar["slices"]) == 1


def test_build_drift_radar_multiple_slices_mixed_drift():
    """Test drift radar with multiple slices, some with large drift."""
    comparisons = [
        {
            "slice_name": "slice_a",
            "l2_distance": 0.2,
            "l1_distance": 0.3,
            "num_sign_flips": 0,
            "top_features_changed": [],
        },
        {
            "slice_name": "slice_b",
            "l2_distance": 0.8,  # Large drift (> 0.5)
            "l1_distance": 1.0,
            "num_sign_flips": 2,
            "top_features_changed": [("len", 0.5, -0.3, -0.8)],
        },
        {
            "slice_name": "slice_c",
            "l2_distance": 1.2,  # Large drift
            "l1_distance": 1.5,
            "num_sign_flips": 1,
            "top_features_changed": [("depth", 0.3, 1.0, 0.7)],
        },
    ]
    
    radar = build_policy_drift_radar(comparisons)
    
    assert radar["num_slices_analyzed"] == 3
    assert radar["max_l2_distance"] == 1.2
    assert len(radar["slices_with_large_drift"]) == 2
    assert "slice_b" in radar["slices_with_large_drift"]
    assert "slice_c" in radar["slices_with_large_drift"]
    assert "slice_a" not in radar["slices_with_large_drift"]


def test_build_drift_radar_threshold_boundary():
    """Test drift radar at threshold boundary (0.5)."""
    comparisons = [
        {
            "slice_name": "just_below",
            "l2_distance": 0.49,
            "l1_distance": 0.6,
            "num_sign_flips": 0,
            "top_features_changed": [],
        },
        {
            "slice_name": "just_above",
            "l2_distance": 0.51,
            "l1_distance": 0.6,
            "num_sign_flips": 0,
            "top_features_changed": [],
        },
    ]
    
    radar = build_policy_drift_radar(comparisons)
    
    # Only "just_above" should be flagged
    assert len(radar["slices_with_large_drift"]) == 1
    assert "just_above" in radar["slices_with_large_drift"]
    assert "just_below" not in radar["slices_with_large_drift"]


# ============================================================================
# Test: summarize_policy_for_governance
# ============================================================================

def test_governance_summary_ok():
    """Test governance summary when all is OK."""
    radar = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 0.1,
        "num_slices_analyzed": 3,
    }
    
    summary = summarize_policy_for_governance(radar)
    
    assert summary["has_large_drift"] is False
    assert summary["slices_with_large_drift"] == []
    assert summary["status"] == "OK"


def test_governance_summary_attention():
    """Test governance summary with 1-2 slices needing attention."""
    radar = {
        "slices": [],
        "slices_with_large_drift": ["slice_a", "slice_b"],
        "max_l2_distance": 0.8,
        "num_slices_analyzed": 5,
    }
    
    summary = summarize_policy_for_governance(radar)
    
    assert summary["has_large_drift"] is True
    assert len(summary["slices_with_large_drift"]) == 2
    assert summary["status"] == "ATTENTION"


def test_governance_summary_volatile():
    """Test governance summary with 3+ slices showing drift."""
    radar = {
        "slices": [],
        "slices_with_large_drift": ["slice_a", "slice_b", "slice_c"],
        "max_l2_distance": 1.5,
        "num_slices_analyzed": 5,
    }
    
    summary = summarize_policy_for_governance(radar)
    
    assert summary["has_large_drift"] is True
    assert len(summary["slices_with_large_drift"]) == 3
    assert summary["status"] == "VOLATILE"


# ============================================================================
# Test: summarize_policy_for_global_health
# ============================================================================

def test_global_health_ok():
    """Test global health summary when all is OK."""
    radar = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 0.3,
        "num_slices_analyzed": 3,
    }
    
    summary = summarize_policy_for_global_health(radar)
    
    assert summary["policy_surface_ok"] is True
    assert summary["max_l2_distance"] == 0.3
    assert summary["num_slices_analyzed"] == 3
    assert summary["status"] == "OK"


def test_global_health_warn():
    """Test global health summary with warning level."""
    radar = {
        "slices": [],
        "slices_with_large_drift": ["slice_a"],
        "max_l2_distance": 0.7,
        "num_slices_analyzed": 3,
    }
    
    summary = summarize_policy_for_global_health(radar)
    
    assert summary["policy_surface_ok"] is False
    assert summary["max_l2_distance"] == 0.7
    assert summary["status"] == "WARN"


def test_global_health_hot():
    """Test global health summary with hot status."""
    radar = {
        "slices": [],
        "slices_with_large_drift": ["slice_a", "slice_b"],
        "max_l2_distance": 1.5,
        "num_slices_analyzed": 3,
    }
    
    summary = summarize_policy_for_global_health(radar)
    
    assert summary["policy_surface_ok"] is False
    assert summary["max_l2_distance"] == 1.5
    assert summary["status"] == "HOT"


def test_global_health_threshold_boundaries():
    """Test global health at exact threshold boundaries."""
    # At warning threshold (0.5)
    radar1 = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 0.5,
        "num_slices_analyzed": 1,
    }
    summary1 = summarize_policy_for_global_health(radar1)
    assert summary1["status"] == "OK"
    assert summary1["policy_surface_ok"] is True
    
    # Just above warning threshold
    radar2 = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 0.51,
        "num_slices_analyzed": 1,
    }
    summary2 = summarize_policy_for_global_health(radar2)
    assert summary2["status"] == "WARN"
    assert summary2["policy_surface_ok"] is False
    
    # At hot threshold (1.0)
    radar3 = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 1.0,
        "num_slices_analyzed": 1,
    }
    summary3 = summarize_policy_for_global_health(radar3)
    assert summary3["status"] == "WARN"
    
    # Just above hot threshold
    radar4 = {
        "slices": [],
        "slices_with_large_drift": [],
        "max_l2_distance": 1.01,
        "num_slices_analyzed": 1,
    }
    summary4 = summarize_policy_for_global_health(radar4)
    assert summary4["status"] == "HOT"


# ============================================================================
# Test: Integration - Full Pipeline
# ============================================================================

def test_full_pipeline_integration():
    """Test complete pipeline from snapshots to governance/health summaries."""
    # Create initial policy state
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="slice_1",
        update_count=10,
        weights={"len": 0.5, "depth": 0.3, "success": 1.0},
    )
    
    # Create evolved policy state (small change)
    after_small = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="slice_1",
        update_count=15,
        weights={"len": 0.6, "depth": 0.35, "success": 1.05},
    )
    
    # Create evolved policy state (large change)
    after_large = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="slice_1",
        update_count=20,
        weights={"len": -1.0, "depth": 2.0, "success": 0.5},
    )
    
    # Extract telemetry
    telemetry_before = build_policy_telemetry_snapshot(before)
    telemetry_after_small = build_policy_telemetry_snapshot(after_small)
    telemetry_after_large = build_policy_telemetry_snapshot(after_large)
    
    assert "weight_norm_l2" in telemetry_before
    assert "weight_norm_l2" in telemetry_after_small
    assert "weight_norm_l2" in telemetry_after_large
    
    # Compare snapshots
    comparison_small = compare_policy_snapshots(before, after_small)
    comparison_large = compare_policy_snapshots(before, after_large)
    
    assert comparison_small["l2_distance"] < comparison_large["l2_distance"]
    
    # Build drift radar
    radar = build_policy_drift_radar([comparison_small, comparison_large])
    
    assert radar["num_slices_analyzed"] == 2
    # Large comparison should be flagged
    assert len(radar["slices_with_large_drift"]) >= 1
    
    # Generate governance summary
    gov_summary = summarize_policy_for_governance(radar)
    assert "status" in gov_summary
    
    # Generate global health summary
    health_summary = summarize_policy_for_global_health(radar)
    assert "status" in health_summary


# ============================================================================
# Test: Determinism
# ============================================================================

def test_determinism_telemetry():
    """Test that telemetry extraction is deterministic."""
    snapshot = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3, "success": 1.0},
    )
    
    # Extract telemetry multiple times
    telemetry1 = build_policy_telemetry_snapshot(snapshot)
    telemetry2 = build_policy_telemetry_snapshot(snapshot)
    telemetry3 = build_policy_telemetry_snapshot(snapshot)
    
    # All results should be identical
    assert telemetry1 == telemetry2
    assert telemetry2 == telemetry3


def test_determinism_comparison():
    """Test that policy comparison is deterministic."""
    before = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=5,
        weights={"len": 0.5, "depth": 0.3},
    )
    after = PolicyStateSnapshot(
        schema_version="v1",
        slice_name="test",
        update_count=6,
        weights={"len": 0.6, "depth": 0.4},
    )
    
    # Compare multiple times
    comp1 = compare_policy_snapshots(before, after)
    comp2 = compare_policy_snapshots(before, after)
    comp3 = compare_policy_snapshots(before, after)
    
    # All results should be identical
    assert comp1 == comp2
    assert comp2 == comp3


def test_determinism_drift_radar():
    """Test that drift radar is deterministic."""
    comparisons = [
        {
            "slice_name": "slice_a",
            "l2_distance": 0.3,
            "l1_distance": 0.4,
            "num_sign_flips": 0,
            "top_features_changed": [],
        },
        {
            "slice_name": "slice_b",
            "l2_distance": 0.8,
            "l1_distance": 1.0,
            "num_sign_flips": 1,
            "top_features_changed": [],
        },
    ]
    
    # Build radar multiple times
    radar1 = build_policy_drift_radar(comparisons)
    radar2 = build_policy_drift_radar(comparisons)
    radar3 = build_policy_drift_radar(comparisons)
    
    # All results should be identical
    assert radar1 == radar2
    assert radar2 == radar3
