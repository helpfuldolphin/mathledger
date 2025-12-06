"""
Policy Learning Radar & Governance Hooks
=========================================

Phase III implementation: Pure metadata extraction and monitoring for policy learning.

Contract:
- All functions are pure (no side effects, no learning behavior changes)
- All functions are deterministic (same inputs → same outputs)
- Provides telemetry snapshots, drift detection, and governance hooks
- Compatible with existing Phase II policy semantics

Public API:
- build_policy_telemetry_snapshot: Extract telemetry from a policy state
- build_policy_drift_radar: Detect drift across multiple policy comparisons
- summarize_policy_for_governance: Generate governance status summary
- summarize_policy_for_global_health: Generate global health status summary
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Sequence, List, Tuple
import numpy as np


@dataclass
class PolicyStateSnapshot:
    """
    Immutable snapshot of policy state for telemetry extraction.
    
    This represents the policy at a specific point in time for a curriculum slice.
    Used to compute telemetry metrics and compare policy evolution.
    """
    schema_version: str  # Version identifier for compatibility
    slice_name: str  # Curriculum slice identifier
    update_count: int  # Number of updates applied to this slice
    weights: Dict[str, float]  # Feature name -> weight value
    clamped: bool = False  # Whether any clamping was applied
    clamp_count: int = 0  # Number of times clamping was triggered
    
    def __post_init__(self):
        """Validate snapshot on creation."""
        if self.update_count < 0:
            raise ValueError(f"update_count must be non-negative, got {self.update_count}")
        if self.clamp_count < 0:
            raise ValueError(f"clamp_count must be non-negative, got {self.clamp_count}")
        if not isinstance(self.weights, dict):
            raise TypeError("weights must be a dictionary")


def build_policy_telemetry_snapshot(snapshot: PolicyStateSnapshot) -> Dict[str, Any]:
    """
    Build telemetry snapshot from policy state.
    
    Extracts key metrics for monitoring policy health and evolution:
    - Weight norms (L1, L2) for magnitude tracking
    - Sparsity (nonzero weights count)
    - Top-K features by weight (positive and negative)
    - Clamping/guardrail status
    
    Args:
        snapshot: Immutable policy state snapshot
        
    Returns:
        Dictionary with telemetry metrics:
        - schema_version: str
        - slice_name: str
        - update_count: int
        - weight_norm_l1: float
        - weight_norm_l2: float
        - nonzero_weights: int
        - top_k_positive_features: List[Tuple[str, float]]
        - top_k_negative_features: List[Tuple[str, float]]
        - clamped: bool
        - clamp_count: int
        
    Example:
        >>> snapshot = PolicyStateSnapshot(
        ...     schema_version="v1",
        ...     slice_name="baseline",
        ...     update_count=10,
        ...     weights={"len": -0.5, "depth": 0.3, "success": 1.2},
        ...     clamped=False,
        ...     clamp_count=0
        ... )
        >>> telemetry = build_policy_telemetry_snapshot(snapshot)
        >>> telemetry["weight_norm_l2"]  # doctest: +ELLIPSIS
        1.34...
    """
    weights = snapshot.weights
    weight_values = np.array(list(weights.values()), dtype=float)
    
    # Compute norms
    weight_norm_l1 = float(np.sum(np.abs(weight_values)))
    weight_norm_l2 = float(np.sqrt(np.sum(weight_values ** 2)))
    
    # Count nonzero weights (using small epsilon for float comparison)
    nonzero_weights = int(np.sum(np.abs(weight_values) > 1e-9))
    
    # Extract top-K positive and negative features (K=3 by default)
    K = 3
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    # Top K positive (largest values)
    top_k_positive = [(name, float(value)) for name, value in sorted_items[:K] if value > 1e-9]
    
    # Top K negative (smallest/most negative values)
    top_k_negative = [(name, float(value)) for name, value in sorted_items[-K:] if value < -1e-9]
    top_k_negative.reverse()  # Most negative first
    
    return {
        "schema_version": snapshot.schema_version,
        "slice_name": snapshot.slice_name,
        "update_count": snapshot.update_count,
        "weight_norm_l1": weight_norm_l1,
        "weight_norm_l2": weight_norm_l2,
        "nonzero_weights": nonzero_weights,
        "top_k_positive_features": top_k_positive,
        "top_k_negative_features": top_k_negative,
        "clamped": snapshot.clamped,
        "clamp_count": snapshot.clamp_count,
    }


def compare_policy_snapshots(
    before: PolicyStateSnapshot,
    after: PolicyStateSnapshot
) -> Dict[str, Any]:
    """
    Compare two policy snapshots to detect changes.
    
    Computes:
    - L2 and L1 distance between weight vectors
    - Number of sign flips (features that changed sign)
    - Features with largest absolute changes
    
    Args:
        before: Earlier policy state
        after: Later policy state
        
    Returns:
        Dictionary with comparison metrics:
        - slice_name: str (from after snapshot)
        - l2_distance: float
        - l1_distance: float
        - num_sign_flips: int
        - top_features_changed: List[Tuple[str, float, float, float]]
            Each tuple: (feature_name, before_weight, after_weight, delta)
    """
    # Ensure both snapshots cover the same features
    all_features = set(before.weights.keys()) | set(after.weights.keys())
    
    # Build aligned weight vectors
    before_vec = np.array([before.weights.get(f, 0.0) for f in sorted(all_features)])
    after_vec = np.array([after.weights.get(f, 0.0) for f in sorted(all_features)])
    
    # Compute distances
    l2_distance = float(np.sqrt(np.sum((after_vec - before_vec) ** 2)))
    l1_distance = float(np.sum(np.abs(after_vec - before_vec)))
    
    # Count sign flips (excluding zero crossings from/to zero)
    sign_flips = 0
    for f in all_features:
        before_val = before.weights.get(f, 0.0)
        after_val = after.weights.get(f, 0.0)
        # Only count as flip if both values are non-zero and signs differ
        if abs(before_val) > 1e-9 and abs(after_val) > 1e-9:
            if (before_val > 0) != (after_val > 0):
                sign_flips += 1
    
    # Find features with largest changes
    deltas = []
    for f in all_features:
        before_val = before.weights.get(f, 0.0)
        after_val = after.weights.get(f, 0.0)
        delta = after_val - before_val
        if abs(delta) > 1e-9:
            deltas.append((f, float(before_val), float(after_val), float(delta)))
    
    # Sort by absolute delta, take top K
    K = 3
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)
    top_features_changed = deltas[:K]
    
    return {
        "slice_name": after.slice_name,
        "l2_distance": l2_distance,
        "l1_distance": l1_distance,
        "num_sign_flips": sign_flips,
        "top_features_changed": top_features_changed,
    }


def build_policy_drift_radar(comparisons: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build drift radar from sequence of policy comparisons.
    
    Aggregates comparison data across multiple slices to detect:
    - Slices with large drift (L2 distance above threshold)
    - Maximum drift magnitude across all slices
    - Per-slice drift metrics
    
    Args:
        comparisons: Sequence of comparison dictionaries from compare_policy_snapshots
        
    Returns:
        Dictionary with drift radar data:
        - slices: List[Dict] - per-slice metrics (slice_name, l2_distance, l1_distance, 
                                num_sign_flips, top_features_changed)
        - slices_with_large_drift: List[str] - slice names exceeding drift threshold
        - max_l2_distance: float - largest L2 distance across all slices
        - num_slices_analyzed: int - number of slices in comparisons
        
    Drift threshold: L2 distance > 0.5 (indicates significant policy change)
    """
    if not comparisons:
        return {
            "slices": [],
            "slices_with_large_drift": [],
            "max_l2_distance": 0.0,
            "num_slices_analyzed": 0,
        }
    
    # Drift threshold for "large" changes
    DRIFT_THRESHOLD = 0.5
    
    # Extract per-slice metrics
    slices = []
    large_drift_slices = []
    max_l2 = 0.0
    
    for comp in comparisons:
        slice_name = comp["slice_name"]
        l2_distance = comp["l2_distance"]
        l1_distance = comp["l1_distance"]
        num_sign_flips = comp["num_sign_flips"]
        top_features_changed = comp["top_features_changed"]
        
        slices.append({
            "slice_name": slice_name,
            "l2_distance": l2_distance,
            "l1_distance": l1_distance,
            "num_sign_flips": num_sign_flips,
            "top_features_changed": top_features_changed,
        })
        
        # Track max L2
        if l2_distance > max_l2:
            max_l2 = l2_distance
        
        # Check for large drift
        if l2_distance > DRIFT_THRESHOLD:
            large_drift_slices.append(slice_name)
    
    return {
        "slices": slices,
        "slices_with_large_drift": large_drift_slices,
        "max_l2_distance": max_l2,
        "num_slices_analyzed": len(comparisons),
    }


def summarize_policy_for_governance(radar: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate governance summary from drift radar.
    
    Provides high-level status for governance/oversight:
    - Whether any slice shows large drift
    - List of affected slices
    - Overall status classification
    
    Status levels:
    - "OK": No large drift detected
    - "ATTENTION": 1-2 slices with large drift
    - "VOLATILE": 3+ slices with large drift
    
    Args:
        radar: Drift radar dictionary from build_policy_drift_radar
        
    Returns:
        Dictionary with governance summary:
        - has_large_drift: bool
        - slices_with_large_drift: List[str]
        - status: str ("OK" | "ATTENTION" | "VOLATILE")
    """
    large_drift_slices = radar["slices_with_large_drift"]
    has_large_drift = len(large_drift_slices) > 0
    
    # Classify status based on number of affected slices
    if len(large_drift_slices) == 0:
        status = "OK"
    elif len(large_drift_slices) <= 2:
        status = "ATTENTION"
    else:
        status = "VOLATILE"
    
    return {
        "has_large_drift": has_large_drift,
        "slices_with_large_drift": large_drift_slices,
        "status": status,
    }


def summarize_policy_for_global_health(radar: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate global health summary from drift radar.
    
    Provides system-wide health indicators:
    - Overall policy surface health
    - Maximum drift magnitude
    - Number of slices analyzed
    - Health status classification
    
    Health status levels:
    - "OK": max_l2_distance <= 0.5
    - "WARN": 0.5 < max_l2_distance <= 1.0
    - "HOT": max_l2_distance > 1.0
    
    Args:
        radar: Drift radar dictionary from build_policy_drift_radar
        
    Returns:
        Dictionary with global health summary:
        - policy_surface_ok: bool
        - max_l2_distance: float
        - num_slices_analyzed: int
        - status: str ("OK" | "WARN" | "HOT")
    """
    max_l2 = radar["max_l2_distance"]
    num_slices = radar["num_slices_analyzed"]
    
    # Policy surface is OK if max drift is below warning threshold
    WARN_THRESHOLD = 0.5
    HOT_THRESHOLD = 1.0
    
    policy_surface_ok = max_l2 <= WARN_THRESHOLD
    
    # Classify health status
    if max_l2 <= WARN_THRESHOLD:
        status = "OK"
    elif max_l2 <= HOT_THRESHOLD:
        status = "WARN"
    else:
        status = "HOT"
    
    return {
        "policy_surface_ok": policy_surface_ok,
        "max_l2_distance": max_l2,
        "num_slices_analyzed": num_slices,
        "status": status,
    }
