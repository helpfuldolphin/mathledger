"""
PHASE II — NOT USED IN PHASE I

Runtime Profile Drift Guard
============================

This module provides utilities to detect and classify changes in runtime profiles
across versions, enabling governance of profile evolution.

The drift guard helps ensure that profile changes are intentional and don't
introduce breaking changes that could compromise runtime safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class ProfileDriftResult:
    """
    Result of comparing two runtime profiles for drift.
    
    Attributes:
        changed_fields: List of field names that changed.
        severity: "NONE" | "NON_BREAKING" | "BREAKING"
        status: "OK" | "WARN" | "BLOCK"
        details: Human-readable explanation of the drift.
    """
    changed_fields: List[str]
    severity: str
    status: str
    details: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "changed_fields": self.changed_fields,
            "severity": self.severity,
            "status": self.status,
            "details": self.details,
        }


def build_runtime_profile_drift_snapshot(
    baseline_profile: Any,  # RuntimeProfile
    current_profile: Any,  # RuntimeProfile
) -> Dict[str, Any]:
    """
    Compare two runtime profiles and detect drift.
    
    This function identifies changes between a baseline (previous version)
    and current profile, classifying them by severity and determining
    whether the drift is acceptable.
    
    BREAKING changes:
    - Previously forbidden flags become allowed in prod/ci profiles
    - Experimental flags enter prod-hardened profile
    - Environment context expectations are loosened (e.g., prod → dev)
    - Required flags are removed from prod/ci profiles
    
    NON_BREAKING changes:
    - Additional required flags added (tightening)
    - Additional forbidden flags added (tightening)
    - Environment context expectations tightened (e.g., dev → prod)
    
    NONE:
    - No changes detected
    
    Args:
        baseline_profile: RuntimeProfile from previous version (baseline).
        current_profile: RuntimeProfile from current version (to compare).
    
    Returns:
        Dictionary with:
        - changed_fields: List of field names that changed
        - severity: "NONE" | "NON_BREAKING" | "BREAKING"
        - status: "OK" | "WARN" | "BLOCK"
        - details: Human-readable explanation
    
    Example:
        >>> baseline = load_runtime_profile("prod-hardened")
        >>> current = RuntimeProfile(..., forbidden_flags={...})  # Modified
        >>> drift = build_runtime_profile_drift_snapshot(baseline, current)
        >>> drift["severity"]
        'BREAKING'
    """
    changed_fields: list = []
    breaking_reasons: list = []
    non_breaking_reasons: list = []
    
    # Check name (shouldn't change, but track it)
    if baseline_profile.name != current_profile.name:
        changed_fields.append("name")
        breaking_reasons.append("Profile name changed (should be immutable)")
    
    # Check expected_env_context
    if baseline_profile.expected_env_context != current_profile.expected_env_context:
        changed_fields.append("expected_env_context")
        baseline_env = baseline_profile.expected_env_context
        current_env = current_profile.expected_env_context
        
        # Loosening env context is breaking
        env_strictness = {"prod": 3, "ci": 2, "dev": 1}
        baseline_strict = env_strictness.get(baseline_env, 0)
        current_strict = env_strictness.get(current_env, 0)
        
        if current_strict < baseline_strict:
            breaking_reasons.append(
                f"Environment context loosened: {baseline_env} → {current_env}"
            )
        else:
            non_breaking_reasons.append(
                f"Environment context tightened: {baseline_env} → {current_env}"
            )
    
    # Check required_flags
    baseline_required = set(baseline_profile.required_flags)
    current_required = set(current_profile.required_flags)
    
    removed_required = baseline_required - current_required
    added_required = current_required - baseline_required
    
    if removed_required or added_required:
        changed_fields.append("required_flags")
        
        if removed_required:
            # Removing required flags is breaking in prod/ci
            if baseline_profile.expected_env_context in ("prod", "ci"):
                breaking_reasons.append(
                    f"Required flags removed in {baseline_profile.expected_env_context}: {sorted(removed_required)}"
                )
            else:
                non_breaking_reasons.append(
                    f"Required flags removed: {sorted(removed_required)}"
                )
        
        if added_required:
            non_breaking_reasons.append(
                f"Required flags added (tightening): {sorted(added_required)}"
            )
    
    # Check forbidden_flags
    baseline_forbidden = set(baseline_profile.forbidden_flags)
    current_forbidden = set(current_profile.forbidden_flags)
    
    removed_forbidden = baseline_forbidden - current_forbidden
    added_forbidden = current_forbidden - baseline_forbidden
    
    if removed_forbidden or added_forbidden:
        changed_fields.append("forbidden_flags")
        
        if removed_forbidden:
            # Allowing previously forbidden flags is breaking in prod/ci
            if baseline_profile.expected_env_context in ("prod", "ci"):
                breaking_reasons.append(
                    f"Forbidden flags allowed in {baseline_profile.expected_env_context}: {sorted(removed_forbidden)}"
                )
            else:
                non_breaking_reasons.append(
                    f"Forbidden flags removed: {sorted(removed_forbidden)}"
                )
        
        if added_forbidden:
            non_breaking_reasons.append(
                f"Forbidden flags added (tightening): {sorted(added_forbidden)}"
            )
    
    # Check for experimental flags entering prod-hardened
    if baseline_profile.name == "prod-hardened" and removed_forbidden:
        # Import here to avoid circular dependency
        from experiments.u2.runtime import FEATURE_FLAGS, FeatureFlagStability
        
        experimental_allowed = {
            flag for flag in removed_forbidden
            if flag in FEATURE_FLAGS
            and FEATURE_FLAGS[flag].stability == FeatureFlagStability.EXPERIMENTAL
        }
        
        if experimental_allowed:
            breaking_reasons.append(
                f"Experimental flags allowed in prod-hardened: {sorted(experimental_allowed)}"
            )
    
    # Check allowed_flags (less critical, but track it)
    baseline_allowed = set(baseline_profile.allowed_flags)
    current_allowed = set(current_profile.allowed_flags)
    
    if baseline_allowed != current_allowed:
        changed_fields.append("allowed_flags")
        # Allowed flags changes are generally non-breaking unless they conflict
        # with required/forbidden changes (already handled above)
        if not removed_forbidden and not removed_required:
            non_breaking_reasons.append("Allowed flags changed")
    
    # Check forbidden_combinations
    baseline_combos = set(baseline_profile.forbidden_combinations)
    current_combos = set(current_profile.forbidden_combinations)
    
    if baseline_combos != current_combos:
        changed_fields.append("forbidden_combinations")
        removed_combos = baseline_combos - current_combos
        if removed_combos:
            non_breaking_reasons.append("Forbidden combinations removed")
        else:
            non_breaking_reasons.append("Forbidden combinations added (tightening)")
    
    # Determine severity and status
    if not changed_fields:
        severity = "NONE"
        status = "OK"
        details = "No changes detected"
    elif breaking_reasons:
        severity = "BREAKING"
        status = "BLOCK"
        details = "; ".join(breaking_reasons)
    else:
        severity = "NON_BREAKING"
        status = "WARN"
        details = "; ".join(non_breaking_reasons) if non_breaking_reasons else "Profile tightened"
    
    return {
        "changed_fields": sorted(changed_fields),
        "severity": severity,
        "status": status,
        "details": details,
        "baseline_profile": baseline_profile.name,
        "current_profile": current_profile.name,
    }


__all__ = [
    "ProfileDriftResult",
    "build_runtime_profile_drift_snapshot",
]

