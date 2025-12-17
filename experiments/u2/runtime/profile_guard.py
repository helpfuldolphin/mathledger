"""
Runtime Profile Drift Guard

Detects and classifies changes between two RuntimeProfile definitions,
identifying "breaking" vs. "non-breaking" drift.

This module helps enforce governance over runtime profiles, ensuring that
changes are intentional and their impact is understood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Set, Tuple

from . import RuntimeProfile, FeatureFlagStability, SYNTHETIC_FEATURE_FLAGS, VALID_ENV_CONTEXTS


@dataclass(frozen=True)
class ProfileDriftResult:
    """
    Result of comparing two RuntimeProfiles for drift.

    Attributes:
        status: Overall status ("OK", "WARN", "BLOCK").
        severity: Severity of the drift ("NONE", "NON_BREAKING", "BREAKING").
        baseline_profile_name: Name of the baseline profile.
        current_profile_name: Name of the current profile.
        schema_version: Version of the drift schema.
        breaking_reasons: List of breaking change reasons.
        non_breaking_reasons: List of non-breaking change reasons.
    """

    status: Literal["OK", "WARN", "BLOCK"]
    severity: Literal["NONE", "NON_BREAKING", "BREAKING"]
    baseline_profile_name: str
    current_profile_name: str
    schema_version: str = "1.0.0"
    breaking_reasons: List[str] = field(default_factory=list)
    non_breaking_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "severity": self.severity,
            "breaking_reasons": self.breaking_reasons,
            "non_breaking_reasons": self.non_breaking_reasons,
            "baseline_profile_name": self.baseline_profile_name,
            "current_profile_name": self.current_profile_name,
        }


def _compare_sets(
    baseline: Set[str],
    current: Set[str],
    field_name: str,
    breaking_reasons: List[str],
    non_breaking_reasons: List[str],
    is_breaking_on_removal: bool = False,
    is_breaking_on_addition: bool = False,
) -> Literal["NONE", "NON_BREAKING", "BREAKING"]:
    """Helper to compare two sets and record changes/violations."""
    severity: Literal["NONE", "NON_BREAKING", "BREAKING"] = "NONE"
    removed = baseline - current
    added = current - baseline

    if removed:
        reason = f"Removed from {field_name}: {sorted(list(removed))}"
        if is_breaking_on_removal:
            breaking_reasons.append(reason)
            severity = "BREAKING"
        else:
            non_breaking_reasons.append(reason)
            if severity == "NONE":
                severity = "NON_BREAKING"

    if added:
        reason = f"Added to {field_name}: {sorted(list(added))}"
        if is_breaking_on_addition:
            breaking_reasons.append(reason)
            severity = "BREAKING"
        else:
            non_breaking_reasons.append(reason)
            if severity == "NONE":
                severity = "NON_BREAKING"

    return severity


def _compare_env_context(
    baseline_env: str,
    current_env: str,
    breaking_reasons: List[str],
    non_breaking_reasons: List[str],
) -> Literal["NONE", "NON_BREAKING", "BREAKING"]:
    """Compare environment contexts for loosening."""
    if baseline_env == current_env:
        return "NONE"

    # Environment strictness order: prod > ci > dev
    env_order = {"prod": 0, "ci": 1, "dev": 2}
    baseline_order = env_order.get(baseline_env, 2)
    current_order = env_order.get(current_env, 2)

    # Loosening (higher order number = less strict)
    if current_order > baseline_order:
        breaking_reasons.append(f"Environment context loosened: {baseline_env} -> {current_env}")
        return "BREAKING"

    # Tightening (lower order number = more strict)
    non_breaking_reasons.append(f"Environment context tightened: {baseline_env} -> {current_env}")
    return "NON_BREAKING"


def build_runtime_profile_drift_snapshot(
    baseline_profile: RuntimeProfile,
    current_profile: RuntimeProfile,
) -> Dict[str, Any]:
    """
    Build a snapshot of the drift between two RuntimeProfile definitions.

    This function compares a baseline profile against a current profile
    and identifies changes, classifying their severity.

    Args:
        baseline_profile: The older or reference RuntimeProfile.
        current_profile: The newer or current RuntimeProfile.

    Returns:
        A dictionary representing the ProfileDriftResult.
    """
    breaking_reasons: List[str] = []
    non_breaking_reasons: List[str] = []
    overall_severity: Literal["NONE", "NON_BREAKING", "BREAKING"] = "NONE"

    def update_overall_severity(new_severity: Literal["NONE", "NON_BREAKING", "BREAKING"]):
        nonlocal overall_severity
        if new_severity == "BREAKING":
            overall_severity = "BREAKING"
        elif new_severity == "NON_BREAKING" and overall_severity == "NONE":
            overall_severity = "NON_BREAKING"

    # Compare expected_env_context
    if baseline_profile.expected_env_context != current_profile.expected_env_context:
        severity = _compare_env_context(
            baseline_profile.expected_env_context,
            current_profile.expected_env_context,
            breaking_reasons,
            non_breaking_reasons,
        )
        update_overall_severity(severity)

    # Compare allowed_flags
    # Loosening allowed flags (removing from allowed) is tightening, NON_BREAKING
    # Tightening allowed flags (adding to allowed) is loosening, NON_BREAKING
    severity = _compare_sets(
        set(baseline_profile.allowed_flags),
        set(current_profile.allowed_flags),
        "allowed_flags",
        breaking_reasons,
        non_breaking_reasons,
        is_breaking_on_removal=False,
        is_breaking_on_addition=False,
    )
    update_overall_severity(severity)

    # Compare required_flags
    # Removing a required flag is BREAKING in ci/prod
    # Adding a required flag is NON_BREAKING (tightening)
    is_prod_ci = current_profile.expected_env_context in ("ci", "prod")
    severity = _compare_sets(
        set(baseline_profile.required_flags),
        set(current_profile.required_flags),
        "required_flags",
        breaking_reasons,
        non_breaking_reasons,
        is_breaking_on_removal=is_prod_ci,
        is_breaking_on_addition=False,
    )
    update_overall_severity(severity)

    # Compare forbidden_flags
    # Removing a forbidden flag is BREAKING (loosening)
    # Adding a forbidden flag is NON_BREAKING (tightening)
    severity = _compare_sets(
        set(baseline_profile.forbidden_flags),
        set(current_profile.forbidden_flags),
        "forbidden_flags",
        breaking_reasons,
        non_breaking_reasons,
        is_breaking_on_removal=True,
        is_breaking_on_addition=False,
    )
    update_overall_severity(severity)

    # Check for experimental flags entering prod-hardened
    if current_profile.name == "prod-hardened":
        for flag_name in current_profile.allowed_flags:
            flag_def = SYNTHETIC_FEATURE_FLAGS.get(flag_name)
            if flag_def and flag_def.stability == FeatureFlagStability.EXPERIMENTAL:
                breaking_reasons.append(
                    f"EXPERIMENTAL flag '{flag_name}' is allowed in prod-hardened profile"
                )
                update_overall_severity("BREAKING")
        for flag_name in current_profile.required_flags:
            flag_def = SYNTHETIC_FEATURE_FLAGS.get(flag_name)
            if flag_def and flag_def.stability == FeatureFlagStability.EXPERIMENTAL:
                breaking_reasons.append(
                    f"EXPERIMENTAL flag '{flag_name}' is required in prod-hardened profile"
                )
                update_overall_severity("BREAKING")

    # Determine final status
    status: Literal["OK", "WARN", "BLOCK"] = "OK"
    if overall_severity == "BREAKING":
        status = "BLOCK"
    elif overall_severity == "NON_BREAKING":
        status = "WARN"

    return ProfileDriftResult(
        status=status,
        severity=overall_severity,
        breaking_reasons=breaking_reasons,
        non_breaking_reasons=non_breaking_reasons,
        baseline_profile_name=baseline_profile.name,
        current_profile_name=current_profile.name,
    ).to_dict()

