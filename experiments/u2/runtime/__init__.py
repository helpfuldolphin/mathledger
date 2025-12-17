"""
U2 Runtime "Thin Waist" — Governance & Profile System

SHADOW MODE ONLY: All governance logic is advisory and diagnostic.
No experiment blocking or enforcement. Designed to feed into global
health dashboards and CI logs for observability.

This module provides:
- Runtime profiles (dev-default, ci-strict, prod-hardened)
- Feature flag governance (STABLE/BETA/EXPERIMENTAL)
- Profile drift detection
- Fail-safe action derivation (advisory only)
- Director console integration
- Runtime health snapshots

Version: 1.5.0
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

__version__ = "1.5.0"

# Valid environment contexts
VALID_ENV_CONTEXTS = frozenset(["dev", "ci", "prod"])

# ============================================================================
# Feature Flag System
# ============================================================================


class FeatureFlagStability(str, Enum):
    """Stability level for feature flags."""

    STABLE = "STABLE"
    BETA = "BETA"
    EXPERIMENTAL = "EXPERIMENTAL"


class RuntimeFeatureFlag:
    """Definition of a runtime feature flag."""

    def __init__(
        self,
        name: str,
        default: bool,
        description: str,
        stability: FeatureFlagStability = FeatureFlagStability.STABLE,
    ):
        self.name = name
        self.default = default
        self.description = description
        self.stability = stability

    def __repr__(self) -> str:
        return f"RuntimeFeatureFlag(name={self.name!r}, default={self.default}, stability={self.stability.value})"


# Synthetic feature flag registry for chaos harness
# NOTE: These are synthetic flags for testing only; real flags would come from actual runtime
SYNTHETIC_FEATURE_FLAGS: Dict[str, RuntimeFeatureFlag] = {
    "u2.use_cycle_orchestrator": RuntimeFeatureFlag(
        name="u2.use_cycle_orchestrator",
        default=True,
        description="Route ordering logic through orchestrator",
        stability=FeatureFlagStability.STABLE,
    ),
    "u2.enable_extra_telemetry": RuntimeFeatureFlag(
        name="u2.enable_extra_telemetry",
        default=False,
        description="Emit additional telemetry events",
        stability=FeatureFlagStability.EXPERIMENTAL,
    ),
    "u2.strict_input_validation": RuntimeFeatureFlag(
        name="u2.strict_input_validation",
        default=True,
        description="Raise ValueError on invalid inputs",
        stability=FeatureFlagStability.STABLE,
    ),
    "u2.trace_hash_chain": RuntimeFeatureFlag(
        name="u2.trace_hash_chain",
        default=False,
        description="Enable hash-chaining for trace logs",
        stability=FeatureFlagStability.BETA,
    ),
    "u2.enable_usla_shadow": RuntimeFeatureFlag(
        name="u2.enable_usla_shadow",
        default=False,
        description="Enable USLA shadow mode integration",
        stability=FeatureFlagStability.BETA,
    ),
    "u2.enable_chaos_testing": RuntimeFeatureFlag(
        name="u2.enable_chaos_testing",
        default=False,
        description="Enable chaos testing harness",
        stability=FeatureFlagStability.EXPERIMENTAL,
    ),
}

# ============================================================================
# Runtime Profiles
# ============================================================================


class RuntimeProfile:
    """
    Defines expected runtime behaviors and flag configurations for an environment.

    Profiles are used to evaluate the current runtime state against desired
    invariants, ensuring experiments run in the correct context.
    """

    def __init__(
        self,
        name: str,
        description: str,
        expected_env_context: str,
        allowed_flags: Optional[Set[str]] = None,
        required_flags: Optional[Set[str]] = None,
        forbidden_flags: Optional[Set[str]] = None,
        forbidden_combinations: Optional[List[Tuple[str, ...]]] = None,
    ):
        if expected_env_context not in VALID_ENV_CONTEXTS:
            raise ValueError(
                f"Invalid env_context: {expected_env_context}. Must be one of {sorted(VALID_ENV_CONTEXTS)}"
            )

        self.name = name
        self.description = description
        self.expected_env_context = expected_env_context
        self.allowed_flags = frozenset(allowed_flags or set())
        self.required_flags = frozenset(required_flags or set())
        self.forbidden_flags = frozenset(forbidden_flags or set())
        self.forbidden_combinations = tuple(forbidden_combinations or [])

        # Validate no conflicts
        conflicts = self.allowed_flags & self.forbidden_flags
        if conflicts:
            raise ValueError(f"Flag conflict: {sorted(conflicts)} appears in both allowed and forbidden")

        conflicts = self.required_flags & self.forbidden_flags
        if conflicts:
            raise ValueError(f"Flag conflict: {sorted(conflicts)} appears in both required and forbidden")

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "expected_env_context": self.expected_env_context,
            "allowed_flags": sorted(list(self.allowed_flags)),
            "required_flags": sorted(list(self.required_flags)),
            "forbidden_flags": sorted(list(self.forbidden_flags)),
            "forbidden_combinations": [list(combo) for combo in self.forbidden_combinations],
        }


# Built-in runtime profiles
RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "dev-default": RuntimeProfile(
        name="dev-default",
        description="Permissive profile for development",
        expected_env_context="dev",
        allowed_flags=set(SYNTHETIC_FEATURE_FLAGS.keys()),
        required_flags=set(),
        forbidden_flags=set(),
    ),
    "ci-strict": RuntimeProfile(
        name="ci-strict",
        description="Strict profile for CI environments",
        expected_env_context="ci",
        allowed_flags={
            "u2.use_cycle_orchestrator",
            "u2.strict_input_validation",
            "u2.trace_hash_chain",
            "u2.enable_usla_shadow",
        },
        required_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation"},
        forbidden_flags={
            "u2.enable_extra_telemetry",
            "u2.enable_chaos_testing",
        },
    ),
    "prod-hardened": RuntimeProfile(
        name="prod-hardened",
        description="Hardened profile for production",
        expected_env_context="prod",
        allowed_flags={
            "u2.use_cycle_orchestrator",
            "u2.strict_input_validation",
            "u2.trace_hash_chain",
        },
        required_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation"},
        forbidden_flags={
            "u2.enable_extra_telemetry",
            "u2.enable_chaos_testing",
            "u2.enable_usla_shadow",
        },
    ),
}


def load_runtime_profile(name: str) -> RuntimeProfile:
    """Load a runtime profile by name."""
    if name not in RUNTIME_PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {sorted(RUNTIME_PROFILES.keys())}")
    return RUNTIME_PROFILES[name]


# ============================================================================
# Runtime Health Snapshot
# ============================================================================


def build_runtime_health_snapshot(
    active_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable snapshot of runtime health.

    Args:
        active_flags: Current flag states (defaults to defaults from registry)

    Returns:
        JSON-serializable health snapshot
    """
    if active_flags is None:
        active_flags = {name: flag.default for name, flag in SYNTHETIC_FEATURE_FLAGS.items()}

    flag_stabilities = {
        name: flag.stability.value for name, flag in SYNTHETIC_FEATURE_FLAGS.items()
    }

    return {
        "schema_version": "1.0.0",
        "runtime_version": __version__,
        "active_flags": active_flags,
        "flag_stabilities": flag_stabilities,
        "config_valid": True,
        "config_errors": [],
    }


# ============================================================================
# Flag Policy Guard
# ============================================================================


def validate_flag_policy(env_context: str, active_flags: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """
    Validate feature flag policy against environment context.

    Args:
        env_context: Environment context (dev/ci/prod)
        active_flags: Current flag states

    Returns:
        Policy validation result
    """
    if env_context not in VALID_ENV_CONTEXTS:
        return {
            "policy_ok": False,
            "violations": [f"Invalid env_context: {env_context}"],
        }

    if active_flags is None:
        active_flags = {name: flag.default for name, flag in SYNTHETIC_FEATURE_FLAGS.items()}

    violations = []
    for flag_name, is_on in active_flags.items():
        if not is_on:
            continue
        flag_def = SYNTHETIC_FEATURE_FLAGS.get(flag_name)
        if not flag_def:
            violations.append((flag_name, "UNKNOWN", f"Flag not in registry: {flag_name}"))
            continue

        if flag_def.stability == FeatureFlagStability.EXPERIMENTAL and env_context in ("ci", "prod"):
            violations.append(
                (flag_name, flag_def.stability.value, f"EXPERIMENTAL flag not allowed in {env_context}")
            )
        elif flag_def.stability == FeatureFlagStability.BETA and env_context == "prod":
            violations.append((flag_name, flag_def.stability.value, f"BETA flag not allowed in prod"))

    return {
        "policy_ok": len(violations) == 0,
        "violations": violations,
    }


# ============================================================================
# Profile Evaluation
# ============================================================================


def evaluate_runtime_profile(
    profile: RuntimeProfile,
    snapshot: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a runtime profile against current state.

    Args:
        profile: Runtime profile to evaluate
        snapshot: Runtime health snapshot
        policy_result: Flag policy validation result

    Returns:
        Profile evaluation result
    """
    active_flags = snapshot.get("active_flags", {})
    violations = []

    # Check environment context
    env_context = os.getenv("RFL_ENV_MODE", "dev").split("-")[0] if "RFL_ENV_MODE" in os.environ else "dev"
    if env_context == "phase1":
        env_context = "dev"
    if env_context not in VALID_ENV_CONTEXTS:
        env_context = "dev"

    if env_context != profile.expected_env_context:
        violations.append(
            f"Environment mismatch: expected {profile.expected_env_context}, got {env_context}"
        )

    # Check required flags
    for flag_name in profile.required_flags:
        if not active_flags.get(flag_name, False):
            violations.append(f"Required flag OFF: {flag_name}")

    # Check forbidden flags
    for flag_name in profile.forbidden_flags:
        if active_flags.get(flag_name, False):
            violations.append(f"Forbidden flag ON: {flag_name}")

    # Check forbidden combinations
    for combo in profile.forbidden_combinations:
        if all(active_flags.get(flag_name, False) for flag_name in combo):
            violations.append(f"Forbidden combination active: {combo}")

    # Add policy violations
    policy_violations = policy_result.get("violations", [])
    for flag_name, stability, reason in policy_violations:
        violations.append(f"Policy violation: {reason}")

    # Determine status
    status = "OK"
    if violations:
        # Check for critical violations (forbidden flags, required flags missing)
        critical_keywords = ["forbidden", "required", "environment mismatch"]
        if any(keyword in v.lower() for v in violations for keyword in critical_keywords):
            status = "BLOCK"
        else:
            status = "WARN"

    return {
        "profile_ok": len(violations) == 0,
        "violations": violations,
        "status": status,
    }


# ============================================================================
# Fail-Safe Action Derivation
# ============================================================================


def derive_runtime_fail_safe_action(profile_eval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive fail-safe action from profile evaluation (advisory only).

    Args:
        profile_eval: Profile evaluation result

    Returns:
        Fail-safe action recommendation
    """
    if profile_eval.get("profile_ok", False):
        return {"action": "ALLOW", "reason": "Profile evaluation passed"}

    violations = profile_eval.get("violations", [])
    status = profile_eval.get("status", "WARN")

    # Critical violations -> NO_RUN
    critical_keywords = ["forbidden", "required flag off", "environment mismatch"]
    has_critical = any(keyword in v.lower() for v in violations for keyword in critical_keywords)

    if has_critical or status == "BLOCK":
        return {
            "action": "NO_RUN",
            "reason": f"Critical violations detected: {', '.join(violations[:3])}",
        }

    # Non-critical violations -> SAFE_DEGRADE
    return {
        "action": "SAFE_DEGRADE",
        "reason": f"Non-critical violations: {', '.join(violations[:2])}",
    }


# ============================================================================
# Director Console Integration
# ============================================================================


def build_runtime_director_panel(
    snapshot: Dict[str, Any],
    policy_result: Dict[str, Any],
    profile_eval: Dict[str, Any],
    fail_safe: Dict[str, Any],
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a compact director console panel for global health dashboard.

    Args:
        snapshot: Runtime health snapshot
        policy_result: Flag policy validation result
        profile_eval: Profile evaluation result
        fail_safe: Fail-safe action
        profile_name: Optional profile name (for panel display)

    Returns:
        Director console panel object
    """
    status_light = "GREEN"
    if profile_eval.get("status") == "BLOCK":
        status_light = "RED"
    elif profile_eval.get("status") == "WARN":
        status_light = "YELLOW"

    violations = profile_eval.get("violations", [])
    key_violations = violations[:3] if violations else []

    env_context = os.getenv("RFL_ENV_MODE", "dev")
    if "-" in env_context:
        env_context = env_context.split("-")[0]

    return {
        "schema_version": "1.0.0",
        "runtime_version": snapshot.get("runtime_version", __version__),
        "env_context": env_context,
        "profile_name": profile_name or "unknown",
        "status_light": status_light,
        "action": fail_safe.get("action", "ALLOW"),
        "key_violations": key_violations,
    }


def build_runtime_profile_snapshot_for_first_light(
    profile_name: str,
    chaos_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact runtime profile snapshot for inclusion in First Light evidence.

    This function extracts key metrics from a chaos harness summary and produces
    a minimal, JSON-serializable snapshot suitable for evidence pack inclusion.

    SHADOW MODE: This snapshot is purely observational and does not gate any
    First Light processes. It provides runtime profile governance signals for
    evidence traceability.

    Args:
        profile_name: Name of the runtime profile (e.g., "prod-hardened")
        chaos_summary: Chaos harness output summary from experiments/u2_runtime_chaos.py

    Returns:
        Compact snapshot dictionary with schema_version, profile, status_light,
        profile_stability, and no_run_rate.

    Example:
        >>> chaos_summary = {
        ...     "profile_name": "prod-hardened",
        ...     "total_runs": 100,
        ...     "actions": {"ALLOW": 95, "NO_RUN": 5},
        ...     "profile_stability": 0.95,
        ... }
        >>> snapshot = build_runtime_profile_snapshot_for_first_light("prod-hardened", chaos_summary)
        >>> assert snapshot["profile"] == "prod-hardened"
        >>> assert snapshot["profile_stability"] == 0.95
    """
    if "error" in chaos_summary:
        return {
            "schema_version": "1.0.0",
            "profile": profile_name,
            "status_light": "RED",
            "profile_stability": 0.0,
            "no_run_rate": 1.0,
        }

    profile_stability = chaos_summary.get("profile_stability", 0.0)
    actions = chaos_summary.get("actions", {})
    total_runs = chaos_summary.get("total_runs", 0)

    no_run_count = actions.get("NO_RUN", 0)
    no_run_rate = no_run_count / total_runs if total_runs > 0 else 0.0

    # Determine status light (advisory only)
    if profile_stability >= 0.9 and no_run_rate < 0.05:
        status_light = "GREEN"
    elif no_run_rate >= 0.2 or profile_stability < 0.7:
        status_light = "RED"
    else:
        status_light = "YELLOW"

    return {
        "schema_version": "1.0.0",
        "profile": profile_name,
        "status_light": status_light,
        "profile_stability": round(profile_stability, 4),
        "no_run_rate": round(no_run_rate, 4),
    }


def summarize_runtime_profile_health_for_global_console(
    chaos_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert chaos harness summary into a global health tile.

    Args:
        chaos_summary: Chaos harness output summary

    Returns:
        Global health tile for runtime_profile_health
    """
    if "error" in chaos_summary:
        return {
            "schema_version": "1.0.0",
            "tile_type": "runtime_profile_health",
            "status_light": "RED",
            "profile_name": chaos_summary.get("profile_name", "unknown"),
            "profile_stability": 0.0,
            "no_run_rate": 1.0,
            "headline": f"Chaos harness error: {chaos_summary['error']}",
            "notes": [],
        }

    profile_stability = chaos_summary.get("profile_stability", 0.0)
    actions = chaos_summary.get("actions", {})
    total_runs = chaos_summary.get("total_runs", 0)

    no_run_count = actions.get("NO_RUN", 0)
    no_run_rate = no_run_count / total_runs if total_runs > 0 else 0.0

    # Determine status light (advisory only)
    if profile_stability >= 0.9 and no_run_rate < 0.05:
        status_light = "GREEN"
    elif no_run_rate >= 0.2 or profile_stability < 0.7:
        status_light = "RED"
    else:
        status_light = "YELLOW"

    # Generate headline
    headline = f"Profile {chaos_summary.get('profile_name', 'unknown')}: {profile_stability:.1%} stability, {no_run_rate:.1%} NO_RUN rate"

    # Collect notes
    notes = []
    if no_run_rate > 0.1:
        notes.append(f"High NO_RUN rate: {no_run_rate:.1%}")
    if profile_stability < 0.8:
        notes.append(f"Low profile stability: {profile_stability:.1%}")

    top_violations = chaos_summary.get("top_violations", [])
    if top_violations:
        notes.append(f"Top violation: {top_violations[0][:80]}")

    return {
        "schema_version": "1.0.0",
        "tile_type": "runtime_profile_health",
        "status_light": status_light,
        "profile_name": chaos_summary.get("profile_name", "unknown"),
        "profile_stability": profile_stability,
        "no_run_rate": no_run_rate,
        "headline": headline,
        "notes": notes,
    }


# ============================================================================
# Profile Drift Guard (exported from profile_guard.py)
# ============================================================================

from .profile_guard import (
    ProfileDriftResult,
    build_runtime_profile_drift_snapshot,
)

from .calibration_correlation import (
    build_runtime_profile_calibration_annex,
    correlate_runtime_profile_with_cal_windows,
    annotate_cal_windows_with_runtime_confounding,
    decompose_divergence_components,
)

__all__ = [
    # Version
    "__version__",
    "VALID_ENV_CONTEXTS",
    # Feature Flags
    "FeatureFlagStability",
    "RuntimeFeatureFlag",
    "SYNTHETIC_FEATURE_FLAGS",
    # Profiles
    "RuntimeProfile",
    "RUNTIME_PROFILES",
    "load_runtime_profile",
    # Health & Policy
    "build_runtime_health_snapshot",
    "validate_flag_policy",
    # Evaluation & Actions
    "evaluate_runtime_profile",
    "derive_runtime_fail_safe_action",
    "build_runtime_director_panel",
    "summarize_runtime_profile_health_for_global_console",
    "build_runtime_profile_snapshot_for_first_light",
    # Profile Drift
    "ProfileDriftResult",
    "build_runtime_profile_drift_snapshot",
    # Calibration Correlation
    "correlate_runtime_profile_with_cal_windows",
    "annotate_cal_windows_with_runtime_confounding",
    "decompose_divergence_components",
    "build_runtime_profile_calibration_annex",
]

