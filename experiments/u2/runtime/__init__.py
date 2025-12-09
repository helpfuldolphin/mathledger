"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Module — The Thin Waist
===================================

This module provides the core runtime utilities for U2 uplift experiments.
It is designed as the "thin waist" of the Phase II architecture: all
experiment runners and analysis tools should depend on these primitives.

GUARANTEES
----------
1. **Determinism**: All functions produce identical outputs for identical inputs.
   No global state is modified; all randomness is instance-scoped.

2. **Stable API**: Symbols exported in `__all__` are considered stable.
   Breaking changes require a version bump and deprecation period.

3. **No Side Effects on Import**: Importing this module performs no I/O,
   no network calls, and no state mutations.

4. **Input Validation**: Invalid inputs raise early with clear messages.
   Never silent failures or implicit coercion.

INTERNAL (may change without notice)
------------------------------------
- Private functions (prefixed with `_`)
- Implementation details within each submodule
- Error message formatting specifics

MODULES
-------
- seed_manager: Deterministic seed schedule generation
- cycle_orchestrator: Per-cycle execution orchestration  
- error_classifier: Runtime error classification utilities
- trace_logger: PHASE II telemetry logging

FEATURE FLAGS
-------------
Runtime behavior can be controlled via feature flags defined in `FEATURE_FLAGS`.
Query flags programmatically via `get_feature_flag()` or via CLI:

    uv run python experiments/u2_runtime_inspect.py --show-feature-flags

Flags are designed for:
- Gradual rollout of new behaviors
- A/B testing in controlled environments
- Safe defaults that preserve existing behavior

USAGE
-----
    from experiments.u2.runtime import (
        generate_seed_schedule,
        CycleState,
        execute_cycle,
        BaselineOrderingStrategy,
        classify_error,
        build_telemetry_record,
        # Feature flags
        get_feature_flag,
        FEATURE_FLAGS,
    )
    
    # Generate deterministic seeds
    schedule = generate_seed_schedule(initial_seed=42, num_cycles=100)
    
    # Check a feature flag
    if get_feature_flag("u2.use_cycle_orchestrator"):
        # Use orchestrator path
        ...
    
    # Execute a cycle through the orchestrator
    state = CycleState(
        cycle=0,
        cycle_seed=schedule.get_seed(0),
        slice_name="arithmetic_simple",
        mode="baseline",
        candidate_items=["1+1", "2+2", "3+3"],
    )
    result = execute_cycle(state, BaselineOrderingStrategy(), substrate_fn)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.u2.runtime.seed_manager import (
    SeedSchedule,
    generate_seed_schedule,
    hash_string,
)
from experiments.u2.runtime.cycle_orchestrator import (
    CycleState,
    CycleResult,
    CycleExecutionError,
    OrderingStrategy,
    BaselineOrderingStrategy,
    RflOrderingStrategy,
    execute_cycle,
    get_ordering_strategy,
)
from experiments.u2.runtime.error_classifier import (
    RuntimeErrorKind,
    ErrorContext,
    classify_error,
    classify_error_with_context,
    build_error_result,
)
from experiments.u2.runtime.trace_logger import (
    PHASE_II_LABEL,
    TelemetryRecord,
    TraceWriter,
    TraceReader,
    build_telemetry_record,
)
from experiments.u2.runtime.profile_guard import (
    ProfileDriftResult,
    build_runtime_profile_drift_snapshot,
)


# ============================================================================
# Feature Flag System
# ============================================================================

class FeatureFlagStability(Enum):
    """
    Stability status for runtime feature flags.
    
    - STABLE: Flag behavior is locked; changes require deprecation.
    - BETA: Flag may change in minor versions.
    - EXPERIMENTAL: Flag may change or be removed at any time.
    """
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class RuntimeFeatureFlag:
    """
    A runtime feature flag definition.
    
    Feature flags allow controlled rollout of new behaviors without
    breaking existing code paths. All flags default to values that
    preserve current behavior.
    
    Attributes:
        name: Unique flag identifier (e.g., "u2.use_cycle_orchestrator").
        default: Default value when flag is not explicitly set.
        description: Human-readable description of what the flag controls.
        stability: Stability status (STABLE, BETA, EXPERIMENTAL).
    
    Example:
        >>> flag = RuntimeFeatureFlag(
        ...     name="u2.enable_extra_telemetry",
        ...     default=False,
        ...     description="Emit additional telemetry events for debugging.",
        ...     stability=FeatureFlagStability.EXPERIMENTAL,
        ... )
    """
    name: str
    default: Any
    description: str
    stability: FeatureFlagStability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "default": self.default,
            "description": self.description,
            "stability": self.stability.value,
        }


# Registry of all runtime feature flags
# NOTE: Default values MUST preserve current behavior (no semantic changes)
FEATURE_FLAGS: Dict[str, RuntimeFeatureFlag] = {
    "u2.use_cycle_orchestrator": RuntimeFeatureFlag(
        name="u2.use_cycle_orchestrator",
        default=True,
        description="Route all ordering logic through cycle_orchestrator (INV-RUN-1).",
        stability=FeatureFlagStability.STABLE,
    ),
    "u2.enable_extra_telemetry": RuntimeFeatureFlag(
        name="u2.enable_extra_telemetry",
        default=False,
        description="Emit additional telemetry events for debugging and analysis.",
        stability=FeatureFlagStability.EXPERIMENTAL,
    ),
    "u2.strict_input_validation": RuntimeFeatureFlag(
        name="u2.strict_input_validation",
        default=True,
        description="Raise ValueError on invalid inputs (vs. silent coercion).",
        stability=FeatureFlagStability.STABLE,
    ),
    "u2.trace_hash_chain": RuntimeFeatureFlag(
        name="u2.trace_hash_chain",
        default=False,
        description="Enable hash-chaining for tamper-evident trace logs.",
        stability=FeatureFlagStability.BETA,
    ),
}

# Runtime flag overrides (set programmatically or via environment)
_flag_overrides: Dict[str, Any] = {}


def get_feature_flag(name: str, default: Optional[Any] = None) -> Any:
    """
    Get the current value of a feature flag.
    
    Resolution order:
    1. Programmatic override (via set_feature_flag)
    2. Registry default
    3. Provided default parameter
    
    Args:
        name: Feature flag name (e.g., "u2.use_cycle_orchestrator").
        default: Fallback if flag is not in registry (None by default).
    
    Returns:
        The flag's current value.
    
    Example:
        >>> get_feature_flag("u2.use_cycle_orchestrator")
        True
        >>> get_feature_flag("unknown.flag", default=False)
        False
    """
    # Check overrides first
    if name in _flag_overrides:
        return _flag_overrides[name]
    
    # Check registry
    if name in FEATURE_FLAGS:
        return FEATURE_FLAGS[name].default
    
    # Fall back to provided default
    return default


def set_feature_flag(name: str, value: Any) -> None:
    """
    Override a feature flag value at runtime.
    
    This is primarily for testing. In production, flags should be
    controlled via configuration or environment variables.
    
    Args:
        name: Feature flag name.
        value: New value for the flag.
    
    Raises:
        ValueError: If the flag name is not in the registry.
    
    Example:
        >>> set_feature_flag("u2.enable_extra_telemetry", True)
        >>> get_feature_flag("u2.enable_extra_telemetry")
        True
    """
    if name not in FEATURE_FLAGS:
        raise ValueError(
            f"Unknown feature flag: {name}. "
            f"Valid flags: {sorted(FEATURE_FLAGS.keys())}"
        )
    _flag_overrides[name] = value


def reset_feature_flags() -> None:
    """
    Reset all feature flag overrides to registry defaults.
    
    Call this in test teardown to ensure clean state.
    """
    _flag_overrides.clear()


def list_feature_flags() -> Dict[str, RuntimeFeatureFlag]:
    """
    Get a copy of the feature flag registry.
    
    Returns:
        Dictionary mapping flag names to RuntimeFeatureFlag objects.
    """
    return dict(FEATURE_FLAGS)


# ============================================================================
# Runtime Health Snapshot
# ============================================================================

# Schema version for health snapshot format
HEALTH_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def build_runtime_health_snapshot(
    config_path: Optional[str] = None,
    config_validation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable runtime health snapshot.
    
    This function produces a deterministic, JSON-serializable snapshot of
    the current runtime state including version, active flags, and config
    validation status.
    
    Args:
        config_path: Optional path to config file for validation.
        config_validation_result: Pre-computed validation result (to avoid re-validation).
    
    Returns:
        Dictionary with:
        - schema_version: Health snapshot format version
        - runtime_version: Current runtime version (e.g., "1.3.0")
        - active_flags: {flag_name: current_value}
        - flag_stabilities: {flag_name: "stable"|"beta"|"experimental"}
        - config_valid: bool (True if no config_path provided or validation passed)
        - config_errors: list of error dicts from dry-run validation
    
    Example:
        >>> snapshot = build_runtime_health_snapshot("config/test.yaml")
        >>> snapshot["runtime_version"]
        '1.3.0'
        >>> snapshot["config_valid"]
        True
    """
    # Gather active flag values
    active_flags: Dict[str, Any] = {}
    flag_stabilities: Dict[str, str] = {}
    
    for name, flag in sorted(FEATURE_FLAGS.items()):
        active_flags[name] = get_feature_flag(name)
        flag_stabilities[name] = flag.stability.value
    
    # Config validation
    config_valid = True
    config_errors: list = []
    
    if config_validation_result is not None:
        # Use pre-computed result
        config_valid = config_validation_result.get("status") == "OK"
        config_errors = config_validation_result.get("errors", [])
    elif config_path is not None:
        # Lazy import to avoid circular dependency
        from pathlib import Path
        
        # Inline validation (simplified - full validation in CLI)
        path = Path(config_path)
        if not path.exists():
            config_valid = False
            config_errors = [{"code": "CONFIG_NOT_FOUND", "message": f"Config not found: {config_path}"}]
    
    return {
        "schema_version": HEALTH_SNAPSHOT_SCHEMA_VERSION,
        "runtime_version": __version__,
        "active_flags": active_flags,
        "flag_stabilities": flag_stabilities,
        "config_valid": config_valid,
        "config_errors": config_errors,
    }


# ============================================================================
# Flag Policy Guard
# ============================================================================

# Valid environment contexts for policy validation
VALID_ENV_CONTEXTS = frozenset({"dev", "ci", "prod"})


@dataclass(frozen=True)
class FlagPolicyViolation:
    """
    A single flag policy violation.
    
    Attributes:
        flag_name: Name of the violating flag.
        stability: Stability level of the flag.
        current_value: Current value of the flag.
        reason: Human-readable explanation of the violation.
    """
    flag_name: str
    stability: str
    current_value: Any
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "flag_name": self.flag_name,
            "stability": self.stability,
            "current_value": self.current_value,
            "reason": self.reason,
        }


def validate_flag_policy(env_context: str) -> Dict[str, Any]:
    """
    Validate feature flag settings against environment policy.
    
    Policy rules:
    - STABLE flags: May be freely toggled in any environment.
    - BETA flags: Allowed only when env_context is "dev" or with explicit override.
    - EXPERIMENTAL flags: Must not be ON in "ci" or "prod" contexts.
    
    Args:
        env_context: Environment context ("dev", "ci", or "prod").
    
    Returns:
        Dictionary with:
        - policy_ok: bool (True if no violations)
        - violations: list of FlagPolicyViolation dicts
        - env_context: The validated environment context
        - flags_checked: Number of flags checked
    
    Raises:
        ValueError: If env_context is not one of "dev", "ci", "prod".
    
    Example:
        >>> result = validate_flag_policy("prod")
        >>> result["policy_ok"]
        True
        >>> result["violations"]
        []
    """
    if env_context not in VALID_ENV_CONTEXTS:
        raise ValueError(
            f"Invalid env_context: {env_context}. "
            f"Must be one of: {sorted(VALID_ENV_CONTEXTS)}"
        )
    
    violations: list = []
    
    for name, flag in sorted(FEATURE_FLAGS.items()):
        current_value = get_feature_flag(name)
        stability = flag.stability
        
        # STABLE flags: Always allowed
        if stability == FeatureFlagStability.STABLE:
            continue
        
        # BETA flags: Only allowed in dev, or if OFF
        if stability == FeatureFlagStability.BETA:
            if current_value and env_context != "dev":
                violations.append(FlagPolicyViolation(
                    flag_name=name,
                    stability=stability.value,
                    current_value=current_value,
                    reason=f"BETA flag is ON in '{env_context}' (only allowed in 'dev')",
                ))
        
        # EXPERIMENTAL flags: Must be OFF in ci/prod
        elif stability == FeatureFlagStability.EXPERIMENTAL:
            if current_value and env_context in ("ci", "prod"):
                violations.append(FlagPolicyViolation(
                    flag_name=name,
                    stability=stability.value,
                    current_value=current_value,
                    reason=f"EXPERIMENTAL flag is ON in '{env_context}' (not allowed)",
                ))
    
    return {
        "policy_ok": len(violations) == 0,
        "violations": [v.to_dict() for v in violations],
        "env_context": env_context,
        "flags_checked": len(FEATURE_FLAGS),
    }


# ============================================================================
# Runtime Profile Catalog
# ============================================================================

@dataclass(frozen=True)
class RuntimeProfile:
    """
    A runtime profile that defines allowed/required/forbidden flag configurations.
    
    Profiles move policy from hard-coded tables to a clean abstraction that
    can be versioned, tested, and extended.
    
    Attributes:
        name: Profile identifier (e.g., "dev-default", "ci-strict", "prod-hardened").
        description: Human-readable description of the profile.
        expected_env_context: Expected environment context ("dev", "ci", "prod").
        allowed_flags: Set of flag names that may be ON in this profile.
        required_flags: Set of flag names that must be ON in this profile.
        forbidden_flags: Set of flag names that must be OFF in this profile.
        forbidden_combinations: List of tuples of flag names that cannot be ON together.
    
    Example:
        >>> profile = RuntimeProfile(
        ...     name="prod-hardened",
        ...     description="Production profile with strict flag controls",
        ...     expected_env_context="prod",
        ...     required_flags={"u2.use_cycle_orchestrator"},
        ...     forbidden_flags={"u2.enable_extra_telemetry"},
        ... )
    """
    name: str
    description: str
    expected_env_context: str
    allowed_flags: Set[str] = field(default_factory=set)
    required_flags: Set[str] = field(default_factory=set)
    forbidden_flags: Set[str] = field(default_factory=set)
    forbidden_combinations: List[Tuple[str, ...]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate profile configuration."""
        if self.expected_env_context not in VALID_ENV_CONTEXTS:
            raise ValueError(
                f"Invalid expected_env_context: {self.expected_env_context}. "
                f"Must be one of: {sorted(VALID_ENV_CONTEXTS)}"
            )
        
        # Check for conflicts
        if self.required_flags & self.forbidden_flags:
            raise ValueError(
                f"Profile {self.name}: flags cannot be both required and forbidden"
            )
        
        # Check forbidden combinations reference valid flags
        all_flags = set(FEATURE_FLAGS.keys())
        for combo in self.forbidden_combinations:
            invalid = set(combo) - all_flags
            if invalid:
                raise ValueError(
                    f"Profile {self.name}: forbidden_combinations references unknown flags: {invalid}"
                )


# Built-in runtime profiles
RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "dev-default": RuntimeProfile(
        name="dev-default",
        description="Default development profile with permissive flag settings",
        expected_env_context="dev",
        allowed_flags=set(FEATURE_FLAGS.keys()),  # All flags allowed
        required_flags={"u2.use_cycle_orchestrator"},  # Must use orchestrator
        forbidden_flags=set(),  # None forbidden
        forbidden_combinations=[],
    ),
    "ci-strict": RuntimeProfile(
        name="ci-strict",
        description="CI profile with strict flag controls (no experimental features)",
        expected_env_context="ci",
        allowed_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation", "u2.trace_hash_chain"},
        required_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation"},
        forbidden_flags={"u2.enable_extra_telemetry"},  # No experimental flags
        forbidden_combinations=[],
    ),
    "prod-hardened": RuntimeProfile(
        name="prod-hardened",
        description="Production profile with maximum hardening (only stable features)",
        expected_env_context="prod",
        allowed_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation"},
        required_flags={"u2.use_cycle_orchestrator", "u2.strict_input_validation"},
        forbidden_flags={"u2.enable_extra_telemetry", "u2.trace_hash_chain"},  # No beta/experimental
        forbidden_combinations=[],
    ),
}


def load_runtime_profile(name: str) -> RuntimeProfile:
    """
    Load a runtime profile by name.
    
    Args:
        name: Profile name (e.g., "dev-default", "ci-strict", "prod-hardened").
    
    Returns:
        RuntimeProfile object.
    
    Raises:
        ValueError: If profile name is not found.
    
    Example:
        >>> profile = load_runtime_profile("prod-hardened")
        >>> profile.expected_env_context
        'prod'
    """
    if name not in RUNTIME_PROFILES:
        raise ValueError(
            f"Unknown runtime profile: {name}. "
            f"Available profiles: {sorted(RUNTIME_PROFILES.keys())}"
        )
    return RUNTIME_PROFILES[name]


def evaluate_runtime_profile(
    profile: RuntimeProfile,
    snapshot: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate runtime state against a profile.
    
    Checks:
    - Required flags are ON
    - Forbidden flags are OFF
    - Forbidden combinations are not present
    - Environment context matches expected
    
    Args:
        profile: RuntimeProfile to evaluate against.
        snapshot: Result from build_runtime_health_snapshot().
        policy_result: Result from validate_flag_policy().
    
    Returns:
        Dictionary with:
        - profile_ok: bool (True if all checks pass)
        - violations: list of violation strings
        - status: "OK" | "WARN" | "BLOCK"
        - profile_name: Name of the evaluated profile
    
    Example:
        >>> snapshot = build_runtime_health_snapshot()
        >>> policy = validate_flag_policy("prod")
        >>> profile = load_runtime_profile("prod-hardened")
        >>> eval_result = evaluate_runtime_profile(profile, snapshot, policy)
        >>> eval_result["profile_ok"]
        True
    """
    violations: list = []
    active_flags = snapshot.get("active_flags", {})
    
    # Check required flags
    for flag_name in profile.required_flags:
        if not active_flags.get(flag_name, False):
            violations.append(f"Required flag '{flag_name}' is OFF")
    
    # Check forbidden flags
    for flag_name in profile.forbidden_flags:
        if active_flags.get(flag_name, False):
            violations.append(f"Forbidden flag '{flag_name}' is ON")
    
    # Check forbidden combinations
    for combo in profile.forbidden_combinations:
        if all(active_flags.get(flag, False) for flag in combo):
            combo_str = " + ".join(combo)
            violations.append(f"Forbidden combination: {combo_str} are all ON")
    
    # Check environment context
    env_context = policy_result.get("env_context", "")
    if env_context != profile.expected_env_context:
        violations.append(
            f"Environment mismatch: expected '{profile.expected_env_context}', got '{env_context}'"
        )
    
    # Determine status
    profile_ok = len(violations) == 0
    
    if not profile_ok:
        # Block if in prod/ci with violations
        if env_context in ("ci", "prod"):
            status = "BLOCK"
        else:
            status = "WARN"
    else:
        status = "OK"
    
    return {
        "profile_ok": profile_ok,
        "violations": violations,
        "status": status,
        "profile_name": profile.name,
    }


# ============================================================================
# Fail-Safe Runtime Modes
# ============================================================================

def derive_runtime_fail_safe_action(
    profile_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Derive a fail-safe action recommendation from profile evaluation.
    
    This function is advisory-only but designed so CI/pipelines can easily
    turn it into a hard gate.
    
    Args:
        profile_eval: Result from evaluate_runtime_profile().
    
    Returns:
        Dictionary with:
        - action: "ALLOW" | "SAFE_DEGRADE" | "NO_RUN"
        - reason: Short neutral explanation
    
    Action determination:
    - ALLOW: Profile passes (profile_ok=True)
    - SAFE_DEGRADE: Profile fails but violations are non-critical (e.g., missing optional flags)
    - NO_RUN: Profile fails with critical violations (e.g., forbidden flags ON, required flags OFF)
    
    Example:
        >>> eval_result = {"profile_ok": False, "violations": ["Required flag 'x' is OFF"]}
        >>> action = derive_runtime_fail_safe_action(eval_result)
        >>> action["action"]
        'NO_RUN'
    """
    profile_ok = profile_eval.get("profile_ok", False)
    violations = profile_eval.get("violations", [])
    status = profile_eval.get("status", "UNKNOWN")
    
    if profile_ok:
        return {
            "action": "ALLOW",
            "reason": "Profile evaluation passed",
        }
    
    # Classify violations by severity
    critical_keywords = ["forbidden", "required", "environment mismatch"]
    has_critical = any(
        any(keyword in v.lower() for keyword in critical_keywords)
        for v in violations
    )
    
    if has_critical or status == "BLOCK":
        return {
            "action": "NO_RUN",
            "reason": f"Critical profile violations: {len(violations)} issue(s)",
        }
    
    # Non-critical violations (e.g., missing optional flags)
    return {
        "action": "SAFE_DEGRADE",
        "reason": f"Non-critical profile violations: {len(violations)} issue(s)",
    }


# ============================================================================
# Global Health Hook
# ============================================================================

def summarize_runtime_for_global_health(
    snapshot: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize runtime state for inclusion in global_health.json.
    
    This function produces a compact summary suitable for the director
    console's global health view.
    
    Args:
        snapshot: Result from build_runtime_health_snapshot()
        policy_result: Result from validate_flag_policy()
    
    Returns:
        Dictionary with:
        - runtime_ok: bool (True if runtime version valid and config ok)
        - flag_policy_ok: bool (True if no policy violations)
        - beta_flags_active: list of active BETA flag names
        - experimental_flags_active: list of active EXPERIMENTAL flag names
        - status: "OK" | "WARN" | "BLOCK"
    
    Status determination:
    - BLOCK: Config invalid or policy violations in prod/ci
    - WARN: Beta/experimental flags active but policy allows
    - OK: Everything nominal
    
    Example:
        >>> snapshot = build_runtime_health_snapshot()
        >>> policy = validate_flag_policy("dev")
        >>> summary = summarize_runtime_for_global_health(snapshot, policy)
        >>> summary["status"]
        'OK'
    """
    # Check runtime basics
    runtime_ok = bool(snapshot.get("runtime_version")) and snapshot.get("config_valid", True)
    
    # Check flag policy
    flag_policy_ok = policy_result.get("policy_ok", True)
    
    # Identify active beta/experimental flags
    active_flags = snapshot.get("active_flags", {})
    flag_stabilities = snapshot.get("flag_stabilities", {})
    
    beta_flags_active: list = []
    experimental_flags_active: list = []
    
    for name, value in active_flags.items():
        if value:  # Flag is ON
            stability = flag_stabilities.get(name, "")
            if stability == "beta":
                beta_flags_active.append(name)
            elif stability == "experimental":
                experimental_flags_active.append(name)
    
    # Determine overall status
    env_context = policy_result.get("env_context", "dev")
    
    if not runtime_ok:
        status = "BLOCK"
    elif not flag_policy_ok:
        # Policy violations are blocking in prod/ci
        if env_context in ("ci", "prod"):
            status = "BLOCK"
        else:
            status = "WARN"
    elif experimental_flags_active or beta_flags_active:
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "runtime_ok": runtime_ok,
        "flag_policy_ok": flag_policy_ok,
        "beta_flags_active": sorted(beta_flags_active),
        "experimental_flags_active": sorted(experimental_flags_active),
        "status": status,
        "runtime_version": snapshot.get("runtime_version"),
        "env_context": env_context,
    }


# ============================================================================
# Director Console Integration
# ============================================================================

def build_runtime_director_panel(
    snapshot: Dict[str, Any],
    policy_result: Dict[str, Any],
    profile_eval: Optional[Dict[str, Any]] = None,
    fail_safe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a compact Director console panel object for dashboard rendering.
    
    This function produces a stable, JSON-safe object suitable for direct
    rendering in a global health dashboard.
    
    Args:
        snapshot: Result from build_runtime_health_snapshot().
        policy_result: Result from validate_flag_policy().
        profile_eval: Optional result from evaluate_runtime_profile().
        fail_safe: Optional result from derive_runtime_fail_safe_action().
    
    Returns:
        Dictionary with:
        - runtime_version: Current runtime version
        - env_context: Environment context (dev/ci/prod)
        - profile_name: Name of evaluated profile (if provided)
        - status_light: "GREEN" | "YELLOW" | "RED"
        - action: "ALLOW" | "SAFE_DEGRADE" | "NO_RUN"
        - key_violations: Top 3 issues (if any)
    
    Example:
        >>> snapshot = build_runtime_health_snapshot()
        >>> policy = validate_flag_policy("prod")
        >>> profile = load_runtime_profile("prod-hardened")
        >>> eval_result = evaluate_runtime_profile(profile, snapshot, policy)
        >>> fail_safe = derive_runtime_fail_safe_action(eval_result)
        >>> panel = build_runtime_director_panel(snapshot, policy, eval_result, fail_safe)
        >>> panel["status_light"]
        'GREEN'
    """
    # Determine status light from summary
    global_summary = summarize_runtime_for_global_health(snapshot, policy_result)
    summary_status = global_summary.get("status", "UNKNOWN")
    
    # Override with profile status if provided
    if profile_eval:
        profile_status = profile_eval.get("status", summary_status)
        # Use more restrictive status
        status_priority = {"BLOCK": 3, "WARN": 2, "OK": 1}
        if status_priority.get(profile_status, 0) > status_priority.get(summary_status, 0):
            summary_status = profile_status
    
    # Map status to light color
    status_to_light = {
        "OK": "GREEN",
        "WARN": "YELLOW",
        "BLOCK": "RED",
    }
    status_light = status_to_light.get(summary_status, "RED")
    
    # Determine action
    if fail_safe:
        action = fail_safe.get("action", "ALLOW")
    else:
        # Derive from status if fail_safe not provided
        if summary_status == "OK":
            action = "ALLOW"
        elif summary_status == "WARN":
            action = "SAFE_DEGRADE"
        else:
            action = "NO_RUN"
    
    # Collect key violations (top 3)
    key_violations: list = []
    
    # From policy violations
    policy_violations = policy_result.get("violations", [])
    for v in policy_violations[:2]:  # Top 2 from policy
        key_violations.append(v.get("reason", "Policy violation"))
    
    # From profile violations
    if profile_eval:
        profile_violations = profile_eval.get("violations", [])
        for v in profile_violations[:1]:  # Top 1 from profile
            key_violations.append(v)
    
    # From config errors
    config_errors = snapshot.get("config_errors", [])
    for err in config_errors[:1]:  # Top 1 from config
        key_violations.append(err.get("message", "Config error"))
    
    # Limit to top 3
    key_violations = key_violations[:3]
    
    return {
        "runtime_version": snapshot.get("runtime_version"),
        "env_context": policy_result.get("env_context", "unknown"),
        "profile_name": profile_eval.get("profile_name") if profile_eval else None,
        "status_light": status_light,
        "action": action,
        "key_violations": key_violations,
    }


__all__ = [
    # seed_manager
    "SeedSchedule",
    "generate_seed_schedule",
    "hash_string",
    # cycle_orchestrator
    "CycleState",
    "CycleResult",
    "OrderingStrategy",
    "BaselineOrderingStrategy",
    "RflOrderingStrategy",
    "execute_cycle",
    "get_ordering_strategy",
    "CycleExecutionError",
    # error_classifier
    "RuntimeErrorKind",
    "ErrorContext",
    "classify_error",
    "classify_error_with_context",
    "build_error_result",
    # trace_logger
    "PHASE_II_LABEL",
    "TelemetryRecord",
    "TraceWriter",
    "TraceReader",
    "build_telemetry_record",
    # feature_flags
    "FeatureFlagStability",
    "RuntimeFeatureFlag",
    "FEATURE_FLAGS",
    "get_feature_flag",
    "set_feature_flag",
    "reset_feature_flags",
    "list_feature_flags",
    # health_snapshot (v1.4.0)
    "HEALTH_SNAPSHOT_SCHEMA_VERSION",
    "build_runtime_health_snapshot",
    # flag_policy (v1.4.0)
    "VALID_ENV_CONTEXTS",
    "FlagPolicyViolation",
    "validate_flag_policy",
    # global_health (v1.4.0)
    "summarize_runtime_for_global_health",
    # runtime_profiles (v1.5.0)
    "RuntimeProfile",
    "RUNTIME_PROFILES",
    "load_runtime_profile",
    "evaluate_runtime_profile",
    # fail_safe (v1.5.0)
    "derive_runtime_fail_safe_action",
    # director_console (v1.5.0)
    "build_runtime_director_panel",
]

# Runtime version for introspection
# Bump on breaking API changes only
__version__ = "1.5.0"
