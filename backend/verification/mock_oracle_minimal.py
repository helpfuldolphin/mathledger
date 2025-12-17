"""
Minimal Mock Oracle for Negative Control Testing

This module provides a deterministic, contract-driven mock verification oracle
for testing governance logic without touching real verifiers.

NEGATIVE CONTROL PURPOSE:
The mock oracle serves as a calibration anchor point for epistemic risk assessment.
It ensures that epistemic, semantic, TDA (Topological Data Analysis), and First-Light
logic do not overfit to deterministic patterns. By monitoring the mock oracle's behavior,
we can detect when governance systems become too rigid or when they fail to account
for expected variability.

The mock oracle drift tile represents the "expected stochasticity baseline" for
governance logic. This baseline helps identify:
- Overfitting: When governance logic becomes too sensitive to deterministic patterns
- Underfitting: When governance logic fails to account for expected variability
- Drift: When the baseline itself shifts unexpectedly, indicating systemic changes

This negative control harness is essential for maintaining epistemic integrity in
the face of complex, evolving verification systems.

IMPORTANT: This is a test-only harness. It must be explicitly enabled via
MATHLEDGER_ALLOW_MOCK_ORACLE=1 environment variable.

Usage:
    import os
    os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"
    from backend.verification.mock_oracle_minimal import mock_verify
    
    result = mock_verify("(p->q)", profile="uniform")
"""

import hashlib
import os
from typing import Dict, Any, Optional, Literal

# Schema version for result format
MOCK_ORACLE_SCHEMA_VERSION = "1.0.0"

# Fixed salt for deterministic hashing
_MOCK_ORACLE_SALT = b"mathledger_mock_oracle_v1.0.0_salt"

# Profile definitions: (success_pct, failure_pct, abstain_pct)
# These define the distribution of outcomes for each profile
_PROFILES = {
    "uniform": {
        "success_pct": 33.33,
        "failure_pct": 33.33,
        "abstain_pct": 33.34,
        "description": "Uniform distribution across all outcomes",
    },
    "timeout_heavy": {
        "success_pct": 10.0,
        "failure_pct": 20.0,
        "abstain_pct": 70.0,
        "description": "Heavy abstention (timeout-like) profile",
    },
    "invalid_heavy": {
        "success_pct": 5.0,
        "failure_pct": 85.0,
        "abstain_pct": 10.0,
        "description": "Heavy failure (invalid-like) profile",
    },
    "success_heavy": {
        "success_pct": 80.0,
        "failure_pct": 10.0,
        "abstain_pct": 10.0,
        "description": "Heavy success profile",
    },
}

# Verdict types
VERDICT_SUCCESS = "success"
VERDICT_FAILURE = "failure"
VERDICT_ABSTAIN = "abstain"


def _check_mock_oracle_enabled() -> None:
    """
    Check if mock oracle is enabled via environment variable.
    
    Raises:
        RuntimeError: If MATHLEDGER_ALLOW_MOCK_ORACLE is not set to "1"
    """
    if os.getenv("MATHLEDGER_ALLOW_MOCK_ORACLE") != "1":
        raise RuntimeError(
            "Mock oracle is disabled. Set MATHLEDGER_ALLOW_MOCK_ORACLE=1 to enable. "
            "This is a test-only feature and must not be used in production."
        )


def _hash_formula(formula: str) -> int:
    """
    Compute deterministic hash of formula.
    
    Args:
        formula: Normalized formula string
    
    Returns:
        Integer hash value (0-99) for bucket selection
    """
    # Normalize: strip whitespace and convert to lowercase for consistency
    normalized = formula.strip().lower()
    
    # Compute SHA-256 hash with fixed salt
    hash_input = _MOCK_ORACLE_SALT + normalized.encode("utf-8")
    hash_bytes = hashlib.sha256(hash_input).digest()
    
    # Convert to integer and take modulo 100 for bucket selection
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    return hash_int % 100


def _determine_verdict(hash_value: int, profile: str) -> tuple[Literal["success", "failure", "abstain"], Optional[str]]:
    """
    Determine verdict based on hash value and profile distribution.
    
    Args:
        hash_value: Integer hash (0-99)
        profile: Profile name
    
    Returns:
        Tuple of (verdict, abstention_reason)
    """
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(_PROFILES.keys())}")
    
    profile_config = _PROFILES[profile]
    success_pct = profile_config["success_pct"]
    failure_pct = profile_config["failure_pct"]
    abstain_pct = profile_config["abstain_pct"]
    
    # Verify percentages sum to ~100 (allow small floating point errors)
    total = success_pct + failure_pct + abstain_pct
    if abs(total - 100.0) > 0.1:
        raise ValueError(f"Profile '{profile}' percentages sum to {total}, not 100")
    
    # Map hash value to bucket
    # 0 to success_bound-1: success
    # success_bound to success_bound+failure_bound-1: failure
    # success_bound+failure_bound to 99: abstain
    success_bound = int(success_pct)
    failure_bound = int(failure_pct)
    abstain_bound = 100 - success_bound - failure_bound
    
    if hash_value < success_bound:
        return VERDICT_SUCCESS, None
    elif hash_value < success_bound + failure_bound:
        return VERDICT_FAILURE, None
    else:
        # Abstain with deterministic reason based on hash
        reasons = [
            "timeout",
            "resource_exhausted",
            "unknown_formula_structure",
            "insufficient_context",
        ]
        reason_idx = hash_value % len(reasons)
        return VERDICT_ABSTAIN, reasons[reason_idx]


def mock_verify(formula: str, profile: str = "uniform") -> Dict[str, Any]:
    """
    Perform deterministic mock verification of a formula.
    
    This function provides a deterministic, contract-driven mock verification
    result based on the formula hash and selected profile. It never touches
    real verifiers and is purely synthetic.
    
    Args:
        formula: Normalized formula string to "verify"
        profile: Profile name (default: "uniform")
                 Options: "uniform", "timeout_heavy", "invalid_heavy", "success_heavy"
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - profile: Profile name used
        - verdict: "success" | "failure" | "abstain"
        - abstention_reason: Optional[str] (only present if verdict is "abstain")
        - trace_hash: Deterministic hash string for traceability
    
    Raises:
        RuntimeError: If mock oracle is not enabled via MATHLEDGER_ALLOW_MOCK_ORACLE=1
        ValueError: If profile is unknown
    """
    _check_mock_oracle_enabled()
    
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string")
    
    # Compute deterministic hash
    hash_value = _hash_formula(formula)
    
    # Determine verdict
    verdict, abstention_reason = _determine_verdict(hash_value, profile)
    
    # Generate trace hash (full SHA-256 hex for traceability)
    normalized = formula.strip().lower()
    hash_input = _MOCK_ORACLE_SALT + normalized.encode("utf-8")
    trace_hash = hashlib.sha256(hash_input).hexdigest()
    
    # Build result
    result: Dict[str, Any] = {
        "schema_version": MOCK_ORACLE_SCHEMA_VERSION,
        "profile": profile,
        "verdict": verdict,
        "trace_hash": trace_hash,
    }
    
    if abstention_reason:
        result["abstention_reason"] = abstention_reason
    
    return result


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """
    List available mock oracle profiles.
    
    Returns:
        Dictionary mapping profile names to their configurations
    """
    return _PROFILES.copy()


def get_profile_info(profile: str) -> Dict[str, Any]:
    """
    Get information about a specific profile.
    
    Args:
        profile: Profile name
    
    Returns:
        Profile configuration dictionary
    
    Raises:
        ValueError: If profile is unknown
    """
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(_PROFILES.keys())}")
    return _PROFILES[profile].copy()


# ============================================================================
# FLEET SUMMARY & CI EVALUATION
# ============================================================================

# Fleet status constants
FLEET_STATUS_OK = "OK"
FLEET_STATUS_DRIFTING = "DRIFTING"
FLEET_STATUS_BROKEN = "BROKEN"

# Fleet summary schema version
FLEET_SUMMARY_SCHEMA_VERSION = "1.0.0"


def build_mock_oracle_fleet_summary(
    results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a fleet summary from a sequence of mock verification results.
    
    This function aggregates mock verification results to compute fleet-level
    metrics for governance and drift detection.
    
    Args:
        results: Sequence of mock verification result dictionaries from mock_verify()
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - total_queries: Total number of results
        - abstention_rate: Fraction of results with verdict "abstain" (0.0-1.0)
        - invalid_rate: Fraction of results with verdict "failure" (0.0-1.0)
        - success_rate: Fraction of results with verdict "success" (0.0-1.0)
        - status: "OK" | "DRIFTING" | "BROKEN"
        - summary_text: Neutral text describing fleet status
    """
    if len(results) == 0:
        return {
            "schema_version": FLEET_SUMMARY_SCHEMA_VERSION,
            "total_queries": 0,
            "abstention_rate": 0.0,
            "invalid_rate": 0.0,
            "success_rate": 0.0,
            "status": FLEET_STATUS_OK,
            "summary_text": "Fleet summary: No queries processed.",
        }
    
    # Count verdicts
    total = len(results)
    success_count = sum(1 for r in results if r.get("verdict") == VERDICT_SUCCESS)
    failure_count = sum(1 for r in results if r.get("verdict") == VERDICT_FAILURE)
    abstain_count = sum(1 for r in results if r.get("verdict") == VERDICT_ABSTAIN)
    
    # Compute rates
    success_rate = success_count / total
    invalid_rate = failure_count / total
    abstention_rate = abstain_count / total
    
    # Determine status based on rules:
    # - invalid_rate > 0.5 → "BROKEN"
    # - abstention_rate > 0.3 → at least "DRIFTING"
    if invalid_rate > 0.5:
        status = FLEET_STATUS_BROKEN
        summary_text = (
            f"Fleet summary: BROKEN. Invalid rate {invalid_rate:.1%} exceeds threshold (50%). "
            f"Total queries: {total}."
        )
    elif abstention_rate > 0.3:
        status = FLEET_STATUS_DRIFTING
        summary_text = (
            f"Fleet summary: DRIFTING. Abstention rate {abstention_rate:.1%} exceeds threshold (30%). "
            f"Total queries: {total}."
        )
    else:
        status = FLEET_STATUS_OK
        summary_text = (
            f"Fleet summary: OK. Success: {success_rate:.1%}, "
            f"Failure: {invalid_rate:.1%}, Abstain: {abstention_rate:.1%}. "
            f"Total queries: {total}."
        )
    
    return {
        "schema_version": FLEET_SUMMARY_SCHEMA_VERSION,
        "total_queries": total,
        "abstention_rate": abstention_rate,
        "invalid_rate": invalid_rate,
        "success_rate": success_rate,
        "status": status,
        "summary_text": summary_text,
    }


def evaluate_mock_oracle_fleet_for_ci(
    fleet_summary: Dict[str, Any],
) -> tuple[int, str]:
    """
    Evaluate fleet summary for CI integration.
    
    This function provides CI-friendly exit codes and messages based on
    fleet summary status.
    
    Args:
        fleet_summary: Fleet summary dictionary from build_mock_oracle_fleet_summary()
    
    Returns:
        Tuple of (exit_code, message):
        - exit_code: 0 (OK), 1 (DRIFTING/WARN), or 2 (BROKEN/BLOCK)
        - message: Human-readable status message
    """
    status = fleet_summary.get("status", FLEET_STATUS_OK)
    total_queries = fleet_summary.get("total_queries", 0)
    invalid_rate = fleet_summary.get("invalid_rate", 0.0)
    abstention_rate = fleet_summary.get("abstention_rate", 0.0)
    
    if status == FLEET_STATUS_BROKEN:
        exit_code = 2
        message = (
            f"BLOCK: Fleet status BROKEN. Invalid rate {invalid_rate:.1%} exceeds 50%. "
            f"Total queries: {total_queries}."
        )
    elif status == FLEET_STATUS_DRIFTING:
        exit_code = 1
        message = (
            f"WARN: Fleet status DRIFTING. Abstention rate {abstention_rate:.1%} exceeds 30%. "
            f"Total queries: {total_queries}."
        )
    else:
        exit_code = 0
        message = (
            f"OK: Fleet status OK. Success: {fleet_summary.get('success_rate', 0.0):.1%}, "
            f"Failure: {invalid_rate:.1%}, Abstain: {abstention_rate:.1%}. "
            f"Total queries: {total_queries}."
        )
    
    return exit_code, message


# ============================================================================
# DRIFT TILE & INTEGRATION
# ============================================================================

# Status light constants
STATUS_LIGHT_GREEN = "GREEN"
STATUS_LIGHT_YELLOW = "YELLOW"
STATUS_LIGHT_RED = "RED"

# Drift status constants
DRIFT_STATUS_OK = "OK"
DRIFT_STATUS_DRIFTING = "DRIFTING"
DRIFT_STATUS_INVALID_HEAVY = "INVALID-HEAVY"

# Drift tile schema version
DRIFT_TILE_SCHEMA_VERSION = "1.0.0"


def build_mock_oracle_drift_tile(
    fleet_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a drift tile for dashboard display from fleet summary.
    
    This function creates a dashboard-friendly tile that represents the mock
    oracle's drift status. The mock oracle serves as a negative control and
    calibration anchor point for epistemic risk assessment.
    
    Purpose:
    The mock oracle drift tile represents the "expected stochasticity baseline"
    for governance logic. It ensures that epistemic, semantic, TDA, and
    First-Light logic do not overfit to deterministic patterns. By monitoring
    the mock oracle's behavior, we can detect when governance systems become
    too rigid or when they fail to account for expected variability.
    
    Args:
        fleet_summary: Fleet summary dictionary from build_mock_oracle_fleet_summary()
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - drift_status: "OK" | "DRIFTING" | "INVALID-HEAVY"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - abstention_rate: Fraction of abstentions (0.0-1.0)
        - invalid_rate: Fraction of invalid results (0.0-1.0)
        - headline: Brief headline text
        - total_queries: Total number of queries processed
    """
    status = fleet_summary.get("status", FLEET_STATUS_OK)
    abstention_rate = fleet_summary.get("abstention_rate", 0.0)
    invalid_rate = fleet_summary.get("invalid_rate", 0.0)
    total_queries = fleet_summary.get("total_queries", 0)
    
    # Map fleet status to drift status
    if status == FLEET_STATUS_BROKEN:
        drift_status = DRIFT_STATUS_INVALID_HEAVY
        status_light = STATUS_LIGHT_RED
        headline = (
            f"Mock Oracle: INVALID-HEAVY. Invalid rate {invalid_rate:.1%} exceeds threshold. "
            f"Baseline stochasticity compromised."
        )
    elif status == FLEET_STATUS_DRIFTING:
        drift_status = DRIFT_STATUS_DRIFTING
        status_light = STATUS_LIGHT_YELLOW
        headline = (
            f"Mock Oracle: DRIFTING. Abstention rate {abstention_rate:.1%} elevated. "
            f"Expected variability baseline shifting."
        )
    else:
        drift_status = DRIFT_STATUS_OK
        status_light = STATUS_LIGHT_GREEN
        headline = (
            f"Mock Oracle: OK. Baseline stochasticity within expected bounds. "
            f"Success: {fleet_summary.get('success_rate', 0.0):.1%}, "
            f"Failure: {invalid_rate:.1%}, Abstain: {abstention_rate:.1%}."
        )
    
    return {
        "schema_version": DRIFT_TILE_SCHEMA_VERSION,
        "drift_status": drift_status,
        "status_light": status_light,
        "abstention_rate": abstention_rate,
        "invalid_rate": invalid_rate,
        "headline": headline,
        "total_queries": total_queries,
    }


# First Light summary schema version
FIRST_LIGHT_SUMMARY_SCHEMA_VERSION = "1.0.0"

# Control arm calibration summary schema version
CONTROL_ARM_CALIBRATION_SCHEMA_VERSION = "1.0.0"

# Control vs twin panel schema version
CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION = "1.0.0"

# Control arm signal consistency check schema version
CONTROL_ARM_CONSISTENCY_SCHEMA_VERSION = "1.0.0"


def build_first_light_mock_oracle_summary(
    fleet_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a First Light evidence summary from fleet summary.
    
    This function creates a simplified summary for First Light evidence that
    explicitly identifies the mock oracle as a negative control arm. The summary
    provides key metrics for comparison against the treatment arm (real verification).
    
    Args:
        fleet_summary: Fleet summary dictionary from build_mock_oracle_fleet_summary()
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - status: Fleet status ("OK" | "DRIFTING" | "BROKEN")
        - abstention_rate: Fraction of abstentions (0.0-1.0)
        - invalid_rate: Fraction of invalid results (0.0-1.0)
        - total_queries: Total number of queries processed
    """
    return {
        "schema_version": FIRST_LIGHT_SUMMARY_SCHEMA_VERSION,
        "status": fleet_summary.get("status", FLEET_STATUS_OK),
        "abstention_rate": fleet_summary.get("abstention_rate", 0.0),
        "invalid_rate": fleet_summary.get("invalid_rate", 0.0),
        "total_queries": fleet_summary.get("total_queries", 0),
    }


def build_control_arm_calibration_summary(
    fleet_summary: Dict[str, Any],
    profile: str = "uniform",
) -> Dict[str, Any]:
    """
    Build a control arm calibration summary for P5 CAL-EXP runs.
    
    This function creates a per-experiment control summary that captures the mock
    oracle's baseline behavior for calibration comparison against twin (real verifier)
    behavior. One summary is emitted per CAL-EXP-* experiment.
    
    The control arm serves to prove we can distinguish noise (expected stochasticity)
    from signal (actual verification behavior). Control ≠ twin is expected and good;
    equality would indicate a red flag (overfitting or lack of sensitivity).
    
    Args:
        fleet_summary: Fleet summary dictionary from build_mock_oracle_fleet_summary()
        profile: Profile name used to generate the fleet summary (default: "uniform")
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - status: Fleet status ("OK" | "DRIFTING" | "BROKEN")
        - abstention_rate: Fraction of abstentions (0.0-1.0)
        - invalid_rate: Fraction of invalid results (0.0-1.0)
        - total_queries: Total number of queries processed
        - profile: Profile name used (e.g., "uniform", "timeout_heavy", "invalid_heavy")
    """
    return {
        "schema_version": CONTROL_ARM_CALIBRATION_SCHEMA_VERSION,
        "status": fleet_summary.get("status", FLEET_STATUS_OK),
        "abstention_rate": fleet_summary.get("abstention_rate", 0.0),
        "invalid_rate": fleet_summary.get("invalid_rate", 0.0),
        "total_queries": fleet_summary.get("total_queries", 0),
        "profile": profile,
    }


def build_control_vs_twin_panel(
    control_summaries: Dict[str, Dict[str, Any]],
    twin_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a control vs twin calibration panel for P5 CAL-EXP runs.
    
    This function compares control arm (mock oracle) summaries against twin (real verifier)
    summaries across multiple experiments. The panel proves we can distinguish noise from
    signal by showing that control ≠ twin (which is expected and good).
    
    Red flags are raised when:
    - Control and twin metrics are too similar (suggests overfitting or lack of sensitivity)
    - Control metrics are outside expected ranges (suggests mock oracle drift)
    - Twin metrics match control exactly (suggests pipeline not responding to real behavior)
    
    Args:
        control_summaries: Dictionary mapping experiment names to control arm summaries
                          (from build_control_arm_calibration_summary())
        twin_summaries: Dictionary mapping experiment names to twin summaries
                       (from real verifier behavior)
    
    Returns:
        Dictionary containing:
        - schema_version: "1.0.0"
        - experiments: List of experiment names analyzed
        - control_vs_twin_delta: Dictionary mapping experiment names to delta metrics
        - red_flags: List of red flag descriptions (empty if all checks pass)
    """
    experiments = sorted(set(control_summaries.keys()) & set(twin_summaries.keys()))
    
    control_vs_twin_delta = {}
    red_flags = []
    
    for exp_name in experiments:
        control = control_summaries[exp_name]
        twin = twin_summaries[exp_name]
        
        # Compute deltas
        abstention_delta = abs(control.get("abstention_rate", 0.0) - twin.get("abstention_rate", 0.0))
        invalid_delta = abs(control.get("invalid_rate", 0.0) - twin.get("invalid_rate", 0.0))
        status_match = control.get("status") == twin.get("status")
        
        control_vs_twin_delta[exp_name] = {
            "abstention_rate_delta": abstention_delta,
            "invalid_rate_delta": invalid_delta,
            "status_match": status_match,
        }
        
        # Red flag: Control and twin are too similar (within 1% on both metrics)
        if abstention_delta < 0.01 and invalid_delta < 0.01:
            red_flags.append(
                f"Experiment '{exp_name}': Control and twin metrics are too similar "
                f"(abstention_delta={abstention_delta:.4f}, invalid_delta={invalid_delta:.4f}). "
                "This may indicate overfitting or lack of sensitivity."
            )
        
        # Red flag: Status matches exactly (suggests pipeline not distinguishing control from twin)
        if status_match and abstention_delta < 0.05 and invalid_delta < 0.05:
            red_flags.append(
                f"Experiment '{exp_name}': Control and twin have matching status with "
                f"similar rates. This suggests the pipeline may not be distinguishing "
                "between expected stochasticity (control) and actual behavior (twin)."
            )
    
    return {
        "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
        "experiments": experiments,
        "control_vs_twin_delta": control_vs_twin_delta,
        "red_flags": red_flags,
    }


def summarize_control_arm_signal_consistency(
    status_signal: Dict[str, Any],
    ggfl_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-check consistency between status signal and GGFL signal for control arm.
    
    This function validates that the status signal and GGFL signal are consistent
    with each other and that the conflict invariant is maintained (conflict must
    always be False).
    
    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Detects inconsistencies for advisory purposes only
    - No gating, just advisory notes
    
    Args:
        status_signal: Control arm signal from status JSON (from generate_first_light_status)
        ggfl_signal: Control arm signal from GGFL adapter (from control_arm_for_alignment_view)
    
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - mode: "SHADOW"
        - consistency: "CONSISTENT" | "PARTIAL" | "INCONSISTENT"
        - notes: List of neutral descriptive notes about inconsistencies
        - conflict_invariant_violated: bool (True if conflict is ever True)
        - top_mismatch_type: Optional[str] (Top mismatch type for INCONSISTENT cases)
    """
    notes: List[str] = []
    consistency = "CONSISTENT"
    conflict_invariant_violated = False
    top_mismatch_type: Optional[str] = None
    
    # Extract status values
    status_status = status_signal.get("status", "").upper()  # "OK" | "WARN"
    ggfl_status = ggfl_signal.get("status", "").lower()  # "ok" | "warn"
    
    # Normalize for comparison
    status_normalized = "warn" if status_status == "WARN" else "ok"
    
    # Check status consistency
    status_mismatch = False
    if status_normalized != ggfl_status:
        notes.append(
            f"Status mismatch: status signal says '{status_status}' but GGFL says '{ggfl_status}'"
        )
        status_mismatch = True
        consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "status_mismatch"
    
    # Extract recommendation values
    status_recommendation = status_signal.get("recommendation")  # May not be present in status
    ggfl_recommendation = ggfl_signal.get("recommendation", "")  # "NONE" | "WARNING"
    
    # Check recommendation consistency (if present in status signal)
    # Missing recommendation should not cause INCONSISTENT, only PARTIAL
    recommendation_mismatch = False
    if status_recommendation is not None:
        if status_recommendation != ggfl_recommendation:
            notes.append(
                f"Recommendation mismatch: status signal says '{status_recommendation}' "
                f"but GGFL says '{ggfl_recommendation}'"
            )
            recommendation_mismatch = True
            if consistency == "CONSISTENT":
                consistency = "PARTIAL"
            if top_mismatch_type is None:
                top_mismatch_type = "recommendation_mismatch"
    
    # Check conflict invariant (MUST always be False)
    # This is the only condition that causes INCONSISTENT
    ggfl_conflict = ggfl_signal.get("conflict", False)
    if ggfl_conflict is True:
        notes.append(
            "CRITICAL: Conflict invariant violated - GGFL signal has conflict=True. "
            "Control arm must never trigger conflict (conflict must always be False)."
        )
        conflict_invariant_violated = True
        consistency = "INCONSISTENT"
        top_mismatch_type = "conflict_invariant_violated"
    
    # If no issues found, return consistent
    if not notes:
        notes.append("Status signal and GGFL signal are consistent")
    
    return {
        "schema_version": CONTROL_ARM_CONSISTENCY_SCHEMA_VERSION,
        "mode": "SHADOW",
        "consistency": consistency,
        "notes": notes,
        "conflict_invariant_violated": conflict_invariant_violated,
        "top_mismatch_type": top_mismatch_type,
    }


def control_arm_for_alignment_view(panel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert control arm calibration panel to GGFL alignment view format.
    
    This function normalizes the control arm panel into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.
    
    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Control arm never triggers conflict directly (conflict: false always, invariant)
    - Control arm has LOW weight to prevent overpowering fusion semantics
    
    Args:
        panel: Control vs twin panel from build_control_vs_twin_panel()
    
    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-CTRL" (identifies this as a control input)
        - status: "ok" | "warn" (warn if any red_flags)
        - summary: Top red flag reason or neutral summary
        - conflict: false (control arm never triggers conflict directly, invariant)
        - weight_hint: "LOW" (ensures control arm doesn't overpower fusion)
        - recommendation: "NONE" | "WARNING" (warning if red_flags > 0)
    """
    red_flags = panel.get("red_flags", [])
    num_experiments = len(panel.get("experiments", []))
    red_flag_count = len(red_flags)
    
    # Determine status: warn if any red flags, otherwise ok
    status = "warn" if red_flags else "ok"
    
    # Determine recommendation: WARNING if red flags present, otherwise NONE
    recommendation = "WARNING" if red_flag_count > 0 else "NONE"
    
    # Build summary from top red flag or neutral message
    if red_flags:
        summary = red_flags[0]  # Top red flag reason
    else:
        summary = f"Control arm calibration: {num_experiments} experiments, no red flags"
    
    return {
        "signal_type": "SIG-CTRL",  # Identifies this as a control input
        "status": status,
        "summary": summary,
        "conflict": False,  # Control arm never triggers conflict directly (invariant)
        "weight_hint": "LOW",  # Ensures control arm doesn't overpower fusion semantics
        "recommendation": recommendation,
    }


def attach_mock_oracle_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    fleet_summary: Optional[Dict[str, Any]] = None,
    control_vs_twin_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach mock oracle drift tile to evidence pack.
    
    This function stores the mock oracle drift tile under the governance
    section of an evidence pack, following the standard evidence pack structure.
    Optionally includes a First Light summary if fleet_summary is provided.
    Optionally includes a control vs twin panel if control_vs_twin_panel is provided.
    
    Args:
        evidence: Evidence pack dictionary (will be modified in-place)
        tile: Mock oracle drift tile from build_mock_oracle_drift_tile()
        fleet_summary: Optional fleet summary for First Light summary attachment
        control_vs_twin_panel: Optional control vs twin panel from build_control_vs_twin_panel()
    
    Returns:
        Modified evidence dictionary with mock oracle tile attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach mock oracle tile
    evidence["governance"]["mock_oracle"] = tile
    
    # Optionally attach First Light summary
    if fleet_summary is not None:
        first_light_summary = build_first_light_mock_oracle_summary(fleet_summary)
        evidence["governance"]["mock_oracle"]["first_light_summary"] = first_light_summary
    
    # Optionally attach control vs twin panel
    if control_vs_twin_panel is not None:
        evidence["governance"]["mock_oracle_panel"] = control_vs_twin_panel
    
    return evidence

