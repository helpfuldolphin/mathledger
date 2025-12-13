"""
Phase X Budget Binding: P3/P4 Integration Layer

This module provides the binding between budget governance signals and Phase X
shadow experiments. It computes noise multipliers for P3 Δp computation and
severity multipliers for P4 divergence analysis.

SHADOW MODE CONTRACT:
- All computations are read-only
- No side effects on governance
- Results are for observation/logging only
- This does NOT alter behavior; only computes multipliers & exposes

See: docs/system_law/Budget_PhaseX_Doctrine.md

Status: SHADOW IMPLEMENTATION (computation only, no enforcement)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "BudgetDriftClass",
    "BudgetStabilityClass",
    "BudgetRiskSignal",
    "compute_noise_multiplier",
    "compute_severity_multiplier",
    "compute_rsi_correction_factor",
    "drift_class_from_value",
    "stability_class_from_health",
    "build_budget_risk_signal",
    "NOISE_MULTIPLIER_TABLE",
    "SEVERITY_MULTIPLIER_TABLE",
    # Task 1: P3 Stability Summary
    "build_budget_risk_summary_for_p3",
    # Task 2: P4 Calibration
    "build_budget_context_for_p4",
    # Task 3: Evidence Hook
    "attach_budget_risk_to_evidence",
    # Task 4: Calibration Evidence Pack Hook
    "attach_calibration_summary_to_evidence",
    "load_calibration_summary_from_file",
    "maybe_attach_calibration_to_evidence",
    # Task 5: GGFL Adapter
    "budget_calibration_for_alignment_view",
    # Task 5b: Reason codes and provenance enums
    "BUDGET_CAL_REASON_CODES",
    "BUDGET_CAL_REASON_PRIORITY",
    "EXTRACTION_SOURCE_ENUM",
]


# =============================================================================
# Constants: Drift Class → Noise Multiplier
# =============================================================================

class BudgetDriftClass(str, Enum):
    """Budget drift classification based on drift_value magnitude."""
    STABLE = "STABLE"          # |drift| <= 0.05
    DRIFTING = "DRIFTING"      # 0.05 < |drift| <= 0.15
    DIVERGING = "DIVERGING"    # 0.15 < |drift| <= 0.25
    CRITICAL = "CRITICAL"      # |drift| > 0.25


class BudgetStabilityClass(str, Enum):
    """Budget stability classification from health metrics."""
    STABLE = "STABLE"          # health >= 80, stability >= 0.95
    DRIFTING = "DRIFTING"      # health in [70, 80) or stability in [0.7, 0.95)
    VOLATILE = "VOLATILE"      # health < 70 or stability < 0.7


# Deterministic mapping: drift_class → noise_multiplier
# Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 2.3
NOISE_MULTIPLIER_TABLE: Dict[BudgetDriftClass, float] = {
    BudgetDriftClass.STABLE: 1.0,
    BudgetDriftClass.DRIFTING: 1.3,
    BudgetDriftClass.DIVERGING: 1.6,
    BudgetDriftClass.CRITICAL: 2.0,
}

# Deterministic mapping: stability_class → severity_multiplier
# Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 3.3
SEVERITY_MULTIPLIER_TABLE: Dict[BudgetStabilityClass, float] = {
    BudgetStabilityClass.STABLE: 1.0,
    BudgetStabilityClass.DRIFTING: 0.7,
    BudgetStabilityClass.VOLATILE: 0.4,
}


# =============================================================================
# Classification Functions
# =============================================================================

def drift_class_from_value(drift_value: float) -> BudgetDriftClass:
    """
    Classify drift_value into BudgetDriftClass.

    Deterministic mapping:
    - |drift| <= 0.05 → STABLE
    - 0.05 < |drift| <= 0.15 → DRIFTING
    - 0.15 < |drift| <= 0.25 → DIVERGING
    - |drift| > 0.25 → CRITICAL

    Args:
        drift_value: Budget drift value (can be negative)

    Returns:
        BudgetDriftClass enum value
    """
    abs_drift = abs(drift_value)

    if abs_drift <= 0.05:
        return BudgetDriftClass.STABLE
    elif abs_drift <= 0.15:
        return BudgetDriftClass.DRIFTING
    elif abs_drift <= 0.25:
        return BudgetDriftClass.DIVERGING
    else:
        return BudgetDriftClass.CRITICAL


def stability_class_from_health(
    health_score: float,
    stability_index: float,
) -> BudgetStabilityClass:
    """
    Classify budget stability from health metrics.

    Deterministic mapping:
    - health >= 80 AND stability >= 0.95 → STABLE
    - health in [70, 80) OR stability in [0.7, 0.95) → DRIFTING
    - health < 70 OR stability < 0.7 → VOLATILE

    Args:
        health_score: Budget health score (0-100)
        stability_index: Stability index (0.0-1.0)

    Returns:
        BudgetStabilityClass enum value
    """
    if health_score < 70.0 or stability_index < 0.7:
        return BudgetStabilityClass.VOLATILE
    elif health_score < 80.0 or stability_index < 0.95:
        return BudgetStabilityClass.DRIFTING
    else:
        return BudgetStabilityClass.STABLE


# =============================================================================
# Multiplier Computation Functions
# =============================================================================

def compute_noise_multiplier(drift_value: float) -> float:
    """
    Compute noise floor multiplier for P3 Δp computation.

    SHADOW MODE: This computes the multiplier but does NOT alter behavior.
    The multiplier is exposed for logging and analysis.

    Two methods are provided:
    1. Table lookup (deterministic, discrete)
    2. Formula-based (continuous): 1.0 / sqrt(1 - |drift_value|), clamped to [1.0, 3.0]

    This function uses the table lookup method for deterministic behavior.

    Args:
        drift_value: Budget drift value

    Returns:
        Noise floor multiplier >= 1.0
    """
    drift_class = drift_class_from_value(drift_value)
    return NOISE_MULTIPLIER_TABLE[drift_class]


def compute_noise_multiplier_continuous(drift_value: float) -> float:
    """
    Compute noise floor multiplier using continuous formula.

    Formula: 1.0 / sqrt(1 - |drift_value|), clamped to [1.0, 3.0]

    This provides a smooth multiplier that increases as drift increases.
    The formula becomes undefined at |drift| = 1.0, so we clamp the result.

    Args:
        drift_value: Budget drift value

    Returns:
        Noise floor multiplier in [1.0, 3.0]
    """
    abs_drift = abs(drift_value)

    # Avoid division by zero or sqrt of negative
    if abs_drift >= 1.0:
        return 3.0

    # Compute multiplier
    try:
        multiplier = 1.0 / math.sqrt(1.0 - abs_drift)
    except (ValueError, ZeroDivisionError):
        multiplier = 3.0

    # Clamp to [1.0, 3.0]
    return max(1.0, min(3.0, multiplier))


def compute_severity_multiplier(
    stability_class: BudgetStabilityClass,
) -> float:
    """
    Compute divergence severity multiplier for P4 analysis.

    SHADOW MODE: This computes the multiplier but does NOT alter behavior.
    The multiplier is exposed for logging and analysis.

    When budget is unstable, divergence severity should be reduced because
    the divergence may be caused by budget effects rather than model inadequacy.

    Multiplier interpretation:
    - 1.0: No adjustment (budget is stable)
    - 0.7: Moderate reduction (budget is drifting)
    - 0.4: Strong reduction (budget is volatile)

    The multiplier is applied as: adjusted_severity = raw_severity * multiplier

    Args:
        stability_class: Budget stability classification

    Returns:
        Severity multiplier in (0, 1]
    """
    return SEVERITY_MULTIPLIER_TABLE[stability_class]


def compute_severity_multiplier_from_health(
    health_score: float,
    stability_index: float,
) -> float:
    """
    Compute severity multiplier directly from health metrics.

    Convenience function that classifies and then looks up multiplier.

    Args:
        health_score: Budget health score (0-100)
        stability_index: Stability index (0.0-1.0)

    Returns:
        Severity multiplier in (0, 1]
    """
    stability_class = stability_class_from_health(health_score, stability_index)
    return compute_severity_multiplier(stability_class)


def compute_rsi_correction_factor(drift_class: BudgetDriftClass) -> float:
    """
    Compute RSI correction factor for budget-induced stability bias.

    When budget drifts, RSI readings may be artificially inflated or depressed.
    This factor represents the uncertainty in RSI due to budget effects.

    Factor interpretation:
    - 1.0: No correction needed (budget stable)
    - < 1.0: RSI may be inflated (apply reduction factor)

    Args:
        drift_class: Budget drift classification

    Returns:
        RSI correction factor in (0, 1]
    """
    correction_table = {
        BudgetDriftClass.STABLE: 1.0,
        BudgetDriftClass.DRIFTING: 0.95,
        BudgetDriftClass.DIVERGING: 0.85,
        BudgetDriftClass.CRITICAL: 0.75,
    }
    return correction_table[drift_class]


# =============================================================================
# Budget Risk Signal
# =============================================================================

@dataclass
class BudgetRiskSignal:
    """
    Budget risk signal for P3/P4 integration.

    This structure aggregates all budget-related risk indicators
    for consumption by P3 Δp computation and P4 divergence analysis.

    SHADOW MODE: This signal is observational only.
    """

    # Classification
    drift_class: BudgetDriftClass = BudgetDriftClass.STABLE
    stability_class: BudgetStabilityClass = BudgetStabilityClass.STABLE

    # Raw metrics
    drift_value: float = 0.0
    health_score: float = 100.0
    stability_index: float = 1.0

    # Computed multipliers
    noise_multiplier: float = 1.0
    severity_multiplier: float = 1.0
    rsi_correction_factor: float = 1.0

    # Flags
    budget_confounded: bool = False
    admissibility_hint: str = "OK"  # OK, WARN, BLOCK

    # Invariant failures
    inv_bud_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "drift_class": self.drift_class.value,
            "stability_class": self.stability_class.value,
            "drift_value": round(self.drift_value, 6),
            "health_score": round(self.health_score, 2),
            "stability_index": round(self.stability_index, 4),
            "noise_multiplier": round(self.noise_multiplier, 4),
            "severity_multiplier": round(self.severity_multiplier, 4),
            "rsi_correction_factor": round(self.rsi_correction_factor, 4),
            "budget_confounded": self.budget_confounded,
            "admissibility_hint": self.admissibility_hint,
            "inv_bud_failures": self.inv_bud_failures,
        }


def build_budget_risk_signal(
    drift_value: float = 0.0,
    health_score: float = 100.0,
    stability_index: float = 1.0,
    inv_bud_failures: Optional[List[str]] = None,
) -> BudgetRiskSignal:
    """
    Build a BudgetRiskSignal from raw metrics.

    This is the main entry point for creating budget risk signals
    for P3/P4 integration.

    Args:
        drift_value: Current budget drift value
        health_score: Budget health score (0-100)
        stability_index: Stability index (0.0-1.0)
        inv_bud_failures: List of failing invariants (e.g., ["INV-BUD-1"])

    Returns:
        BudgetRiskSignal with all computed fields
    """
    inv_bud_failures = inv_bud_failures or []

    # Compute classifications
    drift_class = drift_class_from_value(drift_value)
    stability_class = stability_class_from_health(health_score, stability_index)

    # Compute multipliers
    noise_multiplier = compute_noise_multiplier(drift_value)
    severity_multiplier = compute_severity_multiplier(stability_class)
    rsi_correction_factor = compute_rsi_correction_factor(drift_class)

    # Determine if confounded (any non-stable classification)
    budget_confounded = (
        drift_class != BudgetDriftClass.STABLE or
        stability_class != BudgetStabilityClass.STABLE or
        len(inv_bud_failures) > 0
    )

    # Determine admissibility hint
    if stability_class == BudgetStabilityClass.VOLATILE or len(inv_bud_failures) > 0:
        admissibility_hint = "BLOCK"
    elif stability_class == BudgetStabilityClass.DRIFTING:
        admissibility_hint = "WARN"
    else:
        admissibility_hint = "OK"

    return BudgetRiskSignal(
        drift_class=drift_class,
        stability_class=stability_class,
        drift_value=drift_value,
        health_score=health_score,
        stability_index=stability_index,
        noise_multiplier=noise_multiplier,
        severity_multiplier=severity_multiplier,
        rsi_correction_factor=rsi_correction_factor,
        budget_confounded=budget_confounded,
        admissibility_hint=admissibility_hint,
        inv_bud_failures=list(inv_bud_failures),
    )


# =============================================================================
# P3 Integration: Stability Report Extension
# =============================================================================

def extend_stability_report_with_budget(
    stability_report: Dict[str, Any],
    budget_signal: BudgetRiskSignal,
) -> Dict[str, Any]:
    """
    Extend a stability report with budget risk information.

    SHADOW MODE: This adds observational budget data to the report.
    It does NOT alter the report's conclusions.

    Args:
        stability_report: Existing stability report dict
        budget_signal: Budget risk signal

    Returns:
        Extended stability report with budget_risk section
    """
    # Create a copy to avoid mutation
    extended = dict(stability_report)

    # Add budget risk section
    extended["budget_risk"] = {
        "drift_class": budget_signal.drift_class.value,
        "noise_multiplier": round(budget_signal.noise_multiplier, 4),
        "admissibility_hint": budget_signal.admissibility_hint,
        "stability_class": budget_signal.stability_class.value,
        "severity_multiplier": round(budget_signal.severity_multiplier, 4),
        "budget_confounded": budget_signal.budget_confounded,
        "health_score": round(budget_signal.health_score, 2),
        "stability_index": round(budget_signal.stability_index, 4),
        "inv_bud_failures": budget_signal.inv_bud_failures,
    }

    return extended


# =============================================================================
# P4 Integration: Adjusted Severity
# =============================================================================

def adjust_divergence_severity(
    raw_severity: str,
    budget_signal: BudgetRiskSignal,
) -> Dict[str, Any]:
    """
    Adjust divergence severity based on budget stability.

    SHADOW MODE: This computes the adjusted severity but does NOT
    alter enforcement behavior. The result is for logging/analysis.

    Adjustment rules:
    - When budget is VOLATILE: Consider downgrading CRITICAL → WARN
    - When budget is DRIFTING: Consider downgrading CRITICAL → INFO

    The adjustment is exposed for analysis, not enforcement.

    Args:
        raw_severity: Original severity string (NONE, INFO, WARN, CRITICAL)
        budget_signal: Budget risk signal

    Returns:
        Dict with raw and adjusted severity information
    """
    severity_levels = ["NONE", "INFO", "WARN", "CRITICAL"]
    multiplier = budget_signal.severity_multiplier

    # Get numeric level
    try:
        raw_level = severity_levels.index(raw_severity)
    except ValueError:
        raw_level = 0

    # Compute adjusted level (using multiplier as a reduction factor)
    # Interpretation: multiply level by (1 - adjustment factor)
    # E.g., CRITICAL (3) with multiplier 0.4 → 3 * 0.4 = 1.2 → INFO (1)
    adjusted_level = int(raw_level * multiplier)
    adjusted_level = max(0, min(3, adjusted_level))

    adjusted_severity = severity_levels[adjusted_level]

    return {
        "raw_severity": raw_severity,
        "adjusted_severity": adjusted_severity,
        "severity_multiplier": multiplier,
        "budget_confounded": budget_signal.budget_confounded,
        "stability_class": budget_signal.stability_class.value,
        "adjustment_applied": raw_severity != adjusted_severity,
    }


def compute_tda_context(budget_signal: BudgetRiskSignal) -> str:
    """
    Compute TDA interpretation context from budget signal.

    Returns:
        TDA context: "NOMINAL", "BUDGET_DRIFT", or "BUDGET_UNSTABLE"
    """
    if budget_signal.stability_class == BudgetStabilityClass.VOLATILE:
        return "BUDGET_UNSTABLE"
    elif budget_signal.stability_class == BudgetStabilityClass.DRIFTING:
        return "BUDGET_DRIFT"
    else:
        return "NOMINAL"


# =============================================================================
# Task 1: P3 Stability Summary
# =============================================================================

def build_budget_risk_summary_for_p3(
    drift_value: float = 0.0,
    health_score: float = 100.0,
    stability_index: float = 1.0,
    inv_bud_failures: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build budget risk summary for P3 stability reports.

    This function produces a structured summary suitable for attachment
    under stability_report["budget_risk"]. All fields are deterministic
    and JSON-serializable.

    SHADOW MODE CONTRACT:
    - All values are observational only
    - No governance decisions depend on this summary
    - Used for logging and analysis purposes

    Args:
        drift_value: Current budget drift value
        health_score: Budget health score (0-100)
        stability_index: Stability index (0.0-1.0)
        inv_bud_failures: List of failing invariants

    Returns:
        Dict with structure:
        {
            "drift_class": str,
            "stability_class": str,
            "noise_multiplier": float,
            "severity_multiplier": float,
            "admissibility_hint": str,
            "rsi_correction_factor": float,
            "budget_confounded": bool,
            "health_score": float,
            "stability_index": float,
            "inv_bud_failures": List[str],
        }

    Example:
        >>> summary = build_budget_risk_summary_for_p3(drift_value=0.10, health_score=75.0)
        >>> stability_report["budget_risk"] = summary
    """
    signal = build_budget_risk_signal(
        drift_value=drift_value,
        health_score=health_score,
        stability_index=stability_index,
        inv_bud_failures=inv_bud_failures,
    )

    return {
        "drift_class": signal.drift_class.value,
        "stability_class": signal.stability_class.value,
        "noise_multiplier": round(signal.noise_multiplier, 4),
        "severity_multiplier": round(signal.severity_multiplier, 4),
        "admissibility_hint": signal.admissibility_hint,
        "rsi_correction_factor": round(signal.rsi_correction_factor, 4),
        "budget_confounded": signal.budget_confounded,
        "health_score": round(signal.health_score, 2),
        "stability_index": round(signal.stability_index, 4),
        "inv_bud_failures": signal.inv_bud_failures,
    }


# =============================================================================
# Task 2: P4 Calibration Context
# =============================================================================

def build_budget_context_for_p4(
    divergence_summary: Dict[str, Any],
    budget_signal: BudgetRiskSignal,
) -> Dict[str, Any]:
    """
    Build budget context for P4 divergence calibration.

    This function produces a calibration context that includes:
    - budget_confounded: Whether divergence may be budget-induced
    - effective_severity_shift: Numeric shift in severity levels
    - tda_context: NOMINAL | BUDGET_DRIFT | BUDGET_UNSTABLE

    SHADOW MODE CONTRACT:
    - All values are observational only
    - Severity shift is computed but NOT enforced
    - Used for logging and analysis purposes

    Args:
        divergence_summary: DivergenceSummary.to_dict() or compatible dict
        budget_signal: Budget risk signal

    Returns:
        Dict with structure:
        {
            "budget_confounded": bool,
            "effective_severity_shift": int,  # 0, -1, or -2 levels
            "tda_context": str,
            "stability_class": str,
            "severity_multiplier": float,
            "raw_divergence_rate": float,
            "adjusted_interpretation": str,
            "calibration_note": str,
        }

    Example:
        >>> summary = analyzer.get_summary().to_dict()
        >>> signal = build_budget_risk_signal(health_score=60.0)
        >>> context = build_budget_context_for_p4(summary, signal)
    """
    # Compute TDA context
    tda_context = compute_tda_context(budget_signal)

    # Compute effective severity shift
    # STABLE: 0, DRIFTING: -1, VOLATILE: -2
    severity_shift_map = {
        BudgetStabilityClass.STABLE: 0,
        BudgetStabilityClass.DRIFTING: -1,
        BudgetStabilityClass.VOLATILE: -2,
    }
    effective_severity_shift = severity_shift_map[budget_signal.stability_class]

    # Extract raw divergence rate from summary
    accuracy = divergence_summary.get("accuracy", {})
    raw_divergence_rate = accuracy.get("divergence_rate", 0.0)

    # Generate adjusted interpretation
    if budget_signal.budget_confounded:
        if budget_signal.stability_class == BudgetStabilityClass.VOLATILE:
            adjusted_interpretation = (
                "Divergence may be budget-induced. "
                "Consider VOLATILE budget state before attributing to model error."
            )
        else:
            adjusted_interpretation = (
                "Budget drift detected. "
                "Divergence severity may be inflated due to resource constraints."
            )
    else:
        adjusted_interpretation = "Budget nominal. Divergence reflects model accuracy."

    # Generate calibration note
    if effective_severity_shift == 0:
        calibration_note = "No severity calibration needed."
    elif effective_severity_shift == -1:
        calibration_note = (
            "DRIFTING budget: Consider downgrading CRITICAL→WARN, WARN→INFO."
        )
    else:
        calibration_note = (
            "VOLATILE budget: Consider downgrading all severities by 2 levels."
        )

    return {
        "budget_confounded": budget_signal.budget_confounded,
        "effective_severity_shift": effective_severity_shift,
        "tda_context": tda_context,
        "stability_class": budget_signal.stability_class.value,
        "severity_multiplier": round(budget_signal.severity_multiplier, 4),
        "raw_divergence_rate": round(raw_divergence_rate, 4),
        "adjusted_interpretation": adjusted_interpretation,
        "calibration_note": calibration_note,
    }


# =============================================================================
# Task 3: Evidence Hook
# =============================================================================

def attach_budget_risk_to_evidence(
    evidence: Dict[str, Any],
    budget_signal: BudgetRiskSignal,
) -> Dict[str, Any]:
    """
    Attach budget risk information to evidence structure.

    This function creates a NEW evidence dict with budget risk attached
    under evidence["governance"]["budget_risk"]. The original evidence
    dict is NOT mutated.

    SHADOW MODE CONTRACT:
    - Non-mutating: returns new dict, original unchanged
    - All values are observational only
    - Evidence structure remains valid after attachment

    Args:
        evidence: Evidence dict (from attestation or proof record)
        budget_signal: Budget risk signal to attach

    Returns:
        New evidence dict with budget_risk attached:
        {
            ...original evidence fields...,
            "governance": {
                ...existing governance fields...,
                "budget_risk": {
                    "drift_class": str,
                    "stability_class": str,
                    "noise_multiplier": float,
                    "severity_multiplier": float,
                    "admissibility_hint": str,
                    "budget_confounded": bool,
                    "tda_context": str,
                }
            }
        }

    Example:
        >>> evidence = {"proof_hash": "abc123", "governance": {"aligned": True}}
        >>> signal = build_budget_risk_signal(drift_value=0.15)
        >>> new_evidence = attach_budget_risk_to_evidence(evidence, signal)
        >>> # Original evidence unchanged
        >>> assert "budget_risk" not in evidence.get("governance", {})
        >>> # New evidence has budget_risk
        >>> assert "budget_risk" in new_evidence["governance"]
    """
    import copy

    # Deep copy to ensure non-mutation
    new_evidence = copy.deepcopy(evidence)

    # Ensure governance section exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}

    # Compute TDA context
    tda_context = compute_tda_context(budget_signal)

    # Build budget_risk section
    budget_risk = {
        "drift_class": budget_signal.drift_class.value,
        "stability_class": budget_signal.stability_class.value,
        "noise_multiplier": round(budget_signal.noise_multiplier, 4),
        "severity_multiplier": round(budget_signal.severity_multiplier, 4),
        "admissibility_hint": budget_signal.admissibility_hint,
        "budget_confounded": budget_signal.budget_confounded,
        "tda_context": tda_context,
        "health_score": round(budget_signal.health_score, 2),
        "stability_index": round(budget_signal.stability_index, 4),
    }

    # Attach under governance
    new_evidence["governance"]["budget_risk"] = budget_risk

    return new_evidence


# =============================================================================
# Task 4: Calibration Evidence Pack Hook
# =============================================================================

def attach_calibration_summary_to_evidence(
    evidence: Dict[str, Any],
    calibration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach calibration summary to evidence structure.

    This function creates a NEW evidence dict with calibration summary attached
    under evidence["governance"]["budget_risk"]["calibration"]. The original
    evidence dict is NOT mutated.

    SHADOW MODE CONTRACT:
    - Non-mutating: returns new dict, original unchanged
    - All values are observational only
    - Evidence structure remains valid after attachment
    - Calibration data is advisory, not enforcing

    Args:
        evidence: Evidence dict (from attestation or proof record)
        calibration_summary: Compact calibration summary dict with structure:
            {
                "schema_version": str,
                "experiment_id": str,
                "overall_pass": bool,
                "enablement_recommendation": str,
                "phases": {
                    "phase_1": {"cycles": int, "fp_rate": float, "fn_rate": float, "meets_criteria": bool},
                    "phase_2": {...},
                    "phase_3": {...},
                }
            }

    Returns:
        New evidence dict with calibration attached:
        {
            ...original evidence fields...,
            "governance": {
                ...existing governance fields...,
                "budget_risk": {
                    ...existing budget_risk fields (if any)...,
                    "calibration": {
                        "schema_version": str,
                        "experiment_id": str,
                        "overall_pass": bool,
                        "enablement_recommendation": str,
                        "fp_rate_p1": float,
                        "fn_rate_p1": float,
                        "fp_rate_p2": float,
                        "fn_rate_p2": float,
                        "fp_rate_p3": float,
                        "fn_rate_p3": float,
                    }
                }
            }
        }

    Example:
        >>> evidence = {"proof_hash": "abc123", "governance": {"aligned": True}}
        >>> calibration = {
        ...     "schema_version": "1.0.0",
        ...     "experiment_id": "abc123",
        ...     "overall_pass": True,
        ...     "enablement_recommendation": "PROCEED_TO_STAGE_2",
        ...     "phases": {
        ...         "phase_1": {"cycles": 500, "fp_rate": 0.01, "fn_rate": 0.005, "meets_criteria": True},
        ...     }
        ... }
        >>> new_evidence = attach_calibration_summary_to_evidence(evidence, calibration)
        >>> assert "calibration" in new_evidence["governance"]["budget_risk"]
    """
    import copy

    # Deep copy to ensure non-mutation
    new_evidence = copy.deepcopy(evidence)

    # Ensure governance section exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}

    # Ensure budget_risk section exists
    if "budget_risk" not in new_evidence["governance"]:
        new_evidence["governance"]["budget_risk"] = {}

    # Build compact calibration section
    phases = calibration_summary.get("phases", {})

    calibration_compact = {
        "schema_version": calibration_summary.get("schema_version", "1.0.0"),
        "experiment_id": calibration_summary.get("experiment_id", "unknown"),
        "overall_pass": calibration_summary.get("overall_pass", False),
        "enablement_recommendation": calibration_summary.get("enablement_recommendation", "NOT_RECOMMENDED"),
    }

    # Add phase metrics if available
    if "phase_1" in phases:
        calibration_compact["fp_rate_p1"] = round(phases["phase_1"].get("fp_rate", 0.0), 4)
        calibration_compact["fn_rate_p1"] = round(phases["phase_1"].get("fn_rate", 0.0), 4)

    if "phase_2" in phases:
        calibration_compact["fp_rate_p2"] = round(phases["phase_2"].get("fp_rate", 0.0), 4)
        calibration_compact["fn_rate_p2"] = round(phases["phase_2"].get("fn_rate", 0.0), 4)

    if "phase_3" in phases:
        calibration_compact["fp_rate_p3"] = round(phases["phase_3"].get("fp_rate", 0.0), 4)
        calibration_compact["fn_rate_p3"] = round(phases["phase_3"].get("fn_rate", 0.0), 4)

    # Attach under budget_risk
    new_evidence["governance"]["budget_risk"]["calibration"] = calibration_compact

    return new_evidence


def load_calibration_summary_from_file(summary_path: str) -> Optional[Dict[str, Any]]:
    """
    Load calibration summary from JSON file if it exists.

    Args:
        summary_path: Path to budget_calibration_summary.json

    Returns:
        Compact summary dict if file exists and is valid, None otherwise
    """
    import json
    from pathlib import Path

    path = Path(summary_path)
    if not path.exists():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Return compact_summary if present, otherwise whole file
        return data.get("compact_summary", data)
    except (json.JSONDecodeError, IOError):
        return None


def maybe_attach_calibration_to_evidence(
    evidence: Dict[str, Any],
    calibration_dir: str = "results/budget_calibration",
) -> Dict[str, Any]:
    """
    Attach calibration summary to evidence if calibration artifacts exist.

    This is a convenience function that checks for calibration outputs
    and attaches them if available.

    SHADOW MODE CONTRACT:
    - Non-mutating if no calibration found
    - Calibration is advisory only

    Args:
        evidence: Evidence dict
        calibration_dir: Directory containing calibration outputs

    Returns:
        Evidence with calibration attached (if available), or original evidence
    """
    from pathlib import Path

    summary_path = Path(calibration_dir) / "budget_calibration_summary.json"
    calibration = load_calibration_summary_from_file(str(summary_path))

    if calibration is None:
        return evidence

    return attach_calibration_summary_to_evidence(evidence, calibration)


# =============================================================================
# Task 5: GGFL Adapter — SIG-BUD-CAL Alignment View
# =============================================================================

# Reason codes for GGFL drivers (deterministic, stable enum)
# These codes are used in alignment view drivers and top_reason_code
BUDGET_CAL_REASON_CODES = {
    "DRIVER_DEFER": "DRIVER_DEFER",
    "DRIVER_OVERALL_PASS_FALSE": "DRIVER_OVERALL_PASS_FALSE",
    "DRIVER_FP_FN_PRESENT": "DRIVER_FP_FN_PRESENT",
}

# Priority order for top_reason_code selection (deterministic)
# Lower index = higher priority
BUDGET_CAL_REASON_PRIORITY = [
    "DRIVER_DEFER",
    "DRIVER_OVERALL_PASS_FALSE",
    "DRIVER_FP_FN_PRESENT",
]

# Extraction source provenance enum values
EXTRACTION_SOURCE_ENUM = {
    "MANIFEST": "MANIFEST",
    "EVIDENCE_JSON": "EVIDENCE_JSON",
    "DIRECT_DISCOVERY": "DIRECT_DISCOVERY",
    "MISSING": "MISSING",
}


def budget_calibration_for_alignment_view(
    reference_or_signal: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Map budget calibration signal to GGFL alignment view.

    SHADOW MODE CONTRACT:
    - Purely observational, no side effects
    - Advisory classification only, no gating
    - This function NEVER influences control flow

    Mapping:
        calibration absent -> None (explicit optional, no false health surface)
        ENABLE + overall_pass=True -> "healthy" (status="ok")
        DEFER or overall_pass=False -> "degraded" (status="warn")

    Note: "block" status reserved for future INV-BUD-* integration per
    Budget_PhaseX_Doctrine.md Section 4.2. Current implementation only
    emits "ok" or "warn".

    Reason Codes (drivers):
        DRIVER_DEFER - enablement_recommendation == "DEFER"
        DRIVER_OVERALL_PASS_FALSE - overall_pass == False
        DRIVER_FP_FN_PRESENT - FP/FN rates are present (informational)

    Args:
        reference_or_signal: Budget calibration reference from manifest
            (governance.budget_risk.calibration_reference) or status signal
            (signals.budget_calibration), or None if absent.

    Returns:
        None if calibration artifact is absent (explicit optional).
        Otherwise Dict with:
            - alignment: "healthy" | "degraded"
            - conflict: bool (always False for calibration - no conflict possible)
            - status: "ok" | "warn"
            - advisory: str describing the classification
            - drivers: List[str] deterministic reason codes
            - top_reason_code: str | None (highest priority reason for warn cases)
            - mode: "SHADOW" (always)

    Example:
        >>> ref = manifest["governance"]["budget_risk"]["calibration_reference"]
        >>> view = budget_calibration_for_alignment_view(ref)
        >>> view["alignment"]
        'healthy'
    """
    # Handle missing artifact: return None (explicit optional, no false health)
    if not reference_or_signal:
        return None

    # Extract fields from either reference or signal format
    enablement = reference_or_signal.get("enablement_recommendation")
    overall_pass = reference_or_signal.get("overall_pass")
    fp_rate = reference_or_signal.get("fp_rate")
    fn_rate = reference_or_signal.get("fn_rate")

    # Build deterministic drivers list using reason codes
    # Order: DRIVER_DEFER, DRIVER_OVERALL_PASS_FALSE, DRIVER_FP_FN_PRESENT
    drivers: List[str] = []

    if enablement == "DEFER":
        drivers.append(BUDGET_CAL_REASON_CODES["DRIVER_DEFER"])

    if overall_pass is False:
        drivers.append(BUDGET_CAL_REASON_CODES["DRIVER_OVERALL_PASS_FALSE"])

    if fp_rate is not None or fn_rate is not None:
        drivers.append(BUDGET_CAL_REASON_CODES["DRIVER_FP_FN_PRESENT"])

    # Determine alignment and status based on thresholds
    # Per STRATCOM: warn if DEFER OR overall_pass==false
    should_warn = (
        enablement == "DEFER" or
        overall_pass is False
    )

    # Select top_reason_code deterministically (priority order)
    top_reason_code: Optional[str] = None
    if should_warn:
        for reason in BUDGET_CAL_REASON_PRIORITY:
            if reason in drivers:
                top_reason_code = reason
                break

    if should_warn:
        alignment = "degraded"
        status = "warn"
        if enablement == "DEFER":
            advisory = (
                f"Budget calibration recommends DEFER. "
                f"FP/FN thresholds not yet met for enablement."
            )
        else:
            advisory = (
                f"Budget calibration overall_pass=False. "
                f"One or more calibration phases did not pass success criteria."
            )
    else:
        alignment = "healthy"
        status = "ok"
        advisory = (
            f"Budget calibration criteria met. "
            f"Enablement recommendation: {enablement or 'ENABLE'}."
        )

    return {
        "alignment": alignment,
        "conflict": False,  # Calibration has no conflict concept
        "status": status,
        "advisory": advisory,
        "drivers": drivers,
        "top_reason_code": top_reason_code,
        "mode": "SHADOW",
    }
