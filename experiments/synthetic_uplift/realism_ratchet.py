#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Synthetic Realism Ratchet & Scenario Calibration Console
---------------------------------------------------------

This module implements self-calibrating mechanisms for synthetic scenarios:

    1. Realism Ratchet - Tracks realism pressure and scenario stability
    2. Calibration Console - Determines which scenarios need recalibration

The ratchet mechanism provides deterministic signals that increase pressure
when synthetic scenarios diverge from real-world patterns, and decrease
retention scores for unstable scenarios.

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data in outputs
    - Imply empirical validity

==============================================================================
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# CONSTANTS
# ==============================================================================

SCHEMA_VERSION = "realism_ratchet_v1.0"


# ==============================================================================
# ENUMS
# ==============================================================================

class StabilityClass(Enum):
    """Stability classification for scenarios."""
    STABLE = "STABLE"
    SOFT_DRIFT = "SOFT_DRIFT"
    SHARP_DRIFT = "SHARP_DRIFT"


class CalibrationStatus(Enum):
    """Calibration status for scenarios."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


# ==============================================================================
# TASK 1: SYNTHETIC REALISM RATCHET
# ==============================================================================

def build_synthetic_realism_ratchet(
    consistency_view: Dict[str, Any],
    realism_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a deterministic ratchet signal controlling realism pressure and retention.
    
    Args:
        consistency_view: Output from build_synthetic_real_consistency_view
        realism_summary: Output from summarize_synthetic_realism_for_global_health
    
    Returns:
        Ratchet signal with:
            - realism_pressure: float in [0,1] - increases when synthetic diverges
            - scenario_retention_score: Dict[str, float] - per-scenario scores
            - stability_class: Dict[str, str] - per-scenario stability class
            - global_realism_pressure: float - overall pressure level
    """
    # Extract key metrics
    consistency_status = consistency_view.get("consistency_status", "MISALIGNED")
    envelope_breach_rate = realism_summary.get("envelope_breach_rate", 0.0)
    scenarios_needing_review = set(realism_summary.get("scenarios_needing_review", []))
    
    # Calculate global realism pressure
    # Base pressure from breach rate
    base_pressure = min(1.0, envelope_breach_rate * 2.0)  # Scale: 50% breach = 1.0 pressure
    
    # Add pressure from consistency misalignment
    consistency_pressure = 0.0
    if consistency_status == "MISALIGNED":
        consistency_pressure = 0.5
    elif consistency_status == "PARTIAL":
        consistency_pressure = 0.25
    
    # Total pressure (weighted combination)
    global_realism_pressure = min(1.0, base_pressure * 0.6 + consistency_pressure * 0.4)
    
    # Calculate per-scenario retention scores and stability classes
    per_scenario_retention = {}
    per_scenario_stability = {}
    
    # Get scenario classifications from consistency view
    consistent = set(consistency_view.get("scenarios_consistent_with_real", []))
    aggressive = set(consistency_view.get("scenarios_more_aggressive_than_real", []))
    under_exploring = set(consistency_view.get("scenarios_under_exploring", []))
    
    # Calculate retention scores based on multiple factors
    for scenario_name in consistent | aggressive | under_exploring:
        retention_score = 1.0
        
        # Reduce score if scenario needs review
        if scenario_name in scenarios_needing_review:
            retention_score *= 0.7
        
        # Reduce score if under-exploring (too conservative)
        if scenario_name in under_exploring:
            retention_score *= 0.8
        
        # Reduce score if too aggressive (high breach rate)
        if scenario_name in aggressive:
            retention_score *= 0.6
        
        # Determine stability class
        stability = _determine_stability_class(
            scenario_name=scenario_name,
            needs_review=scenario_name in scenarios_needing_review,
            is_aggressive=scenario_name in aggressive,
            is_under_exploring=scenario_name in under_exploring,
        )
        
        per_scenario_retention[scenario_name] = retention_score
        per_scenario_stability[scenario_name] = stability.value
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realism_pressure": global_realism_pressure,
        "scenario_retention_score": per_scenario_retention,
        "stability_class": per_scenario_stability,
        "global_realism_pressure": global_realism_pressure,
        "pressure_factors": {
            "envelope_breach_rate": envelope_breach_rate,
            "consistency_status": consistency_status,
            "base_pressure": base_pressure,
            "consistency_pressure": consistency_pressure,
        },
    }


def _determine_stability_class(
    scenario_name: str,
    needs_review: bool,
    is_aggressive: bool,
    is_under_exploring: bool,
) -> StabilityClass:
    """
    Determine stability class for a scenario.
    
    Rules:
        - STABLE: No issues, consistent with real
        - SOFT_DRIFT: Minor issues, needs attention
        - SHARP_DRIFT: Major issues, unstable
    """
    # SHARP_DRIFT: Needs review and is aggressive (high instability)
    if needs_review and is_aggressive:
        return StabilityClass.SHARP_DRIFT
    
    # SOFT_DRIFT: Needs review OR is under-exploring
    if needs_review or is_under_exploring:
        return StabilityClass.SOFT_DRIFT
    
    # STABLE: Everything else
    return StabilityClass.STABLE


# ==============================================================================
# TASK 2: SCENARIO CALIBRATION CONSOLE
# ==============================================================================

def build_scenario_calibration_console(
    ratchet: Dict[str, Any],
    scenario_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build calibration console determining which scenarios need recalibration.
    
    Args:
        ratchet: Output from build_synthetic_realism_ratchet
        scenario_policy: Output from derive_synthetic_scenario_policy
    
    Returns:
        Calibration console with:
            - calibration_status: OK / ATTENTION / BLOCK
            - slices_to_recalibrate: List of scenario names
            - advisory_notes: Neutral description of calibration needs
    """
    realism_pressure = ratchet.get("realism_pressure", 0.0)
    retention_scores = ratchet.get("scenario_retention_score", {})
    stability_classes = ratchet.get("stability_class", {})
    
    policy_status = scenario_policy.get("status", "OK")
    scenarios_needing_tuning = set(scenario_policy.get("scenarios_needing_tuning", []))
    
    # Identify scenarios to recalibrate
    slices_to_recalibrate = []
    
    # Add scenarios with low retention scores
    for scenario_name, score in retention_scores.items():
        if score < 0.5:  # Low retention threshold
            slices_to_recalibrate.append(scenario_name)
    
    # Add scenarios with SHARP_DRIFT stability
    for scenario_name, stability in stability_classes.items():
        if stability == StabilityClass.SHARP_DRIFT.value:
            if scenario_name not in slices_to_recalibrate:
                slices_to_recalibrate.append(scenario_name)
    
    # Add scenarios from policy needing tuning
    for scenario_name in scenarios_needing_tuning:
        if scenario_name not in slices_to_recalibrate:
            slices_to_recalibrate.append(scenario_name)
    
    # Remove duplicates and sort
    slices_to_recalibrate = sorted(set(slices_to_recalibrate))
    
    # Determine calibration status
    calibration_status = _determine_calibration_status(
        realism_pressure=realism_pressure,
        policy_status=policy_status,
        recalibrate_count=len(slices_to_recalibrate),
        has_sharp_drift=any(
            s == StabilityClass.SHARP_DRIFT.value
            for s in stability_classes.values()
        ),
    )
    
    # Build advisory notes
    advisory_notes = _build_advisory_notes(
        calibration_status=calibration_status,
        recalibrate_count=len(slices_to_recalibrate),
        realism_pressure=realism_pressure,
        policy_status=policy_status,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calibration_status": calibration_status.value,
        "slices_to_recalibrate": slices_to_recalibrate,
        "advisory_notes": advisory_notes,
        "realism_pressure": realism_pressure,
        "recalibration_count": len(slices_to_recalibrate),
        "stability_summary": {
            "stable_count": sum(1 for s in stability_classes.values() if s == StabilityClass.STABLE.value),
            "soft_drift_count": sum(1 for s in stability_classes.values() if s == StabilityClass.SOFT_DRIFT.value),
            "sharp_drift_count": sum(1 for s in stability_classes.values() if s == StabilityClass.SHARP_DRIFT.value),
        },
    }


def _determine_calibration_status(
    realism_pressure: float,
    policy_status: str,
    recalibrate_count: int,
    has_sharp_drift: bool,
) -> CalibrationStatus:
    """
    Determine calibration status based on ratchet and policy signals.
    
    Rules:
        - BLOCK: High pressure (>0.7) or policy BLOCK or sharp drift present
        - ATTENTION: Moderate pressure (>0.3) or policy ATTENTION or many recalibrations
        - OK: Everything else
    """
    # BLOCK: Critical conditions
    if realism_pressure > 0.7 or policy_status == "BLOCK" or has_sharp_drift:
        return CalibrationStatus.BLOCK
    
    # ATTENTION: Moderate issues
    if realism_pressure > 0.3 or policy_status == "ATTENTION" or recalibrate_count > 5:
        return CalibrationStatus.ATTENTION
    
    # OK: Everything else
    return CalibrationStatus.OK


def _build_advisory_notes(
    calibration_status: CalibrationStatus,
    recalibrate_count: int,
    realism_pressure: float,
    policy_status: str,
) -> str:
    """Build neutral advisory notes describing calibration needs."""
    notes = []
    
    notes.append(f"Calibration console status: {calibration_status.value}.")
    
    if recalibrate_count > 0:
        notes.append(
            f"{recalibrate_count} scenario(s) identified for recalibration based on "
            f"retention scores, stability classes, and policy recommendations."
        )
    else:
        notes.append("No scenarios currently require recalibration.")
    
    notes.append(f"Realism pressure level: {realism_pressure:.1%}.")
    notes.append(f"Policy status: {policy_status}.")
    
    if calibration_status == CalibrationStatus.BLOCK:
        notes.append(
            "Calibration console recommends blocking scenario generation until "
            "recalibration is complete."
        )
    elif calibration_status == CalibrationStatus.ATTENTION:
        notes.append(
            "Calibration console recommends attention to identified scenarios "
            "before proceeding with additional scenario generation."
        )
    else:
        notes.append(
            "Calibration console indicates synthetic universe is within acceptable "
            "calibration parameters."
        )
    
    return " ".join(notes)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def build_complete_calibration_view(
    consistency_view: Dict[str, Any],
    realism_summary: Dict[str, Any],
    scenario_policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build complete calibration view combining ratchet and console.
    
    This is the main entry point for generating a full calibration report.
    """
    # Build ratchet
    ratchet = build_synthetic_realism_ratchet(
        consistency_view=consistency_view,
        realism_summary=realism_summary,
    )
    
    # Build calibration console
    console = build_scenario_calibration_console(
        ratchet=ratchet,
        scenario_policy=scenario_policy,
    )
    
    return {
        "label": SAFETY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ratchet": ratchet,
        "calibration_console": console,
    }


def format_calibration_console(console: Dict[str, Any]) -> str:
    """Format calibration console as human-readable text."""
    lines = [
        f"\n{SAFETY_LABEL}",
        "",
        "=" * 60,
        "SCENARIO CALIBRATION CONSOLE",
        "=" * 60,
        "",
    ]
    
    # Status
    status = console.get("calibration_status", "UNKNOWN")
    lines.append(f"Calibration Status: [{status}]")
    lines.append("")
    
    # Realism pressure
    pressure = console.get("realism_pressure", 0.0)
    lines.append(f"Realism Pressure: {pressure:.1%}")
    lines.append("")
    
    # Scenarios to recalibrate
    slices = console.get("slices_to_recalibrate", [])
    if slices:
        lines.append("SCENARIOS TO RECALIBRATE")
        lines.append("-" * 40)
        for scenario in slices:
            lines.append(f"  - {scenario}")
        lines.append("")
    else:
        lines.append("No scenarios require recalibration.")
        lines.append("")
    
    # Stability summary
    stability = console.get("stability_summary", {})
    if stability:
        lines.append("STABILITY SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Stable:        {stability.get('stable_count', 0)}")
        lines.append(f"  Soft Drift:    {stability.get('soft_drift_count', 0)}")
        lines.append(f"  Sharp Drift:   {stability.get('sharp_drift_count', 0)}")
        lines.append("")
    
    # Advisory notes
    notes = console.get("advisory_notes", "")
    if notes:
        lines.append("ADVISORY NOTES")
        lines.append("-" * 40)
        # Wrap long lines
        words = notes.split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 56:
                lines.append(current_line)
                current_line = f"  {word}"
            else:
                current_line += f" {word}" if current_line != "  " else word
        if current_line != "  ":
            lines.append(current_line)
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"{SAFETY_LABEL}")
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Example usage with mock data
    from experiments.synthetic_uplift.synthetic_real_governance import (
        build_synthetic_real_consistency_view,
        derive_synthetic_scenario_policy,
    )
    from experiments.synthetic_uplift.scenario_governance import (
        build_realism_envelope_timeline,
        summarize_synthetic_realism_for_global_health,
    )
    
    # Mock data
    mock_timeline = build_realism_envelope_timeline([
        {
            "scenario_name": "synthetic_test",
            "envelope_pass": True,
            "violated_checks": [],
            "timestamp": "2025-01-01T00:00:00Z",
        }
    ])
    
    mock_summary = summarize_synthetic_realism_for_global_health(mock_timeline)
    
    mock_consistency = build_synthetic_real_consistency_view(
        synthetic_timeline=mock_timeline,
        real_topology_health={"envelope_breach_rate": 0.05},
        real_metric_health={},
    )
    
    mock_policy = derive_synthetic_scenario_policy(mock_consistency, mock_timeline)
    
    # Build calibration view
    view = build_complete_calibration_view(
        consistency_view=mock_consistency,
        realism_summary=mock_summary,
        scenario_policy=mock_policy,
    )
    
    # Print console
    print(format_calibration_console(view["calibration_console"]))

