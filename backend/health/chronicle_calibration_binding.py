"""Chronicle governance binding for P4 calibration reports.

STATUS: PHASE X — CALIBRATION REPORT INTEGRATION

Provides helper functions to attach chronicle governance data to P4 calibration reports.

SHADOW MODE CONTRACT:
- All functions are read-only (aside from dict modification)
- Chronicle governance data is purely observational
- No control flow depends on this data
"""

from typing import Any, Dict, List

from backend.health.chronicle_governance_adapter import (
    extract_chronicle_drift_signal,
)


def attach_chronicle_governance_to_calibration_report(
    calibration_report: Dict[str, Any],
    recurrence_projection: Dict[str, Any],
    invariant_check: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach chronicle governance section to P4 calibration report.

    STATUS: PHASE X — CALIBRATION REPORT BINDING

    Adds chronicle_governance section to calibration report with:
    - recurrence_likelihood
    - band: "LOW" | "MEDIUM" | "HIGH"
    - invariants_ok: bool
    - drift_notes: List[str] (from highlighted_cases)

    SHADOW MODE CONTRACT:
    - This function modifies the calibration_report dict in-place
    - The attachment is purely observational
    - No control flow depends on the attached data

    Args:
        calibration_report: P4 calibration report dictionary (will be modified in-place)
        recurrence_projection: Recurrence projection from build_recurrence_projection_engine()
        invariant_check: Invariant check from build_phase_transition_drift_invariant_checker()
        tile: Chronicle governance tile from build_chronicle_governance_tile()

    Returns:
        Modified calibration report with chronicle_governance section
    """
    # Extract drift signal
    drift_signal = extract_chronicle_drift_signal(recurrence_projection, invariant_check)
    
    # Build drift notes from highlighted cases
    highlighted_cases = tile.get("highlighted_cases", [])
    drift_notes: List[str] = []
    
    if highlighted_cases:
        drift_notes.extend(highlighted_cases)
    
    # Add invariant explanations if violations exist
    broken_invariants = invariant_check.get("broken_invariants", [])
    if broken_invariants:
        drift_notes.append(f"Invariant violations: {len(broken_invariants)}")
        # Add first few violation descriptions
        explanations = invariant_check.get("explanations", [])
        if explanations:
            drift_notes.extend(explanations[:2])  # Limit to first 2
    
    # Attach chronicle_governance section
    calibration_report["chronicle_governance"] = {
        "recurrence_likelihood": drift_signal.get("recurrence_likelihood", 0.0),
        "band": drift_signal.get("band", "LOW"),
        "invariants_ok": drift_signal.get("invariants_ok", True),
        "drift_notes": drift_notes,
    }
    
    return calibration_report


__all__ = [
    "attach_chronicle_governance_to_calibration_report",
]

