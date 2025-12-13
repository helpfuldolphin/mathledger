"""Coherence integration for P3/P4 reports and evidence.

STATUS: PHASE X — COHERENCE GOVERNANCE LAYER

Provides integration of coherence signals into:
- P3 stability reports
- P4 calibration reports
- Evidence packs
- Uplift council summaries

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Coherence signals are purely observational
- They do NOT influence any other signals or system health classification
- No control flow depends on coherence values
- No governance writes
"""

from typing import Any, Dict, Optional

from backend.health.coherence_adapter import (
    extract_coherence_drift_signal,
)


def attach_coherence_to_p3_stability_report(
    stability_report: Dict[str, Any],
    coherence_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach coherence summary to P3 stability report.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Coherence is observational only
    
    Args:
        stability_report: P3 stability report dictionary
        coherence_tile: Coherence governance tile from build_coherence_governance_tile
        
    Returns:
        Updated stability report with coherence_summary field
    """
    # Extract coherence data from tile
    coherence_summary = {
        "coherence_band": coherence_tile.get("coherence_band", "PARTIAL"),
        "global_coherence_index": coherence_tile.get("global_coherence_index", 0.5),
        "slices_at_risk": coherence_tile.get("slices_at_risk", []),
        "root_incoherence_causes": coherence_tile.get("drivers", []),
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["coherence_summary"] = coherence_summary
    
    return updated_report


def attach_coherence_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    drift_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach coherence calibration to P4 calibration report.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Coherence is observational only
    
    Args:
        calibration_report: P4 calibration report dictionary
        drift_signal: Coherence drift signal from extract_coherence_drift_signal
        
    Returns:
        Updated calibration report with coherence_calibration field
    """
    # Extract coherence data from drift signal
    coherence_calibration = {
        "global_coherence_index": drift_signal.get("global_index", 0.5),
        "coherence_band": drift_signal.get("coherence_band", "PARTIAL"),
        "low_slices": drift_signal.get("low_slices", []),
        "structural_notes": [
            f"Coherence band: {drift_signal.get('coherence_band', 'PARTIAL')}.",
            f"Global coherence index: {drift_signal.get('global_index', 0.5):.3f}.",
            f"{len(drift_signal.get('low_slices', []))} slice{'s' if len(drift_signal.get('low_slices', [])) != 1 else ''} below coherence threshold.",
        ],
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["coherence_calibration"] = coherence_calibration
    
    return updated_report


def attach_coherence_to_evidence(
    evidence: Dict[str, Any],
    coherence_tile: Dict[str, Any],
    drift_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach coherence to evidence pack.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Coherence is observational only
    
    Args:
        evidence: Evidence dictionary
        coherence_tile: Coherence governance tile from build_coherence_governance_tile
        drift_signal: Optional coherence drift signal from extract_coherence_drift_signal
        
    Returns:
        Updated evidence with coherence under evidence["governance"]["coherence"]
    """
    # Build coherence evidence block
    coherence_evidence = {
        "band": coherence_tile.get("coherence_band", "PARTIAL"),
        "global_index": coherence_tile.get("global_coherence_index", 0.5),
        "slices_at_risk": coherence_tile.get("slices_at_risk", []),
    }
    
    # Add First Light summary (compact view)
    coherence_evidence["first_light_summary"] = {
        "coherence_band": coherence_tile.get("coherence_band", "PARTIAL"),
        "global_index": coherence_tile.get("global_coherence_index", 0.5),
        "slices_at_risk": coherence_tile.get("slices_at_risk", []),
    }
    
    # Add drift annotations if provided
    if drift_signal is not None:
        coherence_evidence["drift_annotations"] = {
            "low_slices": drift_signal.get("low_slices", []),
            "coherence_band": drift_signal.get("coherence_band", "PARTIAL"),
        }
    
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])
    
    # Attach coherence
    updated_evidence["governance"]["coherence"] = coherence_evidence
    
    return updated_evidence


def summarize_coherence_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize coherence for uplift council decision (mini-view).
    
    Maps coherence band to council status:
    - MISALIGNED → BLOCK
    - PARTIAL → WARN
    - COHERENT → OK
    
    SHADOW MODE CONTRACT:
    - This is advisory only
    - No hard gates or enforcement
    - Purely observational
    
    Args:
        tile: Coherence governance tile from build_coherence_governance_tile
        
    Returns:
        Council summary with:
        - status: "OK" | "WARN" | "BLOCK"
        - coherence_band: "COHERENT" | "PARTIAL" | "MISALIGNED"
        - slices_at_risk: List[str]
        - headline: str (from tile)
    """
    coherence_band = tile.get("coherence_band", "PARTIAL")
    slices_at_risk = tile.get("slices_at_risk", [])
    headline = tile.get("headline", "")
    
    # Map coherence band to council status
    if coherence_band == "MISALIGNED":
        status = "BLOCK"
    elif coherence_band == "PARTIAL":
        status = "WARN"
    else:  # COHERENT
        status = "OK"
    
    return {
        "status": status,
        "coherence_band": coherence_band,
        "slices_at_risk": slices_at_risk,
        "headline": headline,
    }


__all__ = [
    "attach_coherence_to_p3_stability_report",
    "attach_coherence_to_p4_calibration_report",
    "attach_coherence_to_evidence",
    "summarize_coherence_for_uplift_council",
]

