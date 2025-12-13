"""Epistemic alignment integration for P3/P4 reports and evidence.

STATUS: PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

Provides integration of epistemic alignment signals into:
- P3 stability reports
- P4 calibration reports
- Evidence packs
- Uplift council summaries

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Epistemic alignment signals are purely observational
- They do NOT influence any other signals or system health classification
- No control flow depends on epistemic alignment values
- No governance writes
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.health.epistemic_alignment_adapter import (
    build_epistemic_governance_tile,
)
from backend.health.epistemic_evidence_adapter import (
    extract_epistemic_evidence_for_pack,
)


def attach_epistemic_alignment_to_p3_stability_report(
    stability_report: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach epistemic alignment summary to P3 stability report.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Epistemic alignment is observational only

    Args:
        stability_report: P3 stability report dictionary
        tile: Epistemic alignment governance tile from build_epistemic_governance_tile

    Returns:
        Updated stability report with epistemic_alignment_summary field containing:
        - tensor_norm: float
        - alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - forecast_band: "LOW" | "MEDIUM" | "HIGH"
        - misalignment_hotspots: List[str]
        - status_light: "GREEN" | "YELLOW" | "RED"
    """
    # Extract epistemic alignment data from tile
    epistemic_alignment_summary = {
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "alignment_band": tile.get("alignment_band", "MEDIUM"),
        "forecast_band": tile.get("forecast_band", "MEDIUM"),
        "misalignment_hotspots": tile.get("misalignment_hotspots", []),
        "status_light": tile.get("status_light", "YELLOW"),
    }

    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["epistemic_alignment_summary"] = epistemic_alignment_summary

    return updated_report


def attach_epistemic_alignment_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    tile: Dict[str, Any],
    forecast: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach epistemic alignment calibration to P4 calibration report.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Epistemic alignment is observational only

    Args:
        calibration_report: P4 calibration report dictionary
        tile: Epistemic alignment governance tile from build_epistemic_governance_tile
        forecast: Optional misalignment forecast from forecast_epistemic_misalignment

    Returns:
        Updated calibration report with epistemic_alignment field containing:
        - tensor_norm: float
        - alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - forecast_band: "LOW" | "MEDIUM" | "HIGH"
        - misalignment_hotspots: List[str]
        - confidence: float (from forecast if available)
    """
    # Extract epistemic alignment data from tile
    epistemic_alignment = {
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "alignment_band": tile.get("alignment_band", "MEDIUM"),
        "forecast_band": tile.get("forecast_band", "MEDIUM"),
        "misalignment_hotspots": tile.get("misalignment_hotspots", []),
    }

    # Add confidence from forecast if available
    if forecast is not None:
        epistemic_alignment["confidence"] = forecast.get("confidence", 0.5)
    else:
        epistemic_alignment["confidence"] = 0.5  # Default

    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["epistemic_alignment"] = epistemic_alignment

    return updated_report


def build_first_light_epistemic_annex(
    p3_summary: Dict[str, Any],
    p4_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First-Light epistemic annex combining P3 and P4 alignment data.

    SHADOW-MODE ONLY.
    Observational only. No gating logic.

    Args:
        p3_summary: P3 stability report with epistemic_alignment_summary field
        p4_alignment: P4 calibration report with epistemic_alignment field

    Returns:
        First-Light epistemic annex with:
        - schema_version: "1.0.0"
        - p3_tensor_norm: float
        - p3_alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_tensor_norm: float
        - p4_alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - hotspot_union: List[str] (sorted union of P3 and P4 hotspots, limited to top 5)
    """
    # Extract P3 data
    p3_epistemic = p3_summary.get("epistemic_alignment_summary", {})
    p3_tensor_norm = p3_epistemic.get("tensor_norm", 0.0)
    p3_alignment_band = p3_epistemic.get("alignment_band", "MEDIUM")
    p3_hotspots = p3_epistemic.get("misalignment_hotspots", [])

    # Extract P4 data
    p4_epistemic = p4_alignment.get("epistemic_alignment", {})
    p4_tensor_norm = p4_epistemic.get("tensor_norm", 0.0)
    p4_alignment_band = p4_epistemic.get("alignment_band", "MEDIUM")
    p4_hotspots = p4_epistemic.get("misalignment_hotspots", [])

    # Union of hotspots (sorted, limited to top 5)
    hotspot_union = sorted(set(p3_hotspots) | set(p4_hotspots))[:5]

    return {
        "schema_version": "1.0.0",
        "p3_tensor_norm": p3_tensor_norm,
        "p3_alignment_band": p3_alignment_band,
        "p4_tensor_norm": p4_tensor_norm,
        "p4_alignment_band": p4_alignment_band,
        "hotspot_union": hotspot_union,
    }


def attach_epistemic_alignment_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    compact_signal: Optional[Dict[str, Any]] = None,
    p3_report: Optional[Dict[str, Any]] = None,
    p4_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach epistemic alignment to evidence pack.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Epistemic alignment is observational only

    Args:
        evidence: Evidence dictionary
        tile: Epistemic alignment governance tile from build_epistemic_governance_tile
        compact_signal: Optional compact signal from extract_epistemic_evidence_for_pack
        p3_report: Optional P3 stability report for First-Light annex
        p4_report: Optional P4 calibration report for First-Light annex

    Returns:
        Updated evidence with epistemic_alignment under evidence["governance"]["epistemic_alignment"]
        containing: tensor_norm, alignment_band, forecast_band, hotspots
        Optionally includes first_light_annex if both p3_report and p4_report are provided
    """
    # Build epistemic alignment evidence block
    # Use compact_signal if available, otherwise extract from tile
    if compact_signal is not None:
        epistemic_evidence = {
            "tensor_norm": compact_signal.get("tensor_norm", 0.0),
            "alignment_band": tile.get("alignment_band", "MEDIUM"),
            "forecast_band": compact_signal.get("predicted_band", "MEDIUM"),
            "hotspots": compact_signal.get("misalignment_hotspots", []),
        }
    else:
        epistemic_evidence = {
            "tensor_norm": tile.get("tensor_norm", 0.0),
            "alignment_band": tile.get("alignment_band", "MEDIUM"),
            "forecast_band": tile.get("forecast_band", "MEDIUM"),
            "hotspots": tile.get("misalignment_hotspots", []),
        }

    # Add First-Light annex if both P3 and P4 reports are provided
    if p3_report is not None and p4_report is not None:
        epistemic_evidence["first_light_annex"] = build_first_light_epistemic_annex(
            p3_report, p4_report
        )

    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)

    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])

    # Attach epistemic alignment
    updated_evidence["governance"]["epistemic_alignment"] = epistemic_evidence

    return updated_evidence


def summarize_epistemic_behavior_consistency(
    epistemic_annex: Dict[str, Any],
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize epistemic behavior consistency for cross-checking.

    SHADOW-MODE consistency summary:
    - Flags when alignment bands degrade while evidence_quality trajectory_class is 'IMPROVING' or 'STABLE'
    - Returns neutral advisory_notes only.

    Args:
        epistemic_annex: First-Light epistemic annex from build_first_light_epistemic_annex
        evidence_quality: Optional evidence quality dict with trajectory_class field

    Returns:
        Consistency summary with:
        - consistency_status: "CONSISTENT" | "INCONSISTENT" | "UNKNOWN"
        - advisory_notes: List[str] (neutral, descriptive notes)
    """
    advisory_notes: list[str] = []

    # Extract alignment bands
    p3_band = epistemic_annex.get("p3_alignment_band", "MEDIUM")
    p4_band = epistemic_annex.get("p4_alignment_band", "MEDIUM")

    # Check for degradation from P3 to P4
    band_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    p3_rank = band_order.get(p3_band, 1)
    p4_rank = band_order.get(p4_band, 1)

    degraded = p4_rank < p3_rank

    # Check evidence quality trajectory if available
    if evidence_quality is not None:
        trajectory_class = evidence_quality.get("trajectory_class", "UNKNOWN")
        evidence_ok = trajectory_class in ("IMPROVING", "STABLE")

        # Flag inconsistency: epistemic degraded while evidence quality is OK
        if degraded and evidence_ok:
            consistency_status = "INCONSISTENT"
            advisory_notes.append(
                f"Epistemic alignment degraded from {p3_band} (P3) to {p4_band} (P4) "
                f"while evidence quality trajectory is {trajectory_class}."
            )
        elif degraded:
            consistency_status = "CONSISTENT"
            advisory_notes.append(
                f"Epistemic alignment degraded from {p3_band} (P3) to {p4_band} (P4). "
                f"Evidence quality trajectory: {trajectory_class}."
            )
        else:
            consistency_status = "CONSISTENT"
            advisory_notes.append(
                f"Epistemic alignment bands: P3={p3_band}, P4={p4_band}. "
                f"Evidence quality trajectory: {trajectory_class}."
            )
    else:
        # No evidence quality data
        if degraded:
            consistency_status = "UNKNOWN"
            advisory_notes.append(
                f"Epistemic alignment degraded from {p3_band} (P3) to {p4_band} (P4). "
                "Evidence quality data not available for consistency check."
            )
        else:
            consistency_status = "UNKNOWN"
            advisory_notes.append(
                f"Epistemic alignment bands: P3={p3_band}, P4={p4_band}. "
                "Evidence quality data not available for consistency check."
            )

    return {
        "consistency_status": consistency_status,
        "advisory_notes": advisory_notes,
    }


def summarize_epistemic_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize epistemic alignment for uplift council decision.

    Classification rules (observational):
    - BLOCK: alignment_band="LOW" (mapped from "BROKEN") or forecast_band="HIGH"
    - WARN: alignment_band="MEDIUM" (mapped from "DRIFTING") or forecast_band="MEDIUM"
    - OK: otherwise

    SHADOW MODE CONTRACT:
    - This is advisory only
    - No hard gates or enforcement
    - Purely observational

    Args:
        tile: Epistemic alignment governance tile from build_epistemic_governance_tile

    Returns:
        Council summary with:
        - status: "OK" | "WARN" | "BLOCK"
        - alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - forecast_band: "LOW" | "MEDIUM" | "HIGH"
        - priority_hotspots: List[str] (top 5 misalignment hotspots)
    """
    alignment_band = tile.get("alignment_band", "MEDIUM")
    forecast_band = tile.get("forecast_band", "MEDIUM")
    misalignment_hotspots = tile.get("misalignment_hotspots", [])

    # Map alignment_band: "LOW" maps to "BROKEN", "MEDIUM" maps to "DRIFTING"
    # Classification rules
    if alignment_band == "LOW" or forecast_band == "HIGH":
        status = "BLOCK"
    elif alignment_band == "MEDIUM" or forecast_band == "MEDIUM":
        status = "WARN"
    else:
        status = "OK"

    # Priority hotspots (top 5)
    priority_hotspots = misalignment_hotspots[:5]

    return {
        "status": status,
        "alignment_band": alignment_band,
        "forecast_band": forecast_band,
        "priority_hotspots": priority_hotspots,
    }


def emit_cal_exp_epistemic_annex(
    cal_id: str,
    annex: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Emit calibration experiment epistemic annex with cal_id.

    PHASE X — P5 CALIBRATION EXPERIMENT ANNEX

    Adds schema_version and cal_id to the annex, preserving all original fields.
    Pure function: deterministic output for identical inputs.

    SHADOW MODE CONTRACT:
    - This is evidence-only, not a gate
    - Annex is observational and does not influence calibration experiment behavior

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        annex: First-Light epistemic annex from build_first_light_epistemic_annex()

    Returns:
        Calibration experiment annex with:
        - schema_version: "1.0.0"
        - cal_id: str
        - p3_tensor_norm: float
        - p3_alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_tensor_norm: float
        - p4_alignment_band: "LOW" | "MEDIUM" | "HIGH"
        - hotspot_union: List[str]
    """
    # Preserve all original annex fields
    cal_annex = dict(annex)
    
    # Add schema_version and cal_id
    cal_annex["schema_version"] = "1.0.0"
    cal_annex["cal_id"] = cal_id
    
    return cal_annex


def persist_cal_exp_epistemic_annex(
    annex: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist calibration experiment epistemic annex to disk.

    PHASE X — P5 CALIBRATION SNAPSHOT PERSISTENCE

    Writes annex to calibration/epistemic_annex_<cal_id>.json.
    Creates the output directory if it doesn't exist.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions

    Args:
        annex: Calibration experiment annex from emit_cal_exp_epistemic_annex()
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration"))

    Returns:
        Path to the written annex file

    Raises:
        IOError: If the file cannot be written
    """
    cal_id = annex.get("cal_id", "UNKNOWN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"epistemic_annex_{cal_id}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annex, f, indent=2, sort_keys=True)
    
    return output_path


def _determine_epistemic_reason_code(
    p3_band: str,
    p4_band: str,
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Determine deterministic reason code for epistemic inconsistency.

    SHADOW MODE: Deterministic mapping based on alignment bands and evidence quality.

    Args:
        p3_band: P3 alignment band ("LOW" | "MEDIUM" | "HIGH")
        p4_band: P4 alignment band ("LOW" | "MEDIUM" | "HIGH")
        evidence_quality: Optional evidence quality dict with trajectory_class

    Returns:
        Reason code string:
        - EPI_DEGRADED_EVID_IMPROVING: Epistemic degraded, evidence IMPROVING
        - EPI_DEGRADED_EVID_STABLE: Epistemic degraded, evidence STABLE
        - EPI_IMPROVED_EVID_DEGRADING: Epistemic improved, evidence DEGRADING
        - EPI_UNKNOWN_EVID_PRESENT: Epistemic changed, evidence present but trajectory unclear
        - EPI_DEGRADED_EVID_UNKNOWN: Epistemic degraded, evidence quality unavailable
    """
    band_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    p3_rank = band_order.get(p3_band, 1)
    p4_rank = band_order.get(p4_band, 1)

    degraded = p4_rank < p3_rank
    improved = p4_rank > p3_rank

    if evidence_quality is not None:
        trajectory_class = evidence_quality.get("trajectory_class", "UNKNOWN")
        
        if degraded and trajectory_class == "IMPROVING":
            return "EPI_DEGRADED_EVID_IMPROVING"
        elif degraded and trajectory_class == "STABLE":
            return "EPI_DEGRADED_EVID_STABLE"
        elif improved and trajectory_class == "DEGRADING":
            return "EPI_IMPROVED_EVID_DEGRADING"
        elif degraded or improved:
            # Epistemic changed but evidence trajectory is unclear (not IMPROVING/STABLE/DEGRADING)
            return "EPI_UNKNOWN_EVID_PRESENT"
        else:
            # No change in epistemic, but evidence present
            return "EPI_UNKNOWN_EVID_PRESENT"
    else:
        # No evidence quality data
        if degraded:
            return "EPI_DEGRADED_EVID_UNKNOWN"
        else:
            # No degradation, but no evidence quality data
            return "EPI_UNKNOWN_EVID_PRESENT"


def build_epistemic_consistency_panel(
    annexes: List[Dict[str, Any]],
    consistency_blocks: List[Dict[str, Any]],
    evidence_quality_list: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Build epistemic consistency panel aggregating across calibration experiments.

    PHASE X — P5 CALIBRATION EPISTEMIC PANEL

    Aggregates consistency status across multiple calibration experiments to provide
    a cross-experiment view of epistemic alignment vs evidence quality consistency.

    SHADOW MODE CONTRACT:
    - This is evidence-only, not a gate
    - Panel is observational and does not influence calibration decisions
    - Calibration cross-check, not a direct gate

    Args:
        annexes: List of calibration experiment annexes from emit_cal_exp_epistemic_annex()
        consistency_blocks: List of consistency summaries from summarize_epistemic_behavior_consistency()
            Must be in same order as annexes (one per experiment)
        evidence_quality_list: Optional list of evidence quality dicts (one per experiment)
            Must be in same order as annexes. If None, reason codes will use EPI_DEGRADED_EVID_UNKNOWN
            or EPI_UNKNOWN_EVID_PRESENT as appropriate.

    Returns:
        Consistency panel with:
        - schema_version: "1.0.0"
        - num_experiments: int
        - num_consistent: int
        - num_inconsistent: int
        - num_unknown: int
        - experiments_inconsistent: List[Dict[str, str]] (cal_id, reason, reason_code)
    """
    num_experiments = len(annexes)
    
    # Count consistency statuses
    num_consistent = 0
    num_inconsistent = 0
    num_unknown = 0
    experiments_inconsistent: List[Dict[str, str]] = []
    
    # Process each experiment
    for i, (annex, consistency) in enumerate(zip(annexes, consistency_blocks)):
        cal_id = annex.get("cal_id", f"UNKNOWN_{i}")
        status = consistency.get("consistency_status", "UNKNOWN")
        
        # Get evidence quality for this experiment
        evidence_quality = None
        if evidence_quality_list is not None and i < len(evidence_quality_list):
            evidence_quality = evidence_quality_list[i]
        
        if status == "CONSISTENT":
            num_consistent += 1
        elif status == "INCONSISTENT":
            num_inconsistent += 1
            # Extract brief reason from advisory notes
            notes = consistency.get("advisory_notes", [])
            reason = notes[0] if notes else "Epistemic alignment and evidence quality trajectories disagree."
            
            # Determine reason code
            p3_band = annex.get("p3_alignment_band", "MEDIUM")
            p4_band = annex.get("p4_alignment_band", "MEDIUM")
            reason_code = _determine_epistemic_reason_code(p3_band, p4_band, evidence_quality)
            
            experiments_inconsistent.append({
                "cal_id": cal_id,
                "reason": reason,
                "reason_code": reason_code,
            })
        else:  # UNKNOWN
            num_unknown += 1
    
    # Sort experiments_inconsistent by cal_id for deterministic ordering
    experiments_inconsistent.sort(key=lambda x: x["cal_id"])
    
    return {
        "schema_version": "1.0.0",
        "num_experiments": num_experiments,
        "num_consistent": num_consistent,
        "num_inconsistent": num_inconsistent,
        "num_unknown": num_unknown,
        "experiments_inconsistent": experiments_inconsistent,
    }


def attach_epistemic_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach epistemic consistency panel to evidence pack.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Epistemic panel is observational only

    Args:
        evidence: Evidence dictionary
        panel: Epistemic consistency panel from build_epistemic_consistency_panel()

    Returns:
        Updated evidence with epistemic_panel under evidence["governance"]["epistemic_panel"]
    """
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)

    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])

    # Attach epistemic panel
    updated_evidence["governance"]["epistemic_panel"] = panel

    return updated_evidence


def epistemic_panel_for_alignment_view(
    panel_or_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert epistemic panel to GGFL alignment view format.

    PHASE X — GGFL ADAPTER FOR EPISTEMIC PANEL

    Normalizes the epistemic panel into the Global Governance Fusion Layer (GGFL)
    unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Epistemic panel never triggers conflict directly (conflict: false always)
    - Deterministic output for identical inputs (byte-identical for identical inputs)

    Args:
        panel_or_signal: Epistemic panel from build_epistemic_consistency_panel()
            or signal from first_light_status.json signals["epistemic_panel"]

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-EPI" (identifies this as an epistemic signal)
        - status: "ok" | "warn" (warn if num_inconsistent > 0)
        - conflict: false (epistemic panel never triggers conflict directly, invariant)
        - drivers: List[str] (up to 3 reason-code drivers, deterministic ordering)
        - summary: str (one neutral sentence)
        - weight_hint: "LOW" (ensures epistemic panel doesn't overpower fusion)
        - shadow_mode_invariants: Dict with advisory_only, no_enforcement, conflict_invariant
    """
    num_inconsistent = panel_or_signal.get("num_inconsistent", 0)
    num_experiments = panel_or_signal.get("num_experiments", 0)
    
    # Determine status: warn if any inconsistencies, otherwise ok
    status = "warn" if num_inconsistent > 0 else "ok"
    
    # Extract top reason code using canonical selection (highest count desc, then reason_code asc)
    # Use top_reason_code from signal if present, otherwise compute from histogram
    top_reason_code = panel_or_signal.get("top_reason_code", "UNKNOWN")
    if top_reason_code == "UNKNOWN":
        reason_code_histogram = panel_or_signal.get("reason_code_histogram", {})
        if reason_code_histogram:
            # Canonical selection: highest count desc, then reason_code asc (tie-breaker)
            sorted_by_count = sorted(
                reason_code_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for desc, reason_code for asc
            )
            top_reason_code = sorted_by_count[0][0]
    
    # Extract top cal_ids (use top_inconsistent_cal_ids_top3 from signal if present)
    top_cal_ids = panel_or_signal.get("top_inconsistent_cal_ids_top3", [])
    if not top_cal_ids:
        # Fallback: extract from experiments_inconsistent (truncate to top 3)
        experiments_inconsistent = panel_or_signal.get("experiments_inconsistent", [])
        top_cal_ids = sorted([
            exp.get("cal_id") for exp in experiments_inconsistent
            if exp.get("cal_id")
        ])[:3]  # Truncate to top 3
    
    # Build reason-code drivers (max 3, deterministic ordering)
    drivers: List[str] = []
    
    # 1. Inconsistent present (if any inconsistencies)
    if num_inconsistent > 0:
        drivers.append("DRIVER_EPI_INCONSISTENT_PRESENT")
    
    # 2. Top reason (if not UNKNOWN)
    if top_reason_code != "UNKNOWN":
        drivers.append(f"DRIVER_TOP_REASON_{top_reason_code}")
    
    # 3. Top cal ids (if present)
    if top_cal_ids:
        drivers.append("DRIVER_TOP_CAL_IDS_PRESENT")
    
    # Limit to 3 drivers (already enforced by logic above)
    drivers = drivers[:3]
    
    # Build neutral summary sentence
    if num_inconsistent > 0:
        summary = (
            f"Epistemic panel: {num_inconsistent} out of {num_experiments} "
            f"calibration experiments show inconsistent alignment between "
            f"epistemic signals and evidence quality trajectories."
        )
    else:
        summary = (
            f"Epistemic panel: {num_experiments} calibration experiments "
            f"show consistent alignment between epistemic signals and evidence quality."
        )
    
    return {
        "signal_type": "SIG-EPI",
        "status": status,
        "conflict": False,  # Epistemic panel never triggers conflict directly (invariant)
        "drivers": drivers,
        "summary": summary,
        "weight_hint": "LOW",  # Ensures epistemic panel doesn't overpower fusion semantics
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }


__all__ = [
    "attach_epistemic_alignment_to_p3_stability_report",
    "attach_epistemic_alignment_to_p4_calibration_report",
    "attach_epistemic_alignment_to_evidence",
    "build_first_light_epistemic_annex",
    "build_epistemic_consistency_panel",
    "emit_cal_exp_epistemic_annex",
    "epistemic_panel_for_alignment_view",
    "persist_cal_exp_epistemic_annex",
    "attach_epistemic_panel_to_evidence",
    "summarize_epistemic_behavior_consistency",
    "summarize_epistemic_for_uplift_council",
]

