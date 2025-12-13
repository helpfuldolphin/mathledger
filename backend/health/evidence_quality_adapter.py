"""Evidence quality governance adapter for global health.

STATUS: PHASE X — EVIDENCE QUALITY GOVERNANCE

Provides integration between evidence quality phase-portrait analysis and
the global health surface builder.

GOVERNANCE CONTRACT:
- All functions are read-only and side-effect free
- The evidence_quality tile is purely observational (SHADOW MODE)
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- No governance writes (read-only monitoring)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

EVIDENCE_QUALITY_TILE_SCHEMA_VERSION = "1.0.0"
FAILURE_SHELF_SCHEMA_VERSION = "1.0.0"


def build_evidence_governance_tile(
    phase_portrait: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    director_panel_v2: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build evidence quality governance tile for global health surface.
    
    STATUS: PHASE X — EVIDENCE QUALITY GOVERNANCE
    
    Combines phase portrait, envelope forecast, and director panel v2
    into a unified governance tile for the global health dashboard.
    
    SHADOW MODE: Observational only.
    
    GOVERNANCE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - No governance writes
    
    Args:
        phase_portrait: Optional phase portrait from build_evidence_phase_portrait.
            If None, trajectory_class will be "UNKNOWN".
        forecast: Optional forecast from forecast_evidence_envelope.
            If None, predicted_band will be "UNKNOWN".
        director_panel_v2: Optional director panel from build_evidence_director_panel_v2.
            If None, status_light and other fields will be "UNKNOWN".
    
    Returns:
        Evidence quality governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
        - trajectory_class: "IMPROVING" | "STABLE" | "OSCILLATING" | "DEGRADING" | "UNKNOWN"
        - predicted_band: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
        - cycles_until_risk: int | None
        - regression_status: "OK" | "ATTENTION" | "BLOCK" | "UNKNOWN"
        - flags: List[str]
        - headline: str
    """
    # Extract from director_panel_v2 if available
    if director_panel_v2:
        status_light = director_panel_v2.get("status_light", "UNKNOWN")
        trajectory_class = director_panel_v2.get("trajectory_class", "UNKNOWN")
        regression_status = director_panel_v2.get("regression_status", "UNKNOWN")
        flags = director_panel_v2.get("flags", [])
        headline = director_panel_v2.get("headline", "Evidence quality status unknown")
    else:
        status_light = "UNKNOWN"
        trajectory_class = phase_portrait.get("trajectory_class", "UNKNOWN") if phase_portrait else "UNKNOWN"
        regression_status = "UNKNOWN"
        flags = []
        headline = "Evidence quality data not available"
    
    # Extract from forecast if available
    if forecast:
        predicted_band = forecast.get("predicted_band", "UNKNOWN")
        cycles_until_risk = forecast.get("cycles_until_risk", None)
    else:
        predicted_band = "UNKNOWN"
        cycles_until_risk = None
    
    # Fallback to phase_portrait for trajectory_class if not in director_panel_v2
    if trajectory_class == "UNKNOWN" and phase_portrait:
        trajectory_class = phase_portrait.get("trajectory_class", "UNKNOWN")
    
    return {
        "schema_version": EVIDENCE_QUALITY_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "trajectory_class": trajectory_class,
        "predicted_band": predicted_band,
        "cycles_until_risk": cycles_until_risk,
        "regression_status": regression_status,
        "flags": sorted(flags) if flags else [],
        "headline": headline,
    }


def extract_evidence_quality_summary_for_first_light(
    director_panel_v2: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract compact evidence quality summary for First-Light summary.json.
    
    STATUS: PHASE X — EVIDENCE QUALITY FIRST-LIGHT INTEGRATION
    
    This helper extracts the minimal set of evidence quality signals needed
    for First Light summary.json. It is purely observational and does not
    influence any gates or decisions.
    
    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    
    Args:
        director_panel_v2: Optional director panel from build_evidence_director_panel_v2.
        forecast: Optional forecast from forecast_evidence_envelope.
    
    Returns:
        Compact evidence quality summary dictionary with:
        - trajectory_class: "IMPROVING" | "STABLE" | "OSCILLATING" | "DEGRADING" | "UNKNOWN"
        - predicted_band: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
        - cycles_until_risk: int | None
        - flags: List[str]
    """
    if director_panel_v2:
        trajectory_class = director_panel_v2.get("trajectory_class", "UNKNOWN")
        flags = director_panel_v2.get("flags", [])
    else:
        trajectory_class = "UNKNOWN"
        flags = []
    
    if forecast:
        predicted_band = forecast.get("predicted_band", "UNKNOWN")
        cycles_until_risk = forecast.get("cycles_until_risk", None)
    else:
        predicted_band = "UNKNOWN"
        cycles_until_risk = None
    
    return {
        "trajectory_class": trajectory_class,
        "predicted_band": predicted_band,
        "cycles_until_risk": cycles_until_risk,
        "flags": sorted(flags) if flags else [],
    }


def build_first_light_failure_shelf(
    phase_portrait: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First-Light failure shelf: compact list of most concerning episodes for reviewers.
    
    STATUS: PHASE X — EVIDENCE QUALITY FAILURE SHELF
    
    This function creates a curated summary that highlights the most concerning aspects
    of evidence quality for external reviewers. It uses trajectory_class and flags to
    identify concerning patterns.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The failure shelf is purely observational (curation aid only)
    - No gating logic; this is a curation aid only
    - No control flow depends on the shelf contents
    
    Args:
        phase_portrait: Phase portrait from build_evidence_phase_portrait.
        forecast: Forecast from forecast_evidence_envelope.
    
    Returns:
        Failure shelf dictionary with:
        - schema_version: "1.0.0"
        - trajectory_class: "IMPROVING" | "STABLE" | "OSCILLATING" | "DEGRADING"
        - predicted_band: "LOW" | "MEDIUM" | "HIGH"
        - cycles_until_risk: int | None
        - flags: List[str] (from forecast neutral_explanation, sorted)
    """
    trajectory_class = phase_portrait.get("trajectory_class", "UNKNOWN")
    predicted_band = forecast.get("predicted_band", "UNKNOWN")
    cycles_until_risk = forecast.get("cycles_until_risk", None)
    
    # Extract flags from forecast neutral_explanation
    # These are the concerning signals that reviewers should focus on
    flags = forecast.get("neutral_explanation", [])
    flags = sorted(flags) if flags else []
    
    return {
        "schema_version": "1.0.0",
        "trajectory_class": trajectory_class,
        "predicted_band": predicted_band,
        "cycles_until_risk": cycles_until_risk,
        "flags": flags,
    }


def attach_evidence_quality_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    phase_portrait: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach evidence quality governance tile to an evidence pack (read-only, additive).
    
    STATUS: PHASE X — EVIDENCE QUALITY EVIDENCE INTEGRATION
    
    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["evidence_quality"].
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        tile: Evidence quality governance tile from build_evidence_governance_tile().
        phase_portrait: Optional phase portrait for failure shelf construction.
        forecast: Optional forecast for failure shelf construction.
    
    Returns:
        New dict with evidence contents plus evidence_quality tile attached under governance key.
    
    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_evidence_governance_tile(phase_portrait, forecast, director_panel_v2)
        >>> enriched = attach_evidence_quality_to_evidence(evidence, tile, phase_portrait, forecast)
        >>> enriched["governance"]["evidence_quality"]["trajectory_class"]
        'IMPROVING'
    """
    # Non-mutating: create new dict
    updated = dict(evidence)
    
    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])
    
    # Extract relevant fields for evidence (trajectory, band, cycles, flags)
    evidence_quality_summary = {
        "trajectory_class": tile.get("trajectory_class", "UNKNOWN"),
        "predicted_band": tile.get("predicted_band", "UNKNOWN"),
        "cycles_until_risk": tile.get("cycles_until_risk", None),
        "flags": tile.get("flags", []),
    }
    
    # Attach summary
    updated["governance"]["evidence_quality"] = evidence_quality_summary
    
    # Attach failure shelf if phase_portrait and forecast are provided
    if phase_portrait is not None and forecast is not None:
        failure_shelf = build_first_light_failure_shelf(phase_portrait, forecast)
        updated["governance"]["evidence_quality"]["first_light_failure_shelf"] = failure_shelf
    
    return updated


def attach_evidence_quality_to_p3_stability_report(
    stability_report: Dict[str, Any],
    evidence_quality_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach evidence quality summary to P3 stability report.
    
    STATUS: PHASE X — EVIDENCE QUALITY P3 INTEGRATION
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Evidence quality is observational only
    - Does not alter gates or decisions
    
    Args:
        stability_report: P3 stability report dictionary
        evidence_quality_summary: Evidence quality summary from extract_evidence_quality_summary_for_first_light
        
    Returns:
        Updated stability report with evidence_quality section
    """
    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["evidence_quality"] = evidence_quality_summary
    
    return updated_report


def attach_evidence_quality_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    evidence_quality_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach evidence quality summary to P4 calibration report.
    
    STATUS: PHASE X — EVIDENCE QUALITY P4 INTEGRATION
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Evidence quality is observational only
    - Does not alter gates or decisions
    
    Args:
        calibration_report: P4 calibration report dictionary
        evidence_quality_summary: Evidence quality summary from extract_evidence_quality_summary_for_first_light
        
    Returns:
        Updated calibration report with evidence_quality section
    """
    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["evidence_quality"] = evidence_quality_summary
    
    return updated_report


def emit_cal_exp_failure_shelf(
    cal_id: str,
    shelf: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Emit CAL-EXP failure shelf with cal_id and episode metadata.
    
    STATUS: PHASE X — EVIDENCE QUALITY CAL-EXP SHELF EMISSION
    
    This function enriches a failure shelf with calibration experiment metadata
    and optionally persists it to disk. The shelf is used to track concerning
    episodes per calibration experiment.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction and optional file write)
    - The failure shelf is purely observational (curation aid only)
    - No gating logic; this is a curation aid only
    - File persistence is optional and observational only
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "cal_exp1", "CAL-EXP-1")
        shelf: Failure shelf from build_first_light_failure_shelf()
        output_dir: Optional directory to persist shelf JSON file.
            If None, shelf is returned but not persisted.
    
    Returns:
        Enriched failure shelf dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - episode_id: str (synthetic: "{cal_id}_episode_1")
        - rank: int (always 1 for single-experiment shelf)
        - trajectory_class: from shelf
        - predicted_band: from shelf
        - cycles_until_risk: from shelf
        - flags: from shelf (sorted)
    """
    # Extract fields from input shelf
    trajectory_class = shelf.get("trajectory_class", "UNKNOWN")
    predicted_band = shelf.get("predicted_band", "UNKNOWN")
    cycles_until_risk = shelf.get("cycles_until_risk", None)
    flags = shelf.get("flags", [])
    flags = sorted(flags) if flags else []
    
    # Build enriched shelf with cal_id and episode metadata
    enriched_shelf = {
        "schema_version": FAILURE_SHELF_SCHEMA_VERSION,
        "cal_id": cal_id,
        "episode_id": f"{cal_id}_episode_1",
        "rank": 1,  # Single-experiment shelf always has rank 1
        "trajectory_class": trajectory_class,
        "predicted_band": predicted_band,
        "cycles_until_risk": cycles_until_risk,
        "flags": flags,
    }
    
    # Persist to disk if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"evidence_failure_shelf_{cal_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_shelf, f, indent=2, sort_keys=True)
    
    return enriched_shelf


def build_global_failure_shortlist(
    shelves: List[Dict[str, Any]],
    max_items: int = 10,
) -> Dict[str, Any]:
    """
    Build global failure shortlist from multiple CAL-EXP failure shelves.
    
    STATUS: PHASE X — EVIDENCE QUALITY GLOBAL FAILURE SHORTLIST
    
    This function aggregates failure shelves from multiple calibration experiments,
    sorts them by severity heuristic, and returns the top N most concerning episodes
    for reviewers.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The shortlist is purely observational (curation aid only)
    - No gating logic; this is a reviewer convenience index, not a gate
    - No control flow depends on the shortlist contents
    
    Severity heuristic:
    1. predicted_band: LOW > MEDIUM > HIGH (most concerning first)
    2. cycles_until_risk: lower is more concerning (0 is most concerning)
    3. trajectory_class: DEGRADING > OSCILLATING > STABLE > IMPROVING
    
    Args:
        shelves: List of failure shelves from emit_cal_exp_failure_shelf()
        max_items: Maximum number of items to include in shortlist (default: 10)
    
    Returns:
        Global failure shortlist dictionary with:
        - schema_version: "1.0.0"
        - items: List[Dict] with cal_id, trajectory_class, predicted_band, cycles_until_risk, flags
        - total_shelves: int (total number of shelves processed)
    """
    if not shelves:
        return {
            "schema_version": FAILURE_SHELF_SCHEMA_VERSION,
            "items": [],
            "total_shelves": 0,
        }
    
    # Define severity ordering
    band_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}
    trajectory_order = {"DEGRADING": 0, "OSCILLATING": 1, "STABLE": 2, "IMPROVING": 3, "UNKNOWN": 4}
    
    # Sort shelves by severity heuristic
    def severity_key(shelf: Dict[str, Any]) -> tuple:
        """Sort key: (band_priority, cycles_until_risk, trajectory_priority)."""
        predicted_band = shelf.get("predicted_band", "UNKNOWN")
        cycles_until_risk = shelf.get("cycles_until_risk")
        trajectory_class = shelf.get("trajectory_class", "UNKNOWN")
        
        # Use high value for None cycles_until_risk (less concerning)
        cycles_val = cycles_until_risk if cycles_until_risk is not None else 999
        
        return (
            band_order.get(predicted_band, 3),
            cycles_val,
            trajectory_order.get(trajectory_class, 4),
        )
    
    # Sort by severity (most concerning first)
    sorted_shelves = sorted(shelves, key=severity_key)
    
    # Truncate to max_items
    top_shelves = sorted_shelves[:max_items]
    
    # Build items list (extract relevant fields)
    items = []
    for idx, shelf in enumerate(top_shelves, start=1):
        cal_id = shelf.get("cal_id", "unknown")
        item = {
            "rank": idx,
            "cal_id": cal_id,
            "episode_id": shelf.get("episode_id", f"unknown_episode_{idx}"),
            "trajectory_class": shelf.get("trajectory_class", "UNKNOWN"),
            "predicted_band": shelf.get("predicted_band", "UNKNOWN"),
            "cycles_until_risk": shelf.get("cycles_until_risk", None),
            "flags": sorted(shelf.get("flags", [])) if shelf.get("flags") else [],
            "evidence_path_hint": f"calibration/evidence_failure_shelf_{cal_id}.json",
        }
        items.append(item)
    
    return {
        "schema_version": FAILURE_SHELF_SCHEMA_VERSION,
        "items": items,
        "total_shelves": len(shelves),
    }


def attach_global_failure_shortlist_to_evidence(
    evidence: Dict[str, Any],
    shortlist: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach global failure shortlist to evidence pack.
    
    STATUS: PHASE X — EVIDENCE QUALITY GLOBAL SHORTLIST INTEGRATION
    
    This function attaches the global failure shortlist under
    evidence["governance"]["evidence_failure_shortlist"].
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The shortlist is purely observational (reviewer convenience index)
    - No gating logic; this is a curation aid only
    - Non-mutating: returns new dict, does not modify input
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        shortlist: Global failure shortlist from build_global_failure_shortlist().
    
    Returns:
        New dict with evidence contents plus evidence_failure_shortlist attached under governance key.
    """
    # Non-mutating: create new dict
    updated = dict(evidence)
    
    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])
    
    # Attach shortlist
    updated["governance"]["evidence_failure_shortlist"] = shortlist
    
    return updated


def extract_evidence_failure_shortlist_signal_for_status(
    pack_manifest: Optional[Dict[str, Any]] = None,
    evidence_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract evidence failure shortlist signal for First Light status.json.
    
    STATUS: PHASE X — EVIDENCE QUALITY FAILURE SHORTLIST STATUS EXTRACTION
    
    This function extracts a compact signal from the evidence failure shortlist
    for inclusion in First Light status files. It provides top 5 items and
    advisory warnings if needed.
    
    Manifest-first extraction contract:
    - Primary: pack_manifest["governance"]["evidence_failure_shortlist"]
    - Fallback: evidence_data["governance"]["evidence_failure_shortlist"]
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only; neutral phrasing
    
    Args:
        pack_manifest: Optional manifest dictionary (preferred source).
        evidence_data: Optional evidence.json dictionary (fallback source).
    
    Returns:
        Compact signal dictionary with:
        - extraction_source: "manifest" | "evidence.json" | None
        - total_items: int
        - top5: List[Dict] with cal_id, episode_id, predicted_band, cycles_until_risk
        Or None if evidence_failure_shortlist is not present in either source.
    """
    shortlist = None
    extraction_source = "MISSING"
    
    # Manifest-first extraction
    if pack_manifest:
        governance = pack_manifest.get("governance", {})
        shortlist = governance.get("evidence_failure_shortlist")
        if shortlist:
            extraction_source = "MANIFEST"
    
    # Fallback to evidence.json
    if not shortlist and evidence_data:
        governance = evidence_data.get("governance", {})
        shortlist = governance.get("evidence_failure_shortlist")
        if shortlist:
            extraction_source = "EVIDENCE_JSON"
    
    if not shortlist:
        return None
    
    items = shortlist.get("items", [])
    total_items = len(items)
    
    # Extract top 5 items (or all if fewer than 5)
    top5_items = items[:5]
    
    # Build top5 list with required fields
    top5 = []
    for item in top5_items:
        top5.append({
            "cal_id": item.get("cal_id", "unknown"),
            "episode_id": item.get("episode_id", "unknown_episode"),
            "predicted_band": item.get("predicted_band", "UNKNOWN"),
            "cycles_until_risk": item.get("cycles_until_risk", None),
        })
    
    signal = {
        "extraction_source": extraction_source,
        "total_items": total_items,
        "top5": top5,
    }
    
    return signal


def extract_evidence_failure_shortlist_warnings(
    pack_manifest: Optional[Dict[str, Any]] = None,
    evidence_data: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Extract advisory warnings from evidence failure shortlist.
    
    STATUS: PHASE X — EVIDENCE QUALITY FAILURE SHORTLIST WARNINGS
    
    This function extracts advisory warnings based on the evidence failure shortlist.
    Currently warns if any item in top5 has predicted_band == "HIGH" (unexpected in
    a failure shortlist, may indicate severity ordering issue).
    
    Manifest-first extraction contract:
    - Primary: pack_manifest["governance"]["evidence_failure_shortlist"]
    - Fallback: evidence_data["governance"]["evidence_failure_shortlist"]
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from list construction)
    - Warnings are purely observational (advisory only)
    - No control flow depends on the warnings
    - Neutral phrasing only
    - Returns exactly 1 warning max (even if multiple HIGH items)
    
    Args:
        pack_manifest: Optional manifest dictionary (preferred source).
        evidence_data: Optional evidence.json dictionary (fallback source).
    
    Returns:
        List of advisory warning strings (empty if no warnings, max 1 warning).
    """
    warnings = []
    
    shortlist = None
    
    # Manifest-first extraction
    if pack_manifest:
        governance = pack_manifest.get("governance", {})
        shortlist = governance.get("evidence_failure_shortlist")
    
    # Fallback to evidence.json
    if not shortlist and evidence_data:
        governance = evidence_data.get("governance", {})
        shortlist = governance.get("evidence_failure_shortlist")
    
    if not shortlist:
        return warnings
    
    items = shortlist.get("items", [])
    top5_items = items[:5]
    
    # Check for HIGH predicted_band items in top5 (unexpected in failure shortlist)
    high_band_items = [
        item for item in top5_items
        if item.get("predicted_band") == "HIGH"
    ]
    
    if high_band_items:
        high_band_count = len(high_band_items)
        cal_ids = sorted([item.get("cal_id", "unknown") for item in high_band_items])
        # Limit to top 3 cal_ids for brevity
        top_cal_ids = cal_ids[:3]
        top_cal_ids_str = ", ".join(top_cal_ids) if top_cal_ids else "none"
        
        warnings.append(
            f"Evidence failure shortlist contains {high_band_count} item(s) "
            f"with predicted_band HIGH in top 5 (high_band_count_in_top5: {high_band_count}, "
            f"top_cal_ids: [{top_cal_ids_str}])"
        )
    
    # Cap to exactly 1 warning max
    return warnings[:1]


def evidence_failure_shortlist_for_alignment_view(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build evidence failure shortlist signal for Global Governance First Light (GGFL) alignment view.
    
    STATUS: PHASE X — EVIDENCE QUALITY FAILURE SHORTLIST GGFL ADAPTER
    
    This function converts the evidence failure shortlist status signal into a format
    suitable for the GGFL alignment view. It provides a low-weight, non-conflicting
    signal that indicates evidence quality concerns.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only; neutral phrasing
    - LOW weight, no conflict
    
    Args:
        signal: Evidence failure shortlist signal from extract_evidence_failure_shortlist_signal_for_status().
    
    Returns:
        GGFL alignment view signal dictionary with:
        - signal_type: "SIG-EVID"
        - status: "ok" | "warn" (warn if any HIGH in top5)
        - conflict: false (always)
        - weight_hint: "LOW"
        - drivers: List[str] (reason codes)
        - summary: str (neutral sentence)
        - shadow_mode_invariants: Dict[str, bool] (all_divergence_logged_only, no_governance_modification, no_abort_enforcement)
    """
    # SHADOW MODE invariants (always true for this adapter)
    shadow_mode_invariants = {
        "all_divergence_logged_only": True,
        "no_governance_modification": True,
        "no_abort_enforcement": True,
    }
    
    if not signal:
        return {
            "signal_type": "SIG-EVID",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": [],
            "summary": "Evidence failure shortlist not available",
            "shadow_mode_invariants": shadow_mode_invariants,
        }
    
    top5 = signal.get("top5", [])
    
    # Check for HIGH predicted_band items in top5
    high_band_items = [
        item for item in top5
        if item.get("predicted_band") == "HIGH"
    ]
    
    # Determine status
    if high_band_items:
        status = "warn"
        drivers = ["DRIVER_HIGH_BAND_PRESENT"]
        summary = (
            f"Evidence failure shortlist contains {len(high_band_items)} item(s) "
            f"with predicted_band HIGH in top 5, which may indicate severity ordering issues."
        )
    else:
        status = "ok"
        drivers = []
        total_items = signal.get("total_items", 0)
        if total_items > 0:
            summary = (
                f"Evidence failure shortlist contains {total_items} item(s). "
                f"Top 5 items show no unexpected HIGH predicted_band values."
            )
        else:
            summary = "Evidence failure shortlist is empty."
    
    return {
        "signal_type": "SIG-EVID",
        "status": status,
        "conflict": False,
        "weight_hint": "LOW",
        "drivers": drivers,
        "summary": summary,
        "shadow_mode_invariants": shadow_mode_invariants,
    }

