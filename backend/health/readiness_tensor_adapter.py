"""Readiness Tensor Engine integration adapter for global health.

STATUS: PHASE X — METRIC READINESS GOVERNANCE TILING

Provides integration between the Readiness Tensor Engine (Phase VI) and the
global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The metric_readiness tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- This tile is only attached when tensor + polygraph + phase_transition_eval are available
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

READINESS_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"
READINESS_ANNEX_SCHEMA_VERSION = "1.0.0"
READINESS_PANEL_SCHEMA_VERSION = "1.0.0"


class StatusLight:
    """Status light values for governance tiles."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def build_readiness_governance_tile(
    tensor: Dict[str, Any],
    polygraph: Dict[str, Any],
    phase_transition_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a governance-ready tile from the Readiness Tensor Engine.

    SHADOW MODE: This tile is observational only.
    It MUST NOT influence governance decisions directly.

    Args:
        tensor: From build_metric_readiness_tensor().
        polygraph: From build_metric_drift_polygraph().
        phase_transition_eval: From evaluate_phase_transition_safety_v2().

    Returns:
        Governance tile dict with:
          - status_light: "GREEN" | "YELLOW" | "RED"
          - global_norm: float (from tensor)
          - drift_momentum: float (from polygraph)
          - poly_fail_detected: bool (from polygraph)
          - transition_band: str (from phase_transition_eval)
          - slices_at_risk: List[str] (from polygraph entangled pairs / low readiness)
          - headline: str (neutral, descriptive sentence)
          - schema_version: "1.0.0"

    Guarantees:
        - Deterministic output
        - JSON-safe
        - Neutral language
    """
    global_norm = tensor.get("global_norm", 0.0)
    drift_momentum = polygraph.get("drift_momentum", 0.0)
    poly_fail_detected = polygraph.get("poly_fail_detected", False)
    transition_safe = phase_transition_eval.get("transition_safe", False)
    transition_band = phase_transition_eval.get("transition_band", "LOW")
    entangled_pairs = polygraph.get("entangled_pairs", [])
    
    # Determine status_light
    # RED if transition_safe == False
    if not transition_safe:
        status_light = StatusLight.RED
    # YELLOW if drift_momentum < -0.1 or poly_fail_detected
    elif drift_momentum < -0.1 or poly_fail_detected:
        status_light = StatusLight.YELLOW
    # GREEN otherwise
    else:
        status_light = StatusLight.GREEN
    
    # Collect slices at risk
    slices_at_risk: List[str] = []
    
    # From entangled pairs
    for pair in entangled_pairs:
        if isinstance(pair, list) and len(pair) >= 2:
            slices_at_risk.extend(pair[:2])  # Take first two elements
    
    # From low readiness in tensor
    slice_vectors = tensor.get("slice_vectors", {})
    for slice_name, vector in slice_vectors.items():
        readiness_score = vector.get("readiness_score", 1.0)
        if readiness_score < 0.5:
            if slice_name not in slices_at_risk:
                slices_at_risk.append(slice_name)
    
    # Remove duplicates and sort for determinism
    slices_at_risk = sorted(list(set(slices_at_risk)))
    
    # Generate neutral headline
    if poly_fail_detected:
        headline = "Poly-fail condition detected in readiness tensor"
    elif not transition_safe:
        blocking_count = len(phase_transition_eval.get("blocking_conditions", []))
        headline = f"Transition safety check failed ({blocking_count} blocking condition(s))"
    elif drift_momentum < -0.1:
        headline = f"Negative drift momentum detected ({drift_momentum:.3f})"
    elif global_norm < 0.5:
        headline = f"Global norm below threshold ({global_norm:.3f})"
    elif len(slices_at_risk) > 0:
        headline = f"Global norm: {global_norm:.3f}, {len(slices_at_risk)} slice(s) at risk"
    else:
        headline = f"Global norm: {global_norm:.3f}, transition band: {transition_band}"
    
    return {
        "status_light": status_light,
        "global_norm": global_norm,
        "drift_momentum": drift_momentum,
        "poly_fail_detected": poly_fail_detected,
        "transition_band": transition_band,
        "slices_at_risk": slices_at_risk,
        "headline": headline,
        "schema_version": READINESS_GOVERNANCE_TILE_SCHEMA_VERSION,
    }


def extract_readiness_signal_for_first_light(
    tensor: Dict[str, Any],
    polygraph: Dict[str, Any],
    phase_transition_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a compact readiness signal for First-Light summary JSON.

    This signal will later be embedded in First-Light summary.json.

    Args:
        tensor: From build_metric_readiness_tensor().
        polygraph: From build_metric_drift_polygraph().
        phase_transition_eval: From evaluate_phase_transition_safety_v2().

    Returns:
        Compact signal dict with:
          - global_norm: float
          - drift_momentum: float
          - transition_band: str
          - poly_fail_detected: bool

    Guarantees:
        - Deterministic output
        - JSON-safe
    """
    return {
        "global_norm": tensor.get("global_norm", 0.0),
        "drift_momentum": polygraph.get("drift_momentum", 0.0),
        "transition_band": phase_transition_eval.get("transition_band", "LOW"),
        "poly_fail_detected": polygraph.get("poly_fail_detected", False),
    }


def build_p3_readiness_summary(
    readiness_tile: Dict[str, Any],
    readiness_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build P3 readiness summary for stability report.

    SHADOW MODE: Observational only, does not gate.

    Args:
        readiness_tile: From build_readiness_governance_tile().
        readiness_signal: From extract_readiness_signal_for_first_light().

    Returns:
        P3 readiness summary dict with:
          - global_norm: float
          - drift_momentum: float
          - transition_band: str
          - poly_fail_detected: bool
          - status_light: "GREEN" | "YELLOW" | "RED" (from tile)

    Guarantees:
        - Deterministic output
        - JSON-safe
    """
    return {
        "global_norm": readiness_signal.get("global_norm", 0.0),
        "drift_momentum": readiness_signal.get("drift_momentum", 0.0),
        "transition_band": readiness_signal.get("transition_band", "LOW"),
        "poly_fail_detected": readiness_signal.get("poly_fail_detected", False),
        "status_light": readiness_tile.get("status_light", StatusLight.GREEN),
    }


def build_p4_readiness_calibration(
    tensor: Dict[str, Any],
    polygraph: Dict[str, Any],
    phase_transition_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build P4 readiness calibration for calibration report.

    SHADOW MODE: Observational only, does not gate.

    Args:
        tensor: From build_metric_readiness_tensor().
        polygraph: From build_metric_drift_polygraph().
        phase_transition_eval: From evaluate_phase_transition_safety_v2().

    Returns:
        P4 readiness calibration dict with:
          - global_norm: float
          - drift_momentum: float
          - transition_band: str
          - poly_fail_detected: bool
          - transition_safe: bool

    Guarantees:
        - Deterministic output
        - JSON-safe
    """
    return {
        "global_norm": tensor.get("global_norm", 0.0),
        "drift_momentum": polygraph.get("drift_momentum", 0.0),
        "transition_band": phase_transition_eval.get("transition_band", "LOW"),
        "poly_fail_detected": polygraph.get("poly_fail_detected", False),
        "transition_safe": phase_transition_eval.get("transition_safe", False),
    }


def build_first_light_metric_readiness_annex(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce compact First-Light readiness annex combining P3 and P4 data.

    SHADOW MODE: Observational only, does not gate.

    Args:
        p3_summary: From build_p3_readiness_summary().
        p4_calibration: From build_p4_readiness_calibration().

    Returns:
        Compact annex dict with:
          - schema_version: "1.0.0"
          - p3_global_norm: float
          - p3_transition_band: str
          - p4_global_norm: float
          - p4_transition_band: str
          - poly_fail_detected: bool (OR of P3 and P4)

    Guarantees:
        - Deterministic output
        - JSON-safe
    """
    return {
        "schema_version": READINESS_ANNEX_SCHEMA_VERSION,
        "p3_global_norm": p3_summary.get("global_norm", 0.0),
        "p3_transition_band": p3_summary.get("transition_band", "LOW"),
        "p4_global_norm": p4_calibration.get("global_norm", 0.0),
        "p4_transition_band": p4_calibration.get("transition_band", "LOW"),
        "poly_fail_detected": (
            p3_summary.get("poly_fail_detected", False) or
            p4_calibration.get("poly_fail_detected", False)
        ),
    }


def attach_metric_readiness_to_evidence(
    evidence: Dict[str, Any],
    readiness_tile: Dict[str, Any],
    readiness_signal: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach metric readiness data to evidence pack.

    SHADOW MODE: Non-mutating, returns new dict.
    This is purely for observability and evidence attachment.

    Args:
        evidence: Evidence pack dictionary (will not be mutated).
        readiness_tile: From build_readiness_governance_tile().
        readiness_signal: From extract_readiness_signal_for_first_light().
        p3_summary: Optional P3 readiness summary for annex.
        p4_calibration: Optional P4 readiness calibration for annex.

    Returns:
        New evidence dict with readiness data attached under:
          evidence["governance"]["metric_readiness"]
          If p3_summary and p4_calibration are provided, also includes:
          evidence["governance"]["metric_readiness"]["first_light_annex"]

    Guarantees:
        - Non-mutating (returns new dict)
        - Deterministic output
        - JSON-safe
    """
    # Create a copy to avoid mutation
    new_evidence = dict(evidence)
    
    # Ensure governance key exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}
    else:
        new_evidence["governance"] = dict(new_evidence["governance"])
    
    # Attach readiness data
    readiness_data = {
        "global_norm": readiness_signal.get("global_norm", 0.0),
        "transition_band": readiness_signal.get("transition_band", "LOW"),
        "transition_safe": readiness_tile.get("status_light") != StatusLight.RED,
        "drift_momentum": readiness_signal.get("drift_momentum", 0.0),
        "poly_fail_detected": readiness_signal.get("poly_fail_detected", False),
    }
    
    # Add First-Light annex if both P3 and P4 data are available
    if p3_summary is not None and p4_calibration is not None:
        readiness_data["first_light_annex"] = build_first_light_metric_readiness_annex(
            p3_summary=p3_summary,
            p4_calibration=p4_calibration,
        )
    
    new_evidence["governance"]["metric_readiness"] = readiness_data
    
    return new_evidence


def summarize_readiness_perf_budget_consistency(
    readiness_annex: Dict[str, Any],
    perf_tile: Optional[Dict[str, Any]],
    budget_tile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cross-check readiness against perf/budget tiles for consistency.

    SHADOW MODE: Observational only, flags inconsistencies but does not gate.

    Args:
        readiness_annex: From build_first_light_metric_readiness_annex().
        perf_tile: Optional performance tile (e.g., from global health).
        budget_tile: Optional budget tile (e.g., from global health).

    Returns:
        Consistency summary dict with:
          - consistency_status: "CONSISTENT" | "INCONSISTENT"
          - advisory_notes: List[str] (neutral, descriptive)
          - readiness_status: "OK" | "WARN" | "BLOCK" (derived from annex)
          - perf_status: Optional[str] (if perf_tile available)
          - budget_status: Optional[str] (if budget_tile available)

    Logic:
      - Flags when readiness is OK but perf/budget tiles report BLOCK
      - Returns neutral advisory notes describing inconsistencies

    Guarantees:
        - Deterministic output
        - JSON-safe
        - Neutral language
    """
    # Derive readiness status from annex
    # Consider readiness OK if both P3 and P4 norms are reasonable and no poly-fail
    p3_norm = readiness_annex.get("p3_global_norm", 0.0)
    p4_norm = readiness_annex.get("p4_global_norm", 0.0)
    poly_fail = readiness_annex.get("poly_fail_detected", False)
    
    # Simple heuristic: BLOCK if poly-fail or both norms very low
    if poly_fail or (p3_norm < 0.35 and p4_norm < 0.35):
        readiness_status = "BLOCK"
    # OK if both norms >= 0.5 and no poly-fail
    elif p3_norm >= 0.5 and p4_norm >= 0.5 and not poly_fail:
        readiness_status = "OK"
    # WARN otherwise
    else:
        readiness_status = "WARN"
    
    # Extract perf and budget statuses if available
    perf_status: Optional[str] = None
    if perf_tile is not None:
        perf_status_light = perf_tile.get("status_light")
        if perf_status_light == StatusLight.RED:
            perf_status = "BLOCK"
        elif perf_status_light == StatusLight.YELLOW:
            perf_status = "WARN"
        elif perf_status_light == StatusLight.GREEN:
            perf_status = "OK"
    
    budget_status: Optional[str] = None
    if budget_tile is not None:
        budget_status_light = budget_tile.get("status_light")
        if budget_status_light == StatusLight.RED:
            budget_status = "BLOCK"
        elif budget_status_light == StatusLight.YELLOW:
            budget_status = "WARN"
        elif budget_status_light == StatusLight.GREEN:
            budget_status = "OK"
    
    # Check for inconsistencies
    advisory_notes: List[str] = []
    consistency_status = "CONSISTENT"
    
    # Flag: readiness OK but perf BLOCK
    if readiness_status == "OK" and perf_status == "BLOCK":
        consistency_status = "INCONSISTENT"
        advisory_notes.append("Readiness status OK but performance tile reports BLOCK")
    
    # Flag: readiness OK but budget BLOCK
    if readiness_status == "OK" and budget_status == "BLOCK":
        consistency_status = "INCONSISTENT"
        advisory_notes.append("Readiness status OK but budget tile reports BLOCK")
    
    # Flag: readiness OK but both perf and budget BLOCK
    if readiness_status == "OK" and perf_status == "BLOCK" and budget_status == "BLOCK":
        advisory_notes.append("Readiness status OK but both performance and budget tiles report BLOCK")
    
    # Sort notes for determinism
    advisory_notes = sorted(advisory_notes)
    
    return {
        "consistency_status": consistency_status,
        "advisory_notes": advisory_notes,
        "readiness_status": readiness_status,
        "perf_status": perf_status,
        "budget_status": budget_status,
    }


def summarize_readiness_for_uplift_council(tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize readiness for Uplift Council decision-making.

    SHADOW MODE: This is advisory only, does not gate or abort.
    Provides status mapping and slice recommendations.

    Args:
        tile: From build_readiness_governance_tile().

    Returns:
        Council summary dict with:
          - status: "OK" | "WARN" | "BLOCK"
          - slices_to_hold: List[str] (if available)
          - slices_safe_to_progress: List[str] (if available)
          - global_norm: float
          - transition_band: str
          - transition_safe: bool

    Status mapping:
      - BLOCK if transition_safe == False or poly_fail_detected
      - WARN if transition_band == "MEDIUM" or drift_momentum < -0.1
      - OK otherwise

    Guarantees:
        - Deterministic output
        - JSON-safe
        - Neutral language
    """
    transition_safe = tile.get("status_light") != StatusLight.RED
    poly_fail_detected = tile.get("poly_fail_detected", False)
    transition_band = tile.get("transition_band", "LOW")
    drift_momentum = tile.get("drift_momentum", 0.0)
    global_norm = tile.get("global_norm", 0.0)
    slices_at_risk = tile.get("slices_at_risk", [])
    
    # Determine status
    # BLOCK if transition_safe == False or poly_fail_detected
    if not transition_safe or poly_fail_detected:
        status = "BLOCK"
    # WARN if transition_band == "MEDIUM" or drift_momentum < -0.1
    elif transition_band == "MEDIUM" or drift_momentum < -0.1:
        status = "WARN"
    # OK otherwise
    else:
        status = "OK"
    
    # Determine slices to hold vs safe to progress
    # For now, slices_at_risk are the ones to hold
    slices_to_hold = sorted(slices_at_risk)
    
    # Slices safe to progress would be all slices not at risk
    # But we don't have the full slice list here, so we'll leave it empty
    # or derive from autopilot_policy if available
    slices_safe_to_progress: List[str] = []
    
    return {
        "status": status,
        "slices_to_hold": slices_to_hold,
        "slices_safe_to_progress": slices_safe_to_progress,
        "global_norm": global_norm,
        "transition_band": transition_band,
        "transition_safe": transition_safe,
    }


def emit_cal_exp_metric_readiness_annex(
    cal_id: str,
    annex: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Emit per-experiment metric readiness annex for CAL-EXP-1/2/3.

    STATUS: PHASE X — P5 CALIBRATION EXPERIMENT READINESS CAPTURE

    This function creates a per-experiment annex with required fields and persists
    it to a JSON file under calibration/metric_readiness_annex_<cal_id>.json.

    SHADOW MODE CONTRACT:
    - Pure function: does not mutate input
    - Deterministic output for same cal_id + annex
    - File persistence is side-effect but does not affect control flow

    Args:
        cal_id: Calibration experiment ID (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3").
        annex: First-Light readiness annex from build_first_light_metric_readiness_annex().
        output_dir: Optional output directory. Defaults to Path("calibration").

    Returns:
        Enriched annex dict with:
          - schema_version: "1.0.0"
          - cal_id: str
          - p3_global_norm: float
          - p3_transition_band: str
          - p4_global_norm: float
          - p4_transition_band: str
          - poly_fail_detected: bool

    Guarantees:
        - Deterministic output
        - JSON-safe
        - Non-mutating (returns new dict)
    """
    # Create enriched annex with cal_id
    enriched_annex = {
        "schema_version": READINESS_ANNEX_SCHEMA_VERSION,
        "cal_id": cal_id,
        "p3_global_norm": annex.get("p3_global_norm", 0.0),
        "p3_transition_band": annex.get("p3_transition_band", "LOW"),
        "p4_global_norm": annex.get("p4_global_norm", 0.0),
        "p4_transition_band": annex.get("p4_transition_band", "LOW"),
        "poly_fail_detected": annex.get("poly_fail_detected", False),
    }
    
    # Persist to file if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"metric_readiness_annex_{cal_id}.json"
        file_path = output_dir / filename
        
        # Write JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(enriched_annex, f, indent=2, sort_keys=True)
    
    return enriched_annex


def _derive_readiness_status_from_annex(annex: Dict[str, Any]) -> str:
    """
    Derive readiness status (OK/WARN/BLOCK) from annex.

    Internal helper for panel aggregation.
    """
    p3_norm = annex.get("p3_global_norm", 0.0)
    p4_norm = annex.get("p4_global_norm", 0.0)
    poly_fail = annex.get("poly_fail_detected", False)
    
    # BLOCK if poly-fail or both norms very low
    if poly_fail or (p3_norm < 0.35 and p4_norm < 0.35):
        return "BLOCK"
    # OK if both norms >= 0.5 and no poly-fail
    elif p3_norm >= 0.5 and p4_norm >= 0.5 and not poly_fail:
        return "OK"
    # WARN otherwise
    else:
        return "WARN"


def build_metric_readiness_panel(annexes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build multi-experiment metric readiness panel aggregating across CAL-EXP-1/2/3.

    STATUS: PHASE X — P5 CALIBRATION EXPERIMENT READINESS PANEL

    This function aggregates readiness annexes across calibration experiments
    to produce a panel suitable for evidence pack inclusion.

    SHADOW MODE CONTRACT:
    - Pure function: does not mutate input
    - Deterministic output
    - Advisory only: no gating logic

    Args:
        annexes: List of per-experiment annexes from emit_cal_exp_metric_readiness_annex().
            Each annex must contain:
            - cal_id: str
            - p3_global_norm: float
            - p4_global_norm: float
            - poly_fail_detected: bool

    Returns:
        Panel dict with:
          - schema_version: "1.0.0"
          - num_experiments: int
          - num_ok: int (count of OK status)
          - num_warn: int (count of WARN status)
          - num_block: int (count of BLOCK status)
          - num_poly_fail: int (count with poly_fail_detected=True)
          - global_norm_range: Dict with min/max of p3/p4 norms
          - top_driver_cal_ids: List[str] (up to 3 cal_ids ranked by severity: BLOCK > WARN > OK, tie-break: poly_fail first, then lowest p4_norm, then cal_id)

    Guarantees:
        - Deterministic output
        - JSON-safe
    """
    if not annexes:
        return {
            "schema_version": READINESS_PANEL_SCHEMA_VERSION,
            "num_experiments": 0,
            "num_ok": 0,
            "num_warn": 0,
            "num_block": 0,
            "num_poly_fail": 0,
            "global_norm_range": {
                "p3_min": 0.0,
                "p3_max": 0.0,
                "p4_min": 0.0,
                "p4_max": 0.0,
            },
            "top_driver_cal_ids": [],
        }
    
    # Aggregate status counts
    num_ok = 0
    num_warn = 0
    num_block = 0
    num_poly_fail = 0
    
    # Collect norms for range calculation
    p3_norms: List[float] = []
    p4_norms: List[float] = []
    
    for annex in annexes:
        status = _derive_readiness_status_from_annex(annex)
        if status == "OK":
            num_ok += 1
        elif status == "WARN":
            num_warn += 1
        else:
            num_block += 1
        
        if annex.get("poly_fail_detected", False):
            num_poly_fail += 1
        
        p3_norms.append(annex.get("p3_global_norm", 0.0))
        p4_norms.append(annex.get("p4_global_norm", 0.0))
    
    # Compute norm ranges
    global_norm_range = {
        "p3_min": min(p3_norms) if p3_norms else 0.0,
        "p3_max": max(p3_norms) if p3_norms else 0.0,
        "p4_min": min(p4_norms) if p4_norms else 0.0,
        "p4_max": max(p4_norms) if p4_norms else 0.0,
    }
    
    # Compute top driver cal_ids (up to 3, ranked by severity)
    # Ranking: BLOCK > WARN > OK
    # Tie-break: poly_fail_detected=True first, then lowest p4_global_norm, then cal_id
    def _rank_key(annex: Dict[str, Any]) -> Tuple[int, bool, float, str]:
        """Return tuple for deterministic sorting: (severity_rank, poly_fail, p4_norm, cal_id)."""
        status = _derive_readiness_status_from_annex(annex)
        # Severity rank: BLOCK=0, WARN=1, OK=2 (lower is worse)
        severity_rank = 0 if status == "BLOCK" else (1 if status == "WARN" else 2)
        poly_fail = annex.get("poly_fail_detected", False)
        p4_norm = annex.get("p4_global_norm", 0.0)
        cal_id = annex.get("cal_id", "")
        # Use p4_norm directly so lower norms sort first (worse) in ascending order
        # poly_fail=True should sort first, so use not poly_fail (False sorts before True)
        return (severity_rank, not poly_fail, p4_norm, cal_id)
    
    # Sort annexes by severity (worst first)
    sorted_annexes = sorted(annexes, key=_rank_key)
    
    # Extract top 3 cal_ids
    top_driver_cal_ids = [
        annex.get("cal_id", "")
        for annex in sorted_annexes[:3]
        if annex.get("cal_id")
    ]
    
    return {
        "schema_version": READINESS_PANEL_SCHEMA_VERSION,
        "num_experiments": len(annexes),
        "num_ok": num_ok,
        "num_warn": num_warn,
        "num_block": num_block,
        "num_poly_fail": num_poly_fail,
        "global_norm_range": global_norm_range,
        "top_driver_cal_ids": top_driver_cal_ids,
    }


def attach_metric_readiness_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach metric readiness panel to evidence pack.

    SHADOW MODE: Non-mutating, returns new dict.
    This is purely for observability and evidence attachment.

    Args:
        evidence: Evidence pack dictionary (will not be mutated).
        panel: From build_metric_readiness_panel().

    Returns:
        New evidence dict with panel attached under:
          evidence["governance"]["metric_readiness_panel"]

    Guarantees:
        - Non-mutating (returns new dict)
        - Deterministic output
        - JSON-safe
    """
    # Create a copy to avoid mutation
    new_evidence = dict(evidence)
    
    # Ensure governance key exists
    if "governance" not in new_evidence:
        new_evidence["governance"] = {}
    else:
        new_evidence["governance"] = dict(new_evidence["governance"])
    
    # Attach panel
    new_evidence["governance"]["metric_readiness_panel"] = panel
    
    return new_evidence


def metric_readiness_panel_for_alignment_view(panel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert metric readiness panel to GGFL alignment view format.

    This function normalizes the metric readiness panel into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Metric readiness never triggers conflict directly (conflict: false always, invariant)

    Args:
        panel: Metric readiness panel from build_metric_readiness_panel()

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-READY" (identifies this as a readiness signal)
        - status: "ok" | "warn" (warn if any block or poly_fail)
        - conflict: false (metric readiness never triggers conflict directly, invariant)
        - drivers: List[str] (max 3, deterministic ordering: blocks, poly_fail, low p4 range)
        - summary: str (single neutral sentence)
    """
    num_block = panel.get("num_block", 0)
    num_poly_fail = panel.get("num_poly_fail", 0)
    top_driver_cal_ids = panel.get("top_driver_cal_ids", [])
    num_experiments = panel.get("num_experiments", 0)
    global_norm_range = panel.get("global_norm_range", {})
    p4_min = global_norm_range.get("p4_min", 0.0)
    
    # Determine status: warn if any block or poly_fail, otherwise ok
    status = "warn" if (num_block > 0 or num_poly_fail > 0) else "ok"
    
    # Build drivers list using reason codes (max 3, deterministic ordering)
    # Priority: BLOCK → POLY_FAIL → LOW_NORM
    drivers: List[str] = []
    
    # DRIVER_BLOCK_PRESENT: if any block detected
    if num_block > 0:
        drivers.append("DRIVER_BLOCK_PRESENT")
    
    # DRIVER_POLY_FAIL_PRESENT: if any poly_fail detected
    if num_poly_fail > 0:
        drivers.append("DRIVER_POLY_FAIL_PRESENT")
    
    # DRIVER_LOW_P4_NORM_RANGE: only if p4_min < 0.35 (documented threshold)
    if p4_min < 0.35:
        drivers.append("DRIVER_LOW_P4_NORM_RANGE")
    
    # Limit to max 3 (should never exceed, but enforce for safety)
    drivers = drivers[:3]
    
    # Build summary (single neutral sentence)
    if num_block > 0 and num_poly_fail > 0:
        summary = (
            f"Metric readiness panel: {num_experiments} experiment(s), "
            f"{num_block} BLOCK, {num_poly_fail} poly-fail detected"
        )
    elif num_block > 0:
        summary = (
            f"Metric readiness panel: {num_experiments} experiment(s), "
            f"{num_block} with BLOCK status"
        )
    elif num_poly_fail > 0:
        summary = (
            f"Metric readiness panel: {num_experiments} experiment(s), "
            f"{num_poly_fail} with poly-fail detected"
        )
    elif p4_min < 0.35:
        summary = (
            f"Metric readiness panel: {num_experiments} experiment(s), "
            f"P4 global norm range low (min={p4_min:.2f})"
        )
    else:
        summary = (
            f"Metric readiness panel: {num_experiments} experiment(s), "
            f"all experiments show OK or WARN status"
        )
    
    return {
        "signal_type": "SIG-READY",
        "status": status,
        "conflict": False,  # Metric readiness never triggers conflict directly (invariant)
        "drivers": drivers,
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }


__all__ = [
    "READINESS_ANNEX_SCHEMA_VERSION",
    "READINESS_GOVERNANCE_TILE_SCHEMA_VERSION",
    "READINESS_PANEL_SCHEMA_VERSION",
    "StatusLight",
    "attach_metric_readiness_panel_to_evidence",
    "attach_metric_readiness_to_evidence",
    "build_first_light_metric_readiness_annex",
    "build_metric_readiness_panel",
    "build_p3_readiness_summary",
    "build_p4_readiness_calibration",
    "build_readiness_governance_tile",
    "emit_cal_exp_metric_readiness_annex",
    "extract_readiness_signal_for_first_light",
    "metric_readiness_panel_for_alignment_view",
    "summarize_readiness_for_uplift_council",
    "summarize_readiness_perf_budget_consistency",
]

