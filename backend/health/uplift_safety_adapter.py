"""
Uplift Safety Engine integration adapter for global health.

STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILING

Provides integration between the Uplift Safety Engine (Phase VI) and the
global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The uplift_safety tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- This tile does NOT control deployments; it summarizes risk
- Tile is only attached when safety_tensor + stability_forecaster + gate_decision are available
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


class StatusLight:
    """Status light values for governance tiles."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def _map_decision_to_status_light(decision: str) -> str:
    """
    Map uplift safety decision to status light.

    Args:
        decision: "PASS" | "WARN" | "BLOCK"

    Returns:
        "GREEN" | "YELLOW" | "RED"
    """
    decision_upper = decision.upper()
    if decision_upper == "PASS":
        return StatusLight.GREEN
    elif decision_upper == "WARN":
        return StatusLight.YELLOW
    elif decision_upper == "BLOCK":
        return StatusLight.RED
    else:
        # Default to YELLOW for unknown decisions
        return StatusLight.YELLOW


def build_uplift_safety_governance_tile(
    safety_tensor: Dict[str, Any],
    stability_forecaster: Dict[str, Any],
    gate_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Governance tile summarizing uplift safety and stability.

    SHADOW MODE: This tile does NOT control deployments; it summarizes risk.

    Args:
        safety_tensor: From build_global_uplift_safety_tensor().
            Expected keys: "uplift_risk_band", "tensor_norm", "hotspot_axes"
        stability_forecaster: From build_uplift_stability_forecaster().
            Expected keys: "current_stability", "stability_trend", "instability_prediction"
        gate_decision: From compute_maas_uplift_gate_v3().
            Expected keys: "uplift_safety_decision", "decision_rationale", "risk_band"

    Returns:
        Governance tile dict with:
          - status_light: "GREEN" | "YELLOW" | "RED" (mapped from decision)
          - uplift_safety_decision: "PASS" | "WARN" | "BLOCK" (from gate)
          - risk_band: "LOW" | "MEDIUM" | "HIGH" (from safety tensor)
          - tensor_norm: float (from safety tensor)
          - current_stability: "STABLE" | "UNSTABLE" | "DEGRADING" (from forecaster)
          - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING" (from forecaster)
          - instability_prediction: Dict with predicted cycles and confidence
          - decision_rationale: List[str] (from gate)
          - headline: str (neutral, descriptive sentence)
          - schema_version: "1.0.0"

    Guarantees:
        - Deterministic output
        - JSON-safe
        - Neutral language
    """
    # Extract fields from gate decision
    uplift_safety_decision = gate_decision.get("uplift_safety_decision", "PASS")
    decision_rationale = gate_decision.get("decision_rationale", [])
    gate_risk_band = gate_decision.get("risk_band", "LOW")

    # Extract fields from safety tensor
    tensor_risk_band = safety_tensor.get("uplift_risk_band", "LOW")
    tensor_norm = safety_tensor.get("tensor_norm", 0.0)
    hotspot_axes = safety_tensor.get("hotspot_axes", [])

    # Prefer gate risk_band if available, otherwise use tensor risk_band
    risk_band = gate_risk_band if gate_risk_band else tensor_risk_band

    # Extract fields from stability forecaster
    current_stability = stability_forecaster.get("current_stability", "STABLE")
    stability_trend = stability_forecaster.get("stability_trend", "STABLE")
    instability_prediction = stability_forecaster.get("instability_prediction", {})

    # Map decision to status light
    status_light = _map_decision_to_status_light(uplift_safety_decision)

    # Validate status_light
    if status_light not in (StatusLight.GREEN, StatusLight.YELLOW, StatusLight.RED):
        raise ValueError(
            f"Invalid status_light: {status_light}. "
            f"Must be one of: GREEN, YELLOW, RED"
        )

    # Validate risk_band
    if risk_band not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(
            f"Invalid risk_band: {risk_band}. "
            f"Must be one of: LOW, MEDIUM, HIGH"
        )

    # Build neutral headline
    if uplift_safety_decision == "BLOCK":
        if current_stability == "UNSTABLE":
            headline = f"Uplift safety blocked: stability is {current_stability.lower()}, risk band {risk_band}"
        elif instability_prediction.get("predicted_instability_cycles"):
            cycles = instability_prediction["predicted_instability_cycles"]
            headline = f"Uplift safety blocked: instability predicted in cycles {cycles}, risk band {risk_band}"
        else:
            headline = f"Uplift safety blocked: risk band {risk_band}, tensor norm {tensor_norm:.3f}"
    elif uplift_safety_decision == "WARN":
        if current_stability == "DEGRADING":
            headline = f"Uplift safety warning: stability trend is {stability_trend.lower()}, risk band {risk_band}"
        else:
            headline = f"Uplift safety warning: risk band {risk_band}, tensor norm {tensor_norm:.3f}"
    else:  # PASS
        if hotspot_axes:
            headline = f"Uplift safety pass: risk band {risk_band}, {len(hotspot_axes)} hotspot axis(es)"
        else:
            headline = f"Uplift safety pass: risk band {risk_band}, tensor norm {tensor_norm:.3f}"

    return {
        "status_light": status_light,
        "uplift_safety_decision": uplift_safety_decision,
        "risk_band": risk_band,
        "tensor_norm": tensor_norm,
        "current_stability": current_stability,
        "stability_trend": stability_trend,
        "instability_prediction": instability_prediction,
        "decision_rationale": decision_rationale,
        "headline": headline,
        "schema_version": UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
    }


def extract_uplift_safety_signal_for_first_light(
    safety_tensor: Dict[str, Any],
    stability_forecaster: Dict[str, Any],
    gate_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact signal for First-Light summary.json.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILE

    Extracts a compact uplift safety signal from safety tensor, stability forecaster,
    and gate decision for inclusion in First Light summary.json.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        safety_tensor: From build_global_uplift_safety_tensor().
            Expected keys: "uplift_risk_band"
        stability_forecaster: From build_uplift_stability_forecaster().
            Expected keys: "current_stability", "stability_trend", "instability_prediction"
        gate_decision: From compute_maas_uplift_gate_v3().
            Expected keys: "uplift_safety_decision"

    Returns:
        Dictionary with:
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - uplift_safety_decision: "PASS" | "WARN" | "BLOCK"
        - current_stability: "STABLE" | "UNSTABLE" | "DEGRADING"
        - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - predicted_instability_cycles: List[int] (if any)
    """
    risk_band = safety_tensor.get("uplift_risk_band", "LOW")
    uplift_safety_decision = gate_decision.get("uplift_safety_decision", "PASS")
    current_stability = stability_forecaster.get("current_stability", "STABLE")
    stability_trend = stability_forecaster.get("stability_trend", "STABLE")
    instability_prediction = stability_forecaster.get("instability_prediction", {})
    predicted_instability_cycles = instability_prediction.get(
        "predicted_instability_cycles", []
    )

    signal: Dict[str, Any] = {
        "risk_band": risk_band,
        "uplift_safety_decision": uplift_safety_decision,
        "current_stability": current_stability,
        "stability_trend": stability_trend,
    }

    # Only include predicted_instability_cycles if there are any
    if predicted_instability_cycles:
        signal["predicted_instability_cycles"] = predicted_instability_cycles

    return signal


def build_p3_uplift_safety_summary(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build uplift safety summary for P3 first_light_stability_report.json.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILE

    Adds uplift_safety_summary section with risk band, decision, stability,
    and predicted instability cycles.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        signal: First Light signal from extract_uplift_safety_signal_for_first_light().

    Returns:
        Dictionary with:
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - uplift_safety_decision: "PASS" | "WARN" | "BLOCK"
        - current_stability: "STABLE" | "UNSTABLE" | "DEGRADING"
        - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - predicted_instability_cycles: List[int] (if any)
    """
    summary: Dict[str, Any] = {
        "risk_band": signal.get("risk_band", "LOW"),
        "uplift_safety_decision": signal.get("uplift_safety_decision", "PASS"),
        "current_stability": signal.get("current_stability", "STABLE"),
        "stability_trend": signal.get("stability_trend", "STABLE"),
    }

    # Only include predicted_instability_cycles if present
    if "predicted_instability_cycles" in signal:
        summary["predicted_instability_cycles"] = signal["predicted_instability_cycles"]

    return summary


def build_p4_uplift_safety_calibration(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build uplift safety calibration data for P4 p4_calibration_report.json.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILE

    Adds uplift_safety section with tensor norm, risk band, stability trend,
    decision, predicted cycles, and rationale.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned calibration is purely observational
    - No control flow depends on the calibration contents

    Args:
        tile: Governance tile from build_uplift_safety_governance_tile().

    Returns:
        Dictionary with:
        - tensor_norm: float
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - decision: "PASS" | "WARN" | "BLOCK"
        - predicted_instability_cycles: List[int] (if any)
        - decision_rationale: List[str]
    """
    calibration: Dict[str, Any] = {
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "risk_band": tile.get("risk_band", "LOW"),
        "stability_trend": tile.get("stability_trend", "STABLE"),
        "decision": tile.get("uplift_safety_decision", "PASS"),
        "decision_rationale": tile.get("decision_rationale", []),
    }

    # Extract predicted cycles from instability_prediction if present
    instability_prediction = tile.get("instability_prediction", {})
    if isinstance(instability_prediction, dict):
        predicted_cycles = instability_prediction.get("predicted_instability_cycles", [])
        if predicted_cycles:
            calibration["predicted_instability_cycles"] = predicted_cycles

    return calibration


def attach_uplift_safety_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    signal: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
    alignment_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach uplift safety governance to evidence pack (read-only, additive).

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE EVIDENCE INTEGRATION

    This is a read-only, additive operation. Returns a new dict with
    uplift safety data attached. Does not modify the input evidence dict.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached data is purely observational
    - No control flow depends on the evidence contents
    - Does not influence evidence pack validation or processing

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        tile: Governance tile from build_uplift_safety_governance_tile().
        signal: First Light signal from extract_uplift_safety_signal_for_first_light().
        p3_summary: Optional P3 summary from build_p3_uplift_safety_summary().
        p4_calibration: Optional P4 calibration from build_p4_uplift_safety_calibration().
        alignment_panel: Optional gate alignment panel from build_gate_alignment_panel().

    Returns:
        New dict with evidence contents plus uplift safety data attached under
        "governance.uplift_safety" key. If p3_summary and p4_calibration are provided,
        also includes "first_light_gate_annex". If alignment_panel is provided,
        also includes "uplift_gate_alignment_panel" at top level of governance.
    """
    # Create a deep copy to avoid mutation
    import copy
    result = copy.deepcopy(evidence)

    # Initialize governance section if needed
    if "governance" not in result:
        result["governance"] = {}

    # Attach uplift safety data
    uplift_safety_data: Dict[str, Any] = {
        "schema_version": UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
        "tile": tile,
        "signal": signal,
    }

    # Add gate annex if P3 and P4 data are provided
    if p3_summary is not None and p4_calibration is not None:
        uplift_safety_data["first_light_gate_annex"] = build_first_light_uplift_gate_annex(
            p3_summary, p4_calibration
        )

    result["governance"]["uplift_safety"] = uplift_safety_data

    # Add alignment panel if provided
    if alignment_panel is not None:
        result["governance"]["uplift_gate_alignment_panel"] = alignment_panel

        # Extract and attach status signal
        signal = extract_uplift_gate_alignment_signal(alignment_panel)

        # Ensure signals key exists
        if "signals" not in result:
            result["signals"] = {}
        else:
            result["signals"] = dict(result["signals"])

        # Attach uplift gate alignment signal
        result["signals"]["uplift_gate_alignment"] = signal

    return result


def build_first_light_uplift_gate_annex(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Gate Readiness Annex for First Light.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILE

    Synthesizes P3 and P4 data into a compact annex summarizing the state
    of the uplift safety gate for First Light.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned annex is purely observational
    - No control flow depends on the annex contents
    - SHADOW MODE only, no gating

    Args:
        p3_summary: P3 summary from build_p3_uplift_safety_summary().
            Expected keys: "uplift_safety_decision", "risk_band"
        p4_calibration: P4 calibration from build_p4_uplift_safety_calibration().
            Expected keys: "decision", "risk_band", "stability_trend"

    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - p3_decision: "PASS" | "WARN" | "BLOCK"
        - p3_risk_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_decision: "PASS" | "WARN" | "BLOCK"
        - p4_risk_band: "LOW" | "MEDIUM" | "HIGH"
        - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING"
    """
    return {
        "schema_version": UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
        "p3_decision": p3_summary.get("uplift_safety_decision", "PASS"),
        "p3_risk_band": p3_summary.get("risk_band", "LOW"),
        "p4_decision": p4_calibration.get("decision", "PASS"),
        "p4_risk_band": p4_calibration.get("risk_band", "LOW"),
        "stability_trend": p4_calibration.get("stability_trend", "STABLE"),
    }


def summarize_uplift_safety_for_council(
    tile: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize uplift safety for Uplift Council integration.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILE

    Translates uplift safety decision into council-compatible format.
    Maps PASS→OK, WARN→WARN, BLOCK→BLOCK.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - SHADOW MODE: no gating, advisory only

    Args:
        tile: Governance tile from build_uplift_safety_governance_tile().
        p3_summary: Optional P3 summary from build_p3_uplift_safety_summary().
        p4_calibration: Optional P4 calibration from build_p4_uplift_safety_calibration().

    Returns:
        Dictionary with:
        - status: "OK" | "WARN" | "BLOCK" (mapped from decision)
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - stability_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - predicted_instability_horizon: Optional[List[int]] (predicted cycles)
        - rationale: List[str] (decision rationale)
        - gate_alignment_ok: bool (True if P3 and P4 decisions are both PASS/OK or WARN)
    """
    decision = tile.get("uplift_safety_decision", "PASS")

    # Map decision to council status
    if decision == "BLOCK":
        council_status = "BLOCK"
    elif decision == "WARN":
        council_status = "WARN"
    else:  # PASS
        council_status = "OK"

    summary: Dict[str, Any] = {
        "status": council_status,
        "risk_band": tile.get("risk_band", "LOW"),
        "stability_trend": tile.get("stability_trend", "STABLE"),
        "rationale": tile.get("decision_rationale", []),
    }

    # Extract predicted instability horizon from instability_prediction
    instability_prediction = tile.get("instability_prediction", {})
    if isinstance(instability_prediction, dict):
        predicted_cycles = instability_prediction.get("predicted_instability_cycles", [])
        if predicted_cycles:
            summary["predicted_instability_horizon"] = predicted_cycles

    # Calculate gate_alignment_ok if P3 and P4 data are provided
    if p3_summary is not None and p4_calibration is not None:
        p3_decision = p3_summary.get("uplift_safety_decision", "PASS")
        p4_decision = p4_calibration.get("decision", "PASS")

        # Map decisions to council statuses for comparison
        def _decision_to_council_status(dec: str) -> str:
            if dec == "BLOCK":
                return "BLOCK"
            elif dec == "WARN":
                return "WARN"
            else:  # PASS
                return "OK"

        p3_status = _decision_to_council_status(p3_decision)
        p4_status = _decision_to_council_status(p4_decision)

        # gate_alignment_ok is True if both are OK or both are WARN
        # (BLOCK in either makes it False)
        gate_alignment_ok = (
            (p3_status == "OK" and p4_status == "OK")
            or (p3_status == "WARN" and p4_status == "WARN")
            or (p3_status == "OK" and p4_status == "WARN")
            or (p3_status == "WARN" and p4_status == "OK")
        ) and p3_status != "BLOCK" and p4_status != "BLOCK"

        summary["gate_alignment_ok"] = gate_alignment_ok

    return summary


def export_gate_annex_per_experiment(
    annex: Dict[str, Any],
    cal_id: str,
    output_dir: Path,
) -> Path:
    """
    Export gate annex per CAL-EXP to calibration/uplift_gate_annex_<cal_id>.json.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILING

    Persists a gate readiness annex for a single calibration experiment.
    This enables aggregation across multiple CAL-EXP runs for alignment analysis.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions
    - Non-mutating (returns new path, does not modify annex)

    Args:
        annex: Gate annex from build_first_light_uplift_gate_annex().
            Expected keys: p3_decision, p3_risk_band, p4_decision, p4_risk_band, stability_trend
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3").
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration")).

    Returns:
        Path to the written annex file.

    Raises:
        IOError: If the file cannot be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add cal_id to annex for traceability
    annex_with_cal_id = dict(annex)
    annex_with_cal_id["cal_id"] = cal_id

    annex_path = output_dir / f"uplift_gate_annex_{cal_id}.json"

    with open(annex_path, "w", encoding="utf-8") as f:
        json.dump(annex_with_cal_id, f, indent=2, sort_keys=True)

    return annex_path


def extract_uplift_gate_alignment_signal(
    alignment_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract uplift gate alignment signal for status surface.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILING

    Extracts a compact signal from the gate alignment panel for inclusion
    in the status surface under signals["uplift_gate_alignment"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - SHADOW MODE only, no gating

    Args:
        alignment_panel: Gate alignment panel from build_gate_alignment_panel().

    Returns:
        Dictionary with:
        - alignment_rate: float
        - misaligned_count: int
        - top_misaligned_cal_ids: List[str] (top 3, sorted)
        - reason_code_histogram: Dict[str, int] (counts by reason code)
    """
    alignment_rate = alignment_panel.get("alignment_rate", 0.0)
    misaligned_count = alignment_panel.get("misaligned_count", 0)
    misalignment_details = alignment_panel.get("misalignment_details", [])

    # Extract top 3 misaligned cal_ids (already sorted by cal_id in panel)
    top_misaligned_cal_ids = [
        detail["cal_id"] for detail in misalignment_details[:3]
    ]

    # Build reason code histogram
    reason_code_histogram: Dict[str, int] = {}
    for detail in misalignment_details:
        reason_code = detail.get("reason_code", "UNKNOWN")
        reason_code_histogram[reason_code] = reason_code_histogram.get(reason_code, 0) + 1

    # Sort histogram keys for determinism
    reason_code_histogram = dict(sorted(reason_code_histogram.items()))

    return {
        "alignment_rate": alignment_rate,
        "misaligned_count": misaligned_count,
        "top_misaligned_cal_ids": top_misaligned_cal_ids,
        "reason_code_histogram": reason_code_histogram,
    }


def build_gate_alignment_panel(
    annexes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build gate alignment panel from multiple calibration experiment annexes.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILING

    Aggregates gate readiness annexes across CAL-EXP runs to provide a
    calibration-level alignment panel for auditors. Reports alignment
    statistics and identifies misaligned experiments.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned panel is purely observational
    - No control flow depends on the panel contents
    - SHADOW MODE only, no gating
    - For pre-LastMile reasoning, stays SHADOW-ONLY until Phase Y gating is authorized

    Args:
        annexes: List of gate annexes from build_first_light_uplift_gate_annex().
            Each annex should contain: p3_decision, p4_decision, cal_id (optional)

    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - total_experiments: int
        - aligned_count: int (experiments where P3/P4 decisions are aligned)
        - misaligned_count: int (experiments where P3/P4 decisions are not aligned)
        - experiments_misaligned: List[str] (cal_ids where one is BLOCK and the other is not)
        - alignment_rate: float (aligned_count / total_experiments)
        - misalignment_details: List[Dict] with cal_id, reason_code, p3_decision, p4_decision
            - reason_code: "P3_BLOCK" | "P4_BLOCK" | "BOTH_BLOCK"
    """
    if not annexes:
        return {
            "schema_version": UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
            "total_experiments": 0,
            "aligned_count": 0,
            "misaligned_count": 0,
            "experiments_misaligned": [],
            "alignment_rate": 0.0,
            "misalignment_details": [],
        }

    aligned_count = 0
    misaligned_count = 0
    experiments_misaligned: List[str] = []
    misalignment_details: List[Dict[str, Any]] = []

    for annex in annexes:
        p3_decision = annex.get("p3_decision", "PASS")
        p4_decision = annex.get("p4_decision", "PASS")
        cal_id = annex.get("cal_id", "UNKNOWN")

        # Determine if aligned
        # Aligned: both PASS/WARN, or one PASS and one WARN
        # Misaligned: either is BLOCK
        p3_is_block = p3_decision == "BLOCK"
        p4_is_block = p4_decision == "BLOCK"

        if p3_is_block or p4_is_block:
            misaligned_count += 1
            experiments_misaligned.append(cal_id)

            # Determine reason code
            if p3_is_block and p4_is_block:
                reason_code = "BOTH_BLOCK"
            elif p3_is_block:
                reason_code = "P3_BLOCK"
            else:  # p4_is_block
                reason_code = "P4_BLOCK"

            misalignment_details.append({
                "cal_id": cal_id,
                "reason_code": reason_code,
                "p3_decision": p3_decision,
                "p4_decision": p4_decision,
            })
        else:
            aligned_count += 1

    total_experiments = len(annexes)
    alignment_rate = aligned_count / total_experiments if total_experiments > 0 else 0.0

    # Sort for determinism
    experiments_misaligned = sorted(experiments_misaligned)
    # Sort misalignment_details by cal_id for determinism
    misalignment_details = sorted(misalignment_details, key=lambda x: x["cal_id"])

    return {
        "schema_version": UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION,
        "total_experiments": total_experiments,
        "aligned_count": aligned_count,
        "misaligned_count": misaligned_count,
        "experiments_misaligned": experiments_misaligned,  # Preserved for backward compatibility
        "alignment_rate": round(alignment_rate, 3),  # Round for determinism
        "misalignment_details": misalignment_details,
    }


# Reason code constants for gate alignment drivers
DRIVER_TOP_REASON_P3_BLOCK = "DRIVER_TOP_REASON_P3_BLOCK"
DRIVER_TOP_REASON_P4_BLOCK = "DRIVER_TOP_REASON_P4_BLOCK"
DRIVER_TOP_REASON_BOTH_BLOCK = "DRIVER_TOP_REASON_BOTH_BLOCK"
DRIVER_MISALIGNED_COUNT_PRESENT = "DRIVER_MISALIGNED_COUNT_PRESENT"
DRIVER_TOP_CAL_IDS_PRESENT = "DRIVER_TOP_CAL_IDS_PRESENT"


def uplift_gate_alignment_for_alignment_view(
    signal_or_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert uplift gate alignment signal or panel to GGFL alignment view format.

    STATUS: PHASE X — UPLIFT SAFETY GOVERNANCE TILING

    Normalizes uplift gate alignment signal or panel into Global Governance Fusion Layer
    (GGFL) unified format for cross-subsystem alignment views. Uses reason-code drivers
    to avoid interpretive drift.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - SHADOW MODE only, no gating

    Args:
        signal_or_panel: Either:
            - Alignment panel from build_gate_alignment_panel() (preferred)
            - Status signal from extract_uplift_gate_alignment_signal() (fallback)

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-GATE" (constant)
        - status: "ok" | "warn" (warn if misaligned_count > 0)
        - conflict: False (constant, gate alignment never conflicts)
        - drivers: List[str] (reason codes: DRIVER_TOP_REASON_<CODE>, DRIVER_MISALIGNED_COUNT_PRESENT, DRIVER_TOP_CAL_IDS_PRESENT)
        - summary: str (single neutral sentence)
        - shadow_mode_invariants: Dict with advisory_only, no_enforcement, conflict_invariant (all True)
        - weight_hint: "LOW" (advisory only, low weight)
    """
    # Determine if input is panel or signal
    misaligned_count = signal_or_panel.get("misaligned_count", 0)
    
    # Initialize top_reason_code to None (will be derived if needed)
    top_reason_code = None
    
    # If it's a panel, extract signal first
    if "misalignment_details" in signal_or_panel:
        # It's a panel, extract signal
        alignment_signal = extract_uplift_gate_alignment_signal(signal_or_panel)
        misaligned_count = alignment_signal.get("misaligned_count", 0)
        top_misaligned_cal_ids = alignment_signal.get("top_misaligned_cal_ids", [])
        reason_code_histogram = alignment_signal.get("reason_code_histogram", {})
        # Derive top_reason_code deterministically from histogram
        if reason_code_histogram:
            # Deterministic: sort by count (descending), then by code (ascending) for tie-break
            sorted_items = sorted(
                reason_code_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for descending, then code ascending
            )
            top_reason_code = sorted_items[0][0] if sorted_items else None
    else:
        # It's already a signal
        top_misaligned_cal_ids = signal_or_panel.get("top_misaligned_cal_ids", [])
        reason_code_histogram = signal_or_panel.get("reason_code_histogram", {})
        # Prefer top_reason_code from signal if available (already derived deterministically)
        top_reason_code = signal_or_panel.get("top_reason_code")
        if not top_reason_code and reason_code_histogram:
            # Derive deterministically: sort by count (descending), then by code (ascending) for tie-break
            sorted_items = sorted(
                reason_code_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for descending, then code ascending
            )
            top_reason_code = sorted_items[0][0] if sorted_items else None
    
    # Determine status: warn if misaligned_count > 0, otherwise ok
    status = "warn" if misaligned_count > 0 else "ok"
    
    # Build drivers list using reason codes (deterministic ordering: reason → count → cal ids)
    drivers: List[str] = []
    
    # 1. Top reason code driver (deterministic)
    if top_reason_code:
        if top_reason_code == "P3_BLOCK":
            drivers.append(DRIVER_TOP_REASON_P3_BLOCK)
        elif top_reason_code == "P4_BLOCK":
            drivers.append(DRIVER_TOP_REASON_P4_BLOCK)
        elif top_reason_code == "BOTH_BLOCK":
            drivers.append(DRIVER_TOP_REASON_BOTH_BLOCK)
        # If unknown reason code, skip (should not happen, but defensive)
    
    # 2. Misaligned count present driver
    if misaligned_count > 0:
        drivers.append(DRIVER_MISALIGNED_COUNT_PRESENT)
    
    # 3. Top cal_ids present driver (if space available)
    if len(drivers) < 3 and top_misaligned_cal_ids:
        drivers.append(DRIVER_TOP_CAL_IDS_PRESENT)
    
    # Limit to max 3 drivers (already enforced by logic above, but explicit for safety)
    drivers = drivers[:3]
    
    # Build summary (single neutral sentence)
    total_experiments = signal_or_panel.get("total_experiments", 0)
    if total_experiments == 0:
        summary = "No calibration experiments in uplift gate alignment panel"
    elif misaligned_count == 0:
        summary = f"Uplift gate alignment: {total_experiments} experiment(s), all aligned"
    else:
        # Include top reason code in summary if available
        if top_reason_code:
            summary = (
                f"Uplift gate alignment: {total_experiments} experiment(s), "
                f"{misaligned_count} misaligned (top reason: {top_reason_code})"
            )
        else:
            summary = (
                f"Uplift gate alignment: {total_experiments} experiment(s), "
                f"{misaligned_count} misaligned"
            )
    
    return {
        "signal_type": "SIG-GATE",
        "status": status,
        "conflict": False,  # Gate alignment never triggers conflict directly
        "drivers": drivers,
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,  # Gate alignment is advisory only
            "no_enforcement": True,  # No enforcement or gating
            "conflict_invariant": True,  # Conflict always false (invariant)
        },
        "weight_hint": "LOW",  # Advisory only, low weight
    }


__all__ = [
    "UPLIFT_SAFETY_GOVERNANCE_TILE_SCHEMA_VERSION",
    "StatusLight",
    "build_uplift_safety_governance_tile",
    "extract_uplift_safety_signal_for_first_light",
    "build_p3_uplift_safety_summary",
    "build_p4_uplift_safety_calibration",
    "build_first_light_uplift_gate_annex",
    "export_gate_annex_per_experiment",
    "build_gate_alignment_panel",
    "extract_uplift_gate_alignment_signal",
    "uplift_gate_alignment_for_alignment_view",
    "attach_uplift_safety_to_evidence",
    "summarize_uplift_safety_for_council",
]

