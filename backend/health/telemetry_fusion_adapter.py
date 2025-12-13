"""Telemetry fusion integration adapter for global health.

STATUS: PHASE X — TELEMETRY GOVERNANCE TILE

Provides integration between Phase VI telemetry fusion signals (fusion tile,
uplift gate, director tile v2) for the global health surface builder and
First Light integration.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The telemetry_governance tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TELEMETRY_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"
TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def build_telemetry_governance_tile(
    fusion_tile: Dict[str, Any],
    uplift_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
    telemetry_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build telemetry governance tile summarizing telemetry-driven uplift readiness.

    STATUS: PHASE X — TELEMETRY GOVERNANCE TILE

    Integrates Phase VI telemetry fusion signals (fusion tile, uplift gate,
    director tile v2) into a unified governance tile for the global health
    dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - This does NOT modify decisions

    Args:
        fusion_tile: Fusion tile from build_telemetry_topology_semantic_fusion().
            Must contain: fusion_band, fusion_risk_score, incoherence_vectors, neutral_notes
        uplift_gate: Uplift phase gate from build_telemetry_driven_uplift_phase_gate().
            Must contain: uplift_gate_status, drivers, recommended_hold_slices, headline
        director_tile_v2: Director tile v2 from build_telemetry_director_tile_v2().
            Must contain: status_light, fusion_band, uplift_gate_status, headline
        telemetry_health: Optional telemetry health summary from summarize_telemetry_for_global_health().
            If provided, should contain: telemetry_ok, status

    Returns:
        Telemetry governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED" (from director_tile_v2)
        - fusion_band: "LOW" | "MEDIUM" | "HIGH" (from fusion_tile)
        - uplift_gate_status: "OK" | "ATTENTION" | "BLOCK" (from uplift_gate)
        - telemetry_ok: bool (from telemetry_health or director_tile_v2)
        - incoherence_vectors: List[str] (from fusion_tile)
        - headline: str (from director_tile_v2 or uplift_gate)
    """
    # Extract status_light from director_tile_v2
    status_light = director_tile_v2.get("status_light", "GREEN")
    
    # Extract fusion_band from fusion_tile (prefer director_tile_v2 if available, fallback to fusion_tile)
    fusion_band = director_tile_v2.get("fusion_band") or fusion_tile.get("fusion_band", "LOW")
    
    # Extract uplift_gate_status from uplift_gate (prefer director_tile_v2 if available)
    uplift_gate_status = director_tile_v2.get("uplift_gate_status") or uplift_gate.get("uplift_gate_status", "OK")
    
    # Extract telemetry_ok from telemetry_health or director_tile_v2
    telemetry_ok = True
    if telemetry_health is not None:
        telemetry_ok = telemetry_health.get("telemetry_ok", True)
    else:
        telemetry_ok = director_tile_v2.get("telemetry_ok", True)
    
    # Extract incoherence_vectors from fusion_tile
    incoherence_vectors = fusion_tile.get("incoherence_vectors", [])
    
    # Extract headline from director_tile_v2 (prefer) or uplift_gate
    headline = director_tile_v2.get("headline") or uplift_gate.get("headline", "Telemetry governance: Status unknown")
    
    return {
        "schema_version": TELEMETRY_GOVERNANCE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "fusion_band": fusion_band,
        "uplift_gate_status": uplift_gate_status,
        "telemetry_ok": telemetry_ok,
        "incoherence_vectors": incoherence_vectors,
        "headline": headline,
    }


def extract_telemetry_signal_for_first_light(
    fusion_tile: Dict[str, Any],
    uplift_gate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract telemetry signal for First Light summary.json.

    STATUS: PHASE X — TELEMETRY GOVERNANCE TILE

    Extracts a compact telemetry signal from fusion tile and uplift gate
    for inclusion in First Light summary.json.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        fusion_tile: Fusion tile from build_telemetry_topology_semantic_fusion().
            Must contain: fusion_band, incoherence_vectors
        uplift_gate: Uplift phase gate from build_telemetry_driven_uplift_phase_gate().
            Must contain: uplift_gate_status

    Returns:
        Dictionary with:
        - fusion_band: "LOW" | "MEDIUM" | "HIGH"
        - uplift_gate_status: "OK" | "ATTENTION" | "BLOCK"
        - num_incoherence_vectors: int
    """
    fusion_band = fusion_tile.get("fusion_band", "LOW")
    uplift_gate_status = uplift_gate.get("uplift_gate_status", "OK")
    incoherence_vectors = fusion_tile.get("incoherence_vectors", [])
    num_incoherence_vectors = len(incoherence_vectors) if isinstance(incoherence_vectors, list) else 0
    
    return {
        "fusion_band": fusion_band,
        "uplift_gate_status": uplift_gate_status,
        "num_incoherence_vectors": num_incoherence_vectors,
    }


def attach_telemetry_governance_to_evidence(
    evidence: Dict[str, Any],
    governance_tile: Dict[str, Any],
    first_light_signal: Dict[str, Any],
    readiness_annex: Optional[Dict[str, Any]] = None,
    perf_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach telemetry governance to evidence pack (read-only, additive).

    STATUS: PHASE X — TELEMETRY GOVERNANCE EVIDENCE INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the telemetry
    governance information attached under evidence["governance"]["telemetry"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached data is purely observational
    - No control flow depends on the data contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        governance_tile: Telemetry governance tile from build_telemetry_governance_tile().
        first_light_signal: First Light signal from extract_telemetry_signal_for_first_light().
        readiness_annex: Optional readiness annex/tile from build_readiness_governance_tile().
        perf_tile: Optional performance tile.

    Returns:
        New dict with evidence contents plus telemetry governance attached under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_telemetry_governance_tile(fusion_tile, uplift_gate, director_tile_v2)
        >>> signal = extract_telemetry_signal_for_first_light(fusion_tile, uplift_gate)
        >>> enriched = attach_telemetry_governance_to_evidence(evidence, tile, signal)
        >>> enriched["governance"]["telemetry"]["fusion_band"]
        'LOW'
    """
    # Non-mutating: create new dict
    updated = dict(evidence)

    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])

    # Extract relevant fields for evidence
    telemetry_governance_summary = {
        "fusion_band": governance_tile.get("fusion_band", "LOW"),
        "uplift_gate_status": governance_tile.get("uplift_gate_status", "OK"),
        "num_incoherence_vectors": first_light_signal.get("num_incoherence_vectors", 0),
        "status_light": governance_tile.get("status_light", "GREEN"),
    }

    # Attach summary
    updated["governance"]["telemetry"] = telemetry_governance_summary

    # Optionally include behavior consistency cross-check
    if readiness_annex is not None or perf_tile is not None:
        behavior_consistency = summarize_telemetry_behavior_consistency(
            telemetry_tile=governance_tile,
            readiness_annex=readiness_annex,
            perf_tile=perf_tile,
        )
        updated["governance"]["telemetry"]["behavior_consistency"] = behavior_consistency

    return updated


def attach_telemetry_governance_to_p3_stability_report(
    stability_report: Dict[str, Any],
    governance_tile: Dict[str, Any],
    first_light_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach telemetry governance summary to P3 stability report.

    STATUS: PHASE X — TELEMETRY GOVERNANCE P3 INTEGRATION

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Telemetry governance is observational only

    Args:
        stability_report: P3 stability report dictionary
        governance_tile: Telemetry governance tile from build_telemetry_governance_tile()
        first_light_signal: First Light signal from extract_telemetry_signal_for_first_light()

    Returns:
        Updated stability report with telemetry_governance field
    """
    # Extract telemetry governance data
    telemetry_governance_summary = {
        "fusion_band": governance_tile.get("fusion_band", "LOW"),
        "uplift_gate_status": governance_tile.get("uplift_gate_status", "OK"),
        "num_incoherence_vectors": first_light_signal.get("num_incoherence_vectors", 0),
        "status_light": governance_tile.get("status_light", "GREEN"),
    }

    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["telemetry_governance"] = telemetry_governance_summary

    return updated_report


def attach_telemetry_governance_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    governance_tile: Dict[str, Any],
    first_light_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach telemetry governance calibration to P4 calibration report.

    STATUS: PHASE X — TELEMETRY GOVERNANCE P4 INTEGRATION

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Telemetry governance is observational only

    Args:
        calibration_report: P4 calibration report dictionary
        governance_tile: Telemetry governance tile from build_telemetry_governance_tile()
        first_light_signal: First Light signal from extract_telemetry_signal_for_first_light()

    Returns:
        Updated calibration report with telemetry_governance field
    """
    # Extract telemetry governance data
    telemetry_governance_calibration = {
        "fusion_band": governance_tile.get("fusion_band", "LOW"),
        "uplift_gate_status": governance_tile.get("uplift_gate_status", "OK"),
        "num_incoherence_vectors": first_light_signal.get("num_incoherence_vectors", 0),
        "status_light": governance_tile.get("status_light", "GREEN"),
    }

    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["telemetry_governance"] = telemetry_governance_calibration

    return updated_report


def summarize_telemetry_behavior_consistency(
    telemetry_tile: Dict[str, Any],
    readiness_annex: Optional[Dict[str, Any]] = None,
    perf_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize telemetry-behavior consistency cross-check.

    STATUS: PHASE X — TELEMETRY × BEHAVIOR CROSS-CHECK

    Flags when telemetry governance issues co-occur with metric readiness /
    performance regressions, or when telemetry shows issues while behavior
    metrics appear healthy (potential inconsistency).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational and advisory
    - No control flow depends on the summary contents
    - Advisory only; no gating

    Args:
        telemetry_tile: Telemetry governance tile from build_telemetry_governance_tile().
            Must contain: status_light
        readiness_annex: Optional readiness annex/tile from build_readiness_governance_tile().
            If provided, should contain: status_light
        perf_tile: Optional performance tile (structure similar to readiness).
            If provided, should contain: status_light

    Returns:
        Dictionary with:
        - consistency_status: "CONSISTENT" | "INCONSISTENT" | "PARTIAL"
        - advisory_notes: List[str] (neutral, descriptive notes)
        - telemetry_status: "GREEN" | "YELLOW" | "RED"
        - readiness_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
        - perf_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
    """
    telemetry_status = telemetry_tile.get("status_light", "GREEN")
    readiness_status = readiness_annex.get("status_light", "UNKNOWN") if readiness_annex else "UNKNOWN"
    perf_status = perf_tile.get("status_light", "UNKNOWN") if perf_tile else "UNKNOWN"

    advisory_notes = []
    consistency_status = "CONSISTENT"

    # Check for inconsistencies: telemetry RED/YELLOW while behavior metrics are GREEN
    if telemetry_status in ("RED", "YELLOW"):
        # Prioritize the "both" case when both readiness and perf are provided and both are GREEN
        if readiness_status == "GREEN" and perf_status == "GREEN" and readiness_annex and perf_tile:
            advisory_notes.append(
                f"Telemetry status is {telemetry_status} while both readiness and performance statuses are GREEN. "
                "This may indicate telemetry anomalies that do not align with behavior metrics."
            )
            consistency_status = "INCONSISTENT"
        elif readiness_status == "GREEN" and readiness_annex is not None:
            advisory_notes.append(
                f"Telemetry status is {telemetry_status} while readiness status is GREEN. "
                "This may indicate telemetry anomalies that do not align with metric readiness signals."
            )
            consistency_status = "INCONSISTENT"
        elif perf_status == "GREEN" and perf_tile is not None:
            advisory_notes.append(
                f"Telemetry status is {telemetry_status} while performance status is GREEN. "
                "This may indicate telemetry anomalies that do not align with performance signals."
            )
            if consistency_status == "CONSISTENT":
                consistency_status = "INCONSISTENT"
            else:
                consistency_status = "PARTIAL"

    # Check for alignment: all systems show issues
    if telemetry_status in ("RED", "YELLOW") and readiness_status in ("RED", "YELLOW"):
        advisory_notes.append(
            f"Telemetry status ({telemetry_status}) aligns with readiness status ({readiness_status}). "
            "Both systems indicate potential issues."
        )
        if consistency_status == "CONSISTENT":
            consistency_status = "PARTIAL"

    if telemetry_status in ("RED", "YELLOW") and perf_status in ("RED", "YELLOW"):
        advisory_notes.append(
            f"Telemetry status ({telemetry_status}) aligns with performance status ({perf_status}). "
            "Both systems indicate potential issues."
        )
        if consistency_status == "CONSISTENT":
            consistency_status = "PARTIAL"

    # All systems healthy
    if telemetry_status == "GREEN" and readiness_status == "GREEN" and perf_status == "GREEN":
        if readiness_annex and perf_tile:
            advisory_notes.append(
                "Telemetry, readiness, and performance statuses are all GREEN. "
                "Systems appear aligned and healthy."
            )

    # If no notes generated, add default
    if not advisory_notes:
        advisory_notes.append(
            f"Telemetry status: {telemetry_status}. "
            f"Readiness status: {readiness_status}. "
            f"Performance status: {perf_status}."
        )

    return {
        "consistency_status": consistency_status,
        "advisory_notes": advisory_notes,
        "telemetry_status": telemetry_status,
        "readiness_status": readiness_status,
        "perf_status": perf_status,
    }


def summarize_telemetry_for_uplift_council(tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize telemetry governance tile for uplift council decision-making.

    STATUS: PHASE X — TELEMETRY GOVERNANCE COUNCIL SUMMARY

    Maps telemetry governance signals to council decision status:
    - uplift_gate_status="BLOCK" → BLOCK
    - uplift_gate_status="ATTENTION" or fusion_band="HIGH" → WARN
    - otherwise → OK

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        tile: Telemetry governance tile from build_telemetry_governance_tile().
            Must contain: uplift_gate_status, fusion_band, incoherence_vectors

    Returns:
        Dictionary with:
        - status: "OK" | "WARN" | "BLOCK"
        - num_incoherence_vectors: int
        - fusion_band: "LOW" | "MEDIUM" | "HIGH"
        - uplift_gate_status: "OK" | "ATTENTION" | "BLOCK"
    """
    uplift_gate_status = tile.get("uplift_gate_status", "OK")
    fusion_band = tile.get("fusion_band", "LOW")
    incoherence_vectors = tile.get("incoherence_vectors", [])
    num_incoherence_vectors = len(incoherence_vectors) if isinstance(incoherence_vectors, list) else 0

    # Map to council status
    if uplift_gate_status == "BLOCK":
        status = "BLOCK"
    elif uplift_gate_status == "ATTENTION" or fusion_band == "HIGH":
        status = "WARN"
    else:
        status = "OK"

    return {
        "status": status,
        "num_incoherence_vectors": num_incoherence_vectors,
        "fusion_band": fusion_band,
        "uplift_gate_status": uplift_gate_status,
    }


def emit_cal_exp_telemetry_behavior_consistency(
    cal_id: str,
    consistency: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Emit telemetry-behavior consistency snapshot for a calibration experiment.

    STATUS: PHASE X — TELEMETRY × BEHAVIOR CAL-EXP CONSISTENCY

    Builds a per-experiment consistency snapshot and optionally persists it to disk.
    This snapshot captures the consistency status between telemetry governance
    and behavior metrics (readiness/performance) for a single calibration experiment.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from optional file write)
    - The snapshot is purely observational
    - No control flow depends on the snapshot contents
    - Advisory only; no gating

    Args:
        cal_id: Calibration experiment identifier (e.g., "cal_exp1", "CAL-EXP-1")
        consistency: Consistency summary from summarize_telemetry_behavior_consistency()
        output_dir: Optional directory to persist snapshot (e.g., Path("calibration/"))
            If provided, writes to calibration/telemetry_behavior_consistency_<cal_id>.json

    Returns:
        Snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - consistency_status: "CONSISTENT" | "INCONSISTENT" | "PARTIAL"
        - telemetry_status: "GREEN" | "YELLOW" | "RED"
        - readiness_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
        - perf_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
    """
    snapshot = {
        "schema_version": TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION,
        "cal_id": cal_id,
        "consistency_status": consistency.get("consistency_status", "CONSISTENT"),
        "telemetry_status": consistency.get("telemetry_status", "GREEN"),
        "readiness_status": consistency.get("readiness_status", "UNKNOWN"),
        "perf_status": consistency.get("perf_status", "UNKNOWN"),
    }

    # Optionally persist to disk
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"telemetry_behavior_consistency_{cal_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)

    return snapshot


def _map_inconsistency_to_reason_code(
    telemetry_status: str,
    readiness_status: str,
    perf_status: str,
) -> Optional[str]:
    """
    Map telemetry-behavior status combination to deterministic reason code.

    STATUS: PHASE X — REASON CODE MAPPING

    Deterministically maps status combinations to reason codes for automation
    and categorization. Returns None if no inconsistency pattern matches.

    SHADOW MODE CONTRACT:
    - Pure function, no side effects
    - Deterministic mapping
    - Advisory only

    Args:
        telemetry_status: "GREEN" | "YELLOW" | "RED"
        readiness_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
        perf_status: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"

    Returns:
        Reason code string or None if no inconsistency pattern matches.
        Codes:
        - TEL_WARN_BEHAV_OK: Telemetry YELLOW, behavior OK (both readiness and perf GREEN)
        - TEL_RED_BEHAV_OK: Telemetry RED, behavior OK (both readiness and perf GREEN)
        - TEL_OK_BEHAV_WARN: Telemetry GREEN, behavior WARN (readiness or perf YELLOW)
        - TEL_OK_BEHAV_RED: Telemetry GREEN, behavior RED (readiness or perf RED)
        - TEL_WARN_PERF_OK: Telemetry YELLOW, performance OK (perf GREEN, readiness may vary)
        - TEL_WARN_READINESS_OK: Telemetry YELLOW, readiness OK (readiness GREEN, perf may vary)
    """
    # Normalize UNKNOWN to GREEN for comparison (UNKNOWN is treated as OK)
    readiness_ok = readiness_status in ("GREEN", "UNKNOWN")
    perf_ok = perf_status in ("GREEN", "UNKNOWN")
    readiness_warn = readiness_status == "YELLOW"
    readiness_red = readiness_status == "RED"
    perf_warn = perf_status == "YELLOW"
    perf_red = perf_status == "RED"
    
    # Telemetry YELLOW cases
    if telemetry_status == "YELLOW":
        if readiness_status == "GREEN" and perf_status == "GREEN":
            return "TEL_WARN_BEHAV_OK"
        elif readiness_status == "GREEN":
            return "TEL_WARN_READINESS_OK"
        elif perf_status == "GREEN":
            return "TEL_WARN_PERF_OK"
    
    # Telemetry RED cases
    if telemetry_status == "RED":
        if readiness_ok and perf_ok:
            return "TEL_RED_BEHAV_OK"
    
    # Telemetry GREEN cases (telemetry OK, behavior shows issues)
    if telemetry_status == "GREEN":
        if readiness_red or perf_red:
            return "TEL_OK_BEHAV_RED"
        elif readiness_warn or perf_warn:
            return "TEL_OK_BEHAV_WARN"
    
    return None


def build_consistency_matrix(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build consistency matrix aggregating telemetry-behavior consistency across CAL-EXP runs.

    STATUS: PHASE X — TELEMETRY × BEHAVIOR CONSISTENCY MATRIX

    Aggregates per-experiment consistency snapshots to provide a cross-experiment
    view of telemetry-behavior alignment. Answers: "Do telemetry anomalies agree
    with behavioral issues across CAL-EXP-1/2/3?"

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The matrix is purely observational
    - No control flow depends on the matrix contents
    - Advisory only; no gating

    Args:
        snapshots: List of consistency snapshots from emit_cal_exp_telemetry_behavior_consistency()

    Returns:
        Consistency matrix dictionary with:
        - schema_version: "1.0.0"
        - total_experiments: int
        - consistency_counts: {"CONSISTENT": int, "INCONSISTENT": int, "PARTIAL": int}
        - inconsistent_experiments: List[Dict] with cal_id, reason, and reason_code
        - signals: Dict with counts, top_inconsistent_cal_ids, and reason_code_histogram
        - summary: str (neutral summary)
    """
    if not snapshots:
        return {
            "schema_version": TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION,
            "total_experiments": 0,
            "consistency_counts": {
                "CONSISTENT": 0,
                "INCONSISTENT": 0,
                "PARTIAL": 0,
            },
            "inconsistent_experiments": [],
            "signals": {
                "counts": {
                    "CONSISTENT": 0,
                    "INCONSISTENT": 0,
                    "PARTIAL": 0,
                },
                "top_inconsistent_cal_ids": [],
                "reason_code_histogram": {},
            },
            "summary": "No consistency snapshots provided for analysis.",
        }

    # Count consistency statuses
    consistency_counts = {
        "CONSISTENT": 0,
        "INCONSISTENT": 0,
        "PARTIAL": 0,
    }

    inconsistent_experiments = []
    reason_code_histogram: Dict[str, int] = {}

    for snapshot in snapshots:
        consistency_status = snapshot.get("consistency_status", "CONSISTENT")
        cal_id = snapshot.get("cal_id", "unknown")
        telemetry_status = snapshot.get("telemetry_status", "GREEN")
        readiness_status = snapshot.get("readiness_status", "UNKNOWN")
        perf_status = snapshot.get("perf_status", "UNKNOWN")

        # Count status
        if consistency_status in consistency_counts:
            consistency_counts[consistency_status] += 1

        # Collect inconsistent experiments with brief reasons and reason codes
        if consistency_status == "INCONSISTENT":
            reason_parts = []
            if telemetry_status in ("RED", "YELLOW") and readiness_status == "GREEN":
                reason_parts.append(f"telemetry {telemetry_status} vs readiness GREEN")
            if telemetry_status in ("RED", "YELLOW") and perf_status == "GREEN":
                reason_parts.append(f"telemetry {telemetry_status} vs perf GREEN")
            
            reason = "; ".join(reason_parts) if reason_parts else "inconsistency detected"
            
            # Map to deterministic reason code
            reason_code = _map_inconsistency_to_reason_code(
                telemetry_status, readiness_status, perf_status
            )
            
            # Update histogram
            if reason_code:
                reason_code_histogram[reason_code] = reason_code_histogram.get(reason_code, 0) + 1
            
            inconsistent_experiments.append({
                "cal_id": cal_id,
                "reason": reason,
                "reason_code": reason_code,
            })

    # Build summary
    total = len(snapshots)
    consistent_count = consistency_counts["CONSISTENT"]
    inconsistent_count = consistency_counts["INCONSISTENT"]
    partial_count = consistency_counts["PARTIAL"]

    if inconsistent_count == 0:
        summary = (
            f"All {total} experiment(s) show consistent telemetry-behavior alignment. "
            f"{consistent_count} fully consistent, {partial_count} partially consistent."
        )
    elif inconsistent_count == total:
        summary = (
            f"All {total} experiment(s) show inconsistent telemetry-behavior alignment. "
            "Telemetry anomalies do not align with behavioral issues across experiments."
        )
    else:
        summary = (
            f"Across {total} experiment(s): {consistent_count} consistent, "
            f"{inconsistent_count} inconsistent, {partial_count} partial. "
            "Mixed alignment pattern detected."
        )

    # Build signals section
    # Get top 3 inconsistent cal_ids (sorted by cal_id for determinism)
    top_inconsistent_cal_ids = sorted(
        [exp["cal_id"] for exp in inconsistent_experiments],
        key=lambda x: (len(x), x)  # Sort by length then lexicographically for determinism
    )[:3]
    
    signals = {
        "counts": consistency_counts.copy(),
        "top_inconsistent_cal_ids": top_inconsistent_cal_ids,
        "reason_code_histogram": dict(sorted(reason_code_histogram.items())),  # Sorted for determinism
    }
    
    return {
        "schema_version": TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION,
        "total_experiments": total,
        "consistency_counts": consistency_counts,
        "inconsistent_experiments": inconsistent_experiments,
        "signals": signals,
        "summary": summary,
    }


def extract_telemetry_behavior_panel_signal(
    matrix: Dict[str, Any],
    extraction_source: str = "MISSING",
) -> Dict[str, Any]:
    """
    Extract compact status signal from telemetry-behavior consistency matrix.

    STATUS: PHASE X — TELEMETRY × BEHAVIOR STATUS SIGNAL EXTRACTION

    Extracts a compact signal from the consistency matrix for inclusion in
    first_light_status.json. Provides deterministic top-ids and histogram
    for automation and categorization.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The extracted signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only; no gating

    Args:
        matrix: Consistency matrix from build_consistency_matrix()
        extraction_source: Source of the matrix ("MANIFEST" | "EVIDENCE_JSON" | "MISSING")

    Returns:
        Compact signal dictionary with:
        - schema_version: "1.0.0"
        - mode: "SHADOW"
        - extraction_source: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"
        - consistency_counts: {"CONSISTENT": int, "INCONSISTENT": int, "PARTIAL": int}
        - top_inconsistent_cal_ids: List[str] (top 3, sorted deterministically)
        - reason_code_histogram: Dict[str, int] (sorted for determinism)
    """
    signals = matrix.get("signals", {})
    consistency_counts = signals.get("counts", {
        "CONSISTENT": 0,
        "INCONSISTENT": 0,
        "PARTIAL": 0,
    })
    top_inconsistent_cal_ids = signals.get("top_inconsistent_cal_ids", [])
    reason_code_histogram = signals.get("reason_code_histogram", {})

    # Normalize extraction_source to canonical values
    if extraction_source not in ("MANIFEST", "EVIDENCE_JSON", "MISSING"):
        extraction_source = "MISSING"

    return {
        "schema_version": TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION,
        "mode": "SHADOW",
        "extraction_source": extraction_source,
        "consistency_counts": consistency_counts.copy(),
        "top_inconsistent_cal_ids": top_inconsistent_cal_ids.copy(),
        "reason_code_histogram": dict(sorted(reason_code_histogram.items())),  # Sorted for determinism
    }


def attach_consistency_matrix_to_evidence(
    evidence: Dict[str, Any],
    matrix: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach telemetry-behavior consistency matrix to evidence pack.

    STATUS: PHASE X — TELEMETRY × BEHAVIOR EVIDENCE INTEGRATION

    Attaches the consistency matrix under evidence["governance"]["telemetry_behavior_panel"]
    for inclusion in evidence packs.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached matrix is purely observational
    - No control flow depends on the matrix contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        matrix: Consistency matrix from build_consistency_matrix().

    Returns:
        New dict with evidence contents plus consistency matrix attached under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01"}
        >>> snapshots = [snapshot1, snapshot2, snapshot3]
        >>> matrix = build_consistency_matrix(snapshots)
        >>> enriched = attach_consistency_matrix_to_evidence(evidence, matrix)
        >>> enriched["governance"]["telemetry_behavior_panel"]["total_experiments"]
        3
    """
    # Non-mutating: create new dict
    updated = dict(evidence)

    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])

    # Attach consistency matrix
    updated["governance"]["telemetry_behavior_panel"] = matrix

    return updated


__all__ = [
    "TELEMETRY_GOVERNANCE_TILE_SCHEMA_VERSION",
    "TELEMETRY_BEHAVIOR_CONSISTENCY_SNAPSHOT_SCHEMA_VERSION",
    "build_telemetry_governance_tile",
    "extract_telemetry_signal_for_first_light",
    "attach_telemetry_governance_to_evidence",
    "attach_telemetry_governance_to_p3_stability_report",
    "attach_telemetry_governance_to_p4_calibration_report",
    "summarize_telemetry_behavior_consistency",
    "summarize_telemetry_for_uplift_council",
    "emit_cal_exp_telemetry_behavior_consistency",
    "_map_inconsistency_to_reason_code",
    "build_consistency_matrix",
    "extract_telemetry_behavior_panel_signal",
    "attach_consistency_matrix_to_evidence",
]

