"""Semantic Integrity Grid adapter for global health.

STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

Provides integration between semantic integrity signals (invariants, uplift gate, director tile)
and the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The semantic_integrity tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION = "1.0.0"
SEMANTIC_FOOTPRINT_SCHEMA_VERSION = "1.0.0"
SEMANTIC_SAFETY_PANEL_SCHEMA_VERSION = "1.0.0"


def _validate_invariant_check(invariant_check: Dict[str, Any]) -> None:
    """Validate invariant check structure.
    
    Args:
        invariant_check: Invariant check dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["invariant_status", "broken_invariants", "terms_involved", "neutral_notes"]
    missing = [key for key in required_keys if key not in invariant_check]
    if missing:
        raise ValueError(
            f"invariant_check missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(invariant_check.keys()))}"
        )


def _validate_uplift_preview(uplift_preview: Dict[str, Any]) -> None:
    """Validate uplift preview structure.
    
    Args:
        uplift_preview: Uplift preview dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["uplift_semantic_status", "rationale", "preview_effect_on_curriculum"]
    missing = [key for key in required_keys if key not in uplift_preview]
    if missing:
        raise ValueError(
            f"uplift_preview missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(uplift_preview.keys()))}"
        )


def _validate_director_tile(director_tile: Dict[str, Any]) -> None:
    """Validate director tile structure.
    
    Args:
        director_tile: Director tile dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light", "semantic_uplift_status", "top_risk_terms", "headline"]
    missing = [key for key in required_keys if key not in director_tile]
    if missing:
        raise ValueError(
            f"director_tile missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(director_tile.keys()))}"
        )


def build_semantic_integrity_tile(
    invariant_check: Dict[str, Any],
    uplift_preview: Dict[str, Any],
    director_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic integrity tile for global health surface.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Integrates semantic integrity signals (invariants, uplift gate, director panel)
    to produce a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents

    Args:
        invariant_check: From check_semantic_invariants().
            Must contain: invariant_status, broken_invariants, terms_involved, neutral_notes
        uplift_preview: From preview_semantic_uplift_gate().
            Must contain: uplift_semantic_status, rationale, preview_effect_on_curriculum
        director_tile: From build_semantic_uplift_director_tile().
            Must contain: status_light, semantic_uplift_status, top_risk_terms, headline

    Returns:
        Semantic integrity tile dictionary with:
        - schema_version: "1.0.0"
        - tile_type: "semantic_integrity"
        - status_light: GREEN | YELLOW | RED (from director_tile)
        - invariants_ok: bool (True if invariant_status == "OK")
        - uplift_semantic_status: OK | WARN | BLOCK (from director_tile)
        - broken_invariants: List of broken invariant details
        - top_risk_terms: List[str] (from director_tile)
        - headline: str (from director_tile)
        - notes: List[str] (aggregated neutral notes)
    """
    # Validate inputs
    _validate_invariant_check(invariant_check)
    _validate_uplift_preview(uplift_preview)
    _validate_director_tile(director_tile)
    
    # Extract key values
    status_light = director_tile.get("status_light", "GREEN")
    invariant_status = invariant_check.get("invariant_status", "OK")
    uplift_semantic_status = director_tile.get("semantic_uplift_status", "OK")
    broken_invariants = invariant_check.get("broken_invariants", [])
    top_risk_terms = director_tile.get("top_risk_terms", [])
    headline = director_tile.get("headline", "")
    
    # Determine invariants_ok
    invariants_ok = invariant_status == "OK"
    
    # Aggregate neutral notes
    notes = []
    notes.extend(invariant_check.get("neutral_notes", []))
    notes.extend(uplift_preview.get("rationale", []))
    notes.extend(uplift_preview.get("preview_effect_on_curriculum", []))
    
    # Build tile
    tile = {
        "schema_version": SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION,
        "tile_type": "semantic_integrity",
        "status_light": status_light,
        "invariants_ok": invariants_ok,
        "uplift_semantic_status": uplift_semantic_status,
        "broken_invariants": broken_invariants,
        "top_risk_terms": top_risk_terms,
        "headline": headline,
        "notes": notes,
    }
    
    return tile


def extract_semantic_drift_signal(
    invariant_check: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract semantic drift signal for P4 compatibility hook.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    This function feeds USLAIntegration → Shadow Runner P4 for real-time drift signals.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        invariant_check: From check_semantic_invariants().
            Must contain: invariant_status, broken_invariants, terms_involved

    Returns:
        Semantic drift signal dictionary with:
        - drift_detected: bool
        - severity: OK | ATTENTION | BROKEN
        - broken_invariant_count: int
        - terms_involved: List[str]
        - critical_signals: List[str] (invariant types with BROKEN severity)
    """
    # Validate input
    _validate_invariant_check(invariant_check)
    
    # Extract values
    invariant_status = invariant_check.get("invariant_status", "OK")
    broken_invariants = invariant_check.get("broken_invariants", [])
    terms_involved = invariant_check.get("terms_involved", [])
    
    # Determine drift detection
    drift_detected = invariant_status != "OK"
    
    # Extract critical signals (BROKEN severity)
    critical_signals = [
        inv.get("invariant_type", "unknown")
        for inv in broken_invariants
        if inv.get("severity") == "BROKEN"
    ]
    
    # Build signal
    signal = {
        "drift_detected": drift_detected,
        "severity": invariant_status,
        "broken_invariant_count": len(broken_invariants),
        "terms_involved": terms_involved,
        "critical_signals": critical_signals,
    }
    
    return signal


def build_semantic_integrity_summary_for_p3(
    tile: Dict[str, Any],
    invariant_check: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic integrity summary for P3 stability report.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Extracts key semantic integrity metrics for inclusion in First-Light P3 stability reports.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        tile: From build_semantic_integrity_tile().
        invariant_check: From check_semantic_invariants().

    Returns:
        Semantic integrity summary dictionary with:
        - invariants_ok: bool
        - broken_invariant_count: int
        - uplift_semantic_status: OK | WARN | BLOCK
        - top_risk_terms: List[str]
        - headline: str
    """
    # Extract from tile
    invariants_ok = tile.get("invariants_ok", False)
    uplift_semantic_status = tile.get("uplift_semantic_status", "OK")
    top_risk_terms = tile.get("top_risk_terms", [])
    headline = tile.get("headline", "")
    
    # Extract broken invariant count from invariant_check
    broken_invariants = invariant_check.get("broken_invariants", [])
    broken_invariant_count = len(broken_invariants)
    
    return {
        "invariants_ok": invariants_ok,
        "broken_invariant_count": broken_invariant_count,
        "uplift_semantic_status": uplift_semantic_status,
        "top_risk_terms": top_risk_terms,
        "headline": headline,
    }


def build_semantic_integrity_calibration_for_p4(
    drift_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic integrity calibration for P4 calibration report.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Extracts semantic drift signal for inclusion in P4 calibration reports.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned calibration is purely observational
    - No control flow depends on the calibration contents

    Args:
        drift_signal: From extract_semantic_drift_signal().

    Returns:
        Semantic integrity calibration dictionary with:
        - drift_detected: bool
        - severity: OK | ATTENTION | BROKEN
        - terms_involved: List[str]
        - critical_signals: List[str]
    """
    return {
        "drift_detected": drift_signal.get("drift_detected", False),
        "severity": drift_signal.get("severity", "OK"),
        "terms_involved": drift_signal.get("terms_involved", []),
        "critical_signals": drift_signal.get("critical_signals", []),
    }


def attach_semantic_integrity_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    drift_signal: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach semantic integrity to evidence pack.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Stores semantic integrity data under evidence["governance"]["semantic_integrity"].
    If p3_summary and p4_calibration are provided, also includes first_light_footprint.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - Returns a new dict, does not mutate input evidence
    - The attached data is purely observational
    - No control flow depends on the evidence contents
    - Advisory only; council still decides; no new gate semantics

    Args:
        evidence: Existing evidence pack dictionary (read-only, not modified).
        tile: From build_semantic_integrity_tile().
        drift_signal: From extract_semantic_drift_signal().
        p3_summary: Optional P3 summary from build_semantic_integrity_summary_for_p3().
        p4_calibration: Optional P4 calibration from build_semantic_integrity_calibration_for_p4().

    Returns:
        New dict with evidence contents plus semantic_integrity attached under governance.
    """
    # Create a copy to avoid mutation
    result = dict(evidence)
    
    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}
    
    # Extract key fields
    invariants_ok = tile.get("invariants_ok", False)
    broken_invariants = tile.get("broken_invariants", [])
    broken_count = len(broken_invariants)
    severity = drift_signal.get("severity", "OK")
    terms_involved = drift_signal.get("terms_involved", [])
    critical_signals = drift_signal.get("critical_signals", [])
    
    # Attach semantic integrity data
    result["governance"]["semantic_integrity"] = {
        "invariants_ok": invariants_ok,
        "broken_invariant_count": broken_count,
        "severity": severity,
        "terms_involved": terms_involved,
        "critical_signals": critical_signals,
    }
    
    # Add first_light_footprint if both P3 and P4 data provided
    if p3_summary is not None and p4_calibration is not None:
        result["governance"]["semantic_integrity"]["first_light_footprint"] = (
            build_first_light_semantic_footprint(p3_summary, p4_calibration)
        )
    
    return result


def build_first_light_semantic_footprint(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic footprint for First Light experiments.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Produces a single, human-readable semantic footprint record summarizing
    invariants and uplift status from both P3 and P4 perspectives.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned footprint is purely observational
    - No control flow depends on the footprint contents
    - Advisory only; council still decides; no new gate semantics

    Args:
        p3_summary: From build_semantic_integrity_summary_for_p3().
            Must contain: invariants_ok, broken_invariant_count, uplift_semantic_status
        p4_calibration: From build_semantic_integrity_calibration_for_p4().
            Must contain: severity, terms_involved

    Returns:
        Semantic footprint dictionary with:
        - schema_version: "1.0.0"
        - invariants_ok: bool
        - broken_invariant_count: int
        - p3_uplift_semantic_status: OK | WARN | BLOCK
        - p4_severity: OK | ATTENTION | BROKEN
        - terms_involved: List[str] (limited to top 5)
    """
    return {
        "schema_version": "1.0.0",
        "invariants_ok": p3_summary.get("invariants_ok", False),
        "broken_invariant_count": p3_summary.get("broken_invariant_count", 0),
        "p3_uplift_semantic_status": p3_summary.get("uplift_semantic_status", "OK"),
        "p4_severity": p4_calibration.get("severity", "OK"),
        "terms_involved": p4_calibration.get("terms_involved", [])[:5],
    }


def emit_cal_exp_semantic_footprint(
    cal_id: str,
    footprint: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Emit calibration experiment semantic footprint record.

    STATUS: PHASE X — P5 CALIBRATION EXPERIMENT SEMANTIC FOOTPRINT

    Creates a per-experiment semantic footprint record with cal_id for persistence
    and cross-experiment aggregation.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned footprint is purely observational
    - No control flow depends on the footprint contents
    - Advisory only; council still decides; no new gate semantics

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3").
        footprint: Semantic footprint from build_first_light_semantic_footprint().
            Must contain: schema_version, invariants_ok, broken_invariant_count,
            p3_uplift_semantic_status, p4_severity, terms_involved

    Returns:
        Calibration experiment footprint dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - p3_status: OK | WARN | BLOCK (from p3_uplift_semantic_status)
        - p4_status: OK | ATTENTION | BROKEN (from p4_severity)
        - broken_invariant_count: int
    """
    # Extract status values
    p3_status = footprint.get("p3_uplift_semantic_status", "OK")
    p4_status = footprint.get("p4_severity", "OK")
    broken_invariant_count = footprint.get("broken_invariant_count", 0)

    return {
        "schema_version": SEMANTIC_FOOTPRINT_SCHEMA_VERSION,
        "cal_id": cal_id,
        "p3_status": p3_status,
        "p4_status": p4_status,
        "broken_invariant_count": broken_invariant_count,
    }


def persist_cal_exp_semantic_footprint(
    footprint: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist calibration experiment semantic footprint to disk.

    STATUS: PHASE X — P5 CALIBRATION EXPERIMENT SEMANTIC FOOTPRINT PERSISTENCE

    Writes footprint to calibration/semantic_footprint_<cal_id>.json.
    Creates the output directory if it doesn't exist.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions
    - Non-mutating: does not modify input footprint

    Args:
        footprint: Calibration experiment footprint from emit_cal_exp_semantic_footprint().
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration")).

    Returns:
        Path to the written footprint file.

    Raises:
        IOError: If the file cannot be written.
    """
    cal_id = footprint.get("cal_id", "UNKNOWN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"semantic_footprint_{cal_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(footprint, f, indent=2, sort_keys=True)

    return output_path


def build_semantic_safety_panel(
    footprints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build semantic safety panel from multiple calibration experiment footprints.

    STATUS: PHASE X — P5 SEMANTIC SAFETY PANEL

    Aggregates semantic footprints across calibration experiments into a 2x2 safety grid
    showing the distribution of P3 vs P4 status combinations.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned panel is purely observational
    - No control flow depends on the panel contents
    - Advisory only; council still decides; no new gate semantics

    Args:
        footprints: List of calibration experiment footprints from emit_cal_exp_semantic_footprint().
            Each footprint must contain: cal_id, p3_status, p4_status, broken_invariant_count

    Returns:
        Semantic safety panel dictionary with:
        - schema_version: "1.0.0"
        - total_experiments: int
        - grid_counts:
            - ok_ok: int (P3 OK × P4 OK)
            - ok_not_ok: int (P3 OK × P4 Not-OK)
            - not_ok_ok: int (P3 Not-OK × P4 OK)
            - not_ok_not_ok: int (P3 Not-OK × P4 Not-OK)
        - top_drivers: List[str] (up to 3 cal_ids ranked by worst bucket, tie-broken by broken_invariant_count then cal_id)
        - experiments: List[Dict] (one per footprint with cal_id and grid_bucket)
    """
    # Initialize grid counts
    ok_ok = 0
    ok_not_ok = 0
    not_ok_ok = 0
    not_ok_not_ok = 0

    experiments = []

    for footprint in footprints:
        cal_id = footprint.get("cal_id", "UNKNOWN")
        p3_status = footprint.get("p3_status", "OK")
        p4_status = footprint.get("p4_status", "OK")
        broken_invariant_count = footprint.get("broken_invariant_count", 0)

        # Classify into grid bucket
        p3_ok = p3_status == "OK"
        p4_ok = p4_status == "OK"

        if p3_ok and p4_ok:
            grid_bucket = "OK×OK"
            ok_ok += 1
        elif p3_ok and not p4_ok:
            grid_bucket = "OK×Not-OK"
            ok_not_ok += 1
        elif not p3_ok and p4_ok:
            grid_bucket = "Not-OK×OK"
            not_ok_ok += 1
        else:
            grid_bucket = "Not-OK×Not-OK"
            not_ok_not_ok += 1

        experiments.append({
            "cal_id": cal_id,
            "p3_status": p3_status,
            "p4_status": p4_status,
            "broken_invariant_count": broken_invariant_count,
            "grid_bucket": grid_bucket,
        })

    # Compute top drivers: up to 3 cal_ids ranked by worst bucket
    # Ranking: not_ok_not_ok > ok_not_ok/not_ok_ok > ok_ok
    # Tie-break: higher broken_invariant_count, then cal_id (alphabetical)
    def _rank_key(exp: Dict[str, Any]) -> tuple:
        """Return ranking key for sorting experiments by severity."""
        grid_bucket = exp.get("grid_bucket", "OK×OK")
        broken_count = exp.get("broken_invariant_count", 0)
        cal_id = exp.get("cal_id", "")
        
        # Bucket priority: not_ok_not_ok (0) > ok_not_ok/not_ok_ok (1) > ok_ok (2)
        if grid_bucket == "Not-OK×Not-OK":
            bucket_priority = 0
        elif grid_bucket in ("OK×Not-OK", "Not-OK×OK"):
            bucket_priority = 1
        else:  # OK×OK
            bucket_priority = 2
        
        # Return tuple for sorting: (bucket_priority, -broken_count, cal_id)
        # Negative broken_count for descending order (higher count = worse)
        # cal_id for alphabetical tie-break
        return (bucket_priority, -broken_count, cal_id)

    # Sort experiments by severity (worst first)
    sorted_experiments = sorted(experiments, key=_rank_key)
    
    # Extract top 3 cal_ids
    top_drivers = [exp["cal_id"] for exp in sorted_experiments[:3]]

    return {
        "schema_version": SEMANTIC_SAFETY_PANEL_SCHEMA_VERSION,
        "total_experiments": len(footprints),
        "grid_counts": {
            "ok_ok": ok_ok,
            "ok_not_ok": ok_not_ok,
            "not_ok_ok": not_ok_ok,
            "not_ok_not_ok": not_ok_not_ok,
        },
        "top_drivers": top_drivers,
        "experiments": experiments,
    }


def extract_semantic_safety_panel_signal(
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact semantic safety panel signal for status integration.

    STATUS: PHASE X — P5 SEMANTIC SAFETY PANEL STATUS EXTRACTION

    This function extracts a minimal signal from the semantic safety panel
    for inclusion in First Light status files.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only; neutral phrasing

    Args:
        panel: Semantic safety panel from build_semantic_safety_panel().

    Returns:
        Compact signal dictionary with:
        - ok_ok: int
        - ok_not_ok: int
        - not_ok_ok: int
        - not_ok_not_ok: int
        - top_drivers: List[str] (up to 3 cal_ids)
    """
    grid_counts = panel.get("grid_counts", {})
    
    return {
        "ok_ok": grid_counts.get("ok_ok", 0),
        "ok_not_ok": grid_counts.get("ok_not_ok", 0),
        "not_ok_ok": grid_counts.get("not_ok_ok", 0),
        "not_ok_not_ok": grid_counts.get("not_ok_not_ok", 0),
        "top_drivers": panel.get("top_drivers", []),
    }


# Reason code constants for semantic safety panel drivers
SEM_DRV_001 = "SEM-DRV-001"  # Top driver (first cal_id)
SEM_DRV_002 = "SEM-DRV-002"  # Second driver (second cal_id)
SEM_DRV_003 = "SEM-DRV-003"  # Third driver (third cal_id)


def semantic_safety_panel_for_alignment_view(
    panel_or_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert semantic safety panel to GGFL alignment view format.

    PHASE X — GGFL ADAPTER FOR SEMANTIC SAFETY PANEL

    Normalizes the semantic safety panel into the Global Governance Fusion Layer (GGFL)
    unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Semantic safety panel never triggers conflict directly (conflict: false always)
    - Deterministic output for identical inputs
    - Low weight hint (advisory only)

    Args:
        panel_or_signal: Semantic safety panel from build_semantic_safety_panel()
            or signal from first_light_status.json signals["semantic_safety_panel"]

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-SEM" (identifies this as a semantic signal)
        - status: "ok" | "warn" (warn if not_ok_not_ok > 0)
        - conflict: false (semantic safety panel never triggers conflict directly)
        - weight_hint: "LOW" (advisory only, low weight)
        - drivers: List[str] (up to 3 human-readable cal_ids: top_drivers from panel, sorted)
        - drivers_reason_codes: List[str] (up to 3 deterministic reason codes: SEM-DRV-001, SEM-DRV-002, SEM-DRV-003)
        - summary: str (one neutral sentence describing panel state)
    """
    # Extract grid counts (works for both panel and signal)
    grid_counts = panel_or_signal.get("grid_counts", {})
    if not grid_counts:
        # If signal format, extract directly
        ok_ok = panel_or_signal.get("ok_ok", 0)
        ok_not_ok = panel_or_signal.get("ok_not_ok", 0)
        not_ok_ok = panel_or_signal.get("not_ok_ok", 0)
        not_ok_not_ok = panel_or_signal.get("not_ok_not_ok", 0)
    else:
        ok_ok = grid_counts.get("ok_ok", 0)
        ok_not_ok = grid_counts.get("ok_not_ok", 0)
        not_ok_ok = grid_counts.get("not_ok_ok", 0)
        not_ok_not_ok = grid_counts.get("not_ok_not_ok", 0)
    
    # Determine status: warn if any not_ok_not_ok, otherwise ok
    # Frozen enum: status ∈ {"ok", "warn"}
    status = "warn" if not_ok_not_ok > 0 else "ok"
    
    # Extract top drivers (deterministic, sorted)
    top_drivers = panel_or_signal.get("top_drivers", [])
    drivers = sorted(top_drivers[:3]) if top_drivers else []
    
    # Build drivers_reason_codes list (deterministic ordering: SEM-DRV-001, SEM-DRV-002, SEM-DRV-003)
    drivers_reason_codes: List[str] = []
    reason_code_map = [SEM_DRV_001, SEM_DRV_002, SEM_DRV_003]
    for i, driver in enumerate(drivers[:3]):
        if i < len(reason_code_map):
            drivers_reason_codes.append(f"{reason_code_map[i]}:{driver}")
    
    # Build neutral summary sentence
    total_experiments = panel_or_signal.get("total_experiments", ok_ok + ok_not_ok + not_ok_ok + not_ok_not_ok)
    if not_ok_not_ok > 0:
        summary = f"Semantic safety panel: {not_ok_not_ok} of {total_experiments} experiments show semantic issues in both P3 and P4 phases."
    elif ok_not_ok > 0 or not_ok_ok > 0:
        summary = f"Semantic safety panel: {ok_not_ok + not_ok_ok} of {total_experiments} experiments show semantic issues in one phase."
    else:
        summary = f"Semantic safety panel: all {total_experiments} experiments show no semantic issues."
    
    return {
        "signal_type": "SIG-SEM",
        "status": status,  # Frozen enum: "ok" | "warn"
        "conflict": False,  # Semantic safety panel never triggers conflict directly
        "weight_hint": "LOW",  # Advisory only, low weight
        "drivers": drivers,  # Human-readable cal_ids (up to 3)
        "drivers_reason_codes": drivers_reason_codes,  # Deterministic reason codes (up to 3)
        "summary": summary,
    }


def attach_semantic_safety_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach semantic safety panel to evidence pack.

    STATUS: PHASE X — P5 SEMANTIC SAFETY PANEL EVIDENCE INTEGRATION

    Attaches the semantic safety panel under evidence["governance"]["semantic_safety_panel"]
    and extracts a compact signal under evidence["signals"]["semantic_safety_panel"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - Returns a new dict, does not mutate input evidence
    - The attached panel is purely observational
    - No control flow depends on the panel contents
    - Advisory only; council still decides; no new gate semantics

    Args:
        evidence: Existing evidence pack dictionary (read-only, not modified).
        panel: Semantic safety panel from build_semantic_safety_panel().

    Returns:
        New dict with evidence contents plus semantic_safety_panel attached under governance
        and signals.semantic_safety_panel attached under signals.
    """
    # Create a copy to avoid mutation
    result = dict(evidence)

    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}

    # Attach semantic safety panel
    result["governance"]["semantic_safety_panel"] = panel

    # Extract and attach status signal
    signal = extract_semantic_safety_panel_signal(panel)
    
    # Ensure signals key exists
    if "signals" not in result:
        result["signals"] = {}
    else:
        result["signals"] = dict(result["signals"])

    # Attach semantic safety panel signal
    result["signals"]["semantic_safety_panel"] = signal

    return result


def summarize_semantic_integrity_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize semantic integrity for uplift council.

    STATUS: PHASE X — SEMANTIC–CURRICULUM OVERSIGHT LAYER

    Maps semantic integrity status to council-friendly format.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - No gating, no aborts - purely observability

    Args:
        tile: From build_semantic_integrity_tile().

    Returns:
        Council summary dictionary with:
        - status: OK | WARN | BLOCK
        - invariants_ok: bool
        - uplift_semantic_status: OK | WARN | BLOCK
        - top_risk_terms: List[str]
        - headline: str
    """
    # Extract values
    invariants_ok = tile.get("invariants_ok", False)
    uplift_semantic_status = tile.get("uplift_semantic_status", "OK")
    top_risk_terms = tile.get("top_risk_terms", [])
    headline = tile.get("headline", "")
    
    # Map to council status: OK | WARN | BLOCK
    # BLOCK if uplift_semantic_status is BLOCK or invariants_ok is False
    # WARN if uplift_semantic_status is WARN
    # OK otherwise
    if uplift_semantic_status == "BLOCK" or not invariants_ok:
        status = "BLOCK"
    elif uplift_semantic_status == "WARN":
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "status": status,
        "invariants_ok": invariants_ok,
        "uplift_semantic_status": uplift_semantic_status,
        "top_risk_terms": top_risk_terms,
        "headline": headline,
    }


__all__ = [
    "SEMANTIC_INTEGRITY_TILE_SCHEMA_VERSION",
    "SEMANTIC_FOOTPRINT_SCHEMA_VERSION",
    "SEMANTIC_SAFETY_PANEL_SCHEMA_VERSION",
    "SEM_DRV_001",
    "SEM_DRV_002",
    "SEM_DRV_003",
    "build_semantic_integrity_tile",
    "extract_semantic_drift_signal",
    "build_semantic_integrity_summary_for_p3",
    "build_semantic_integrity_calibration_for_p4",
    "build_first_light_semantic_footprint",
    "emit_cal_exp_semantic_footprint",
    "persist_cal_exp_semantic_footprint",
    "build_semantic_safety_panel",
    "extract_semantic_safety_panel_signal",
    "semantic_safety_panel_for_alignment_view",
    "attach_semantic_integrity_to_evidence",
    "attach_semantic_safety_panel_to_evidence",
    "summarize_semantic_integrity_for_uplift_council",
]

