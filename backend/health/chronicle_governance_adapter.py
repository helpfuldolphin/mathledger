"""Chronicle governance adapter for global health.

STATUS: PHASE X — CHRONICLE-LEVEL GOVERNANCE TILE

Provides integration of Phase V recurrence projection, invariant checking,
and director tile functionality into the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The chronicle_governance tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

from typing import Any, Dict, List, Optional

CHRONICLE_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


def _validate_recurrence_projection(projection: Dict[str, Any]) -> None:
    """Validate recurrence projection structure.
    
    Args:
        projection: Recurrence projection dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = [
        "recurrence_likelihood",
        "drivers",
        "projected_recurrence_horizon",
        "neutral_explanation"
    ]
    missing = [key for key in required_keys if key not in projection]
    if missing:
        raise ValueError(
            f"recurrence_projection missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(projection.keys()))}"
        )


def _validate_invariant_check(check: Dict[str, Any]) -> None:
    """Validate invariant check structure.
    
    Args:
        check: Invariant check dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["invariant_status", "broken_invariants", "explanations"]
    missing = [key for key in required_keys if key not in check]
    if missing:
        raise ValueError(
            f"invariant_check missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(check.keys()))}"
        )


def _validate_stability_scores(scores: Dict[str, Any]) -> None:
    """Validate stability scores structure.
    
    Args:
        scores: Stability scores dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["stability_band", "axes_contributing", "headline"]
    missing = [key for key in required_keys if key not in scores]
    if missing:
        raise ValueError(
            f"stability_scores missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(scores.keys()))}"
        )


def build_chronicle_governance_tile(
    recurrence_projection: Dict[str, Any],
    invariant_check: Dict[str, Any],
    stability_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build chronicle governance tile for global health surface.

    STATUS: PHASE X — CHRONICLE-LEVEL GOVERNANCE TILE

    Integrates Phase V recurrence projection, invariant checking, and stability
    scores into a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents

    Args:
        recurrence_projection: Recurrence projection from build_recurrence_projection_engine().
            Must contain: recurrence_likelihood, drivers, projected_recurrence_horizon,
            neutral_explanation
        invariant_check: Invariant check from build_phase_transition_drift_invariant_checker().
            Must contain: invariant_status, broken_invariants, explanations
        stability_scores: Stability scores from estimate_multi_axis_chronicle_stability().
            Must contain: stability_band, axes_contributing, headline

    Returns:
        Chronicle governance tile dictionary with:
        - schema_version
        - tile_type: "chronicle_governance"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - recurrence_band: "LOW" | "MEDIUM" | "HIGH"
        - projected_horizon_days: int
        - invariants_ok: bool
        - highlighted_cases: List[str] (top 3)
        - headline: str (neutral summary)
    """
    # Validate inputs
    _validate_recurrence_projection(recurrence_projection)
    _validate_invariant_check(invariant_check)
    _validate_stability_scores(stability_scores)
    
    # Extract key metrics
    recurrence_likelihood = recurrence_projection.get("recurrence_likelihood", 0.0)
    projected_horizon = recurrence_projection.get("projected_recurrence_horizon", 30)
    drivers = recurrence_projection.get("drivers", [])
    
    invariant_status = invariant_check.get("invariant_status", "OK")
    broken_invariants = invariant_check.get("broken_invariants", [])
    invariants_ok = invariant_status == "OK"
    
    stability_band = stability_scores.get("stability_band", "MEDIUM")
    axes_contributing = stability_scores.get("axes_contributing", [])
    
    # Determine recurrence band
    if recurrence_likelihood >= 0.7:
        recurrence_band = "HIGH"
    elif recurrence_likelihood >= 0.4:
        recurrence_band = "MEDIUM"
    else:
        recurrence_band = "LOW"
    
    # Determine status light (RED on invariant violations, YELLOW on high recurrence/low stability, GREEN otherwise)
    if not invariants_ok:
        status_light = "RED"
    elif recurrence_band == "HIGH" or stability_band == "LOW":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Build highlighted cases (top 3)
    highlighted_cases: List[str] = []
    
    if recurrence_band == "HIGH":
        top_drivers = drivers[:2] if len(drivers) >= 2 else drivers
        if top_drivers:
            highlighted_cases.append(f"High recurrence likelihood: {', '.join(top_drivers)}")
    
    if broken_invariants:
        violation_count = len(broken_invariants)
        highlighted_cases.append(f"{violation_count} invariant violation(s) detected")
    
    if stability_band == "LOW" and axes_contributing:
        highlighted_cases.append(f"Stability concerns: {', '.join(axes_contributing[:2])}")
    
    # Limit to top 3
    highlighted_cases = highlighted_cases[:3]
    
    if not highlighted_cases:
        highlighted_cases.append("No notable concerns")
    
    # Build neutral headline
    status_desc = status_light.lower()
    headline = (
        f"Chronicle status: {status_desc}. "
        f"Recurrence likelihood: {recurrence_likelihood:.2f} "
        f"(projected horizon: {projected_horizon} days). "
        f"Invariants: {'satisfied' if invariants_ok else 'violated'}."
    )
    
    return {
        "schema_version": CHRONICLE_GOVERNANCE_TILE_SCHEMA_VERSION,
        "tile_type": "chronicle_governance",
        "status_light": status_light,
        "recurrence_band": recurrence_band,
        "projected_horizon_days": projected_horizon,
        "invariants_ok": invariants_ok,
        "highlighted_cases": highlighted_cases,
        "headline": headline,
    }


def extract_chronicle_drift_signal(
    recurrence_projection: Dict[str, Any],
    invariant_check: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract chronicle drift signal for P4 digital twin divergence logs.

    STATUS: PHASE X — DIGITAL TWIN DRIFT HOOK

    Extracts key signals from recurrence projection and invariant check
    for integration into P4 shadow-mode divergence logs.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        recurrence_projection: Recurrence projection from build_recurrence_projection_engine().
        invariant_check: Invariant check from build_phase_transition_drift_invariant_checker().

    Returns:
        Chronicle drift signal dictionary with:
        - recurrence_likelihood: float [0.0, 1.0]
        - invariants_ok: bool
        - band: "LOW" | "MEDIUM" | "HIGH" (recurrence band)
    """
    recurrence_likelihood = recurrence_projection.get("recurrence_likelihood", 0.0)
    invariant_status = invariant_check.get("invariant_status", "OK")
    invariants_ok = invariant_status == "OK"
    
    # Determine recurrence band
    if recurrence_likelihood >= 0.7:
        band = "HIGH"
    elif recurrence_likelihood >= 0.4:
        band = "MEDIUM"
    else:
        band = "LOW"
    
    return {
        "recurrence_likelihood": recurrence_likelihood,
        "invariants_ok": invariants_ok,
        "band": band,
    }


def build_first_light_chronicle_annex(
    recurrence_projection: Dict[str, Any],
    invariant_check: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First Light chronicle recurrence annex.

    STATUS: PHASE X — FIRST LIGHT RECURRENCE ANNEX

    Provides a recurrence annex that answers: "how likely is this anomaly to recur?"

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned annex is purely observational
    - No control flow depends on the annex contents
    - Annex is descriptive only; council + gates unchanged

    Args:
        recurrence_projection: Recurrence projection from build_recurrence_projection_engine()
        invariant_check: Invariant check from build_phase_transition_drift_invariant_checker()

    Returns:
        First Light chronicle annex dictionary with:
        - schema_version: "1.0.0"
        - recurrence_likelihood: float [0.0, 1.0]
        - band: "LOW" | "MEDIUM" | "HIGH"
        - invariants_ok: bool
    """
    recurrence_likelihood = recurrence_projection.get("recurrence_likelihood", 0.0)
    invariant_status = invariant_check.get("invariant_status", "OK")
    invariants_ok = invariant_status == "OK"
    
    # Determine band from recurrence likelihood
    if recurrence_likelihood >= 0.7:
        band = "HIGH"
    elif recurrence_likelihood >= 0.4:
        band = "MEDIUM"
    else:
        band = "LOW"
    
    return {
        "schema_version": "1.0.0",
        "recurrence_likelihood": recurrence_likelihood,
        "band": band,
        "invariants_ok": invariants_ok,
    }


def attach_chronicle_governance_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    drift_signal: Dict[str, Any],
    recurrence_projection: Optional[Dict[str, Any]] = None,
    invariant_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach chronicle governance tile and drift signal to evidence pack.

    STATUS: PHASE X — EVIDENCE PACK INTEGRATION

    Stores chronicle governance information under evidence["governance"]["chronicle"]
    for inclusion in evidence packs. Includes First Light recurrence annex if
    recurrence_projection and invariant_check are provided.

    SHADOW MODE CONTRACT:
    - This function modifies the evidence dict in-place
    - The attachment is purely observational
    - No control flow depends on the attached data

    Args:
        evidence: Evidence pack dictionary (will be modified in-place)
        tile: Chronicle governance tile from build_chronicle_governance_tile()
        drift_signal: Chronicle drift signal from extract_chronicle_drift_signal()
        recurrence_projection: Optional recurrence projection for First Light annex
        invariant_check: Optional invariant check for First Light annex

    Returns:
        Modified evidence dictionary with chronicle governance attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach chronicle governance
    evidence["governance"]["chronicle"] = {
        "recurrence_likelihood": drift_signal.get("recurrence_likelihood", 0.0),
        "band": drift_signal.get("band", "LOW"),
        "invariants_ok": drift_signal.get("invariants_ok", True),
        "highlighted_cases": tile.get("highlighted_cases", []),
        "status_light": tile.get("status_light", "GREEN"),
        "recurrence_band": tile.get("recurrence_band", "LOW"),
        "projected_horizon_days": tile.get("projected_horizon_days", 30),
    }
    
    # Attach First Light recurrence annex if projection and check provided
    if recurrence_projection is not None and invariant_check is not None:
        evidence["governance"]["chronicle"]["first_light_annex"] = (
            build_first_light_chronicle_annex(recurrence_projection, invariant_check)
        )
    
    return evidence


def summarize_chronicle_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize chronicle governance for uplift council decision-making.

    STATUS: PHASE X — UPLIFT COUNCIL HOOK

    Maps chronicle governance tile to council decision signals:
    - BLOCK: invariants_ok=False OR band="HIGH"
    - WARN: band="MEDIUM"
    - OK: otherwise

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - This is a drift & invariants observer, not a hard control path

    Args:
        tile: Chronicle governance tile from build_chronicle_governance_tile()

    Returns:
        Council summary dictionary with:
        - council_status: "OK" | "WARN" | "BLOCK"
        - invariants_ok: bool
        - recurrence_band: "LOW" | "MEDIUM" | "HIGH"
        - rationale: str (neutral explanation)
    """
    invariants_ok = tile.get("invariants_ok", True)
    recurrence_band = tile.get("recurrence_band", "LOW")
    
    # Determine council status
    if not invariants_ok or recurrence_band == "HIGH":
        council_status = "BLOCK"
    elif recurrence_band == "MEDIUM":
        council_status = "WARN"
    else:
        council_status = "OK"
    
    # Build rationale
    if not invariants_ok:
        rationale = "Invariant violations detected in chronicle drift analysis."
    elif recurrence_band == "HIGH":
        rationale = "High recurrence likelihood indicates elevated drift risk."
    elif recurrence_band == "MEDIUM":
        rationale = "Moderate recurrence likelihood suggests monitoring recommended."
    else:
        rationale = "Chronicle governance signals indicate stable drift patterns."
    
    return {
        "council_status": council_status,
        "invariants_ok": invariants_ok,
        "recurrence_band": recurrence_band,
        "rationale": rationale,
    }


def build_cal_exp_recurrence_snapshot(
    cal_id: str,
    annex: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build per-experiment recurrence snapshot for calibration experiments.

    STATUS: PHASE X — CAL-EXP RECURRENCE SNAPSHOT

    Creates a snapshot of recurrence data for a single calibration experiment.
    Optionally emits to `calibration/chronicle_recurrence_<cal_id>.json`.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from file writing)
    - The snapshot is purely observational
    - No control flow depends on the snapshot contents

    Args:
        cal_id: Calibration experiment identifier
        annex: First Light recurrence annex from build_first_light_chronicle_annex()
        output_dir: Optional directory to emit JSON file. If None, no file is written.

    Returns:
        Recurrence snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - recurrence_likelihood: float [0.0, 1.0]
        - band: "LOW" | "MEDIUM" | "HIGH"
        - invariants_ok: bool
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    snapshot = {
        "schema_version": "1.0.0",
        "cal_id": cal_id,
        "recurrence_likelihood": annex.get("recurrence_likelihood", 0.0),
        "band": annex.get("band", "LOW"),
        "invariants_ok": annex.get("invariants_ok", True),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Emit to file if output_dir provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"chronicle_recurrence_{cal_id}.json"
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)

    return snapshot


def build_chronicle_risk_register(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build chronicle risk register from calibration experiment snapshots.

    STATUS: PHASE X — CAL-EXP RISK REGISTER

    Aggregates recurrence snapshots across calibration experiments to provide
    a risk register for reviewers. Identifies high-risk calibrations and
    provides band distribution statistics.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned register is purely observational (advisory)
    - No control flow depends on the register contents

    Args:
        snapshots: List of recurrence snapshots from build_cal_exp_recurrence_snapshot()

    Returns:
        Risk register dictionary with:
        - schema_version: "1.0.0"
        - total_calibrations: int
        - band_counts: Dict[str, int] (LOW/MEDIUM/HIGH)
        - high_risk_calibrations: List[str] (cal_ids with HIGH + invariants_ok=False)
        - risk_summary: str (neutral summary)
    """
    total = len(snapshots)
    
    # Count bands
    band_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    high_risk_calibrations: List[str] = []
    high_risk_details: List[Dict[str, Any]] = []
    
    for snapshot in snapshots:
        band = snapshot.get("band", "LOW")
        if band in band_counts:
            band_counts[band] += 1
        
        # High risk: HIGH band + invariants violated
        if (band == "HIGH" and not snapshot.get("invariants_ok", True)):
            cal_id = snapshot.get("cal_id", "unknown")
            high_risk_calibrations.append(cal_id)
            
            # Build detail entry with evidence path hint
            high_risk_details.append({
                "cal_id": cal_id,
                "recurrence_likelihood": snapshot.get("recurrence_likelihood", 0.0),
                "invariants_ok": snapshot.get("invariants_ok", True),
                "evidence_path_hint": f"calibration/chronicle_recurrence_{cal_id}.json",
            })
    
    # Sort high_risk_details deterministically by cal_id
    high_risk_details.sort(key=lambda x: x["cal_id"])
    
    # Build risk summary
    if total == 0:
        risk_summary = "No calibration experiments analyzed."
    elif len(high_risk_calibrations) > 0:
        risk_summary = (
            f"Analyzed {total} calibration experiment(s). "
            f"{len(high_risk_calibrations)} high-risk calibration(s) identified "
            f"(HIGH recurrence + invariants violated)."
        )
    elif band_counts["HIGH"] > 0:
        risk_summary = (
            f"Analyzed {total} calibration experiment(s). "
            f"{band_counts['HIGH']} calibration(s) with HIGH recurrence likelihood, "
            f"but invariants intact."
        )
    else:
        risk_summary = (
            f"Analyzed {total} calibration experiment(s). "
            f"Recurrence patterns indicate low to moderate risk across experiments."
        )
    
    return {
        "schema_version": "1.0.0",
        "total_calibrations": total,
        "band_counts": band_counts,
        "high_risk_calibrations": sorted(high_risk_calibrations),
        "high_risk_details": high_risk_details,
        "risk_summary": risk_summary,
    }


def attach_chronicle_risk_register_to_evidence(
    evidence: Dict[str, Any],
    risk_register: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach chronicle risk register to evidence pack.

    STATUS: PHASE X — EVIDENCE PACK RISK REGISTER INTEGRATION

    Stores the chronicle risk register under evidence["governance"]["chronicle_risk_register"]
    for inclusion in evidence packs. The register is advisory only.

    SHADOW MODE CONTRACT:
    - This function modifies the evidence dict in-place
    - The attachment is purely observational (advisory)
    - No control flow depends on the attached data

    Args:
        evidence: Evidence pack dictionary (will be modified in-place)
        risk_register: Risk register from build_chronicle_risk_register()

    Returns:
        Modified evidence dictionary with chronicle risk register attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach chronicle risk register
    evidence["governance"]["chronicle_risk_register"] = risk_register
    
    return evidence


def chronicle_risk_for_alignment_view(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert chronicle risk register signal to GGFL alignment view format.

    STATUS: PHASE X — GGFL ADAPTER FOR CHRONICLE RISK REGISTER

    Normalizes the chronicle risk register signal into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Chronicle risk never triggers conflict directly (conflict: false always)
    - Deterministic output for identical inputs
    - Advisory only; no enforcement

    Args:
        signal: Chronicle risk signal from first_light_status.json signals["chronicle_risk"]
            Must contain: total_calibrations, high_risk_count, high_risk_cal_ids_top3,
            has_any_invariants_violated

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-CHR" (identifies this as a chronicle risk signal)
        - status: "ok" | "warn" (warn if high_risk_count > 0 or has_any_invariants_violated)
        - conflict: false (chronicle risk never triggers conflict directly)
        - weight_hint: "LOW" (chronicle risk is low-weight advisory signal)
        - drivers: List[str] (deterministic: high_risk_count, invariants_violated, top cal_ids)
        - summary: str (one neutral sentence)
        - extraction_source: str (MANIFEST | EVIDENCE_JSON | MISSING)
    """
    total_calibrations = signal.get("total_calibrations", 0)
    high_risk_count = signal.get("high_risk_count", 0)
    high_risk_cal_ids_top3 = signal.get("high_risk_cal_ids_top3", [])
    has_any_invariants_violated = signal.get("has_any_invariants_violated", False)
    extraction_source = signal.get("extraction_source", "MISSING")
    
    # Determine status: warn if any high-risk calibrations or invariants violated
    status = "warn" if (high_risk_count > 0 or has_any_invariants_violated) else "ok"
    
    # Build drivers (deterministic list with frozen prefixes)
    # SIG-CHR CONTRACT v1 FREEZE: Drivers must use deterministic prefixes CHR-DRV-001, CHR-DRV-002, CHR-DRV-003
    drivers: List[str] = []
    
    if high_risk_count > 0:
        drivers.append(f"CHR-DRV-001: high_risk_count={high_risk_count}")
    
    if has_any_invariants_violated:
        drivers.append("CHR-DRV-002: HIGH recurrence + invariants violated")
    
    if high_risk_cal_ids_top3:
        # Join top 3 cal_ids deterministically (already sorted)
        cal_ids_str = ", ".join(high_risk_cal_ids_top3)
        drivers.append(f"CHR-DRV-003: top_risk_cal_ids={cal_ids_str}")
    
    # Limit to 3 drivers (SIG-CHR CONTRACT v1 FREEZE: max 3 drivers)
    drivers = drivers[:3]
    
    # Build neutral summary sentence
    if total_calibrations == 0:
        summary = "No calibration experiments analyzed for chronicle recurrence risk."
    elif high_risk_count > 0:
        if has_any_invariants_violated:
            summary = (
                f"Chronicle risk register: {high_risk_count} out of {total_calibrations} "
                f"calibration experiment(s) show HIGH recurrence likelihood with invariants violated."
            )
        else:
            summary = (
                f"Chronicle risk register: {high_risk_count} out of {total_calibrations} "
                f"calibration experiment(s) show HIGH recurrence likelihood, invariants intact."
            )
    elif has_any_invariants_violated:
        # Edge case: invariants violated but no high-risk calibrations
        summary = (
            f"Chronicle risk register: {total_calibrations} calibration experiment(s) "
            f"show invariants violated, but recurrence risk is low to moderate."
        )
    else:
        summary = (
            f"Chronicle risk register: {total_calibrations} calibration experiment(s) "
            f"show low to moderate recurrence risk patterns."
        )
    
    # SIG-CHR CONTRACT v1 FREEZE: status ∈ {"ok", "warn"}
    # Coerce to lowercase to ensure enum compliance
    status_normalized = status.lower()
    if status_normalized not in ("ok", "warn"):
        status_normalized = "ok"  # Default to "ok" if invalid
    
    # SIG-CHR CONTRACT v1 FREEZE: extraction_source ∈ {"MANIFEST", "EVIDENCE_JSON", "MISSING"}
    # Coerce to valid enum value
    if extraction_source is None:
        extraction_source_normalized = "MISSING"
    else:
        extraction_source_normalized = str(extraction_source).upper()
        if extraction_source_normalized not in ("MANIFEST", "EVIDENCE_JSON", "MISSING"):
            extraction_source_normalized = "MISSING"  # Default to "MISSING" if invalid
    
    return {
        "signal_type": "SIG-CHR",
        "status": status_normalized,  # Frozen enum: "ok" | "warn"
        "conflict": False,  # SIG-CHR CONTRACT v1 FREEZE: conflict must always be False (invariant)
        "weight_hint": "LOW",
        "drivers": drivers,  # Frozen: max 3, deterministic prefixes CHR-DRV-001, CHR-DRV-002, CHR-DRV-003
        "summary": summary,
        "extraction_source": extraction_source_normalized,  # Frozen enum: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"
    }


CHRONICLE_RISK_CONSISTENCY_SCHEMA_VERSION = "1.0.0"


def summarize_chronicle_risk_signal_consistency(
    status_signal: Dict[str, Any],
    ggfl_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-check consistency between status signal and GGFL signal for chronicle risk.
    
    SIG-CHR CONTRACT v1 FREEZE: Status↔GGFL Consistency Checker
    
    This function validates that the status signal and GGFL signal are consistent
    with each other and that the conflict invariant is maintained (conflict must
    always be False).
    
    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Detects inconsistencies for advisory purposes only
    - No gating, just advisory notes
    
    Args:
        status_signal: Chronicle risk signal from status JSON (from generate_first_light_status)
            Must contain: total_calibrations, high_risk_count, has_any_invariants_violated
        ggfl_signal: Chronicle risk signal from GGFL adapter (from chronicle_risk_for_alignment_view)
            Must contain: signal_type, status, conflict, drivers
    
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - mode: "SHADOW"
        - consistency: "CONSISTENT" | "PARTIAL" | "INCONSISTENT"
        - notes: List of neutral descriptive notes about inconsistencies
        - conflict_invariant_violated: bool (True if conflict is ever True)
        - top_mismatch_type: Optional[str] (Top mismatch type for INCONSISTENT cases)
    """
    notes: List[str] = []
    consistency = "CONSISTENT"
    conflict_invariant_violated = False
    top_mismatch_type: Optional[str] = None
    
    # Extract status values
    # Status signal may have status field or derive from high_risk_count/invariants
    status_high_risk_count = status_signal.get("high_risk_count", 0)
    status_has_invariants_violated = status_signal.get("has_any_invariants_violated", False)
    status_derived = "warn" if (status_high_risk_count > 0 or status_has_invariants_violated) else "ok"
    
    # GGFL signal status (already normalized to lowercase)
    ggfl_status = ggfl_signal.get("status", "ok").lower()
    
    # Check status consistency
    status_mismatch = False
    if status_derived != ggfl_status:
        notes.append(
            f"Status mismatch: status signal derived status '{status_derived}' "
            f"(high_risk_count={status_high_risk_count}, invariants_violated={status_has_invariants_violated}) "
            f"but GGFL says '{ggfl_status}'"
        )
        status_mismatch = True
        consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "status_mismatch"
    
    # Check conflict invariant (MUST always be False)
    # This is the only condition that causes INCONSISTENT
    ggfl_conflict = ggfl_signal.get("conflict", False)
    if ggfl_conflict is True:
        notes.append(
            "CRITICAL: Conflict invariant violated - GGFL signal has conflict=True. "
            "Chronicle risk must never trigger conflict (conflict must always be False)."
        )
        conflict_invariant_violated = True
        consistency = "INCONSISTENT"
        top_mismatch_type = "conflict_invariant_violated"
    
    # Check driver count (should be <= 3 per contract)
    ggfl_drivers = ggfl_signal.get("drivers", [])
    if len(ggfl_drivers) > 3:
        notes.append(
            f"Driver count violation: GGFL signal has {len(ggfl_drivers)} drivers, "
            f"but contract requires max 3 drivers"
        )
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "driver_count_violation"
    
    # Check driver prefix format (should use CHR-DRV-001, CHR-DRV-002, CHR-DRV-003)
    for i, driver in enumerate(ggfl_drivers):
        expected_prefix = f"CHR-DRV-{i+1:03d}"
        if not driver.startswith(expected_prefix):
            notes.append(
                f"Driver prefix violation: Driver {i+1} has prefix '{driver.split(':')[0] if ':' in driver else driver}', "
                f"but contract requires '{expected_prefix}'"
            )
            if consistency == "CONSISTENT":
                consistency = "PARTIAL"
            if top_mismatch_type is None:
                top_mismatch_type = "driver_prefix_violation"
    
    # If no issues found, return consistent
    if not notes:
        notes.append("Status signal and GGFL signal are consistent")
    
    return {
        "schema_version": CHRONICLE_RISK_CONSISTENCY_SCHEMA_VERSION,
        "mode": "SHADOW",
        "consistency": consistency,
        "notes": notes,
        "conflict_invariant_violated": conflict_invariant_violated,
        "top_mismatch_type": top_mismatch_type,
    }


__all__ = [
    "CHRONICLE_GOVERNANCE_TILE_SCHEMA_VERSION",
    "CHRONICLE_RISK_CONSISTENCY_SCHEMA_VERSION",
    "build_chronicle_governance_tile",
    "extract_chronicle_drift_signal",
    "build_first_light_chronicle_annex",
    "attach_chronicle_governance_to_evidence",
    "summarize_chronicle_for_uplift_council",
    "build_cal_exp_recurrence_snapshot",
    "build_chronicle_risk_register",
    "attach_chronicle_risk_register_to_evidence",
    "chronicle_risk_for_alignment_view",
    "summarize_chronicle_risk_signal_consistency",
]

