"""Epistemic drift integration for P3/P4 reports and evidence packs.

Provides integration functions to attach epistemic drift analysis to:
- P3 stability reports (first_light_stability_report.json)
- P4 calibration reports (p4_calibration_report.json)
- Evidence packs (evidence["governance"]["epistemic_drift"])

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Epistemic drift is purely observational
- Never blocks or modifies pipeline behavior
- Deterministic and JSON-serializable
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from rfl.verification.abstention_semantics import (
    build_abstention_storyline,
    build_epistemic_abstention_profile,
    build_epistemic_drift_timeline,
)

EPISTEMIC_SUMMARY_SCHEMA_VERSION = "1.0.0"
EPISTEMIC_CALIBRATION_SCHEMA_VERSION = "1.0.0"
EPISTEMIC_EVIDENCE_SCHEMA_VERSION = "1.0.0"
FIRST_LIGHT_FOOTPRINT_SCHEMA_VERSION = "1.0.0"
CAL_EXP_FOOTPRINT_SCHEMA_VERSION = "1.0.0"
CALIBRATION_PANEL_SCHEMA_VERSION = "1.0.0"
EXTRACTION_AUDIT_SCHEMA_VERSION = "1.0.0"

# Frozen allowed extraction path values
EXTRACTION_PATHS = frozenset(["DIRECT", "NESTED", "FALLBACK", "DEFAULTS"])


def build_epistemic_summary_for_p3(
    profiles: Sequence[Dict[str, Any]],
    drift_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build epistemic_summary block for P3 stability report.

    Computes mean epistemic risk, drift band, and identifies abstention anomalies.

    Args:
        profiles: Sequence of epistemic abstention profiles
        drift_timeline: Optional pre-computed drift timeline (if None, computed from profiles)

    Returns:
        {
            "schema_version": "1.0.0",
            "mean_epistemic_risk": "LOW" | "MEDIUM" | "HIGH",
            "drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "abstention_anomalies": List[str],
            "risk_distribution": Dict[str, int]  # Counts per risk band
        }
    """
    if not profiles:
        return {
            "schema_version": EPISTEMIC_SUMMARY_SCHEMA_VERSION,
            "mean_epistemic_risk": "LOW",
            "drift_band": "STABLE",
            "abstention_anomalies": [],
            "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
        }

    # Compute drift timeline if not provided
    if drift_timeline is None:
        drift_timeline = build_epistemic_drift_timeline(list(profiles))

    # Compute mean epistemic risk
    risk_bands = [p.get("epistemic_risk_band", "LOW") for p in profiles]
    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risk_values = [risk_map.get(band, 0) for band in risk_bands]
    mean_risk_value = sum(risk_values) / len(risk_values) if risk_values else 0.0

    if mean_risk_value >= 1.5:
        mean_epistemic_risk = "HIGH"
    elif mean_risk_value >= 0.5:
        mean_epistemic_risk = "MEDIUM"
    else:
        mean_epistemic_risk = "LOW"

    # Get drift band from timeline
    drift_band = drift_timeline.get("risk_band", "STABLE")

    # Identify abstention anomalies
    anomalies: List[str] = []
    if drift_band == "VOLATILE":
        anomalies.append("High drift volatility detected")
    if mean_epistemic_risk == "HIGH":
        anomalies.append("High mean epistemic risk")
    
    # Check for significant transitions
    change_points = drift_timeline.get("change_points", [])
    if len(change_points) > 3:
        anomalies.append(f"Multiple risk transitions ({len(change_points)} change points)")

    # Count risk distribution
    risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for band in risk_bands:
        risk_distribution[band] = risk_distribution.get(band, 0) + 1

    return {
        "schema_version": EPISTEMIC_SUMMARY_SCHEMA_VERSION,
        "mean_epistemic_risk": mean_epistemic_risk,
        "drift_band": drift_band,
        "abstention_anomalies": anomalies,
        "risk_distribution": risk_distribution,
    }


def build_epistemic_calibration_for_p4(
    profiles: Sequence[Dict[str, Any]],
    drift_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build epistemic_calibration section for P4 calibration report.

    Uses timeline function to compute mean risk, drift band, variance, and structural anomalies.

    Args:
        profiles: Sequence of epistemic abstention profiles
        drift_timeline: Optional pre-computed drift timeline (if None, computed from profiles)

    Returns:
        {
            "schema_version": "1.0.0",
            "mean_risk": "LOW" | "MEDIUM" | "HIGH",
            "drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "variance": float,  # Normalized variance of risk bands
            "structural_anomalies": List[str],
            "change_points": List[Dict],
            "drift_index": float
        }
    """
    if not profiles:
        return {
            "schema_version": EPISTEMIC_CALIBRATION_SCHEMA_VERSION,
            "mean_risk": "LOW",
            "drift_band": "STABLE",
            "variance": 0.0,
            "structural_anomalies": [],
            "change_points": [],
            "drift_index": 0.0,
        }

    # Compute drift timeline if not provided
    if drift_timeline is None:
        drift_timeline = build_epistemic_drift_timeline(list(profiles))

    # Extract metrics from timeline
    drift_band = drift_timeline.get("risk_band", "STABLE")
    drift_index = drift_timeline.get("drift_index", 0.0)
    change_points = drift_timeline.get("change_points", [])

    # Compute mean risk
    risk_bands = [p.get("epistemic_risk_band", "LOW") for p in profiles]
    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    risk_values = [risk_map.get(band, 0) for band in risk_bands]
    mean_risk_value = sum(risk_values) / len(risk_values) if risk_values else 0.0

    if mean_risk_value >= 1.5:
        mean_risk = "HIGH"
    elif mean_risk_value >= 0.5:
        mean_risk = "MEDIUM"
    else:
        mean_risk = "LOW"

    # Compute variance (normalized)
    if len(risk_values) > 1:
        mean_val = sum(risk_values) / len(risk_values)
        variance = sum((v - mean_val) ** 2 for v in risk_values) / len(risk_values)
        # Normalize to 0-1 range (max variance for 3-band system is 1.33)
        normalized_variance = min(1.0, variance / 1.33) if 1.33 > 0 else 0.0
    else:
        normalized_variance = 0.0

    # Identify structural anomalies
    structural_anomalies: List[str] = []
    if drift_band == "VOLATILE":
        structural_anomalies.append("Volatile drift pattern detected")
    if drift_index > 0.7:
        structural_anomalies.append(f"High drift index ({drift_index:.2f})")
    if len(change_points) > 5:
        structural_anomalies.append(f"Excessive transitions ({len(change_points)} change points)")
    if mean_risk == "HIGH" and drift_band != "STABLE":
        structural_anomalies.append("High risk combined with drift instability")

    return {
        "schema_version": EPISTEMIC_CALIBRATION_SCHEMA_VERSION,
        "mean_risk": mean_risk,
        "drift_band": drift_band,
        "variance": round(normalized_variance, 3),
        "structural_anomalies": structural_anomalies,
        "change_points": change_points,
        "drift_index": round(drift_index, 3),
    }


def build_first_light_epistemic_footprint(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine P3 + P4 into a compact First Light epistemic footprint.

    Creates a single, compact summary combining P3 stability and P4 calibration
    epistemic drift signals.

    Args:
        p3_summary: P3 epistemic summary from build_epistemic_summary_for_p3
        p4_calibration: P4 epistemic calibration from build_epistemic_calibration_for_p4

    Returns:
        {
            "schema_version": "1.0.0",
            "p3_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p4_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p3_mean_risk": "LOW" | "MEDIUM" | "HIGH",
            "p4_mean_risk": "LOW" | "MEDIUM" | "HIGH",
        }
    """
    return {
        "schema_version": FIRST_LIGHT_FOOTPRINT_SCHEMA_VERSION,
        "p3_drift_band": p3_summary.get("drift_band", "STABLE"),
        "p4_drift_band": p4_calibration.get("drift_band", "STABLE"),
        "p3_mean_risk": p3_summary.get("mean_epistemic_risk", "LOW"),
        "p4_mean_risk": p4_calibration.get("mean_risk", "LOW"),
    }


def attach_epistemic_drift_to_evidence(
    evidence: Dict[str, Any],
    drift_timeline: Dict[str, Any],
    storyline: Optional[Dict[str, Any]] = None,
    first_light_footprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach epistemic drift to evidence pack.

    Stores under evidence["governance"]["epistemic_drift"].
    Includes drift band, key transitions, storyline episode summaries, and
    First Light epistemic footprint (if provided).

    Args:
        evidence: Evidence pack dictionary (will be modified in-place, but function is deterministic)
        drift_timeline: Drift timeline from build_epistemic_drift_timeline
        storyline: Optional storyline from build_abstention_storyline
        first_light_footprint: Optional First Light footprint from build_first_light_epistemic_footprint

    Returns:
        Updated evidence dictionary (same object, deterministic mutation)

    Note:
        Function mutates evidence dict but is deterministic (same inputs = same outputs).
        This is acceptable for evidence pack construction.
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Extract key information from drift timeline
    drift_band = drift_timeline.get("risk_band", "STABLE")
    drift_index = drift_timeline.get("drift_index", 0.0)
    change_points = drift_timeline.get("change_points", [])

    # Extract key transitions (most significant)
    key_transitions = sorted(
        change_points,
        key=lambda x: x.get("change_magnitude", 0.0),
        reverse=True,
    )[:5]  # Top 5 transitions

    # Build storyline episode summaries if available
    storyline_episodes = None
    if storyline is not None:
        storyline_episodes = {
            "trend": storyline.get("global_epistemic_trend", "STABLE"),
            "story": storyline.get("story", ""),
        }

    # Build epistemic drift section
    epistemic_drift = {
        "schema_version": EPISTEMIC_EVIDENCE_SCHEMA_VERSION,
        "drift_band": drift_band,
        "drift_index": round(drift_index, 3),
        "key_transitions": [
            {
                "slice_name": t.get("slice_name", "unknown"),
                "transition": t.get("transition", ""),
                "change_magnitude": round(t.get("change_magnitude", 0.0), 3),
            }
            for t in key_transitions
        ],
        "storyline_episodes": storyline_episodes,
        "summary_text": drift_timeline.get("summary_text", ""),
    }

    # Add First Light footprint if provided
    if first_light_footprint is not None:
        epistemic_drift["first_light_footprint"] = first_light_footprint

    # Attach to evidence
    evidence["governance"]["epistemic_drift"] = epistemic_drift

    return evidence


def attach_epistemic_calibration_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: Dict[str, Any],
    advisory_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Attach epistemic calibration panel to evidence pack.

    Stores under evidence["governance"]["epistemic_calibration_panel"].

    Args:
        evidence: Evidence pack dictionary (will be modified in-place, but function is deterministic)
        panel: Calibration panel from build_epistemic_calibration_panel
        advisory_notes: Optional list of neutral advisory note strings

    Returns:
        Updated evidence dictionary (same object, deterministic mutation)

    Note:
        Function mutates evidence dict but is deterministic (same inputs = same outputs).
        This is acceptable for evidence pack construction.
        Advisory notes are neutral observation strings only (no enforcement semantics).
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}

    # Build enriched panel with advisory notes
    enriched_panel = dict(panel)
    if advisory_notes is not None:
        enriched_panel["advisory_notes"] = advisory_notes
    else:
        enriched_panel["advisory_notes"] = []

    # Attach panel
    evidence["governance"]["epistemic_calibration_panel"] = enriched_panel

    return evidence


def emit_epistemic_footprint_from_cal_exp_report(
    cal_id: str,
    cal_exp_report: Dict[str, Any],
    strict_extraction: bool = False,
) -> Dict[str, Any]:
    """
    Auto-emit epistemic footprint from CAL-EXP report.

    Extracts P3 and P4 epistemic data from a calibration experiment report
    and emits a footprint. Uses safe defaults if data is missing.

    EXTRACTION PRECEDENCE (enforced in order):
    1. DIRECT: report["epistemic_summary"] and report["epistemic_calibration"]
    2. NESTED: report["p3"]["epistemic_summary"] and report["p4"]["epistemic_calibration"]
    3. FALLBACK: report["epistemic_alignment_summary"] and report["epistemic_alignment"]
    4. DEFAULTS: Safe defaults (STABLE, LOW) with confidence="LOW" and advisory_note="DEFAULTS_USED"

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        cal_exp_report: CAL-EXP report dictionary containing P3/P4 data
        strict_extraction: If True, detects when multiple sources are present simultaneously
            and adds advisory_note="MULTIPLE_SOURCES_PRESENT" (advisory only, no exceptions)

    Returns:
        Epistemic footprint with extraction_path and extraction_audit fields:
        {
            "schema_version": "1.0.0",
            "cal_id": str,
            "p3_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p4_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p3_mean_risk": "LOW" | "MEDIUM" | "HIGH",
            "p4_mean_risk": "LOW" | "MEDIUM" | "HIGH",
            "extraction_path": "DIRECT" | "NESTED" | "FALLBACK" | "DEFAULTS",
            "extraction_audit": [
                {"path": "DIRECT", "found": bool, "fields": List[str]},
                {"path": "NESTED", "found": bool, "fields": List[str]},
                {"path": "FALLBACK", "found": bool, "fields": List[str]},
                {"path": "DEFAULTS", "found": bool, "fields": List[str]}
            ],
            "confidence": "LOW" | "MEDIUM" | "HIGH" (only if DEFAULTS),
            "advisory_note": str (only if DEFAULTS or MULTIPLE_SOURCES_PRESENT)
        }
    """
    p3_summary = {}
    p4_calibration = {}
    extraction_path = None
    used_defaults = False
    
    # Build audit log (deterministic ordering: DIRECT, NESTED, FALLBACK, DEFAULTS)
    extraction_audit: List[Dict[str, Any]] = []
    found_paths_complete: List[str] = []  # Tracks complete sources (both P3 and P4)
    found_paths_partial: List[str] = []  # Tracks partial sources (only P3 or only P4)

    # PRECEDENCE 1: DIRECT - epistemic_summary/epistemic_calibration at top level
    direct_p3_found = "epistemic_summary" in cal_exp_report
    direct_p4_found = "epistemic_calibration" in cal_exp_report
    direct_found = direct_p3_found and direct_p4_found
    direct_fields = []
    if direct_p3_found:
        direct_fields.append("epistemic_summary")
    if direct_p4_found:
        direct_fields.append("epistemic_calibration")
    
    extraction_audit.append({
        "path": "DIRECT",
        "found": direct_found,
        "fields": direct_fields,
    })
    
    if direct_found:
        p3_summary = cal_exp_report["epistemic_summary"]
        p4_calibration = cal_exp_report["epistemic_calibration"]
        extraction_path = "DIRECT"
        found_paths_complete.append("DIRECT")
    elif direct_p3_found or direct_p4_found:
        found_paths_partial.append("DIRECT")
    
    # PRECEDENCE 2: NESTED - p3/p4 blocks
    nested_p3_found = (
        "p3" in cal_exp_report
        and isinstance(cal_exp_report["p3"], dict)
        and "epistemic_summary" in cal_exp_report["p3"]
    )
    nested_p4_found = (
        "p4" in cal_exp_report
        and isinstance(cal_exp_report["p4"], dict)
        and "epistemic_calibration" in cal_exp_report["p4"]
    )
    nested_found = nested_p3_found and nested_p4_found
    nested_fields = []
    if nested_p3_found:
        nested_fields.append("p3.epistemic_summary")
    if nested_p4_found:
        nested_fields.append("p4.epistemic_calibration")
    
    extraction_audit.append({
        "path": "NESTED",
        "found": nested_found,
        "fields": nested_fields,
    })
    
    if nested_found and not extraction_path:
        p3_summary = cal_exp_report["p3"]["epistemic_summary"]
        p4_calibration = cal_exp_report["p4"]["epistemic_calibration"]
        extraction_path = "NESTED"
    if nested_found:
        found_paths_complete.append("NESTED")
    elif nested_p3_found or nested_p4_found:
        found_paths_partial.append("NESTED")
    
    # PRECEDENCE 3: FALLBACK - alignment data
    fallback_p3_found = "epistemic_alignment_summary" in cal_exp_report
    fallback_p4_found = "epistemic_alignment" in cal_exp_report
    fallback_found = fallback_p3_found and fallback_p4_found
    fallback_fields = []
    if fallback_p3_found:
        fallback_fields.append("epistemic_alignment_summary")
    if fallback_p4_found:
        fallback_fields.append("epistemic_alignment")
    
    # Handle partial fallback
    if not extraction_path:
        if fallback_found:
            alignment_p3 = cal_exp_report["epistemic_alignment_summary"]
            alignment_p4 = cal_exp_report["epistemic_alignment"]
            p3_summary = {
                "drift_band": _derive_drift_band_from_alignment(alignment_p3),
                "mean_epistemic_risk": _derive_risk_from_alignment(alignment_p3),
            }
            p4_calibration = {
                "drift_band": _derive_drift_band_from_alignment(alignment_p4),
                "mean_risk": _derive_risk_from_alignment(alignment_p4),
            }
            extraction_path = "FALLBACK"
        if fallback_found:
            found_paths_complete.append("FALLBACK")
        elif (fallback_p3_found and not p3_summary) or (fallback_p4_found and not p4_calibration):
            found_paths_partial.append("FALLBACK")
        if not extraction_path:
            if fallback_p3_found and not p3_summary:
                alignment = cal_exp_report["epistemic_alignment_summary"]
                p3_summary = {
                    "drift_band": _derive_drift_band_from_alignment(alignment),
                    "mean_epistemic_risk": _derive_risk_from_alignment(alignment),
                }
                extraction_path = "FALLBACK"
            elif fallback_p4_found and not p4_calibration:
                alignment = cal_exp_report["epistemic_alignment"]
                p4_calibration = {
                    "drift_band": _derive_drift_band_from_alignment(alignment),
                    "mean_risk": _derive_risk_from_alignment(alignment),
                }
                extraction_path = "FALLBACK"
    
    extraction_audit.append({
        "path": "FALLBACK",
        "found": fallback_found or (fallback_p3_found and not p3_summary) or (fallback_p4_found and not p4_calibration),
        "fields": fallback_fields,
    })

    # PRECEDENCE 4: DEFAULTS - safe defaults if extraction failed
    defaults_used = False
    defaults_fields = []
    if not p3_summary:
        p3_summary = {"drift_band": "STABLE", "mean_epistemic_risk": "LOW"}
        used_defaults = True
        defaults_used = True
        defaults_fields.append("p3_defaults")
    if not p4_calibration:
        p4_calibration = {"drift_band": "STABLE", "mean_risk": "LOW"}
        used_defaults = True
        defaults_used = True
        defaults_fields.append("p4_defaults")
    
    extraction_audit.append({
        "path": "DEFAULTS",
        "found": defaults_used,
        "fields": defaults_fields,
    })
    
    if used_defaults:
        extraction_path = "DEFAULTS"
        # Don't add DEFAULTS to found_paths - it's not a "source", it's a fallback

    # Build footprint
    footprint = emit_cal_exp_epistemic_footprint(cal_id, p3_summary, p4_calibration)
    
    # Add extraction metadata
    footprint["extraction_path"] = extraction_path
    footprint["extraction_audit"] = extraction_audit
    footprint["extraction_audit_schema_version"] = EXTRACTION_AUDIT_SCHEMA_VERSION
    
    # Validate extraction_path is in allowed set
    if extraction_path not in EXTRACTION_PATHS:
        raise ValueError(f"Invalid extraction_path: {extraction_path}. Must be one of {sorted(EXTRACTION_PATHS)}")
    
    # If defaults were used, add confidence and advisory note
    advisory_notes = []
    if used_defaults:
        footprint["confidence"] = "LOW"
        advisory_notes.append("DEFAULTS_USED")
    
    # Strict mode: detect multiple sources (advisory only, no exceptions)
    if strict_extraction:
        if len(found_paths_complete) > 1:
            # Multiple complete sources found
            advisory_notes.append("MULTIPLE_SOURCES_PRESENT_COMPLETE")
        elif len(found_paths_complete) == 1 and len(found_paths_partial) > 0:
            # One complete source + partial sources
            advisory_notes.append("MULTIPLE_SOURCES_PRESENT_PARTIAL")
    
    # Combine advisory notes if any (deterministic ordering)
    if advisory_notes:
        footprint["advisory_note"] = "; ".join(sorted(advisory_notes))

    return footprint


def _derive_drift_band_from_alignment(alignment: Dict[str, Any]) -> str:
    """Derive drift_band from alignment data (safe default: STABLE)."""
    alignment_band = alignment.get("alignment_band", "MEDIUM")
    forecast_band = alignment.get("forecast_band", "MEDIUM")

    # Map alignment bands to drift bands
    if alignment_band == "HIGH" or forecast_band == "HIGH":
        return "DRIFTING"
    elif alignment_band == "LOW" and forecast_band == "LOW":
        return "STABLE"
    else:
        return "DRIFTING"  # Default to DRIFTING for MEDIUM


def _derive_risk_from_alignment(alignment: Dict[str, Any]) -> str:
    """Derive risk level from alignment data (safe default: LOW)."""
    alignment_band = alignment.get("alignment_band", "MEDIUM")
    status_light = alignment.get("status_light", "YELLOW")

    # Map to risk levels
    if status_light == "RED" or alignment_band == "HIGH":
        return "HIGH"
    elif status_light == "YELLOW" or alignment_band == "MEDIUM":
        return "MEDIUM"
    else:
        return "LOW"


def emit_cal_exp_epistemic_footprint(
    cal_id: str,
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Emit epistemic footprint for a calibration experiment (CAL-EXP-*).

    Creates a per-experiment footprint that can be persisted as
    `calibration/epistemic_footprint_<cal_id>.json`.

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        p3_summary: P3 epistemic summary from build_epistemic_summary_for_p3
        p4_calibration: P4 epistemic calibration from build_epistemic_calibration_for_p4

    Returns:
        {
            "schema_version": "1.0.0",
            "cal_id": str,
            "p3_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p4_drift_band": "STABLE" | "DRIFTING" | "VOLATILE",
            "p3_mean_risk": "LOW" | "MEDIUM" | "HIGH",
            "p4_mean_risk": "LOW" | "MEDIUM" | "HIGH",
        }
    """
    footprint = build_first_light_epistemic_footprint(p3_summary, p4_calibration)
    footprint["schema_version"] = CAL_EXP_FOOTPRINT_SCHEMA_VERSION
    footprint["cal_id"] = cal_id
    return footprint


def build_epistemic_calibration_panel(
    footprints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build multi-experiment epistemic calibration panel from footprints.

    Aggregates multiple calibration experiment footprints into a panel summary
    for triage and governance review.

    Args:
        footprints: List of calibration experiment footprints from emit_cal_exp_epistemic_footprint

    Returns:
        {
            "schema_version": "1.0.0",
            "num_experiments": int,
            "num_conservative": int,  # P3 > P4
            "num_divergent": int,  # P4 > P3
            "num_high_risk_both": int,  # Both HIGH risk
            "dominant_pattern": "CONSERVATIVE" | "CONVERGENT" | "DIVERGENT" | "HIGH_RISK_BOTH" | "MIXED",
            "dominant_pattern_confidence": float,  # 0.0-1.0, margin between top and runner-up
            "experiments": List[Dict],  # Original footprints
        }
    """
    if not footprints:
        return {
            "schema_version": CALIBRATION_PANEL_SCHEMA_VERSION,
            "num_experiments": 0,
            "num_conservative": 0,
            "num_divergent": 0,
            "num_high_risk_both": 0,
            "dominant_pattern": "MIXED",
            "dominant_pattern_confidence": 0.0,
            "experiments": [],
        }

    # Classify experiments
    num_conservative = 0
    num_divergent = 0
    num_high_risk_both = 0

    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    drift_map = {"STABLE": 0, "DRIFTING": 1, "VOLATILE": 2}

    for footprint in footprints:
        p3_risk = risk_map.get(footprint.get("p3_mean_risk", "LOW"), 0)
        p4_risk = risk_map.get(footprint.get("p4_mean_risk", "LOW"), 0)
        p3_drift = drift_map.get(footprint.get("p3_drift_band", "STABLE"), 0)
        p4_drift = drift_map.get(footprint.get("p4_drift_band", "STABLE"), 0)

        # High risk in both (check first, independent of conservative/divergent)
        if p3_risk == 2 and p4_risk == 2:  # Both HIGH
            num_high_risk_both += 1

        # Conservative: P3 shows higher risk than P4 (risk takes precedence)
        if p3_risk > p4_risk:
            num_conservative += 1
        # If risks equal, use drift as tiebreaker (but only if not both HIGH)
        elif p3_risk == p4_risk and p3_risk < 2 and p3_drift > p4_drift:
            num_conservative += 1

        # Divergent: P4 shows higher risk than P3 (risk takes precedence)
        if p4_risk > p3_risk:
            num_divergent += 1
        # If risks equal, use drift as tiebreaker (but only if not both HIGH)
        elif p4_risk == p3_risk and p4_risk < 2 and p4_drift > p3_drift:
            num_divergent += 1

    # Determine dominant pattern and confidence
    dominant_pattern, pattern_confidence = _determine_dominant_pattern_with_confidence(
        num_experiments=len(footprints),
        num_conservative=num_conservative,
        num_divergent=num_divergent,
        num_high_risk_both=num_high_risk_both,
    )

    return {
        "schema_version": CALIBRATION_PANEL_SCHEMA_VERSION,
        "num_experiments": len(footprints),
        "num_conservative": num_conservative,
        "num_divergent": num_divergent,
        "num_high_risk_both": num_high_risk_both,
        "dominant_pattern": dominant_pattern,
        "dominant_pattern_confidence": round(pattern_confidence, 3),
        "experiments": footprints,
    }


def _determine_dominant_pattern_with_confidence(
    num_experiments: int,
    num_conservative: int,
    num_divergent: int,
    num_high_risk_both: int,
) -> Tuple[str, float]:
    """
    Determine dominant pattern from panel counts with confidence score.

    Confidence is based on the margin between the top pattern count and the runner-up.
    Higher margin = higher confidence.

    Returns:
        Tuple of (pattern, confidence) where:
        - pattern: "CONSERVATIVE" | "CONVERGENT" | "DIVERGENT" | "HIGH_RISK_BOTH" | "MIXED"
        - confidence: float in [0.0, 1.0], margin between top and runner-up
    """
    if num_experiments == 0:
        return ("MIXED", 0.0)

    # Calculate percentages
    conservative_pct = num_conservative / num_experiments if num_experiments > 0 else 0.0
    divergent_pct = num_divergent / num_experiments if num_experiments > 0 else 0.0
    high_risk_pct = num_high_risk_both / num_experiments if num_experiments > 0 else 0.0

    # Collect all pattern counts for margin calculation
    pattern_counts = {
        "CONSERVATIVE": conservative_pct,
        "DIVERGENT": divergent_pct,
        "HIGH_RISK_BOTH": high_risk_pct,
        "CONVERGENT": 1.0 - conservative_pct - divergent_pct - high_risk_pct,  # Remaining
    }

    # Find top pattern and runner-up
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    top_pattern, top_pct = sorted_patterns[0]
    runner_up_pct = sorted_patterns[1][1] if len(sorted_patterns) > 1 else 0.0

    # Calculate confidence as margin (clamped to [0.0, 1.0])
    margin = top_pct - runner_up_pct
    confidence = max(0.0, min(1.0, margin))

    # High risk both takes precedence if present and >= 50%
    if num_high_risk_both > 0 and num_high_risk_both >= num_experiments / 2:
        return ("HIGH_RISK_BOTH", confidence)

    # Conservative dominant
    if conservative_pct > 0.5 and conservative_pct > divergent_pct:
        return ("CONSERVATIVE", confidence)

    # Divergent dominant
    if divergent_pct > 0.5 and divergent_pct > conservative_pct:
        return ("DIVERGENT", confidence)

    # Convergent (both low, similar)
    if conservative_pct < 0.3 and divergent_pct < 0.3:
        return ("CONVERGENT", confidence)

    # Mixed (no clear pattern)
    return ("MIXED", confidence)


__all__ = [
    "EPISTEMIC_SUMMARY_SCHEMA_VERSION",
    "EPISTEMIC_CALIBRATION_SCHEMA_VERSION",
    "EPISTEMIC_EVIDENCE_SCHEMA_VERSION",
    "FIRST_LIGHT_FOOTPRINT_SCHEMA_VERSION",
    "CAL_EXP_FOOTPRINT_SCHEMA_VERSION",
    "CALIBRATION_PANEL_SCHEMA_VERSION",
    "build_epistemic_summary_for_p3",
    "build_epistemic_calibration_for_p4",
    "build_first_light_epistemic_footprint",
    "emit_cal_exp_epistemic_footprint",
    "emit_epistemic_footprint_from_cal_exp_report",
    "build_epistemic_calibration_panel",
    "attach_epistemic_drift_to_evidence",
    "attach_epistemic_calibration_panel_to_evidence",
]

