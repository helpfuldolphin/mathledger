#!/usr/bin/env python3
"""
Taxonomy Governance & CI Integration - Phase V

Provides:
- Taxonomy integrity radar
- Global console tile
- CI blocking evaluation
- Drift timeline analysis
- CLI commands for CI integration
- First Light curriculum coherence summary (Phase X)

Curriculum Coherence Summary:
----------------------------
The First Light curriculum coherence summary combines P3 (synthetic) and P4
(real-runner shadow) taxonomy observations into a single alignment witness.

Reading the Summary:
- alignment_score (0.0-1.0): Overall taxonomy alignment across metrics, docs,
  and curriculum. 1.0 = perfect alignment, <0.5 = significant misalignment.
- integrity_status ("OK" | "WARN" | "BLOCK"): Current taxonomy integrity state.
  BLOCK indicates curriculum slices reference removed types (critical).
- drift_band ("STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT"): Historical
  taxonomy change intensity. STABLE = no changes, HIGH_DRIFT = frequent breaking changes.
- projected_horizon (0.0-1.0): Extrapolated change intensity forward. Higher values
  indicate anticipated taxonomy instability.

Interpretation:
- High alignment_score + STABLE drift_band = curriculum philosophically consistent
  between synthetic and shadow modes.
- Low alignment_score or HIGH_DRIFT = taxonomy changes may have broken curriculum
  coherence; review docs_impacted list.
- BLOCK integrity_status = curriculum slices must be updated before proceeding.

SHADOW MODE: This summary is an alignment witness, not an automatic blocker.
It provides observability into curriculum coherence but does not influence
P3/P4 execution behavior.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Phase V: Semantic Integrity Grid
# ---------------------------------------------------------------------------

def build_taxonomy_integrity_radar(
    metrics_impact: Dict[str, Any],
    docs_alignment: Dict[str, Any],
    curriculum_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the Taxonomy Integrity Radar - unified view of taxonomy health.
    
    Aggregates metrics, docs, and curriculum alignment into a single
    integrity status with alignment score.
    
    Args:
        metrics_impact: Output from analyze_taxonomy_impact_on_metrics()
            Expected keys: "affected_metric_kinds" (List[str]), "status" (str)
        docs_alignment: Output from analyze_taxonomy_alignment_with_docs_and_curriculum()
            Expected keys: "missing_doc_updates" (List[str]), "alignment_status" (str)
        curriculum_alignment: Same structure as docs_alignment
            Expected keys: "slices_with_outdated_types" (List[str]), "alignment_status" (str)
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - integrity_status: "OK" | "WARN" | "BLOCK"
        - alignment_score: Float 0.0-1.0 (1.0 = perfect alignment)
        - metrics_impacted: List[str]
        - docs_impacted: List[str]
        - curriculum_impacted: List[str]
    """
    # Extract affected items
    affected_metrics = metrics_impact.get("affected_metric_kinds", [])
    affected_docs = docs_alignment.get("missing_doc_updates", [])
    affected_curriculum = curriculum_alignment.get("slices_with_outdated_types", [])
    
    # Determine integrity status
    # BLOCK: Curriculum slices affected (critical - breaks runtime)
    # WARN: Docs or metrics out of date (non-critical but needs attention)
    # OK: Everything aligned
    if affected_curriculum:
        integrity_status = "BLOCK"
    elif affected_docs or affected_metrics or metrics_impact.get("status") != "OK":
        integrity_status = "WARN"
    else:
        integrity_status = "OK"
    
    # Calculate alignment score (0.0 = misaligned, 1.0 = perfect)
    # Score components:
    # - Metrics: 0.3 weight (OK=1.0, PARTIAL=0.5, MISALIGNED=0.0)
    # - Docs: 0.3 weight (ALIGNED=1.0, PARTIAL=0.5, OUT_OF_DATE=0.0)
    # - Curriculum: 0.4 weight (no affected slices=1.0, affected=0.0)
    
    metrics_status = metrics_impact.get("status", "OK")
    docs_status = docs_alignment.get("alignment_status", "ALIGNED")
    curriculum_status = curriculum_alignment.get("alignment_status", "ALIGNED")
    
    metrics_score = {
        "OK": 1.0,
        "PARTIAL": 0.5,
        "MISALIGNED": 0.0,
    }.get(metrics_status, 0.0)
    
    docs_score = {
        "ALIGNED": 1.0,
        "PARTIAL": 0.5,
        "OUT_OF_DATE": 0.0,
    }.get(docs_status, 0.0)
    
    curriculum_score = {
        "ALIGNED": 1.0,
        "PARTIAL": 0.5,
        "OUT_OF_DATE": 0.0,
    }.get(curriculum_status, 1.0 if not affected_curriculum else 0.0)
    
    alignment_score = (
        metrics_score * 0.3 +
        docs_score * 0.3 +
        curriculum_score * 0.4
    )
    
    return {
        "schema_version": "1.0.0",
        "integrity_status": integrity_status,
        "alignment_score": round(alignment_score, 3),
        "metrics_impacted": affected_metrics,
        "docs_impacted": affected_docs[:10],  # Limit to 10 for readability
        "curriculum_impacted": affected_curriculum,
    }


def build_global_console_tile(
    radar: Dict[str, Any],
    risk_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a Global Console Tile for high-level dashboard view.
    
    Provides executive-level summary of taxonomy health.
    
    Args:
        radar: Output from build_taxonomy_integrity_radar()
        risk_analysis: Output from analyze_taxonomy_change() (as dict)
            Expected keys: "risk_level" (str), "breaking_changes" (List)
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - critical_breaks_count: int
        - headline: str (neutral summary)
    """
    integrity_status = radar.get("integrity_status", "OK")
    alignment_score = radar.get("alignment_score", 1.0)
    affected_curriculum = radar.get("curriculum_impacted", [])
    
    # Determine status_light
    if integrity_status == "BLOCK" or alignment_score < 0.5:
        status_light = "RED"
    elif integrity_status == "WARN" or alignment_score < 0.8:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Count critical breaks (curriculum slices are critical)
    critical_breaks_count = len(affected_curriculum)
    
    # Build headline (neutral, descriptive language only)
    if critical_breaks_count > 0:
        headline = f"Taxonomy integrity: {critical_breaks_count} critical break(s) detected"
    elif integrity_status == "WARN":
        headline = "Taxonomy integrity: warnings present, alignment partial"
    else:
        headline = "Taxonomy integrity: all systems aligned"
    
    return {
        "schema_version": "1.0.0",
        "status_light": status_light,
        "critical_breaks_count": critical_breaks_count,
        "headline": headline,
    }


def evaluate_taxonomy_for_ci(radar: Dict[str, Any]) -> Tuple[int, str]:
    """
    Evaluate taxonomy integrity for CI blocking.
    
    Exit code rules:
    - BLOCK (exit 1): Any curriculum slice referencing removed types
    - WARN (exit 0 with message): Docs or metrics out of date
    - OK (exit 0): Everything aligned
    
    Args:
        radar: Output from build_taxonomy_integrity_radar()
        
    Returns:
        Tuple of (exit_code, message)
        - exit_code: 0 (OK/WARN) or 1 (BLOCK)
        - message: Human-readable status message
    """
    integrity_status = radar.get("integrity_status", "OK")
    affected_curriculum = radar.get("curriculum_impacted", [])
    affected_metrics = radar.get("metrics_impacted", [])
    affected_docs = radar.get("docs_impacted", [])
    
    if integrity_status == "BLOCK":
        # BLOCK: Curriculum slices affected
        slice_names = ", ".join(affected_curriculum[:5])  # Limit to 5 for readability
        if len(affected_curriculum) > 5:
            slice_names += f" (and {len(affected_curriculum) - 5} more)"
        message = (
            f"BLOCK: {len(affected_curriculum)} curriculum slice(s) reference removed types: {slice_names}. "
            f"Update curriculum.yaml before proceeding."
        )
        return 1, message
    
    elif integrity_status == "WARN":
        # WARN: Docs or metrics out of date
        warn_parts = []
        if affected_metrics:
            warn_parts.append(f"{len(affected_metrics)} metric(s) affected")
        if affected_docs:
            warn_parts.append(f"{len(affected_docs)} doc location(s) need updates")
        
        message = f"WARN: {', '.join(warn_parts)}. Review and update as needed."
        return 0, message
    
    else:
        # OK: Everything aligned
        message = "OK: Taxonomy integrity maintained across all systems."
        return 0, message


def build_taxonomy_drift_timeline(
    historical_impacts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate Taxonomy Drift Timeline from historical impact objects.
    
    Analyzes a sequence of taxonomy changes to identify drift patterns.
    
    Args:
        historical_impacts: List of impact analysis dictionaries (from analyze_taxonomy_change)
            Expected keys: "breaking_changes" (List), "non_breaking_changes" (List)
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT"
        - change_intensity: Float 0.0-1.0 (cumulative change intensity)
        - first_break_index: Optional[int] (index of first breaking change, or None if none)
    """
    if not historical_impacts:
        return {
            "schema_version": "1.0.0",
            "drift_band": "STABLE",
            "change_intensity": 0.0,
            "first_break_index": None,
        }
    
    # Analyze historical impacts
    total_changes = 0
    breaking_changes = 0
    first_break_index = None
    
    for i, impact in enumerate(historical_impacts):
        breaking_count = len(impact.get("breaking_changes", []))
        non_breaking_count = len(impact.get("non_breaking_changes", []))
        
        total_changes += breaking_count + non_breaking_count
        breaking_changes += breaking_count
        
        if breaking_count > 0 and first_break_index is None:
            first_break_index = i
    
    # Calculate change intensity
    # Intensity = (breaking_changes * 2 + non_breaking_changes) / (total_impacts * max_expected_changes)
    # Normalize to 0.0-1.0 range
    non_breaking_changes = total_changes - breaking_changes
    max_expected_changes = len(historical_impacts) * 10  # Assume max 10 changes per impact
    
    if max_expected_changes > 0:
        change_intensity = min(1.0, (breaking_changes * 2 + non_breaking_changes) / max_expected_changes)
    else:
        change_intensity = 0.0
    
    # Determine drift band
    if change_intensity == 0.0:
        drift_band = "STABLE"
    elif change_intensity < 0.2:
        drift_band = "LOW_DRIFT"
    elif change_intensity < 0.5:
        drift_band = "MEDIUM_DRIFT"
    else:
        drift_band = "HIGH_DRIFT"
    
    return {
        "schema_version": "1.0.0",
        "drift_band": drift_band,
        "change_intensity": round(change_intensity, 3),
        "first_break_index": first_break_index,
    }


# ---------------------------------------------------------------------------
# Evidence Pack Integration
# ---------------------------------------------------------------------------

def attach_taxonomy_to_evidence(
    evidence: Dict[str, Any],
    radar: Dict[str, Any],
    tile: Dict[str, Any],
    drift_timeline: Optional[Dict[str, Any]] = None,
    curriculum_coherence_summary: Optional[Dict[str, Any]] = None,
    p3_taxonomy_summary: Optional[Dict[str, Any]] = None,
    p4_taxonomy_calibration: Optional[Dict[str, Any]] = None,
    coherence_crosscheck: Optional[Dict[str, Any]] = None,
    curriculum_coherence_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach taxonomy integrity data to an evidence pack.
    
    This is a read-only, additive operation. Returns a new dict with
    taxonomy data attached. Does not modify the input evidence dict.
    
    SHADOW MODE: Taxonomy data is observational only and does not
    influence evidence pack validation or processing.
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        radar: Taxonomy integrity radar from build_taxonomy_integrity_radar().
        tile: Taxonomy console tile from build_global_console_tile().
        drift_timeline: Optional drift timeline from build_taxonomy_drift_timeline().
        curriculum_coherence_summary: Optional First Light curriculum coherence summary
            from build_first_light_curriculum_coherence_summary().
        p3_taxonomy_summary: Optional P3 taxonomy summary from build_p3_taxonomy_summary().
            If provided along with p4_taxonomy_calibration, will automatically build
            and attach first_light_curriculum_coherence_tile.
        p4_taxonomy_calibration: Optional P4 taxonomy calibration from build_p4_taxonomy_calibration().
            If provided along with p3_taxonomy_summary, will automatically build
            and attach first_light_curriculum_coherence_tile.
        coherence_crosscheck: Optional coherence vs curriculum governance cross-check
            from summarize_coherence_vs_curriculum_governance().
        curriculum_coherence_panel: Optional curriculum coherence panel from
            build_curriculum_coherence_panel().
        
    Returns:
        New dict with evidence contents plus taxonomy data attached under
        "governance.taxonomy" key.
    """
    # Create a copy to avoid mutation
    result = dict(evidence)
    
    # Initialize governance section if needed
    if "governance" not in result:
        result["governance"] = {}
    
    # Attach taxonomy data
    taxonomy_data = {
        "schema_version": "1.0.0",
        "radar": radar,
        "tile": tile,
    }
    
    if drift_timeline is not None:
        taxonomy_data["drift_timeline"] = drift_timeline
    
    # Extract key breakpoints from radar
    if radar.get("curriculum_impacted"):
        taxonomy_data["key_breakpoints"] = {
            "curriculum_slices_affected": radar["curriculum_impacted"],
            "critical_breaks_count": len(radar["curriculum_impacted"]),
        }
    
    # Attach curriculum coherence summary if provided
    if curriculum_coherence_summary is not None:
        taxonomy_data["curriculum_coherence_summary"] = curriculum_coherence_summary
    
    # Phase X: Automatically build and attach first_light_curriculum_coherence_tile
    # if both P3 and P4 summaries are provided
    if p3_taxonomy_summary is not None and p4_taxonomy_calibration is not None:
        coherence_tile = build_first_light_curriculum_coherence_tile(
            p3_taxonomy_summary=p3_taxonomy_summary,
            p4_taxonomy_calibration=p4_taxonomy_calibration,
        )
        taxonomy_data["first_light_curriculum_coherence"] = coherence_tile
    
    # Phase X: Attach coherence vs curriculum governance cross-check if provided
    if coherence_crosscheck is not None:
        if "governance" not in result:
            result["governance"] = {}
        result["governance"]["curriculum_coherence_crosscheck"] = coherence_crosscheck
    
    # Phase X: Attach curriculum coherence panel if provided
    if curriculum_coherence_panel is not None:
        if "governance" not in result:
            result["governance"] = {}
        result["governance"]["curriculum_coherence_panel"] = curriculum_coherence_panel
    
    result["governance"]["taxonomy"] = taxonomy_data
    
    return result


# ---------------------------------------------------------------------------
# P3/P4 Integration Helpers
# ---------------------------------------------------------------------------

def build_p3_taxonomy_summary(
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build taxonomy summary for P3 first_light_stability_report.json.
    
    Adds taxonomy_summary section with alignment score and docs_impacted.
    
    Args:
        radar: Taxonomy integrity radar from build_taxonomy_integrity_radar().
        
    Returns:
        Dictionary with:
        - alignment_score: float (0.0-1.0)
        - docs_impacted: List[str]
        - integrity_status: "OK" | "WARN" | "BLOCK"
    """
    return {
        "alignment_score": radar.get("alignment_score", 1.0),
        "docs_impacted": radar.get("docs_impacted", []),
        "integrity_status": radar.get("integrity_status", "OK"),
    }


def build_p4_taxonomy_calibration(
    drift_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build taxonomy calibration data for P4 p4_calibration_report.json.
    
    Adds taxonomy_calibration section with drift band, projected horizon,
    and critical break index.
    
    Args:
        drift_timeline: Taxonomy drift timeline from build_taxonomy_drift_timeline().
        
    Returns:
        Dictionary with:
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT"
        - projected_horizon: float (change_intensity projected forward)
        - critical_break_index: Optional[int] (first_break_index from timeline)
    """
    change_intensity = drift_timeline.get("change_intensity", 0.0)
    
    # Projected horizon: extrapolate change_intensity forward
    # Simple linear projection: assume same rate continues
    projected_horizon = min(1.0, change_intensity * 1.5)  # Conservative projection
    
    return {
        "drift_band": drift_timeline.get("drift_band", "STABLE"),
        "projected_horizon": round(projected_horizon, 3),
        "critical_break_index": drift_timeline.get("first_break_index"),
    }


def build_first_light_curriculum_coherence_summary(
    p3_taxonomy_summary: Dict[str, Any],
    p4_taxonomy_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First Light curriculum coherence summary combining P3 + P4 views.
    
    This provides a unified view of taxonomy integrity across both P3 (synthetic)
    and P4 (real-runner shadow) experiments, serving as a coherence witness
    for curriculum alignment.
    
    Reading the Summary Together:
    -----------------------------
    The four key fields should be interpreted together:
    
    1. alignment_score + integrity_status: Current taxonomy health snapshot.
       - alignment_score 1.0 + integrity_status "OK" = perfect alignment
       - alignment_score <0.5 or integrity_status "BLOCK" = critical issues
    
    2. drift_band + projected_horizon: Historical and projected taxonomy stability.
       - drift_band "STABLE" + projected_horizon 0.0 = no taxonomy changes expected
       - drift_band "HIGH_DRIFT" + projected_horizon >0.5 = ongoing instability
    
    3. Combined interpretation:
       - High alignment + STABLE drift = curriculum philosophically consistent
         between synthetic (P3) and shadow (P4) modes
       - Low alignment or HIGH_DRIFT = taxonomy changes may have broken curriculum
         coherence; requires review
    
    SHADOW MODE: This is evidence-only, not a gate. It provides observability
    into curriculum coherence but does not influence P3/P4 behavior. External
    reviewers use this summary to answer: "Did the curriculum stay philosophically
    consistent between synthetic and shadow modes?"
    
    Args:
        p3_taxonomy_summary: Taxonomy summary from build_p3_taxonomy_summary().
        p4_taxonomy_calibration: Taxonomy calibration from build_p4_taxonomy_calibration().
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - alignment_score: float (from P3, 0.0-1.0)
        - integrity_status: "OK" | "WARN" | "BLOCK" (from P3)
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT" (from P4)
        - projected_horizon: float (from P4, 0.0-1.0)
    """
    return {
        "schema_version": "1.0.0",
        "alignment_score": p3_taxonomy_summary.get("alignment_score", 1.0),
        "integrity_status": p3_taxonomy_summary.get("integrity_status", "OK"),
        "drift_band": p4_taxonomy_calibration.get("drift_band", "STABLE"),
        "projected_horizon": p4_taxonomy_calibration.get("projected_horizon", 0.0),
    }


def build_first_light_curriculum_coherence_tile(
    p3_taxonomy_summary: Dict[str, Any],
    p4_taxonomy_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First Light curriculum coherence tile for evidence pack.
    
    STATUS: PHASE X — FIRST LIGHT CURRICULUM COHERENCE
    
    Collapses taxonomy radar/tile into a one-page curriculum coherence summary
    for First Light evidence packs. This provides a unified view of taxonomy
    integrity across both P3 (synthetic) and P4 (real-runner shadow) experiments.
    
    SHADOW MODE: This is evidence-only, not a gate. It provides observability
    into curriculum coherence but does not influence P3/P4 behavior.
    
    Args:
        p3_taxonomy_summary: Taxonomy summary from build_p3_taxonomy_summary().
            Expected keys: alignment_score, integrity_status, docs_impacted
        p4_taxonomy_calibration: Taxonomy calibration from build_p4_taxonomy_calibration().
            Expected keys: drift_band, projected_horizon
    
    Returns:
        Evidence-ready summary dictionary with:
        - schema_version: "1.0.0"
        - alignment_score: float (from P3)
        - integrity_status: "OK" | "WARN" | "BLOCK" (from P3)
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT" (from P4)
        - projected_horizon: float (from P4)
        - docs_impacted: List[str] (from P3)
    """
    return {
        "schema_version": "1.0.0",
        "alignment_score": p3_taxonomy_summary.get("alignment_score", 1.0),
        "integrity_status": p3_taxonomy_summary.get("integrity_status", "OK"),
        "drift_band": p4_taxonomy_calibration.get("drift_band", "STABLE"),
        "projected_horizon": p4_taxonomy_calibration.get("projected_horizon", 0.0),
        "docs_impacted": sorted(p3_taxonomy_summary.get("docs_impacted", [])),  # Sorted for determinism
    }


# ---------------------------------------------------------------------------
# Phase X: Curriculum Coherence Calibration Panel (P5)
# ---------------------------------------------------------------------------

def build_cal_exp_curriculum_coherence_snapshot(
    cal_id: str,
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a curriculum coherence snapshot for a single calibration experiment.
    
    STATUS: PHASE X — P5 CALIBRATION EXPERIMENT SNAPSHOT
    
    Extracts key coherence fields from a curriculum coherence tile and creates
    a per-experiment snapshot suitable for aggregation across CAL-EXP-1/2/3.
    
    SHADOW MODE: This is evidence-only, not a gate. Snapshots are observational
    and do not influence calibration experiment behavior.
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2").
        tile: Curriculum coherence tile from build_first_light_curriculum_coherence_tile().
            Expected keys: alignment_score, integrity_status, drift_band, projected_horizon
    
    Returns:
        Snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str (experiment identifier)
        - alignment_score: float (0.0-1.0)
        - integrity_status: "OK" | "WARN" | "BLOCK"
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT"
        - projected_horizon: float (0.0-1.0)
    """
    return {
        "schema_version": "1.0.0",
        "cal_id": cal_id,
        "alignment_score": round(tile.get("alignment_score", 1.0), 3),  # Round for determinism
        "integrity_status": tile.get("integrity_status", "OK"),
        "drift_band": tile.get("drift_band", "STABLE"),
        "projected_horizon": round(tile.get("projected_horizon", 0.0), 3),  # Round for determinism
    }


def persist_curriculum_coherence_snapshot(
    snapshot: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist a curriculum coherence snapshot to disk.
    
    STATUS: PHASE X — P5 CALIBRATION SNAPSHOT PERSISTENCE
    
    Writes a snapshot to calibration/curriculum_coherence_<cal_id>.json.
    Creates the output directory if it doesn't exist.
    
    SHADOW MODE: This is a write-only operation for recording snapshots.
    It does not affect any governance state.
    
    Args:
        snapshot: Snapshot dictionary from build_cal_exp_curriculum_coherence_snapshot().
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration")).
    
    Returns:
        Path to the written snapshot file.
    
    Raises:
        IOError: If the file cannot be written.
    """
    cal_id = snapshot.get("cal_id", "UNKNOWN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_path = output_dir / f"curriculum_coherence_{cal_id}.json"
    
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    
    return snapshot_path


def curriculum_coherence_panel_for_alignment_view(
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert curriculum coherence panel to GGFL alignment view format.
    
    STATUS: PHASE X — P5 CALIBRATION GGFL ADAPTER
    
    Normalizes curriculum coherence panel into Global Governance Fusion Layer
    (GGFL) unified format for cross-subsystem alignment views.
    
    SHADOW MODE: This is read-only, observational, and does not influence
    governance decisions. The output is advisory only.
    
    Args:
        panel: Curriculum coherence panel from build_curriculum_coherence_panel().
            Expected keys: num_block, num_high_drift, num_experiments, median_alignment_score
    
    Returns:
        GGFL-normalized dict with stable output contract:
        - signal_type: "SIG-CURR" (constant)
        - status: "ok" | "warn" (lowercase)
        - conflict: False (constant, curriculum coherence never conflicts)
        - drivers: List[str] (max 3, deterministic ordering: blocks first, then high drift, then median)
        - summary: str (single neutral sentence)
    """
    num_block = panel.get("num_block", 0)
    num_high_drift = panel.get("num_high_drift", 0)
    num_experiments = panel.get("num_experiments", 0)
    median_alignment_score = panel.get("median_alignment_score", 0.0)
    
    # Determine status: ok if no blocks and no high drift, warn otherwise
    if num_block == 0 and num_high_drift == 0:
        status = "ok"
    else:
        status = "warn"
    
    # Extract drivers with deterministic ordering and prefixed reason codes:
    # 1. Blocks first (DRIVER_BLOCK_COUNT)
    # 2. High drift second (DRIVER_HIGH_DRIFT_COUNT)
    # 3. Median alignment third (DRIVER_MEDIAN_BELOW_THRESHOLD, only if status is warn and below threshold)
    # ENUM LOCK: All drivers must use prefixed reason codes; no freeform text allowed
    drivers: List[str] = []
    if num_block > 0:
        drivers.append(f"DRIVER_BLOCK_COUNT:{num_block}")
    if num_high_drift > 0:
        drivers.append(f"DRIVER_HIGH_DRIFT_COUNT:{num_high_drift}")
    # Add median alignment as third driver only if status is warn and below threshold
    if status == "warn" and len(drivers) < 3 and median_alignment_score < 0.7:
        drivers.append(f"DRIVER_MEDIAN_BELOW_THRESHOLD:{median_alignment_score:.3f}")
    
    # Limit to top 3 drivers (already enforced by logic above, but explicit for safety)
    drivers = drivers[:3]
    
    # ENUM LOCK: Verify all drivers use prefixed reason codes (defensive check)
    allowed_prefixes = ("DRIVER_BLOCK_COUNT:", "DRIVER_HIGH_DRIFT_COUNT:", "DRIVER_MEDIAN_BELOW_THRESHOLD:")
    for driver in drivers:
        if not any(driver.startswith(prefix) for prefix in allowed_prefixes):
            # This should never happen, but defensive check ensures enum lock
            raise ValueError(f"Driver must use prefixed reason code, got: {driver}")
    
    # Generate neutral summary sentence
    if num_experiments == 0:
        summary = "No calibration experiments in curriculum coherence panel"
    elif status == "ok":
        summary = f"Curriculum coherence panel: {num_experiments} experiment(s), all aligned"
    elif num_block > 0 and num_high_drift > 0:
        summary = f"Curriculum coherence panel: {num_experiments} experiment(s), {num_block} with BLOCK status, {num_high_drift} with HIGH_DRIFT"
    elif num_block > 0:
        summary = f"Curriculum coherence panel: {num_experiments} experiment(s), {num_block} with BLOCK status"
    elif num_high_drift > 0:
        summary = f"Curriculum coherence panel: {num_experiments} experiment(s), {num_high_drift} with HIGH_DRIFT"
    else:
        summary = f"Curriculum coherence panel: {num_experiments} experiment(s)"
    
    return {
        "signal_type": "SIG-CURR",  # Constant signal type identifier
        "status": status,
        "conflict": False,  # Constant: curriculum coherence never conflicts (hard-coded invariant)
        "drivers": drivers,  # Prefixed reason codes: DRIVER_BLOCK_COUNT, DRIVER_HIGH_DRIFT_COUNT, DRIVER_MEDIAN_BELOW_THRESHOLD
        "summary": summary,  # Single neutral sentence
        "shadow_mode_invariants": {
            "advisory_only": True,  # This signal is advisory only, not a gate
            "no_enforcement": True,  # No enforcement logic depends on this signal
            "conflict_invariant": True,  # Conflict is always False (hard-coded invariant)
        },
    }


def build_curriculum_coherence_panel(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a multi-experiment curriculum coherence panel from snapshots.
    
    STATUS: PHASE X — P5 CALIBRATION COHERENCE PANEL
    
    Aggregates curriculum coherence snapshots across multiple calibration
    experiments (CAL-EXP-1/2/3) into a single panel for cross-experiment
    alignment witness.
    
    SHADOW MODE: This is evidence-only, not a gate. The panel is observational
    and does not influence calibration experiment behavior.
    
    Args:
        snapshots: List of snapshot dictionaries from build_cal_exp_curriculum_coherence_snapshot().
            Each snapshot must contain: cal_id, alignment_score, integrity_status, drift_band.
    
    Returns:
        Panel dictionary with:
        - schema_version: "1.0.0"
        - num_experiments: int (number of snapshots)
        - num_ok: int (count of snapshots with integrity_status == "OK")
        - num_warn: int (count of snapshots with integrity_status == "WARN")
        - num_block: int (count of snapshots with integrity_status == "BLOCK")
        - num_high_drift: int (count of snapshots with drift_band == "HIGH_DRIFT")
        - median_alignment_score: float (median alignment_score across snapshots)
    """
    if not snapshots:
        return {
            "schema_version": "1.0.0",
            "num_experiments": 0,
            "num_ok": 0,
            "num_warn": 0,
            "num_block": 0,
            "num_high_drift": 0,
            "median_alignment_score": 0.0,
        }
    
    # Count integrity statuses
    num_ok = sum(1 for s in snapshots if s.get("integrity_status") == "OK")
    num_warn = sum(1 for s in snapshots if s.get("integrity_status") == "WARN")
    num_block = sum(1 for s in snapshots if s.get("integrity_status") == "BLOCK")
    
    # Count high drift
    num_high_drift = sum(1 for s in snapshots if s.get("drift_band") == "HIGH_DRIFT")
    
    # Compute median alignment score
    alignment_scores = [s.get("alignment_score", 0.0) for s in snapshots]
    alignment_scores.sort()  # Sort for median calculation
    n = len(alignment_scores)
    if n % 2 == 0:
        median_alignment_score = (alignment_scores[n // 2 - 1] + alignment_scores[n // 2]) / 2.0
    else:
        median_alignment_score = alignment_scores[n // 2]
    median_alignment_score = round(median_alignment_score, 3)  # Round for determinism
    
    return {
        "schema_version": "1.0.0",
        "num_experiments": len(snapshots),
        "num_ok": num_ok,
        "num_warn": num_warn,
        "num_block": num_block,
        "num_high_drift": num_high_drift,
        "median_alignment_score": median_alignment_score,
    }


# ---------------------------------------------------------------------------
# Phase X: Curriculum Coherence Time-Series & Alignment Dashboard
# ---------------------------------------------------------------------------

def build_curriculum_coherence_timeseries(
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build curriculum coherence time-series from a sequence of coherence summaries.
    
    STATUS: PHASE X — ALIGNMENT TREND LINE FOR CAL-EXP-*
    
    Extracts time-series data from curriculum coherence summaries for dashboard
    visualization. Used in P5 (CAL-EXP-2, 1000 cycles) to track alignment trends
    over time.
    
    The time-series provides a visible "Alignment Trend Line" showing how
    curriculum coherence evolves across cycles or runs.
    
    Args:
        summaries: List of curriculum coherence summaries from
            build_first_light_curriculum_coherence_summary().
            Each summary should have: alignment_score, drift_band, and optionally
            cycle_or_run_idx. If cycle_or_run_idx is missing, indices are assigned
            sequentially starting from 0.
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - points: List of time-series points, each with:
          - cycle_or_run_idx: int (monotone increasing)
          - alignment_score: float (0.0-1.0)
          - drift_band: str (STABLE | LOW_DRIFT | MEDIUM_DRIFT | HIGH_DRIFT)
    """
    if not summaries:
        return {
            "schema_version": "1.0.0",
            "points": [],
        }
    
    points = []
    next_idx = 0
    
    for summary in summaries:
        # Extract cycle_or_run_idx, or assign sequentially
        idx = summary.get("cycle_or_run_idx")
        if idx is None:
            idx = next_idx
            next_idx += 1
        else:
            # Ensure monotone increasing
            if idx < next_idx:
                idx = next_idx
            next_idx = idx + 1
        
        points.append({
            "cycle_or_run_idx": idx,
            "alignment_score": summary.get("alignment_score", 1.0),
            "drift_band": summary.get("drift_band", "STABLE"),
        })
    
    # Verify monotone indexing (invariant check)
    for i in range(1, len(points)):
        if points[i]["cycle_or_run_idx"] < points[i-1]["cycle_or_run_idx"]:
            # This should never happen due to correction logic above, but verify
            raise ValueError(
                f"Time-series points must be monotone increasing: "
                f"point {i} has idx {points[i]['cycle_or_run_idx']} < "
                f"point {i-1} idx {points[i-1]['cycle_or_run_idx']}"
            )
    
    return {
        "schema_version": "1.0.0",
        "points": points,
    }


def _calculate_episode_severity_score(
    status: str,
    max_drift_band_seen: str,
    min_alignment_score_seen: float,
    alignment_ok_threshold: float,
    high_drift_value: str,
) -> float:
    """
    Calculate deterministic episode severity score.
    
    Scoring formula:
    - Base by status: CONSISTENT=0, TENSION=50, CONFLICT=100
    - Drift bump: +20 if max_drift_band_seen == high_drift_value
    - Alignment bump: max(0, (alignment_ok_threshold - min_alignment_score_seen) * 10) clamped
    
    Higher scores indicate more severe episodes.
    
    Args:
        status: Episode status ("CONSISTENT" | "TENSION" | "CONFLICT").
        max_drift_band_seen: Highest drift band in episode.
        min_alignment_score_seen: Lowest alignment score in episode.
        alignment_ok_threshold: Alignment threshold for CONSISTENT status.
        high_drift_value: Drift band value that triggers HIGH_DRIFT logic.
        
    Returns:
        Episode severity score (float, higher = more severe).
    """
    # Base score by status
    base_scores = {"CONSISTENT": 0.0, "TENSION": 50.0, "CONFLICT": 100.0}
    score = base_scores.get(status, 0.0)
    
    # Drift bump: +20 if HIGH_DRIFT
    if max_drift_band_seen == high_drift_value:
        score += 20.0
    
    # Alignment bump: based on how far below threshold
    # Clamp to ensure non-negative and reasonable range
    alignment_gap = max(0.0, alignment_ok_threshold - min_alignment_score_seen)
    alignment_bump = min(alignment_gap * 10.0, 30.0)  # Cap at 30 to keep scores reasonable
    score += alignment_bump
    
    return score


def summarize_coherence_vs_curriculum_governance(
    coherence_ts: Dict[str, Any],
    curriculum_signals: List[Dict[str, Any]],
    alignment_ok_threshold: float = 0.8,
    alignment_conflict_threshold: float = 0.6,
    high_drift_value: str = "HIGH_DRIFT",
) -> Dict[str, Any]:
    """
    Cross-check curriculum coherence time-series against curriculum governance signals.
    
    STATUS: PHASE X — COHERENCE VS GOVERNANCE CROSS-CHECK
    
    Compares curriculum coherence trends with curriculum governance signals to
    identify consistency, tension, or conflict between taxonomy alignment and
    curriculum evolution.
    
    Status determination rules (applied per point, then aggregated):
    - CONSISTENT: drift_band ≤ MEDIUM_DRIFT AND alignment_score ≥ alignment_ok_threshold
    - TENSION: (drift_band == high_drift_value XOR alignment_score < alignment_conflict_threshold)
    - CONFLICT: drift_band == high_drift_value AND alignment_score < alignment_conflict_threshold
    
    Episodes are window ranges where status != CONSISTENT, enriched with metadata.
    
    SHADOW MODE: This is advisory only, not an acceptance gate. Results are
    observational and do not block P5 acceptance.
    
    Args:
        coherence_ts: Time-series from build_curriculum_coherence_timeseries().
            Expected structure: {"schema_version": "1.0.0", "points": [...]}
            Each point has: cycle_or_run_idx, alignment_score, drift_band.
        curriculum_signals: List of curriculum governance signals (from
            curriculum-governance-signal schema). Expected keys: timestamp,
            signal_type, severity, status, active_slice. Currently unused but
            reserved for future cross-check logic.
        alignment_ok_threshold: Minimum alignment score for CONSISTENT status (default: 0.8).
        alignment_conflict_threshold: Maximum alignment score for CONFLICT status (default: 0.6).
        high_drift_value: Drift band value that triggers HIGH_DRIFT logic (default: "HIGH_DRIFT").
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - status: "CONSISTENT" | "TENSION" | "CONFLICT" (worst status across all points)
        - episodes: List of episode dicts with start_idx, end_idx, status, point_count,
          max_drift_band_seen, min_alignment_score_seen
        - advisory_notes: List of 1-3 neutral advisory notes
        - summary: Dict with num_points, num_episodes, worst_status, worst_episode
    """
    points = coherence_ts.get("points", [])
    if not points:
        severity_score_basis = {
            "status_weights": {
                "CONSISTENT": 0.0,
                "TENSION": 50.0,
                "CONFLICT": 100.0,
            },
            "drift_bump": 20.0,
            "alignment_bump_formula": f"min(30.0, (alignment_ok_threshold - min_alignment_score_seen) * 10.0)",
            "alignment_ok_threshold_used": alignment_ok_threshold,
            "high_drift_value_used": high_drift_value,
        }
        return {
            "schema_version": "1.0.0",
            "status": "CONSISTENT",
            "episodes": [],
            "advisory_notes": ["No time-series points available for cross-check"],
            "summary": {
                "num_points": 0,
                "num_episodes": 0,
                "worst_status": "CONSISTENT",
                "worst_episode": None,
            },
            "severity_score_basis": severity_score_basis,
        }
    
    # Status priority: CONFLICT > TENSION > CONSISTENT
    status_priority = {"CONFLICT": 3, "TENSION": 2, "CONSISTENT": 1}
    drift_priority = {"HIGH_DRIFT": 4, "MEDIUM_DRIFT": 3, "LOW_DRIFT": 2, "STABLE": 1}
    
    # Analyze each point
    point_statuses = []
    for point in points:
        drift_band = point.get("drift_band", "STABLE")
        alignment_score = point.get("alignment_score", 1.0)
        cycle_idx = point.get("cycle_or_run_idx", 0)
        
        # Determine status for this point using parameterized thresholds
        if drift_band == high_drift_value and alignment_score < alignment_conflict_threshold:
            point_status = "CONFLICT"
        elif drift_band in ["STABLE", "LOW_DRIFT", "MEDIUM_DRIFT"] and alignment_score >= alignment_ok_threshold:
            point_status = "CONSISTENT"
        else:
            # TENSION: (high_drift_value XOR alignment_score < alignment_conflict_threshold)
            # This covers: (high_drift_value AND alignment_score >= alignment_conflict_threshold) OR
            #              (NOT high_drift_value AND alignment_score < alignment_conflict_threshold)
            point_status = "TENSION"
        
        point_statuses.append({
            "cycle_idx": cycle_idx,
            "status": point_status,
            "drift_band": drift_band,
            "alignment_score": alignment_score,
        })
    
    # Determine overall status (worst across all points)
    worst_status = "CONSISTENT"
    for ps in point_statuses:
        if status_priority[ps["status"]] > status_priority[worst_status]:
            worst_status = ps["status"]
    
    # Build episodes (contiguous windows where status != CONSISTENT)
    episodes = []
    current_episode = None
    current_episode_points = []
    
    for ps in point_statuses:
        if ps["status"] != "CONSISTENT":
            if current_episode is None:
                # Start new episode
                current_episode = {
                    "start_idx": ps["cycle_idx"],
                    "end_idx": ps["cycle_idx"],
                    "status": ps["status"],
                }
                current_episode_points = [ps]
            else:
                # Extend current episode if same status, otherwise start new one
                if current_episode["status"] == ps["status"]:
                    current_episode["end_idx"] = ps["cycle_idx"]
                    current_episode_points.append(ps)
                else:
                    # Close current episode and enrich with metadata
                    current_episode["point_count"] = len(current_episode_points)
                    current_episode["max_drift_band_seen"] = max(
                        current_episode_points,
                        key=lambda p: drift_priority.get(p["drift_band"], 0)
                    )["drift_band"]
                    current_episode["min_alignment_score_seen"] = min(
                        p["alignment_score"] for p in current_episode_points
                    )
                    # Calculate episode severity score
                    current_episode["episode_severity_score"] = _calculate_episode_severity_score(
                        current_episode["status"],
                        current_episode["max_drift_band_seen"],
                        current_episode["min_alignment_score_seen"],
                        alignment_ok_threshold,
                        high_drift_value,
                    )
                    episodes.append(current_episode)
                    # Start new episode
                    current_episode = {
                        "start_idx": ps["cycle_idx"],
                        "end_idx": ps["cycle_idx"],
                        "status": ps["status"],
                    }
                    current_episode_points = [ps]
        else:
            # CONSISTENT point - close any open episode
            if current_episode is not None:
                current_episode["point_count"] = len(current_episode_points)
                current_episode["max_drift_band_seen"] = max(
                    current_episode_points,
                    key=lambda p: drift_priority.get(p["drift_band"], 0)
                )["drift_band"]
                current_episode["min_alignment_score_seen"] = min(
                    p["alignment_score"] for p in current_episode_points
                )
                # Calculate episode severity score
                current_episode["episode_severity_score"] = _calculate_episode_severity_score(
                    current_episode["status"],
                    current_episode["max_drift_band_seen"],
                    current_episode["min_alignment_score_seen"],
                    alignment_ok_threshold,
                    high_drift_value,
                )
                episodes.append(current_episode)
                current_episode = None
                current_episode_points = []
    
    # Close final episode if open
    if current_episode is not None:
        current_episode["point_count"] = len(current_episode_points)
        current_episode["max_drift_band_seen"] = max(
            current_episode_points,
            key=lambda p: drift_priority.get(p["drift_band"], 0)
        )["drift_band"]
        current_episode["min_alignment_score_seen"] = min(
            p["alignment_score"] for p in current_episode_points
        )
        # Calculate episode severity score
        current_episode["episode_severity_score"] = _calculate_episode_severity_score(
            current_episode["status"],
            current_episode["max_drift_band_seen"],
            current_episode["min_alignment_score_seen"],
            alignment_ok_threshold,
            high_drift_value,
        )
        episodes.append(current_episode)
    
    # Generate advisory notes (1-3 neutral notes) using parameterized thresholds
    advisory_notes = []
    
    if worst_status == "CONFLICT":
        advisory_notes.append(
            f"High drift band ({high_drift_value}) observed with alignment score below {alignment_conflict_threshold} threshold"
        )
        conflict_count = sum(1 for ps in point_statuses if ps["status"] == "CONFLICT")
        if conflict_count > 0:
            advisory_notes.append(
                f"Conflict status detected in {conflict_count} time-series point(s)"
            )
    elif worst_status == "TENSION":
        high_drift_count = sum(1 for ps in point_statuses if ps["drift_band"] == high_drift_value)
        low_alignment_count = sum(1 for ps in point_statuses if ps["alignment_score"] < alignment_conflict_threshold)
        
        if high_drift_count > 0:
            advisory_notes.append(
                f"High drift band observed in {high_drift_count} point(s) with alignment score >= {alignment_conflict_threshold}"
            )
        if low_alignment_count > 0:
            advisory_notes.append(
                f"Alignment score below {alignment_conflict_threshold} threshold in {low_alignment_count} point(s) with drift band <= MEDIUM_DRIFT"
            )
        if not advisory_notes:
            advisory_notes.append(
                "Tension status detected between drift band and alignment score thresholds"
            )
    else:
        # CONSISTENT
        advisory_notes.append(
            f"All time-series points meet consistency criteria (drift_band <= MEDIUM_DRIFT, alignment_score >= {alignment_ok_threshold})"
        )
    
    # Limit to 3 notes
    advisory_notes = advisory_notes[:3]
    
    # Build summary block
    num_points = len(points)
    num_episodes = len(episodes)
    
    # Find worst episode: max episode_severity_score, tie-break by longer duration, then smaller start_idx
    worst_episode = None
    selected_by = []
    if episodes:
        # Determine selection criteria
        max_severity = max(e.get("episode_severity_score", 0) for e in episodes)
        episodes_with_max_severity = [e for e in episodes if e.get("episode_severity_score", 0) == max_severity]
        
        if len(episodes_with_max_severity) == 1:
            selected_by = ["severity_score"]
            worst_episode = episodes_with_max_severity[0]
        else:
            # Tie-break by duration
            max_duration = max(e.get("end_idx", 0) - e.get("start_idx", 0) for e in episodes_with_max_severity)
            episodes_with_max_duration = [e for e in episodes_with_max_severity 
                                         if (e.get("end_idx", 0) - e.get("start_idx", 0)) == max_duration]
            
            if len(episodes_with_max_duration) == 1:
                selected_by = ["severity_score", "duration"]
                worst_episode = episodes_with_max_duration[0]
            else:
                # Tie-break by start_idx (smaller wins)
                min_start_idx = min(e.get("start_idx", 0) for e in episodes_with_max_duration)
                worst_episode = next(e for e in episodes_with_max_duration if e.get("start_idx", 0) == min_start_idx)
                selected_by = ["severity_score", "duration", "start_idx"]
        
        # Add selection trace to worst_episode
        worst_episode = dict(worst_episode)  # Make a copy to avoid mutating original
        worst_episode["selected_by"] = selected_by
    
    summary = {
        "num_points": num_points,
        "num_episodes": num_episodes,
        "worst_status": worst_status,
        "worst_episode": worst_episode,
    }
    
    # Build severity score basis metadata for auditability
    severity_score_basis = {
        "status_weights": {
            "CONSISTENT": 0.0,
            "TENSION": 50.0,
            "CONFLICT": 100.0,
        },
        "drift_bump": 20.0,
        "alignment_bump_formula": f"min(30.0, (alignment_ok_threshold - min_alignment_score_seen) * 10.0)",
        "alignment_ok_threshold_used": alignment_ok_threshold,
        "high_drift_value_used": high_drift_value,
    }
    
    return {
        "schema_version": "1.0.0",
        "status": worst_status,
        "episodes": episodes,
        "advisory_notes": advisory_notes,
        "summary": summary,
        "severity_score_basis": severity_score_basis,
    }


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def load_json_file(path: str) -> Dict[str, Any]:
    """Load JSON file, exit on error."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(2)


def cmd_integrity_radar(args: argparse.Namespace) -> int:
    """Generate integrity radar from analysis files."""
    metrics_impact = load_json_file(args.metrics_impact)
    docs_alignment = load_json_file(args.docs_alignment)
    curriculum_alignment = load_json_file(args.curriculum_alignment)
    
    radar = build_taxonomy_integrity_radar(metrics_impact, docs_alignment, curriculum_alignment)
    print(json.dumps(radar, indent=2))
    return 0


def cmd_integrity_ci_check(args: argparse.Namespace) -> int:
    """Evaluate taxonomy for CI blocking."""
    radar_data = load_json_file(args.radar)
    
    exit_code, message = evaluate_taxonomy_for_ci(radar_data)
    
    if args.json:
        result = {
            "exit_code": exit_code,
            "message": message,
            "integrity_status": radar_data.get("integrity_status", "UNKNOWN"),
        }
        print(json.dumps(result, indent=2))
    else:
        print(message)
    
    return exit_code


def cmd_integrity_console_tile(args: argparse.Namespace) -> int:
    """Generate global console tile."""
    radar = load_json_file(args.radar)
    risk_analysis = load_json_file(args.risk_analysis)
    
    tile = build_global_console_tile(radar, risk_analysis)
    print(json.dumps(tile, indent=2))
    return 0


def cmd_drift_timeline(args: argparse.Namespace) -> int:
    """Generate drift timeline from historical impacts."""
    historical_impacts = load_json_file(args.historical_impacts)
    
    if not isinstance(historical_impacts, list):
        print("ERROR: historical_impacts must be a JSON array", file=sys.stderr)
        return 2
    
    timeline = build_taxonomy_drift_timeline(historical_impacts)
    print(json.dumps(timeline, indent=2))
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Taxonomy Governance & CI Integration - Phase V"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # integrity-radar command
    radar_parser = subparsers.add_parser("integrity-radar", help="Generate integrity radar")
    radar_parser.add_argument("--metrics-impact", required=True, help="Path to metrics impact JSON")
    radar_parser.add_argument("--docs-alignment", required=True, help="Path to docs alignment JSON")
    radar_parser.add_argument("--curriculum-alignment", required=True, help="Path to curriculum alignment JSON")
    radar_parser.set_defaults(func=cmd_integrity_radar)
    
    # integrity-ci-check command
    ci_parser = subparsers.add_parser("integrity-ci-check", help="Evaluate taxonomy for CI blocking")
    ci_parser.add_argument("--radar", required=True, help="Path to radar JSON")
    ci_parser.add_argument("--json", action="store_true", help="Output JSON format")
    ci_parser.set_defaults(func=cmd_integrity_ci_check)
    
    # integrity-console-tile command
    tile_parser = subparsers.add_parser("integrity-console-tile", help="Generate global console tile")
    tile_parser.add_argument("--radar", required=True, help="Path to radar JSON")
    tile_parser.add_argument("--risk-analysis", required=True, help="Path to risk analysis JSON")
    tile_parser.set_defaults(func=cmd_integrity_console_tile)
    
    # drift-timeline command
    timeline_parser = subparsers.add_parser("drift-timeline", help="Generate drift timeline")
    timeline_parser.add_argument("--historical-impacts", required=True, help="Path to historical impacts JSON array")
    timeline_parser.set_defaults(func=cmd_drift_timeline)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

