"""Global Health Envelope v4 integration for P3/P4 reports and evidence.

STATUS: PHASE VI — GLOBAL HEALTH ENVELOPE v4 INTEGRATION

Provides integration of envelope v4 signals into:
- P3 stability reports (First-Light Integration)
- P4 calibration reports
- Evidence packs
- Uplift council summaries

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Envelope v4 signals are purely observational
- They do NOT influence any other signals or system health classification
- No control flow depends on envelope v4 values
- No governance writes
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def attach_envelope_v4_to_p3_stability_report(
    stability_report: Dict[str, Any],
    envelope_v4: Dict[str, Any],
    coherence_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach Global Health Envelope v4 summary to P3 stability report (First-Light Integration).
    
    Adds envelope v4 summary with:
    - coherence_score: Derived from coherence analysis
    - system_alignment: Overall system alignment status
    - pressure_status: Pressure assessment from envelope
    - status_light: GREEN|YELLOW|RED
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Envelope v4 is observational only
    
    Args:
        stability_report: P3 stability report dictionary
        envelope_v4: Global Health Envelope v4 from build_global_health_envelope_v4
        coherence_analysis: Optional coherence analysis from analyze_system_coherence
        
    Returns:
        Updated stability report with global_health_envelope_v4 field
    """
    global_band = envelope_v4.get("global_band", "GREEN")
    
    # Extract coherence score from coherence analysis if available
    coherence_score = None
    if coherence_analysis:
        coherence_status = coherence_analysis.get("coherence_status", "COHERENT")
        if coherence_status == "COHERENT":
            coherence_score = 1.0
        elif coherence_status == "MISMATCHED":
            coherence_score = 0.3
        else:  # INSUFFICIENT_DATA
            coherence_score = 0.5
    
    # Determine system alignment from envelope components
    component_count = 0
    aligned_count = 0
    envelope_components = envelope_v4.get("envelope_components", {})
    for comp_name, comp_data in envelope_components.items():
        if comp_data.get("present", False):
            component_count += 1
            band = comp_data.get("band", "UNKNOWN")
            if band == "GREEN":
                aligned_count += 1
    
    system_alignment = "ALIGNED"
    if component_count > 0:
        alignment_ratio = aligned_count / component_count
        if alignment_ratio < 0.5:
            system_alignment = "MISALIGNED"
        elif alignment_ratio < 0.8:
            system_alignment = "PARTIAL"
    
    # Determine pressure status from cross-signal hotspots
    hotspots = envelope_v4.get("cross_signal_hotspots", [])
    pressure_status = "STABLE"
    if len(hotspots) >= 3:
        pressure_status = "HIGH"
    elif len(hotspots) >= 1:
        pressure_status = "ELEVATED"
    
    # Map global band to status light
    status_light = "GREEN"
    if global_band == "RED":
        status_light = "RED"
    elif global_band == "YELLOW":
        status_light = "YELLOW"
    
    # Build envelope v4 summary
    envelope_summary = {
        "coherence_score": coherence_score,
        "system_alignment": system_alignment,
        "pressure_status": pressure_status,
        "status_light": status_light,
        "global_band": global_band,
        "component_count": component_count,
        "aligned_count": aligned_count,
        "hotspot_count": len(hotspots),
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["global_health_envelope_v4"] = envelope_summary
    
    return updated_report


def attach_envelope_v4_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    envelope_v4: Dict[str, Any],
    coherence_analysis: Dict[str, Any],
    director_mega_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach Global Health Envelope v4 to P4 calibration report.
    
    Adds:
    - envelope_v4: Full envelope v4 output
    - coherence_analysis: Coherence analysis output
    - director_mega_panel: Director mega-panel output
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Envelope v4 is observational only
    
    Args:
        calibration_report: P4 calibration report dictionary
        envelope_v4: Global Health Envelope v4 from build_global_health_envelope_v4
        coherence_analysis: Coherence analysis from analyze_system_coherence
        director_mega_panel: Director mega-panel from build_director_mega_panel
        
    Returns:
        Updated calibration report with envelope_v4 field
    """
    # Build envelope v4 calibration block
    envelope_calibration = {
        "envelope_v4": envelope_v4,
        "coherence_analysis": coherence_analysis,
        "director_mega_panel": director_mega_panel,
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["envelope_v4"] = envelope_calibration
    
    return updated_report


def attach_envelope_v4_to_evidence(
    evidence: Dict[str, Any],
    envelope_v4: Dict[str, Any],
    coherence_analysis: Dict[str, Any],
    director_mega_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach Global Health Envelope v4 to evidence pack.
    
    Stores under evidence["governance"]["envelope_v4"]:
    - envelope_v4: Full envelope v4 output
    - coherence_analysis: Coherence analysis output
    - director_mega_panel: Director mega-panel output
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Envelope v4 is observational only
    
    Args:
        evidence: Evidence dictionary
        envelope_v4: Global Health Envelope v4 from build_global_health_envelope_v4
        coherence_analysis: Coherence analysis from analyze_system_coherence
        director_mega_panel: Director mega-panel from build_director_mega_panel
        
    Returns:
        Updated evidence with envelope_v4 under evidence["governance"]["envelope_v4"]
    """
    # Build envelope v4 evidence block
    envelope_evidence = {
        "envelope_v4": envelope_v4,
        "coherence_analysis": coherence_analysis,
        "director_mega_panel": director_mega_panel,
    }
    
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])
    
    # Attach envelope v4
    updated_evidence["governance"]["envelope_v4"] = envelope_evidence
    
    # Build and attach release attitude annex
    release_attitude_annex = build_first_light_release_attitude_annex(
        envelope_v4, director_mega_panel
    )
    updated_evidence["governance"]["envelope_v4"]["first_light_release_attitude"] = (
        release_attitude_annex
    )
    
    return updated_evidence


def build_first_light_release_attitude_annex(
    envelope_v4: Dict[str, Any],
    director_mega_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First-Light Release Attitude Annex — cross-pillar system attitude summary.
    
    Provides a concise summary of the overall system's "release attitude" based on
    global health envelope v4 and director mega-panel signals.
    
    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The returned annex is purely observational
    - No control flow depends on the annex contents
    - This is a health observer, not a hard control path
    
    Args:
        envelope_v4: Global Health Envelope v4 from build_global_health_envelope_v4
        director_mega_panel: Director mega-panel from build_director_mega_panel
        
    Returns:
        Release attitude annex dictionary with:
        - schema_version: "1.0.0"
        - global_band: "GREEN" | "YELLOW" | "RED"
        - system_alignment: Derived from envelope components
        - release_ready: bool (from director mega-panel)
        - status_light: "🟢" | "🟡" | "🔴" (from director mega-panel)
    """
    global_band = envelope_v4.get("global_band", "GREEN")
    
    # Derive system_alignment from envelope components
    component_count = 0
    aligned_count = 0
    envelope_components = envelope_v4.get("envelope_components", {})
    for comp_name, comp_data in envelope_components.items():
        if comp_data.get("present", False):
            component_count += 1
            band = comp_data.get("band", "UNKNOWN")
            if band == "GREEN":
                aligned_count += 1
    
    system_alignment = "ALIGNED"
    if component_count > 0:
        alignment_ratio = aligned_count / component_count
        if alignment_ratio < 0.5:
            system_alignment = "MISALIGNED"
        elif alignment_ratio < 0.8:
            system_alignment = "PARTIAL"
    
    # Extract from director mega-panel
    release_ready = director_mega_panel.get("release_ready", True)
    status_light = director_mega_panel.get("mega_status_light", "🟢")
    
    return {
        "schema_version": "1.0.0",
        "global_band": global_band,
        "system_alignment": system_alignment,
        "release_ready": release_ready,
        "status_light": status_light,
    }


def summarize_envelope_v4_for_uplift_council(
    mega_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize Global Health Envelope v4 for uplift council decision-making.
    
    Maps director mega-panel status to council decision signals:
    - RED → BLOCK
    - YELLOW → WARN
    - GREEN → OK
    
    Extracts:
    - top_drivers: Primary health drivers from component summary
    - coherence_discrepancies: Coherence mismatches if any
    - cross_pillar_faults: Cross-signal hotspots
    
    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - This is a health observer, not a hard control path
    
    Args:
        mega_panel: Director mega-panel from build_director_mega_panel
        
    Returns:
        Council summary dictionary with:
        - council_status: "OK" | "WARN" | "BLOCK"
        - status_light: "🟢" | "🟡" | "🔴"
        - top_drivers: List[str] (primary health drivers)
        - coherence_discrepancies: List[Dict] (coherence mismatches)
        - cross_pillar_faults: List[Dict] (cross-signal hotspots)
        - rationale: str (neutral explanation)
    """
    # Extract status light and global envelope
    status_light = mega_panel.get("mega_status_light", "🟢")
    global_envelope = mega_panel.get("global_envelope", {})
    coherence_analysis = mega_panel.get("coherence_analysis", {})
    component_summary = mega_panel.get("component_summary", {})
    
    # Map status light to council status
    if status_light == "🔴":
        council_status = "BLOCK"
    elif status_light == "🟡":
        council_status = "WARN"
    else:  # 🟢
        council_status = "OK"
    
    # Extract top drivers from component summary
    top_drivers = []
    for agent_name, agent_data in component_summary.items():
        band = agent_data.get("band", "GREEN")
        status = agent_data.get("status", "OK")
        if band == "RED" or status == "CRITICAL":
            top_drivers.append(f"{agent_name}: {status}")
        elif band == "YELLOW" or status == "WARN":
            top_drivers.append(f"{agent_name}: {status}")
    
    # Extract coherence discrepancies
    coherence_discrepancies = []
    mismatches = coherence_analysis.get("mismatches", [])
    for mismatch in mismatches:
        coherence_discrepancies.append({
            "components": mismatch.get("components", []),
            "issue": mismatch.get("issue", ""),
            "severity": mismatch.get("severity", "MEDIUM"),
        })
    
    # Extract cross-pillar faults from global envelope
    cross_pillar_faults = global_envelope.get("cross_signal_hotspots", [])
    
    # Build rationale
    if council_status == "BLOCK":
        if len(top_drivers) > 0:
            rationale = f"Critical health issues detected in {len(top_drivers)} component(s)."
        elif len(coherence_discrepancies) > 0:
            rationale = f"{len(coherence_discrepancies)} coherence mismatch(es) detected."
        else:
            rationale = "Global health envelope indicates critical system state."
    elif council_status == "WARN":
        if len(top_drivers) > 0:
            rationale = f"Health attention required in {len(top_drivers)} component(s)."
        elif len(coherence_discrepancies) > 0:
            rationale = f"{len(coherence_discrepancies)} coherence discrepancy(ies) detected."
        else:
            rationale = "Global health envelope indicates elevated risk state."
    else:
        rationale = "Global health envelope indicates stable system state."
    
    return {
        "council_status": council_status,
        "status_light": status_light,
        "top_drivers": top_drivers,
        "coherence_discrepancies": coherence_discrepancies,
        "cross_pillar_faults": cross_pillar_faults,
        "rationale": rationale,
        "release_ready": mega_panel.get("release_ready", True),
    }


def build_cal_exp_release_attitude_annex(
    cal_id: str,
    release_attitude_annex: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a per-experiment release attitude annex for a calibration experiment.
    
    STATUS: PHASE VI — P5 CALIBRATION EXPERIMENT RELEASE ATTITUDE CAPTURE
    
    Wraps a release attitude annex with calibration experiment identifier,
    suitable for per-experiment export and aggregation into a release attitude strip.
    
    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The returned annex is purely observational
    - No control flow depends on the annex contents
    - This is evidence-only, not a gate
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        release_attitude_annex: Release attitude annex from build_first_light_release_attitude_annex()
        
    Returns:
        Per-experiment release attitude annex dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str (experiment identifier)
        - global_band: "GREEN" | "YELLOW" | "RED"
        - system_alignment: "ALIGNED" | "PARTIAL" | "MISALIGNED"
        - release_ready: bool
        - status_light: "🟢" | "🟡" | "🔴"
    """
    return {
        "schema_version": "1.0.0",
        "cal_id": cal_id,
        "global_band": release_attitude_annex.get("global_band", "GREEN"),
        "system_alignment": release_attitude_annex.get("system_alignment", "ALIGNED"),
        "release_ready": release_attitude_annex.get("release_ready", True),
        "status_light": release_attitude_annex.get("status_light", "🟢"),
    }


def export_cal_exp_release_attitude_annex(
    cal_id: str,
    release_attitude_annex: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Export per-experiment release attitude annex to JSON file.
    
    STATUS: PHASE VI — P5 CALIBRATION EXPERIMENT RELEASE ATTITUDE EXPORT
    
    Writes release attitude annex to calibration/release_attitude_annex_<cal_id>.json.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from file writing)
    - The exported file is purely observational
    - No control flow depends on the file contents
    
    Args:
        cal_id: Calibration experiment identifier
        release_attitude_annex: Release attitude annex (with or without cal_id)
        output_dir: Output directory (calibration/ subdirectory will be created)
        
    Returns:
        Path to the exported JSON file
    """
    # Build per-experiment annex with cal_id
    cal_exp_annex = build_cal_exp_release_attitude_annex(cal_id, release_attitude_annex)
    
    # Ensure calibration directory exists
    calibration_dir = output_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to file
    filename = f"release_attitude_annex_{cal_id}.json"
    output_path = calibration_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cal_exp_annex, f, indent=2, ensure_ascii=False)
    
    return output_path


def build_release_attitude_strip(
    annexes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build Release Attitude Strip — aggregates per-experiment release attitude annexes.
    
    STATUS: PHASE VI — P5 CALIBRATION RELEASE ATTITUDE STRIP
    
    Aggregates multiple calibration experiment release attitude annexes into a
    single strip for quick cross-experiment posture assessment. Provides a visual
    shorthand for auditors across calibration experiments.
    
    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The returned strip is purely observational
    - No control flow depends on the strip contents
    - This is a summary badge, not an enforcement gate
    - Decisions remain with human governance processes
    
    Args:
        annexes: List of per-experiment release attitude annexes (each should have cal_id)
        
    Returns:
        Release attitude strip dictionary with:
        - schema_version: "1.0.0"
        - experiments: List[Dict] with cal_id, global_band, system_alignment, release_ready, status_light
        - summary: Dict with release_ready_count, total_count, release_ready_ratio, trend
    """
    # Extract experiment summaries (sorted by cal_id for determinism)
    experiments = []
    for annex in annexes:
        cal_id = annex.get("cal_id", "UNKNOWN")
        experiments.append({
            "cal_id": cal_id,
            "global_band": annex.get("global_band", "GREEN"),
            "system_alignment": annex.get("system_alignment", "ALIGNED"),
            "release_ready": annex.get("release_ready", True),
            "status_light": annex.get("status_light", "🟢"),
        })
    
    # Sort by cal_id for deterministic output
    experiments.sort(key=lambda x: x["cal_id"])
    
    # Compute summary statistics
    total_count = len(experiments)
    release_ready_count = sum(1 for exp in experiments if exp.get("release_ready", False))
    release_ready_ratio = release_ready_count / total_count if total_count > 0 else 0.0
    
    # Compute trend based on status_light sequence
    trend = "STABLE"
    if total_count >= 2:
        # Map status_light to numeric values: 🟢=2, 🟡=1, 🔴=0
        def status_light_to_value(light: str) -> int:
            if light == "🟢":
                return 2
            elif light == "🟡":
                return 1
            elif light == "🔴":
                return 0
            else:
                # Unknown status defaults to 1 (neutral)
                return 1
        
        # Get sequence of status values
        status_values = [status_light_to_value(exp.get("status_light", "🟢")) for exp in experiments]
        
        # Compute slope (delta from first to last, normalized by sequence length)
        if len(status_values) >= 2:
            first_value = status_values[0]
            last_value = status_values[-1]
            delta = last_value - first_value
            
            # Simple threshold: if delta > 0.3 → IMPROVING, if delta < -0.3 → DEGRADING
            # Normalize by max possible delta (2) to get a ratio
            normalized_delta = delta / 2.0
            
            if normalized_delta > 0.3:
                trend = "IMPROVING"
            elif normalized_delta < -0.3:
                trend = "DEGRADING"
            else:
                trend = "STABLE"
    
    return {
        "schema_version": "1.0.0",
        "experiments": experiments,
        "summary": {
            "total_count": total_count,
            "release_ready_count": release_ready_count,
            "release_ready_ratio": round(release_ready_ratio, 3),  # Round for determinism
            "trend": trend,
        },
    }


def extract_release_attitude_strip_signal(
    strip: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact release attitude strip signal for status surface.
    
    STATUS: PHASE VI — P5 CALIBRATION RELEASE ATTITUDE STRIP SIGNAL EXTRACTION
    
    Extracts key metrics from release attitude strip for inclusion in signals surface.
    Provides a compact summary suitable for status dashboards and advisory warnings.
    
    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The returned signal is advisory-only
    - No control flow should depend on this signal for blocking releases
    - This is purely for observability and logging
    
    Args:
        strip: Release attitude strip from build_release_attitude_strip()
        
    Returns:
        Compact signal dictionary with:
        - total_count: int
        - release_ready_ratio: float (0.0-1.0)
        - trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - first_status_light: str (from first experiment)
        - last_status_light: str (from last experiment)
        - advisory_warning: Optional[str] (warning if trend == DEGRADING)
    """
    summary = strip.get("summary", {})
    experiments = strip.get("experiments", [])
    
    total_count = summary.get("total_count", 0)
    release_ready_ratio = summary.get("release_ready_ratio", 0.0)
    trend = summary.get("trend", "STABLE")
    
    # Extract first and last status lights
    first_status_light = None
    last_status_light = None
    if len(experiments) > 0:
        first_status_light = experiments[0].get("status_light", "🟢")
        last_status_light = experiments[-1].get("status_light", "🟢")
    else:
        first_status_light = "🟢"
        last_status_light = "🟢"
    
    # Generate advisory warning if trend is DEGRADING
    advisory_warning = None
    if trend == "DEGRADING":
        advisory_warning = "Release attitude trend is DEGRADING across calibration experiments"
    
    return {
        "total_count": total_count,
        "release_ready_ratio": release_ready_ratio,
        "trend": trend,
        "first_status_light": first_status_light,
        "last_status_light": last_status_light,
        "advisory_warning": advisory_warning,
    }


def attach_release_attitude_strip_to_evidence(
    evidence: Dict[str, Any],
    strip: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach Release Attitude Strip to evidence pack and extract signal.
    
    STATUS: PHASE VI — P5 CALIBRATION RELEASE ATTITUDE STRIP EVIDENCE INTEGRATION
    
    Stores release attitude strip under evidence["governance"]["release_attitude_strip"]
    and extracts a compact signal under evidence["signals"]["release_attitude_strip"].
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Strip is observational only
    - Signal is advisory-only, not an enforcement gate
    
    Args:
        evidence: Evidence dictionary
        strip: Release attitude strip from build_release_attitude_strip()
        
    Returns:
        Updated evidence with:
        - release_attitude_strip under evidence["governance"]["release_attitude_strip"]
        - signal under evidence["signals"]["release_attitude_strip"]
    """
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])
    
    # Attach release attitude strip
    updated_evidence["governance"]["release_attitude_strip"] = strip
    
    # Extract and attach signal
    signal = extract_release_attitude_strip_signal(strip)
    
    # Ensure signals structure exists
    if "signals" not in updated_evidence:
        updated_evidence["signals"] = {}
    else:
        updated_evidence["signals"] = dict(updated_evidence["signals"])
    
    # Attach release attitude strip signal
    updated_evidence["signals"]["release_attitude_strip"] = signal
    
    return updated_evidence


def release_attitude_strip_for_alignment_view(
    strip_or_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert release attitude strip to GGFL alignment view format.

    PHASE X — GGFL ADAPTER FOR RELEASE ATTITUDE STRIP

    Normalizes the release attitude strip into the Global Governance Fusion Layer (GGFL)
    unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Deterministic output for identical inputs
    - Low weight hint (advisory only)

    Args:
        strip_or_signal: Release attitude strip from build_release_attitude_strip()
            or signal from first_light_status.json signals["release_attitude_strip"]

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-REL" (identifies this as a release attitude signal)
        - status: "ok" | "warn" (warn if last_status_light != GREEN or release_ready_ratio < 1.0)
        - conflict: false (release attitude strip never triggers conflict directly)
        - weight_hint: "LOW" (advisory only, low weight)
        - drivers: List[str] (reason codes: DRIVER_RELEASE_READY_RATIO_LT_1, DRIVER_STATUS_LIGHT_NOT_GREEN)
        - shadow_mode_invariants: Dict with all_divergence_logged_only, no_governance_modification, no_abort_enforcement
    """
    # Extract signal if strip provided, otherwise use signal directly
    if "summary" in strip_or_signal:
        # It's a strip, extract signal
        signal = extract_release_attitude_strip_signal(strip_or_signal)
    else:
        # It's already a signal
        signal = strip_or_signal
    
    # Determine status: "ok" if last_status_light is GREEN and release_ready_ratio is 1.0, else "warn"
    last_status_light = signal.get("last_status_light", "🟢")
    release_ready_ratio = signal.get("release_ready_ratio", 1.0)
    
    status = "ok"
    if last_status_light != "🟢" or release_ready_ratio < 1.0:
        status = "warn"
    
    # Build drivers list with correct reason codes
    drivers = []
    if release_ready_ratio < 1.0:
        drivers.append("DRIVER_RELEASE_READY_RATIO_LT_1")
    
    if last_status_light != "🟢":
        drivers.append("DRIVER_STATUS_LIGHT_NOT_GREEN")
    
    return {
        "signal_type": "SIG-REL",
        "status": status,
        "conflict": False,
        "weight_hint": "LOW",
        "drivers": drivers,
        "shadow_mode_invariants": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
    }


__all__ = [
    "attach_envelope_v4_to_p3_stability_report",
    "attach_envelope_v4_to_p4_calibration_report",
    "attach_envelope_v4_to_evidence",
    "build_first_light_release_attitude_annex",
    "build_cal_exp_release_attitude_annex",
    "export_cal_exp_release_attitude_annex",
    "build_release_attitude_strip",
    "extract_release_attitude_strip_signal",
    "attach_release_attitude_strip_to_evidence",
    "release_attitude_strip_for_alignment_view",
    "summarize_envelope_v4_for_uplift_council",
]

