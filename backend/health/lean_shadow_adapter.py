"""Lean shadow mode adapter for global health.

STATUS: PHASE 1 — LEAN SHADOW TILE FOR GLOBAL HEALTH

Provides integration between Lean shadow mode capability radar and the global
health surface builder. This tile provides structural integrity signals for
P3 (synthetic) and P4 (real shadow) verification layers.

P3 NARRATIVE (Synthetic Lean Evaluation):
- In P3 mode, Lean shadow checks internal logical structural coherence
- The shadow mode validates that synthetic verification pathways maintain
  structural integrity across formula complexity bands
- This provides early warning signals for structural degradation in synthetic
  verification workflows before real-runner integration

P4 NARRATIVE (Real-Runner + Lean Shadow Observations):
- In P4 mode, Lean shadow checks regression in real-runner Lean pathways
- The shadow mode observes real Lean subprocess execution and compares against
  expected structural patterns
- This detects regressions and structural drift in live verification pathways
  without interfering with actual verification outcomes

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The lean_shadow tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
- No abort logic, no control feedback, no modification of pipeline/runner decisions
- Tile is observational only; never a control surface
"""

from typing import Any, Dict, List, Optional, Sequence

LEAN_SHADOW_TILE_SCHEMA_VERSION = "1.0.0"
CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION = "1.0.0"
STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION = "1.0.0"


def _validate_shadow_radar(radar: Dict[str, Any]) -> None:
    """Validate shadow radar structure.
    
    Args:
        radar: Shadow radar dictionary from build_lean_shadow_capability_radar()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["structural_error_rate", "shadow_resource_band", "anomaly_signatures"]
    missing = [key for key in required_keys if key not in radar]
    if missing:
        raise ValueError(
            f"shadow_radar missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(radar.keys()))}"
        )


def _determine_shadow_status(
    structural_error_rate: float,
    shadow_resource_band: str,
) -> str:
    """
    Determine shadow status based on error rate and resource band.
    
    Status Logic:
    - BLOCK: structural_error_rate > 0.5 OR shadow_resource_band == "HIGH"
    - WARN: structural_error_rate >= 0.2 OR shadow_resource_band == "MEDIUM"
    - OK: otherwise
    
    Args:
        structural_error_rate: Ratio of structural errors (0.0-1.0)
        shadow_resource_band: Resource band classification
    
    Returns:
        "OK" | "WARN" | "BLOCK"
    """
    if structural_error_rate > 0.5 or shadow_resource_band == "HIGH":
        return "BLOCK"
    elif structural_error_rate >= 0.2 or shadow_resource_band == "MEDIUM":
        return "WARN"
    else:
        return "OK"


def _build_shadow_headline(
    status: str,
    structural_error_rate: float,
    shadow_resource_band: str,
    total_requests: int,
) -> str:
    """
    Build neutral headline for shadow tile.
    
    Args:
        status: Shadow status ("OK" | "WARN" | "BLOCK")
        structural_error_rate: Structural error rate
        shadow_resource_band: Resource band
        total_requests: Total shadow requests processed
    
    Returns:
        Neutral headline string
    """
    if total_requests == 0:
        return "No Lean shadow mode activity recorded."
    
    # Build base headline
    error_rate_pct = structural_error_rate * 100
    headline = (
        f"Lean shadow mode: {total_requests} requests processed, "
        f"{error_rate_pct:.1f}% structural error rate, "
        f"{shadow_resource_band} resource band."
    )
    
    # Add status context
    if status == "BLOCK":
        headline += " Status: BLOCK."
    elif status == "WARN":
        headline += " Status: WARN."
    else:
        headline += " Status: OK."
    
    return headline


def build_lean_shadow_tile_for_global_health(
    shadow_radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Lean shadow tile for global health surface.

    STATUS: PHASE 1 — LEAN SHADOW TILE FOR GLOBAL HEALTH

    Maps shadow capability radar output into a global health tile that provides
    structural integrity signals for P3 (synthetic) and P4 (real shadow)
    verification layers.

    P3 NARRATIVE (Synthetic Lean Evaluation):
    - Lean shadow checks internal logical structural coherence
    - Validates that synthetic verification pathways maintain structural integrity
    - Provides early warning signals for structural degradation in synthetic workflows

    P4 NARRATIVE (Real-Runner + Lean Shadow Observations):
    - Lean shadow checks regression in real-runner Lean pathways
    - Observes real Lean subprocess execution and compares against expected patterns
    - Detects regressions and structural drift in live verification pathways

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - Compatible with both P3 (synthetic Lean evaluation) and P4 (real-runner + Lean shadow observations)
    - No abort logic, no control feedback, no modification of pipeline/runner decisions
    - Tile is observational only; never a control surface

    Args:
        shadow_radar: Shadow capability radar from build_lean_shadow_capability_radar().
            Must contain:
            - structural_error_rate: float (0.0-1.0)
            - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
            - anomaly_signatures: List[str]
            - total_shadow_requests: int (optional, defaults to 0)

    Returns:
        Lean shadow health tile dictionary with:
        - schema_version: "1.0.0"
        - status: "OK" | "WARN" | "BLOCK"
        - structural_error_rate: float
        - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
        - dominant_anomalies: List[str] (top 3 anomaly signatures)
        - headline: str (neutral summary)
    
    Example:
        >>> radar = {
        ...     "structural_error_rate": 0.15,
        ...     "shadow_resource_band": "LOW",
        ...     "anomaly_signatures": ["abc123", "def456"],
        ...     "total_shadow_requests": 100,
        ... }
        >>> tile = build_lean_shadow_tile_for_global_health(radar)
        >>> tile["status"]
        'OK'
    """
    # Validate input
    _validate_shadow_radar(shadow_radar)
    
    # Extract radar fields
    structural_error_rate = float(shadow_radar.get("structural_error_rate", 0.0))
    shadow_resource_band = shadow_radar.get("shadow_resource_band", "LOW")
    anomaly_signatures = shadow_radar.get("anomaly_signatures", [])
    total_requests = shadow_radar.get("total_shadow_requests", 0)
    
    # Validate resource band
    if shadow_resource_band not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(
            f"Invalid shadow_resource_band: {shadow_resource_band}. "
            f"Must be one of: LOW, MEDIUM, HIGH"
        )
    
    # Determine status
    status = _determine_shadow_status(structural_error_rate, shadow_resource_band)
    
    # Get dominant anomalies (top 3)
    dominant_anomalies = anomaly_signatures[:3] if anomaly_signatures else []
    
    # Build headline
    headline = _build_shadow_headline(
        status,
        structural_error_rate,
        shadow_resource_band,
        total_requests,
    )
    
    # Build tile
    tile = {
        "schema_version": LEAN_SHADOW_TILE_SCHEMA_VERSION,
        "status": status,
        "structural_error_rate": round(structural_error_rate, 3),
        "shadow_resource_band": shadow_resource_band,
        "dominant_anomalies": dominant_anomalies,
        "headline": headline,
    }
    
    return tile


def build_first_light_lean_shadow_summary(
    shadow_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First Light structural summary from Lean shadow tile.

    STATUS: PHASE X — FIRST LIGHT STRUCTURE ANNEX

    This function creates an evidence-ready structural summary that explains
    if the proof pipeline itself is stable. The summary is designed for
    inclusion in First Light evidence packs as a structural annex.

    **For External Readers:** This summary is intended as a structural health
    annex for the proof pipeline, not a hard gate.

    FIRST LIGHT NARRATIVE:
    - Explains if the proof pipeline itself is stable
    - Provides structural integrity signals for proof generation pathways
    - Documents anomaly patterns that may indicate pipeline instability

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Purely observational; never a control surface

    Args:
        shadow_tile: Lean shadow tile from build_lean_shadow_tile_for_global_health().
            Must contain:
            - status: "OK" | "WARN" | "BLOCK"
            - structural_error_rate: float
            - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
            - dominant_anomalies: List[str] (optional)

    Returns:
        First Light structural summary dictionary with:
        - schema_version: "1.0.0"
        - status: "OK" | "WARN" | "BLOCK"
        - structural_error_rate: float
        - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
        - dominant_anomalies: List[str] (top 5 anomaly signatures)

    Example:
        >>> tile = build_lean_shadow_tile_for_global_health(radar)
        >>> summary = build_first_light_lean_shadow_summary(tile)
        >>> summary["status"]
        'OK'
        >>> len(summary["dominant_anomalies"]) <= 5
        True
    """
    # Extract key fields from shadow tile
    status = shadow_tile.get("status", "OK")
    structural_error_rate = shadow_tile.get("structural_error_rate", 0.0)
    shadow_resource_band = shadow_tile.get("shadow_resource_band", "LOW")
    dominant_anomalies = shadow_tile.get("dominant_anomalies", [])
    
    # Truncate anomalies to top 5 for First Light summary
    truncated_anomalies = dominant_anomalies[:5] if dominant_anomalies else []
    
    # Build summary
    summary = {
        "schema_version": "1.0.0",
        "status": status,
        "structural_error_rate": structural_error_rate,
        "shadow_resource_band": shadow_resource_band,
        "dominant_anomalies": truncated_anomalies,
    }
    
    return summary


def attach_cal_exp_structural_summary_to_report(
    report: Dict[str, Any],
    lean_tile_sequence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Attach calibration experiment structural summary to a calibration report.

    STATUS: PHASE X — P5 STRUCTURAL REGIME MONITOR

    This helper attaches the Lean shadow structural summary to CAL-EXP reports
    (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3) under report["structural_summary"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        report: Existing calibration experiment report (read-only, not modified).
        lean_tile_sequence: Sequence of Lean shadow tiles from the experiment.

    Returns:
        New dict with report contents plus structural_summary attached.

    Example:
        >>> report = {"schema_version": "1.0.0", "summary": {...}}
        >>> tiles = [tile1, tile2, tile3]
        >>> enriched = attach_cal_exp_structural_summary_to_report(report, tiles)
        >>> "structural_summary" in enriched
        True
    """
    # Create a copy to avoid mutating the original
    enriched = report.copy()
    
    # Build structural summary
    structural_summary = build_cal_exp_structural_summary(lean_tile_sequence)
    
    # Attach to report
    enriched["structural_summary"] = structural_summary
    
    return enriched


def build_structural_calibration_panel(
    cal_exp_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build structural calibration panel from multiple calibration experiment reports.

    STATUS: PHASE X — P5 STRUCTURAL REGIME MONITOR

    This function aggregates structural summaries from multiple calibration
    experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3) into a single panel for
    cross-experiment analysis.

    P5 NARRATIVE:
    - Provides dashboard view of structural health across calibration experiments
    - Detects patterns (bursts, conflicts) across experiments
    - Enables quick assessment of proof pipeline stability across calibration runs

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned panel is purely observational
    - No control flow depends on the panel contents
    - Advisory only; no gating

    Args:
        cal_exp_reports: List of calibration experiment report dictionaries.
            Each report should contain:
            - cal_id: str (experiment identifier, e.g., "CAL-EXP-1")
            - structural_summary: Dict (from build_cal_exp_structural_summary())
            - Optional: governance.structure.lean_cross_check (from compare_lean_vs_structural_signal())

    Returns:
        Structural calibration panel dictionary with:
        - schema_version: "1.0.0"
        - experiments: List[Dict] with cal_id, mean_error_rate, max_error_rate, burst_detected, cross_check_status
        - counts: Dict with num_experiments, num_burst, num_conflict

    Example:
        >>> report1 = {
        ...     "cal_id": "CAL-EXP-1",
        ...     "structural_summary": {"mean_structural_error_rate": 0.1, "max_structural_error_rate": 0.2, "anomaly_bursts": []},
        ... }
        >>> report2 = {
        ...     "cal_id": "CAL-EXP-2",
        ...     "structural_summary": {"mean_structural_error_rate": 0.3, "max_structural_error_rate": 0.4, "anomaly_bursts": [{"start_index": 0}]},
        ... }
        >>> panel = build_structural_calibration_panel([report1, report2])
        >>> panel["counts"]["num_experiments"]
        2
        >>> panel["counts"]["num_burst"]
        1
    """
    experiments: List[Dict[str, Any]] = []
    num_burst = 0
    num_conflict = 0
    
    for report in cal_exp_reports:
        # Extract cal_id (try multiple possible keys)
        cal_id = report.get("cal_id")
        if not cal_id:
            cal_id = report.get("experiment_id")
        if not cal_id:
            run_id = report.get("run_id")
            if run_id:
                # Extract prefix from run_id (e.g., "cal_exp3_20250101_120000" -> "cal_exp3")
                # Take first two parts if they exist, otherwise just first part
                parts = run_id.split("_")
                if len(parts) >= 2:
                    cal_id = "_".join(parts[:2])  # "cal_exp3"
                else:
                    cal_id = parts[0]  # Just "cal" if no underscores
        if not cal_id:
            # Fallback to generated ID
            cal_id = f"CAL-EXP-{len(experiments) + 1}"
        
        # Extract structural summary
        structural_summary = report.get("structural_summary", {})
        mean_error_rate = structural_summary.get("mean_structural_error_rate", 0.0)
        max_error_rate = structural_summary.get("max_structural_error_rate", 0.0)
        anomaly_bursts = structural_summary.get("anomaly_bursts", [])
        burst_detected = len(anomaly_bursts) > 0
        
        if burst_detected:
            num_burst += 1
        
        # Extract cross-check status (if available)
        # Harden to one of: CONSISTENT / TENSION / CONFLICT / UNKNOWN
        cross_check_status: Optional[str] = None
        governance = report.get("governance", {})
        structure = governance.get("structure", {})
        lean_cross_check = structure.get("lean_cross_check")
        if lean_cross_check:
            raw_status = lean_cross_check.get("status")
            # Normalize to valid values
            if raw_status in ("CONSISTENT", "TENSION", "CONFLICT"):
                cross_check_status = raw_status
            else:
                cross_check_status = "UNKNOWN"
            if cross_check_status == "CONFLICT":
                num_conflict += 1
        else:
            # Missing cross-check -> UNKNOWN (do not drop experiment)
            cross_check_status = "UNKNOWN"
        
        experiments.append({
            "cal_id": cal_id,
            "mean_error_rate": round(mean_error_rate, 3),
            "max_error_rate": round(max_error_rate, 3),
            "burst_detected": burst_detected,
            "cross_check_status": cross_check_status,
        })
    
    return {
        "schema_version": STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION,
        "experiments": experiments,
        "counts": {
            "num_experiments": len(experiments),
            "num_burst": num_burst,
            "num_conflict": num_conflict,
        },
    }


def extract_structural_calibration_panel_signal(
    calibration_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact structural calibration panel signal for status integration.

    STATUS: PHASE X — P5 STRUCTURAL REGIME MONITOR

    This function extracts a minimal signal from the structural calibration panel
    for inclusion in First Light status files.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        calibration_panel: Structural calibration panel from build_structural_calibration_panel().

    Returns:
        Compact signal dictionary with:
        - num_conflict: int
        - num_burst: int
        - last_cross_check_status: str | None (status from last experiment)
        - most_recent_cal_id: str | None (cal_id from last experiment)

    Example:
        >>> panel = build_structural_calibration_panel([report1, report2])
        >>> signal = extract_structural_calibration_panel_signal(panel)
        >>> signal["num_conflict"]
        0
    """
    counts = calibration_panel.get("counts", {})
    experiments = calibration_panel.get("experiments", [])
    
    # Get last cross-check status and most recent cal_id
    last_cross_check_status: Optional[str] = None
    most_recent_cal_id: Optional[str] = None
    for exp in reversed(experiments):  # Start from last experiment
        status = exp.get("cross_check_status")
        cal_id = exp.get("cal_id")
        if status:
            last_cross_check_status = status
        if cal_id:
            most_recent_cal_id = cal_id
        if last_cross_check_status and most_recent_cal_id:
            break  # Found both, no need to continue
    
    return {
        "num_conflict": counts.get("num_conflict", 0),
        "num_burst": counts.get("num_burst", 0),
        "last_cross_check_status": last_cross_check_status,
        "most_recent_cal_id": most_recent_cal_id,
    }


def attach_lean_shadow_to_evidence(
    evidence: Dict[str, Any],
    shadow_tile: Dict[str, Any],
    include_first_light_summary: bool = False,
    lean_cross_check: Optional[Dict[str, Any]] = None,
    structural_calibration_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach Lean shadow tile to an evidence pack (read-only, additive).

    STATUS: PHASE X — EVIDENCE PACK INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["lean_shadow"].

    P3 NARRATIVE (Synthetic Lean Evaluation):
    - Evidence includes structural coherence signals from synthetic verification
    - Documents internal logical structural integrity across complexity bands

    P4 NARRATIVE (Real-Runner + Lean Shadow Observations):
    - Evidence includes regression detection signals from real-runner observations
    - Documents structural drift and anomaly patterns in live verification pathways

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input
    - Tile is observational only; never a control surface

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        shadow_tile: Lean shadow tile from build_lean_shadow_tile_for_global_health().

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        shadow_tile: Lean shadow tile from build_lean_shadow_tile_for_global_health().
        include_first_light_summary: If True, include first_light_summary in the attached tile.
        lean_cross_check: Optional cross-check comparison from compare_lean_vs_structural_signal().
            If provided, attaches to evidence["governance"]["structure"]["lean_cross_check"].
        structural_calibration_panel: Optional structural calibration panel from build_structural_calibration_panel().
            If provided, attaches to evidence["governance"]["structure"]["calibration_panel"].

    Returns:
        New dict with evidence contents plus lean_shadow tile attached under governance key.
        The attached tile includes:
        - status: "OK" | "WARN" | "BLOCK"
        - structural_error_rate: float
        - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
        - dominant_anomalies: List[str] (top 3)
        - first_light_summary: Dict[str, Any] (optional, if include_first_light_summary=True)

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> radar = {"structural_error_rate": 0.1, "shadow_resource_band": "LOW", ...}
        >>> tile = build_lean_shadow_tile_for_global_health(radar)
        >>> enriched = attach_lean_shadow_to_evidence(evidence, tile)
        >>> "governance" in enriched
        True
        >>> "lean_shadow" in enriched["governance"]
        True
        >>> enriched = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=True)
        >>> "first_light_summary" in enriched["governance"]["lean_shadow"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach lean shadow tile (extract key fields for evidence pack)
    enriched["governance"] = enriched["governance"].copy()
    lean_shadow_data = {
        "status": shadow_tile.get("status", "OK"),
        "structural_error_rate": shadow_tile.get("structural_error_rate", 0.0),
        "shadow_resource_band": shadow_tile.get("shadow_resource_band", "LOW"),
        "dominant_anomalies": shadow_tile.get("dominant_anomalies", []),
    }
    
    # Optionally add First Light summary
    if include_first_light_summary:
        lean_shadow_data["first_light_summary"] = build_first_light_lean_shadow_summary(shadow_tile)
    
    enriched["governance"]["lean_shadow"] = lean_shadow_data
    
    # Optionally add Lean vs Structural cross-check
    if lean_cross_check is not None:
        if "governance" not in enriched:
            enriched["governance"] = {}
        if "structure" not in enriched["governance"]:
            enriched["governance"]["structure"] = {}
        enriched["governance"]["structure"]["lean_cross_check"] = lean_cross_check
    
    # Optionally add structural calibration panel
    if structural_calibration_panel is not None:
        if "governance" not in enriched:
            enriched["governance"] = {}
        if "structure" not in enriched["governance"]:
            enriched["governance"]["structure"] = {}
        enriched["governance"]["structure"]["calibration_panel"] = structural_calibration_panel
    
    return enriched


def build_cal_exp_structural_summary(
    lean_tile_sequence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build per-experiment structural summary from a sequence of Lean shadow tiles.

    STATUS: PHASE X — P5 STRUCTURAL REGIME MONITOR

    This function aggregates Lean shadow tiles across a calibration experiment
    to produce a structural regime summary. It detects anomaly bursts and
    distinguishes single spikes from sustained bursts.

    P5 NARRATIVE:
    - Monitors proof pipeline structural health across calibration experiments
    - Detects regime changes (sustained anomalies vs isolated spikes)
    - Provides evidence for structural stability assessment

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Purely observational; never a control surface

    Args:
        lean_tile_sequence: Sequence of Lean shadow tiles from build_lean_shadow_tile_for_global_health().
            Each tile must contain:
            - structural_error_rate: float
            - shadow_resource_band: "LOW" | "MEDIUM" | "HIGH"
            - dominant_anomalies: List[str] (optional)

    Returns:
        Calibration experiment structural summary dictionary with:
        - schema_version: "1.0.0"
        - mean_structural_error_rate: float (mean across all tiles)
        - max_structural_error_rate: float (maximum across all tiles)
        - anomaly_bursts: List[Dict] (detected burst patterns)
        - dominant_anomalies: List[str] (aggregated top anomalies across experiment)

    Example:
        >>> tiles = [
        ...     {"structural_error_rate": 0.1, "shadow_resource_band": "LOW", "dominant_anomalies": ["a"]},
        ...     {"structural_error_rate": 0.3, "shadow_resource_band": "MEDIUM", "dominant_anomalies": ["b"]},
        ... ]
        >>> summary = build_cal_exp_structural_summary(tiles)
        >>> summary["mean_structural_error_rate"]
        0.2
        >>> summary["max_structural_error_rate"]
        0.3
    """
    if not lean_tile_sequence:
        return {
            "schema_version": CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION,
            "mean_structural_error_rate": 0.0,
            "max_structural_error_rate": 0.0,
            "anomaly_bursts": [],
            "dominant_anomalies": [],
        }
    
    # Extract error rates
    error_rates = [
        float(tile.get("structural_error_rate", 0.0))
        for tile in lean_tile_sequence
    ]
    
    # Compute statistics
    mean_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0.0
    max_error_rate = max(error_rates) if error_rates else 0.0
    
    # Detect anomaly bursts (sustained high error rates)
    # A burst is defined as 3+ consecutive tiles with error_rate >= 0.3
    anomaly_bursts: List[Dict[str, Any]] = []
    burst_start: int | None = None
    
    for i, error_rate in enumerate(error_rates):
        if error_rate >= 0.3:
            if burst_start is None:
                burst_start = i
        else:
            if burst_start is not None:
                burst_length = i - burst_start
                if burst_length >= 3:  # Sustained burst (3+ consecutive)
                    anomaly_bursts.append({
                        "start_index": burst_start,
                        "end_index": i - 1,
                        "length": burst_length,
                        "mean_error_rate": sum(error_rates[burst_start:i]) / burst_length,
                        "max_error_rate": max(error_rates[burst_start:i]),
                    })
                burst_start = None
    
    # Handle burst at end of sequence
    if burst_start is not None:
        burst_length = len(error_rates) - burst_start
        if burst_length >= 3:
            anomaly_bursts.append({
                "start_index": burst_start,
                "end_index": len(error_rates) - 1,
                "length": burst_length,
                "mean_error_rate": sum(error_rates[burst_start:]) / burst_length,
                "max_error_rate": max(error_rates[burst_start:]),
            })
    
    # Aggregate dominant anomalies across all tiles
    all_anomalies: Dict[str, int] = {}
    for tile in lean_tile_sequence:
        anomalies = tile.get("dominant_anomalies", [])
        for anomaly in anomalies:
            all_anomalies[anomaly] = all_anomalies.get(anomaly, 0) + 1
    
    # Sort by frequency and take top 10
    sorted_anomalies = sorted(
        all_anomalies.items(),
        key=lambda x: (x[1], x[0]),  # Sort by frequency, then by anomaly ID
        reverse=True,
    )
    dominant_anomalies = [anomaly for anomaly, _ in sorted_anomalies[:10]]
    
    return {
        "schema_version": CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION,
        "mean_structural_error_rate": round(mean_error_rate, 3),
        "max_structural_error_rate": round(max_error_rate, 3),
        "anomaly_bursts": anomaly_bursts,
        "dominant_anomalies": dominant_anomalies,
    }


def compare_lean_vs_structural_signal(
    lean_summary: Dict[str, Any],
    structural_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare Lean shadow signal with structural cohesion signal.

    STATUS: PHASE X — P5 STRUCTURAL CROSS-CHECK

    This function compares the Lean shadow structural summary with the structural
    cohesion summary to detect consistency, tension, or conflict between signals.

    P5 NARRATIVE:
    - Cross-checks proof pipeline structural health (Lean) vs architectural
      structural health (DAG/Topology/HT)
    - Detects alignment or misalignment between verification and architecture layers
    - Provides advisory notes for human review

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned comparison is purely observational
    - No control flow depends on the comparison contents
    - Advisory notes only; no gating

    Args:
        lean_summary: Calibration experiment structural summary from build_cal_exp_structural_summary().
            Must contain:
            - mean_structural_error_rate: float
            - max_structural_error_rate: float
            - anomaly_bursts: List[Dict]
        structural_summary: Structural cohesion summary (from structural governance signal).
            Expected to contain:
            - combined_severity: "CONSISTENT" | "TENSION" | "CONFLICT"
            - cohesion_score: float (0.0-1.0)
            - dag_status, topology_status, ht_status: str

    Returns:
        Cross-check comparison dictionary with:
        - schema_version: "1.0.0"
        - status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - lean_signal_severity: "OK" | "WARN" | "BLOCK" (derived from lean summary)
        - structural_signal_severity: "CONSISTENT" | "TENSION" | "CONFLICT" (from structural summary)
        - advisory_notes: List[str] (human-readable notes)

    Example:
        >>> lean_summary = {"mean_structural_error_rate": 0.1, "max_structural_error_rate": 0.2, "anomaly_bursts": []}
        >>> structural_summary = {"combined_severity": "CONSISTENT", "cohesion_score": 0.95}
        >>> comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        >>> comparison["status"]
        'CONSISTENT'
    """
    # Determine Lean signal severity
    mean_error = lean_summary.get("mean_structural_error_rate", 0.0)
    max_error = lean_summary.get("max_structural_error_rate", 0.0)
    has_bursts = len(lean_summary.get("anomaly_bursts", [])) > 0
    
    if max_error > 0.5 or has_bursts:
        lean_severity = "BLOCK"
    elif mean_error >= 0.2 or max_error >= 0.3:
        lean_severity = "WARN"
    else:
        lean_severity = "OK"
    
    # Get structural signal severity
    structural_severity = structural_summary.get("combined_severity", "CONSISTENT")
    cohesion_score = structural_summary.get("cohesion_score", 1.0)
    
    # Determine comparison status
    # CONSISTENT: Both signals agree (both OK/CONSISTENT or both WARN/TENSION or both BLOCK/CONFLICT)
    # TENSION: Signals partially disagree (one is worse than the other but not opposite)
    # CONFLICT: Signals strongly disagree (one is OK/CONSISTENT while other is BLOCK/CONFLICT)
    
    status: str
    advisory_notes: List[str] = []
    
    if lean_severity == "OK" and structural_severity == "CONSISTENT":
        status = "CONSISTENT"
        advisory_notes.append("Both Lean shadow and structural cohesion signals indicate healthy state.")
    elif lean_severity == "WARN" and structural_severity == "TENSION":
        status = "CONSISTENT"
        advisory_notes.append("Both signals indicate moderate concerns; signals are aligned.")
    elif lean_severity == "BLOCK" and structural_severity == "CONFLICT":
        status = "CONSISTENT"
        advisory_notes.append("Both signals indicate significant issues; signals are aligned.")
    elif (lean_severity == "OK" and structural_severity == "CONFLICT") or \
         (lean_severity == "BLOCK" and structural_severity == "CONSISTENT"):
        status = "CONFLICT"
        advisory_notes.append(
            f"Signals strongly disagree: Lean shadow is {lean_severity} but structural cohesion is {structural_severity}. "
            "This suggests a potential mismatch between verification pipeline and architectural layers."
        )
    else:
        status = "TENSION"
        advisory_notes.append(
            f"Signals partially disagree: Lean shadow is {lean_severity} while structural cohesion is {structural_severity}. "
            "Review both signals for context."
        )
    
    # Add specific notes based on metrics
    if mean_error > 0.3:
        advisory_notes.append(
            f"Lean shadow shows elevated mean error rate ({mean_error:.1%}). "
            "Consider reviewing verification pipeline stability."
        )
    
    if cohesion_score < 0.7:
        advisory_notes.append(
            f"Structural cohesion score is low ({cohesion_score:.2f}). "
            "Review DAG/Topology/HT layer consistency."
        )
    
    if has_bursts:
        burst_count = len(lean_summary.get("anomaly_bursts", []))
        advisory_notes.append(
            f"Detected {burst_count} anomaly burst(s) in Lean shadow. "
            "This indicates sustained structural issues rather than isolated spikes."
        )
    
    return {
        "schema_version": "1.0.0",
        "status": status,
        "lean_signal_severity": lean_severity,
        "structural_signal_severity": structural_severity,
        "advisory_notes": advisory_notes,
    }


__all__ = [
    "LEAN_SHADOW_TILE_SCHEMA_VERSION",
    "CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION",
    "STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION",
    "build_lean_shadow_tile_for_global_health",
    "build_first_light_lean_shadow_summary",
    "build_cal_exp_structural_summary",
    "compare_lean_vs_structural_signal",
    "build_structural_calibration_panel",
    "extract_structural_calibration_panel_signal",
    "attach_cal_exp_structural_summary_to_report",
    "attach_lean_shadow_to_evidence",
]

