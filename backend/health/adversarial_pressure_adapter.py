"""Adversarial pressure adapter for global health.

STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE TILE

Provides integration between Phase V adversarial pressure model, scenario evolution plan,
and failover plan v2 components for the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The adversarial_governance tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- No modification of adversarial test state or metric promotion decisions
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"
ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION = "1.0.0"
ADVERSARIAL_COVERAGE_GRID_SCHEMA_VERSION = "1.0.0"
ADVERSARIAL_PRIORITY_SCENARIO_LEDGER_SCHEMA_VERSION = "1.0.0"


def _validate_pressure_model(pressure_model: Dict[str, Any]) -> None:
    """Validate pressure model structure.
    
    Args:
        pressure_model: Pressure model dictionary from build_adversarial_pressure_model()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["metric_pressure_scores", "global_pressure_index", "pressure_band"]
    missing = [key for key in required_keys if key not in pressure_model]
    if missing:
        raise ValueError(
            f"pressure_model missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(pressure_model.keys()))}"
        )


def _validate_scenario_plan(scenario_plan: Dict[str, Any]) -> None:
    """Validate scenario plan structure.
    
    Args:
        scenario_plan: Scenario plan dictionary from build_evolving_adversarial_scenario_plan()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["scenario_backlog", "priority_order"]
    missing = [key for key in required_keys if key not in scenario_plan]
    if missing:
        raise ValueError(
            f"scenario_plan missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(scenario_plan.keys()))}"
        )


def _validate_failover_plan_v2(failover_plan_v2: Dict[str, Any]) -> None:
    """Validate failover plan v2 structure.
    
    Args:
        failover_plan_v2: Failover plan v2 dictionary from build_adversarial_failover_plan_v2()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["has_failover", "status", "metrics_without_failover"]
    missing = [key for key in required_keys if key not in failover_plan_v2]
    if missing:
        raise ValueError(
            f"failover_plan_v2 missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(failover_plan_v2.keys()))}"
        )


def build_adversarial_governance_tile(
    pressure_model: Dict[str, Any],
    scenario_plan: Dict[str, Any],
    failover_plan_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build adversarial governance tile for global health surface.

    STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE TILE

    Integrates Phase V adversarial pressure model, scenario evolution plan, and failover plan v2
    components into a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - No modification of adversarial test state or metric promotion decisions

    Args:
        pressure_model: Pressure model from build_adversarial_pressure_model().
            Must contain: metric_pressure_scores, global_pressure_index, pressure_band
        scenario_plan: Scenario plan from build_evolving_adversarial_scenario_plan().
            Must contain: scenario_backlog, priority_order
        failover_plan_v2: Failover plan v2 from build_adversarial_failover_plan_v2().
            Must contain: has_failover, status, metrics_without_failover

    Returns:
        Adversarial governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - global_pressure_index: float (0.0-1.0)
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - priority_scenarios: List[str] (top 3 scenario names)
        - has_failover: bool
        - metrics_without_failover: List[str]
        - headline: str (neutral summary)
    
    Example:
        >>> pressure_model = {
        ...     "global_pressure_index": 0.32,
        ...     "pressure_band": "LOW",
        ...     "metric_pressure_scores": {"goal_hit": 0.2},
        ... }
        >>> scenario_plan = {
        ...     "scenario_backlog": [{"name": "goal_hit_critical"}],
        ...     "priority_order": ["goal_hit_critical"],
        ... }
        >>> failover_plan_v2 = {
        ...     "has_failover": True,
        ...     "status": "OK",
        ...     "metrics_without_failover": [],
        ... }
        >>> tile = build_adversarial_governance_tile(
        ...     pressure_model, scenario_plan, failover_plan_v2
        ... )
        >>> tile["status_light"]
        'GREEN'
    """
    # Validate inputs
    _validate_pressure_model(pressure_model)
    _validate_scenario_plan(scenario_plan)
    _validate_failover_plan_v2(failover_plan_v2)
    
    # Extract fields
    global_pressure_index = pressure_model.get("global_pressure_index", 0.0)
    pressure_band = pressure_model.get("pressure_band", "LOW")
    failover_status = failover_plan_v2.get("status", "OK")
    has_failover = failover_plan_v2.get("has_failover", True)
    metrics_without_failover = failover_plan_v2.get("metrics_without_failover", [])
    
    # Get top 3 priority scenarios
    priority_order = scenario_plan.get("priority_order", [])
    priority_scenarios = priority_order[:3]
    
    # Determine status_light
    # RED if status == "BLOCK" or global_pressure_index > 0.7
    # YELLOW if "WARN" or global_pressure_index > 0.4
    # GREEN otherwise
    if failover_status == "BLOCK" or global_pressure_index > 0.7:
        status_light = "RED"
    elif failover_status == "WARN" or global_pressure_index > 0.4:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Validate status_light
    if status_light not in ("GREEN", "YELLOW", "RED"):
        raise ValueError(
            f"Invalid status_light: {status_light}. "
            f"Must be one of: GREEN, YELLOW, RED"
        )
    
    # Validate pressure_band
    if pressure_band not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(
            f"Invalid pressure_band: {pressure_band}. "
            f"Must be one of: LOW, MEDIUM, HIGH"
        )
    
    # Build headline (neutral summary)
    if status_light == "RED":
        if failover_status == "BLOCK":
            headline = f"Adversarial governance: {len(metrics_without_failover)} metrics blocking promotion"
        else:
            headline = f"Adversarial governance: High pressure (index {global_pressure_index:.2f})"
    elif status_light == "YELLOW":
        if failover_status == "WARN":
            headline = f"Adversarial governance: {len(metrics_without_failover)} metrics with sparse coverage"
        else:
            headline = f"Adversarial governance: Moderate pressure (index {global_pressure_index:.2f})"
    else:
        headline = f"Adversarial governance: Low pressure (index {global_pressure_index:.2f}), {len(priority_scenarios)} priority scenarios"
    
    # Build tile
    tile = {
        "schema_version": ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "global_pressure_index": round(global_pressure_index, 3),
        "pressure_band": pressure_band,
        "priority_scenarios": priority_scenarios,
        "has_failover": has_failover,
        "metrics_without_failover": sorted(metrics_without_failover),
        "headline": headline,
    }
    
    return tile


def extract_adversarial_signal_for_release(
    pressure_model: Dict[str, Any],
    failover_plan_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact adversarial signal for release gating.

    STATUS: PHASE X — RELEASE READINESS HOOK

    Provides a compact, advisory-only signal for release/CI harnesses.
    This signal does NOT block releases; it is purely informational.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is advisory-only
    - No control flow should depend on this signal for blocking releases
    - This is purely for observability and logging

    Args:
        pressure_model: Pressure model from build_adversarial_pressure_model().
        failover_plan_v2: Failover plan v2 from build_adversarial_failover_plan_v2().

    Returns:
        Compact adversarial signal dictionary with:
        - global_pressure_index: float (0.0-1.0)
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - has_failover: bool
        - metrics_without_failover: List[str]
    
    Example:
        >>> pressure_model = {
        ...     "global_pressure_index": 0.32,
        ...     "pressure_band": "LOW",
        ... }
        >>> failover_plan_v2 = {
        ...     "has_failover": True,
        ...     "metrics_without_failover": [],
        ... }
        >>> signal = extract_adversarial_signal_for_release(pressure_model, failover_plan_v2)
        >>> signal["pressure_band"]
        'LOW'
    """
    # Extract fields (gracefully handle missing keys)
    global_pressure_index = pressure_model.get("global_pressure_index", 0.0)
    pressure_band = pressure_model.get("pressure_band", "LOW")
    has_failover = failover_plan_v2.get("has_failover", True)
    metrics_without_failover = failover_plan_v2.get("metrics_without_failover", [])
    
    return {
        "global_pressure_index": round(global_pressure_index, 3),
        "pressure_band": pressure_band,
        "has_failover": has_failover,
        "metrics_without_failover": sorted(metrics_without_failover),
    }


def attach_adversarial_governance_to_evidence(
    evidence: Dict[str, Any],
    governance_tile: Dict[str, Any],
    release_signal: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach adversarial governance tile to an evidence pack (read-only, additive).

    STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["adversarial"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        governance_tile: Adversarial governance tile from build_adversarial_governance_tile().
        release_signal: Release signal from extract_adversarial_signal_for_release().
        p3_summary: Optional P3 adversarial pressure summary from extract_adversarial_summary_for_p3_stability().
        p4_calibration: Optional P4 adversarial calibration from extract_adversarial_calibration_for_p4().

    Returns:
        New dict with evidence contents plus adversarial tile attached under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_adversarial_governance_tile(pressure_model, scenario_plan, failover_plan_v2)
        >>> signal = extract_adversarial_signal_for_release(pressure_model, failover_plan_v2)
        >>> enriched = attach_adversarial_governance_to_evidence(evidence, tile, signal)
        >>> "governance" in enriched
        True
        >>> "adversarial" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach adversarial governance tile
    enriched["governance"] = enriched["governance"].copy()
    adversarial_data = {
        "tile": governance_tile,
        "release_signal": release_signal,
        "global_pressure_index": governance_tile.get("global_pressure_index", 0.0),
        "pressure_band": governance_tile.get("pressure_band", "LOW"),
        "priority_scenarios": governance_tile.get("priority_scenarios", []),
        "has_failover": governance_tile.get("has_failover", True),
        "metrics_without_failover": governance_tile.get("metrics_without_failover", []),
    }
    
    # Attach First-Light coverage annex if P3 and P4 data are provided
    if p3_summary is not None and p4_calibration is not None:
        adversarial_data["first_light_coverage"] = build_first_light_adversarial_coverage_annex(
            p3_summary, p4_calibration
        )
    
    enriched["governance"]["adversarial"] = adversarial_data
    
    return enriched


def summarize_adversarial_for_uplift_council(tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize adversarial governance tile for Uplift Council.

    STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE

    Maps adversarial pressure signals to council decision status:
    - HIGH pressure AND core metrics lack failover → always BLOCK (tightened rule)
    - HIGH pressure OR missing failover for core metrics → BLOCK
    - MEDIUM pressure → WARN
    - Otherwise → OK

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is advisory-only
    - No control flow should depend on this summary for blocking promotions
    - This is purely for observability and logging

    Args:
        tile: Adversarial governance tile from build_adversarial_governance_tile().

    Returns:
        Council summary dictionary with:
        - status: "OK" | "WARN" | "BLOCK"
        - priority_scenarios: List[str] (top priority scenarios)
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - has_failover: bool
        - metrics_without_failover: List[str]
    
    Example:
        >>> tile = build_adversarial_governance_tile(pressure_model, scenario_plan, failover_plan_v2)
        >>> summary = summarize_adversarial_for_uplift_council(tile)
        >>> summary["status"]
        'OK'
    """
    global_pressure_index = tile.get("global_pressure_index", 0.0)
    pressure_band = tile.get("pressure_band", "LOW")
    has_failover = tile.get("has_failover", True)
    metrics_without_failover = tile.get("metrics_without_failover", [])
    priority_scenarios = tile.get("priority_scenarios", [])
    
    # Core uplift metrics (from CORE_UPLIFT_METRICS constant)
    # These are typically goal_hit and density
    core_metrics = {"goal_hit", "density"}
    
    # Check if core metrics are without failover
    core_without_failover = [
        mk for mk in metrics_without_failover
        if mk in core_metrics
    ]
    
    # Determine status
    # Tightened: HIGH pressure AND core metrics lack failover → always BLOCK
    if pressure_band == "HIGH" and len(core_without_failover) > 0:
        status = "BLOCK"
    # Otherwise keep existing semantics: HIGH pressure OR missing failover for core metrics → BLOCK
    elif pressure_band == "HIGH" or len(core_without_failover) > 0:
        status = "BLOCK"
    # WARN if MEDIUM pressure
    elif pressure_band == "MEDIUM":
        status = "WARN"
    # OK otherwise
    else:
        status = "OK"
    
    return {
        "status": status,
        "priority_scenarios": priority_scenarios,
        "pressure_band": pressure_band,
        "has_failover": has_failover,
        "metrics_without_failover": sorted(metrics_without_failover),
        "core_metrics_without_failover": sorted(core_without_failover),
    }


def extract_adversarial_summary_for_p3_stability(
    governance_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract adversarial pressure summary for P3 First-Light stability report.

    STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE

    Provides a compact summary suitable for inclusion in P3 stability report JSON.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is advisory-only
    - No control flow should depend on this summary

    Args:
        governance_tile: Adversarial governance tile from build_adversarial_governance_tile().

    Returns:
        Summary dictionary with:
        - global_pressure_index: float
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - priority_scenarios: List[str]
        - has_failover: bool
        - metrics_without_failover: List[str]
    """
    return {
        "global_pressure_index": governance_tile.get("global_pressure_index", 0.0),
        "pressure_band": governance_tile.get("pressure_band", "LOW"),
        "status_light": governance_tile.get("status_light", "GREEN"),
        "priority_scenarios": governance_tile.get("priority_scenarios", []),
        "has_failover": governance_tile.get("has_failover", True),
        "metrics_without_failover": sorted(governance_tile.get("metrics_without_failover", [])),
    }


def build_first_light_adversarial_coverage_annex(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First-Light adversarial coverage annex combining P3 and P4 signals.

    STATUS: PHASE X — ADVERSARIAL COVERAGE ANNEX

    Combines adversarial pressure signals from P3 stability report and P4 calibration
    report into a unified coverage annex for First-Light experiments.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned annex is purely observational
    - No control flow depends on the annex contents

    Args:
        p3_summary: P3 adversarial pressure summary from extract_adversarial_summary_for_p3_stability().
        p4_calibration: P4 adversarial calibration from extract_adversarial_calibration_for_p4().

    Returns:
        Coverage annex dictionary with:
        - schema_version: "1.0.0"
        - p3_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - priority_scenarios: List[str] (top 5 unique scenarios from both P3 and P4)
        - has_failover: bool (true if both P3 and P4 have failover)
    
    Example:
        >>> p3_summary = {"pressure_band": "LOW", "priority_scenarios": ["scenario1"]}
        >>> p4_calibration = {"pressure_band": "MEDIUM", "priority_scenarios": ["scenario2"]}
        >>> annex = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        >>> annex["p3_pressure_band"]
        'LOW'
    """
    # Extract pressure bands
    p3_pressure_band = p3_summary.get("pressure_band", "LOW")
    p4_pressure_band = p4_calibration.get("pressure_band", "LOW")
    
    # Combine priority scenarios (preserve order, remove duplicates, limit to 5)
    p3_scenarios = p3_summary.get("priority_scenarios", [])
    p4_scenarios = p4_calibration.get("priority_scenarios", [])
    # Use dict.fromkeys to preserve order while removing duplicates
    combined_scenarios = list(dict.fromkeys(p3_scenarios + p4_scenarios))[:5]
    
    # Both must have failover for overall has_failover to be true
    p3_has_failover = p3_summary.get("has_failover", True)
    p4_has_failover = p4_calibration.get("has_failover", True)
    has_failover = p3_has_failover and p4_has_failover
    
    return {
        "schema_version": "1.0.0",
        "p3_pressure_band": p3_pressure_band,
        "p4_pressure_band": p4_pressure_band,
        "priority_scenarios": combined_scenarios,
        "has_failover": has_failover,
    }


def extract_adversarial_calibration_for_p4(
    governance_tile: Dict[str, Any],
    release_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract adversarial calibration summary for P4 calibration report.

    STATUS: PHASE X — ADVERSARIAL PRESSURE GOVERNANCE

    Provides a compact summary suitable for inclusion in P4 calibration report JSON.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is advisory-only
    - No control flow should depend on this summary

    Args:
        governance_tile: Adversarial governance tile from build_adversarial_governance_tile().
        release_signal: Release signal from extract_adversarial_signal_for_release().

    Returns:
        Calibration dictionary with:
        - global_pressure_index: float
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - has_failover: bool
        - metrics_without_failover: List[str]
        - notes: List[str] (neutral observation strings)
    """
    notes = []
    
    pressure_band = governance_tile.get("pressure_band", "LOW")
    if pressure_band == "HIGH":
        notes.append("High adversarial pressure detected")
    elif pressure_band == "MEDIUM":
        notes.append("Moderate adversarial pressure")
    
    metrics_without_failover = release_signal.get("metrics_without_failover", [])
    if len(metrics_without_failover) > 0:
        notes.append(f"{len(metrics_without_failover)} metric(s) without failover coverage")
    
    if len(notes) == 0:
        notes.append("Adversarial coverage: OK")
    
    return {
        "global_pressure_index": release_signal.get("global_pressure_index", 0.0),
        "pressure_band": release_signal.get("pressure_band", "LOW"),
        "has_failover": release_signal.get("has_failover", True),
        "metrics_without_failover": sorted(metrics_without_failover),
        "notes": sorted(notes),
    }


def emit_cal_exp_adversarial_coverage(
    cal_id: str,
    annex: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Emit per-experiment adversarial coverage snapshot for calibration experiment.

    STATUS: PHASE X — CAL-EXP ADVERSARIAL COVERAGE GRID

    Creates a snapshot of adversarial coverage for a single calibration experiment
    and optionally persists it to disk.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from file I/O if output_dir provided)
    - The snapshot is purely observational
    - No control flow depends on the snapshot contents

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1").
        annex: Coverage annex from build_first_light_adversarial_coverage_annex().
        output_dir: Optional directory to persist snapshot. If None, no file is written.

    Returns:
        Coverage snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - p3_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - priority_scenarios: List[str] (≤5)
        - has_failover: bool
    
    Example:
        >>> annex = build_first_light_adversarial_coverage_annex(p3_summary, p4_calibration)
        >>> snapshot = emit_cal_exp_adversarial_coverage("CAL-EXP-1", annex)
        >>> snapshot["cal_id"]
        'CAL-EXP-1'
    """
    # Extract fields from annex
    p3_pressure_band = annex.get("p3_pressure_band", "LOW")
    p4_pressure_band = annex.get("p4_pressure_band", "LOW")
    priority_scenarios = annex.get("priority_scenarios", [])[:5]  # Limit to 5
    has_failover = annex.get("has_failover", True)
    
    snapshot = {
        "schema_version": ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION,
        "cal_id": cal_id,
        "p3_pressure_band": p3_pressure_band,
        "p4_pressure_band": p4_pressure_band,
        "priority_scenarios": priority_scenarios,
        "has_failover": has_failover,
    }
    
    # Persist to file if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"adversarial_coverage_{cal_id}.json"
        
        with open(output_path, "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
    
    return snapshot


def build_adversarial_coverage_grid(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build adversarial coverage grid from per-experiment snapshots.

    STATUS: PHASE X — CAL-EXP ADVERSARIAL COVERAGE GRID

    Aggregates coverage snapshots across multiple calibration experiments to provide
    a grid view of adversarial coverage across the experiment suite.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The grid is purely observational
    - No control flow depends on the grid contents

    Args:
        snapshots: List of coverage snapshots from emit_cal_exp_adversarial_coverage().

    Returns:
        Coverage grid dictionary with:
        - schema_version: "1.0.0"
        - total_experiments: int
        - pressure_band_counts: Dict[str, int] (counts by (p3_band, p4_band) tuple key)
        - experiments_missing_failover: List[str] (cal_ids where has_failover == False)
        - experiments_by_pressure_band: Dict[str, List[str]] (cal_ids grouped by pressure band combo)
    
    Example:
        >>> snapshots = [
        ...     {"cal_id": "CAL-EXP-1", "p3_pressure_band": "LOW", "p4_pressure_band": "MEDIUM", "has_failover": True},
        ...     {"cal_id": "CAL-EXP-2", "p3_pressure_band": "HIGH", "p4_pressure_band": "HIGH", "has_failover": False},
        ... ]
        >>> grid = build_adversarial_coverage_grid(snapshots)
        >>> grid["total_experiments"]
        2
    """
    # Count experiments by pressure band combination
    pressure_band_counts: Dict[str, int] = {}
    experiments_missing_failover: List[str] = []
    experiments_by_pressure_band: Dict[str, List[str]] = {}
    
    for snapshot in snapshots:
        cal_id = snapshot.get("cal_id", "")
        p3_band = snapshot.get("p3_pressure_band", "LOW")
        p4_band = snapshot.get("p4_pressure_band", "LOW")
        has_failover = snapshot.get("has_failover", True)
        
        # Create pressure band key
        band_key = f"({p3_band},{p4_band})"
        
        # Count by pressure band combination
        pressure_band_counts[band_key] = pressure_band_counts.get(band_key, 0) + 1
        
        # Track experiments missing failover
        if not has_failover:
            experiments_missing_failover.append(cal_id)
        
        # Group experiments by pressure band combination
        if band_key not in experiments_by_pressure_band:
            experiments_by_pressure_band[band_key] = []
        experiments_by_pressure_band[band_key].append(cal_id)
    
    return {
        "schema_version": ADVERSARIAL_COVERAGE_GRID_SCHEMA_VERSION,
        "total_experiments": len(snapshots),
        "pressure_band_counts": dict(sorted(pressure_band_counts.items())),
        "experiments_missing_failover": sorted(experiments_missing_failover),
        "experiments_by_pressure_band": {
            k: sorted(v) for k, v in sorted(experiments_by_pressure_band.items())
        },
    }


def attach_adversarial_coverage_grid_to_evidence(
    evidence: Dict[str, Any],
    coverage_grid: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach adversarial coverage grid to an evidence pack (read-only, additive).

    STATUS: PHASE X — CAL-EXP ADVERSARIAL COVERAGE GRID

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the grid attached
    under evidence["governance"]["adversarial_coverage_panel"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached grid is purely observational
    - No control flow depends on the grid contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        coverage_grid: Coverage grid from build_adversarial_coverage_grid().

    Returns:
        New dict with evidence contents plus coverage grid attached under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> grid = build_adversarial_coverage_grid(snapshots)
        >>> enriched = attach_adversarial_coverage_grid_to_evidence(evidence, grid)
        >>> "governance" in enriched
        True
        >>> "adversarial_coverage_panel" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach coverage grid
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["adversarial_coverage_panel"] = coverage_grid
    
    return enriched


def build_adversarial_priority_scenario_ledger(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build priority scenario ledger from per-experiment snapshots.

    STATUS: PHASE X — PRIORITY SCENARIO LEDGER

    Aggregates priority scenarios across all calibration experiments to identify
    the most frequently occurring scenarios that need adversarial test coverage.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The ledger is purely observational
    - No control flow depends on the ledger contents

    Args:
        snapshots: List of coverage snapshots from emit_cal_exp_adversarial_coverage().

    Returns:
        Priority scenario ledger dictionary with:
        - schema_version: "1.0.0"
        - scenario_counts: Dict[str, int] (scenario name -> frequency count, sorted)
        - top_priority_scenarios: List[str] (top 10 scenarios, deterministic order)
        - experiments_missing_failover: List[str] (cal_ids where has_failover == False, reused from grid)
    
    Example:
        >>> snapshots = [
        ...     {"cal_id": "CAL-EXP-1", "priority_scenarios": ["s1", "s2"], "has_failover": True},
        ...     {"cal_id": "CAL-EXP-2", "priority_scenarios": ["s1", "s3"], "has_failover": False},
        ... ]
        >>> ledger = build_adversarial_priority_scenario_ledger(snapshots)
        >>> ledger["scenario_counts"]["s1"]
        2
        >>> "CAL-EXP-2" in ledger["experiments_missing_failover"]
        True
    """
    # Count scenario frequency across all snapshots
    scenario_counts: Dict[str, int] = {}
    experiments_missing_failover: List[str] = []
    
    for snapshot in snapshots:
        cal_id = snapshot.get("cal_id", "")
        priority_scenarios = snapshot.get("priority_scenarios", [])
        has_failover = snapshot.get("has_failover", True)
        
        # Count scenarios
        for scenario in priority_scenarios:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        # Track experiments missing failover
        if not has_failover:
            experiments_missing_failover.append(cal_id)
    
    # Sort scenario counts by frequency (descending), then by name (ascending) for determinism
    sorted_scenarios = sorted(
        scenario_counts.items(),
        key=lambda x: (-x[1], x[0])  # Negative count for descending, then name ascending
    )
    
    # Top 10 priority scenarios (deterministic order)
    top_priority_scenarios = [scenario for scenario, _ in sorted_scenarios[:10]]
    
    # Build sorted scenario_counts dict (for JSON determinism)
    scenario_counts_sorted = dict(sorted_scenarios)
    
    return {
        "schema_version": ADVERSARIAL_PRIORITY_SCENARIO_LEDGER_SCHEMA_VERSION,
        "scenario_counts": scenario_counts_sorted,
        "top_priority_scenarios": top_priority_scenarios,
        "experiments_missing_failover": sorted(experiments_missing_failover),
    }


def attach_priority_scenario_ledger_to_evidence(
    evidence: Dict[str, Any],
    priority_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach priority scenario ledger to evidence pack under coverage panel.

    STATUS: PHASE X — PRIORITY SCENARIO LEDGER

    Attaches the ledger under evidence["governance"]["adversarial_coverage_panel"]["priority_scenario_ledger"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached ledger is purely observational
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        priority_ledger: Priority scenario ledger from build_adversarial_priority_scenario_ledger().

    Returns:
        New dict with evidence contents plus priority ledger attached.

    Example:
        >>> evidence = {"governance": {"adversarial_coverage_panel": {...}}}
        >>> ledger = build_adversarial_priority_scenario_ledger(snapshots)
        >>> enriched = attach_priority_scenario_ledger_to_evidence(evidence, ledger)
        >>> "priority_scenario_ledger" in enriched["governance"]["adversarial_coverage_panel"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance and coverage panel keys exist
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    enriched["governance"] = enriched["governance"].copy()
    
    if "adversarial_coverage_panel" not in enriched["governance"]:
        enriched["governance"]["adversarial_coverage_panel"] = {}
    else:
        enriched["governance"]["adversarial_coverage_panel"] = enriched["governance"]["adversarial_coverage_panel"].copy()
    
    # Attach priority ledger
    enriched["governance"]["adversarial_coverage_panel"]["priority_scenario_ledger"] = priority_ledger
    
    return enriched


def extract_adversarial_coverage_signal_for_status(
    manifest: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract adversarial coverage signal for status generation (manifest-first).

    STATUS: PHASE X — STATUS SURFACE

    Extracts compact adversarial coverage information from evidence pack for
    inclusion in status signals. Reads from manifest first, falls back to evidence.json.
    Tracks extraction provenance to avoid interpretive drift.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The extracted signal is purely observational
    - No control flow depends on the signal contents

    Args:
        manifest: Evidence pack manifest dict (preferred source, read-only).
        evidence: Evidence pack dict (fallback source, read-only).

    Returns:
        Optional dict with:
        - schema_version: str (if available from panel)
        - mode: "SHADOW"
        - extraction_source: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"
        - total_experiments: int
        - missing_failover_count: int
        - top_priority_scenarios_top5: List[str] (top 5 scenarios, empty if ledger missing)
        - priority_scenario_ledger_present: bool
        None if coverage panel not found in either source.

    Example:
        >>> manifest = {
        ...     "governance": {
        ...         "adversarial_coverage_panel": {
        ...             "schema_version": "1.0.0",
        ...             "total_experiments": 3,
        ...             "experiments_missing_failover": ["CAL-EXP-2"],
        ...             "priority_scenario_ledger": {
        ...                 "top_priority_scenarios": ["s1", "s2", "s3", "s4", "s5", "s6"]
        ...             }
        ...         }
        ...     }
        ... }
        >>> signal = extract_adversarial_coverage_signal_for_status(manifest=manifest)
        >>> signal["total_experiments"]
        3
        >>> signal["extraction_source"]
        'MANIFEST'
        >>> signal["priority_scenario_ledger_present"]
        True
    """
    # Try manifest first (preferred)
    coverage_panel = None
    extraction_source = "MISSING"
    
    if manifest:
        governance = manifest.get("governance", {})
        coverage_panel = governance.get("adversarial_coverage_panel")
        if coverage_panel is not None:
            extraction_source = "MANIFEST"
    
    # Fallback to evidence.json if manifest didn't have it
    if coverage_panel is None and evidence:
        governance = evidence.get("governance", {})
        coverage_panel = governance.get("adversarial_coverage_panel")
        if coverage_panel is not None:
            extraction_source = "EVIDENCE_JSON"
    
    # If panel still not found, return None
    if coverage_panel is None:
        return None
    
    # Extract grid data
    schema_version = coverage_panel.get("schema_version")
    total_experiments = coverage_panel.get("total_experiments", 0)
    experiments_missing_failover = coverage_panel.get("experiments_missing_failover", [])
    missing_failover_count = len(experiments_missing_failover)
    
    # Extract priority ledger data (may be missing)
    priority_ledger = coverage_panel.get("priority_scenario_ledger")
    priority_scenario_ledger_present = priority_ledger is not None
    
    if priority_ledger:
        top_priority_scenarios = priority_ledger.get("top_priority_scenarios", [])
        top_priority_scenarios_top5 = top_priority_scenarios[:5]
    else:
        # Safe defaults when ledger missing
        top_priority_scenarios_top5 = []
    
    signal = {
        "mode": "SHADOW",
        "extraction_source": extraction_source,
        "total_experiments": total_experiments,
        "missing_failover_count": missing_failover_count,
        "top_priority_scenarios_top5": top_priority_scenarios_top5,
        "priority_scenario_ledger_present": priority_scenario_ledger_present,
    }
    
    # Include schema_version if available
    if schema_version:
        signal["schema_version"] = schema_version
    
    return signal


# Reason code constants for adversarial coverage drivers
DRIVER_MISSING_FAILOVER_COUNT = "DRIVER_MISSING_FAILOVER_COUNT"
DRIVER_REPEATED_PRIORITY_SCENARIOS = "DRIVER_REPEATED_PRIORITY_SCENARIOS"


def adversarial_coverage_for_alignment_view(
    signal_or_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert adversarial coverage signal/panel to GGFL alignment view format.

    STATUS: PHASE X — GGFL ADAPTER

    Converts adversarial coverage signal or panel into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.
    Uses reason-code drivers to avoid interpretive drift.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The output is purely observational
    - No control flow depends on the output contents

    Args:
        signal_or_panel: Either:
            - Status signal from extract_adversarial_coverage_signal_for_status()
            - Coverage panel from evidence["governance"]["adversarial_coverage_panel"]

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-ADV"
        - status: "ok" | "warn" (warn if missing_failover_count > 0 or any scenario repeated >= 2)
        - conflict: false (always, as adversarial coverage is advisory)
        - drivers: List[str] (reason codes: DRIVER_MISSING_FAILOVER_COUNT, DRIVER_REPEATED_PRIORITY_SCENARIOS)
        - summary: str (one neutral sentence)
        - shadow_mode_invariants: Dict with conflict: false always

    Example:
        >>> signal = {
        ...     "total_experiments": 3,
        ...     "missing_failover_count": 2,
        ...     "top_priority_scenarios_top5": ["s1", "s2", "s1"],
        ... }
        >>> view = adversarial_coverage_for_alignment_view(signal)
        >>> view["signal_type"]
        'SIG-ADV'
        >>> view["status"]
        'warn'
        >>> view["conflict"]
        False
        >>> DRIVER_MISSING_FAILOVER_COUNT in view["drivers"]
        True
    """
    # Extract fields (handle both signal and panel formats)
    total_experiments = signal_or_panel.get("total_experiments", 0)
    missing_failover_count = signal_or_panel.get("missing_failover_count", 0)
    top_priority_scenarios = signal_or_panel.get("top_priority_scenarios_top5", [])
    
    # If panel format, extract from nested structure
    if "top_priority_scenarios_top5" not in signal_or_panel:
        priority_ledger = signal_or_panel.get("priority_scenario_ledger", {})
        top_priority_scenarios = priority_ledger.get("top_priority_scenarios", [])[:5]
        experiments_missing_failover = signal_or_panel.get("experiments_missing_failover", [])
        missing_failover_count = len(experiments_missing_failover)
    
    # Determine status: warn if missing failover or repeated scenarios
    status = "ok"
    drivers: List[str] = []
    
    # Check for missing failover (first priority) - use reason code
    if missing_failover_count > 0:
        status = "warn"
        drivers.append(DRIVER_MISSING_FAILOVER_COUNT)
    
    # Check for repeated scenarios (scenario appears >= 2 times in top 5) - use reason code
    scenario_counts: Dict[str, int] = {}
    for scenario in top_priority_scenarios:
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    repeated_scenarios = [
        scenario for scenario, count in scenario_counts.items() if count >= 2
    ]
    if repeated_scenarios:
        status = "warn"
        drivers.append(DRIVER_REPEATED_PRIORITY_SCENARIOS)
    
    # Build summary (one neutral sentence, includes both issues if present)
    if total_experiments == 0:
        summary = "No calibration experiments with adversarial coverage data"
    elif missing_failover_count == 0 and not repeated_scenarios:
        summary = f"Adversarial coverage: {total_experiments} experiment(s), all with failover"
    elif missing_failover_count > 0 and repeated_scenarios:
        # Both issues: single sentence mentioning both
        summary = f"Adversarial coverage: {total_experiments} experiment(s), {missing_failover_count} missing failover, priority scenarios repeated"
    elif missing_failover_count > 0:
        summary = f"Adversarial coverage: {total_experiments} experiment(s), {missing_failover_count} missing failover"
    else:
        summary = f"Adversarial coverage: {total_experiments} experiment(s), priority scenarios repeated"
    
    return {
        "signal_type": "SIG-ADV",
        "status": status,
        "conflict": False,  # Always false, adversarial coverage is advisory
        "drivers": drivers,  # Reason codes only
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,  # Adversarial coverage is advisory only
            "no_enforcement": True,  # No enforcement or gating
            "conflict_invariant": True,  # Conflict always false (invariant)
        },
    }


__all__ = [
    "ADVERSARIAL_GOVERNANCE_TILE_SCHEMA_VERSION",
    "ADVERSARIAL_COVERAGE_ANNEX_SCHEMA_VERSION",
    "ADVERSARIAL_COVERAGE_GRID_SCHEMA_VERSION",
    "ADVERSARIAL_PRIORITY_SCENARIO_LEDGER_SCHEMA_VERSION",
    "build_adversarial_governance_tile",
    "extract_adversarial_signal_for_release",
    "attach_adversarial_governance_to_evidence",
    "summarize_adversarial_for_uplift_council",
    "extract_adversarial_summary_for_p3_stability",
    "extract_adversarial_calibration_for_p4",
    "build_first_light_adversarial_coverage_annex",
    "emit_cal_exp_adversarial_coverage",
    "build_adversarial_coverage_grid",
    "attach_adversarial_coverage_grid_to_evidence",
    "build_adversarial_priority_scenario_ledger",
    "attach_priority_scenario_ledger_to_evidence",
    "extract_adversarial_coverage_signal_for_status",
    "adversarial_coverage_for_alignment_view",
]

