"""Topology pressure adapter for global health.

STATUS: PHASE X — TOPOLOGY GOVERNANCE TILE

Provides integration between Phase V topology pressure field, promotion gate,
and console tile components for the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The topology_pressure tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- No modification of topology state or curriculum decisions
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TOPOLOGY_PRESSURE_TILE_SCHEMA_VERSION = "1.0.0"


def _validate_pressure_field(pressure_field: Dict[str, Any]) -> None:
    """Validate pressure field structure.
    
    Args:
        pressure_field: Pressure field dictionary from build_topological_pressure_field()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["pressure_band", "slice_pressure", "pressure_components"]
    missing = [key for key in required_keys if key not in pressure_field]
    if missing:
        raise ValueError(
            f"pressure_field missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(pressure_field.keys()))}"
        )


def _validate_promotion_gate(promotion_gate: Dict[str, Any]) -> None:
    """Validate promotion gate structure.
    
    Args:
        promotion_gate: Promotion gate dictionary from topology_curriculum_promotion_gate()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["promotion_status"]
    missing = [key for key in required_keys if key not in promotion_gate]
    if missing:
        raise ValueError(
            f"promotion_gate missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(promotion_gate.keys()))}"
        )


def _validate_console_tile(console_tile: Dict[str, Any]) -> None:
    """Validate console tile structure.
    
    Args:
        console_tile: Console tile dictionary from build_topology_console_tile()
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light", "headline"]
    missing = [key for key in required_keys if key not in console_tile]
    if missing:
        raise ValueError(
            f"console_tile missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(console_tile.keys()))}"
        )


def build_topology_pressure_governance_tile(
    pressure_field: Dict[str, Any],
    promotion_gate: Dict[str, Any],
    console_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topology pressure governance tile for global health surface.

    STATUS: PHASE X — TOPOLOGY GOVERNANCE TILE

    Integrates Phase V topology pressure field, promotion gate, and console tile
    components into a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - No modification of topology state or curriculum decisions

    Args:
        pressure_field: Pressure field from build_topological_pressure_field().
            Must contain: pressure_band, slice_pressure, pressure_components
        promotion_gate: Promotion gate from topology_curriculum_promotion_gate().
            Must contain: promotion_status
        console_tile: Console tile from build_topology_console_tile().
            Must contain: status_light, headline

    Returns:
        Topology pressure governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - promotion_gate_status: "OK" | "ATTENTION" | "BLOCK"
        - pressure_hotspots: List[str]
        - headline: str (neutral summary)
    
    Example:
        >>> pressure_field = {
        ...     "pressure_band": "MEDIUM",
        ...     "slice_pressure": 0.5,
        ...     "pressure_components": {"depth": 0.4, "branching": 0.3, "risk": 0.3},
        ... }
        >>> promotion_gate = {"promotion_status": "OK"}
        >>> console_tile = {
        ...     "status_light": "GREEN",
        ...     "headline": "Topology status: stable",
        ...     "pressure_hotspots": [],
        ... }
        >>> tile = build_topology_pressure_governance_tile(
        ...     pressure_field, promotion_gate, console_tile
        ... )
        >>> tile["status_light"]
        'GREEN'
    """
    # Validate inputs
    _validate_pressure_field(pressure_field)
    _validate_promotion_gate(promotion_gate)
    _validate_console_tile(console_tile)
    
    # Extract fields
    status_light = console_tile.get("status_light", "GREEN")
    pressure_band = pressure_field.get("pressure_band", "LOW")
    promotion_gate_status = promotion_gate.get("promotion_status", "OK")
    pressure_hotspots = console_tile.get("pressure_hotspots", [])
    headline = console_tile.get("headline", "Topology pressure status: no data")
    
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
    
    # Validate promotion_gate_status
    if promotion_gate_status not in ("OK", "ATTENTION", "BLOCK"):
        raise ValueError(
            f"Invalid promotion_gate_status: {promotion_gate_status}. "
            f"Must be one of: OK, ATTENTION, BLOCK"
        )
    
    # Build tile
    tile = {
        "schema_version": TOPOLOGY_PRESSURE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "pressure_band": pressure_band,
        "promotion_gate_status": promotion_gate_status,
        "pressure_hotspots": pressure_hotspots,
        "headline": headline,
    }
    
    return tile


def build_first_light_topology_stress_summary(
    p3_topology_summary: Dict[str, Any],
    p4_topology_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topology stress summary from P3 and P4 topology data.

    STATUS: PHASE X — TOPOLOGY STRESS BLOCK FOR FIRST LIGHT

    Combines P3 stability report topology summary and P4 calibration report
    topology calibration into a unified stress summary for First Light runs.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: does not modify input dicts

    Args:
        p3_topology_summary: P3 topology summary from topology_pressure_summary field.
            Must contain: pressure_band
        p4_topology_calibration: P4 topology calibration from topology_pressure_calibration field.
            Must contain: pressure_band

    Returns:
        Topology stress summary dictionary with:
        - schema_version: "1.0.0"
        - p3_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - pressure_hotspots: List[str] (merged, deduplicated, max 5)

    Example:
        >>> p3_summary = {"pressure_band": "MEDIUM", "pressure_hotspots": ["Depth trend"]}
        >>> p4_calibration = {"pressure_band": "HIGH", "pressure_hotspots": ["Risk envelope"]}
        >>> summary = build_first_light_topology_stress_summary(p3_summary, p4_calibration)
        >>> summary["p3_pressure_band"]
        'MEDIUM'
    """
    # Extract pressure bands
    p3_band = p3_topology_summary.get("pressure_band", "LOW")
    p4_band = p4_topology_calibration.get("pressure_band", "LOW")

    # Merge and deduplicate hotspots (max 5)
    p3_hotspots = p3_topology_summary.get("pressure_hotspots", [])
    p4_hotspots = p4_topology_calibration.get("pressure_hotspots", [])
    merged_hotspots = list(set(p3_hotspots + p4_hotspots))[:5]

    return {
        "schema_version": "1.0.0",
        "p3_pressure_band": p3_band,
        "p4_pressure_band": p4_band,
        "pressure_hotspots": merged_hotspots,
    }


def attach_topology_pressure_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    p3_topology_summary: Optional[Dict[str, Any]] = None,
    p4_topology_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach topology pressure governance tile to an evidence pack (read-only, additive).

    STATUS: PHASE X — TOPOLOGY PRESSURE EVIDENCE INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["topology_pressure"].

    If both p3_topology_summary and p4_topology_calibration are provided, also
    attaches a first_light_topology_stress summary combining both.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        tile: Topology pressure governance tile from build_topology_pressure_governance_tile().
        p3_topology_summary: Optional P3 topology summary from topology_pressure_summary field.
        p4_topology_calibration: Optional P4 topology calibration from topology_pressure_calibration field.

    Returns:
        New dict with evidence contents plus topology_pressure tile attached under governance key.
        If both P3 and P4 summaries provided, also includes first_light_topology_stress.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_topology_pressure_governance_tile(pressure_field, promotion_gate, console_tile)
        >>> p3_summary = {"pressure_band": "MEDIUM", "pressure_hotspots": []}
        >>> p4_calibration = {"pressure_band": "HIGH", "pressure_hotspots": []}
        >>> enriched = attach_topology_pressure_to_evidence(evidence, tile, p3_summary, p4_calibration)
        >>> enriched["governance"]["topology_pressure"]["pressure_band"]
        'MEDIUM'
        >>> enriched["governance"]["first_light_topology_stress"]["p3_pressure_band"]
        'MEDIUM'
    """
    # Non-mutating: create new dict
    updated = dict(evidence)

    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])

    # Extract relevant fields for evidence (band, hotspots, headline)
    topology_pressure_summary = {
        "pressure_band": tile.get("pressure_band", "LOW"),
        "pressure_hotspots": tile.get("pressure_hotspots", []),
        "headline": tile.get("headline", "Topology pressure status: no data"),
    }

    # Attach summary
    updated["governance"]["topology_pressure"] = topology_pressure_summary

    # If both P3 and P4 summaries provided, build and attach stress summary
    if p3_topology_summary is not None and p4_topology_calibration is not None:
        stress_summary = build_first_light_topology_stress_summary(
            p3_topology_summary, p4_topology_calibration
        )
        updated["governance"]["first_light_topology_stress"] = stress_summary

    return updated


def build_topology_hotspot_ledger(
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topology hotspot ledger from topology stress panel.

    STATUS: PHASE X — TOPOLOGY HOTSPOT LEDGER

    Analyzes the topology stress panel to identify which slices (or hotspot
    identifiers) recur as hotspots across multiple calibration experiments.
    This ledger provides a cross-experiment view of recurring topology stress
    patterns.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned ledger is purely observational
    - No control flow depends on the ledger contents
    - Non-mutating: does not modify input panel
    - Ledger is for analysis/visualization, not a gate

    Args:
        panel: Topology stress panel from build_topology_stress_panel().
            Must contain: experiments (list of experiment dicts with pressure_hotspots)

    Returns:
        Hotspot ledger dictionary with:
        - schema_version: "1.0.0"
        - hotspot_counts: Dict[str, int] (slice/hotspot name -> count, sorted by key)
        - top_hotspots: List[str] (top 10 hotspots by count, deterministic ordering)
        - num_experiments: int (total number of experiments analyzed)

    Example:
        >>> panel = {
        ...     "experiments": [
        ...         {"pressure_hotspots": ["slice_uplift_tree", "Depth trend"]},
        ...         {"pressure_hotspots": ["slice_uplift_tree", "Risk envelope"]},
        ...     ]
        ... }
        >>> ledger = build_topology_hotspot_ledger(panel)
        >>> ledger["hotspot_counts"]["slice_uplift_tree"]
        2
    """
    experiments = panel.get("experiments", [])

    # Count occurrences of each hotspot across all experiments
    hotspot_counts: Dict[str, int] = {}
    for exp in experiments:
        hotspots = exp.get("pressure_hotspots", [])
        for hotspot in hotspots:
            if hotspot:  # Skip empty strings
                hotspot_counts[hotspot] = hotspot_counts.get(hotspot, 0) + 1

    # Sort counts by key for determinism
    sorted_counts = dict(sorted(hotspot_counts.items()))

    # Get top 10 hotspots by count (with deterministic tie-breaking by name)
    # Sort by count (descending), then by name (ascending) for tie-breaking
    sorted_by_count = sorted(
        hotspot_counts.items(),
        key=lambda x: (-x[1], x[0]),  # Negative count for descending, name for tie-break
    )
    top_hotspots = [hotspot for hotspot, _ in sorted_by_count[:10]]

    return {
        "schema_version": "1.0.0",
        "hotspot_counts": sorted_counts,
        "top_hotspots": top_hotspots,
        "num_experiments": len(experiments),
    }


def extract_topology_hotspot_ledger_signal(
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact topology hotspot ledger signal for status integration.

    STATUS: PHASE X — TOPOLOGY HOTSPOT LEDGER STATUS SIGNAL

    Extracts a compact status signal from the topology hotspot ledger for
    inclusion in First Light status files. Provides a quick view of recurring
    hotspot patterns without requiring full ledger inspection.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Non-mutating: does not modify input ledger

    Args:
        ledger: Topology hotspot ledger from build_topology_hotspot_ledger().
            Must contain: hotspot_counts, top_hotspots, num_experiments

    Returns:
        Compact signal dictionary with:
        - schema_version: "1.0.0"
        - num_experiments: int
        - unique_hotspot_count: int (number of unique hotspots)
        - top_hotspots_top3: List[str] (top 3 hotspot names only)
        - top_hotspot_counts_top3: List[int] (corresponding counts for top 3)

    Example:
        >>> ledger = {
        ...     "hotspot_counts": {"slice_a": 3, "slice_b": 2, "slice_c": 1},
        ...     "top_hotspots": ["slice_a", "slice_b", "slice_c"],
        ...     "num_experiments": 3,
        ... }
        >>> signal = extract_topology_hotspot_ledger_signal(ledger)
        >>> signal["top_hotspots_top3"]
        ['slice_a', 'slice_b', 'slice_c']
    """
    hotspot_counts = ledger.get("hotspot_counts", {})
    top_hotspots = ledger.get("top_hotspots", [])
    num_experiments = ledger.get("num_experiments", 0)

    # Get unique hotspot count
    unique_hotspot_count = len(hotspot_counts)

    # Extract top 3 hotspots and their counts
    top_hotspots_top3 = top_hotspots[:3]
    top_hotspot_counts_top3 = [
        hotspot_counts.get(hotspot, 0) for hotspot in top_hotspots_top3
    ]

    return {
        "schema_version": "1.0.0",
        "num_experiments": num_experiments,
        "unique_hotspot_count": unique_hotspot_count,
        "top_hotspots_top3": top_hotspots_top3,
        "top_hotspot_counts_top3": top_hotspot_counts_top3,
    }


def extract_topology_hotspot_ledger_signal_from_evidence(
    evidence: Dict[str, Any],
    extraction_source: str = "UNKNOWN",
) -> Optional[Dict[str, Any]]:
    """
    Extract topology hotspot ledger signal from evidence pack.

    STATUS: PHASE X — TOPOLOGY HOTSPOT LEDGER STATUS EXTRACTION

    Extracts the topology hotspot ledger signal from the evidence pack if present.
    Returns None if the panel or ledger is not found (graceful degradation).

    CANONICAL LOCATION:
    The topology stress panel should be attached to the evidence pack manifest at:
        manifest["governance"]["topology_stress_panel"]
    
    The hotspot ledger is automatically included as:
        manifest["governance"]["topology_stress_panel"]["hotspot_ledger"]
    
    This canonical location is used by the status generator (generate_first_light_status.py)
    for manifest-first signal extraction.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The extracted signal is purely observational
    - No control flow depends on the signal contents
    - Returns None if ledger not present (safe behavior)

    Args:
        evidence: Evidence pack dictionary. May contain:
            evidence["governance"]["topology_stress_panel"]["hotspot_ledger"]
        extraction_source: Source identifier ("MANIFEST" | "EVIDENCE_JSON" | "MISSING").
            Used for provenance tracking in the signal.

    Returns:
        Status signal dictionary from extract_topology_hotspot_ledger_signal(),
        with extraction_source field added, or None if ledger not present in evidence.

    Example:
        >>> evidence = {
        ...     "governance": {
        ...         "topology_stress_panel": {
        ...             "hotspot_ledger": {
        ...                 "hotspot_counts": {"slice_a": 2},
        ...                 "top_hotspots": ["slice_a"],
        ...                 "num_experiments": 3,
        ...             }
        ...         }
        ...     }
        ... }
        >>> signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence, "MANIFEST")
        >>> signal["num_experiments"]
        3
        >>> signal["extraction_source"]
        'MANIFEST'
    """
    # Navigate to ledger in evidence structure
    governance = evidence.get("governance", {})
    panel = governance.get("topology_stress_panel")
    if panel is None:
        return None

    ledger = panel.get("hotspot_ledger")
    if ledger is None:
        return None

    # Extract signal from ledger and add provenance
    signal = extract_topology_hotspot_ledger_signal(ledger)
    if signal:
        signal["extraction_source"] = extraction_source
    return signal


def extract_topology_hotspot_ledger_warnings(
    ledger: Dict[str, Any],
) -> List[str]:
    """
    Extract warnings from topology hotspot ledger.

    STATUS: PHASE X — TOPOLOGY HOTSPOT LEDGER WARNING EXTRACTION

    Generates advisory warnings if any hotspot recurs across experiments
    (count >= 2). Warnings are neutral and observational only.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from list construction)
    - The returned warnings are purely observational
    - No control flow depends on the warnings
    - Non-mutating: does not modify input ledger

    Args:
        ledger: Topology hotspot ledger from build_topology_hotspot_ledger().
            Must contain: hotspot_counts

    Returns:
        List of warning strings (empty if no recurring hotspots).

    Example:
        >>> ledger = {
        ...     "hotspot_counts": {"slice_a": 3, "slice_b": 1},
        ... }
        >>> warnings = extract_topology_hotspot_ledger_warnings(ledger)
        >>> len(warnings)
        1
    """
    warnings = []
    hotspot_counts = ledger.get("hotspot_counts", {})

    # Check for recurring hotspots (count >= 2)
    recurring_hotspots = [
        hotspot for hotspot, count in hotspot_counts.items() if count >= 2
    ]

    if recurring_hotspots:
        # Sort for deterministic output
        sorted_recurring = sorted(recurring_hotspots)
        for hotspot in sorted_recurring:
            count = hotspot_counts[hotspot]
            warnings.append(
                f"Hotspot '{hotspot}' recurs across {count} experiment(s)"
            )

    return warnings


def attach_topology_hotspot_ledger_signal_to_evidence(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach topology hotspot ledger signal to evidence pack signals section.

    STATUS: PHASE X — TOPOLOGY HOTSPOT LEDGER STATUS INTEGRATION

    Extracts the topology hotspot ledger signal from the evidence pack and
    attaches it to evidence["signals"]["topology_hotspot_ledger"] for quick
    status access. Also attaches warnings if any hotspot recurs across experiments.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached signal is purely observational
    - No control flow depends on the signal contents
    - Non-mutating: returns new dict, does not modify input
    - Gracefully handles missing ledger (no signal attached)

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
            May contain evidence["governance"]["topology_stress_panel"]["hotspot_ledger"].

    Returns:
        New dict with evidence contents plus topology hotspot ledger signal
        attached under signals key (if ledger present in evidence).

    Example:
        >>> evidence = {
        ...     "governance": {
        ...         "topology_stress_panel": {
        ...             "hotspot_ledger": {
        ...                 "hotspot_counts": {"slice_a": 2},
        ...                 "top_hotspots": ["slice_a"],
        ...                 "num_experiments": 3,
        ...             }
        ...         }
        ...     }
        ... }
        >>> enriched = attach_topology_hotspot_ledger_signal_to_evidence(evidence)
        >>> "signals" in enriched
        True
        >>> "topology_hotspot_ledger" in enriched["signals"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = dict(evidence)

    # Extract signal from ledger if present
    signal = extract_topology_hotspot_ledger_signal_from_evidence(evidence)

    if signal is not None:
        # Ensure signals key exists
        if "signals" not in enriched:
            enriched["signals"] = {}
        else:
            enriched["signals"] = dict(enriched["signals"])

        # Attach signal
        enriched["signals"]["topology_hotspot_ledger"] = signal

        # Extract and attach warnings if any hotspot recurs (count >= 2)
        governance = evidence.get("governance", {})
        panel = governance.get("topology_stress_panel")
        if panel is not None:
            ledger = panel.get("hotspot_ledger")
            if ledger is not None:
                warnings = extract_topology_hotspot_ledger_warnings(ledger)
                if warnings:
                    # Attach warnings to signal
                    enriched["signals"]["topology_hotspot_ledger"]["warnings"] = warnings

    return enriched


def attach_topology_stress_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach topology stress panel to an evidence pack (read-only, additive).

    STATUS: PHASE X — TOPOLOGY STRESS PANEL EVIDENCE INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the panel attached
    under evidence["governance"]["topology_stress_panel"]. Also attaches the
    hotspot ledger under the panel's hotspot_ledger field.

    CANONICAL LOCATION:
    The topology stress panel should be attached to the evidence pack manifest at:
        manifest["governance"]["topology_stress_panel"]
    
    The hotspot ledger is automatically included as:
        manifest["governance"]["topology_stress_panel"]["hotspot_ledger"]
    
    This canonical location is used by the status generator (generate_first_light_status.py)
    for manifest-first signal extraction.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached panel is purely observational
    - No control flow depends on the panel contents
    - Non-mutating: returns new dict, does not modify input
    - Panel is for heatmap/visualization, not a gate

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        panel: Topology stress panel from build_topology_stress_panel().

    Returns:
        New dict with evidence contents plus topology_stress_panel attached under governance key.
        The panel includes a hotspot_ledger field.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> snapshots = [snapshot1, snapshot2, snapshot3]
        >>> panel = build_topology_stress_panel(snapshots)
        >>> enriched = attach_topology_stress_panel_to_evidence(evidence, panel)
        >>> enriched["governance"]["topology_stress_panel"]["panel_type"]
        'topology_stress_heatmap'
        >>> "hotspot_ledger" in enriched["governance"]["topology_stress_panel"]
        True
    """
    # Non-mutating: create new dict
    updated = dict(evidence)

    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])

    # Build hotspot ledger and attach to panel
    panel_with_ledger = dict(panel)
    hotspot_ledger = build_topology_hotspot_ledger(panel)
    panel_with_ledger["hotspot_ledger"] = hotspot_ledger

    # Attach panel with ledger
    updated["governance"]["topology_stress_panel"] = panel_with_ledger

    return updated


def summarize_topology_pressure_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize topology pressure tile for uplift council decision.

    STATUS: PHASE X — TOPOLOGY PRESSURE COUNCIL ADAPTER

    Maps topology pressure band to council status:
    - HIGH → BLOCK
    - MEDIUM → WARN
    - LOW → OK

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Topology pressure is a stress indicator, not a direct gate

    Args:
        tile: Topology pressure governance tile from build_topology_pressure_governance_tile().

    Returns:
        Council summary with:
        - status: "OK" | "WARN" | "BLOCK"
        - pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - pressure_hotspots: List[str]
        - headline: str

    Example:
        >>> tile = build_topology_pressure_governance_tile(pressure_field, promotion_gate, console_tile)
        >>> summary = summarize_topology_pressure_for_uplift_council(tile)
        >>> summary["status"]
        'WARN'
    """
    pressure_band = tile.get("pressure_band", "LOW")

    # Map pressure band to council status
    if pressure_band == "HIGH":
        status = "BLOCK"
    elif pressure_band == "MEDIUM":
        status = "WARN"
    else:  # LOW
        status = "OK"

    return {
        "status": status,
        "pressure_band": pressure_band,
        "pressure_hotspots": tile.get("pressure_hotspots", []),
        "headline": tile.get("headline", "Topology pressure status: no data"),
    }


def add_topology_pressure_to_p3_stability_report(
    stability_report: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add topology pressure summary to P3 stability report.

    STATUS: PHASE X — TOPOLOGY PRESSURE P3 BINDING

    Adds topology_pressure_summary field to stability report containing:
    - pressure_band: "LOW" | "MEDIUM" | "HIGH"
    - status_light: "GREEN" | "YELLOW" | "RED"
    - pressure_hotspots: List[str]
    - headline: str

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The added summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        stability_report: P3 stability report dictionary (read-only, not modified).
        tile: Topology pressure governance tile from build_topology_pressure_governance_tile().

    Returns:
        New stability report dict with topology_pressure_summary added.

    Example:
        >>> report = {"schema_version": "1.0.0", "run_id": "test", ...}
        >>> tile = build_topology_pressure_governance_tile(pressure_field, promotion_gate, console_tile)
        >>> enriched = add_topology_pressure_to_p3_stability_report(report, tile)
        >>> enriched["topology_pressure_summary"]["pressure_band"]
        'MEDIUM'
    """
    # Non-mutating: create new dict
    updated = dict(stability_report)

    # Extract relevant fields for stability report
    topology_pressure_summary = {
        "pressure_band": tile.get("pressure_band", "LOW"),
        "status_light": tile.get("status_light", "GREEN"),
        "pressure_hotspots": tile.get("pressure_hotspots", []),
        "headline": tile.get("headline", "Topology pressure status: no data"),
    }

    # Add summary to report
    updated["topology_pressure_summary"] = topology_pressure_summary

    return updated


def add_topology_pressure_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add topology pressure calibration to P4 calibration report.

    STATUS: PHASE X — TOPOLOGY PRESSURE P4 BINDING

    Adds topology_pressure_calibration field to calibration report containing:
    - pressure_band: "LOW" | "MEDIUM" | "HIGH"
    - status_light: "GREEN" | "YELLOW" | "RED"
    - pressure_hotspots: List[str]
    - structural_notes: List[str] (from tile neutral_notes if available)

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The added calibration is purely observational
    - No control flow depends on the calibration contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        calibration_report: P4 calibration report dictionary (read-only, not modified).
        tile: Topology pressure governance tile from build_topology_pressure_governance_tile().

    Returns:
        New calibration report dict with topology_pressure_calibration added.

    Example:
        >>> report = {"schema_version": "1.0.0", "run_id": "test", ...}
        >>> tile = build_topology_pressure_governance_tile(pressure_field, promotion_gate, console_tile)
        >>> enriched = add_topology_pressure_to_p4_calibration_report(report, tile)
        >>> enriched["topology_pressure_calibration"]["pressure_band"]
        'MEDIUM'
    """
    # Non-mutating: create new dict
    updated = dict(calibration_report)

    # Extract relevant fields for calibration report
    # Note: structural_notes can come from pressure_field if available
    structural_notes = []
    if "pressure_components" in tile:
        # Extract notes from pressure components if available
        components = tile.get("pressure_components", {})
        if components:
            structural_notes.append(
                f"Depth component: {components.get('depth', 0.0):.3f}, "
                f"Branching component: {components.get('branching', 0.0):.3f}, "
                f"Risk component: {components.get('risk', 0.0):.3f}"
            )

    topology_pressure_calibration = {
        "pressure_band": tile.get("pressure_band", "LOW"),
        "status_light": tile.get("status_light", "GREEN"),
        "pressure_hotspots": tile.get("pressure_hotspots", []),
        "structural_notes": structural_notes,
    }

    # Add calibration to report
    updated["topology_pressure_calibration"] = topology_pressure_calibration

    return updated


def build_cal_exp_topology_stress_snapshot(
    cal_id: str,
    stress_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topology stress snapshot for a calibration experiment.

    STATUS: PHASE X — CAL-EXP TOPOLOGY STRESS SNAPSHOT

    Creates a per-experiment topology stress snapshot from a First Light
    topology stress summary. This snapshot is designed to be persisted as
    `calibration/topology_stress_<cal_id>.json` and aggregated into a
    heatmap panel for visualization.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned snapshot is purely observational
    - No control flow depends on the snapshot contents
    - Non-mutating: does not modify input dicts

    Args:
        cal_id: Calibration experiment identifier (e.g., "cal_exp1", "cal_exp2").
        stress_summary: First Light topology stress summary from build_first_light_topology_stress_summary().
            Must contain: p3_pressure_band, p4_pressure_band, pressure_hotspots

    Returns:
        Topology stress snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str
        - p3_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - p4_pressure_band: "LOW" | "MEDIUM" | "HIGH"
        - pressure_hotspots: List[str] (max 5, preserved from summary)

    Example:
        >>> summary = {
        ...     "p3_pressure_band": "MEDIUM",
        ...     "p4_pressure_band": "HIGH",
        ...     "pressure_hotspots": ["Depth trend contributing"],
        ... }
        >>> snapshot = build_cal_exp_topology_stress_snapshot("cal_exp1", summary)
        >>> snapshot["cal_id"]
        'cal_exp1'
    """
    # Extract fields from stress summary
    p3_band = stress_summary.get("p3_pressure_band", "LOW")
    p4_band = stress_summary.get("p4_pressure_band", "LOW")
    hotspots = stress_summary.get("pressure_hotspots", [])

    # Ensure hotspots are limited to 5 (preserve limit from summary)
    limited_hotspots = hotspots[:5] if len(hotspots) > 5 else hotspots

    return {
        "schema_version": "1.0.0",
        "cal_id": cal_id,
        "p3_pressure_band": p3_band,
        "p4_pressure_band": p4_band,
        "pressure_hotspots": limited_hotspots,
    }


def persist_cal_exp_topology_stress_snapshot(
    snapshot: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist calibration experiment topology stress snapshot to disk.

    STATUS: PHASE X — CAL-EXP TOPOLOGY STRESS SNAPSHOT PERSISTENCE

    Writes snapshot to calibration/topology_stress_<cal_id>.json.
    Creates the output directory if it doesn't exist.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions
    - Non-mutating: does not modify input snapshot

    Args:
        snapshot: Topology stress snapshot from build_cal_exp_topology_stress_snapshot().
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration")).

    Returns:
        Path to the written snapshot file.

    Raises:
        IOError: If the file cannot be written.

    Example:
        >>> snapshot = build_cal_exp_topology_stress_snapshot("cal_exp1", stress_summary)
        >>> path = persist_cal_exp_topology_stress_snapshot(snapshot, Path("calibration"))
        >>> path.exists()
        True
    """
    cal_id = snapshot.get("cal_id", "UNKNOWN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"topology_stress_{cal_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)

    return output_path


def build_topology_stress_panel(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build topology stress panel from multiple calibration experiment snapshots.

    STATUS: PHASE X — TOPOLOGY STRESS HEATMAP PANEL (STUB)

    Aggregates per-experiment topology stress snapshots into a panel structure
    suitable for heatmap visualization. This is a data preparation function;
    actual plotting/visualization is handled by downstream visualization tools.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned panel is purely observational
    - No control flow depends on the panel contents
    - Non-mutating: does not modify input list or dicts
    - This is a data feed, not a gate

    Args:
        snapshots: List of topology stress snapshots from build_cal_exp_topology_stress_snapshot().
            Each snapshot must contain: cal_id, p3_pressure_band, p4_pressure_band, pressure_hotspots

    Returns:
        Topology stress panel dictionary with:
        - schema_version: "1.0.0"
        - panel_type: "topology_stress_heatmap"
        - experiments: List[Dict[str, Any]] (list of snapshot entries)
        - summary: Dict[str, Any] (aggregate statistics)

    Example:
        >>> snapshot1 = build_cal_exp_topology_stress_snapshot("cal_exp1", summary1)
        >>> snapshot2 = build_cal_exp_topology_stress_snapshot("cal_exp2", summary2)
        >>> panel = build_topology_stress_panel([snapshot1, snapshot2])
        >>> panel["panel_type"]
        'topology_stress_heatmap'
    """
    # Build experiment entries (preserve all snapshot fields)
    experiments = []
    for snapshot in snapshots:
        experiments.append({
            "cal_id": snapshot.get("cal_id", ""),
            "p3_pressure_band": snapshot.get("p3_pressure_band", "LOW"),
            "p4_pressure_band": snapshot.get("p4_pressure_band", "LOW"),
            "pressure_hotspots": snapshot.get("pressure_hotspots", []),
        })

    # Compute aggregate statistics
    p3_bands = [e["p3_pressure_band"] for e in experiments]
    p4_bands = [e["p4_pressure_band"] for e in experiments]

    # Count band frequencies
    p3_band_counts = {
        "LOW": p3_bands.count("LOW"),
        "MEDIUM": p3_bands.count("MEDIUM"),
        "HIGH": p3_bands.count("HIGH"),
    }
    p4_band_counts = {
        "LOW": p4_bands.count("LOW"),
        "MEDIUM": p4_bands.count("MEDIUM"),
        "HIGH": p4_bands.count("HIGH"),
    }

    # Collect all unique hotspots across experiments
    all_hotspots = []
    for exp in experiments:
        all_hotspots.extend(exp.get("pressure_hotspots", []))
    unique_hotspots = list(set(all_hotspots))[:10]  # Limit to 10 for summary

    summary = {
        "total_experiments": len(experiments),
        "p3_band_distribution": p3_band_counts,
        "p4_band_distribution": p4_band_counts,
        "common_hotspots": unique_hotspots,
    }

    return {
        "schema_version": "1.0.0",
        "panel_type": "topology_stress_heatmap",
        "experiments": experiments,
        "summary": summary,
    }


__all__ = [
    "TOPOLOGY_PRESSURE_TILE_SCHEMA_VERSION",
    "build_topology_pressure_governance_tile",
    "attach_topology_pressure_to_evidence",
    "summarize_topology_pressure_for_uplift_council",
    "add_topology_pressure_to_p3_stability_report",
    "add_topology_pressure_to_p4_calibration_report",
    "build_first_light_topology_stress_summary",
    "build_cal_exp_topology_stress_snapshot",
    "persist_cal_exp_topology_stress_snapshot",
    "build_topology_stress_panel",
    "build_topology_hotspot_ledger",
    "extract_topology_hotspot_ledger_signal",
    "extract_topology_hotspot_ledger_signal_from_evidence",
    "extract_topology_hotspot_ledger_warnings",
    "attach_topology_stress_panel_to_evidence",
    "attach_topology_hotspot_ledger_signal_to_evidence",
]

