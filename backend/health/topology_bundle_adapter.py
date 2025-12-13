"""Topology × Bundle adapter for global health.

STATUS: PHASE X — TOPOLOGY/BUNDLE GOVERNANCE TILE

Provides integration between topology drift compass, bundle joint view,
and director panel components for the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The topology_bundle tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- No modification of topology state or bundle decisions
- Zero gating logic — SHADOW MODE only
"""

from typing import Any, Dict, List, Optional

TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION = "1.0.0"


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_joint_view(joint_view: Dict[str, Any]) -> None:
    """Validate topology bundle joint view structure.

    Args:
        joint_view: Joint view dictionary per topology_bundle_joint_view.schema.json

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["topology_snapshot", "bundle_snapshot", "alignment_status"]
    missing = [key for key in required_keys if key not in joint_view]
    if missing:
        raise ValueError(
            f"joint_view missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(joint_view.keys()))}"
        )


def _validate_consistency_result(consistency_result: Dict[str, Any]) -> None:
    """Validate cross-system consistency result structure.

    Args:
        consistency_result: Consistency evaluation result

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["consistent", "status"]
    missing = [key for key in required_keys if key not in consistency_result]
    if missing:
        raise ValueError(
            f"consistency_result missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(consistency_result.keys()))}"
        )


def _validate_director_panel(director_panel: Dict[str, Any]) -> None:
    """Validate topology bundle director panel structure.

    Args:
        director_panel: Director panel dictionary per topology_bundle_director_panel.schema.json

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["topology_tile", "bundle_tile", "correlation_tile", "overall_health"]
    missing = [key for key in required_keys if key not in director_panel]
    if missing:
        raise ValueError(
            f"director_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(director_panel.keys()))}"
        )


# -----------------------------------------------------------------------------
# Status mapping helpers
# -----------------------------------------------------------------------------

def _map_topology_mode_to_stability(mode: str) -> str:
    """Map topology mode to stability status.

    Args:
        mode: Topology mode (STABLE, DRIFT, TURBULENT, CRITICAL)

    Returns:
        Stability status string
    """
    mapping = {
        "STABLE": "STABLE",
        "DRIFT": "DRIFTING",
        "TURBULENT": "TURBULENT",
        "CRITICAL": "CRITICAL",
    }
    return mapping.get(mode, "UNKNOWN")


def _map_bundle_status_to_stability(status: str) -> str:
    """Map bundle chain status to stability status.

    Args:
        status: Bundle status (VALID, WARN, BROKEN, MISSING)

    Returns:
        Stability status string
    """
    mapping = {
        "VALID": "VALID",
        "WARN": "ATTENTION",
        "BROKEN": "BROKEN",
        "MISSING": "MISSING",
    }
    return mapping.get(status, "UNKNOWN")


def _determine_status_light(
    topology_mode: str,
    bundle_status: str,
    alignment_status: str,
) -> str:
    """Determine overall status light from component states.

    Args:
        topology_mode: Topology mode (STABLE, DRIFT, TURBULENT, CRITICAL)
        bundle_status: Bundle chain status (VALID, WARN, BROKEN, MISSING)
        alignment_status: Alignment status (ALIGNED, TENSION, DIVERGENT)

    Returns:
        Status light: GREEN, YELLOW, or RED
    """
    # CRITICAL conditions → RED
    if topology_mode == "CRITICAL":
        return "RED"
    if bundle_status in ("BROKEN", "MISSING"):
        return "RED"
    if alignment_status == "DIVERGENT":
        return "RED"

    # WARNING conditions → YELLOW
    if topology_mode in ("DRIFT", "TURBULENT"):
        return "YELLOW"
    if bundle_status == "WARN":
        return "YELLOW"
    if alignment_status == "TENSION":
        return "YELLOW"

    # All nominal → GREEN
    return "GREEN"


def _extract_conflict_codes(
    joint_view: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]],
) -> List[str]:
    """Extract XCOR conflict codes from joint view and director panel.

    Args:
        joint_view: Topology bundle joint view
        director_panel: Optional director panel

    Returns:
        List of active XCOR conflict codes
    """
    conflict_codes: List[str] = []

    # Extract from joint_view governance_signals
    if "governance_signals" in joint_view:
        signals = joint_view["governance_signals"]
        correlation_codes = signals.get("correlation_codes", [])
        # Filter for WARN/CRIT codes (XCOR-WARN-*, XCOR-CRIT-*)
        for code in correlation_codes:
            if "WARN" in code or "CRIT" in code:
                conflict_codes.append(code)

    # Extract from director_panel correlation_tile
    if director_panel and "correlation_tile" in director_panel:
        correlation_tile = director_panel["correlation_tile"]
        active_signals = correlation_tile.get("active_signals", [])
        for signal in active_signals:
            if isinstance(signal, dict):
                code = signal.get("code", "")
                if code and ("WARN" in code or "CRIT" in code):
                    if code not in conflict_codes:
                        conflict_codes.append(code)
            elif isinstance(signal, str):
                if "WARN" in signal or "CRIT" in signal:
                    if signal not in conflict_codes:
                        conflict_codes.append(signal)

    return conflict_codes


def _build_neutral_headline(
    topology_mode: str,
    bundle_status: str,
    alignment_status: str,
    status_light: str,
) -> str:
    """Build neutral headline for console tile.

    Args:
        topology_mode: Topology mode
        bundle_status: Bundle chain status
        alignment_status: Alignment status
        status_light: Overall status light

    Returns:
        Neutral headline string (no judgmental language)
    """
    # Build descriptive headline based on state
    topology_desc = _map_topology_mode_to_stability(topology_mode)
    bundle_desc = _map_bundle_status_to_stability(bundle_status)

    parts = [
        f"Topology: {topology_desc}",
        f"Bundle: {bundle_desc}",
        f"Alignment: {alignment_status}",
    ]

    return " | ".join(parts)


# -----------------------------------------------------------------------------
# Main tile builder
# -----------------------------------------------------------------------------

def build_topology_bundle_console_tile(
    joint_view: Dict[str, Any],
    consistency_result: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build topology × bundle console tile for global health surface.

    STATUS: PHASE X — TOPOLOGY/BUNDLE GOVERNANCE TILE

    Integrates topology drift compass, bundle joint view, and director panel
    components into a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - No modification of topology state or bundle decisions
    - Zero gating logic — SHADOW MODE only

    Args:
        joint_view: Topology bundle joint view per topology_bundle_joint_view.schema.json
            Must contain: topology_snapshot, bundle_snapshot, alignment_status
        consistency_result: Cross-system consistency evaluation result
            Must contain: consistent, status
        director_panel: Optional director panel per topology_bundle_director_panel.schema.json
            Must contain (if provided): topology_tile, bundle_tile, correlation_tile, overall_health

    Returns:
        Topology bundle console tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - topology_stability: str (topology mode mapped to stability)
        - bundle_stability: str (bundle status mapped to stability)
        - cross_system_consistency: bool (from consistency_result)
        - joint_status: str (alignment status from joint_view)
        - conflict_codes: List[str] (XCOR codes active)
        - headline: str (neutral description)

    Example:
        >>> joint_view = {
        ...     "topology_snapshot": {"topology_mode": "STABLE", ...},
        ...     "bundle_snapshot": {"bundle_status": "VALID", ...},
        ...     "alignment_status": {"overall_status": "ALIGNED", ...},
        ... }
        >>> consistency_result = {"consistent": True, "status": "OK"}
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> tile["status_light"]
        'GREEN'
    """
    # Validate inputs
    _validate_joint_view(joint_view)
    _validate_consistency_result(consistency_result)
    if director_panel is not None:
        _validate_director_panel(director_panel)

    # Extract topology state
    topology_snapshot = joint_view.get("topology_snapshot", {})
    topology_mode = topology_snapshot.get("topology_mode", "UNKNOWN")

    # Extract bundle state
    bundle_snapshot = joint_view.get("bundle_snapshot", {})
    bundle_status = bundle_snapshot.get("bundle_status", "UNKNOWN")

    # Extract alignment status
    alignment_status_obj = joint_view.get("alignment_status", {})
    alignment_status = alignment_status_obj.get("overall_status", "UNKNOWN")

    # Compute derived fields
    status_light = _determine_status_light(topology_mode, bundle_status, alignment_status)
    topology_stability = _map_topology_mode_to_stability(topology_mode)
    bundle_stability = _map_bundle_status_to_stability(bundle_status)
    cross_system_consistency = consistency_result.get("consistent", False)
    conflict_codes = _extract_conflict_codes(joint_view, director_panel)
    headline = _build_neutral_headline(
        topology_mode, bundle_status, alignment_status, status_light
    )

    # Build tile
    tile: Dict[str, Any] = {
        "schema_version": TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "topology_stability": topology_stability,
        "bundle_stability": bundle_stability,
        "cross_system_consistency": cross_system_consistency,
        "joint_status": alignment_status,
        "conflict_codes": conflict_codes,
        "headline": headline,
    }

    return tile


# -----------------------------------------------------------------------------
# Governance signal builder
# -----------------------------------------------------------------------------

def topology_bundle_to_governance_signal(
    joint_view: Dict[str, Any],
    consistency_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert topology bundle state to governance signal format.

    STATUS: PHASE X — TOPOLOGY/BUNDLE GOVERNANCE SIGNAL

    Follows the GovernanceSignal semantics from replay_safety_governance_signal.schema.json.
    Prefixes reasons with [Topology] or [Bundle] as appropriate.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Zero gating logic — SHADOW MODE only

    Args:
        joint_view: Topology bundle joint view per topology_bundle_joint_view.schema.json
        consistency_result: Cross-system consistency evaluation result

    Returns:
        Governance signal dictionary with:
        - schema_version: "1.0.0"
        - signal_type: "topology_bundle"
        - status: "OK" | "WARN" | "BLOCK"
        - governance_status: str (mirrors status)
        - governance_alignment: "ALIGNED" | "TENSION" | "DIVERGENT"
        - topology_status: "OK" | "WARN" | "CRITICAL"
        - bundle_status: "OK" | "WARN" | "CRITICAL"
        - conflict: bool
        - reasons: List[str] (prefixed with [Topology] or [Bundle])
        - safe_for_policy_update: bool (SHADOW: always True)
        - safe_for_promotion: bool (SHADOW: always True)

    Example:
        >>> joint_view = {
        ...     "topology_snapshot": {"topology_mode": "DRIFT", ...},
        ...     "bundle_snapshot": {"bundle_status": "VALID", ...},
        ...     "alignment_status": {"overall_status": "TENSION", ...},
        ... }
        >>> consistency_result = {"consistent": True, "status": "OK"}
        >>> signal = topology_bundle_to_governance_signal(joint_view, consistency_result)
        >>> signal["status"]
        'WARN'
    """
    # Validate inputs
    _validate_joint_view(joint_view)
    _validate_consistency_result(consistency_result)

    # Extract states
    topology_snapshot = joint_view.get("topology_snapshot", {})
    topology_mode = topology_snapshot.get("topology_mode", "UNKNOWN")

    bundle_snapshot = joint_view.get("bundle_snapshot", {})
    bundle_chain_status = bundle_snapshot.get("bundle_status", "UNKNOWN")

    alignment_status_obj = joint_view.get("alignment_status", {})
    alignment_status = alignment_status_obj.get("overall_status", "UNKNOWN")

    # Map topology mode to governance status
    topology_status_map = {
        "STABLE": "OK",
        "DRIFT": "WARN",
        "TURBULENT": "WARN",
        "CRITICAL": "CRITICAL",
    }
    topology_status = topology_status_map.get(topology_mode, "OK")

    # Map bundle status to governance status
    bundle_status_map = {
        "VALID": "OK",
        "WARN": "WARN",
        "BROKEN": "CRITICAL",
        "MISSING": "CRITICAL",
    }
    bundle_status = bundle_status_map.get(bundle_chain_status, "OK")

    # Determine overall status (worst of topology and bundle)
    status_priority = {"CRITICAL": 3, "WARN": 2, "OK": 1}
    if status_priority.get(topology_status, 0) >= status_priority.get(bundle_status, 0):
        overall_status = topology_status
    else:
        overall_status = bundle_status

    # Map to governance signal status (CRITICAL → BLOCK)
    if overall_status == "CRITICAL":
        signal_status = "BLOCK"
    elif overall_status == "WARN":
        signal_status = "WARN"
    else:
        signal_status = "OK"

    # Determine governance alignment
    alignment_map = {
        "ALIGNED": "ALIGNED",
        "TENSION": "TENSION",
        "DIVERGENT": "DIVERGENT",
    }
    governance_alignment = alignment_map.get(alignment_status, "ALIGNED")

    # Determine conflict
    conflict = alignment_status == "DIVERGENT" or (
        topology_status == "CRITICAL" and bundle_status == "OK"
    ) or (
        topology_status == "OK" and bundle_status == "CRITICAL"
    )

    # Build reasons list with prefixes
    reasons: List[str] = []

    # Topology reasons
    if topology_mode == "STABLE":
        reasons.append("[Topology] Topology mode: STABLE")
    elif topology_mode == "DRIFT":
        reasons.append("[Topology] Topology drift detected")
    elif topology_mode == "TURBULENT":
        reasons.append("[Topology] Topology turbulence detected")
    elif topology_mode == "CRITICAL":
        reasons.append("[Topology] Critical topology invariant violation")
    else:
        reasons.append(f"[Topology] Topology mode: {topology_mode}")

    # Bundle reasons
    if bundle_chain_status == "VALID":
        reasons.append("[Bundle] Bundle chain: VALID")
    elif bundle_chain_status == "WARN":
        reasons.append("[Bundle] Bundle chain warning detected")
    elif bundle_chain_status == "BROKEN":
        reasons.append("[Bundle] Bundle chain integrity broken")
    elif bundle_chain_status == "MISSING":
        reasons.append("[Bundle] Bundle chain missing")
    else:
        reasons.append(f"[Bundle] Bundle status: {bundle_chain_status}")

    # Alignment reason
    if alignment_status == "ALIGNED":
        reasons.append("[Topology] Topology-bundle alignment: ALIGNED")
    elif alignment_status == "TENSION":
        reasons.append("[Topology] Topology-bundle alignment: TENSION")
    elif alignment_status == "DIVERGENT":
        reasons.append("[Topology] Topology-bundle alignment: DIVERGENT")

    # Consistency reason
    if consistency_result.get("consistent", False):
        reasons.append("[Bundle] Cross-system consistency: VERIFIED")
    else:
        reasons.append("[Bundle] Cross-system consistency: NOT VERIFIED")

    # Build signal
    signal: Dict[str, Any] = {
        "schema_version": TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION,
        "signal_type": "topology_bundle",
        "status": signal_status,
        "governance_status": signal_status,
        "governance_alignment": governance_alignment,
        "topology_status": topology_status,
        "bundle_status": bundle_status,
        "conflict": conflict,
        "reasons": reasons,
        # SHADOW MODE: always permissive
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
    }

    return signal


# -----------------------------------------------------------------------------
# Global health integration
# -----------------------------------------------------------------------------

def build_topology_bundle_tile_for_global_health(
    joint_view: Dict[str, Any],
    consistency_result: Dict[str, Any],
    director_panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build topology bundle tile for global health surface attachment.

    STATUS: PHASE X — TOPOLOGY/BUNDLE GLOBAL HEALTH INTEGRATION

    This is the entry point for global_surface.py to build the topology_bundle tile.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - Zero gating logic — SHADOW MODE only

    Args:
        joint_view: Topology bundle joint view per topology_bundle_joint_view.schema.json
        consistency_result: Cross-system consistency evaluation result
        director_panel: Optional director panel per topology_bundle_director_panel.schema.json

    Returns:
        Topology bundle tile suitable for global health surface attachment
    """
    return build_topology_bundle_console_tile(
        joint_view=joint_view,
        consistency_result=consistency_result,
        director_panel=director_panel,
    )


# -----------------------------------------------------------------------------
# P3 Stability Report Integration
# -----------------------------------------------------------------------------

def build_topology_bundle_summary_for_p3(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build topology bundle summary for P3 stability report.

    STATUS: PHASE X — TOPOLOGY/BUNDLE P3 BINDING

    Non-mutating helper that returns a dict suitable for embedding in
    first_light_stability_report.json.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: returns new dict

    Args:
        tile: Topology bundle console tile from build_topology_bundle_console_tile().

    Returns:
        P3 stability summary dict with:
        - topology_stability: str
        - bundle_stability: str
        - joint_status: str
        - conflict_codes: List[str]
        - status_light: str

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> summary = build_topology_bundle_summary_for_p3(tile)
        >>> summary["topology_stability"]
        'STABLE'
    """
    return {
        "topology_stability": tile.get("topology_stability", "UNKNOWN"),
        "bundle_stability": tile.get("bundle_stability", "UNKNOWN"),
        "joint_status": tile.get("joint_status", "UNKNOWN"),
        "conflict_codes": list(tile.get("conflict_codes", [])),
        "status_light": tile.get("status_light", "GREEN"),
    }


def add_topology_bundle_to_p3_stability_report(
    stability_report: Dict[str, Any],
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add topology bundle summary to P3 stability report.

    STATUS: PHASE X — TOPOLOGY/BUNDLE P3 BINDING

    Adds topology_bundle_summary field to stability report.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The added summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        stability_report: P3 stability report dictionary (read-only, not modified).
        tile: Topology bundle console tile from build_topology_bundle_console_tile().

    Returns:
        New stability report dict with topology_bundle_summary added.

    Example:
        >>> report = {"schema_version": "1.0.0", "run_id": "test", ...}
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> enriched = add_topology_bundle_to_p3_stability_report(report, tile)
        >>> enriched["topology_bundle_summary"]["topology_stability"]
        'STABLE'
    """
    # Non-mutating: create new dict
    updated = dict(stability_report)

    # Build and add summary
    updated["topology_bundle_summary"] = build_topology_bundle_summary_for_p3(tile)

    return updated


# -----------------------------------------------------------------------------
# P4 Calibration Report Integration
# -----------------------------------------------------------------------------

def build_topology_bundle_calibration_for_p4(
    tile: Dict[str, Any],
    joint_view: Optional[Dict[str, Any]] = None,
    consistency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build topology bundle calibration summary for P4 calibration report.

    STATUS: PHASE X — TOPOLOGY/BUNDLE P4 BINDING

    Focuses on structural calibration data: topology_mode, bundle_integration_status,
    cross_system_consistency, and any XCOR-* codes.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned calibration is purely observational
    - No control flow depends on the calibration contents
    - Non-mutating: returns new dict

    Args:
        tile: Topology bundle console tile from build_topology_bundle_console_tile().
        joint_view: Optional joint view for extracting raw topology_mode.
        consistency_result: Optional consistency result for cross-system status.

    Returns:
        P4 calibration summary dict with:
        - topology_mode: str (raw mode from joint_view or derived from stability)
        - bundle_integration_status: str
        - cross_system_consistency: bool
        - xcor_codes: List[str] (XCOR-* conflict codes only)
        - status_light: str
        - structural_notes: List[str]

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> calibration = build_topology_bundle_calibration_for_p4(tile, joint_view, consistency_result)
        >>> calibration["topology_mode"]
        'STABLE'
    """
    # Extract raw topology mode from joint_view if available
    topology_mode = "UNKNOWN"
    if joint_view is not None:
        topology_snapshot = joint_view.get("topology_snapshot", {})
        topology_mode = topology_snapshot.get("topology_mode", "UNKNOWN")
    else:
        # Derive from stability status
        stability = tile.get("topology_stability", "UNKNOWN")
        stability_to_mode = {
            "STABLE": "STABLE",
            "DRIFTING": "DRIFT",
            "TURBULENT": "TURBULENT",
            "CRITICAL": "CRITICAL",
        }
        topology_mode = stability_to_mode.get(stability, "UNKNOWN")

    # Extract bundle integration status
    bundle_stability = tile.get("bundle_stability", "UNKNOWN")
    bundle_integration_status_map = {
        "VALID": "INTEGRATED",
        "ATTENTION": "PARTIAL",
        "BROKEN": "BROKEN",
        "MISSING": "MISSING",
    }
    bundle_integration_status = bundle_integration_status_map.get(bundle_stability, "UNKNOWN")

    # Extract cross-system consistency
    cross_system_consistency = tile.get("cross_system_consistency", False)
    if consistency_result is not None:
        cross_system_consistency = consistency_result.get("consistent", cross_system_consistency)

    # Extract XCOR codes only
    all_codes = tile.get("conflict_codes", [])
    xcor_codes = [code for code in all_codes if code.startswith("XCOR-")]

    # Build structural notes
    structural_notes: List[str] = []
    if topology_mode == "DRIFT":
        structural_notes.append("Topology drift detected during calibration window")
    elif topology_mode == "TURBULENT":
        structural_notes.append("Topology turbulence observed during calibration")
    elif topology_mode == "CRITICAL":
        structural_notes.append("Critical topology invariant violation during calibration")

    if bundle_integration_status == "PARTIAL":
        structural_notes.append("Bundle integration incomplete")
    elif bundle_integration_status == "BROKEN":
        structural_notes.append("Bundle chain integrity broken")

    if not cross_system_consistency:
        structural_notes.append("Cross-system consistency not verified")

    if xcor_codes:
        structural_notes.append(f"Active correlation codes: {', '.join(xcor_codes)}")

    return {
        "topology_mode": topology_mode,
        "bundle_integration_status": bundle_integration_status,
        "cross_system_consistency": cross_system_consistency,
        "xcor_codes": xcor_codes,
        "status_light": tile.get("status_light", "GREEN"),
        "structural_notes": structural_notes,
    }


def add_topology_bundle_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    tile: Dict[str, Any],
    joint_view: Optional[Dict[str, Any]] = None,
    consistency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add topology bundle calibration to P4 calibration report.

    STATUS: PHASE X — TOPOLOGY/BUNDLE P4 BINDING

    Adds topology_bundle_calibration field to calibration report.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The added calibration is purely observational
    - No control flow depends on the calibration contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        calibration_report: P4 calibration report dictionary (read-only, not modified).
        tile: Topology bundle console tile from build_topology_bundle_console_tile().
        joint_view: Optional joint view for extracting raw topology_mode.
        consistency_result: Optional consistency result for cross-system status.

    Returns:
        New calibration report dict with topology_bundle_calibration added.

    Example:
        >>> report = {"schema_version": "1.0.0", "run_id": "test", ...}
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> enriched = add_topology_bundle_to_p4_calibration_report(report, tile, joint_view)
        >>> enriched["topology_bundle_calibration"]["topology_mode"]
        'STABLE'
    """
    # Non-mutating: create new dict
    updated = dict(calibration_report)

    # Build and add calibration
    updated["topology_bundle_calibration"] = build_topology_bundle_calibration_for_p4(
        tile=tile,
        joint_view=joint_view,
        consistency_result=consistency_result,
    )

    return updated


# -----------------------------------------------------------------------------
# Evidence Integration (Enhanced)
# -----------------------------------------------------------------------------

def attach_topology_bundle_to_evidence(
    evidence: Dict[str, Any],
    console_tile: Dict[str, Any],
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach topology bundle tile and signal to an evidence pack (read-only, additive).

    STATUS: PHASE X — TOPOLOGY/BUNDLE EVIDENCE INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["topology_bundle"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        console_tile: Topology bundle console tile from build_topology_bundle_console_tile().
        governance_signal: Optional governance signal from topology_bundle_to_governance_signal().

    Returns:
        New dict with evidence contents plus topology_bundle attached under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> signal = topology_bundle_to_governance_signal(joint_view, consistency_result)
        >>> enriched = attach_topology_bundle_to_evidence(evidence, tile, signal)
        >>> enriched["governance"]["topology_bundle"]["status_light"]
        'GREEN'
    """
    # Non-mutating: create new dict
    updated = dict(evidence)

    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])

    # Extract relevant fields from console tile for evidence
    topology_bundle_summary: Dict[str, Any] = {
        "status_light": console_tile.get("status_light", "GREEN"),
        "topology_stability": console_tile.get("topology_stability", "UNKNOWN"),
        "bundle_stability": console_tile.get("bundle_stability", "UNKNOWN"),
        "cross_system_consistency": console_tile.get("cross_system_consistency", False),
        "joint_status": console_tile.get("joint_status", "UNKNOWN"),
        "conflict_codes": list(console_tile.get("conflict_codes", [])),
    }

    # Include governance signal fields if provided
    if governance_signal is not None:
        topology_bundle_summary["governance_status"] = governance_signal.get("status", "OK")
        topology_bundle_summary["governance_alignment"] = governance_signal.get(
            "governance_alignment", "ALIGNED"
        )
        topology_bundle_summary["topology_signal_status"] = governance_signal.get(
            "topology_status", "OK"
        )
        topology_bundle_summary["bundle_signal_status"] = governance_signal.get(
            "bundle_status", "OK"
        )
        topology_bundle_summary["conflict"] = governance_signal.get("conflict", False)
        # Extract prefixed reasons
        reasons = governance_signal.get("reasons", [])
        topology_bundle_summary["reasons"] = list(reasons)

    # Attach summary
    updated["governance"]["topology_bundle"] = topology_bundle_summary

    return updated


# -----------------------------------------------------------------------------
# First Light Signal Extraction
# -----------------------------------------------------------------------------

def extract_topology_bundle_signal_for_first_light(
    tile: Dict[str, Any],
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract minimal topology bundle signal for First Light reporting.

    STATUS: PHASE X — TOPOLOGY/BUNDLE FIRST LIGHT INTEGRATION

    Returns a concise signal dict suitable for First Light evidence chains.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        tile: Topology bundle console tile from build_topology_bundle_console_tile().
        governance_signal: Optional governance signal for status extraction.

    Returns:
        Minimal First Light signal dict with:
        - status: str (OK, WARN, BLOCK)
        - topology_mode: str (derived from stability)
        - bundle_status: str (derived from stability)
        - conflict_count: int

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency_result)
        >>> signal = extract_topology_bundle_signal_for_first_light(tile)
        >>> signal["status"]
        'OK'
    """
    # Determine status from governance signal or derive from status_light
    if governance_signal is not None:
        status = governance_signal.get("status", "OK")
    else:
        status_light = tile.get("status_light", "GREEN")
        status_light_to_status = {
            "GREEN": "OK",
            "YELLOW": "WARN",
            "RED": "BLOCK",
        }
        status = status_light_to_status.get(status_light, "OK")

    # Map stability back to mode
    topology_stability = tile.get("topology_stability", "UNKNOWN")
    stability_to_mode = {
        "STABLE": "STABLE",
        "DRIFTING": "DRIFT",
        "TURBULENT": "TURBULENT",
        "CRITICAL": "CRITICAL",
    }
    topology_mode = stability_to_mode.get(topology_stability, "UNKNOWN")

    # Map bundle stability to status
    bundle_stability = tile.get("bundle_stability", "UNKNOWN")
    stability_to_status = {
        "VALID": "VALID",
        "ATTENTION": "WARN",
        "BROKEN": "BROKEN",
        "MISSING": "MISSING",
    }
    bundle_status = stability_to_status.get(bundle_stability, "UNKNOWN")

    # Count conflict codes
    conflict_codes = tile.get("conflict_codes", [])
    conflict_count = len(conflict_codes)

    return {
        "status": status,
        "topology_mode": topology_mode,
        "bundle_status": bundle_status,
        "conflict_count": conflict_count,
    }


__all__ = [
    "TOPOLOGY_BUNDLE_TILE_SCHEMA_VERSION",
    "build_topology_bundle_console_tile",
    "topology_bundle_to_governance_signal",
    "build_topology_bundle_tile_for_global_health",
    "build_topology_bundle_summary_for_p3",
    "add_topology_bundle_to_p3_stability_report",
    "build_topology_bundle_calibration_for_p4",
    "add_topology_bundle_to_p4_calibration_report",
    "attach_topology_bundle_to_evidence",
    "extract_topology_bundle_signal_for_first_light",
]
