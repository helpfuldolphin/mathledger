"""
Metrics Governance Console Tile Adapter.

Provides metrics governance tile for global health integration.
Synthesizes drift compass, budget view, and governance signal into
a unified console tile.

SHADOW MODE: Observation-only. No control paths.
This tile does NOT influence safety, replay, or any other governance layer.

Phase X: Metrics Conformance Layer (CLAUDE D)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

METRICS_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"

# Status light thresholds
_DRIFT_CRITICAL_THRESHOLD = 0.7
_DRIFT_WARN_THRESHOLD = 0.3
_BUDGET_CRITICAL_THRESHOLD = 95.0
_BUDGET_WARN_THRESHOLD = 80.0
_SUCCESS_RATE_CRITICAL_THRESHOLD = 50.0
_SUCCESS_RATE_WARN_THRESHOLD = 80.0


def _determine_status_light(
    drift_compass: Optional[Dict[str, Any]],
    budget_view: Optional[Dict[str, Any]],
    governance_signal: Optional[Dict[str, Any]],
) -> str:
    """
    Determine status light from input signals.

    SHADOW MODE: Observation-only. No control paths.
    This function is a pure mapper with no side effects.

    Returns:
        "GREEN" | "YELLOW" | "RED"
    """
    # If governance_signal has explicit status, prefer it
    if governance_signal is not None:
        gov_status = governance_signal.get("governance_status", "").upper()
        if gov_status == "BLOCK":
            return "RED"
        if gov_status == "WARN":
            return "YELLOW"
        if gov_status == "OK":
            return "GREEN"

    # Check drift compass
    if drift_compass is not None:
        heading = drift_compass.get("compass_heading", "").upper()
        drift_mag = drift_compass.get("drift_magnitude", 0.0)

        if heading == "CRITICAL" or drift_mag >= _DRIFT_CRITICAL_THRESHOLD:
            return "RED"
        if heading in ("DRIFTING", "DIVERGING") or drift_mag >= _DRIFT_WARN_THRESHOLD:
            return "YELLOW"

    # Check budget view
    if budget_view is not None:
        budget_status = budget_view.get("budget_status", "").upper()
        utilization = budget_view.get("utilization_pct", 0.0)

        if budget_status in ("CRITICAL", "EXCEEDED") or utilization >= _BUDGET_CRITICAL_THRESHOLD:
            return "RED"
        if budget_status == "ELEVATED" or utilization >= _BUDGET_WARN_THRESHOLD:
            return "YELLOW"

    # Check FO vital signs via governance_signal
    if governance_signal is not None:
        sub_signals = governance_signal.get("sub_signals", {})
        fo_signals = sub_signals.get("fo_vital_signs", {})
        success_rate = fo_signals.get("success_rate", 100.0)
        health_status = fo_signals.get("health_status", "").upper()

        if health_status == "CRITICAL" or success_rate < _SUCCESS_RATE_CRITICAL_THRESHOLD:
            return "RED"
        if health_status == "DEGRADED" or success_rate < _SUCCESS_RATE_WARN_THRESHOLD:
            return "YELLOW"

    return "GREEN"


def _build_headline(
    status_light: str,
    drift_compass: Optional[Dict[str, Any]],
    budget_view: Optional[Dict[str, Any]],
    governance_signal: Optional[Dict[str, Any]],
) -> str:
    """
    Build neutral descriptive headline for metrics console tile.

    SHADOW MODE: Observation-only. No control paths.
    Language must be purely descriptive with no evaluative terms.
    """
    parts: List[str] = []

    # Drift summary
    if drift_compass is not None:
        heading = drift_compass.get("compass_heading", "STABLE")
        drift_mag = drift_compass.get("drift_magnitude", 0.0)
        parts.append(f"Drift: {heading} ({drift_mag:.2f})")

    # Budget summary
    if budget_view is not None:
        budget_status = budget_view.get("budget_status", "NOMINAL")
        utilization = budget_view.get("utilization_pct", 0.0)
        parts.append(f"Budget: {budget_status} ({utilization:.1f}%)")

    # FO summary
    if governance_signal is not None:
        sub_signals = governance_signal.get("sub_signals", {})
        fo_signals = sub_signals.get("fo_vital_signs", {})
        health = fo_signals.get("health_status", "UNKNOWN")
        success_rate = fo_signals.get("success_rate", 0.0)
        parts.append(f"FO: {health} ({success_rate:.1f}%)")

    if not parts:
        return "No metrics data available."

    return ". ".join(parts) + "."


def _extract_blocked_metrics(
    drift_compass: Optional[Dict[str, Any]],
    budget_view: Optional[Dict[str, Any]],
    governance_signal: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Extract list of metrics that are in blocking state.

    SHADOW MODE: Observation-only. No control paths.
    """
    blocked: List[str] = []

    # Check drift axes
    if drift_compass is not None:
        axes = drift_compass.get("axes", [])
        for axis in axes:
            if axis.get("status", "").upper() == "CRITICAL":
                blocked.append(f"drift:{axis.get('axis_name', 'unknown')}")

    # Check budget resources
    if budget_view is not None:
        blocking_resources = budget_view.get("governance_implication", {}).get("blocking_resources", [])
        for resource in blocking_resources:
            blocked.append(f"budget:{resource}")

    # Check governance signal blocking
    if governance_signal is not None:
        if governance_signal.get("governance_status", "").upper() == "BLOCK":
            # Add sub-signal blocking reasons
            sub_signals = governance_signal.get("sub_signals", {})
            for name, signal in sub_signals.items():
                if signal.get("status", "").upper() == "BLOCK":
                    blocked.append(f"signal:{name}")

    return sorted(set(blocked))


def _extract_budget_risk(budget_view: Optional[Dict[str, Any]]) -> str:
    """
    Extract budget risk band from budget view.

    SHADOW MODE: Observation-only. No control paths.
    """
    if budget_view is None:
        return "UNKNOWN"

    budget_status = budget_view.get("budget_status", "NOMINAL").upper()

    if budget_status in ("CRITICAL", "EXCEEDED"):
        return "HIGH"
    if budget_status == "ELEVATED":
        return "MEDIUM"
    if budget_status in ("NOMINAL", "UNDER"):
        return "LOW"

    return "UNKNOWN"


def _extract_fo_vitality_summary(
    governance_signal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract FO vitality summary from governance signal.

    SHADOW MODE: Observation-only. No control paths.
    """
    if governance_signal is None:
        return {
            "health_status": "UNKNOWN",
            "success_rate": 0.0,
            "throttle_recommended": False,
            "boost_allowed": False,
        }

    sub_signals = governance_signal.get("sub_signals", {})
    fo_signals = sub_signals.get("fo_vital_signs", {})

    return {
        "health_status": fo_signals.get("health_status", "UNKNOWN"),
        "success_rate": fo_signals.get("success_rate", 0.0),
        "throttle_recommended": fo_signals.get("should_throttle", False),
        "boost_allowed": fo_signals.get("should_boost", False),
    }


def build_metrics_console_tile(
    drift_compass: Optional[Dict[str, Any]] = None,
    budget_view: Optional[Dict[str, Any]] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metrics console tile for global health surface.

    SHADOW MODE: Observation-only. No control paths.
    This tile is purely observational and does NOT influence other tiles.
    It does NOT affect safety, replay, or any other governance layer.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_compass: Dict from metric_drift_compass.schema.json.
            Expected keys: compass_heading, drift_magnitude, axes, poly_cause_detected.
        budget_view: Dict from metric_budget_joint_view.schema.json.
            Expected keys: budget_status, utilization_pct, allocations, governance_implication.
        governance_signal: Dict from metric_governance_signal.schema.json.
            Expected keys: governance_status, sub_signals, safe_for_promotion.

    Returns:
        Dict with console tile:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - headline: str (neutral descriptive text)
        - drift_heading: str (compass heading)
        - blocked_metrics: List[str] (metrics in blocking state)
        - budget_risk: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
        - fo_vitality_summary: Dict with FO health summary
        - governance_status: "OK" | "WARN" | "BLOCK" | "UNKNOWN"
        - safe_for_promotion: bool
    """
    status_light = _determine_status_light(drift_compass, budget_view, governance_signal)
    headline = _build_headline(status_light, drift_compass, budget_view, governance_signal)
    blocked_metrics = _extract_blocked_metrics(drift_compass, budget_view, governance_signal)
    budget_risk = _extract_budget_risk(budget_view)
    fo_vitality_summary = _extract_fo_vitality_summary(governance_signal)

    # Extract drift heading
    drift_heading = "UNKNOWN"
    if drift_compass is not None:
        drift_heading = drift_compass.get("compass_heading", "UNKNOWN")

    # Extract governance status
    governance_status = "UNKNOWN"
    safe_for_promotion = True
    if governance_signal is not None:
        governance_status = governance_signal.get("governance_status", "UNKNOWN")
        safe_for_promotion = governance_signal.get("safe_for_promotion", True)

    return {
        "schema_version": METRICS_GOVERNANCE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "headline": headline,
        "drift_heading": drift_heading,
        "blocked_metrics": blocked_metrics,
        "budget_risk": budget_risk,
        "fo_vitality_summary": fo_vitality_summary,
        "governance_status": governance_status,
        "safe_for_promotion": safe_for_promotion,
    }


def build_metrics_governance_tile_for_global_health(
    drift_compass: Optional[Dict[str, Any]] = None,
    budget_view: Optional[Dict[str, Any]] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metrics governance tile specifically for global health surface attachment.

    SHADOW MODE: Observation-only. No control paths.
    This is an alias for build_metrics_console_tile() with same contract.

    Args:
        drift_compass: Drift compass data.
        budget_view: Budget view data.
        governance_signal: Governance signal data.

    Returns:
        Metrics governance tile dict.
    """
    return build_metrics_console_tile(
        drift_compass=drift_compass,
        budget_view=budget_view,
        governance_signal=governance_signal,
    )


def metrics_for_alignment_view(
    governance_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert governance signal to alignment view format for CLAUDE I fusion.

    SHADOW MODE: Observation-only. No control paths.
    This function performs ONLY mapping and summarization.
    No semantics expansion. No governance decisions.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        governance_signal: Dict from metric_governance_signal.schema.json.
            Required keys: governance_status, governance_alignment, sub_signals.

    Returns:
        Dict with alignment view:
        - layer: "metrics" (constant)
        - status: "ok" | "warn" | "block" (lowercase for GovernanceSignal compatibility)
        - alignment: "ALIGNED" | "TENSION" | "DIVERGENT"
        - severity: "info" | "warning" | "critical"
        - summary: str (brief description)
        - sub_signal_statuses: Dict[str, str] (sub-signal name → status)
        - safe_for_promotion: bool
    """
    # Map governance_status to lowercase status
    gov_status = governance_signal.get("governance_status", "OK").upper()
    status_map = {"OK": "ok", "WARN": "warn", "BLOCK": "block"}
    status = status_map.get(gov_status, "ok")

    # Map severity
    severity_map = {"OK": "info", "WARN": "warning", "BLOCK": "critical"}
    severity = severity_map.get(gov_status, "info")

    # Extract alignment
    alignment = governance_signal.get("governance_alignment", "ALIGNED")

    # Extract sub-signal statuses
    sub_signals = governance_signal.get("sub_signals", {})
    sub_signal_statuses: Dict[str, str] = {}
    for name, signal in sub_signals.items():
        sig_status = signal.get("status", "OK").upper()
        sub_signal_statuses[name] = status_map.get(sig_status, "ok")

    # Build summary from reasons
    reasons = governance_signal.get("reasons", [])
    if reasons:
        # Take first reason as summary, strip prefix
        first_reason = reasons[0]
        # Remove prefix like "[DriftCompass] " if present
        if "]" in first_reason:
            first_reason = first_reason.split("]", 1)[1].strip()
        summary = first_reason
    else:
        summary = f"Metrics layer status: {gov_status}"

    return {
        "layer": "metrics",
        "status": status,
        "alignment": alignment,
        "severity": severity,
        "summary": summary,
        "sub_signal_statuses": sub_signal_statuses,
        "safe_for_promotion": governance_signal.get("safe_for_promotion", True),
    }


def summarize_metrics_for_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize metrics governance tile for uplift council decision-making.

    SHADOW MODE: Observation-only. No control paths.
    This function provides structural monitoring information only.
    It does NOT make governance decisions.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        tile: Metrics governance tile from build_metrics_console_tile().

    Returns:
        Dict with council summary:
        - status: "OK" | "WARN" | "BLOCK"
        - drift_heading: str
        - budget_risk: str
        - fo_health: str
        - blocked_metrics: List[str]
        - summary: str (neutral descriptive summary)
    """
    status_light = tile.get("status_light", "GREEN")

    # Map status_light to governance status
    status_map = {"GREEN": "OK", "YELLOW": "WARN", "RED": "BLOCK"}
    status = status_map.get(status_light, "OK")

    drift_heading = tile.get("drift_heading", "UNKNOWN")
    budget_risk = tile.get("budget_risk", "UNKNOWN")
    fo_vitality = tile.get("fo_vitality_summary", {})
    fo_health = fo_vitality.get("health_status", "UNKNOWN")
    blocked_metrics = tile.get("blocked_metrics", [])

    # Build summary
    summary_parts: List[str] = []
    summary_parts.append(f"Drift heading: {drift_heading}")
    summary_parts.append(f"Budget risk: {budget_risk}")
    summary_parts.append(f"FO health: {fo_health}")
    if blocked_metrics:
        summary_parts.append(f"Blocked: {', '.join(blocked_metrics)}")

    summary = ". ".join(summary_parts) + "."

    return {
        "status": status,
        "drift_heading": drift_heading,
        "budget_risk": budget_risk,
        "fo_health": fo_health,
        "blocked_metrics": blocked_metrics,
        "summary": summary,
    }


def build_metrics_summary_for_p3(
    drift_compass: Optional[Dict[str, Any]] = None,
    budget_view: Optional[Dict[str, Any]] = None,
    governance_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metrics governance summary for P3 First Light stability reports.

    SHADOW MODE: Observation-only. No control paths.
    This summary is attached to P3 stability reports under `metrics_governance_summary`.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_compass: Optional drift compass data from metric_drift_compass.schema.json.
        budget_view: Optional budget view data from metric_budget_joint_view.schema.json.
        governance_signal: Optional governance signal from metric_governance_signal.schema.json.

    Returns:
        Dict with P3 summary:
        - drift_heading: str (STABLE | DRIFTING | DIVERGING | CRITICAL | UNKNOWN)
        - blocked_metrics: List[str] (metrics in blocking state)
        - budget_risk: "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
        - fo_vitality_summary: Dict with FO health summary
        - status_light: "GREEN" | "YELLOW" | "RED"
        - schema_version: "1.0.0"
    """
    status_light = _determine_status_light(drift_compass, budget_view, governance_signal)
    blocked_metrics = _extract_blocked_metrics(drift_compass, budget_view, governance_signal)
    budget_risk = _extract_budget_risk(budget_view)
    fo_vitality_summary = _extract_fo_vitality_summary(governance_signal)

    # Extract drift heading
    drift_heading = "UNKNOWN"
    if drift_compass is not None:
        drift_heading = drift_compass.get("compass_heading", "UNKNOWN")

    return {
        "schema_version": METRICS_GOVERNANCE_TILE_SCHEMA_VERSION,
        "drift_heading": drift_heading,
        "blocked_metrics": blocked_metrics,
        "budget_risk": budget_risk,
        "fo_vitality_summary": fo_vitality_summary,
        "status_light": status_light,
    }


def attach_metrics_governance_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach metrics governance data to an evidence pack (read-only, non-mutating).

    SHADOW MODE: Observation-only. No control paths.
    This function creates a NEW dict with evidence contents plus metrics governance
    data attached under evidence["governance"]["metrics"].

    The original evidence dict is NOT modified.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        tile: Metrics governance tile from build_metrics_console_tile().
        signal: Optional governance signal for additional context.

    Returns:
        New dict with evidence contents plus metrics governance data attached
        under evidence["governance"]["metrics"].

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_metrics_console_tile(drift_compass, budget_view, signal)
        >>> enriched = attach_metrics_governance_to_evidence(evidence, tile, signal)
        >>> "governance" in enriched
        True
        >>> "metrics" in enriched["governance"]
        True
    """
    # Create a shallow copy to avoid mutating the original
    enriched: Dict[str, Any] = {}
    for key, value in evidence.items():
        enriched[key] = value

    # Ensure governance key exists (copy if present)
    if "governance" not in enriched:
        enriched["governance"] = {}
    else:
        # Copy the governance dict to avoid mutation
        enriched["governance"] = dict(enriched["governance"])

    # Extract governance status from signal if available
    governance_status = "UNKNOWN"
    if signal is not None:
        governance_status = signal.get("governance_status", "UNKNOWN")

    # Attach metrics governance data
    enriched["governance"]["metrics"] = {
        "status_light": tile.get("status_light", "GREEN"),
        "drift_heading": tile.get("drift_heading", "UNKNOWN"),
        "blocked_metrics": tile.get("blocked_metrics", []),
        "budget_risk": tile.get("budget_risk", "UNKNOWN"),
        "governance_status": governance_status,
        "schema_version": METRICS_GOVERNANCE_TILE_SCHEMA_VERSION,
    }

    return enriched


__all__ = [
    "METRICS_GOVERNANCE_TILE_SCHEMA_VERSION",
    "attach_metrics_governance_to_evidence",
    "build_metrics_console_tile",
    "build_metrics_governance_tile_for_global_health",
    "build_metrics_summary_for_p3",
    "metrics_for_alignment_view",
    "summarize_metrics_for_council",
]
