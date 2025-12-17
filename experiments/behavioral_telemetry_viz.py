"""Behavioral telemetry visualization module.

Provides visualization utilities for telemetry data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryVisualization:
    """Telemetry visualization result."""
    chart_type: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def visualize_telemetry(
    data: List[Dict[str, Any]],
    chart_type: str = "line",
) -> TelemetryVisualization:
    """Create telemetry visualization."""
    return TelemetryVisualization(
        chart_type=chart_type,
        data=data,
        metadata={"point_count": len(data)},
    )


def export_telemetry_chart(
    viz: TelemetryVisualization,
    output_path: str,
    format: str = "json",
) -> None:
    """Export telemetry chart to file."""
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "chart_type": viz.chart_type,
            "data": viz.data,
            "metadata": viz.metadata,
        }, f, indent=2)


def aggregate_behavioral_metrics(
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate behavioral metrics from telemetry events."""
    return {
        "event_count": len(events),
        "unique_types": len(set(e.get("type", "unknown") for e in events)),
        "status": "ok",
    }


@dataclass
class TelemetryDirectorTile:
    """Telemetry director tile for dashboard."""
    status: str = "ok"
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


def build_telemetry_director_tile_v2(
    events: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> TelemetryDirectorTile:
    """Build telemetry director tile v2."""
    config = config or {}
    aggregated = aggregate_behavioral_metrics(events)

    alerts = []
    if aggregated["event_count"] == 0:
        alerts.append("No events")

    status = "ok" if not alerts else "warn"

    return TelemetryDirectorTile(
        status=status,
        metrics=aggregated,
        alerts=alerts,
    )


def build_telemetry_drift_envelope(
    events: List[Dict[str, Any]],
    window_size: int = 10,
) -> Dict[str, Any]:
    """Build telemetry drift envelope from events."""
    if not events:
        return {
            "drift_score": 0.0,
            "drift_band": "LOW",
            "envelope_status": "stable",
            "window_size": window_size,
        }

    # Simple drift calculation
    drift_score = len(events) * 0.02

    if drift_score < 0.3:
        drift_band = "LOW"
    elif drift_score < 0.6:
        drift_band = "MEDIUM"
    else:
        drift_band = "HIGH"

    envelope_status = "stable" if drift_score < 0.5 else "drifting"

    return {
        "drift_score": min(1.0, drift_score),
        "drift_band": drift_band,
        "envelope_status": envelope_status,
        "event_count": len(events),
        "window_size": window_size,
        "plots_with_repeated_drift": [],
        "neutral_notes": [],
    }


def build_telemetry_driven_uplift_phase_gate(
    events: List[Dict[str, Any]],
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """Build telemetry-driven uplift phase gate."""
    aggregated = aggregate_behavioral_metrics(events)
    event_count = aggregated.get("event_count", 0)
    score = min(1.0, event_count * 0.1)
    passed = score >= threshold

    return {
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "event_count": event_count,
    }


def build_telemetry_topology_semantic_fusion(
    telemetry_envelope: Dict[str, Any],
    topology_struct: Dict[str, Any],
    semantic_struct: Dict[str, Any],
) -> Dict[str, Any]:
    """Build telemetry–topology–semantic fusion tile."""
    # Extract individual status
    telemetry_band = telemetry_envelope.get("drift_band", "LOW")
    topology_stability = topology_struct.get("stability_score", 1.0)
    semantic_stability = semantic_struct.get("stability_score", 1.0)

    # Compute fusion band
    if telemetry_band == "LOW" and topology_stability >= 0.8 and semantic_stability >= 0.8:
        fusion_band = "LOW"
    elif telemetry_band == "HIGH" or topology_stability < 0.5 or semantic_stability < 0.5:
        fusion_band = "HIGH"
    else:
        fusion_band = "MEDIUM"

    return {
        "fusion_band": fusion_band,
        "telemetry_band": telemetry_band,
        "topology_stability": topology_stability,
        "semantic_stability": semantic_stability,
        "warnings": [],
        "status": "ok" if fusion_band == "LOW" else "warn",
    }


def summarize_telemetry_for_global_health(
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize telemetry for global health display."""
    aggregated = aggregate_behavioral_metrics(events)
    envelope = build_telemetry_drift_envelope(events)

    return {
        "event_count": aggregated["event_count"],
        "unique_types": aggregated["unique_types"],
        "drift_band": envelope["drift_band"],
        "status": "ok" if envelope["drift_band"] == "LOW" else "warn",
    }


def summarize_telemetry_for_uplift_safety(
    events: List[Dict[str, Any]],
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """Summarize telemetry for uplift safety assessment."""
    gate = build_telemetry_driven_uplift_phase_gate(events, threshold)
    envelope = build_telemetry_drift_envelope(events)

    return {
        "safe_for_uplift": gate["passed"],
        "score": gate["score"],
        "threshold": threshold,
        "drift_band": envelope["drift_band"],
        "status": "ok" if gate["passed"] else "blocked",
    }


__all__ = [
    "TelemetryVisualization",
    "TelemetryDirectorTile",
    "visualize_telemetry",
    "export_telemetry_chart",
    "aggregate_behavioral_metrics",
    "build_telemetry_director_tile_v2",
    "build_telemetry_drift_envelope",
    "build_telemetry_driven_uplift_phase_gate",
    "build_telemetry_topology_semantic_fusion",
    "summarize_telemetry_for_global_health",
    "summarize_telemetry_for_uplift_safety",
]
