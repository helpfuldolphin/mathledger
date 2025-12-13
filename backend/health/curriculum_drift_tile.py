"""Curriculum drift tile module.

Provides curriculum drift monitoring tiles for health dashboard.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CurriculumDriftTile:
    """Curriculum drift tile data."""
    name: str
    status: str = "ok"
    drift_score: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_curriculum_drift_tile(
    name: str,
    drift_events: Optional[List[Dict[str, Any]]] = None,
) -> CurriculumDriftTile:
    """Build curriculum drift tile from events."""
    drift_events = drift_events or []
    drift_score = len(drift_events) * 0.1 if drift_events else 0.0
    return CurriculumDriftTile(
        name=name,
        status="ok" if drift_score < 0.5 else "warn",
        drift_score=drift_score,
        events=drift_events,
        metadata={"event_count": len(drift_events)},
    )


def summarize_drift_for_health(
    tiles: List[CurriculumDriftTile],
) -> Dict[str, Any]:
    """Summarize drift tiles for global health."""
    total_drift = sum(t.drift_score for t in tiles)
    return {
        "tile_count": len(tiles),
        "total_drift": total_drift,
        "status": "ok" if total_drift < 1.0 else "warn",
    }


def build_curriculum_drift_tile_for_global_health(
    name: str,
    drift_events: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build curriculum drift tile for global health dashboard."""
    tile = build_curriculum_drift_tile(name, drift_events)
    return {
        "name": tile.name,
        "status": tile.status,
        "drift_score": tile.drift_score,
        "event_count": len(tile.events),
        "config": config or {},
    }


__all__ = [
    "CurriculumDriftTile",
    "build_curriculum_drift_tile",
    "build_curriculum_drift_tile_for_global_health",
    "summarize_drift_for_health",
]
