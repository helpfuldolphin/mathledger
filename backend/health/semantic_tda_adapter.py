"""
Semantic/TDA Cross-Correlation Adapter — Phase V Integration

Operation CORTEX: Phase V Semantic/TDA Neural Link
==================================================

This module provides integration helpers for wiring the semantic/TDA governance tile
into the global health builder.

The semantic/TDA tile combines:
- Semantic graph drift signals (from experiments/semantic_consistency_audit.py)
- TDA topology health signals (from backend/health/tda_adapter.py)

Usage:
    from backend.health.semantic_tda_adapter import (
        build_semantic_tda_tile_for_global_health,
    )

    semantic_tile = build_semantic_tda_tile_for_global_health(
        semantic_panel=director_panel,
        tda_tile=tda_health_tile,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

# Import from semantic consistency audit (Phase V functions)
from experiments.semantic_consistency_audit import (
    build_semantic_drift_timeline,
    build_semantic_director_panel,
    correlate_semantic_and_tda_signals,
    build_semantic_tda_governance_tile,
)

# Import TDA adapter
from .tda_adapter import summarize_tda_for_global_health


def build_semantic_tda_tile_for_global_health(
    semantic_panel: Mapping[str, Any],
    tda_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic/TDA governance tile for global health integration.
    
    This is the main integration point for wiring semantic/TDA cross-correlation
    into global_health.json.
    
    Args:
        semantic_panel: Semantic director panel from build_semantic_director_panel.
            Expected keys: semantic_status_light, alignment_status, critical_run_ids, headline
        tda_tile: TDA health tile from summarize_tda_for_global_health.
            Expected keys: tda_status, block_rate, hss_trend, governance_signal
    
    Returns:
        Semantic/TDA governance tile dictionary ready for inclusion in global_health.json.
        The tile will be placed at global_health["semantic_tda"].
    
    Raises:
        ValueError: If required keys are missing from inputs.
    
    Example:
        >>> semantic_panel = {
        ...     "semantic_status_light": "RED",
        ...     "alignment_status": "MISALIGNED",
        ...     "critical_run_ids": ["run1"],
        ...     "headline": "Semantic graph shows critical drift",
        ... }
        >>> tda_tile = {
        ...     "tda_status": "ALERT",
        ...     "block_rate": 0.25,
        ...     "hss_trend": "DEGRADING",
        ...     "governance_signal": "BLOCK",
        ... }
        >>> tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_tile)
        >>> assert tile["status"] == "BLOCK"
        >>> assert tile["status_light"] == "RED"
    """
    # Validate inputs
    _validate_semantic_panel(semantic_panel)
    _validate_tda_tile(tda_tile)
    
    # Build correlation
    # Note: We need a semantic_timeline for correlation, but we only have semantic_panel.
    # For integration purposes, we'll create a minimal timeline from the panel.
    semantic_timeline = _extract_timeline_from_panel(semantic_panel)
    
    # Build correlation
    correlation = correlate_semantic_and_tda_signals(
        semantic_timeline=semantic_timeline,
        tda_health=tda_tile,
    )
    
    # Build governance tile
    tile = build_semantic_tda_governance_tile(
        semantic_panel=dict(semantic_panel),
        tda_panel=dict(tda_tile),
        correlation=correlation,
    )
    
    return tile


def _validate_semantic_panel(panel: Mapping[str, Any]) -> None:
    """Validate semantic panel has required keys."""
    required_keys = ["semantic_status_light"]
    missing = [key for key in required_keys if key not in panel]
    if missing:
        raise ValueError(
            f"semantic_panel missing required keys: {missing}. "
            f"Expected keys: {required_keys}"
        )


def _validate_tda_tile(tile: Mapping[str, Any]) -> None:
    """Validate TDA tile has required keys."""
    required_keys = ["tda_status"]
    missing = [key for key in required_keys if key not in tile]
    if missing:
        raise ValueError(
            f"tda_tile missing required keys: {missing}. "
            f"Expected keys: {required_keys}"
        )


def _extract_timeline_from_panel(panel: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract a minimal semantic timeline from semantic panel.
    
    This is a helper for integration when we only have the panel, not the full timeline.
    Creates a synthetic timeline that captures the essential drift signals.
    """
    critical_run_ids = panel.get("critical_run_ids", [])
    trend = panel.get("trend", "STABLE")
    node_disappearance_count = panel.get("node_disappearance_count", 0)
    
    # Build minimal timeline entry
    timeline_entry = {
        "run_id": "current",
        "term_count": 0,  # Not available from panel
        "critical_signal_count": len(critical_run_ids),
        "status": "CRITICAL" if critical_run_ids else "OK",
    }
    
    # Build node disappearance events if count > 0
    node_disappearance_events = []
    if node_disappearance_count > 0:
        # We don't have term names from panel, so create generic markers
        for i in range(node_disappearance_count):
            node_disappearance_events.append({
                "run_id": "current",
                "term": f"term_disappeared_{i}",
            })
    
    return {
        "schema_version": "semantic-drift-timeline-1.0.0",
        "timeline": [timeline_entry],
        "runs_with_critical_signals": list(critical_run_ids),
        "node_disappearance_events": node_disappearance_events,
        "trend": trend,
    }


__all__ = [
    "build_semantic_tda_tile_for_global_health",
]

