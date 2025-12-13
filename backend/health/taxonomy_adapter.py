"""Taxonomy integrity adapter for global health surface.

Provides taxonomy tile attachment for global health surface.
SHADOW MODE: all analysis is observational only.
"""

from typing import Dict, Any, Optional

from scripts.taxonomy_governance import (
    build_global_console_tile,
    build_taxonomy_integrity_radar,
)


def build_taxonomy_tile_for_global_health(
    radar: Optional[Dict[str, Any]] = None,
    tile: Optional[Dict[str, Any]] = None,
    metrics_impact: Optional[Dict[str, Any]] = None,
    docs_alignment: Optional[Dict[str, Any]] = None,
    curriculum_alignment: Optional[Dict[str, Any]] = None,
    risk_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build taxonomy tile for global health surface.
    
    SHADOW MODE CONTRACT:
    - The taxonomy tile is purely observational
    - It does NOT influence any other tiles or system health classification
    - No control flow depends on this tile
    - Tile is only attached when radar/tile data is available
    
    Args:
        radar: Optional pre-computed integrity radar
        tile: Optional pre-computed console tile
        metrics_impact: Optional metrics impact data (used to build radar if radar not provided)
        docs_alignment: Optional docs alignment data (used to build radar if radar not provided)
        curriculum_alignment: Optional curriculum alignment data (used to build radar if radar not provided)
        risk_analysis: Optional risk analysis data (used to build tile if tile not provided)
        
    Returns:
        Taxonomy tile dict with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - alignment_score: float (0.0-1.0)
        - critical_breaks_count: int
        - headline: str
        - docs_impacted: List[str] (from radar)
        Or None if insufficient data
    """
    # If tile is provided, use it directly
    if tile is not None:
        return {
            "status_light": tile.get("status_light", "GREEN"),
            "alignment_score": radar.get("alignment_score", 1.0) if radar else 1.0,
            "critical_breaks_count": tile.get("critical_breaks_count", 0),
            "headline": tile.get("headline", "Taxonomy integrity: status unknown"),
            "docs_impacted": radar.get("docs_impacted", []) if radar else [],
        }
    
    # If radar is provided, build tile from it
    if radar is not None:
        risk_analysis = risk_analysis or {"risk_level": "LOW", "breaking_changes": []}
        tile = build_global_console_tile(radar, risk_analysis)
        return {
            "status_light": tile.get("status_light", "GREEN"),
            "alignment_score": radar.get("alignment_score", 1.0),
            "critical_breaks_count": tile.get("critical_breaks_count", 0),
            "headline": tile.get("headline", "Taxonomy integrity: status unknown"),
            "docs_impacted": radar.get("docs_impacted", []),
        }
    
    # If we have component data, build radar and tile
    if metrics_impact is not None and docs_alignment is not None and curriculum_alignment is not None:
        try:
            radar = build_taxonomy_integrity_radar(
                metrics_impact,
                docs_alignment,
                curriculum_alignment,
            )
            risk_analysis = risk_analysis or {"risk_level": "LOW", "breaking_changes": []}
            tile = build_global_console_tile(radar, risk_analysis)
            return {
                "status_light": tile.get("status_light", "GREEN"),
                "alignment_score": radar.get("alignment_score", 1.0),
                "critical_breaks_count": tile.get("critical_breaks_count", 0),
                "headline": tile.get("headline", "Taxonomy integrity: status unknown"),
                "docs_impacted": radar.get("docs_impacted", []),
            }
        except Exception:
            # Graceful degradation: return None if building fails
            return None
    
    # Insufficient data
    return None

