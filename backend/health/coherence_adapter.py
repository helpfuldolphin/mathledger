"""Coherence governance adapter for global health.

STATUS: PHASE X — COHERENCE GOVERNANCE LAYER

Provides integration between confusability-topology coherence signals and
the global health surface builder.

GOVERNANCE CONTRACT:
- All functions are read-only and side-effect free
- The coherence_health tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- No governance writes (read-only monitoring)
"""

from typing import Any, Dict, Optional

COHERENCE_TILE_SCHEMA_VERSION = "1.0.0"


def extract_coherence_drift_signal(
    coherence_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract coherence drift signal for timeline integration.
    
    This will later feed the P4 real-runner integration stream.
    
    Args:
        coherence_map: Output from build_confusability_topology_coherence_map
        
    Returns:
        {
            "coherence_band": "COHERENT" | "PARTIAL" | "MISALIGNED",
            "low_slices": List[str],  # Slices below threshold
            "global_index": float,
        }
    """
    coherence_band = coherence_map.get("coherence_band", "PARTIAL")
    global_index = coherence_map.get("global_coherence_index", 0.5)
    slice_scores = coherence_map.get("slice_coherence_scores", {})
    
    # Identify low slices (below PARTIAL threshold)
    threshold = 0.45
    low_slices = [
        slice_name for slice_name, score in slice_scores.items()
        if score < threshold
    ]
    low_slices = sorted(low_slices)
    
    return {
        "coherence_band": coherence_band,
        "low_slices": low_slices,
        "global_index": global_index,
    }


def build_coherence_governance_tile(
    coherence_map: Dict[str, Any],
    drift_horizon: Optional[Dict[str, Any]] = None,
    console_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build coherence governance tile for global health surface.
    
    STATUS: PHASE X — COHERENCE GOVERNANCE LAYER
    
    Combines coherence map, drift horizon predictor, and console tile
    into a unified governance tile for the global health dashboard.
    
    GOVERNANCE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    - No governance writes
    
    Args:
        coherence_map: Output from build_confusability_topology_coherence_map.
            Must contain: slice_coherence_scores, global_coherence_index,
            coherence_band, root_incoherence_causes
        drift_horizon: Optional output from build_confusability_drift_horizon_predictor.
            If None, horizon fields will be None.
        console_tile: Optional output from build_global_coherence_console_tile.
            If None, will extract from coherence_map.
    
    Returns:
        Coherence governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - coherence_band: "COHERENT" | "PARTIAL" | "MISALIGNED"
        - global_coherence_index: float
        - slices_at_risk: List[str]
        - drivers: List[str]  # Root incoherence causes
        - horizon_estimate: Optional[int]
        - headline: str
    """
    # Extract from coherence_map
    coherence_band = coherence_map.get("coherence_band", "PARTIAL")
    global_index = coherence_map.get("global_coherence_index", 0.5)
    slice_scores = coherence_map.get("slice_coherence_scores", {})
    root_causes = coherence_map.get("root_incoherence_causes", [])
    
    # Determine status light from coherence band
    if coherence_band == "COHERENT":
        status_light = "GREEN"
    elif coherence_band == "PARTIAL":
        status_light = "YELLOW"
    else:  # MISALIGNED
        status_light = "RED"
    
    # Extract slices at risk (low coherence)
    threshold = 0.45
    slices_at_risk = [
        slice_name for slice_name, score in slice_scores.items()
        if score < threshold
    ]
    slices_at_risk = sorted(slices_at_risk)
    
    # Extract drivers from root causes
    drivers = root_causes[:5]  # Top 5 causes
    
    # Extract horizon estimate from drift_horizon
    horizon_estimate = None
    if drift_horizon is not None:
        horizon_estimate = drift_horizon.get("horizon_estimate")
    
    # Build headline
    if console_tile is not None:
        headline = console_tile.get("headline", "")
    else:
        # Build headline from coherence_map
        headline_parts = []
        headline_parts.append(f"Coherence status: {status_light} ({coherence_band}).")
        headline_parts.append(f"Global coherence index: {global_index:.3f}.")
        
        if len(slices_at_risk) > 0:
            headline_parts.append(
                f"{len(slices_at_risk)} slice{'s' if len(slices_at_risk) != 1 else ''} "
                f"below coherence threshold."
            )
        else:
            headline_parts.append("All slices above coherence threshold.")
        
        headline = " ".join(headline_parts)
    
    # Build tile
    tile = {
        "schema_version": COHERENCE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "coherence_band": coherence_band,
        "global_coherence_index": global_index,
        "slices_at_risk": slices_at_risk,
        "drivers": drivers,
        "horizon_estimate": horizon_estimate,
        "headline": headline,
    }
    
    return tile


def build_coherence_tile_for_global_health(
    coherence_map: Dict[str, Any],
    drift_horizon: Optional[Dict[str, Any]] = None,
    console_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build coherence tile for global health surface (alias for build_coherence_governance_tile).
    
    This is the main entry point for global health integration.
    
    Args:
        coherence_map: Output from build_confusability_topology_coherence_map
        drift_horizon: Optional output from build_confusability_drift_horizon_predictor
        console_tile: Optional output from build_global_coherence_console_tile
    
    Returns:
        Coherence governance tile dictionary
    """
    return build_coherence_governance_tile(
        coherence_map=coherence_map,
        drift_horizon=drift_horizon,
        console_tile=console_tile,
    )


__all__ = [
    "COHERENCE_TILE_SCHEMA_VERSION",
    "extract_coherence_drift_signal",
    "build_coherence_governance_tile",
    "build_coherence_tile_for_global_health",
]

