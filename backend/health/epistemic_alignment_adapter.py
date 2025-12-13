"""Epistemic alignment integration adapter for global health.

STATUS: PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

Provides integration between epistemic alignment tensor, misalignment forecast,
and director panel for the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The epistemic_alignment tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
- Tile is only attached when all three inputs are present
"""

from typing import Any, Dict

EPISTEMIC_ALIGNMENT_TILE_SCHEMA_VERSION = "1.0.0"


def build_epistemic_governance_tile(
    alignment_tensor: Dict[str, Any],
    misalignment_forecast: Dict[str, Any],
    director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Governance tile summarizing epistemic alignment and misalignment risk.

    SHADOW MODE: Observational only. No control paths, no gating, no aborts.

    Args:
        alignment_tensor: Output from build_epistemic_alignment_tensor()
        misalignment_forecast: Output from forecast_epistemic_misalignment()
        director_panel: Output from build_epistemic_director_panel()

    Returns:
        Epistemic alignment governance tile dictionary with:
        - status_light: from director panel (GREEN/YELLOW/RED)
        - alignment_band: from alignment tensor norm (HIGH/MEDIUM/LOW)
        - forecast_band: from forecast predicted_band
        - tensor_norm: from alignment_tensor["alignment_tensor_norm"]
        - misalignment_hotspots: from alignment_tensor["misalignment_hotspots"]
        - headline: from director panel
        - flags: from director panel
        - schema_version: "1.0.0"

    All outputs are JSON-safe and deterministic.
    """
    # Extract status_light from director panel
    status_light = director_panel.get("status_light", "YELLOW")

    # Extract alignment_band from alignment tensor norm
    tensor_norm = alignment_tensor.get("alignment_tensor_norm", 0.5)
    if tensor_norm >= 0.7:
        alignment_band = "HIGH"
    elif tensor_norm >= 0.4:
        alignment_band = "MEDIUM"
    else:
        alignment_band = "LOW"

    # Extract forecast_band from misalignment forecast
    forecast_band = misalignment_forecast.get("predicted_band", "MEDIUM")

    # Extract tensor_norm from alignment tensor
    tensor_norm_value = alignment_tensor.get("alignment_tensor_norm", 0.0)

    # Extract misalignment_hotspots from alignment tensor
    misalignment_hotspots = alignment_tensor.get("misalignment_hotspots", [])

    # Extract headline from director panel
    headline = director_panel.get("headline", "Epistemic alignment status available.")

    # Extract flags from director panel
    flags = director_panel.get("flags", [])

    # Build tile
    tile: Dict[str, Any] = {
        "schema_version": EPISTEMIC_ALIGNMENT_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "alignment_band": alignment_band,
        "forecast_band": forecast_band,
        "tensor_norm": tensor_norm_value,
        "misalignment_hotspots": misalignment_hotspots,
        "headline": headline,
        "flags": flags,
    }

    return tile


def build_epistemic_alignment_tile_for_global_health(
    alignment_tensor: Dict[str, Any],
    misalignment_forecast: Dict[str, Any],
    director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build epistemic alignment tile for global health surface (alias for build_epistemic_governance_tile).

    This is the main entry point for global health integration.

    Args:
        alignment_tensor: Output from build_epistemic_alignment_tensor()
        misalignment_forecast: Output from forecast_epistemic_misalignment()
        director_panel: Output from build_epistemic_director_panel()

    Returns:
        Epistemic alignment governance tile dictionary
    """
    return build_epistemic_governance_tile(
        alignment_tensor=alignment_tensor,
        misalignment_forecast=misalignment_forecast,
        director_panel=director_panel,
    )


__all__ = [
    "EPISTEMIC_ALIGNMENT_TILE_SCHEMA_VERSION",
    "build_epistemic_governance_tile",
    "build_epistemic_alignment_tile_for_global_health",
]

