"""
PRNG Governance Adapter for Global Health Surface.

Provides PRNG drift radar integration into the global health console.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rfl.prng.governance import (
    DriftStatus,
    build_prng_drift_radar,
    build_prng_governance_tile,
)


class StatusLight:
    """Status light values for global health tiles."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


PRNG_TILE_SCHEMA_VERSION = "1.0.0"


def build_prng_tile_for_global_health(
    history: Dict[str, Any],
    radar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build PRNG governance tile for global health surface.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The PRNG tile does NOT influence any other tiles
    - No control flow depends on the PRNG tile contents
    - The PRNG tile is purely observational

    Args:
        history: PRNG governance history from build_prng_governance_history().
        radar: Optional drift radar from build_prng_drift_radar().
               If None, will be computed from history.

    Returns:
        PRNG governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - drift_status: "STABLE" | "DRIFTING" | "VOLATILE"
        - status: "OK" | "WARN" | "BLOCK"
        - frequent_violations: Dict[str, int]
        - blocking_rules: List[str]
        - headline: str (neutral summary)

    Status Light Mapping:
        - VOLATILE → RED
        - DRIFTING → YELLOW
        - STABLE → GREEN
    """
    if radar is None:
        radar = build_prng_drift_radar(history)

    tile = build_prng_governance_tile(history, radar=radar)

    # Map drift_status to status_light
    drift_status = radar.get("drift_status", DriftStatus.STABLE.value)
    if drift_status == DriftStatus.VOLATILE.value:
        status_light = StatusLight.RED
    elif drift_status == DriftStatus.DRIFTING.value:
        status_light = StatusLight.YELLOW
    else:  # STABLE
        status_light = StatusLight.GREEN

    return {
        "schema_version": PRNG_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "drift_status": drift_status,
        "status": tile.get("status", "OK"),
        "frequent_violations": radar.get("frequent_violations", {}),
        "blocking_rules": tile.get("blocking_rules", []),
        "headline": tile.get("headline", "PRNG governance: compliant"),
    }


__all__ = [
    "PRNG_TILE_SCHEMA_VERSION",
    "StatusLight",
    "build_prng_tile_for_global_health",
]
