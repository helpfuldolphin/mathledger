"""TDA health adapter module.

Provides TDA-specific health tile adapters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


TDA_HEALTH_TILE_SCHEMA_VERSION = "1.0.0"


@dataclass
class TDAHealthTile:
    """TDA health tile for dashboard."""
    status: str = "ok"
    coverage_pct: float = 0.0
    alignment_status: str = "ALIGNED"
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_tda_health_tile(
    tda_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> TDAHealthTile:
    """Build TDA health tile from TDA data."""
    config = config or {}

    coverage = tda_data.get("coverage_pct", 0.0)
    alignment = tda_data.get("alignment_status", "ALIGNED")

    status = "ok"
    if coverage < 50.0:
        status = "warn"
    if alignment not in ("ALIGNED", "SPARSE"):
        status = "error"

    return TDAHealthTile(
        status=status,
        coverage_pct=coverage,
        alignment_status=alignment,
        metadata={"schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION},
    )


def adapt_tda_for_global_health(
    tda_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Adapt TDA data for global health display."""
    tile = build_tda_health_tile(tda_data)
    return {
        "status": tile.status,
        "coverage_pct": tile.coverage_pct,
        "alignment_status": tile.alignment_status,
    }


__all__ = [
    "TDA_HEALTH_TILE_SCHEMA_VERSION",
    "TDAHealthTile",
    "build_tda_health_tile",
    "adapt_tda_for_global_health",
]
