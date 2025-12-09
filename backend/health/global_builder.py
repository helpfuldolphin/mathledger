"""
Global Health Surface Builder — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Global Health Assembly
==================================================

This module assembles the canonical global_health.json by merging tiles from:
1. FM canonicalization (scripts/fm_canonicalize.py)
2. TDA governance (backend/health/tda_adapter.py)
3. Replay safety (experiments/u2/runner.py)
4. Learning health (analysis/conjecture_engine_contract.py)

The builder enforces:
- Status aggregation (BLOCK > WARN > OK)
- Deterministic output
- Schema validation via global_schema.py

Usage:
    from backend.health.global_builder import (
        build_global_health_surface,
        GlobalHealthSurface,
    )

    surface = build_global_health_surface(
        fm_health={"fm_ok": True, "status": "OK", ...},
        tda_snapshot={"tda_status": "OK", ...},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .global_schema import (
    GLOBAL_HEALTH_SCHEMA_VERSION,
    validate_global_health,
    SchemaValidationError,
)
from .tda_adapter import summarize_tda_for_global_health

# Schema version for builder output
GLOBAL_BUILDER_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# GlobalHealthSurface Dataclass
# ============================================================================

@dataclass
class GlobalHealthSurface:
    """
    Assembled global health surface with all tiles.

    This is the canonical representation of system-wide health,
    combining FM health, TDA governance, replay safety, and learning health.

    Attributes:
        schema_version: Schema version for this surface.
        generated_at: ISO8601 UTC timestamp of generation.
        status: Aggregate status (OK, WARN, BLOCK).
        fm_ok: FM health flag.
        fm_coverage_pct: FM coverage percentage [0, 100].
        tda: Optional TDA governance tile.
        replay: Optional replay safety tile.
        learning: Optional learning health tile.
    """
    schema_version: str
    generated_at: str
    status: str  # "OK" | "WARN" | "BLOCK"
    fm_ok: bool
    fm_coverage_pct: float
    alignment_status: str
    external_only_labels: int

    # Optional tiles
    tda: Optional[Dict[str, Any]] = None
    replay: Optional[Dict[str, Any]] = None
    learning: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "status": self.status,
            "fm_ok": self.fm_ok,
            "coverage_pct": round(self.fm_coverage_pct, 4),
            "alignment_status": self.alignment_status,
            "external_only_labels": self.external_only_labels,
        }

        if self.tda is not None:
            result["tda"] = self.tda

        if self.replay is not None:
            result["replay"] = self.replay

        if self.learning is not None:
            result["learning"] = self.learning

        return result


# ============================================================================
# Status Aggregation Logic
# ============================================================================

def _compute_aggregate_status(
    fm_status: str,
    tda_status: Optional[str],
    replay_ok: Optional[bool],
    learning_status: Optional[str],
) -> str:
    """
    Compute aggregate status from component statuses.

    Rules:
    - BLOCK if any component is BLOCK/ALERT
    - WARN if any component is WARN/ATTENTION
    - OK only if all components are OK

    Args:
        fm_status: FM health status (OK, WARN, BLOCK).
        tda_status: TDA health status (OK, ATTENTION, ALERT) or None.
        replay_ok: Replay safety OK flag or None.
        learning_status: Learning health status or None.

    Returns:
        Aggregate status: "OK", "WARN", or "BLOCK".
    """
    # Map TDA status to global status
    tda_mapped = None
    if tda_status is not None:
        tda_mapped = {
            "OK": "OK",
            "ATTENTION": "WARN",
            "ALERT": "BLOCK",
        }.get(tda_status.upper(), "WARN")

    # Map replay status
    replay_mapped = None
    if replay_ok is not None:
        replay_mapped = "OK" if replay_ok else "BLOCK"

    # Map learning status
    learning_mapped = None
    if learning_status is not None:
        learning_mapped = {
            "HEALTHY": "OK",
            "DEGRADED": "WARN",
            "CRITICAL": "BLOCK",
        }.get(learning_status.upper(), "WARN")

    # Aggregate: most severe wins
    statuses = [fm_status]
    if tda_mapped:
        statuses.append(tda_mapped)
    if replay_mapped:
        statuses.append(replay_mapped)
    if learning_mapped:
        statuses.append(learning_mapped)

    if "BLOCK" in statuses:
        return "BLOCK"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


# ============================================================================
# Tile Extraction Functions
# ============================================================================

def _extract_tda_tile(tda_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract TDA tile from governance snapshot.

    Uses summarize_tda_for_global_health() from tda_adapter.py.
    """
    return summarize_tda_for_global_health(tda_snapshot)


def _extract_replay_tile(replay_result: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract replay safety tile from replay result.

    Args:
        replay_result: Replay verification result dict.

    Returns:
        Replay safety tile with:
        - replay_safety_ok: Boolean
        - status: String status
        - confidence_score: Float
    """
    return {
        "replay_safety_ok": bool(replay_result.get("governance_admissible", False)),
        "status": str(replay_result.get("status", "UNKNOWN")),
        "confidence_score": float(replay_result.get("confidence_score", 0.0)),
    }


def _extract_learning_tile(learning_report: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract learning health tile from conjecture engine report.

    Args:
        learning_report: Conjecture engine analysis result.

    Returns:
        Learning health tile with:
        - status: Health status string
        - supports: Count of supporting evidence
        - contradicts: Count of contradicting evidence
        - inconclusive: Count of inconclusive evidence
    """
    metrics = learning_report.get("metrics", {})
    return {
        "status": str(learning_report.get("status", "UNKNOWN")),
        "supports": int(metrics.get("supports", 0)),
        "contradicts": int(metrics.get("contradicts", 0)),
        "inconclusive": int(metrics.get("inconclusive", 0)),
    }


# ============================================================================
# Main Builder Function
# ============================================================================

def build_global_health_surface(
    fm_health: Mapping[str, Any],
    tda_snapshot: Optional[Mapping[str, Any]] = None,
    replay_result: Optional[Mapping[str, Any]] = None,
    learning_report: Optional[Mapping[str, Any]] = None,
) -> GlobalHealthSurface:
    """
    Build the global health surface from component tiles.

    Args:
        fm_health: Core FM health dict. Required fields:
            - fm_ok: bool
            - coverage_pct: float
            - status: str (OK, WARN, BLOCK)
            - alignment_status: str
            - external_only_labels: int
        tda_snapshot: Optional TDA governance console snapshot.
        replay_result: Optional replay safety verification result.
        learning_report: Optional conjecture engine report.

    Returns:
        GlobalHealthSurface with all tiles merged.

    Raises:
        SchemaValidationError: If fm_health fails validation.
    """
    # Validate and normalize FM health
    validated_fm = validate_global_health(fm_health)

    # Build TDA tile
    tda_tile = None
    tda_status = None
    if tda_snapshot is not None:
        tda_tile = _extract_tda_tile(tda_snapshot)
        tda_status = tda_tile.get("tda_status")

    # Build replay safety tile
    replay_tile = None
    replay_ok = None
    if replay_result is not None:
        replay_tile = _extract_replay_tile(replay_result)
        replay_ok = replay_tile.get("replay_safety_ok")

    # Build learning health tile
    learning_tile = None
    learning_status = None
    if learning_report is not None:
        learning_tile = _extract_learning_tile(learning_report)
        learning_status = learning_tile.get("status")

    # Compute aggregate status
    aggregate_status = _compute_aggregate_status(
        fm_status=validated_fm["status"],
        tda_status=tda_status,
        replay_ok=replay_ok,
        learning_status=learning_status,
    )

    return GlobalHealthSurface(
        schema_version=GLOBAL_HEALTH_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=aggregate_status,
        fm_ok=validated_fm["fm_ok"],
        fm_coverage_pct=validated_fm["coverage_pct"],
        alignment_status=validated_fm["alignment_status"],
        external_only_labels=validated_fm["external_only_labels"],
        tda=tda_tile,
        replay=replay_tile,
        learning=learning_tile,
    )


__all__ = [
    "GlobalHealthSurface",
    "build_global_health_surface",
    "GLOBAL_BUILDER_SCHEMA_VERSION",
]
