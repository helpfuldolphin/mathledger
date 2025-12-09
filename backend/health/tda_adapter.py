"""
TDA Global Health Adapter — Phase VI Global Health Coupler

Operation CORTEX: Phase VI Auto-Watchdog & Global Health Coupler
=================================================================

This module adapts TDAGovernanceSnapshot to a first-class tile in
global_health.json, enabling unified system health monitoring.

Design Principles:
- Pure adapter: no mutation of upstream snapshots
- Deterministic: same inputs always produce same outputs
- Neutral language: no "good/bad", only structure and numbers

Usage:
    from backend.health.tda_adapter import summarize_tda_for_global_health

    snapshot = build_governance_console_snapshot(...)
    tile = summarize_tda_for_global_health(snapshot.to_dict())
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

TDA_HEALTH_TILE_SCHEMA_VERSION = "1.0.0"

# Status values for TDA health tile
TDA_STATUS_OK = "OK"
TDA_STATUS_ATTENTION = "ATTENTION"
TDA_STATUS_ALERT = "ALERT"

# HSS trend values (normalized to uppercase for global health)
HSS_TREND_IMPROVING = "IMPROVING"
HSS_TREND_STABLE = "STABLE"
HSS_TREND_DEGRADING = "DEGRADING"
HSS_TREND_UNKNOWN = "UNKNOWN"

# Governance signal mapping to global health vocabulary
GOVERNANCE_SIGNAL_MAP = {
    "HEALTHY": "OK",
    "DEGRADED": "WARN",
    "CRITICAL": "BLOCK",
}


def _normalize_hss_trend(trend: Optional[str]) -> str:
    """
    Normalize HSS trend to uppercase global health vocabulary.

    Args:
        trend: HSS trend from snapshot (lowercase or mixed case).

    Returns:
        Normalized trend: IMPROVING | STABLE | DEGRADING | UNKNOWN
    """
    if trend is None:
        return HSS_TREND_UNKNOWN

    trend_upper = trend.upper()
    if trend_upper in (HSS_TREND_IMPROVING, HSS_TREND_STABLE,
                       HSS_TREND_DEGRADING, HSS_TREND_UNKNOWN):
        return trend_upper

    return HSS_TREND_UNKNOWN


def _normalize_governance_signal(signal: Optional[str]) -> str:
    """
    Normalize governance signal to global health vocabulary.

    Args:
        signal: Governance signal from snapshot.

    Returns:
        Normalized signal: OK | WARN | BLOCK
    """
    if signal is None:
        return "OK"

    return GOVERNANCE_SIGNAL_MAP.get(signal.upper(), "OK")


def _classify_tda_status(
    block_rate: float,
    hss_trend: str,
    governance_signal: str,
    golden_alignment: Optional[str] = None,
) -> str:
    """
    Classify TDA status based on multiple signals.

    Status Rules:
    - ALERT if:
        - governance_signal == "BLOCK", or
        - block_rate >= 0.2 AND hss_trend == "DEGRADING", or
        - golden_alignment == "BROKEN"
    - ATTENTION if:
        - any block_rate > 0 but not ALERT, or
        - hss_trend == "DEGRADING" with low block_rate, or
        - golden_alignment == "DRIFTING"
    - OK otherwise

    Args:
        block_rate: Block rate from snapshot (0.0-1.0).
        hss_trend: Normalized HSS trend.
        governance_signal: Normalized governance signal.
        golden_alignment: Golden alignment status (optional).

    Returns:
        TDA status: OK | ATTENTION | ALERT
    """
    # ALERT conditions
    if governance_signal == "BLOCK":
        return TDA_STATUS_ALERT

    if block_rate >= 0.2 and hss_trend == HSS_TREND_DEGRADING:
        return TDA_STATUS_ALERT

    if golden_alignment and golden_alignment.upper() == "BROKEN":
        return TDA_STATUS_ALERT

    # ATTENTION conditions
    if block_rate > 0:
        return TDA_STATUS_ATTENTION

    if hss_trend == HSS_TREND_DEGRADING:
        return TDA_STATUS_ATTENTION

    if golden_alignment and golden_alignment.upper() == "DRIFTING":
        return TDA_STATUS_ATTENTION

    if governance_signal == "WARN":
        return TDA_STATUS_ATTENTION

    # Default to OK
    return TDA_STATUS_OK


def _build_notes(
    block_rate: float,
    hss_trend: str,
    governance_signal: str,
    golden_alignment: Optional[str],
    cycle_count: int,
    exception_windows_active: int,
) -> List[str]:
    """
    Build neutral, structural notes for the TDA health tile.

    Notes are descriptive, not judgmental. No "good/bad" language.

    Args:
        block_rate: Block rate from snapshot.
        hss_trend: Normalized HSS trend.
        governance_signal: Normalized governance signal.
        golden_alignment: Golden alignment status.
        cycle_count: Number of cycles analyzed.
        exception_windows_active: Number of active exception windows.

    Returns:
        List of neutral note strings.
    """
    notes = []

    # Block rate notes
    if block_rate >= 0.2:
        notes.append(f"block_rate={block_rate:.2%} exceeds 20% threshold")
    elif block_rate >= 0.1:
        notes.append(f"block_rate={block_rate:.2%} elevated above 10%")
    elif block_rate > 0:
        notes.append(f"block_rate={block_rate:.2%} non-zero")

    # HSS trend notes
    if hss_trend == HSS_TREND_DEGRADING:
        notes.append(f"hss_trend classified as {hss_trend} over {cycle_count} cycles")
    elif hss_trend == HSS_TREND_IMPROVING:
        notes.append(f"hss_trend classified as {hss_trend} over {cycle_count} cycles")

    # Governance signal notes
    if governance_signal == "BLOCK":
        notes.append("governance_signal indicates BLOCK condition")
    elif governance_signal == "WARN":
        notes.append("governance_signal indicates WARN condition")

    # Golden alignment notes
    if golden_alignment:
        alignment_upper = golden_alignment.upper()
        if alignment_upper == "BROKEN":
            notes.append("golden_alignment status is BROKEN")
        elif alignment_upper == "DRIFTING":
            notes.append("golden_alignment status is DRIFTING")

    # Exception window notes
    if exception_windows_active > 0:
        notes.append(f"{exception_windows_active} exception window(s) currently active")

    # Empty notes case
    if not notes:
        notes.append("TDA metrics within normal operating range")

    return notes


def summarize_tda_for_global_health(
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Adapter from TDAGovernanceSnapshot → global_health tile.

    Transforms a TDA governance snapshot into a standardized health tile
    suitable for inclusion in global_health.json.

    Args:
        snapshot: Output of build_governance_console_snapshot().to_dict()
            Expected keys:
            - block_rate: float (0.0-1.0)
            - mean_hss: float (0.0-1.0)
            - hss_trend: str
            - governance_signal: str
            - golden_alignment: str (optional)
            - cycle_count: int
            - exception_windows_active: int

    Returns:
        Dictionary conforming to TDA health tile schema:
        {
            "schema_version": "1.0.0",
            "tda_status": "OK" | "ATTENTION" | "ALERT",
            "block_rate": float,
            "mean_hss": float | None,
            "hss_trend": "IMPROVING" | "STABLE" | "DEGRADING" | "UNKNOWN",
            "governance_signal": "OK" | "WARN" | "BLOCK",
            "notes": list[str],
        }

    Example:
        >>> snapshot = {
        ...     "block_rate": 0.05,
        ...     "mean_hss": 0.72,
        ...     "hss_trend": "stable",
        ...     "governance_signal": "HEALTHY",
        ...     "cycle_count": 100,
        ... }
        >>> tile = summarize_tda_for_global_health(snapshot)
        >>> assert tile["tda_status"] == "ATTENTION"  # block_rate > 0
    """
    # Extract values with defaults
    block_rate = float(snapshot.get("block_rate", 0.0))
    mean_hss = snapshot.get("mean_hss")
    if mean_hss is not None:
        mean_hss = float(mean_hss)

    raw_hss_trend = snapshot.get("hss_trend")
    hss_trend = _normalize_hss_trend(raw_hss_trend)

    raw_governance_signal = snapshot.get("governance_signal")
    governance_signal = _normalize_governance_signal(raw_governance_signal)

    golden_alignment = snapshot.get("golden_alignment")
    cycle_count = int(snapshot.get("cycle_count", 0))
    exception_windows_active = int(snapshot.get("exception_windows_active", 0))

    # Classify TDA status
    tda_status = _classify_tda_status(
        block_rate=block_rate,
        hss_trend=hss_trend,
        governance_signal=governance_signal,
        golden_alignment=golden_alignment,
    )

    # Build notes
    notes = _build_notes(
        block_rate=block_rate,
        hss_trend=hss_trend,
        governance_signal=governance_signal,
        golden_alignment=golden_alignment,
        cycle_count=cycle_count,
        exception_windows_active=exception_windows_active,
    )

    return {
        "schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION,
        "tda_status": tda_status,
        "block_rate": round(block_rate, 4),
        "mean_hss": round(mean_hss, 4) if mean_hss is not None else None,
        "hss_trend": hss_trend,
        "governance_signal": governance_signal,
        "notes": notes,
    }


def validate_tda_health_tile(tile: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate a TDA health tile conforms to schema.

    Args:
        tile: TDA health tile dictionary.

    Returns:
        Validated tile dictionary.

    Raises:
        ValueError: If tile fails validation.
    """
    required_fields = ["schema_version", "tda_status", "block_rate", "hss_trend",
                       "governance_signal", "notes"]

    for field in required_fields:
        if field not in tile:
            raise ValueError(f"Missing required field: {field}")

    if tile["schema_version"] != TDA_HEALTH_TILE_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: expected {TDA_HEALTH_TILE_SCHEMA_VERSION}, "
            f"got {tile['schema_version']}"
        )

    if tile["tda_status"] not in (TDA_STATUS_OK, TDA_STATUS_ATTENTION, TDA_STATUS_ALERT):
        raise ValueError(f"Invalid tda_status: {tile['tda_status']}")

    if tile["hss_trend"] not in (HSS_TREND_IMPROVING, HSS_TREND_STABLE,
                                  HSS_TREND_DEGRADING, HSS_TREND_UNKNOWN):
        raise ValueError(f"Invalid hss_trend: {tile['hss_trend']}")

    if tile["governance_signal"] not in ("OK", "WARN", "BLOCK"):
        raise ValueError(f"Invalid governance_signal: {tile['governance_signal']}")

    if not isinstance(tile["notes"], list):
        raise ValueError("notes must be a list")

    return dict(tile)


# ============================================================================
# Global Health Integration
# ============================================================================

def extend_global_health_with_tda(
    global_health: Dict[str, Any],
    tda_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Extend global_health.json with TDA health tile.

    This function adds a "tda" section to the global health payload,
    populated by the TDA adapter.

    Args:
        global_health: Existing global_health dictionary.
        tda_snapshot: TDA governance snapshot from build_governance_console_snapshot().

    Returns:
        Extended global_health dictionary with "tda" section.
    """
    tda_tile = summarize_tda_for_global_health(tda_snapshot)

    # Create copy to avoid mutation
    extended = dict(global_health)
    extended["tda"] = tda_tile

    return extended


__all__ = [
    "TDA_HEALTH_TILE_SCHEMA_VERSION",
    "TDA_STATUS_OK",
    "TDA_STATUS_ATTENTION",
    "TDA_STATUS_ALERT",
    "HSS_TREND_IMPROVING",
    "HSS_TREND_STABLE",
    "HSS_TREND_DEGRADING",
    "HSS_TREND_UNKNOWN",
    "summarize_tda_for_global_health",
    "validate_tda_health_tile",
    "extend_global_health_with_tda",
]
