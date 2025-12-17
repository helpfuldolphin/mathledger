"""
Global health tile helpers.

Provides a thin wrapper for aggregating PQ policy guard verdicts into the
global health dictionary consumed by the console.  Each tile is independent so
callers can incrementally adopt PQ telemetry without restructuring the entire
health payload.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from substrate.crypto.pq_policy_guard import summarize_pq_policy_for_global_health


def build_pq_policy_tile(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the pq_policy tile from raw guard verdicts.
    """
    summary = summarize_pq_policy_for_global_health(results)
    tile = {
        "status": summary["status"],
        "violation_count": summary["violations"],
        "latest_epoch": summary["current_epoch"],
        "latest_block": summary["latest_block"],
        "violation_codes": summary["violation_codes"],
    }
    tile["headline"] = _pq_policy_headline(tile)
    return tile


def attach_pq_policy_tile(
    global_health: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Attach the pq_policy tile to the provided global health dictionary.
    """
    updated = dict(global_health)
    updated["pq_policy"] = build_pq_policy_tile(results)
    return updated


def _pq_policy_headline(tile: Mapping[str, Any]) -> str:
    if tile["status"] == "pass":
        return "PQ policy clean"
    count = tile["violation_count"]
    return f"PQ policy violations detected ({count})"


__all__ = [
    "attach_pq_policy_tile",
    "build_pq_policy_tile",
]
