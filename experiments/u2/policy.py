"""
PHASE II — NOT USED IN PHASE I

Planner-Oriented Policy Utilities.

Currently exposes a single global-health summarizer that converts raw
LeanFailureSignal telemetry into a stable JSON tile for governance and
console dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from backend.lean_interface import LeanFailureSignal

_SUMMARY_SCHEMA_VERSION = "1.0.0"
_KIND_ORDER = ("timeout", "type_error", "tactic_failure", "unknown")


def _clamp_rate(value: float) -> float:
    """Round rates to four decimal places for stable JSON."""
    return round(value, 4)


def summarize_lean_failures_for_global_health(
    signals: Sequence[LeanFailureSignal],
) -> Dict[str, Any]:
    """
    Summarize Lean failure telemetry for governance/global console.

    Returns a small, stable JSON object with:
        {
            "schema_version": "1.0.0",
            "status": "OK"|"WARN"|"BLOCK",
            "total_events": int,
            "counts": {"timeout": n, "type_error": n, ...},
            "rates": {"timeout": float, ...},   # normalized to [0, 1]
            "avg_duration_ms": float,
            "alerts": [str, ...]                # optional machine-readable notes
        }
    """
    counts = {kind: 0 for kind in _KIND_ORDER}
    total = 0
    total_elapsed = 0

    for signal in signals:
        kind = signal.kind if signal.kind in counts else "unknown"
        counts[kind] += 1
        total += 1
        total_elapsed += max(0, signal.elapsed_ms)

    rates = {
        kind: _clamp_rate(counts[kind] / total) if total else 0.0
        for kind in _KIND_ORDER
    }

    status = "OK"
    alerts: list[str] = []

    if total:
        timeout_rate = rates["timeout"]
        type_rate = rates["type_error"]
        tactic_rate = rates["tactic_failure"]

        if timeout_rate >= 0.4 or type_rate >= 0.3:
            status = "BLOCK"
            if timeout_rate >= 0.4:
                alerts.append("excessive_timeouts")
            if type_rate >= 0.3:
                alerts.append("type_error_spike")
        elif timeout_rate >= 0.2 or type_rate >= 0.15 or tactic_rate >= 0.25:
            status = "WARN"
            if timeout_rate >= 0.2:
                alerts.append("timeout_trend")
            if type_rate >= 0.15:
                alerts.append("type_error_trend")
            if tactic_rate >= 0.25:
                alerts.append("tactic_failure_trend")

    avg_duration = round(total_elapsed / total, 2) if total else 0.0

    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "status": status,
        "total_events": total,
        "counts": counts,
        "rates": rates,
        "avg_duration_ms": avg_duration,
        "alerts": alerts,
    }


__all__ = ["summarize_lean_failures_for_global_health"]
