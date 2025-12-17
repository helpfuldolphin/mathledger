"""U2 Dynamics analysis and console adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Core state structures
# ---------------------------------------------------------------------------


@dataclass
class DynamicsState:
    """Aggregated dynamics metrics for a single run."""

    cycle: int = 0
    pressure: float = 0.0
    velocity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicsTile:
    """Legacy tile wrapper retained for compatibility."""

    status: str = "ok"
    cycle: int = 0
    pressure_score: float = 0.0
    velocity_score: float = 0.0
    headline: str = "Dynamics stable"


@dataclass
class DynamicsDebugSnapshot:
    """Debug payload emitted by runners for console inspection."""

    run_id: str
    slice_name: str
    mode: str
    oscillation_index: float
    plateau_score: float
    uplift_delta: float
    pattern_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Runner hook helpers
# ---------------------------------------------------------------------------


def compute_dynamics_state(
    metrics: Sequence[Dict[str, Any]],
    cycle: int = 0,
) -> DynamicsState:
    """Compute a coarse-grained dynamics state from raw metrics."""
    if not metrics:
        return DynamicsState(cycle=cycle)

    pressure = min(1.0, len(metrics) * 0.05)
    velocity = sum(m.get("delta", 0.0) for m in metrics) / max(1, len(metrics))
    return DynamicsState(
        cycle=cycle,
        pressure=pressure,
        velocity=velocity,
        metadata={"metric_count": len(metrics)},
    )


def build_dynamics_tile(state: DynamicsState) -> DynamicsTile:
    """Map a dynamics state to the legacy tile format."""
    if state.pressure < 0.3 and abs(state.velocity) < 0.2:
        status = "ok"
        headline = "Dynamics stable"
    elif state.pressure < 0.6 or abs(state.velocity) < 0.4:
        status = "warn"
        headline = "Dynamics showing drift"
    else:
        status = "critical"
        headline = "Dynamics critical"

    return DynamicsTile(
        status=status,
        cycle=state.cycle,
        pressure_score=state.pressure,
        velocity_score=state.velocity,
        headline=headline,
    )


def build_dynamics_debug_snapshot(
    state: DynamicsState,
    *,
    run_id: str,
    mode: str,
    slice_name: str,
    oscillation_index: float,
    plateau_score: float,
    uplift_delta: float,
    pattern_label: str = "plateau",
    metadata: Optional[Dict[str, Any]] = None,
) -> DynamicsDebugSnapshot:
    """
    Build a debug snapshot suitable for console ingestion.

    Runners should invoke this once per run (baseline, RFL, and negative-control)
    immediately after computing the final abstention/uplift metrics. The
    resulting snapshot should be appended to the run manifest and later attached
    under ``global_health["dynamics"]`` via ``attach_dynamics_tile``. Negative
    control runs must set ``mode`` to ``"baseline"`` or ``"nc"`` so the NC
    auditor can enforce Phase II safeguards.
    """

    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.setdefault("cycle", state.cycle)
    snapshot_metadata.setdefault("pressure", state.pressure)
    snapshot_metadata.setdefault("velocity", state.velocity)
    return DynamicsDebugSnapshot(
        run_id=run_id,
        slice_name=slice_name,
        mode=mode,
        oscillation_index=float(oscillation_index),
        plateau_score=float(plateau_score),
        uplift_delta=float(uplift_delta),
        pattern_label=pattern_label,
        metadata=snapshot_metadata,
    )


# ---------------------------------------------------------------------------
# Negative control and console adapters
# ---------------------------------------------------------------------------


def check_negative_control_dynamics(
    snapshots: Sequence[DynamicsDebugSnapshot],
    max_allowed_oscillation_index: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluate NC snapshots and classify their health.
    """
    status = "OK"
    notes: List[str] = []
    max_osc = 0.0
    max_plateau = 0.0

    def escalate(new_status: str) -> None:
        nonlocal status
        order = {"OK": 0, "DRIFTING": 1, "BROKEN": 2}
        if order[new_status] > order[status]:
            status = new_status

    for snapshot in snapshots:
        max_osc = max(max_osc, snapshot.oscillation_index)
        max_plateau = max(max_plateau, snapshot.plateau_score)
        if snapshot.pattern_label not in {"plateau", "degenerate"}:
            notes.append(
                f"{snapshot.run_id} pattern={snapshot.pattern_label} violates NC expectation"
            )
            escalate("DRIFTING")
        if snapshot.oscillation_index > max_allowed_oscillation_index:
            notes.append(
                f"{snapshot.run_id} oscillation_index="
                f"{snapshot.oscillation_index:.3f} exceeds "
                f"{max_allowed_oscillation_index:.3f}"
            )
            escalate("BROKEN")

    return {
        "schema_version": "1.0.0",
        "run_count": len(snapshots),
        "max_oscillation_index": max_osc,
        "max_plateau_score": max_plateau,
        "status": status,
        "notes": notes,
    }


def _round(value: float) -> float:
    return round(float(value), 6)


def summarize_dynamics_for_global_health(
    metrics: Sequence[Dict[str, Any]],
    cycle: int = 0,
) -> Dict[str, Any]:
    """Backward-compatible helper for legacy callers."""
    state = compute_dynamics_state(metrics, cycle)
    tile = build_dynamics_tile(state)
    return {
        "status": tile.status,
        "cycle": tile.cycle,
        "pressure_score": tile.pressure_score,
        "velocity_score": tile.velocity_score,
        "headline": tile.headline,
    }


def summarize_dynamics_for_global_console(
    snapshots: Sequence[DynamicsDebugSnapshot],
) -> Dict[str, Any]:
    """
    Summarize dynamics snapshots for the global console tile.
    """
    run_count = len(snapshots)
    if run_count:
        mean_osc = _round(mean(s.oscillation_index for s in snapshots))
        max_osc = _round(max(s.oscillation_index for s in snapshots))
    else:
        mean_osc = 0.0
        max_osc = 0.0

    nc_snapshots = [s for s in snapshots if s.mode in {"baseline", "nc"}]
    if nc_snapshots:
        nc_report = check_negative_control_dynamics(nc_snapshots)
        nc_status = nc_report["status"]
        nc_notes = nc_report["notes"]
    else:
        nc_status = "OK"
        nc_notes = []

    headline = (
        f"{run_count} dynamics runs; mean oscillation {mean_osc:.3f}, "
        f"max {max_osc:.3f} ({nc_status})."
    )

    return {
        "schema_version": "1.0.0",
        "run_count": run_count,
        "mean_oscillation_index": mean_osc,
        "max_oscillation_index": max_osc,
        "nc_status": nc_status,
        "nc_notes": nc_notes,
        "headline": headline,
    }


def attach_dynamics_tile(
    global_health: Dict[str, Any],
    snapshots: Sequence[DynamicsDebugSnapshot],
) -> Dict[str, Any]:
    """Attach the console summary to a global health payload."""
    updated = dict(global_health)
    updated["dynamics"] = summarize_dynamics_for_global_console(snapshots)
    return updated


__all__ = [
    "DynamicsDebugSnapshot",
    "DynamicsState",
    "DynamicsTile",
    "attach_dynamics_tile",
    "build_dynamics_debug_snapshot",
    "build_dynamics_tile",
    "check_negative_control_dynamics",
    "compute_dynamics_state",
    "summarize_dynamics_for_global_console",
    "summarize_dynamics_for_global_health",
]
