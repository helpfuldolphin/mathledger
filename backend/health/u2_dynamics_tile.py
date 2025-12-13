"""Self-contained U2 dynamics tile builder."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Status thresholds kept explicit for clarity and future tuning.
GREEN_SUCCESS_RATE = 0.9
GREEN_MAX_DEPTH = 100.0
RED_SUCCESS_RATE = 0.6
RED_MAX_DEPTH = 180.0

# Fixed default window size for P4 U2 dynamics summaries.
DEFAULT_U2_DYNAMICS_WINDOW_SIZE = 50

# U2 dynamics reason codes (GGFL + status warnings).
DRIVER_U2_SIGNAL_MISSING = "DRIVER_U2_SIGNAL_MISSING"
DRIVER_U2_RUNS_ZERO = "DRIVER_U2_RUNS_ZERO"
DRIVER_U2_SUCCESS_RATE_LOW = "DRIVER_U2_SUCCESS_RATE_LOW"
DRIVER_U2_MAX_DEPTH_HIGH = "DRIVER_U2_MAX_DEPTH_HIGH"
DRIVER_U2_STATUS_NON_GREEN = "DRIVER_U2_STATUS_NON_GREEN"

# Allowed extraction_source values for signals.u2_dynamics.
U2_DYNAMICS_EXTRACTION_SOURCES = ("P4_SUMMARY", "MANIFEST", "EVIDENCE_JSON", "MISSING")

# ---------------------------------------------------------------------------
# Canonical semantics (SHADOW-only, observational)
# ---------------------------------------------------------------------------
# u2_dynamics.windows[*].tile (from build_u2_dynamics_tile())
# - schema_version (safety/meta): semver for parsing; unit: n/a.
# - status_light (safety): {"GREEN","YELLOW","RED"} derived from success_rate/max_depth.
# - headline (safety): neutral human-readable summary; non-gating.
# - metrics.mean_depth (state): mean proof/search depth; unit: depth steps; range: >= 0.
# - metrics.max_depth (state): max proof/search depth; unit: depth steps; range: >= 0.
# - metrics.success_rate (outcome): successes/runs; unit: fraction; range: [0.0, 1.0].
# - metrics.runs (state): sample size; unit: cycles; range: int >= 0.
#
# signals.u2_dynamics (from build_u2_dynamics_first_light_status_signal())
# - mode (safety/meta): always "SHADOW".
# - action (safety/meta): always "LOGGED_ONLY".
# - extraction_source (safety/meta): {"P4_SUMMARY","MANIFEST","EVIDENCE_JSON","MISSING"}.
# - status_light (safety): overall traffic-light status for the run.
# - success_rate (outcome): overall success fraction; [0.0, 1.0].
# - max_depth (state): overall max depth across the run; depth steps; int >= 0.
# - window_size (state): cycles/window; int >= 1; default 50.
# - windows[*].window_index (state): 0-based index; int >= 0.
# - windows[*].start_cycle (state): inclusive cycle id; int >= 1.
# - windows[*].end_cycle (state): inclusive cycle id; int >= start_cycle.
# - windows[*].status_light (safety): per-window traffic-light status.
# - windows[*].success_rate (outcome): per-window success fraction; [0.0, 1.0].
# - windows[*].max_depth (state): per-window max depth; depth steps; >= 0.
# - decomposition_summary (safety/meta): cross-window vector reconciliation summary (observational).
#   - state_components.window_count: int >= 0
#   - state_components.mean_depth_mean: float >= 0 or null
#   - state_components.max_depth_mean: float >= 0 or null
#   - state_components.runs_total: int >= 0
#   - outcome_components.success_rate_mean: float in [0,1] or null
#   - outcome_components.success_rate_weighted_mean: float in [0,1] or null
#   - safety_components.status_light_counts: counts by {"GREEN","YELLOW","RED","UNKNOWN"}
#   - safety_components.status_risk_mean: mean risk score (GREEN=0,YELLOW=1,RED=2); float >= 0 or null
# - warning (safety): optional 1-line advisory, format: "window_count=<n> driver=<REASON_CODE>".


def _determine_status(success_rate: float, max_depth: float, runs: int) -> str:
    """Map summary metrics to a traffic-light status."""
    if runs <= 0:
        return "RED"
    if success_rate >= GREEN_SUCCESS_RATE and max_depth <= GREEN_MAX_DEPTH:
        return "GREEN"
    if success_rate < RED_SUCCESS_RATE or max_depth > RED_MAX_DEPTH:
        return "RED"
    return "YELLOW"


def build_u2_dynamics_tile(dynamics_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-safe U2 dynamics tile from summary metrics."""
    mean_depth = max(0.0, float(dynamics_summary.get("mean_depth", 0.0)))
    max_depth = max(0.0, float(dynamics_summary.get("max_depth", 0.0)))
    success_rate = float(dynamics_summary.get("success_rate", 0.0))
    runs = max(0, int(dynamics_summary.get("runs", 0)))

    # Keep success rates within [0, 1] to avoid skewed statuses.
    success_rate = max(0.0, min(1.0, success_rate))
    status_light = _determine_status(success_rate, max_depth, runs)
    headline = (
        f"U2 dynamics: success rate {success_rate:.1%} across {runs} runs "
        f"(mean depth {mean_depth:.1f}, max depth {max_depth:.1f})."
    )

    return {
        "schema_version": "1.0.0",
        "status_light": status_light,
        "headline": headline,
        "metrics": {
            "mean_depth": mean_depth,
            "max_depth": max_depth,
            "success_rate": success_rate,
            "runs": runs,
        },
    }


def attach_u2_dynamics_tile(global_health: Dict[str, Any], dynamics_tile: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the health surface with the U2 dynamics tile attached."""
    updated = dict(global_health or {})
    updated["u2_dynamics"] = dynamics_tile
    return updated


def attach_u2_dynamics_to_p4_summary(
    p4_summary: Dict[str, Any],
    dynamics_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach a compact U2 dynamics summary to the P4 summary (non-mutating)."""
    updated = dict(p4_summary or {})
    metrics = dynamics_tile.get("metrics", {})
    summary = {
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "max_depth": int(metrics.get("max_depth", 0)),
        "status_light": str(dynamics_tile.get("status_light", "RED")),
        "headline": str(dynamics_tile.get("headline", "")),
    }
    if "windows" in dynamics_tile:
        summary["windows"] = dynamics_tile.get("windows", [])
    if "window_size" in dynamics_tile:
        summary["window_size"] = dynamics_tile.get("window_size")
    updated["u2_dynamics"] = summary
    return updated


def attach_u2_dynamics_to_first_light_status(
    status_json: Dict[str, Any],
    dynamics_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach compact U2 dynamics signal to first-light status JSON (non-mutating)."""
    updated = dict(status_json or {})
    signals = dict(updated.get("signals", {}))
    signals["u2_dynamics"] = {
        "success_rate": float(dynamics_signal.get("success_rate", 0.0)),
        "max_depth": int(dynamics_signal.get("max_depth", 0)),
        "status_light": str(dynamics_signal.get("status_light", "RED")),
    }
    updated["signals"] = signals
    return updated


def attach_u2_dynamics_to_evidence(
    evidence: Dict[str, Any],
    dynamics_signal: Dict[str, Any],
    telemetry_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach U2 dynamics governance evidence (non-mutating, SHADOW/advisory only)."""
    updated = dict(evidence or {})
    governance = dict(updated.get("governance", {}))
    success_rate = float(dynamics_signal.get("success_rate", 0.0))
    max_depth = int(dynamics_signal.get("max_depth", 0))
    status_light = str(dynamics_signal.get("status_light", "RED"))
    headline = (
        f"U2 dynamics signal: success rate {success_rate:.1%} with max depth {max_depth}."
    )
    governance["u2_dynamics"] = {
        "success_rate": success_rate,
        "max_depth": max_depth,
        "status_light": status_light,
        "headline": headline,
    }
    if telemetry_tile is not None:
        governance["u2_dynamics"]["u2_telemetry_consistency"] = summarize_u2_dynamics_vs_telemetry(
            dynamics_signal, telemetry_tile
        )
    updated["governance"] = governance
    return updated


def extract_u2_dynamics_signal_for_first_light(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a minimal signal for the first-light path (spec-only, no wiring).

    Returns:
        Dict with success_rate (float), max_depth (int), and status_light (str).
    """
    tile = build_u2_dynamics_tile(stats)
    metrics = tile.get("metrics", {})
    return {
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "max_depth": int(metrics.get("max_depth", 0)),
        "status_light": tile.get("status_light", "RED"),
    }


def build_u2_dynamics_first_light_status_signal(
    dynamics_tile: Dict[str, Any],
    extraction_source: str = "P4_SUMMARY",
) -> Dict[str, Any]:
    """
    Build a compact U2 dynamics status signal for first_light_status.json (SHADOW-only).

    This is an observational stub describing the intended surface:
        signals.u2_dynamics = {
            "mode": "SHADOW",
            "action": "LOGGED_ONLY",
            "extraction_source": "P4_SUMMARY|MANIFEST|EVIDENCE_JSON|MISSING",
            "status_light": <traffic light>,
            "success_rate": <overall success>,
            "max_depth": <overall max depth>,
            "window_size": <cycles per window>,
            "windows": [
                {
                    "window_index": i,
                    "start_cycle": a,
                    "end_cycle": b,
                    "status_light": <window light>,
                    "success_rate": <window success>,
                    "max_depth": <window max depth>,
                },
                ...
            ],
            "decomposition_summary": {"state_components": ..., "outcome_components": ..., "safety_components": ...},
            "warning": "window_count=<n> driver=<REASON_CODE>",  # optional, non-GREEN only
        }

    Args:
        dynamics_tile: U2 dynamics tile (full tile or compact P4 summary form).

    Returns:
        JSON-safe dict suitable for first_light_status.json signals section.
    """
    if not isinstance(dynamics_tile, dict):
        dynamics_tile = {}

    metrics = dynamics_tile.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    windows = dynamics_tile.get("windows")
    windows = windows if isinstance(windows, list) else []

    def _status_label(value: Any) -> str:
        if value is None:
            return "RED"
        return str(value).upper()

    def _extract_numeric(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_success_rate(tile: Dict[str, Any]) -> float:
        raw = tile.get("success_rate")
        if raw is None and isinstance(tile.get("metrics"), dict):
            raw = tile["metrics"].get("success_rate")
        value = _extract_numeric(raw)
        if value is None:
            return 0.0
        return max(0.0, min(1.0, value))

    def _extract_max_depth(tile: Dict[str, Any]) -> int:
        raw = tile.get("max_depth")
        if raw is None and isinstance(tile.get("metrics"), dict):
            raw = tile["metrics"].get("max_depth")
        value = _extract_numeric(raw)
        if value is None:
            return 0
        return max(0, int(value))

    compact_windows: list[Dict[str, Any]] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        tile = window.get("tile")
        tile = tile if isinstance(tile, dict) else window
        compact_windows.append(
            {
                "window_index": window.get("window_index"),
                "start_cycle": window.get("start_cycle"),
                "end_cycle": window.get("end_cycle"),
                "status_light": _status_label(tile.get("status_light") or tile.get("status")),
                "success_rate": _extract_success_rate(tile),
                "max_depth": _extract_max_depth(tile),
            }
        )

    if compact_windows:
        compact_windows.sort(
            key=lambda w: (w.get("window_index") is None, w.get("window_index", 0))
        )

    success_rate = _extract_numeric(metrics.get("success_rate"))
    if success_rate is None:
        success_rate = _extract_numeric(dynamics_tile.get("success_rate"))
    success_rate = max(0.0, min(1.0, float(success_rate or 0.0)))

    max_depth_val = _extract_numeric(metrics.get("max_depth"))
    if max_depth_val is None:
        max_depth_val = _extract_numeric(dynamics_tile.get("max_depth"))
    max_depth = max(0, int(max_depth_val or 0))

    status_light = _status_label(dynamics_tile.get("status_light") or "RED")

    source = str(extraction_source or "MISSING").upper()
    if source not in U2_DYNAMICS_EXTRACTION_SOURCES:
        source = "MISSING"

    signal: Dict[str, Any] = {
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
        "extraction_source": source,
        "status_light": status_light,
        "success_rate": success_rate,
        "max_depth": max_depth,
        "window_size": int(
            dynamics_tile.get("window_size", DEFAULT_U2_DYNAMICS_WINDOW_SIZE)
        ),
    }
    if compact_windows:
        signal["windows"] = compact_windows
        signal["decomposition_summary"] = build_u2_dynamics_decomposition_summary(windows)

        if status_light != "GREEN":
            warning = build_u2_dynamics_warning(windows, status_light, success_rate, max_depth)
            if warning:
                signal["warning"] = warning
    return signal


def build_u2_dynamics_decomposition_summary(windows: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decompose per-window U2 dynamics into state/outcome/safety components (SHADOW-only).

    This is intentionally vector-shaped: it reconciles multiple windows into
    separate component groups without collapsing them into a single verdict.

    Args:
        windows: List of window dicts from build_u2_dynamics_window_metrics()["windows"] or
            the compact u2_dynamics.windows in p4_summary.json.

    Returns:
        JSON-safe dict with state_components, outcome_components, safety_components.
    """
    window_count = 0

    mean_depths: list[float] = []
    max_depths: list[float] = []
    runs: list[int] = []
    success_rates: list[float] = []
    weighted_success_sum = 0.0
    weighted_success_weight = 0

    status_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0, "UNKNOWN": 0}
    risk_scores: list[int] = []

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

    def _status_label(tile: Optional[Dict[str, Any]]) -> str:
        if not isinstance(tile, dict):
            return "UNKNOWN"
        val = tile.get("status_light") or tile.get("status")
        if val is None:
            return "UNKNOWN"
        return str(val).upper()

    def _risk_score(status: str) -> int:
        mapping = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        return mapping.get(status, 2)

    for window in windows:
        if not isinstance(window, dict):
            continue
        tile = window.get("tile")
        tile = tile if isinstance(tile, dict) else window
        metrics = tile.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}

        window_count += 1

        mean_depth = _as_float(metrics.get("mean_depth"))
        if mean_depth is not None:
            mean_depths.append(max(0.0, mean_depth))

        max_depth = _as_float(metrics.get("max_depth"))
        if max_depth is None:
            max_depth = _as_float(tile.get("max_depth"))
        if max_depth is not None:
            max_depths.append(max(0.0, max_depth))

        success_rate = _as_float(metrics.get("success_rate"))
        if success_rate is None:
            success_rate = _as_float(tile.get("success_rate"))
        if success_rate is not None:
            success_rate = max(0.0, min(1.0, float(success_rate)))
            success_rates.append(success_rate)

        run_count = _as_int(metrics.get("runs"))
        if run_count is None:
            run_count = _as_int(tile.get("runs"))
        if run_count is not None and run_count >= 0:
            runs.append(run_count)
            if success_rate is not None:
                weighted_success_sum += success_rate * run_count
                weighted_success_weight += run_count

        status = _status_label(tile)
        if status not in status_counts:
            status = "UNKNOWN"
        status_counts[status] += 1
        risk_scores.append(_risk_score(status))

    def _mean(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def _mean_int(values: list[int]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def _round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), ndigits)

    mean_depth_mean = _round(_mean(mean_depths))
    max_depth_mean = _round(_mean(max_depths))
    runs_total = int(sum(runs)) if runs else 0
    runs_mean = _round(_mean_int(runs))

    success_rate_mean = _round(_mean(success_rates))
    success_rate_weighted_mean = None
    if weighted_success_weight > 0:
        success_rate_weighted_mean = _round(weighted_success_sum / weighted_success_weight)

    status_risk_mean = _round(_mean([float(r) for r in risk_scores])) if risk_scores else None

    return {
        "state_components": {
            "window_count": window_count,
            "mean_depth_mean": mean_depth_mean,
            "max_depth_mean": max_depth_mean,
            "runs_total": runs_total,
            "runs_mean": runs_mean,
        },
        "outcome_components": {
            "window_count": window_count,
            "success_rate_mean": success_rate_mean,
            "success_rate_weighted_mean": success_rate_weighted_mean,
        },
        "safety_components": {
            "window_count": window_count,
            "status_light_counts": status_counts,
            "status_risk_mean": status_risk_mean,
        },
    }


def build_u2_dynamics_warning(
    windows: list[Dict[str, Any]],
    status_light: str,
    success_rate: float,
    max_depth: int,
) -> Optional[str]:
    """
    Emit a single-line u2_dynamics warning (SHADOW-only).

    Hygiene contract:
    - At most one line
    - Includes only: window_count + top driver field
    """
    window_count = len(windows) if isinstance(windows, list) else 0
    status = str(status_light or "RED").upper()

    if status == "GREEN":
        return None

    runs_total = build_u2_dynamics_decomposition_summary(windows)["state_components"].get("runs_total")
    if isinstance(runs_total, int) and runs_total <= 0:
        driver = DRIVER_U2_RUNS_ZERO
    elif status == "RED":
        if success_rate < RED_SUCCESS_RATE:
            driver = DRIVER_U2_SUCCESS_RATE_LOW
        elif max_depth > RED_MAX_DEPTH:
            driver = DRIVER_U2_MAX_DEPTH_HIGH
        else:
            driver = DRIVER_U2_STATUS_NON_GREEN
    else:
        if success_rate < GREEN_SUCCESS_RATE:
            driver = DRIVER_U2_SUCCESS_RATE_LOW
        elif max_depth > GREEN_MAX_DEPTH:
            driver = DRIVER_U2_MAX_DEPTH_HIGH
        else:
            driver = DRIVER_U2_STATUS_NON_GREEN

    warning = f"window_count={window_count} driver={driver}"
    return warning.replace("\n", " ")


def u2_dynamics_for_alignment_view(signal_or_tile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert U2 dynamics signal/tile to GGFL alignment view format (SHADOW-only).

    Fixed output contract:
        {
            "signal_type": "SIG-U2",
            "status": "ok" | "warn",
            "conflict": False,
            "weight_hint": "LOW",
            "drivers": [REASON_CODE, ...],  # <= 3, deterministic order
            "summary": "<one sentence>",
            "shadow_mode_invariants": {...},
        }
    """
    payload = signal_or_tile if isinstance(signal_or_tile, dict) else {}

    status_light = str(payload.get("status_light") or payload.get("status") or "RED").upper()

    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    success_rate_val = metrics.get("success_rate")
    if success_rate_val is None:
        success_rate_val = payload.get("success_rate")
    try:
        success_rate = float(success_rate_val) if success_rate_val is not None else 0.0
    except (TypeError, ValueError):
        success_rate = 0.0
    success_rate = max(0.0, min(1.0, success_rate))

    max_depth_val = metrics.get("max_depth")
    if max_depth_val is None:
        max_depth_val = payload.get("max_depth")
    try:
        max_depth = int(max_depth_val) if max_depth_val is not None else 0
    except (TypeError, ValueError):
        try:
            max_depth = int(float(max_depth_val))
        except (TypeError, ValueError):
            max_depth = 0
    max_depth = max(0, max_depth)

    windows = payload.get("windows")
    windows = windows if isinstance(windows, list) else []
    window_count = len(windows)

    decomposition = payload.get("decomposition_summary")
    if not isinstance(decomposition, dict) and windows:
        decomposition = build_u2_dynamics_decomposition_summary(windows)

    runs_total = None
    if isinstance(decomposition, dict):
        state_components = decomposition.get("state_components")
        if isinstance(state_components, dict):
            runs_total = state_components.get("runs_total")

    extraction_source = payload.get("extraction_source")
    if isinstance(extraction_source, str):
        extraction_source = extraction_source.upper()
    else:
        extraction_source = None

    status = "ok" if status_light == "GREEN" else "warn"
    drivers: list[str] = []
    if status != "ok":
        if extraction_source == "MISSING":
            drivers.append(DRIVER_U2_SIGNAL_MISSING)
        if (isinstance(runs_total, int) and runs_total <= 0) or window_count == 0:
            drivers.append(DRIVER_U2_RUNS_ZERO)
        if success_rate < GREEN_SUCCESS_RATE:
            drivers.append(DRIVER_U2_SUCCESS_RATE_LOW)
        if max_depth > GREEN_MAX_DEPTH:
            drivers.append(DRIVER_U2_MAX_DEPTH_HIGH)
        if not drivers:
            drivers.append(DRIVER_U2_STATUS_NON_GREEN)

        # Deduplicate while preserving order, then cap to 3.
        deduped: list[str] = []
        seen = set()
        for driver in drivers:
            if driver in seen:
                continue
            seen.add(driver)
            deduped.append(driver)
        drivers = deduped[:3]

    top_driver = drivers[0] if drivers else "NONE"
    summary = f"U2 dynamics: status={status}, window_count={window_count}, top_driver={top_driver}."

    return {
        "signal_type": "SIG-U2",
        "status": status,
        "conflict": False,
        "weight_hint": "LOW",
        "drivers": drivers,
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }


def summarize_u2_dynamics_vs_telemetry(
    dynamics_tile: Optional[Dict[str, Any]],
    telemetry_tile: Optional[Dict[str, Any]],
    window_context: Optional[Dict[str, int]] = None,
    telemetry_windows: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compare dynamics status vs telemetry health and emit a neutral advisory.

    CAL-EXP-1: This consistency block can be embedded in CAL-EXP evidence packs
    (SHADOW/advisory only) to cross-check U2 dynamics against telemetry without
    affecting gating or promotion decisions.
    """
    def _status_label(tile: Optional[Dict[str, Any]]) -> str:
        if tile is None:
            return "RED"
        val = tile.get("status_light") or tile.get("status") or tile.get("overall_status")
        if val is None:
            return "RED"
        return str(val).upper()

    def _status_value(tile: Optional[Dict[str, Any]]) -> int:
        val = _status_label(tile)
        mapping = {
            "GREEN": 0,
            "OK": 0,
            "YELLOW": 1,
            "WARN": 1,
            "ATTENTION": 1,
            "RED": 2,
            "CRITICAL": 2,
        }
        return mapping.get(val, 2)

    def _compare_status(
        dyn_tile: Optional[Dict[str, Any]],
        tel_tile: Optional[Dict[str, Any]],
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        dyn_status_val = _status_value(dyn_tile) if dyn_tile is not None else 2
        tel_status_val = _status_value(tel_tile) if tel_tile is not None else 2

        if dyn_tile is None or tel_tile is None:
            consistency_status = "PARTIAL"
            advisory_notes = ["Missing dynamics or telemetry signal; reporting partial view."]
        elif dyn_status_val == tel_status_val:
            consistency_status = "CONSISTENT"
            advisory_notes = ["Dynamics and telemetry health align."]
        elif dyn_status_val > tel_status_val:
            consistency_status = "INCONSISTENT"
            advisory_notes = ["Dynamics indicates elevated risk relative to telemetry."]
        else:
            consistency_status = "PARTIAL"
            advisory_notes = ["Telemetry indicates elevated risk relative to dynamics."]

        if start_cycle is not None and end_cycle is not None:
            advisory_notes.append(f"Window cycles {start_cycle}-{end_cycle}.")

        return consistency_status, advisory_notes

    dynamics_status = _status_label(dynamics_tile)
    telemetry_status = _status_label(telemetry_tile)

    window_consistency: list[Dict[str, Any]] = []
    dyn_windows = dynamics_tile.get("windows") if isinstance(dynamics_tile, dict) else None
    tel_windows = telemetry_windows
    if tel_windows is None and isinstance(telemetry_tile, dict):
        maybe_windows = telemetry_tile.get("windows")
        if isinstance(maybe_windows, list):
            tel_windows = maybe_windows

    tel_by_index: Dict[int, Dict[str, Any]] = {}
    if isinstance(tel_windows, list):
        for win in tel_windows:
            idx = win.get("window_index")
            if idx is not None:
                try:
                    tel_by_index[int(idx)] = win
                except (TypeError, ValueError):
                    continue

    if isinstance(dyn_windows, list) and dyn_windows:
        for pos, dyn_window in enumerate(dyn_windows):
            idx = dyn_window.get("window_index")
            start = dyn_window.get("start_cycle")
            end = dyn_window.get("end_cycle")
            dyn_tile_entry = dyn_window.get("tile") if isinstance(dyn_window, dict) else None

            tel_window = None
            if idx is not None and idx in tel_by_index:
                tel_window = tel_by_index[idx]
            elif isinstance(tel_windows, list) and pos < len(tel_windows):
                tel_window = tel_windows[pos]
            else:
                tel_window = telemetry_tile

            tel_tile_entry = tel_window.get("tile") if isinstance(tel_window, dict) and "tile" in tel_window else tel_window

            status, notes = _compare_status(dyn_tile_entry, tel_tile_entry, start, end)
            window_consistency.append(
                {
                    "window_index": idx if idx is not None else pos,
                    "start_cycle": start,
                    "end_cycle": end,
                    "consistency_status": status,
                    "advisory_notes": notes,
                    "dynamics_status": _status_label(dyn_tile_entry),
                    "telemetry_status": _status_label(tel_tile_entry),
                }
            )

    if window_consistency:
        statuses = [w["consistency_status"] for w in window_consistency]
        if any(s == "INCONSISTENT" for s in statuses):
            consistency_status = "INCONSISTENT"
            advisory_notes = [
                "At least one window shows elevated dynamics risk relative to telemetry."
            ]
        elif any(s == "PARTIAL" for s in statuses):
            consistency_status = "PARTIAL"
            advisory_notes = [
                "Some windows show telemetry risk above dynamics or missing signals."
            ]
        else:
            consistency_status = "CONSISTENT"
            advisory_notes = ["All evaluated windows show aligned risk signals."]
        advisory_notes.append(f"Evaluated {len(window_consistency)} windows.")
    else:
        start = window_context.get("start_cycle") if window_context else None
        end = window_context.get("end_cycle") if window_context else None
        consistency_status, advisory_notes = _compare_status(
            dynamics_tile, telemetry_tile, start, end
        )

    return {
        "consistency_status": consistency_status,
        "advisory_notes": advisory_notes,
        "dynamics_status": dynamics_status,
        "telemetry_status": telemetry_status,
        "window_consistency": window_consistency,
    }


def _extract_success_and_depth(entry: Dict[str, Any]) -> tuple[bool, Optional[int]]:
    """Pull success/depth from common real_cycles.jsonl record shapes."""
    success = entry.get("success")
    depth = entry.get("depth")
    if success is None:
        success = entry.get("runner_state", {}).get("success")
    if depth is None:
        depth = entry.get("runner_state", {}).get("depth")
    if success is None:
        success = entry.get("runner", {}).get("success")
    if depth is None:
        depth = entry.get("runner", {}).get("depth")
    return bool(success), depth if depth is not None else None


def build_u2_dynamics_window_metrics(
    real_cycles_jsonl_path: Union[str, Path],
    window_size: int = DEFAULT_U2_DYNAMICS_WINDOW_SIZE,
) -> Dict[str, Any]:
    """
    Compute U2 dynamics tiles over fixed windows from real_cycles.jsonl (SHADOW only).

    Returns:
        JSON-safe dict with base dynamics tile + per-window tiles.
    """
    path = Path(real_cycles_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"real_cycles.jsonl not found: {path}")

    windows: Dict[int, Dict[str, Any]] = {}
    total_success = 0
    total_count = 0
    total_depths: list[int] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cycle = int(entry.get("cycle", 0))
            success, depth = _extract_success_and_depth(entry)
            window_index = (cycle - 1) // window_size if cycle > 0 else 0
            window = windows.setdefault(
                window_index,
                {"start_cycle": window_index * window_size + 1, "end_cycle": (window_index + 1) * window_size, "successes": 0, "total": 0, "depths": []},
            )
            window["total"] += 1
            if success:
                window["successes"] += 1
            if depth is not None:
                window["depths"].append(int(depth))

            total_count += 1
            if success:
                total_success += 1
            if depth is not None:
                total_depths.append(int(depth))

    windows_list = []
    for index in sorted(windows.keys()):
        data = windows[index]
        total = data["total"]
        success_rate = (data["successes"] / total) if total else 0.0
        depths = data["depths"]
        mean_depth = sum(depths) / len(depths) if depths else 0.0
        max_depth = max(depths) if depths else 0.0
        tile = build_u2_dynamics_tile(
            {
                "mean_depth": mean_depth,
                "max_depth": max_depth,
                "success_rate": success_rate,
                "runs": total,
            }
        )
        windows_list.append(
            {
                "window_index": index,
                "start_cycle": data["start_cycle"],
                "end_cycle": min(data["end_cycle"], data["start_cycle"] + total - 1),
                "tile": tile,
            }
        )

    overall_mean_depth = sum(total_depths) / len(total_depths) if total_depths else 0.0
    overall_max_depth = max(total_depths) if total_depths else 0.0
    overall_success_rate = (total_success / total_count) if total_count else 0.0
    base_tile = build_u2_dynamics_tile(
        {
            "mean_depth": overall_mean_depth,
            "max_depth": overall_max_depth,
            "success_rate": overall_success_rate,
            "runs": total_count,
        }
    )
    enriched_tile = dict(base_tile)
    enriched_tile["window_size"] = window_size
    enriched_tile["windows"] = windows_list
    return enriched_tile


__all__ = [
    "build_u2_dynamics_tile",
    "attach_u2_dynamics_tile",
    "attach_u2_dynamics_to_p4_summary",
    "extract_u2_dynamics_signal_for_first_light",
    "attach_u2_dynamics_to_first_light_status",
    "attach_u2_dynamics_to_evidence",
    "summarize_u2_dynamics_vs_telemetry",
    "build_u2_dynamics_window_metrics",
    "build_u2_dynamics_first_light_status_signal",
    "build_u2_dynamics_decomposition_summary",
    "build_u2_dynamics_warning",
    "DEFAULT_U2_DYNAMICS_WINDOW_SIZE",
    "U2_DYNAMICS_EXTRACTION_SOURCES",
    "u2_dynamics_for_alignment_view",
]
