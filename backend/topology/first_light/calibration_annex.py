from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rfl.prng.calibration_integration import build_prng_calibration_annex
except ImportError:
    # PRNG module not available - PRNG integration is optional
    build_prng_calibration_annex = None

ANNEX_SCHEMA_VERSION = "1.1.0"
METRIC_DEFINITIONS_REF = (
    "docs/system_law/calibration/METRIC_DEFINITIONS.md@v1.1.0"
)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_divergence_component(
    report: Dict[str, Any],
    key: str,
    *,
    last_window: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    candidates: List[Any] = []
    candidates.append(report.get(key))
    candidates.append((report.get("summary") or {}).get(key))
    candidates.append((report.get("true_divergence") or {}).get(key))
    candidates.append((report.get("divergence_decomposition") or {}).get(key))
    candidates.append((report.get("decomposition") or {}).get(key))
    if last_window is not None:
        candidates.append(last_window.get(key))
    for candidate in candidates:
        parsed = _coerce_float(candidate)
        if parsed is not None:
            return parsed
    return None


def _extract_state_delta_p_mean(window: Dict[str, Any]) -> Optional[float]:
    for key in ("mean_delta_p", "state_delta_p_mean", "mean_abs_delta_p", "mean_delta_p_abs"):
        parsed = _coerce_float(window.get(key))
        if parsed is not None:
            return parsed
    return None


def _state_delta_p_trend(
    first_window: Dict[str, Any],
    last_window: Dict[str, Any],
    windows: List[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    if len(windows) < 2:
        return None
    start = _extract_state_delta_p_mean(first_window)
    end = _extract_state_delta_p_mean(last_window)
    if start is None or end is None:
        return None
    return start, end


def load_cal_exp1_annex(
    report_path: Path,
    prng_tiles: Optional[List[Dict[str, Any]]] = None,
    prng_history: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load CAL-EXP-1 annex data from a report JSON.

    Optionally integrates PRNG drift analysis if PRNG data is provided.

    Args:
        report_path: Path to CAL-EXP-1 report JSON.
        prng_tiles: Optional list of PRNG governance tiles (one per window).
        prng_history: Optional PRNG governance history.

    Returns:
        Compact annex dictionary or None if unavailable.
        If PRNG data is provided, includes PRNG calibration annex.
    """
    if not report_path.exists():
        return None

    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    windows: List[Dict[str, Any]] = report.get("windows") or []
    summary = report.get("summary") or {}
    if not windows:
        last_window: Dict[str, Any] = {}
        first_window: Dict[str, Any] = {}
    else:
        last_window = windows[-1]
        first_window = windows[0]

    legacy_learning_occurring = _learning_occurring_legacy(first_window, last_window, windows)

    state_delta_p_mean = _extract_state_delta_p_mean(last_window)
    success_div_rate = _extract_divergence_component(report, "success_div_rate", last_window=last_window)
    omega_div_rate = _extract_divergence_component(report, "omega_div_rate", last_window=last_window)
    blocked_div_rate = _extract_divergence_component(report, "blocked_div_rate", last_window=last_window)
    true_divergence_vector_present = any(
        component is not None
        for component in (
            state_delta_p_mean,
            success_div_rate,
            omega_div_rate,
            blocked_div_rate,
        )
    )

    if len(windows) < 2:
        learning_occurring = legacy_learning_occurring
        learning_occurring_basis = "INSUFFICIENT_WINDOWS"
    else:
        state_trend = _state_delta_p_trend(first_window, last_window, windows)
        if state_trend is None:
            learning_occurring = legacy_learning_occurring
            learning_occurring_basis = "LEGACY_DIVERGENCE_HEURISTIC"
        else:
            start, end = state_trend
            learning_occurring = end < start
            learning_occurring_basis = "MEAN_DELTA_P_TREND"

    annex = {
        "schema_version": ANNEX_SCHEMA_VERSION,
        "metric_definitions_ref": METRIC_DEFINITIONS_REF,
        "final_divergence_rate": summary.get("final_divergence_rate"),
        "delta_bias_end": last_window.get("delta_bias"),
        "delta_variance_end": last_window.get("delta_variance"),
        "phase_lag_xcorr_end": last_window.get("phase_lag_xcorr"),
        "learning_occurring": learning_occurring,
        "learning_occurring_legacy": legacy_learning_occurring,
        "learning_occurring_basis": learning_occurring_basis,
        "pattern_tag": last_window.get("pattern_tag") or summary.get("pattern_tag") or "UNKNOWN",
        "true_divergence_vector_present": true_divergence_vector_present,
        "state_delta_p_mean": state_delta_p_mean,
        "success_div_rate": success_div_rate,
        "omega_div_rate": omega_div_rate,
        "blocked_div_rate": blocked_div_rate,
    }
    
    # Integrate PRNG calibration annex if PRNG data provided and module available
    if (prng_tiles or prng_history) and build_prng_calibration_annex is not None:
        prng_annex = build_prng_calibration_annex(windows, prng_tiles, prng_history)
        annex["prng"] = prng_annex
    
    return annex


def _learning_occurring_legacy(
    first_window: Dict[str, Any],
    last_window: Dict[str, Any],
    windows: List[Dict[str, Any]],
) -> bool:
    """
    Determine if learning is occurring using a simple divergence heuristic.
    """
    if not windows:
        return False

    first_div = float(first_window.get("divergence_rate", 0.0) or 0.0)
    last_div = float(last_window.get("divergence_rate", 0.0) or 0.0)
    if len(windows) == 1:
        return last_div <= 0.5
    return last_div <= first_div
