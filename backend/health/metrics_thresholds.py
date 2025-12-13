"""
Metrics Threshold Registry for P5 Migration.

Provides dual threshold management for MOCK (P3/P4) and REAL (P5) modes,
with HYBRID mode for safe transition comparison.

SHADOW MODE: Observation-only. No control paths.
This module provides threshold values but does NOT enforce governance.

REAL-READY: This module is designed for P5 migration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ==============================================================================
# Threshold Mode Constants
# ==============================================================================

MODE_MOCK = "MOCK"      # P3/P4 synthetic thresholds (default)
MODE_HYBRID = "HYBRID"  # Dual evaluation, MOCK authoritative
MODE_REAL = "REAL"      # P5 real telemetry thresholds

VALID_MODES = frozenset({MODE_MOCK, MODE_HYBRID, MODE_REAL})

# Environment variable for threshold mode
ENV_THRESHOLD_MODE = "METRIC_THRESHOLDS_MODE"


# ==============================================================================
# Dual Threshold Registry
# ==============================================================================

# REAL-READY: Dual threshold registry
# P3/P4 (MOCK) vs P5 (REAL) thresholds
_THRESHOLDS: Dict[str, Dict[str, float]] = {
    MODE_MOCK: {
        # Drift thresholds
        "drift_warn": 0.30,
        "drift_critical": 0.70,
        # Success rate thresholds (percentage)
        "success_rate_warn": 80.0,
        "success_rate_critical": 50.0,
        # Budget thresholds (percentage)
        "budget_warn": 80.0,
        "budget_critical": 95.0,
        # Abstention thresholds (percentage)
        "abstention_warn": 5.0,
        "abstention_critical": 15.0,
        # Block rate thresholds (ratio)
        "block_rate_warn": 0.08,
        "block_rate_critical": 0.20,
    },
    MODE_REAL: {
        # Drift thresholds - relaxed for real variance
        "drift_warn": 0.35,
        "drift_critical": 0.75,
        # Success rate thresholds - relaxed for real proofs
        "success_rate_warn": 75.0,
        "success_rate_critical": 45.0,
        # Budget thresholds - adjusted for real utilization
        "budget_warn": 85.0,
        "budget_critical": 92.0,
        # Abstention thresholds - relaxed for bursty queues
        "abstention_warn": 8.0,
        "abstention_critical": 18.0,
        # Block rate thresholds - relaxed for real derivation
        "block_rate_warn": 0.12,
        "block_rate_critical": 0.25,
    },
}

# Safe comparison bands (from Metrics_PhaseX_Spec.md Section 6.3)
_SAFE_COMPARISON_BANDS: Dict[str, float] = {
    "success_rate": 15.0,       # ±15%
    "block_rate": 0.08,         # ±0.08
    "abstention_rate": 5.0,     # ±5%
    "drift_magnitude": 0.15,    # ±0.15
    "budget_utilization": 10.0, # ±10%
}


# ==============================================================================
# Mode Management
# ==============================================================================

def get_threshold_mode() -> str:
    """
    Get current threshold mode from environment.

    REAL-READY: Reads METRIC_THRESHOLDS_MODE env var.

    Returns:
        MODE_MOCK | MODE_HYBRID | MODE_REAL
    """
    mode = os.environ.get(ENV_THRESHOLD_MODE, MODE_MOCK).upper()
    if mode not in VALID_MODES:
        return MODE_MOCK
    return mode


def _resolve_mode(mode: Optional[str]) -> str:
    """
    Resolve mode for threshold lookup.

    HYBRID mode resolves to MOCK for authoritative thresholds.

    Args:
        mode: Explicit mode or None to use env.

    Returns:
        MODE_MOCK | MODE_REAL (never HYBRID for lookups)
    """
    if mode is None:
        mode = get_threshold_mode()

    mode = mode.upper()

    # HYBRID uses MOCK as authoritative
    if mode == MODE_HYBRID:
        return MODE_MOCK

    if mode not in {MODE_MOCK, MODE_REAL}:
        return MODE_MOCK

    return mode


# ==============================================================================
# Threshold Access
# ==============================================================================

def get_threshold(name: str, mode: Optional[str] = None) -> float:
    """
    Get threshold by name and mode.

    REAL-READY: Supports MOCK, HYBRID, REAL modes.

    Args:
        name: Threshold name (e.g., "drift_warn", "success_rate_critical")
        mode: Override mode, defaults to METRIC_THRESHOLDS_MODE env

    Returns:
        Threshold value for the specified mode.

    Raises:
        KeyError: If threshold name not found.

    Example:
        >>> get_threshold("drift_warn")  # Uses env mode
        0.30
        >>> get_threshold("drift_warn", "REAL")
        0.35
    """
    resolved_mode = _resolve_mode(mode)
    return _THRESHOLDS[resolved_mode][name]


def get_all_thresholds(mode: Optional[str] = None) -> Dict[str, float]:
    """
    Get all thresholds for a mode.

    REAL-READY: Returns copy to prevent mutation.

    Args:
        mode: Override mode, defaults to METRIC_THRESHOLDS_MODE env

    Returns:
        Dict of all threshold name → value pairs.
    """
    resolved_mode = _resolve_mode(mode)
    return _THRESHOLDS[resolved_mode].copy()


def get_threshold_pair(name: str) -> Dict[str, float]:
    """
    Get both MOCK and REAL thresholds for a metric.

    REAL-READY: Useful for comparison logging.

    Args:
        name: Threshold name

    Returns:
        Dict with "mock" and "real" values.

    Example:
        >>> get_threshold_pair("drift_warn")
        {"mock": 0.30, "real": 0.35}
    """
    return {
        "mock": _THRESHOLDS[MODE_MOCK][name],
        "real": _THRESHOLDS[MODE_REAL][name],
    }


def list_threshold_names() -> List[str]:
    """
    List all available threshold names.

    Returns:
        Sorted list of threshold names.
    """
    return sorted(_THRESHOLDS[MODE_MOCK].keys())


# ==============================================================================
# Safe Comparison Bands
# ==============================================================================

def get_safe_band(metric: str) -> float:
    """
    Get safe comparison band for a metric.

    REAL-READY: Used for P3 vs P5 comparison.

    Args:
        metric: Metric name (success_rate, block_rate, etc.)

    Returns:
        Band width (±value for comparison).

    Raises:
        KeyError: If metric not found.
    """
    return _SAFE_COMPARISON_BANDS[metric]


def get_all_safe_bands() -> Dict[str, float]:
    """
    Get all safe comparison bands.

    Returns:
        Copy of safe bands dict.
    """
    return _SAFE_COMPARISON_BANDS.copy()


def check_in_band(
    metric: str,
    p3_value: float,
    p5_value: float,
) -> Dict[str, Any]:
    """
    Check if metric delta is within safe comparison band.

    REAL-READY: Core band position check.

    Args:
        metric: Metric name
        p3_value: P3/P4 (mock) value
        p5_value: P5 (real) value

    Returns:
        Dict with delta, band, and in_band status.

    Example:
        >>> check_in_band("success_rate", 90.0, 82.0)
        {"p3": 90.0, "p5": 82.0, "delta": 8.0, "band": 15.0, "in_band": True}
    """
    band = _SAFE_COMPARISON_BANDS.get(metric)
    if band is None:
        return {
            "p3": p3_value,
            "p5": p5_value,
            "delta": abs(p3_value - p5_value),
            "band": None,
            "in_band": None,
            "error": f"Unknown metric: {metric}",
        }

    delta = abs(p3_value - p5_value)
    return {
        "p3": p3_value,
        "p5": p5_value,
        "delta": round(delta, 6),
        "band": band,
        "in_band": delta <= band,
    }


def log_band_position(
    p3_metrics: Dict[str, float],
    p5_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Log where metrics fall relative to safe comparison bands.

    REAL-READY: Full band position tracking.

    Args:
        p3_metrics: Dict of P3/P4 metric values
        p5_metrics: Dict of P5 metric values

    Returns:
        Dict with positions for each metric and aggregate status.

    Example:
        >>> log_band_position(
        ...     {"success_rate": 92.5, "block_rate": 0.05},
        ...     {"success_rate": 88.1, "block_rate": 0.09},
        ... )
        {
            "positions": {...},
            "all_in_band": True,
            "out_of_band_count": 0,
            "out_of_band_metrics": [],
        }
    """
    positions: Dict[str, Dict[str, Any]] = {}
    out_of_band_metrics: List[str] = []

    for metric in _SAFE_COMPARISON_BANDS:
        p3_val = p3_metrics.get(metric, 0.0)
        p5_val = p5_metrics.get(metric, 0.0)
        result = check_in_band(metric, p3_val, p5_val)
        positions[metric] = result

        if not result.get("in_band", True):
            out_of_band_metrics.append(metric)

    return {
        "positions": positions,
        "all_in_band": len(out_of_band_metrics) == 0,
        "out_of_band_count": len(out_of_band_metrics),
        "out_of_band_metrics": sorted(out_of_band_metrics),
    }


# ==============================================================================
# Dual Threshold Evaluation
# ==============================================================================

def evaluate_with_dual_thresholds(
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Evaluate metrics against both MOCK and REAL thresholds.

    REAL-READY: Core dual evaluation for HYBRID mode.

    In HYBRID mode, returns both verdicts with divergence detection.
    In MOCK/REAL mode, returns single verdict.

    Args:
        metrics: Dict with metric values:
            - drift_magnitude: float
            - success_rate: float (percentage)
            - budget_utilization: float (percentage)
            - abstention_rate: float (percentage)
            - block_rate: float (ratio)

    Returns:
        Dict with evaluation results:
            - mode: current mode
            - verdict: authoritative verdict (MOCK in HYBRID)
            - dual_evaluation: bool
            - p5_verdict: P5 verdict (HYBRID only)
            - diverges: bool (HYBRID only)
            - divergence_detail: dict (if diverges)

    Example:
        >>> os.environ["METRIC_THRESHOLDS_MODE"] = "HYBRID"
        >>> evaluate_with_dual_thresholds({"drift_magnitude": 0.32})
        {
            "mode": "HYBRID",
            "verdict": {"status": "YELLOW", ...},
            "p5_verdict": {"status": "GREEN", ...},
            "dual_evaluation": True,
            "diverges": True,
            "divergence_detail": {...}
        }
    """
    mode = get_threshold_mode()

    if mode != MODE_HYBRID:
        # Single evaluation
        verdict = _evaluate_single(metrics, mode)
        return {
            "mode": mode,
            "verdict": verdict,
            "dual_evaluation": False,
        }

    # HYBRID: evaluate both
    mock_verdict = _evaluate_single(metrics, MODE_MOCK)
    real_verdict = _evaluate_single(metrics, MODE_REAL)

    diverges = mock_verdict["status"] != real_verdict["status"]

    result: Dict[str, Any] = {
        "mode": MODE_HYBRID,
        "verdict": mock_verdict,  # MOCK is authoritative
        "p5_verdict": real_verdict,  # P5 for logging
        "dual_evaluation": True,
        "diverges": diverges,
    }

    if diverges:
        result["divergence_detail"] = {
            "mock_status": mock_verdict["status"],
            "real_status": real_verdict["status"],
            "triggered_thresholds": _find_divergent_thresholds(metrics),
        }

    return result


def _evaluate_single(
    metrics: Dict[str, float],
    mode: str,
) -> Dict[str, Any]:
    """
    Evaluate metrics against a single threshold set.

    REAL-READY: Internal single-mode evaluation.

    Args:
        metrics: Dict with metric values
        mode: MODE_MOCK or MODE_REAL

    Returns:
        Dict with status and reasons.
    """
    resolved_mode = _resolve_mode(mode)
    thresholds = _THRESHOLDS[resolved_mode]

    status = "GREEN"
    reasons: List[str] = []

    # Check drift
    drift_mag = metrics.get("drift_magnitude", 0.0)
    if drift_mag >= thresholds["drift_critical"]:
        status = "RED"
        reasons.append(f"drift_magnitude {drift_mag:.3f} >= critical {thresholds['drift_critical']}")
    elif drift_mag >= thresholds["drift_warn"]:
        if status == "GREEN":
            status = "YELLOW"
        reasons.append(f"drift_magnitude {drift_mag:.3f} >= warn {thresholds['drift_warn']}")

    # Check success rate
    success_rate = metrics.get("success_rate", 100.0)
    if success_rate < thresholds["success_rate_critical"]:
        status = "RED"
        reasons.append(f"success_rate {success_rate:.1f}% < critical {thresholds['success_rate_critical']}%")
    elif success_rate < thresholds["success_rate_warn"]:
        if status == "GREEN":
            status = "YELLOW"
        reasons.append(f"success_rate {success_rate:.1f}% < warn {thresholds['success_rate_warn']}%")

    # Check budget utilization
    budget_util = metrics.get("budget_utilization", 0.0)
    if budget_util >= thresholds["budget_critical"]:
        status = "RED"
        reasons.append(f"budget_utilization {budget_util:.1f}% >= critical {thresholds['budget_critical']}%")
    elif budget_util >= thresholds["budget_warn"]:
        if status == "GREEN":
            status = "YELLOW"
        reasons.append(f"budget_utilization {budget_util:.1f}% >= warn {thresholds['budget_warn']}%")

    # Check abstention rate
    abstention_rate = metrics.get("abstention_rate", 0.0)
    if abstention_rate >= thresholds["abstention_critical"]:
        status = "RED"
        reasons.append(f"abstention_rate {abstention_rate:.1f}% >= critical {thresholds['abstention_critical']}%")
    elif abstention_rate >= thresholds["abstention_warn"]:
        if status == "GREEN":
            status = "YELLOW"
        reasons.append(f"abstention_rate {abstention_rate:.1f}% >= warn {thresholds['abstention_warn']}%")

    # Check block rate
    block_rate = metrics.get("block_rate", 0.0)
    if block_rate >= thresholds["block_rate_critical"]:
        status = "RED"
        reasons.append(f"block_rate {block_rate:.3f} >= critical {thresholds['block_rate_critical']}")
    elif block_rate >= thresholds["block_rate_warn"]:
        if status == "GREEN":
            status = "YELLOW"
        reasons.append(f"block_rate {block_rate:.3f} >= warn {thresholds['block_rate_warn']}")

    return {
        "status": status,
        "mode": resolved_mode,
        "reasons": reasons,
        "thresholds_used": thresholds,
    }


def _find_divergent_thresholds(metrics: Dict[str, float]) -> List[str]:
    """
    Find thresholds where MOCK and REAL would give different verdicts.

    Args:
        metrics: Dict with metric values

    Returns:
        List of threshold names causing divergence.
    """
    divergent: List[str] = []

    # Drift
    drift_mag = metrics.get("drift_magnitude", 0.0)
    mock_drift_warn = _THRESHOLDS[MODE_MOCK]["drift_warn"]
    real_drift_warn = _THRESHOLDS[MODE_REAL]["drift_warn"]
    if mock_drift_warn <= drift_mag < real_drift_warn:
        divergent.append("drift_warn")

    mock_drift_crit = _THRESHOLDS[MODE_MOCK]["drift_critical"]
    real_drift_crit = _THRESHOLDS[MODE_REAL]["drift_critical"]
    if mock_drift_crit <= drift_mag < real_drift_crit:
        divergent.append("drift_critical")

    # Success rate (inverted - lower is worse)
    success_rate = metrics.get("success_rate", 100.0)
    mock_sr_warn = _THRESHOLDS[MODE_MOCK]["success_rate_warn"]
    real_sr_warn = _THRESHOLDS[MODE_REAL]["success_rate_warn"]
    if real_sr_warn <= success_rate < mock_sr_warn:
        divergent.append("success_rate_warn")

    mock_sr_crit = _THRESHOLDS[MODE_MOCK]["success_rate_critical"]
    real_sr_crit = _THRESHOLDS[MODE_REAL]["success_rate_critical"]
    if real_sr_crit <= success_rate < mock_sr_crit:
        divergent.append("success_rate_critical")

    # Budget utilization
    budget_util = metrics.get("budget_utilization", 0.0)
    mock_budget_warn = _THRESHOLDS[MODE_MOCK]["budget_warn"]
    real_budget_warn = _THRESHOLDS[MODE_REAL]["budget_warn"]
    if mock_budget_warn <= budget_util < real_budget_warn:
        divergent.append("budget_warn")

    # Budget critical is inverted (REAL is lower)
    mock_budget_crit = _THRESHOLDS[MODE_MOCK]["budget_critical"]
    real_budget_crit = _THRESHOLDS[MODE_REAL]["budget_critical"]
    if real_budget_crit <= budget_util < mock_budget_crit:
        divergent.append("budget_critical")

    # Abstention rate
    abstention = metrics.get("abstention_rate", 0.0)
    mock_abs_warn = _THRESHOLDS[MODE_MOCK]["abstention_warn"]
    real_abs_warn = _THRESHOLDS[MODE_REAL]["abstention_warn"]
    if mock_abs_warn <= abstention < real_abs_warn:
        divergent.append("abstention_warn")

    mock_abs_crit = _THRESHOLDS[MODE_MOCK]["abstention_critical"]
    real_abs_crit = _THRESHOLDS[MODE_REAL]["abstention_critical"]
    if mock_abs_crit <= abstention < real_abs_crit:
        divergent.append("abstention_critical")

    # Block rate
    block_rate = metrics.get("block_rate", 0.0)
    mock_br_warn = _THRESHOLDS[MODE_MOCK]["block_rate_warn"]
    real_br_warn = _THRESHOLDS[MODE_REAL]["block_rate_warn"]
    if mock_br_warn <= block_rate < real_br_warn:
        divergent.append("block_rate_warn")

    mock_br_crit = _THRESHOLDS[MODE_MOCK]["block_rate_critical"]
    real_br_crit = _THRESHOLDS[MODE_REAL]["block_rate_critical"]
    if mock_br_crit <= block_rate < real_br_crit:
        divergent.append("block_rate_critical")

    return sorted(divergent)


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Constants
    "MODE_MOCK",
    "MODE_HYBRID",
    "MODE_REAL",
    "VALID_MODES",
    "ENV_THRESHOLD_MODE",
    # Mode management
    "get_threshold_mode",
    # Threshold access
    "get_threshold",
    "get_all_thresholds",
    "get_threshold_pair",
    "list_threshold_names",
    # Safe comparison bands
    "get_safe_band",
    "get_all_safe_bands",
    "check_in_band",
    "log_band_position",
    # Dual evaluation
    "evaluate_with_dual_thresholds",
]
