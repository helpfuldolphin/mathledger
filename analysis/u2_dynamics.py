
"""
Canonical implementations of the U2 dynamics helpers used by the
conjecture engine. Everything in this module is deterministic so that the
governance classification surfaces remain hash-stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


@dataclass(frozen=True)
class DynamicsThresholds:
    """Typed wrapper around the thresholds used in the dynamics module."""

    stagnation_std_thresh: float
    trend_tau_thresh: float
    oscillation_omega_thresh: float
    step_size_thresh: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "DynamicsThresholds":
        return cls(
            stagnation_std_thresh=float(values.get("stagnation_std_thresh", 0.01)),
            trend_tau_thresh=float(values.get("trend_tau_thresh", -0.2)),
            oscillation_omega_thresh=float(values.get("oscillation_omega_thresh", 0.3)),
            step_size_thresh=float(values.get("step_size_thresh", 0.1)),
        )

    def to_mapping(self) -> Dict[str, float]:
        return {
            "stagnation_std_thresh": self.stagnation_std_thresh,
            "trend_tau_thresh": self.trend_tau_thresh,
            "oscillation_omega_thresh": self.oscillation_omega_thresh,
            "step_size_thresh": self.step_size_thresh,
        }


@dataclass(frozen=True)
class DynamicsDebugSnapshot:
    """Canonical debug payload for introspecting a single dynamics run."""

    schema_version: str
    run_id: str
    slice_name: str
    mode: str
    abstention_series: List[float]
    oscillation_index: float
    plateau_score: float
    uplift_delta: float
    uplift_ci: tuple[float, float]
    pattern_label: str


def _extract_abstention_series(records: list) -> pd.Series:
    """Converts a list of record dicts into a pandas Series."""
    if not records:
        return pd.Series(dtype=float)

    samples: List[tuple[int, float]] = []
    for record in records:
        cycle = record.get("cycle")
        metrics = record.get("metrics") or {}
        abstention = metrics.get("abstention_rate")
        if cycle is None or abstention is None:
            continue
        samples.append((int(cycle), float(abstention)))

    if not samples:
        return pd.Series(dtype=float)

    samples.sort(key=lambda item: item[0])
    indices, values = zip(*samples)
    return pd.Series(values, index=pd.Index(indices, name="cycle"))


def _extract_policy_vectors(records: list) -> np.ndarray:
    """Extracts policy theta vectors as a NumPy array."""
    vectors: List[np.ndarray] = []
    for record in records:
        policy = record.get("policy") or {}
        theta = policy.get("theta")
        if theta is None:
            continue
        vectors.append(np.asarray(theta, dtype=float))

    if not vectors:
        return np.empty((0, 0), dtype=float)

    return np.vstack(vectors)


def estimate_A_t(records: list) -> pd.Series:
    """Returns the abstention-rate time series."""
    return _extract_abstention_series(records)


def detect_pattern(a_t_series: pd.Series, records: list, thresholds: dict) -> str:
    """Detects the qualitative pattern of the abstention series."""
    if a_t_series.empty:
        return "Irregular"

    values = a_t_series.values.astype(float)
    stagnation_std = thresholds.get("stagnation_std_thresh", 0.01)
    tau_thresh = thresholds.get("trend_tau_thresh", -0.2)
    step_thresh = thresholds.get("step_size_thresh", 0.1)
    osc_thresh = thresholds.get("oscillation_omega_thresh", 0.3)

    total_range = float(np.max(values) - np.min(values))
    std = float(np.std(values))

    if std < stagnation_std and total_range < step_thresh:
        return "Stagnation"

    if estimate_oscillation_index(records) > osc_thresh:
        return "Oscillation"

    tau, _ = kendalltau(a_t_series.index, values)
    tau = float(tau) if tau is not None else float("nan")
    if math.isnan(tau):
        tau = 0.0

    first_window = max(5, len(values) // 5)
    first_segment = values[:first_window]
    last_segment = values[-first_window:]
    delta = float(np.mean(first_segment) - np.mean(last_segment))
    plateau_std = float(np.std(first_segment))

    if tau <= tau_thresh and delta > step_thresh:
        if plateau_std < stagnation_std * 3 and delta > step_thresh * 2:
            return "Logistic-like Decay"
        return "Negative Drift"

    if total_range < step_thresh * 0.5:
        return "Stagnation"

    return "Irregular"


def estimate_G_i(baseline_records: list, rfl_records: list) -> dict:
    """Estimates uplift delta and a normal-approximate 95% confidence interval."""
    baseline = estimate_A_t(baseline_records).values.astype(float)
    rfl = estimate_A_t(rfl_records).values.astype(float)

    if baseline.size == 0 or rfl.size == 0:
        return {"delta": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}

    delta = float(np.mean(baseline) - np.mean(rfl))
    var_baseline = float(np.var(baseline, ddof=1)) if baseline.size > 1 else 0.0
    var_rfl = float(np.var(rfl, ddof=1)) if rfl.size > 1 else 0.0
    stderr = math.sqrt(var_baseline / baseline.size + var_rfl / rfl.size)
    margin = 1.96 * stderr

    return {
        "delta": delta,
        "ci_95_lower": delta - margin,
        "ci_95_upper": delta + margin,
    }


def estimate_policy_stability(records: list) -> float:
    """Returns a scalar in (0, 1] describing how smoothly the policy updates."""
    vectors = _extract_policy_vectors(records)
    if len(vectors) < 2:
        return 1.0

    steps = np.diff(vectors, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    if norms.size == 0:
        return 1.0

    mean_step = float(np.mean(norms))
    return 1.0 / (1.0 + mean_step)


def estimate_oscillation_index(records: list) -> float:
    """Combines policy reversals and abstention sign flips into a single score."""
    vectors = _extract_policy_vectors(records)
    policy_ratio = 0.0
    if len(vectors) >= 3:
        steps = np.diff(vectors, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        valid = (norms[:-1] > 0) & (norms[1:] > 0)
        if np.any(valid):
            reversals = 0
            comparisons = 0
            for prev, curr, prev_norm, curr_norm in zip(
                steps[:-1][valid],
                steps[1:][valid],
                norms[:-1][valid],
                norms[1:][valid],
            ):
                cosine = float(np.dot(prev, curr) / (prev_norm * curr_norm))
                if cosine < 0:
                    reversals += 1
                comparisons += 1
            if comparisons:
                policy_ratio = min(1.0, (reversals / comparisons) * 4.0)

    abstention_series = estimate_A_t(records).values.astype(float)
    abstention_ratio = 0.0
    if abstention_series.size >= 3:
        diffs = np.diff(abstention_series)
        magnitude_mask = np.abs(diffs) >= 0.01
        filtered = np.sign(diffs[magnitude_mask])
        filtered = filtered[filtered != 0]
        if filtered.size >= 2:
            changes = np.sum(filtered[1:] * filtered[:-1] < 0)
            coverage = filtered.size / diffs.size
            abstention_ratio = (changes / (filtered.size - 1)) * coverage

    return float(np.mean([policy_ratio, abstention_ratio]))


# Expose kendalltau so callers can import it directly from this module.
kendalltau = kendalltau


def _normalize_pattern_label(pattern: str) -> str:
    mapping = {
        "Logistic-like Decay": "logistic",
        "Negative Drift": "logistic",
        "Oscillation": "oscillatory",
        "Stagnation": "plateau",
        "Irregular": "degenerate",
    }
    return mapping.get(pattern, "degenerate")


def _compute_plateau_score(series: pd.Series, thresholds: DynamicsThresholds) -> float:
    if series.empty:
        return 0.0
    std = float(np.std(series.values))
    normalized = std / (thresholds.stagnation_std_thresh + 1e-9)
    return 1.0 / (1.0 + normalized)


def build_dynamics_debug_snapshot(
    run_summary: Dict[str, Any],
    thresholds: DynamicsThresholds,
) -> DynamicsDebugSnapshot:
    """
    Build a deterministic debug snapshot from a U2 runner summary.

    U2 runners should call this once per run (baseline, RFL, NC) after they have
    the final abstention series plus uplift deltas/CI so downstream consumers can
    attach the resulting snapshot under ``global_health["dynamics"]``.
    Negative-control runs (baseline/nc modes) must be passed through this same
    builder; the NC auditor will later filter those snapshots and ensure they
    remain quiescent before surfacing a console tile.
    """
    records = run_summary.get("records") or []
    slice_name = str(run_summary.get("slice_name", "default"))
    mode = str(run_summary.get("mode", "nc"))
    if mode not in {"baseline", "rfl", "nc"}:
        mode = "nc"

    run_id = str(run_summary.get("run_id") or f"{slice_name}-{mode}")

    series = estimate_A_t(records)
    oscillation = float(estimate_oscillation_index(records))
    plateau_score = _compute_plateau_score(series, thresholds)

    comparison = run_summary.get("comparison") or {}
    baseline_records = run_summary.get("baseline_records") or comparison.get("baseline_records")
    rfl_records = run_summary.get("rfl_records") or comparison.get("rfl_records")
    if isinstance(baseline_records, list) and isinstance(rfl_records, list):
        uplift_info = estimate_G_i(baseline_records, rfl_records)
    else:
        uplift_info = {"delta": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}

    pattern = run_summary.get("pattern")
    if not pattern:
        pattern = detect_pattern(series, records, thresholds.to_mapping())
    pattern_label = _normalize_pattern_label(pattern)

    return DynamicsDebugSnapshot(
        schema_version="1.0.0",
        run_id=run_id,
        slice_name=slice_name,
        mode=mode,
        abstention_series=[float(x) for x in series.to_list()],
        oscillation_index=oscillation,
        plateau_score=plateau_score,
        uplift_delta=float(uplift_info["delta"]),
        uplift_ci=(
            float(uplift_info["ci_95_lower"]),
            float(uplift_info["ci_95_upper"]),
        ),
        pattern_label=pattern_label,
    )


def check_negative_control_dynamics(
    debug_snapshots: Sequence[DynamicsDebugSnapshot],
    max_allowed_oscillation_index: float = 0.1,
) -> Dict[str, Any]:
    """
    Ensures that negative-control runs never appear oscillatory.

    Returns a deterministic summary dict per the NC audit schema.
    """
    status_order = {"OK": 0, "DRIFTING": 1, "BROKEN": 2}
    status = "OK"
    notes: List[str] = []
    max_osc = 0.0
    max_plateau = 0.0

    def _escalate(new_status: str) -> None:
        nonlocal status
        if status_order[new_status] > status_order[status]:
            status = new_status

    for snapshot in debug_snapshots:
        max_osc = max(max_osc, snapshot.oscillation_index)
        max_plateau = max(max_plateau, snapshot.plateau_score)
        if snapshot.pattern_label not in {"plateau", "degenerate"}:
            notes.append(
                f"{snapshot.run_id} pattern={snapshot.pattern_label} violates NC expectation"
            )
            _escalate("DRIFTING")
        if snapshot.oscillation_index > max_allowed_oscillation_index:
            notes.append(
                f"{snapshot.run_id} oscillation_index={snapshot.oscillation_index:.3f} exceeds {max_allowed_oscillation_index:.3f}"
            )
            _escalate("BROKEN")

    return {
        "schema_version": "1.0.0",
        "run_count": len(debug_snapshots),
        "max_oscillation_index": max_osc,
        "max_plateau_score": max_plateau,
        "status": status,
        "notes": notes,
    }


def summarize_dynamics_for_global_console(
    snapshots: Sequence[DynamicsDebugSnapshot],
) -> Dict[str, Any]:
    """
    Produces a deterministic aggregate suitable for the global console tile.
    """

    def _round(value: float) -> float:
        return round(float(value), 6)

    run_count = len(snapshots)
    if run_count:
        mean_osc = _round(sum(s.oscillation_index for s in snapshots) / run_count)
        max_osc = _round(max(s.oscillation_index for s in snapshots))
    else:
        mean_osc = 0.0
        max_osc = 0.0

    nc_candidates = [snap for snap in snapshots if snap.mode in {"baseline", "nc"}]
    if nc_candidates:
        nc_report = check_negative_control_dynamics(nc_candidates)
        nc_status = nc_report["status"]
    else:
        nc_status = "OK"

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
        "headline": headline,
    }
