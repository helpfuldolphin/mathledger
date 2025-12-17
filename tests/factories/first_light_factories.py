"""Reusable factory helpers for First-Light tests."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "make_synthetic_raw_record",
    "make_red_flag_entry",
    "make_divergence_entry",
    "make_metrics_window",
    "make_tda_window",
    "make_summary_payload",
    "make_stability_report_payload",
    "make_p4_divergence_log_record",
    "make_real_telemetry_snapshot",
    "make_cal_exp1_report",
    "make_cal_exp2_report",
    "make_cal_exp3_report",
    "emit_cal_exp_reports_for_evidence_pack",
    "emit_cal_exp1_report_for_evidence_pack",
    "emit_cal_exp2_report_for_evidence_pack",
    "emit_cal_exp3_report_for_evidence_pack",
]

_BASE_TIMESTAMP = datetime(2025, 1, 1, tzinfo=timezone.utc)
_DEFAULT_FACTORY_SEED = 1337


def _timestamp_for_cycle(cycle: int) -> str:
    """Return a deterministic ISO8601 timestamp for a cycle."""
    ts = _BASE_TIMESTAMP + timedelta(seconds=cycle * 17)
    return ts.isoformat().replace("+00:00", "Z")


def _namespace_seed(namespace: str) -> int:
    """Derive a numeric offset from the namespace string."""
    total = 0
    for ch in namespace:
        total = (total * 131 + ord(ch)) % 1_000_000
    return total


def _rng_for(namespace: str, key: int, seed: Optional[int]) -> random.Random:
    """Return a deterministic Random for the namespace/key combination."""
    base = _DEFAULT_FACTORY_SEED if seed is None else seed
    mix = base + _namespace_seed(namespace) + key * 7919
    return random.Random(mix)


def make_synthetic_raw_record(cycle: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a dictionary compatible with CycleLogEntry(**payload).

    Args:
        cycle: Synthetic cycle number.

    Returns:
        Dictionary with CycleLogEntry field names and realistic values.
    """
    rng = _rng_for("raw_record", cycle, seed)
    timestamp = _timestamp_for_cycle(cycle)
    runner_success = rng.random() > 0.25
    runner_depth = rng.randint(3, 8) if runner_success else rng.randint(1, 4)

    runner_type = rng.choice(["u2", "rfl"])
    runner_slice = rng.choice(["arithmetic_simple", "safety_mix", "drift_probe"])

    usla_H = round(0.45 + 0.4 * rng.random(), 4)
    usla_tau = round(0.18 + 0.05 * rng.random(), 4)
    usla_rho = round(0.5 + 0.45 * rng.random(), 4)
    usla_beta = round(0.2 + 0.6 * rng.random(), 4)

    in_omega = usla_H > usla_tau and usla_rho > 0.5
    hard_ok = rng.random() > 0.05
    abstained = (not runner_success) and rng.random() > 0.6
    real_blocked = usla_beta > 0.65
    governance_aligned = rng.random() > 0.1
    sim_blocked = real_blocked if governance_aligned else not real_blocked

    return {
        "schema": "first-light-cycle/1.0.0",
        "cycle": cycle,
        "timestamp": timestamp,
        "mode": "SHADOW",
        "runner_type": runner_type,
        "runner_slice": runner_slice,
        "runner_success": runner_success,
        "runner_depth": runner_depth,
        "usla_H": round(usla_H, 4),
        "usla_rho": usla_rho,
        "usla_tau": usla_tau,
        "usla_beta": usla_beta,
        "real_blocked": real_blocked,
        "sim_blocked": sim_blocked,
        "governance_aligned": governance_aligned,
        "hard_ok": hard_ok,
        "in_omega": in_omega,
        "abstained": abstained,
    }


_RED_FLAG_SEVERITY = {
    "RSI_COLLAPSE": "WARNING",
    "OMEGA_EXIT": "INFO",
    "HARD_FAIL": "CRITICAL",
    "GOVERNANCE_DIVERGENCE": "WARNING",
    "CDI_007": "WARNING",
    "CDI_010": "CRITICAL",
}

_RED_FLAG_THRESHOLD = {
    "RSI_COLLAPSE": 0.2,
    "OMEGA_EXIT": 1.0,
    "HARD_FAIL": 1.0,
    "GOVERNANCE_DIVERGENCE": 1.0,
    "CDI_007": 0.5,
    "CDI_010": 0.2,
}


def make_red_flag_entry(cycle: int, kind: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a dictionary compatible with RedFlagLogEntry(**payload).

    Args:
        cycle: Synthetic cycle number.
        kind: Red-flag type string such as "RSI_COLLAPSE".
    """
    rng = _rng_for(f"red_flag_{kind}", cycle, seed)
    timestamp = _timestamp_for_cycle(cycle)
    severity = _RED_FLAG_SEVERITY.get(kind, "INFO")
    threshold = _RED_FLAG_THRESHOLD.get(kind, 1.0)

    consecutive_cycles = rng.randint(1, 5)
    observed_value = threshold

    if kind == "RSI_COLLAPSE":
        observed_value = round(0.1 + 0.1 * rng.random(), 4)
    elif kind == "OMEGA_EXIT":
        observed_value = 0.0
    elif kind == "HARD_FAIL":
        observed_value = 0.0
    elif kind == "GOVERNANCE_DIVERGENCE":
        observed_value = float(consecutive_cycles)
    elif kind == "CDI_007":
        observed_value = round(0.55 + 0.1 * rng.random(), 4)
    elif kind == "CDI_010":
        observed_value = round(0.19 + 0.02 * rng.random(), 4)

    return {
        "schema": "first-light-red-flag/1.0.0",
        "cycle": cycle,
        "timestamp": timestamp,
        "mode": "SHADOW",
        "flag_type": kind,
        "flag_severity": severity,
        "observed_value": observed_value,
        "threshold": threshold,
        "consecutive_cycles": consecutive_cycles,
        "action": "LOGGED_ONLY",
        "hypothetical_would_abort": False,
        "hypothetical_reason": None,
    }


def make_divergence_entry(cycle: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a dictionary compatible with DivergenceLogEntry(**payload).

    Args:
        cycle: Synthetic cycle number.
    """
    rng = _rng_for("divergence_entry", cycle, seed)
    timestamp = _timestamp_for_cycle(cycle)
    success_diverged = rng.random() > 0.6
    omega_diverged = rng.random() > 0.75
    hard_ok_diverged = rng.random() > 0.85
    blocked_diverged = rng.random() > 0.9

    diverged = success_diverged or omega_diverged or hard_ok_diverged or blocked_diverged

    if diverged:
        severity = "MODERATE" if rng.random() > 0.5 else "MINOR"
        if success_diverged and omega_diverged:
            divergence_type = "BOTH"
        elif success_diverged or hard_ok_diverged or blocked_diverged:
            divergence_type = "OUTCOME"
        else:
            divergence_type = "STATE"
    else:
        severity = "NONE"
        divergence_type = "NONE"

    return {
        "schema": "first-light-p4-divergence/1.0.0",
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
        "cycle": cycle,
        "timestamp": timestamp,
        "success_diverged": success_diverged,
        "blocked_diverged": blocked_diverged,
        "omega_diverged": omega_diverged,
        "hard_ok_diverged": hard_ok_diverged,
        "H_delta": round(0.01 * (cycle % 4), 4),
        "rho_delta": round(0.02 * (cycle % 3), 4),
        "tau_delta": round(0.005 * (cycle % 5), 4),
        "beta_delta": round(0.01 * rng.random(), 4),
        "severity": severity,
        "divergence_type": divergence_type,
        "consecutive_count": rng.randint(1, 4) if diverged else 0,
    }


def make_real_telemetry_snapshot(
    cycle: int,
    *,
    runner_type: str = "u2",
    slice_name: str = "arithmetic_simple",
    source: str = "REAL_RUNNER",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a dictionary that matches the real telemetry contract (14+ fields).
    """
    rng = _rng_for("real_telemetry", cycle, seed)
    timestamp = _timestamp_for_cycle(cycle)

    success = rng.random() > 0.3
    hard_ok = rng.random() > 0.1
    real_blocked = rng.random() > 0.85
    governance_aligned = rng.random() > 0.1
    sim_blocked = real_blocked if governance_aligned else not real_blocked
    abstained = (not success) and rng.random() > 0.6

    H_val = round(0.4 + 0.4 * rng.random(), 4)
    rho_val = round(0.5 + 0.4 * rng.random(), 4)
    tau_val = round(0.2 + 0.05 * rng.random(), 4)
    beta_val = round(0.1 + 0.25 * rng.random(), 4)
    in_omega = H_val > tau_val and rho_val > 0.5

    telemetry_hash = "sha256:" + "".join(rng.choice("0123456789abcdef") for _ in range(64))
    reasoning_hash = "sha256:" + "".join(rng.choice("0123456789abcdef") for _ in range(64))

    return {
        "cycle": cycle,
        "timestamp": timestamp,
        "runner_type": runner_type,
        "slice_name": slice_name,
        "success": success,
        "depth": rng.randint(2, 8),
        "proof_hash": "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(16)),
        "H": H_val,
        "rho": rho_val,
        "rsi": rho_val,
        "tau": tau_val,
        "beta": beta_val,
        "in_omega": in_omega,
        "real_blocked": real_blocked,
        "sim_blocked": sim_blocked,
        "governance_aligned": governance_aligned,
        "governance_reason": "ALIGNED" if governance_aligned else "DIVERGED",
        "hard_ok": hard_ok,
        "abstained": abstained,
        "abstention_reason": "LOW_CONFIDENCE" if abstained else None,
        "mode": "SHADOW",
        "source": source,
        "telemetry_hash": telemetry_hash,
        "telemetry_snapshot_hash": telemetry_hash,
        "reasoning_graph_hash": reasoning_hash,
        "proof_dag_size": rng.randint(50, 150),
        "snapshot_hash": telemetry_hash,
    }


def make_p4_divergence_log_record(cycle: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a dictionary matching p4_divergence_log.schema.json.
    """
    rng = _rng_for("p4_divergence_log", cycle, seed)
    twin_snapshot = make_real_telemetry_snapshot(
        cycle,
        source="SHADOW_TWIN",
        seed=None if seed is None else seed * 2,
    )
    real_snapshot = make_real_telemetry_snapshot(
        cycle,
        source="REAL_RUNNER",
        seed=None if seed is None else seed * 2 + 1,
    )

    twin_delta_p = round(0.01 + 0.02 * rng.random(), 6)
    real_delta_p = twin_delta_p + round(rng.uniform(-0.005, 0.005), 6)
    divergence = abs(real_delta_p - twin_delta_p)
    divergence_pct = round(divergence / max(abs(real_delta_p), 0.001), 6)

    if divergence_pct < 0.05:
        severity = "NONE"
    elif divergence_pct < 0.1:
        severity = "INFO"
    elif divergence_pct < 0.2:
        severity = "WARN"
    else:
        severity = "CRITICAL"

    twin_state = {
        "H": twin_snapshot["H"],
        "rho": twin_snapshot["rho"],
        "tau": twin_snapshot["tau"],
        "beta": twin_snapshot["beta"],
        "in_omega": twin_snapshot["in_omega"],
        "success": twin_snapshot["success"],
    }
    real_state = {
        "H": real_snapshot["H"],
        "rho": real_snapshot["rho"],
        "tau": real_snapshot["tau"],
        "beta": real_snapshot["beta"],
        "in_omega": real_snapshot["in_omega"],
        "success": real_snapshot["success"],
    }

    return {
        "cycle": cycle,
        "timestamp": _timestamp_for_cycle(cycle),
        # Compatibility: evidence plotters expect delta_p/delta_p_percent.
        "delta_p": round(divergence, 6),
        "delta_p_percent": round(divergence * 100, 6),
        "twin_delta_p": twin_delta_p,
        "real_delta_p": real_delta_p,
        "divergence": round(divergence, 6),
        "divergence_pct": divergence_pct,
        "severity": severity,
        "twin_state": twin_state,
        "real_state": real_state,
        "state_divergences": {
            "H_diff": round(abs(twin_state["H"] - real_state["H"]), 6),
            "rho_diff": round(abs(twin_state["rho"] - real_state["rho"]), 6),
            "tau_diff": round(abs(twin_state["tau"] - real_state["tau"]), 6),
            "beta_diff": round(abs(twin_state["beta"] - real_state["beta"]), 6),
            "omega_mismatch": twin_state["in_omega"] != real_state["in_omega"],
            "success_mismatch": twin_state["success"] != real_state["success"],
        },
        "telemetry_snapshot_hash": real_snapshot["telemetry_snapshot_hash"],
        "streak_length": rng.randint(0, 3),
    }


def make_metrics_window(
    window_index: int = 0,
    *,
    window_size: int = 10,
    start_cycle: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a dictionary matching MetricsWindow.finalize() output.

    Includes a ``cycles`` entry with per-cycle data so tests can replay windows.
    """
    rng = _rng_for("metrics_window", window_index, seed)
    total_count = window_size
    start_cycle = start_cycle if start_cycle is not None else window_index * window_size + 1
    end_cycle = start_cycle + max(total_count - 1, 0)

    success_count = 0
    abstention_count = 0
    omega_count = 0
    hard_ok_count = 0
    blocked_count = 0
    rsi_sum = 0.0
    min_rsi = None
    max_rsi = None
    H_traj: List[float] = []
    rho_traj: List[float] = []
    success_traj: List[bool] = []
    per_cycle: List[Dict[str, Any]] = []

    for _ in range(total_count):
        success = rng.random() > 0.35
        abstained = (not success) and rng.random() > 0.7
        in_omega = rng.random() > 0.2
        hard_ok = rng.random() > 0.1
        rsi = round(0.5 + 0.45 * rng.random(), 4)
        blocked = rng.random() > 0.85
        H_val = round(0.4 + 0.5 * rng.random(), 4)

        success_count += 1 if success else 0
        abstention_count += 1 if abstained else 0
        omega_count += 1 if in_omega else 0
        hard_ok_count += 1 if hard_ok else 0
        blocked_count += 1 if blocked else 0
        rsi_sum += rsi
        min_rsi = rsi if min_rsi is None else min(min_rsi, rsi)
        max_rsi = rsi if max_rsi is None else max(max_rsi, rsi)

        H_traj.append(H_val)
        rho_traj.append(rsi)
        success_traj.append(success)

        per_cycle.append(
            {
                "success": success,
                "abstained": abstained,
                "in_omega": in_omega,
                "hard_ok": hard_ok,
                "rsi": rsi,
                "blocked": blocked,
                "H": H_val,
            }
        )

    mean_rsi = rsi_sum / total_count if total_count > 0 else 0.0
    block_rate = blocked_count / total_count if total_count > 0 else 0.0

    return {
        "window_index": window_index,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "total_count": total_count,
        "mode": "SHADOW",
        "success_metrics": {
            "success_count": success_count,
            "success_rate": round(success_count / total_count, 4) if total_count else 0.0,
        },
        "abstention_metrics": {
            "abstention_count": abstention_count,
            "abstention_rate": round(abstention_count / total_count, 4) if total_count else 0.0,
        },
        "safe_region_metrics": {
            "omega_count": omega_count,
            "omega_occupancy": round(omega_count / total_count, 4) if total_count else 0.0,
        },
        "hard_mode_metrics": {
            "hard_ok_count": hard_ok_count,
            "hard_ok_rate": round(hard_ok_count / total_count, 4) if total_count else 0.0,
        },
        "stability_metrics": {
            "mean_rsi": round(mean_rsi, 4),
            "min_rsi": round(min_rsi or 0.0, 4),
            "max_rsi": round(max_rsi or 0.0, 4),
        },
        "block_metrics": {
            "blocked_count": blocked_count,
            "block_rate": round(block_rate, 4),
        },
        "tda_inputs": {
            "H_trajectory": H_traj,
            "rho_trajectory": rho_traj,
            "success_trajectory": success_traj,
        },
        "cycles": per_cycle,
    }


def make_tda_window(
    window_index: int = 0,
    *,
    length: int = 12,
    start_cycle: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a dictionary for aggregated TDA metrics with SNS/PCS/DRS/HSS sections.
    """
    rng = _rng_for("tda_window", window_index, seed)
    length = max(1, length)
    start_cycle = start_cycle if start_cycle is not None else window_index * length + 1
    end_cycle = start_cycle + length - 1

    sns_values = [round(0.15 + 0.45 * rng.random(), 6) for _ in range(length)]
    pcs_values = [round(0.55 + 0.4 * rng.random(), 6) for _ in range(length)]
    hss_values = [round(0.5 + 0.4 * rng.random(), 6) for _ in range(length)]
    drs_values = [round(0.02 + 0.15 * rng.random(), 6) for _ in range(length)]

    def stats(values: List[float]) -> Dict[str, float]:
        mean = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return {"mean": mean, "min": min_val, "max": max_val, "std": std}

    sns_stats = stats(sns_values)
    pcs_stats = stats(pcs_values)
    hss_stats = stats(hss_values)
    drs_stats = stats(drs_values)

    sns_anomaly_count = sum(1 for v in sns_values if v > 0.6)
    sns_elevated_count = sum(1 for v in sns_values if v > 0.4)
    pcs_low_count = sum(1 for v in pcs_values if v < 0.6)
    pcs_incoherent_count = sum(1 for v in pcs_values if v < 0.4)
    hss_degradation_count = sum(1 for v in hss_values if v < 0.4)
    hss_unstable_count = sum(1 for v in hss_values if v < 0.6)
    envelope_in_count = sum(
        1
        for s, p, h in zip(sns_values, pcs_values, hss_values)
        if s <= 0.4 and p >= 0.6 and h >= 0.6
    )
    envelope_occupancy = envelope_in_count / length

    return {
        "schema_version": "1.0.0",
        "window_index": window_index,
        "window_start_cycle": start_cycle,
        "window_end_cycle": end_cycle,
        "mode": "SHADOW",
        "sns": {
            "mean": round(sns_stats["mean"], 6),
            "max": round(sns_stats["max"], 6),
            "min": round(sns_stats["min"], 6),
            "std": round(sns_stats["std"], 6),
            "anomaly_count": sns_anomaly_count,
            "elevated_count": sns_elevated_count,
        },
        "pcs": {
            "mean": round(pcs_stats["mean"], 6),
            "max": round(pcs_stats["max"], 6),
            "min": round(pcs_stats["min"], 6),
            "std": round(pcs_stats["std"], 6),
            "low_coherence_count": pcs_low_count,
            "incoherent_count": pcs_incoherent_count,
        },
        "hss": {
            "mean": round(hss_stats["mean"], 6),
            "max": round(hss_stats["max"], 6),
            "min": round(hss_stats["min"], 6),
            "std": round(hss_stats["std"], 6),
            "degradation_count": hss_degradation_count,
            "unstable_count": hss_unstable_count,
            "betti_snapshot": {
                "b0": rng.randint(1, 3),
                "b1": rng.randint(0, 2),
                "b2": 0,
            },
        },
        "drs": {
            "mean": round(drs_stats["mean"], 6),
            "max": round(drs_stats["max"], 6),
            "min": round(drs_stats["min"], 6),
            "std": round(drs_stats["std"], 6),
        },
        "envelope": {
            "occupancy_rate": round(envelope_occupancy, 6),
            "exit_count": length - envelope_in_count,
            "max_exit_streak": max(0, length - envelope_in_count),
        },
        "red_flags": {
            "tda_sns_anomaly": sns_anomaly_count,
            "tda_pcs_collapse": pcs_incoherent_count,
            "tda_hss_degradation": hss_degradation_count,
            "tda_envelope_exit": length - envelope_in_count,
        },
        "trajectories": {
            "sns": sns_values,
            "pcs": pcs_values,
            "hss": hss_values,
            "drs": drs_values,
        },
    }


def make_summary_payload(
    *,
    run_id: Optional[str] = None,
    total_cycles: int = 100,
    slice_name: str = "arithmetic_simple",
    runner_type: str = "u2",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a dict matching SummarySchema.to_dict()."""
    rng = _rng_for("summary_payload", total_cycles, seed)
    run_id = run_id or f"fl_factory_{total_cycles}_{rng.randint(1000, 9999)}"
    start_ts = _timestamp_for_cycle(0)
    duration_seconds = total_cycles * 0.8
    end_ts = (_BASE_TIMESTAMP + timedelta(seconds=duration_seconds)).isoformat().replace("+00:00", "Z")

    success_rate = round(0.7 + 0.2 * rng.random(), 4)
    rsi_mean = round(0.6 + 0.3 * rng.random(), 4)
    omega = round(0.7 + 0.2 * rng.random(), 4)
    hard_ok = round(0.8 + 0.15 * rng.random(), 4)

    return {
        "schema": "first-light-summary/1.0.0",
        "run_id": run_id,
        "mode": "SHADOW",
        "config": {
            "slice_name": slice_name,
            "runner_type": runner_type,
            "total_cycles": total_cycles,
            "tau_0": round(0.18 + 0.05 * rng.random(), 4),
        },
        "execution": {
            "start_time": start_ts,
            "end_time": end_ts,
            "duration_seconds": duration_seconds,
            "cycles_completed": total_cycles,
        },
        "success_criteria": {
            "u2_success_rate": {
                "target": 0.75,
                "actual": success_rate,
                "passed": success_rate >= 0.75,
            },
            "omega_occupancy": {
                "target": 0.9,
                "actual": omega,
                "passed": omega >= 0.9,
            },
            "hard_ok_rate": {
                "target": 0.85,
                "actual": hard_ok,
                "passed": hard_ok >= 0.85,
            },
        },
        "red_flag_summary": {
            "total_observations": rng.randint(0, 5),
            "by_type": {"RSI_COLLAPSE": rng.randint(0, 2)},
            "hypothetical_aborts": 0,
        },
    }


def make_stability_report_payload(
    *,
    run_id: Optional[str] = None,
    total_cycles: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a dict for stability_report.json using summary + metrics factories."""
    summary = make_summary_payload(run_id=run_id, total_cycles=total_cycles, seed=seed)
    rng = _rng_for("stability_report", total_cycles, seed)
    metrics = {
        "delta_p": {
            "success_final": round(0.01 * rng.random(), 4),
            "abstention_final": round(0.005 * rng.random(), 4),
        },
        "rsi": {
            "mean": summary["success_criteria"]["u2_success_rate"]["actual"],
            "min": round(0.4 + 0.2 * rng.random(), 4),
            "max": round(0.8 + 0.2 * rng.random(), 4),
            "std": round(0.05 * rng.random(), 4),
        },
        "omega": {
            "occupancy_rate": summary["success_criteria"]["omega_occupancy"]["actual"],
            "exit_count": rng.randint(0, 3),
            "max_exit_streak": rng.randint(0, 2),
        },
        "hard_mode": {
            "ok_rate": summary["success_criteria"]["hard_ok_rate"]["actual"],
            "fail_count": rng.randint(0, 3),
            "max_fail_streak": rng.randint(0, 2),
        },
    }

    return {
        "schema_version": "1.0.0",
        "run_id": summary["run_id"],
        "config": summary["config"],
        "timing": summary["execution"],
        "metrics": metrics,
        "criteria_evaluation": {
            "all_passed": all(c["passed"] for c in summary["success_criteria"].values()),
            "criteria": [
                {
                    "name": name,
                    "threshold": crit["target"],
                    "actual": crit["actual"],
                    "passed": crit["passed"],
                }
                for name, crit in summary["success_criteria"].items()
            ],
        },
        "mode": "SHADOW",
    }


def make_cal_exp1_report(
    *,
    cycles: int = 200,
    window_size: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic CAL-EXP-1 report (evidence-pack ready)."""
    rng = _rng_for("cal_exp1_report", cycles + window_size, seed)
    total_cycles = max(0, cycles)
    window_size = max(1, window_size)
    window_count = max(1, (total_cycles + window_size - 1) // window_size)
    pattern_vocab = ["NONE", "DRIFT", "OSCILLATION", "PLATEAU"]
    windows: List[Dict[str, Any]] = []

    for window_index in range(window_count):
        window_start = window_index * window_size
        window_end = min(total_cycles - 1, (window_index + 1) * window_size - 1)
        cycles_in_window = max(0, window_end - window_start + 1)

        divergence_rate_est = 0.05 + 0.15 * rng.random()
        divergence_count = int(round(divergence_rate_est * cycles_in_window))
        divergence_count = max(0, min(divergence_count, cycles_in_window))
        divergence_rate = round(divergence_count / cycles_in_window, 6) if cycles_in_window else 0.0

        mean_delta_p = round(rng.uniform(-0.02, 0.02), 6)
        delta_bias = mean_delta_p
        delta_variance = round(abs(mean_delta_p) * 0.1 + 0.001 * rng.random(), 6)

        windows.append(
            {
                "window_index": window_index,
                "window_start": window_start,
                "window_end": window_end,
                "cycles_in_window": cycles_in_window,
                "divergence_count": divergence_count,
                "divergence_rate": divergence_rate,
                "mean_delta_p": mean_delta_p,
                "delta_bias": delta_bias,
                "delta_variance": delta_variance,
                "phase_lag_xcorr": round(rng.uniform(-0.1, 0.1), 6),
                "pattern_tag": rng.choice(pattern_vocab),
            }
        )

    final_divergence_rate = windows[-1]["divergence_rate"] if windows else None
    final_delta_bias = windows[-1]["delta_bias"] if windows else None
    mean_divergence_over_run = (
        round(sum(w["divergence_rate"] for w in windows) / len(windows), 6) if windows else None
    )
    pattern_progression = [w["pattern_tag"] for w in windows]

    run_id = f"cal_exp1_{seed or _DEFAULT_FACTORY_SEED}"
    timestamp = _timestamp_for_cycle(total_cycles)

    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "run_id": run_id,
        "timestamp": timestamp,
        "total_cycles": total_cycles,
        "window_size": window_size,
        "window_count": window_count,
        "windows": windows,
        "_calibration_note": (
            "Per-window metrics for P5 calibration (factory). "
            "Observational only; no gating."
        ),
        "summary": {
            "final_divergence_rate": final_divergence_rate,
            "final_delta_bias": final_delta_bias,
            "mean_divergence_over_run": mean_divergence_over_run,
            "pattern_progression": pattern_progression,
        },
        "provisional_verdict": {
            "verdict": "DEFER",
            "reason": "Factory scaffold: advisory only.",
            "enforcement": "SHADOW_ONLY",
            "_note": "This verdict is observational. No gating occurs.",
        },
    }


def make_cal_exp2_report(
    learning_rates: Optional[List[float]] = None,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic CAL-EXP-2 scaffold report."""
    learning_rates = learning_rates or [0.05, 0.1, 0.2]
    rng = _rng_for("cal_exp2_report", len(learning_rates), seed)
    trials: List[Dict[str, Any]] = []

    for lr in learning_rates:
        trajectory = []
        for idx in range(1, 6):
            base = max(0.0, 0.2 - lr * 0.5)
            noise = rng.uniform(-0.02, 0.02)
            trajectory.append(round(max(0.0, base + noise + idx * 0.005), 6))
        trials.append({"lr": lr, "divergence_trajectory": trajectory})

    return {
        "schema_version": "0.1.0",
        "mode": "SHADOW",
        "generated_at": _timestamp_for_cycle(0),
        "timestamp": _timestamp_for_cycle(0),
        "trials": trials,
    }


def make_cal_exp3_report(
    *,
    cycles: int = 300,
    change_after: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic CAL-EXP-3 scaffold report."""
    rng = _rng_for("cal_exp3_report", cycles + change_after, seed)

    def _rate(sample_size: int) -> Dict[str, Any]:
        return {
            "divergence_rate": round(rng.uniform(0.05, 0.25), 6),
            "sample_size": sample_size,
        }

    return {
        "schema_version": "0.1.0",
        "mode": "SHADOW",
        "generated_at": _timestamp_for_cycle(0),
        "params": {
            "cycles": cycles,
            "change_after": change_after,
            "delta_H": 0.2,
            "seed": seed or _DEFAULT_FACTORY_SEED,
        },
        "pre_change": _rate(change_after),
        "post_change": _rate(max(0, cycles - change_after)),
        "timestamp": _timestamp_for_cycle(cycles),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def emit_cal_exp_reports_for_evidence_pack(
    pack_dir: Path,
    *,
    cal_exp1_report: Optional[Dict[str, Any]] = None,
    cal_exp2_report: Optional[Dict[str, Any]] = None,
    cal_exp3_report: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Emit CAL-EXP report JSON files using the evidence-pack layout.

    Layout (relative to pack_dir):
      - calibration/cal_exp1_report.json
      - calibration/cal_exp2_report.json
      - calibration/cal_exp3_report.json
    """
    cal_dir = pack_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    report_1 = cal_exp1_report if cal_exp1_report is not None else make_cal_exp1_report(seed=seed)
    report_2 = cal_exp2_report if cal_exp2_report is not None else make_cal_exp2_report(seed=seed)
    report_3 = cal_exp3_report if cal_exp3_report is not None else make_cal_exp3_report(seed=seed)

    path_1 = cal_dir / "cal_exp1_report.json"
    path_2 = cal_dir / "cal_exp2_report.json"
    path_3 = cal_dir / "cal_exp3_report.json"

    _write_json(path_1, report_1)
    _write_json(path_2, report_2)
    _write_json(path_3, report_3)

    return {
        "cal_exp1_report": path_1,
        "cal_exp2_report": path_2,
        "cal_exp3_report": path_3,
    }


def emit_cal_exp1_report_for_evidence_pack(
    pack_dir: Path,
    *,
    cycles: int = 200,
    window_size: int = 50,
    seed: Optional[int] = None,
) -> Path:
    """Emit calibration/cal_exp1_report.json under pack_dir."""
    report = make_cal_exp1_report(cycles=cycles, window_size=window_size, seed=seed)
    report_path = pack_dir / "calibration" / "cal_exp1_report.json"
    _write_json(report_path, report)
    return report_path


def emit_cal_exp2_report_for_evidence_pack(
    pack_dir: Path,
    learning_rates: Optional[List[float]] = None,
    *,
    seed: Optional[int] = None,
) -> Path:
    """Emit calibration/cal_exp2_report.json under pack_dir."""
    report = make_cal_exp2_report(learning_rates, seed=seed)
    report_path = pack_dir / "calibration" / "cal_exp2_report.json"
    _write_json(report_path, report)
    return report_path


def emit_cal_exp3_report_for_evidence_pack(
    pack_dir: Path,
    *,
    cycles: int = 300,
    change_after: int = 100,
    seed: Optional[int] = None,
) -> Path:
    """Emit calibration/cal_exp3_report.json under pack_dir."""
    report = make_cal_exp3_report(cycles=cycles, change_after=change_after, seed=seed)
    report_path = pack_dir / "calibration" / "cal_exp3_report.json"
    _write_json(report_path, report)
    return report_path
