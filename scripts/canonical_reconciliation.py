#!/usr/bin/env python3
"""
CAL-EXP-1 Canonical Reconciliation Script

This script recomputes ALL metrics from raw artifacts (synthetic_trace.jsonl)
by re-running the TwinRunner to reconstruct twin predictions and divergence
decomposition.

Purpose: Produce a single canonical verdict for UPGRADE-1 with full decomposition.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Run directories
RUNS = {
    "baseline_seed42": {
        "path": "results/p5_cal_exp1/synthetic_seed42/cal_exp1_20251212_091922",
        "seed": 42,
        "lr_overrides": None,
    },
    "baseline_seed43": {
        "path": "results/p5_cal_exp1/synthetic_seed43/cal_exp1_20251212_091945",
        "seed": 43,
        "lr_overrides": None,
    },
    "upgrade1_seed42": {
        "path": "results/p5_cal_exp1/upgrade1_seed42/cal_exp1_20251212_095143",
        "seed": 42,
        "lr_overrides": {"H": 0.2, "rho": 0.15, "tau": 0.02, "beta": 0.12},
    },
    "upgrade1_seed43": {
        "path": "results/p5_cal_exp1/upgrade1_seed43/cal_exp1_20251212_095159",
        "seed": 43,
        "lr_overrides": {"H": 0.2, "rho": 0.15, "tau": 0.02, "beta": 0.12},
    },
}

WINDOW_SIZE = 50


@dataclass
class CycleData:
    """Parsed cycle data from synthetic_trace.jsonl."""
    cycle: int
    H: float
    rho: float
    tau: float
    beta: float
    success: bool
    in_omega: bool
    blocked: bool
    hard_ok: bool


@dataclass
class TwinPrediction:
    """Twin prediction for a cycle."""
    cycle: int
    twin_H: float
    twin_rho: float
    twin_tau: float
    twin_beta: float
    predicted_success: bool
    predicted_in_omega: bool
    predicted_blocked: bool
    predicted_hard_ok: bool


@dataclass
class DivergenceDecomposition:
    """Full divergence decomposition for a cycle."""
    cycle: int
    # State deltas
    H_delta: float
    rho_delta: float
    tau_delta: float
    beta_delta: float
    delta_p: float
    # Binary divergences
    success_diverged: bool
    omega_diverged: bool
    blocked_diverged: bool
    hard_ok_diverged: bool
    state_diverged: bool  # delta_p > 0.05
    # Aggregate
    any_diverged: bool


def load_trace(path: str) -> List[CycleData]:
    """Load synthetic_trace.jsonl and parse into CycleData."""
    cycles = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            cycles.append(CycleData(
                cycle=data["cycle"],
                H=data["H"],
                rho=data["rho"],
                tau=data["tau"],
                beta=data["beta"],
                success=data["success"],
                in_omega=data["in_omega"],
                blocked=data["blocked"],
                hard_ok=data["hard_ok"],
            ))
    return cycles


def run_twin(
    real_cycles: List[CycleData],
    seed: int,
    lr_overrides: Optional[Dict[str, float]],
) -> List[TwinPrediction]:
    """
    Re-run TwinRunner to reconstruct twin predictions.

    This replicates the exact logic from runner_p4.py TwinRunner.
    """
    import random

    rng = random.Random(seed)
    lr = 0.1
    noise = 0.02

    # Extract per-component LRs
    lr_H = (lr_overrides or {}).get("H", lr)
    lr_rho = (lr_overrides or {}).get("rho", lr)
    lr_tau = (lr_overrides or {}).get("tau", lr * 0.5)
    lr_beta = (lr_overrides or {}).get("beta", lr)

    # Initialize twin state
    twin_H = 0.5
    twin_rho = 0.7
    twin_tau = 0.20
    twin_beta = 0.1

    predictions = []

    for real in real_cycles:
        # Predict based on current twin state (BEFORE update)
        predicted_in_omega = twin_H > twin_tau and twin_rho > 0.5
        predicted_blocked = twin_beta > 0.7

        # Success prediction (matches RealTelemetryAdapter logic)
        base_success_prob = 0.5 + 0.3 * twin_H + 0.2 * twin_rho - 0.3 * twin_beta
        base_success_prob = max(0.1, min(0.95, base_success_prob))
        predicted_success = base_success_prob > 0.5

        # Hard OK prediction
        hard_ok_prob = 0.9 + 0.1 * twin_rho
        predicted_hard_ok = hard_ok_prob > 0.5

        predictions.append(TwinPrediction(
            cycle=real.cycle,
            twin_H=twin_H,
            twin_rho=twin_rho,
            twin_tau=twin_tau,
            twin_beta=twin_beta,
            predicted_success=predicted_success,
            predicted_in_omega=predicted_in_omega,
            predicted_blocked=predicted_blocked,
            predicted_hard_ok=predicted_hard_ok,
        ))

        # Update twin state based on real observation (AFTER prediction)
        twin_H = twin_H * (1 - lr_H) + real.H * lr_H
        twin_H += rng.gauss(0, noise)
        twin_H = max(0.0, min(1.0, twin_H))

        twin_rho = twin_rho * (1 - lr_rho) + real.rho * lr_rho
        twin_rho += rng.gauss(0, noise)
        twin_rho = max(0.0, min(1.0, twin_rho))

        twin_tau = twin_tau * (1 - lr_tau) + real.tau * lr_tau
        twin_tau += rng.gauss(0, noise * 0.5)
        twin_tau = max(0.0, min(1.0, twin_tau))

        # Beta learns from blocked status
        if real.blocked:
            twin_beta = min(1.0, twin_beta + 0.05 * lr_beta / max(lr, 1e-6))
        else:
            twin_beta = max(0.0, twin_beta - 0.01 * lr_beta / max(lr, 1e-6) + rng.gauss(0, noise))

    return predictions


def compute_decomposition(
    real_cycles: List[CycleData],
    twin_predictions: List[TwinPrediction],
) -> List[DivergenceDecomposition]:
    """Compute full divergence decomposition."""
    decompositions = []

    for real, twin in zip(real_cycles, twin_predictions):
        H_delta = abs(real.H - twin.twin_H)
        rho_delta = abs(real.rho - twin.twin_rho)
        tau_delta = abs(real.tau - twin.twin_tau)
        beta_delta = abs(real.beta - twin.twin_beta)
        delta_p = (H_delta + rho_delta + tau_delta + beta_delta) / 4.0

        success_diverged = real.success != twin.predicted_success
        omega_diverged = real.in_omega != twin.predicted_in_omega
        blocked_diverged = real.blocked != twin.predicted_blocked
        hard_ok_diverged = real.hard_ok != twin.predicted_hard_ok
        state_diverged = delta_p > 0.05

        any_diverged = (
            success_diverged or
            omega_diverged or
            blocked_diverged or
            hard_ok_diverged or
            state_diverged
        )

        decompositions.append(DivergenceDecomposition(
            cycle=real.cycle,
            H_delta=H_delta,
            rho_delta=rho_delta,
            tau_delta=tau_delta,
            beta_delta=beta_delta,
            delta_p=delta_p,
            success_diverged=success_diverged,
            omega_diverged=omega_diverged,
            blocked_diverged=blocked_diverged,
            hard_ok_diverged=hard_ok_diverged,
            state_diverged=state_diverged,
            any_diverged=any_diverged,
        ))

    return decompositions


def compute_window_metrics(
    decompositions: List[DivergenceDecomposition],
    window_start: int,
    window_end: int,
) -> Dict[str, Any]:
    """Compute metrics for a window."""
    window = [d for d in decompositions if window_start <= d.cycle <= window_end]
    n = len(window)

    if n == 0:
        return {}

    # Aggregate divergence rates
    any_diverged_count = sum(1 for d in window if d.any_diverged)
    success_diverged_count = sum(1 for d in window if d.success_diverged)
    omega_diverged_count = sum(1 for d in window if d.omega_diverged)
    blocked_diverged_count = sum(1 for d in window if d.blocked_diverged)
    hard_ok_diverged_count = sum(1 for d in window if d.hard_ok_diverged)
    state_diverged_count = sum(1 for d in window if d.state_diverged)

    # Mean delta_p
    mean_delta_p = sum(d.delta_p for d in window) / n

    # Per-component mean absolute errors
    mean_H_error = sum(d.H_delta for d in window) / n
    mean_rho_error = sum(d.rho_delta for d in window) / n
    mean_tau_error = sum(d.tau_delta for d in window) / n
    mean_beta_error = sum(d.beta_delta for d in window) / n

    return {
        "window_start": window_start,
        "window_end": window_end,
        "cycle_count": n,
        "divergence_rate": any_diverged_count / n,
        "success_divergence_rate": success_diverged_count / n,
        "omega_divergence_rate": omega_diverged_count / n,
        "blocked_divergence_rate": blocked_diverged_count / n,
        "hard_ok_divergence_rate": hard_ok_diverged_count / n,
        "state_divergence_rate": state_diverged_count / n,
        "mean_delta_p": mean_delta_p,
        "mean_H_error": mean_H_error,
        "mean_rho_error": mean_rho_error,
        "mean_tau_error": mean_tau_error,
        "mean_beta_error": mean_beta_error,
        # Phase lag: null with reason_code per METRIC_DEFINITIONS v1.1.0
        "phase_lag_xcorr": None,
        "phase_lag_xcorr_reason_code": "NOT_COMPUTED_DASHBOARD_ONLY",
    }


def compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of file."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    print("=" * 80)
    print("CAL-EXP-1 CANONICAL RECONCILIATION")
    print("=" * 80)
    print()

    root = Path(__file__).parent.parent

    results = {}

    for run_name, run_config in RUNS.items():
        print(f"Processing: {run_name}")
        run_path = root / run_config["path"]
        trace_path = run_path / "synthetic_trace.jsonl"

        if not trace_path.exists():
            print(f"  ERROR: {trace_path} not found")
            continue

        # Compute file hash for provenance
        trace_hash = compute_file_hash(str(trace_path))
        print(f"  Trace hash: {trace_hash}")

        # Load real telemetry
        real_cycles = load_trace(str(trace_path))
        print(f"  Loaded {len(real_cycles)} cycles")

        # Re-run twin to reconstruct predictions
        twin_predictions = run_twin(
            real_cycles,
            run_config["seed"],
            run_config["lr_overrides"],
        )
        print(f"  Generated {len(twin_predictions)} twin predictions")

        # Compute full decomposition
        decompositions = compute_decomposition(real_cycles, twin_predictions)

        # Compute per-window metrics
        windows = []
        for i in range(0, len(decompositions), WINDOW_SIZE):
            window_start = i + 1
            window_end = min(i + WINDOW_SIZE, len(decompositions))
            metrics = compute_window_metrics(decompositions, window_start, window_end)
            if metrics:
                windows.append(metrics)

        # Compute full-run aggregates
        n = len(decompositions)
        full_run = {
            "total_cycles": n,
            "divergence_rate": sum(1 for d in decompositions if d.any_diverged) / n,
            "success_divergence_rate": sum(1 for d in decompositions if d.success_diverged) / n,
            "omega_divergence_rate": sum(1 for d in decompositions if d.omega_diverged) / n,
            "blocked_divergence_rate": sum(1 for d in decompositions if d.blocked_diverged) / n,
            "hard_ok_divergence_rate": sum(1 for d in decompositions if d.hard_ok_diverged) / n,
            "state_divergence_rate": sum(1 for d in decompositions if d.state_diverged) / n,
            "mean_delta_p": sum(d.delta_p for d in decompositions) / n,
            "mean_H_error": sum(d.H_delta for d in decompositions) / n,
            "mean_rho_error": sum(d.rho_delta for d in decompositions) / n,
            "mean_tau_error": sum(d.tau_delta for d in decompositions) / n,
            "mean_beta_error": sum(d.beta_delta for d in decompositions) / n,
            # Phase lag: null with reason_code per METRIC_DEFINITIONS v1.1.0
            "phase_lag_xcorr": None,
            "phase_lag_xcorr_reason_code": "NOT_COMPUTED_DASHBOARD_ONLY",
        }

        results[run_name] = {
            "config": {
                "seed": run_config["seed"],
                "lr_overrides": run_config["lr_overrides"],
                "path": run_config["path"],
            },
            "provenance": {
                "trace_path": str(trace_path.relative_to(root)),
                "trace_hash": trace_hash,
            },
            "windows": windows,
            "full_run": full_run,
        }

        print(f"  Full-run divergence_rate: {full_run['divergence_rate']:.4f}")
        print(f"  Full-run mean_delta_p: {full_run['mean_delta_p']:.6f}")
        print()

    # Compute comparisons
    print("=" * 80)
    print("DECOMPOSITION COMPARISON")
    print("=" * 80)
    print()

    comparisons = {}

    for seed in [42, 43]:
        baseline_key = f"baseline_seed{seed}"
        upgrade_key = f"upgrade1_seed{seed}"

        if baseline_key not in results or upgrade_key not in results:
            continue

        baseline = results[baseline_key]["full_run"]
        upgrade = results[upgrade_key]["full_run"]

        comparison = {
            "seed": seed,
            "baseline": baseline,
            "upgrade1": upgrade,
            "deltas": {
                "divergence_rate": upgrade["divergence_rate"] - baseline["divergence_rate"],
                "success_divergence_rate": upgrade["success_divergence_rate"] - baseline["success_divergence_rate"],
                "omega_divergence_rate": upgrade["omega_divergence_rate"] - baseline["omega_divergence_rate"],
                "blocked_divergence_rate": upgrade["blocked_divergence_rate"] - baseline["blocked_divergence_rate"],
                "state_divergence_rate": upgrade["state_divergence_rate"] - baseline["state_divergence_rate"],
                "mean_delta_p": upgrade["mean_delta_p"] - baseline["mean_delta_p"],
                "mean_H_error": upgrade["mean_H_error"] - baseline["mean_H_error"],
                "mean_rho_error": upgrade["mean_rho_error"] - baseline["mean_rho_error"],
                "mean_tau_error": upgrade["mean_tau_error"] - baseline["mean_tau_error"],
                "mean_beta_error": upgrade["mean_beta_error"] - baseline["mean_beta_error"],
            },
        }
        comparisons[f"seed{seed}"] = comparison

        print(f"Seed={seed} Decomposition:")
        print(f"  {'Metric':<30} {'BASELINE':>12} {'UPGRADE-1':>12} {'Delta':>12}")
        print("  " + "-" * 66)
        for metric in ["divergence_rate", "success_divergence_rate", "omega_divergence_rate",
                       "blocked_divergence_rate", "state_divergence_rate", "mean_delta_p",
                       "mean_H_error", "mean_rho_error", "mean_tau_error", "mean_beta_error"]:
            b = baseline[metric]
            u = upgrade[metric]
            d = comparison["deltas"][metric]
            print(f"  {metric:<30} {b:>12.6f} {u:>12.6f} {d:>+12.6f}")
        print()

    # Determine dominant contributor
    print("=" * 80)
    print("DOMINANT DIVERGENCE CONTRIBUTOR ANALYSIS")
    print("=" * 80)
    print()

    for seed in [42, 43]:
        baseline_key = f"baseline_seed{seed}"
        if baseline_key not in results:
            continue

        baseline = results[baseline_key]["full_run"]

        # Compute contribution percentages
        total_div = baseline["divergence_rate"]
        if total_div > 0:
            success_contrib = baseline["success_divergence_rate"] / total_div * 100
            omega_contrib = baseline["omega_divergence_rate"] / total_div * 100
            blocked_contrib = baseline["blocked_divergence_rate"] / total_div * 100
            state_contrib = baseline["state_divergence_rate"] / total_div * 100
        else:
            success_contrib = omega_contrib = blocked_contrib = state_contrib = 0

        print(f"Seed={seed} BASELINE divergence_rate composition:")
        print(f"  success_divergence contributes: {success_contrib:.1f}%")
        print(f"  omega_divergence contributes:   {omega_contrib:.1f}%")
        print(f"  blocked_divergence contributes: {blocked_contrib:.1f}%")
        print(f"  state_divergence contributes:   {state_contrib:.1f}%")

        # Identify dominant
        dominant = max([
            ("success", baseline["success_divergence_rate"]),
            ("omega", baseline["omega_divergence_rate"]),
            ("blocked", baseline["blocked_divergence_rate"]),
            ("state", baseline["state_divergence_rate"]),
        ], key=lambda x: x[1])
        print(f"  DOMINANT: {dominant[0]} ({dominant[1]:.4f})")
        print()

    # Build final JSON artifact
    output = {
        "schema_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "metric_definitions": {
            "divergence_rate": "count(any_diverged) / total_cycles",
            "success_divergence_rate": "count(real.success != twin.predicted_success) / total_cycles",
            "omega_divergence_rate": "count(real.in_omega != twin.predicted_in_omega) / total_cycles",
            "blocked_divergence_rate": "count(real.blocked != twin.predicted_blocked) / total_cycles",
            "state_divergence_rate": "count(delta_p > 0.05) / total_cycles",
            "mean_delta_p": "mean((H_delta + rho_delta + tau_delta + beta_delta) / 4)",
            "mean_H_error": "mean(|real.H - twin.H|)",
            "mean_rho_error": "mean(|real.rho - twin.rho|)",
            "mean_tau_error": "mean(|real.tau - twin.tau|)",
            "mean_beta_error": "mean(|real.beta - twin.beta|)",
        },
        "thresholds": {
            "state_divergence_threshold": 0.05,
            "status": "PROVISIONAL",
        },
        "runs": results,
        "comparisons": comparisons,
    }

    # Save JSON
    output_dir = root / "results" / "p5_reconciliation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "upgrade1_reconciliation.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    # Determine verdict
    print()
    print("=" * 80)
    print("CANONICAL VERDICT")
    print("=" * 80)
    print()

    # Single metric of truth: mean_delta_p
    seed42_improved = comparisons.get("seed42", {}).get("deltas", {}).get("mean_delta_p", 0) < 0
    seed43_improved = comparisons.get("seed43", {}).get("deltas", {}).get("mean_delta_p", 0) < 0

    print("Primary Metric: mean_delta_p (state tracking error)")
    print()
    print(f"  Seed=42: {'IMPROVED' if seed42_improved else 'WORSENED'} (delta={comparisons.get('seed42', {}).get('deltas', {}).get('mean_delta_p', 0):+.6f})")
    print(f"  Seed=43: {'IMPROVED' if seed43_improved else 'WORSENED'} (delta={comparisons.get('seed43', {}).get('deltas', {}).get('mean_delta_p', 0):+.6f})")
    print()

    if seed42_improved and seed43_improved:
        verdict = "PROCEED"
        verdict_reason = "mean_delta_p improved in BOTH seeds"
    elif seed42_improved or seed43_improved:
        verdict = "MIXED"
        verdict_reason = "mean_delta_p improved in 1 seed, worsened in 1 seed"
    else:
        verdict = "ADJUST"
        verdict_reason = "mean_delta_p worsened in BOTH seeds"

    print(f"VERDICT: {verdict}")
    print(f"REASON: {verdict_reason}")
    print()

    # Note about divergence_rate saturation
    print("NOTE: divergence_rate is saturated near 1.0 because state_divergence_rate")
    print("dominates and mean_delta_p (~0.05) is at the threshold boundary.")
    print("Use mean_delta_p as the calibration objective, not divergence_rate.")
    print("=" * 80)

    # Save verdict to output
    output["verdict"] = {
        "decision": verdict,
        "reason": verdict_reason,
        "primary_metric": "mean_delta_p",
        "seed42_improved": seed42_improved,
        "seed43_improved": seed43_improved,
    }

    # Re-save with verdict
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    main()
