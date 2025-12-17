#!/usr/bin/env python3
"""
CAL-EXP-3: Baseline vs Learning-Enabled A/B Comparison

Two arms:
  - ARM A (baseline): learning_rate=0.0 (twin does not track real state)
  - ARM B (treatment): learning_rate=0.1 (twin learns from real observations)

All other configuration identical between arms.
Measures: delta_p (state tracking error) over time.

SHADOW MODE - observational only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Toolchain fingerprint
TOOLCHAIN_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


@dataclass
class CycleMetrics:
    """Metrics for a single cycle."""
    cycle: int
    H: float
    rho: float
    tau: float
    beta: float
    twin_H: float
    twin_rho: float
    twin_tau: float
    twin_beta: float
    delta_p: float
    success: bool
    predicted_success: bool
    in_omega: bool
    predicted_in_omega: bool
    blocked: bool
    predicted_blocked: bool


@dataclass
class WindowMetrics:
    """Aggregated metrics for a window."""
    window_id: int
    window_start: int
    window_end: int
    mean_delta_p: float
    divergence_rate: float
    success_divergence_rate: float


@dataclass
class ArmResult:
    """Result for one experiment arm."""
    arm_id: str
    learning_rate: float
    seed: int
    total_cycles: int
    final_mean_delta_p: float
    final_divergence_rate: float
    windows: List[WindowMetrics] = field(default_factory=list)
    cycle_metrics: List[CycleMetrics] = field(default_factory=list)


@dataclass
class RunConfig:
    """Complete run configuration for reproducibility."""
    experiment_id: str
    arm_id: str
    seed: int
    learning_rate: float
    noise_scale: float
    total_cycles: int
    window_size: int
    initial_H: float
    initial_rho: float
    initial_tau: float
    initial_beta: float
    toolchain_version: str
    toolchain_fingerprint: str
    timestamp: str


def compute_toolchain_fingerprint() -> str:
    """Compute deterministic toolchain fingerprint."""
    components = [
        f"python:{platform.python_version()}",
        f"platform:{platform.system()}",
        f"script:run_cal_exp_3.py",
        f"schema:{SCHEMA_VERSION}",
    ]
    content = "|".join(components)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class SyntheticTelemetry:
    """Generates synthetic telemetry for reproducible experiments."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._cycle = 0
        self._H = 0.5
        self._rho = 0.7
        self._tau = 0.20
        self._beta = 0.1

    def step(self) -> Dict[str, Any]:
        """Generate next cycle telemetry."""
        self._cycle += 1

        # Evolve state with drift and noise
        self._H += self._rng.gauss(0.002, 0.03)
        self._H = max(0.0, min(1.0, self._H))

        self._rho += self._rng.gauss(0.001, 0.02)
        self._rho = max(0.0, min(1.0, self._rho))

        self._tau += self._rng.gauss(0, 0.01)
        self._tau = max(0.1, min(0.3, self._tau))

        # Beta occasionally spikes
        if self._rng.random() < 0.05:
            self._beta = min(1.0, self._beta + 0.1)
        else:
            self._beta = max(0.0, self._beta - 0.01)

        # Compute derived signals
        in_omega = self._H > self._tau and self._rho > 0.5
        blocked = self._beta > 0.7

        success_prob = 0.5 + 0.3 * self._H + 0.2 * self._rho - 0.3 * self._beta
        success_prob = max(0.1, min(0.95, success_prob))
        success = self._rng.random() < success_prob

        return {
            "cycle": self._cycle,
            "H": self._H,
            "rho": self._rho,
            "tau": self._tau,
            "beta": self._beta,
            "success": success,
            "in_omega": in_omega,
            "blocked": blocked,
        }


class TwinModel:
    """Twin model with configurable learning rate."""

    def __init__(
        self,
        seed: int,
        learning_rate: float,
        noise_scale: float = 0.02,
    ):
        self._rng = random.Random(seed + 1000)  # Offset seed for twin
        self._lr = learning_rate
        self._noise = noise_scale

        # Initial state
        self._H = 0.5
        self._rho = 0.7
        self._tau = 0.20
        self._beta = 0.1

    def predict(self, real: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on current twin state."""
        predicted_in_omega = self._H > self._tau and self._rho > 0.5
        predicted_blocked = self._beta > 0.7

        success_prob = 0.5 + 0.3 * self._H + 0.2 * self._rho - 0.3 * self._beta
        success_prob = max(0.1, min(0.95, success_prob))
        predicted_success = success_prob > 0.5

        return {
            "twin_H": self._H,
            "twin_rho": self._rho,
            "twin_tau": self._tau,
            "twin_beta": self._beta,
            "predicted_success": predicted_success,
            "predicted_in_omega": predicted_in_omega,
            "predicted_blocked": predicted_blocked,
        }

    def update(self, real: Dict[str, Any]) -> None:
        """Update twin state based on real observation."""
        if self._lr == 0.0:
            # Baseline: no learning
            return

        # Blend toward real state
        self._H = self._H * (1 - self._lr) + real["H"] * self._lr
        self._H += self._rng.gauss(0, self._noise)
        self._H = max(0.0, min(1.0, self._H))

        self._rho = self._rho * (1 - self._lr) + real["rho"] * self._lr
        self._rho += self._rng.gauss(0, self._noise)
        self._rho = max(0.0, min(1.0, self._rho))

        lr_tau = self._lr * 0.5  # Slower tau learning
        self._tau = self._tau * (1 - lr_tau) + real["tau"] * lr_tau
        self._tau += self._rng.gauss(0, self._noise * 0.5)
        self._tau = max(0.0, min(1.0, self._tau))

        # Beta learns from blocked status
        if real["blocked"]:
            self._beta = min(1.0, self._beta + 0.05)
        else:
            self._beta = max(0.0, self._beta - 0.01 + self._rng.gauss(0, self._noise))


def compute_delta_p(real: Dict[str, Any], twin: Dict[str, Any]) -> float:
    """Compute mean absolute state error (delta_p)."""
    H_delta = abs(real["H"] - twin["twin_H"])
    rho_delta = abs(real["rho"] - twin["twin_rho"])
    tau_delta = abs(real["tau"] - twin["twin_tau"])
    beta_delta = abs(real["beta"] - twin["twin_beta"])
    return (H_delta + rho_delta + tau_delta + beta_delta) / 4.0


def run_arm(
    arm_id: str,
    seed: int,
    learning_rate: float,
    total_cycles: int,
    window_size: int,
    output_dir: Path,
) -> ArmResult:
    """Run one arm of the experiment."""
    telemetry = SyntheticTelemetry(seed)
    twin = TwinModel(seed, learning_rate)

    cycle_metrics: List[CycleMetrics] = []
    windows: List[WindowMetrics] = []

    for _ in range(total_cycles):
        real = telemetry.step()
        pred = twin.predict(real)
        delta_p = compute_delta_p(real, pred)

        cycle_metrics.append(CycleMetrics(
            cycle=real["cycle"],
            H=real["H"],
            rho=real["rho"],
            tau=real["tau"],
            beta=real["beta"],
            twin_H=pred["twin_H"],
            twin_rho=pred["twin_rho"],
            twin_tau=pred["twin_tau"],
            twin_beta=pred["twin_beta"],
            delta_p=delta_p,
            success=real["success"],
            predicted_success=pred["predicted_success"],
            in_omega=real["in_omega"],
            predicted_in_omega=pred["predicted_in_omega"],
            blocked=real["blocked"],
            predicted_blocked=pred["predicted_blocked"],
        ))

        twin.update(real)

    # Compute window aggregates
    for i in range(0, total_cycles, window_size):
        window_cycles = cycle_metrics[i:i + window_size]
        if not window_cycles:
            continue

        mean_delta_p = sum(c.delta_p for c in window_cycles) / len(window_cycles)
        diverged_count = sum(1 for c in window_cycles if c.delta_p > 0.05)
        success_div = sum(1 for c in window_cycles if c.success != c.predicted_success)

        windows.append(WindowMetrics(
            window_id=i // window_size,
            window_start=window_cycles[0].cycle,
            window_end=window_cycles[-1].cycle,
            mean_delta_p=mean_delta_p,
            divergence_rate=diverged_count / len(window_cycles),
            success_divergence_rate=success_div / len(window_cycles),
        ))

    # Compute final aggregates
    final_mean_delta_p = sum(c.delta_p for c in cycle_metrics) / len(cycle_metrics)
    final_div_rate = sum(1 for c in cycle_metrics if c.delta_p > 0.05) / len(cycle_metrics)

    # Save run_config.json
    config = RunConfig(
        experiment_id="CAL-EXP-3",
        arm_id=arm_id,
        seed=seed,
        learning_rate=learning_rate,
        noise_scale=0.02,
        total_cycles=total_cycles,
        window_size=window_size,
        initial_H=0.5,
        initial_rho=0.7,
        initial_tau=0.20,
        initial_beta=0.1,
        toolchain_version=TOOLCHAIN_VERSION,
        toolchain_fingerprint=compute_toolchain_fingerprint(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    arm_dir = output_dir / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)

    with open(arm_dir / "run_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Save cycle-level trace
    with open(arm_dir / "delta_p_trace.jsonl", "w") as f:
        for cm in cycle_metrics:
            f.write(json.dumps(asdict(cm)) + "\n")

    return ArmResult(
        arm_id=arm_id,
        learning_rate=learning_rate,
        seed=seed,
        total_cycles=total_cycles,
        final_mean_delta_p=final_mean_delta_p,
        final_divergence_rate=final_div_rate,
        windows=windows,
        cycle_metrics=cycle_metrics,
    )


def generate_summary(
    baseline: ArmResult,
    treatment: ArmResult,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate summary comparison."""
    delta_mean_delta_p = treatment.final_mean_delta_p - baseline.final_mean_delta_p
    delta_div_rate = treatment.final_divergence_rate - baseline.final_divergence_rate

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "CAL-EXP-3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "toolchain_fingerprint": compute_toolchain_fingerprint(),
        "arms": {
            "baseline": {
                "arm_id": baseline.arm_id,
                "learning_rate": baseline.learning_rate,
                "seed": baseline.seed,
                "total_cycles": baseline.total_cycles,
                "final_mean_delta_p": baseline.final_mean_delta_p,
                "final_divergence_rate": baseline.final_divergence_rate,
            },
            "treatment": {
                "arm_id": treatment.arm_id,
                "learning_rate": treatment.learning_rate,
                "seed": treatment.seed,
                "total_cycles": treatment.total_cycles,
                "final_mean_delta_p": treatment.final_mean_delta_p,
                "final_divergence_rate": treatment.final_divergence_rate,
            },
        },
        "comparison": {
            "delta_mean_delta_p": delta_mean_delta_p,
            "delta_divergence_rate": delta_div_rate,
            "delta_p_favors_treatment": delta_mean_delta_p < 0,
            "div_rate_favors_treatment": delta_div_rate < 0,
        },
        "windows": {
            "baseline": [asdict(w) for w in baseline.windows],
            "treatment": [asdict(w) for w in treatment.windows],
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_summary_table(summary: Dict[str, Any], output_dir: Path) -> None:
    """Generate markdown summary table."""
    baseline = summary["arms"]["baseline"]
    treatment = summary["arms"]["treatment"]
    comp = summary["comparison"]

    lines = [
        "# CAL-EXP-3 Summary",
        "",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Mode**: {summary['mode']}",
        f"**Toolchain**: {summary['toolchain_fingerprint']}",
        "",
        "## Arm Configuration",
        "",
        "| Arm | learning_rate | seed | cycles |",
        "|-----|---------------|------|--------|",
        f"| baseline | {baseline['learning_rate']} | {baseline['seed']} | {baseline['total_cycles']} |",
        f"| treatment | {treatment['learning_rate']} | {treatment['seed']} | {treatment['total_cycles']} |",
        "",
        "## Results",
        "",
        "| Metric | Baseline | Treatment | Delta |",
        "|--------|----------|-----------|-------|",
        f"| mean_delta_p | {baseline['final_mean_delta_p']:.6f} | {treatment['final_mean_delta_p']:.6f} | {comp['delta_mean_delta_p']:+.6f} |",
        f"| divergence_rate | {baseline['final_divergence_rate']:.4f} | {treatment['final_divergence_rate']:.4f} | {comp['delta_divergence_rate']:+.4f} |",
        "",
        "## Verdict",
        "",
        f"- Delta_p favors treatment: **{comp['delta_p_favors_treatment']}**",
        f"- Divergence rate favors treatment: **{comp['div_rate_favors_treatment']}**",
        "",
        "---",
        "",
        "**SHADOW MODE - observational only.**",
    ]

    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(lines))


def generate_plots(
    baseline: ArmResult,
    treatment: ArmResult,
    output_dir: Path,
) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[cal-exp-3] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: delta_p over time
    ax1 = axes[0]
    baseline_cycles = [c.cycle for c in baseline.cycle_metrics]
    baseline_delta_p = [c.delta_p for c in baseline.cycle_metrics]
    treatment_cycles = [c.cycle for c in treatment.cycle_metrics]
    treatment_delta_p = [c.delta_p for c in treatment.cycle_metrics]

    ax1.plot(baseline_cycles, baseline_delta_p, 'b-', alpha=0.5, label='Baseline (lr=0.0)')
    ax1.plot(treatment_cycles, treatment_delta_p, 'g-', alpha=0.5, label='Treatment (lr=0.1)')
    ax1.axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('delta_p')
    ax1.set_title('CAL-EXP-3: delta_p Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Windowed comparison
    ax2 = axes[1]
    baseline_window_x = [w.window_id for w in baseline.windows]
    baseline_window_y = [w.mean_delta_p for w in baseline.windows]
    treatment_window_x = [w.window_id for w in treatment.windows]
    treatment_window_y = [w.mean_delta_p for w in treatment.windows]

    width = 0.35
    x = range(len(baseline_window_x))
    ax2.bar([i - width/2 for i in x], baseline_window_y, width, label='Baseline', color='blue', alpha=0.7)
    ax2.bar([i + width/2 for i in x], treatment_window_y, width, label='Treatment', color='green', alpha=0.7)
    ax2.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('mean_delta_p')
    ax2.set_title('CAL-EXP-3: Windowed Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cal_exp_3_plots.png", dpi=150)
    plt.close()

    print(f"[cal-exp-3] Saved plots to {output_dir / 'cal_exp_3_plots.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAL-EXP-3: Baseline vs Learning-Enabled Comparison"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=500,
        help="Total cycles per arm (default: 500)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Window size for aggregation (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cal_exp_3"),
        help="Output directory (default: results/cal_exp_3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAL-EXP-3: Baseline vs Learning-Enabled")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Cycles: {args.cycles}")
    print(f"Window size: {args.window_size}")
    print(f"Output: {output_dir}")
    print()

    # ARM A: Baseline (no learning)
    print("[ARM A] Running baseline (learning_rate=0.0)...")
    baseline = run_arm(
        arm_id="baseline",
        seed=args.seed,
        learning_rate=0.0,
        total_cycles=args.cycles,
        window_size=args.window_size,
        output_dir=output_dir,
    )
    print(f"  mean_delta_p: {baseline.final_mean_delta_p:.6f}")
    print(f"  divergence_rate: {baseline.final_divergence_rate:.4f}")
    print()

    # ARM B: Treatment (learning enabled)
    print("[ARM B] Running treatment (learning_rate=0.1)...")
    treatment = run_arm(
        arm_id="treatment",
        seed=args.seed,
        learning_rate=0.1,
        total_cycles=args.cycles,
        window_size=args.window_size,
        output_dir=output_dir,
    )
    print(f"  mean_delta_p: {treatment.final_mean_delta_p:.6f}")
    print(f"  divergence_rate: {treatment.final_divergence_rate:.4f}")
    print()

    # Generate summary
    print("[SUMMARY] Generating comparison...")
    summary = generate_summary(baseline, treatment, output_dir)
    generate_summary_table(summary, output_dir)

    # Generate plots
    print("[PLOTS] Generating visualizations...")
    generate_plots(baseline, treatment, output_dir)

    # Print verdict
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    delta = summary["comparison"]["delta_mean_delta_p"]
    print(f"ΔΔp = {delta:+.6f}")
    if delta < 0:
        print("ΔΔp sign: NEGATIVE (delta_p favors treatment)")
    else:
        print("ΔΔp sign: POSITIVE (delta_p favors baseline)")
    print()
    print("SHADOW MODE - observational only.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
