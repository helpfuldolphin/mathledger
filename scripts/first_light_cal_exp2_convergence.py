#!/usr/bin/env python3
"""
CAL-EXP-2 Convergence Scaffold (P5 Calibration Campaign)

Runs short shadow trials with different learning rates and records divergence
trajectories. This is scaffolding for the full convergence sweep.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAL-EXP-2 convergence scaffold")
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.2],
        help="Learning rates to sweep (default: 0.05 0.1 0.2)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Total cycles (scaffold runs 200 per trial as placeholder)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for reproducibility (default: 123)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for calibration artifacts.",
    )
    return parser.parse_args()


def run_trial(lr: float, seed: int, cycles: int) -> Dict[str, object]:
    cfg = FirstLightConfigP4()
    cfg.total_cycles = cycles
    cfg.telemetry_adapter = MockTelemetryProvider(seed=seed)
    cfg.run_id = f"cal_exp2_lr_{lr}"
    runner = FirstLightShadowRunnerP4(cfg, seed=seed)
    runner.run()
    history = runner._divergence_analyzer.get_divergence_history()  # type: ignore[attr-defined]

    trajectory: List[float] = []
    for end in range(20, len(history) + 1, 20):
        window = history[:end]
        rate = sum(1 for snap in window if snap.is_diverged()) / len(window)
        trajectory.append(rate)
    return {"lr": lr, "divergence_trajectory": trajectory}


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = [
        run_trial(lr, seed=args.seed, cycles=200) for lr in args.learning_rates
    ]

    report = {
        "schema_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    report_path = output_dir / "cal_exp2_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[cal-exp2] Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
