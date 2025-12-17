#!/usr/bin/env python3
"""
CAL-EXP-3 Regime Change Scaffold (P5 Calibration Campaign)

Injects a simple regime change into mock telemetry and records divergence
metrics before and after the change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
from dataclasses import replace


class RegimeChangeAdapter(MockTelemetryProvider):
    """Mock adapter that injects a regime change after N cycles."""

    def __init__(self, change_after: int = 100, delta_H: float = 0.2, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.change_after = change_after
        self.delta_H = delta_H

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        if snapshot and snapshot.cycle >= self.change_after:
            adjusted_H = min(1.0, snapshot.H + self.delta_H)
            snapshot = replace(snapshot, H=adjusted_H)
        return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAL-EXP-3 regime change scaffold")
    parser.add_argument(
        "--cycles",
        type=int,
        default=300,
        help="Total cycles to run (default: 300)",
    )
    parser.add_argument(
        "--change-after",
        type=int,
        default=100,
        help="Cycle after which regime change occurs (default: 100)",
    )
    parser.add_argument(
        "--delta-H",
        type=float,
        default=0.2,
        dest="delta_h",
        help="Offset to apply to H after regime change (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=321,
        help="Seed for reproducibility (default: 321)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for calibration artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = RegimeChangeAdapter(seed=args.seed, change_after=args.change_after, delta_H=args.delta_h)
    cfg = FirstLightConfigP4()
    cfg.total_cycles = args.cycles
    cfg.telemetry_adapter = adapter
    cfg.run_id = f"cal_exp3_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    runner = FirstLightShadowRunnerP4(cfg, seed=args.seed)
    runner.run()

    history = runner._divergence_analyzer.get_divergence_history()  # type: ignore[attr-defined]
    pre = [snap for snap in history if snap.cycle < args.change_after]
    post = [snap for snap in history if snap.cycle >= args.change_after]

    def _rate(snaps):
        return (sum(1 for s in snaps if s.is_diverged()) / len(snaps)) if snaps else 0.0

    report = {
        "schema_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "cycles": args.cycles,
            "change_after": args.change_after,
            "delta_H": args.delta_h,
            "seed": args.seed,
        },
        "pre_change": {
            "divergence_rate": _rate(pre),
            "sample_size": len(pre),
        },
        "post_change": {
            "divergence_rate": _rate(post),
            "sample_size": len(post),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path = output_dir / "cal_exp3_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[cal-exp3] Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
