#!/usr/bin/env python3
"""
Offline verifier-noise guard consistency check.

Reads metrics emitted by derivation.noise_guard and recomputes epsilon_total
using the configured channel weights. Emits a status line suitable for CI gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from derivation.noise_guard import VerifierNoiseConfig


def compute_epsilon_total(channels: Dict[str, float], config: VerifierNoiseConfig) -> float:
    value = 1.0
    for name, prob in channels.items():
        weight = getattr(config, f"{name}_weight", 1.0)
        value *= (1.0 - weight * prob)
    value = max(0.0, value)
    return 1.0 - value


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify verifier-noise guard telemetry.")
    parser.add_argument(
        "--metrics",
        default="metrics/verifier_noise_window.json",
        help="Path to metrics/verifier_noise_window.json (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        default="config/verifier_noise_phase2.yaml",
        help="Noise guard config path (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed absolute difference between recorded and recomputed epsilon_total.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    config_path = Path(args.config)
    if not metrics_path.exists():
        raise SystemExit(f"metrics file not found: {metrics_path}")
    if not config_path.exists():
        raise SystemExit(f"config file not found: {config_path}")

    cfg = VerifierNoiseConfig.from_file(config_path)
    snapshot = json.loads(metrics_path.read_text())

    channels = snapshot.get("channels")
    if not isinstance(channels, dict):
        raise SystemExit("metrics snapshot missing 'channels' object")

    recomputed = compute_epsilon_total({k: float(v) for k, v in channels.items()}, cfg)
    recorded = float(snapshot.get("epsilon_total", recomputed))
    delta = abs(recomputed - recorded)
    guard_state = "stable"
    if snapshot.get("timeout_noisy"):
        guard_state = "timeout-noisy"
    elif snapshot.get("unstable_buckets"):
        guard_state = "bucket-unstable"

    print(
        json.dumps(
            {
                "epsilon_total_recorded": recorded,
                "epsilon_total_recomputed": recomputed,
                "delta": delta,
                "guard_state": guard_state,
                "timeout_cusum": snapshot.get("timeout_cusum", 0.0),
                "window_id": snapshot.get("window_id"),
            },
            indent=2,
        )
    )

    if delta > args.tolerance:
        raise SystemExit(
            f"epsilon_total mismatch exceeds tolerance: recorded={recorded:.6f} recomputed={recomputed:.6f}"
        )


if __name__ == "__main__":
    main()
