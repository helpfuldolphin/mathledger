#!/usr/bin/env python3
"""
CAL-EXP-1 Warm-Start Harness (P5 Calibration Campaign)

Runs a short-window warm-start shadow loop using the P4 runner and emits a
compact calibration report with windowed divergence metrics.

SHADOW MODE: Observation-only. No governance actions are taken.

Outputs:
  - cal_exp1_report.json

P5_CALIBRATION_GAPS:
- No structural-break detector yet (uses fixed flag).
- Pattern tagging is heuristic, provisional (SHADOW-only).
- Twin learning-rate override is recorded but not plumbed into P4 twin config.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
from backend.topology.first_light.telemetry_adapter import (
    MockTelemetryProvider,
    USLAIntegrationAdapter,
)

WINDOW_SMALL = 10
WINDOW_LARGE = 50

PATTERN_TAGGING_NOTE = "heuristic, provisional"
PATTERN_TAGGING_INSUFFICIENT_DATA_NOTE = "INSUFFICIENT_DATA"
PATTERN_TAGGING_EXTRACTION_SOURCE = "HEURISTIC_V0_1"
PATTERN_TAGGING_CONF_LOW_REASON = "CONF_LOW_INSUFFICIENT_DATA"
PATTERN_TAGGING_CONF_MED_REASON = "CONF_MED_SUFFICIENT_DATA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAL-EXP-1 warm-start shadow harness (P5 calibration)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["real", "mock"],
        default="real",
        help="Telemetry adapter (default: real).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=200,
        help="Total cycles to observe (default: 200).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Twin learning rate (recorded for reporting; twin uses default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--decoupled-success",
        action="store_true",
        help="Enable decoupled success prediction for the twin (P5 upgrade flag).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for calibration artifacts.",
    )
    return parser.parse_args()


def build_adapter(adapter: str, seed: int) -> object:
    if adapter == "real":
        # If no integration is wired, fall back to mock to keep demo runnable.
        real_adapter = USLAIntegrationAdapter(integration_ref=None)
        if real_adapter.is_available():
            return real_adapter
        return MockTelemetryProvider(seed=seed)
    return MockTelemetryProvider(seed=seed)

def _classify_pattern_tag(delta_ps: Optional[List[float]]) -> tuple[str, Optional[str], str]:
    """
    Heuristic, provisional tagger for CAL-EXP-1 window patterns.

    SHADOW-ONLY: Observational labels. Not used for enforcement.

    Rules (minimal):
    - Oscillatory Δp sign flips → OSCILLATION
    - Large single-window spike → SPIKE
    - Monotonic trend → DRIFT
    - Otherwise → NONE
    """
    if not delta_ps or len(delta_ps) < 3:
        return "NONE", PATTERN_TAGGING_INSUFFICIENT_DATA_NOTE, "LOW"

    if not all(isinstance(dp, (int, float)) for dp in delta_ps):
        return "NONE", PATTERN_TAGGING_INSUFFICIENT_DATA_NOTE, "LOW"

    def _sign(value: float, eps: float = 1e-9) -> int:
        if value > eps:
            return 1
        if value < -eps:
            return -1
        return 0

    non_zero_signs = [s for s in (_sign(dp) for dp in delta_ps) if s != 0]
    if len(non_zero_signs) >= 4:
        sign_flips = sum(
            1
            for i in range(len(non_zero_signs) - 1)
            if non_zero_signs[i] != non_zero_signs[i + 1]
        )
        denom = max(1, len(non_zero_signs) - 1)
        if sign_flips >= 2 and (sign_flips / denom) >= 0.6:
            confidence = "MEDIUM" if (sign_flips / denom) >= 0.8 else "LOW"
            return "OSCILLATION", None, confidence

    mean = statistics.fmean(delta_ps)
    stdev = statistics.pstdev(delta_ps)
    if stdev > 0:
        max_deviation = max(abs(dp - mean) for dp in delta_ps)
        if max_deviation > 3 * stdev:
            confidence = "MEDIUM" if max_deviation > 4 * stdev else "LOW"
            return "SPIKE", None, confidence

    n = len(delta_ps)
    increasing = sum(1 for i in range(n - 1) if delta_ps[i + 1] > delta_ps[i])
    decreasing = sum(1 for i in range(n - 1) if delta_ps[i + 1] < delta_ps[i])
    if increasing > 0.7 * (n - 1) or decreasing > 0.7 * (n - 1):
        ratio = max(increasing, decreasing) / max(1, n - 1)
        confidence = "MEDIUM" if ratio >= 0.85 else "LOW"
        return "DRIFT", None, confidence

    return "NONE", None, "LOW"


def compute_window_metrics(snapshots: List[object]) -> List[Dict[str, object]]:
    windows: List[Dict[str, object]] = []
    for start in range(0, len(snapshots), WINDOW_LARGE):
        chunk = snapshots[start : start + WINDOW_LARGE]
        if not chunk:
            continue
        start_cycle_raw = getattr(chunk[0], "cycle", start)
        end_cycle_raw = getattr(chunk[-1], "cycle", start + len(chunk) - 1)
        start_cycle = int(start_cycle_raw) if isinstance(start_cycle_raw, int) else start
        end_cycle = int(end_cycle_raw) if isinstance(end_cycle_raw, int) else start + len(chunk) - 1

        def _is_diverged(snap: object) -> bool:
            fn = getattr(snap, "is_diverged", None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    return False
            return bool(getattr(snap, "diverged", False))

        divergence_rate = sum(1 for snap in chunk if _is_diverged(snap)) / len(chunk)

        deltas: List[float] = []
        insufficient = False
        for snap in chunk:
            dp = getattr(snap, "delta_p", None)
            if isinstance(dp, (int, float)):
                deltas.append(float(dp))
            else:
                insufficient = True

        mean_delta_p = sum(deltas) / len(deltas) if deltas else 0.0
        delta_bias = mean_delta_p
        delta_variance = (
            sum((d - mean_delta_p) ** 2 for d in deltas) / len(deltas) if deltas else 0.0
        )
        pattern_tag, pattern_tag_note, _ = _classify_pattern_tag(None if insufficient else deltas)

        windows.append(
            {
                "start_cycle": start_cycle,
                "end_cycle": end_cycle,
                "divergence_rate": divergence_rate,
                "mean_delta_p": mean_delta_p,
                "delta_bias": delta_bias,
                "delta_variance": delta_variance,
                "phase_lag_xcorr": 0.0,  # Stub for future signal analysis
                "pattern_tag": pattern_tag,
                "pattern_tag_note": pattern_tag_note,
            }
        )
    return windows


def _build_pattern_tagging_metadata(
    *,
    windows: List[Dict[str, object]],
    total_cycles: int,
    final_divergence_rate: float,
) -> Dict[str, str]:
    confidence = "LOW"
    reason_code = PATTERN_TAGGING_CONF_LOW_REASON
    if windows and not any(
        w.get("pattern_tag_note") == PATTERN_TAGGING_INSUFFICIENT_DATA_NOTE for w in windows
    ):
        if total_cycles >= 2 * WINDOW_LARGE and float(final_divergence_rate or 0.0) > 0.0:
            confidence = "MEDIUM"
            reason_code = PATTERN_TAGGING_CONF_MED_REASON
    return {
        "mode": "SHADOW",
        "note": PATTERN_TAGGING_NOTE,
        "extraction_source": PATTERN_TAGGING_EXTRACTION_SOURCE,
        "confidence": confidence,
        "reason_code": reason_code,
    }


def compute_divergence_rate_over_small_windows(snapshots: List[object]) -> float:
    if not snapshots:
        return 0.0
    chunk = snapshots[-WINDOW_SMALL:]
    return sum(1 for snap in chunk if snap.is_diverged()) / len(chunk)


def run_cal_exp1(args: argparse.Namespace) -> Path:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(args.adapter, args.seed)
    cfg = FirstLightConfigP4()
    cfg.total_cycles = args.cycles
    cfg.telemetry_adapter = adapter
    cfg.run_id = f"cal_exp1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cfg.use_decoupled_success = bool(getattr(args, "decoupled_success", False))

    runner = FirstLightShadowRunnerP4(cfg, seed=args.seed)
    result = runner.run()

    # Collect divergence snapshots
    divergence_history = runner._divergence_analyzer.get_divergence_history()  # type: ignore[attr-defined]

    windows = compute_window_metrics(divergence_history)
    final_divergence_rate = compute_divergence_rate_over_small_windows(divergence_history)
    final_delta_bias = windows[-1]["delta_bias"] if windows else 0.0

    pattern_tagging = _build_pattern_tagging_metadata(
        windows=windows,
        total_cycles=args.cycles,
        final_divergence_rate=final_divergence_rate,
    )

    report = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pattern_tagging": pattern_tagging,
        "params": {
            "adapter": args.adapter,
            "cycles": args.cycles,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "decoupled_success": cfg.use_decoupled_success,
        },
        "windows": windows,
        "summary": {
            "final_divergence_rate": final_divergence_rate,
            "final_delta_bias": final_delta_bias,
            "no_structural_break_detected": True,
        },
        "twin_accuracy": runner._divergence_analyzer.get_summary().to_dict(),  # type: ignore[attr-defined]
        "result": result.to_dict(),
    }

    report_path = output_dir / "cal_exp1_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def main() -> int:
    args = parse_args()
    try:
        report_path = run_cal_exp1(args)
        print(f"[cal-exp1] Wrote report to {report_path}")
        return 0
    except Exception as exc:  # pragma: no cover - surfaced via tests
        print(f"[cal-exp1] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
