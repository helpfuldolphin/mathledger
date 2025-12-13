from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from scripts.first_light_cal_exp1_warm_start import (
    _build_pattern_tagging_metadata,
    compute_window_metrics,
    run_cal_exp1,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Snapshot:
    cycle: int
    delta_p: float
    diverged: bool = False

    def is_diverged(self) -> bool:
        return self.diverged


def _snapshots(delta_ps: List[float]) -> List[_Snapshot]:
    return [_Snapshot(cycle=i, delta_p=dp, diverged=False) for i, dp in enumerate(delta_ps)]


def test_pattern_tag_oscillation_from_delta_sign_flips() -> None:
    delta_ps = [0.1 if i % 2 == 0 else -0.1 for i in range(50)]
    windows = compute_window_metrics(_snapshots(delta_ps))
    assert windows[0]["pattern_tag"] == "OSCILLATION"


def test_pattern_tag_spike_from_single_outlier() -> None:
    delta_ps = [0.01] * 50
    delta_ps[25] = 5.0
    windows = compute_window_metrics(_snapshots(delta_ps))
    assert windows[0]["pattern_tag"] == "SPIKE"


def test_pattern_tag_drift_from_monotonic_trend() -> None:
    delta_ps = [i / 100 for i in range(50)]
    windows = compute_window_metrics(_snapshots(delta_ps))
    assert windows[0]["pattern_tag"] == "DRIFT"


def test_pattern_tag_none_when_no_clear_signature() -> None:
    delta_ps = [0.10, 0.12, 0.11, 0.13, 0.12] * 10
    windows = compute_window_metrics(_snapshots(delta_ps))
    assert windows[0]["pattern_tag"] == "NONE"


def test_pattern_tag_oscillation_takes_precedence_over_spike() -> None:
    delta_ps = [0.1 if i % 2 == 0 else -0.1 for i in range(50)]
    delta_ps[24] = 5.0
    windows = compute_window_metrics(_snapshots(delta_ps))
    assert windows[0]["pattern_tag"] == "OSCILLATION"

def test_pattern_tag_none_with_insufficient_data_note_when_delta_p_missing() -> None:
    class _BadSnapshot:
        def __init__(self, cycle: int) -> None:
            self.cycle = cycle

        def is_diverged(self) -> bool:
            return False

    good = _snapshots([0.01] * 50)
    good[3] = _BadSnapshot(cycle=3)
    windows = compute_window_metrics(good)
    assert windows[0]["pattern_tag"] == "NONE"
    assert windows[0]["pattern_tag_note"] == "INSUFFICIENT_DATA"


def test_report_labels_pattern_tagging_heuristic_provisional(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    args = type(
        "Args",
        (),
        {
            "adapter": "mock",
            "cycles": 60,
            "learning_rate": 0.1,
            "seed": 7,
            "output_dir": out_dir,
        },
    )
    report_path = run_cal_exp1(args)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    tagging = payload["pattern_tagging"]
    assert tagging["mode"] == "SHADOW"
    assert tagging["note"] == "heuristic, provisional"
    assert tagging["extraction_source"] == "HEURISTIC_V0_1"
    assert tagging["confidence"] == "LOW"
    assert tagging["reason_code"] == "CONF_LOW_INSUFFICIENT_DATA"


def test_pattern_tagging_confidence_reason_code_allowed_and_deterministic() -> None:
    allowed_extraction_sources = {"HEURISTIC_V0_1"}
    allowed_confidence = {"LOW", "MEDIUM"}
    allowed_reason_codes = {"CONF_LOW_INSUFFICIENT_DATA", "CONF_MED_SUFFICIENT_DATA"}
    allowed_pairs = {
        ("LOW", "CONF_LOW_INSUFFICIENT_DATA"),
        ("MEDIUM", "CONF_MED_SUFFICIENT_DATA"),
    }

    windows = [
        {"pattern_tag_note": None},
        {"pattern_tag_note": None},
    ]
    tag_a = _build_pattern_tagging_metadata(
        windows=windows,
        total_cycles=120,
        final_divergence_rate=0.1,
    )
    tag_b = _build_pattern_tagging_metadata(
        windows=windows,
        total_cycles=120,
        final_divergence_rate=0.1,
    )

    assert tag_a == tag_b
    assert tag_a["extraction_source"] in allowed_extraction_sources
    assert tag_a["confidence"] in allowed_confidence
    assert tag_a["reason_code"] in allowed_reason_codes
    assert (tag_a["confidence"], tag_a["reason_code"]) in allowed_pairs
    assert (tag_a["confidence"], tag_a["reason_code"]) == ("MEDIUM", "CONF_MED_SUFFICIENT_DATA")


def test_pattern_tagging_confidence_low_when_insufficient_data_present() -> None:
    tag = _build_pattern_tagging_metadata(
        windows=[{"pattern_tag_note": "INSUFFICIENT_DATA"}],
        total_cycles=120,
        final_divergence_rate=0.1,
    )
    assert tag["confidence"] == "LOW"
    assert tag["reason_code"] == "CONF_LOW_INSUFFICIENT_DATA"
