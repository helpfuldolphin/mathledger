from __future__ import annotations

import json
from pathlib import Path

from scripts.first_light_cal_exp1_warm_start import run_cal_exp1
from tests.factories.first_light_factories import make_cal_exp1_report


def test_cal_exp1_runs_200_cycles_real_adapter_shadow_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    args = type(
        "Args",
        (),
        {
            "adapter": "real",
            "cycles": 200,
            "learning_rate": 0.1,
            "seed": 7,
            "output_dir": out_dir,
        },
    )
    report_path = run_cal_exp1(args)
    payload = json.loads(report_path.read_text())
    assert report_path.exists()
    assert len(payload.get("windows", [])) >= 4
    assert payload["summary"]["final_divergence_rate"] is not None


def test_cal_exp1_factory_report_shape_and_determinism() -> None:
    report1 = make_cal_exp1_report(cycles=200, seed=1234)
    report2 = make_cal_exp1_report(cycles=200, seed=1234)

    assert report1 == report2
    assert report1["schema_version"] == "1.0.0"
    assert report1["mode"] == "SHADOW"
    assert len(report1["windows"]) >= 4
    for window in report1["windows"]:
        assert set(window.keys()) == {
            "window_index",
            "window_start",
            "window_end",
            "cycles_in_window",
            "divergence_count",
            "divergence_rate",
            "mean_delta_p",
            "delta_bias",
            "delta_variance",
            "phase_lag_xcorr",
            "pattern_tag",
        }
        assert window["window_start"] <= window["window_end"]
        assert window["cycles_in_window"] >= 0
    assert "final_divergence_rate" in report1["summary"]
