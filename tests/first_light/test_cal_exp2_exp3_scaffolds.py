from __future__ import annotations

import json
from pathlib import Path

from scripts import first_light_cal_exp2_convergence as exp2
from scripts import first_light_cal_exp3_regime_change as exp3
from tests.factories.first_light_factories import (
    make_cal_exp2_report,
    make_cal_exp3_report,
)

def test_cal_exp2_scaffold_generates_report(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "learning_rates": [0.05, 0.1],
            "cycles": 200,
            "seed": 1,
            "output_dir": tmp_path / "exp2",
        },
    )
    exp2.main.__globals__["parse_args"] = lambda: args  # type: ignore
    assert exp2.main() == 0
    report_path = args.output_dir / "cal_exp2_report.json"
    payload = json.loads(report_path.read_text())
    assert "trials" in payload
    assert len(payload["trials"]) == 2


def test_cal_exp3_scaffold_generates_report(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "cycles": 120,
            "change_after": 40,
            "delta_h": 0.1,
            "seed": 2,
            "output_dir": tmp_path / "exp3",
        },
    )
    exp3.main.__globals__["parse_args"] = lambda: args  # type: ignore
    assert exp3.main() == 0
    report_path = args.output_dir / "cal_exp3_report.json"
    payload = json.loads(report_path.read_text())
    assert "pre_change" in payload and "post_change" in payload


def test_cal_exp2_factory_report_shape_and_determinism() -> None:
    report1 = make_cal_exp2_report([0.05, 0.15], seed=321)
    report2 = make_cal_exp2_report([0.05, 0.15], seed=321)

    assert report1 == report2
    assert report1["schema_version"] == "0.1.0"
    assert len(report1["trials"]) == 2
    for trial in report1["trials"]:
        assert "lr" in trial and "divergence_trajectory" in trial
        assert len(trial["divergence_trajectory"]) == 5


def test_cal_exp3_factory_report_shape() -> None:
    report = make_cal_exp3_report(cycles=300, change_after=120, seed=987)
    assert report["schema_version"] == "0.1.0"
    assert report["params"]["change_after"] == 120
    assert "divergence_rate" in report["pre_change"]
    assert "divergence_rate" in report["post_change"]
