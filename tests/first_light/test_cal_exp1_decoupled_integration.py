from __future__ import annotations

import json
from pathlib import Path

from scripts.first_light_cal_exp1_warm_start import run_cal_exp1


def test_cal_exp1_decoupled_success_improves_accuracy(tmp_path: Path) -> None:
    def _run(decoupled: bool) -> dict:
        out_dir = tmp_path / ("decoupled" if decoupled else "baseline")
        args = type(
            "Args",
            (),
            {
                "adapter": "mock",
                "cycles": 100,
                "learning_rate": 0.1,
                "seed": 999,
                "output_dir": out_dir,
                "decoupled_success": decoupled,
            },
        )
        report_path = run_cal_exp1(args)
        payload = json.loads(report_path.read_text())
        return payload["twin_accuracy"]["accuracy"]["success_accuracy"]

    acc_base = _run(False)
    acc_decoupled = _run(True)

    # PROVISIONAL: Expect decoupled mode to be at least as good
    assert acc_decoupled >= acc_base
