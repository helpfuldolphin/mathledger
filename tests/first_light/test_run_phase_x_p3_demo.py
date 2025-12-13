from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path

from scripts import run_phase_x_p3_demo


def test_run_phase_x_p3_demo_py_compile() -> None:
    py_compile.compile("scripts/run_phase_x_p3_demo.py", doraise=True)


def test_run_first_light_harness_prefers_real(monkeypatch, tmp_path: Path) -> None:
    invoked: list[str] = []

    def fake_run(cmd, check=True):
        script_name = Path(cmd[1]).name
        invoked.append(script_name)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        (output_dir / "run_real").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(run_phase_x_p3_demo.subprocess, "run", fake_run)

    run_dir = run_phase_x_p3_demo.run_first_light_harness(tmp_path, cycles=10, seed=1)
    assert run_dir.name == "run_real"
    assert invoked == [run_phase_x_p3_demo.REAL_HARNESS_SCRIPT.name]


def test_run_first_light_harness_falls_back_to_wrapper(monkeypatch, tmp_path: Path, capsys) -> None:
    invoked: list[str] = []
    failed_once = {"real": False}

    def fake_run(cmd, check=True):
        script_name = Path(cmd[1]).name
        invoked.append(script_name)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        if script_name == run_phase_x_p3_demo.REAL_HARNESS_SCRIPT.name and not failed_once["real"]:
            failed_once["real"] = True
            raise subprocess.CalledProcessError(1, cmd)
        (output_dir / "run_wrapper").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(run_phase_x_p3_demo.subprocess, "run", fake_run)

    run_dir = run_phase_x_p3_demo.run_first_light_harness(tmp_path, cycles=10, seed=2)
    assert run_dir.name == "run_wrapper"

    captured = capsys.readouterr()
    assert "Real harness invocation failed" in captured.out
    assert invoked == [
        run_phase_x_p3_demo.REAL_HARNESS_SCRIPT.name,
        run_phase_x_p3_demo.WRAPPER_SCRIPT.name,
    ]
