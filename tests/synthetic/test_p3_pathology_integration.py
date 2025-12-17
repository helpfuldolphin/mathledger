import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _run_harness_and_read_H(tmp_dir: Path, pathology: str) -> tuple[list[float], Path]:
    """Run the P3 harness for a short synthetic run and return H trajectory and run dir."""
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_dir / pathology
    cmd = [
        sys.executable,
        "scripts/usla_first_light_harness.py",
        "--cycles",
        "20",
        "--seed",
        "123",
        "--output-dir",
        str(output_dir),
        "--pathology",
        pathology,
    ]
    subprocess.run(cmd, cwd=root, check=True)

    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1, "expected a single harness run directory"
    run_dir = run_dirs[0]
    synthetic_raw_path = run_dir / "synthetic_raw.jsonl"

    with synthetic_raw_path.open("r", encoding="utf-8") as f:
        h_series = [json.loads(line)["usla_state"]["H"] for line in f]

    return h_series, run_dir


def test_pathology_spike_injects_into_h_series(tmp_path: Path) -> None:
    baseline, _ = _run_harness_and_read_H(tmp_path, pathology="none")
    spiked, run_dir = _run_harness_and_read_H(tmp_path, pathology="spike")

    assert len(baseline) == len(spiked) == 20

    deltas = [s - b for s, b in zip(spiked, baseline)]
    spike_indices = [i for i, d in enumerate(deltas) if abs(d) > 1e-3]

    assert spike_indices == [10], "expected single spike near run midpoint"
    assert deltas[spike_indices[0]] == pytest.approx(0.75, abs=1e-3)

    report_path = run_dir / "stability_report.json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["pathology"] == "spike"
    assert report["pathology_params"] == {"magnitude": 0.75, "at": 10}
