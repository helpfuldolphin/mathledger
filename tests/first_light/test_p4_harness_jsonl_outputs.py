from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
from backend.health.u2_dynamics_tile import build_u2_dynamics_window_metrics


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.integration
def test_p4_harness_emits_expected_jsonl_artifacts(tmp_path: Path) -> None:
    """Run the real harness and ensure the JsonlWriter-backed files look healthy."""
    output_root = tmp_path / "p4_runs"
    cycles = 50

    cmd = [
        sys.executable,
        "scripts/usla_first_light_p4_harness.py",
        "--cycles",
        str(cycles),
        "--output-dir",
        str(output_root),
        "--seed",
        "7",
    ]
    subprocess.run(cmd, check=True)

    run_dirs = sorted(p for p in output_root.iterdir() if p.is_dir())
    assert run_dirs, "Harness did not produce an output directory"
    run_dir = run_dirs[-1]

    expectations = {
        "real_cycles.jsonl": cycles,
        "twin_predictions.jsonl": cycles,
        "divergence_log.jsonl": cycles,
    }
    for filename, expected_count in expectations.items():
        target = run_dir / filename
        assert target.exists(), f"Expected {filename} in {run_dir}"
        records = _read_jsonl(target)
        assert len(records) == expected_count, f"{filename} record mismatch"

    # Compute U2 dynamics window metrics for CAL-EXP observational plumbing
    real_cycles_path = run_dir / "real_cycles.jsonl"
    tile_first = build_u2_dynamics_window_metrics(real_cycles_path)
    tile_second = build_u2_dynamics_window_metrics(real_cycles_path)
    assert tile_first == tile_second
    assert json.loads(json.dumps(tile_first)) == tile_first
