import json
import subprocess
import sys
from pathlib import Path

from scripts.policy_drift_lint import summarize_policy_drift_for_global_health


def write_config(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def run_linter(old_path: Path, new_path: Path, warn_exit_zero: bool = False) -> dict:
    cmd = [
        sys.executable,
        "scripts/policy_drift_lint.py",
        "--old",
        str(old_path),
        "--new",
        str(new_path),
    ]
    if warn_exit_zero:
        cmd.append("--warn-exit-zero")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "payload": json.loads(result.stdout),
    }


def test_abstention_gate_disable_blocks(tmp_path):
    baseline = tmp_path / "base.yaml"
    candidate = tmp_path / "cand.yaml"
    write_config(
        baseline,
        """
abstention_gate:
  enabled: true
""",
    )
    write_config(
        candidate,
        """
abstention_gate:
  enabled: false
""",
    )

    result = run_linter(baseline, candidate)
    tile = summarize_policy_drift_for_global_health(result["payload"])
    assert result["payload"]["status"] == "BLOCK"
    assert tile["policy_ok"] is False
    assert "blocking change" in tile["headline"]


def test_reward_weight_adjustment_warns(tmp_path):
    baseline = tmp_path / "base.yaml"
    candidate = tmp_path / "cand.yaml"
    write_config(baseline, "abstention:\n  reward_weight: -0.8\n")
    write_config(candidate, "abstention:\n  reward_weight: -0.5\n")

    result = run_linter(baseline, candidate)
    tile = summarize_policy_drift_for_global_health(result["payload"])
    assert result["payload"]["status"] == "WARN"
    assert tile["status"] == "WARN"
    assert tile["policy_ok"] is False
    assert "soft change" in tile["headline"]


def test_workflow_modes_share_tile(tmp_path):
    baseline = tmp_path / "base.yaml"
    candidate = tmp_path / "cand.yaml"
    write_config(baseline, "abstention:\n  reward_weight: -0.9\n")
    write_config(candidate, "abstention:\n  reward_weight: -0.7\n")

    strict = run_linter(baseline, candidate)
    advisory = run_linter(baseline, candidate, warn_exit_zero=True)

    strict_tile = summarize_policy_drift_for_global_health(strict["payload"])
    advisory_tile = summarize_policy_drift_for_global_health(advisory["payload"])

    assert strict["payload"]["status"] == "WARN"
    assert advisory["payload"]["status"] == "WARN"
    assert strict_tile == advisory_tile
    assert strict["exit_code"] == 1
    assert advisory["exit_code"] == 0
