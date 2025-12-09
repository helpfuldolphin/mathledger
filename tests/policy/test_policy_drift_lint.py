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
    baseline = tmp_path / "old.yaml"
    candidate = tmp_path / "new.yaml"
    write_config(
        baseline,
        """
abstention_ctrl:
  gate_enabled: true
""",
    )
    write_config(
        candidate,
        """
abstention_ctrl:
  gate_enabled: false
""",
    )
    result = run_linter(baseline, candidate)
    assert result["exit_code"] == 1
    assert result["payload"]["status"] == "BLOCK"
    categories = {entry["category"] for entry in result["payload"]["breaking_changes"]}
    assert "abstention_controls" in categories


def test_reward_weight_adjustment_warns(tmp_path):
    baseline = tmp_path / "old.yaml"
    candidate = tmp_path / "new.yaml"
    write_config(
        baseline,
        """
abstention:
  reward_weight: -0.8
""",
    )
    write_config(
        candidate,
        """
abstention:
  reward_weight: -0.5
""",
    )
    result = run_linter(baseline, candidate)
    assert result["exit_code"] == 1  # WARN still returns non-zero
    assert result["payload"]["status"] == "WARN"
    assert not result["payload"]["breaking_changes"]
    assert result["payload"]["soft_changes"]


def test_warn_exit_zero_flag(tmp_path):
    baseline = tmp_path / "base.yaml"
    candidate = tmp_path / "cand.yaml"
    write_config(baseline, "abstention:\n  reward_weight: -0.8\n")
    write_config(candidate, "abstention:\n  reward_weight: -0.7\n")
    result = run_linter(baseline, candidate, warn_exit_zero=True)
    assert result["payload"]["status"] == "WARN"
    assert result["exit_code"] == 0


def test_global_health_summary():
    ok = summarize_policy_drift_for_global_health({"status": "OK"})
    warn = summarize_policy_drift_for_global_health({"status": "WARN"})
    block = summarize_policy_drift_for_global_health({"status": "BLOCK"})

    assert ok == {
        "schema_version": "1.0.0",
        "policy_ok": True,
        "status": "OK",
        "headline": "Policy stable; no tracked drift.",
    }
    assert warn["policy_ok"] is False and warn["status"] == "WARN"
    assert block["policy_ok"] is False and block["status"] == "BLOCK"
