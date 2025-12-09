import json
import subprocess
import sys
from pathlib import Path


def write_policy(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_lint(old_path: Path, new_path: Path, text_mode: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "scripts/policy_drift_lint.py",
        "--old",
        str(old_path),
        "--new",
        str(new_path),
    ]
    if text_mode:
        cmd.append("--text")
    return subprocess.run(cmd, capture_output=True, text=True)


def test_policy_drift_lint_no_changes(tmp_path):
    old_file = tmp_path / "old.yaml"
    new_file = tmp_path / "new.yaml"
    data = """
trainer:
  learning_rate: 0.1
  clip_norm: 0.5
abstention:
  reward_weight: -0.8
promotion:
  threshold: 0.9
"""
    write_policy(old_file, data)
    write_policy(new_file, data)

    result = run_lint(old_file, new_file)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0.0"
    assert payload["status"] == "OK"
    assert payload["summary"]["total_changes"] == 0


def test_policy_drift_lint_blocks_learning_rate_spike(tmp_path):
    old_file = tmp_path / "old.yaml"
    new_file = tmp_path / "new.yaml"
    write_policy(
        old_file,
        """
trainer:
  learning_rate: 0.1
  clip_norm: 0.5
promotion_policy:
  required_level: L2
""",
    )
    write_policy(
        new_file,
        """
trainer:
  learning_rate: 0.6
  clip_norm: 0.5
promotion_policy:
  required_level: L3
""",
    )

    result = run_lint(old_file, new_file)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCK"
    paths = {entry["path"] for entry in payload["breaking_changes"]}
    assert "trainer.learning_rate" in paths
    assert "promotion_policy.required_level" in paths


def test_policy_drift_lint_warns_on_small_learning_rate_change(tmp_path):
    old_file = tmp_path / "old.yaml"
    new_file = tmp_path / "new.yaml"
    write_policy(
        old_file,
        """
trainer:
  learning_rate: 0.1
""",
    )
    write_policy(
        new_file,
        """
trainer:
  learning_rate: 0.2
""",
    )
    result = run_lint(old_file, new_file)
    payload = json.loads(result.stdout)
    assert payload["status"] == "WARN"
    assert payload["soft_changes"], "Expected soft change entry"


def test_policy_drift_lint_blocks_clipping_removal(tmp_path):
    old_file = tmp_path / "old.yaml"
    new_file = tmp_path / "new.yaml"
    write_policy(
        old_file,
        """
trainer:
  clip_norm: 0.5
""",
    )
    write_policy(
        new_file,
        """
trainer:
  learning_rate: 0.1
""",
    )
    result = run_lint(old_file, new_file)
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCK"
    removed = [entry for entry in payload["breaking_changes"] if entry["category"] == "clipping_thresholds"]
    assert removed, "Expected removal of clipping threshold to be flagged as breaking"


def test_policy_drift_lint_blocks_abstention_gating_disable(tmp_path):
    old_file = tmp_path / "old.yaml"
    new_file = tmp_path / "new.yaml"
    write_policy(
        old_file,
        """
abstention_gate:
  enabled: true
""",
    )
    write_policy(
        new_file,
        """
abstention_gate:
  enabled: false
""",
    )
    result = run_lint(old_file, new_file)
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCK"
    disabled = [entry for entry in payload["breaking_changes"] if entry["category"] == "abstention_controls"]
    assert disabled, "Disabling abstention gating should be breaking"
