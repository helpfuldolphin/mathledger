# tests/hash_observatory/test_history_and_ci.py
import subprocess
import json
from pathlib import Path
import pytest
import sys

@pytest.fixture
def mock_auditor_script(tmp_path: Path):
    p = tmp_path / "mock_auditor.py"
    p.write_text('import sys, json; print(json.dumps({"status": "OK"})); sys.exit(0)')
    return p

def test_history_builder_script_execution(tmp_path: Path, mock_auditor_script: Path):
    project_root_for_test = tmp_path
    artifacts_dir = project_root_for_test / "artifacts" / "hash_observatory"
    history_log = artifacts_dir / "history.jsonl"
    config_dir = project_root_for_test / "config"
    
    config_dir.mkdir()
    (config_dir / "curriculum_uplift_phase2_hashed.yaml").touch()
    (project_root_for_test / "PREREG_UPLIFT_U2.yaml").touch()
    (project_root_for_test / "execution_manifest.json").touch()

    # Correct path to the script relative to the project root
    real_history_script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "hash_observatory_history.py"

    result = subprocess.run([
        sys.executable, str(real_history_script_path),
        "--auditor-script", str(mock_auditor_script),
        "--project-root", str(project_root_for_test)
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"History script failed: {result.stderr}"
    assert history_log.exists(), "History log file was not created."
    assert len(history_log.read_text().strip().splitlines()) == 1
