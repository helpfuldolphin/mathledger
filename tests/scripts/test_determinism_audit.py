# tests/scripts/test_determinism_audit.py

import pytest
import subprocess
import sys
import json
from pathlib import Path

# --- Mock Runner Scripts ---

MOCK_DETERMINISTIC_RUNNER = """
import json
import sys
import os
from pathlib import Path

# This mock runner simulates a fully deterministic experiment.
# It always outputs the same result hash.

def main():
    # Assume the audit script provides an isolated output dir via env var
    output_dir = Path(os.environ.get("RFL_ARTIFACTS_DIR", "tmp/default_mock_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    
    results_data = {
        "policy": {
            "ledger": [
                {}, # dummy entry
                { "composite_root": "DETERMINISTIC_HASH_VALUE_SUCCESS" }
            ]
        }
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f)
    
    print("Mock Deterministic Runner: Run complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
"""

MOCK_NONDETERMINISTIC_RUNNER = """
import json
import sys
import os
import uuid
from pathlib import Path

# This mock runner simulates a non-deterministic experiment.
# It generates a new unique hash for each run.

def main():
    output_dir = Path(os.environ.get("RFL_ARTIFACTS_DIR", "tmp/default_mock_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    
    results_data = {
        "policy": {
            "ledger": [
                { "composite_root": str(uuid.uuid4()) }
            ]
        }
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f)
    
    print("Mock Non-Deterministic Runner: Run complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
"""

# --- Fixtures ---

@pytest.fixture
def mock_runner(tmp_path: Path):
    """Fixture to create mock runner files."""
    deterministic_script = tmp_path / "deterministic_runner.py"
    nondeterministic_script = tmp_path / "nondeterministic_runner.py"
    
    deterministic_script.write_text(MOCK_DETERMINISTIC_RUNNER)
    nondeterministic_script.write_text(MOCK_NONDETERMINISTIC_RUNNER)
    
    # Also create a dummy manifest
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text('{"experiment_id": "mock_test"}')
    
    return {
        "deterministic": deterministic_script,
        "nondeterministic": nondeterministic_script,
        "manifest": manifest_file,
    }

# --- Tests ---

def test_audit_script_reports_deterministic(mock_runner, monkeypatch):
    """
    Verify the audit script correctly identifies a deterministic process.
    """
    # We monkeypatch the command to call our mock runner instead of the real one.
    # This is a bit complex, so an easier way is to just point the audit script
    # to our mock runner file. The audit script calls `rfl/runner.py`. We can
    # temporarily replace it. A cleaner way for a test would be to make the
    # runner script an argument to the audit script.
    
    # For this test, we'll patch the subprocess call.
    # But a simpler method is to just run the test from a directory where
    # `rfl/runner.py` is our mock script. Let's do that.
    
    # Let's create a temporary `rfl` directory and place our mock runner there.
    rfl_dir = mock_runner["deterministic"].parent / "rfl"
    rfl_dir.mkdir()
    (rfl_dir / "runner.py").write_text(MOCK_DETERMINISTIC_RUNNER)

    audit_script_path = Path.cwd() / "scripts/check_determinism_over_history.py"
    
    # Run the audit script from the temp directory where the mock runner is located
    result = subprocess.run(
        [sys.executable, str(audit_script_path), "--manifest", str(mock_runner["manifest"]), "--runs", "2"],
        capture_output=True,
        text=True,
        cwd=mock_runner["deterministic"].parent
    )
    
    assert result.returncode == 0, f"Audit script failed unexpectedly. Stderr:\n{result.stderr}"
    assert "Audit Verdict: DETERMINISTIC" in result.stdout
    assert "DIVERGENCE" not in result.stdout

def test_audit_script_reports_nondeterministic(mock_runner):
    """
    Verify the audit script correctly identifies a non-deterministic process.
    """
    rfl_dir = mock_runner["nondeterministic"].parent / "rfl"
    rfl_dir.mkdir(exist_ok=True)
    (rfl_dir / "runner.py").write_text(MOCK_NONDETERMINISTIC_RUNNER)

    audit_script_path = Path.cwd() / "scripts/check_determinism_over_history.py"
    
    result = subprocess.run(
        [sys.executable, str(audit_script_path), "--manifest", str(mock_runner["manifest"]), "--runs", "2"],
        capture_output=True,
        text=True,
        cwd=mock_runner["nondeterministic"].parent
    )
    
    assert result.returncode == 1, "Audit script should have failed but passed."
    assert "Audit Verdict: NONDETERMINISTIC" in result.stderr
    assert "DIVERGENCE" in result.stdout
