# tests/scripts/test_scenario_simulator.py

import pytest
import subprocess
import sys
import json
from pathlib import Path

# --- Fixtures ---

@pytest.fixture
def baseline_posture_file(tmp_path: Path) -> Path:
    """Creates a temporary baseline posture file for the simulator."""
    posture_file = tmp_path / "baseline_posture.json"
    posture_data = {
        "replay_status": "OK",
        "seed_classification": "PURE",
        "last_mile_validation": "PASS",
    }
    posture_file.write_text(json.dumps(posture_data))
    return posture_file

# --- Tests ---

def run_simulator(args: list) -> subprocess.CompletedProcess:
    """Helper to run the simulator script as a subprocess."""
    base_command = [sys.executable, "scripts/security_scenario_simulator.py"]
    return subprocess.run(base_command + args, capture_output=True, text=True)

def test_simulator_no_change(baseline_posture_file: Path):
    """Test the simulator with no changes, should report no impact."""
    result = run_simulator(["--baseline-posture-file", str(baseline_posture_file)])
    
    assert result.returncode == 0
    assert "IMPACT: No change in overall security level." in result.stdout
    assert "'GREEN' -> 'GREEN'" not in result.stdout # Ensure it doesn't report a change

def test_simulator_change_to_amber(baseline_posture_file: Path):
    """Test simulating a change that results in an AMBER state."""
    result = run_simulator([
        "--baseline-posture-file", str(baseline_posture_file),
        "--set-last-mile-validation", "FAIL"
    ])
    
    assert result.returncode == 0
    assert "IMPACT: Security level changed from 'GREEN' -> 'AMBER'." in result.stdout
    assert "WARNING:" in result.stdout

def test_simulator_change_to_red(baseline_posture_file: Path):
    """Test simulating a change that results in a RED state."""
    result = run_simulator([
        "--baseline-posture-file", str(baseline_posture_file),
        "--set-replay-status", "FAIL"
    ])
    
    assert result.returncode == 0
    assert "IMPACT: Security level changed from 'GREEN' -> 'RED'." in result.stdout
    assert "CRITICAL:" in result.stdout

def test_simulator_multiple_changes(baseline_posture_file: Path):
    """Test applying multiple simulated changes at once."""
    result = run_simulator([
        "--baseline-posture-file", str(baseline_posture_file),
        "--set-replay-status", "FAIL",
        "--set-seed-classification", "DRIFT"
    ])
    
    assert result.returncode == 0
    assert "IMPACT: Security level changed from 'GREEN' -> 'RED'." in result.stdout
    
    # Verify the final state in the JSON output
    output_data = json.loads(result.stdout.split("[2] SIMULATED POSTURE (What-if)")[1].split("-----------------------------")[1].split("[3] IMPACT ANALYSIS]")[0].strip())
    assert output_data["components"]["replay_status"] == "FAIL"
    assert output_data["components"]["seed_classification"] == "DRIFT"

def test_simulator_file_not_found():
    """Test that the simulator exits gracefully if the baseline file is not found."""
    result = run_simulator(["--baseline-posture-file", "non_existent_file.json"])
    assert result.returncode == 1
    assert "Error: Baseline posture file not found" in result.stderr
