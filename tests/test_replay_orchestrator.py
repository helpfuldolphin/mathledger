# tests/test_replay_orchestrator.py
import json
import os
import subprocess
import sys
import pytest

# Note: The test runner will execute this from the root directory.
SCRIPT_PATH = "scripts/replay_governance_orchestrator.py"

@pytest.fixture
def mock_rules(tmp_path):
    """Creates a mock rules file for tests."""
    rules_path = tmp_path / "replay_criticality_rules.yaml"
    rules_path.write_text("""
thresholds:
  min_determinism_rate: 99.5
  max_drift_metric: 0.10
""")
    # Change CWD for the test, as the script expects the rules file in CWD
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(original_cwd)


@pytest.fixture
def input_dir(tmp_path):
    """Provides a temporary directory for test inputs."""
    d = tmp_path / "input"
    d.mkdir()
    return d

def run_orchestrator(args):
    """Helper to run the script as a subprocess."""
    base_command = [sys.executable, SCRIPT_PATH]
    return subprocess.run(base_command + args, capture_output=True, text=True, check=False)

def write_component(path, name, determinism, drift, version="1.0"):
    """Helper to write a component metric file."""
    with open(path, 'w') as f:
        json.dump({
            "schema_version": version,
            "name": name,
            "determinism_rate": determinism,
            "drift_metric": drift
        }, f)

def test_stable_case(input_dir, tmp_path, mock_rules):
    """(a) Tests a successful run that should result in a STABLE verdict."""
    output_file = tmp_path / "output.json"
    write_component(input_dir / "comp_a.json", "component-a", 100.0, 0.05)
    write_component(input_dir / "comp_b.json", "component-b", 99.8, 0.01)

    result = run_orchestrator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert output_file.exists()
    
    with open(output_file, 'r') as f:
        snapshot = json.load(f)

    assert snapshot["radar_status"] == "STABLE"
    assert snapshot["promotion_eval"]["verdict"] == "promotion_ok"
    assert snapshot["determinism_rate"] == 99.9

def test_empty_directory_case(input_dir, tmp_path, mock_rules):
    """(b) Tests the case where no valid component files are found."""
    output_file = tmp_path / "output.json"

    result = run_orchestrator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert output_file.exists()

    with open(output_file, 'r') as f:
        snapshot = json.load(f)

    assert snapshot["radar_status"] == "UNSTABLE"
    assert snapshot["promotion_eval"]["verdict"] == "BLOCK"
    assert snapshot["promotion_eval"]["reasons"] == ["no component data found"]

def test_malformed_json_ignored(input_dir, tmp_path, mock_rules):
    """(c) Tests that a malformed JSON file is skipped and a warning is printed."""
    output_file = tmp_path / "output.json"
    write_component(input_dir / "comp_a.json", "component-a", 100.0, 0.05)
    
    malformed_file = input_dir / "invalid.json"
    malformed_file.write_text("{ not json }")

    result = run_orchestrator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert "Warning: Skipping malformed JSON file" in result.stderr
    
    with open(output_file, 'r') as f:
        snapshot = json.load(f)
    
    assert len(snapshot["components"]) == 1
    assert snapshot["components"][0]["name"] == "component-a"

def test_dry_run_no_file_written(input_dir, tmp_path, mock_rules):
    """(d) Tests that --dry-run prints to stdout and does not write a file."""
    output_file = tmp_path / "output.json"
    write_component(input_dir / "comp_a.json", "component-a", 100.0, 0.05)

    result = run_orchestrator(["--input-dir", str(input_dir), "--output-file", str(output_file), "--dry-run"])

    assert result.returncode == 0
    assert not output_file.exists()
    assert "DRY RUN MODE" in result.stdout
    
    # Verify the stdout contains valid JSON
    json_output = "".join(result.stdout.splitlines()[1:]) # Remove banner
    snapshot = json.loads(json_output)
    assert snapshot["radar_status"] == "STABLE"

def test_deterministic_ordering(input_dir, tmp_path, mock_rules):
    """(e) Tests that components are sorted alphabetically in the output."""
    output_file = tmp_path / "output.json"
    write_component(input_dir / "z_comp.json", "z-component", 100.0, 0.01)
    write_component(input_dir / "a_comp.json", "a-component", 100.0, 0.01)
    write_component(input_dir / "b_comp.json", "b-component", 100.0, 0.01)

    result = run_orchestrator(["--input-dir", str(input_dir), "--output-file", str(output_file)])
    
    assert result.returncode == 0
    with open(output_file, 'r') as f:
        snapshot = json.load(f)
        
    component_names = [c["name"] for c in snapshot["components"]]
    assert component_names == ["a-component", "b-component", "z-component"]
