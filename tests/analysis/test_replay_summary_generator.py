# tests/analysis/test_replay_summary_generator.py
import json
import os
import subprocess
import sys
import pytest
import datetime # For mocking timestamp if needed, but subprocess is fine

# Add script directory to path to allow import
SCRIPT_PATH = "analysis/replay_summary_generator.py"

@pytest.fixture
def input_dir(tmp_path):
    """Provides a temporary directory for test inputs."""
    d = tmp_path / "input"
    d.mkdir()
    return d

def run_generator(args, cwd=None):
    """Helper to run the script as a subprocess."""
    base_command = [sys.executable, SCRIPT_PATH]
    return subprocess.run(base_command + args, capture_output=True, text=True, check=False, cwd=cwd)

def write_component(path, name, determinism, drift, version="1.0"):
    """Helper to write a component metric file."""
    with open(path, 'w') as f:
        json.dump({
            "schema_version": version,
            "name": name,
            "determinism_rate": determinism,
            "drift_metric": drift
        }, f)

def test_summary_output_and_fields(input_dir, tmp_path):
    """
    Tests that the summary generator produces the correct output format,
    includes required fields, and excludes forbidden verdict fields.
    """
    output_file = tmp_path / "summary.json"

    # Create two component files with deterministic names
    write_component(input_dir / "comp_b.json", "component-b", 99.0, 0.08)
    write_component(input_dir / "comp_a.json", "component-a", 100.0, 0.02)

    result = run_generator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert output_file.exists()

    with open(output_file, 'r') as f:
        summary = json.load(f)
    
    # Assert required top-level fields
    assert summary["schema_version"] == "1.0"
    assert summary["mode"] == "ANALYSIS"
    assert summary["scope_note"] == "NOT_GOVERNANCE_NOT_GATING"
    assert "run_id" in summary
    assert "timestamp_utc" in summary
    assert "summary_statistics" in summary
    assert "components" in summary

    # Assert no forbidden verdict fields
    assert "radar_status" not in summary
    assert "promotion_eval" not in summary
    for component in summary["components"]:
        assert "is_blocking" not in component

    # Assert deterministic ordering of components
    component_names = [c["name"] for c in summary["components"]]
    assert component_names == ["component-a", "component-b"]

    # Assert statistical calculations (basic check)
    assert summary["summary_statistics"]["component_count"] == 2
    assert summary["summary_statistics"]["determinism_rate"]["mean"] == 99.5
    assert summary["summary_statistics"]["drift_metric"]["mean"] == 0.05
    assert summary["summary_statistics"]["determinism_rate"]["min"] == 99.0
    assert summary["summary_statistics"]["determinism_rate"]["max"] == 100.0

def test_empty_input_directory(input_dir, tmp_path):
    """Tests behavior with an empty input directory."""
    output_file = tmp_path / "summary_empty.json"
    result = run_generator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert output_file.exists()

    with open(output_file, 'r') as f:
        summary = json.load(f)

    assert summary["summary_statistics"]["component_count"] == 0
    assert summary["summary_statistics"]["determinism_rate"] == {"mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    assert summary["components"] == []

def test_malformed_json_ignored(input_dir, tmp_path):
    """Tests that malformed JSON files are ignored."""
    output_file = tmp_path / "summary_malformed.json"
    write_component(input_dir / "valid_comp.json", "valid-component", 99.9, 0.03)
    (input_dir / "malformed.json").write_text("{ this is not valid json")

    result = run_generator(["--input-dir", str(input_dir), "--output-file", str(output_file)])

    assert result.returncode == 0
    assert "Warning: Skipping malformed JSON file" in result.stderr
    
    with open(output_file, 'r') as f:
        summary = json.load(f)
    
    assert summary["summary_statistics"]["component_count"] == 1
    assert summary["components"][0]["name"] == "valid-component"

