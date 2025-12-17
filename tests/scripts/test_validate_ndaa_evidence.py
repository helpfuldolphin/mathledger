# tests/scripts/test_validate_ndaa_evidence.py

import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

# Ensure the scripts directory is in the Python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from validate_ndaa_evidence import validate_checklist, main # Import the function directly for testing

# Path to the schema for the checklist itself
CHECKLIST_SCHEMA_PATH = os.path.abspath("docs/system_law/schemas/ndaa_evidence_checklist.schema.json")
# Path to the machine-readable checklist JSON (used as input to validator)
CHECKLIST_JSON_PATH = os.path.abspath("docs/system_law/ndaa_evidence_checklist.json")

@pytest.fixture
def temp_evidence_pack(tmp_path):
    """Creates a temporary evidence pack directory with some mock files."""
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()

    (pack_dir / "audit_p3").mkdir()
    (pack_dir / "audit_p4").mkdir()

    # Create a minimal set of expected files for a passing scenario
    (pack_dir / "audit_p3" / "first_light_red_flag_matrix.json").write_text(
        json.dumps({"summary": {"total_flags": 1}, "schema_version": "1.0.0"})
    )
    (pack_dir / "audit_p3" / "first_light_stability_report.json").write_text(
        json.dumps({"criteria_evaluation": {"all_passed": True}, "schema_version": "1.0.0"})
    )
    (pack_dir / "audit_p4" / "p4_divergence_log.jsonl").write_text(
        '{"action": "LOGGED_ONLY", "cycle": 1, "timestamp": "2025-01-01T00:00:00Z", "twin_delta_p": 0.1, "real_delta_p": 0.1, "divergence": 0, "divergence_pct": 0, "severity": "NONE"}\n'
    )
    
    # Mock verify_config_hashes script for testing purposes
    mock_verify_config_hashes_script = tmp_path / "verify_config_hashes.py"
    mock_verify_config_hashes_script.write_text(
        """import sys; sys.exit(0) # Always succeed for test"""
    )
    # Mock ledgerctl.py
    mock_ledgerctl_script = tmp_path / "ledgerctl.py"
    mock_ledgerctl_script.write_text(
        """import sys; sys.exit(0) # Always succeed for test"""
    )
    # Mock test_replay_safety_governance_signal.py
    mock_replay_safety_test = tmp_path / "tests" / "test_replay_safety_governance_signal.py"
    mock_replay_safety_test.parent.mkdir(parents=True, exist_ok=True)
    mock_replay_safety_test.write_text(
        """import pytest; def test_signal_fusion(): assert True"""
    )
    # Mock test_slice_drift_guard.py
    mock_drift_guard_test = tmp_path / "curriculum" / "test_slice_drift_guard.py"
    mock_drift_guard_test.parent.mkdir(parents=True, exist_ok=True)
    mock_drift_guard_test.write_text(
        """import pytest; def test_drift_detection(): assert True"""
    )
    return pack_dir

@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mocks subprocess.run to allow controlling command outcomes."""
    def _mock_run(command, **kwargs):
        # For testing purposes, we assume specific commands pass.
        # In a real scenario, this would be more sophisticated.
        if "validate_ndaa_evidence.py" in command and "--check-schema" in command:
            # This is a specific check-schema call within the validator itself
            # For tests, we assume it passes if the path is valid
            if "first_light_red_flag_matrix.schema.json" in command or \
               "first_light_stability_report.schema.json" in command or \
               "p4_divergence_log.schema.json" in command:
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="Schema check passed", stderr="")
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="Schema check failed (mock)")
        
        # For other commands, just return success
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="Mock command output", stderr="")
    monkeypatch.setattr(subprocess, "run", _mock_run)


def test_validate_ndaa_evidence_full_success(temp_evidence_pack, monkeypatch, capsys):
    """Tests a full successful validation run."""
    # Adjust CHECKLIST_JSON_PATH to point to a test-specific version if needed
    # For now, we assume the system-wide one is sufficient.
    
    # Mock calls to `python verify_config_hashes.py` to succeed
    monkeypatch.setattr(os, 'path.exists', lambda path: True)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout='mocked success', stderr=''))

    report_path = temp_evidence_pack / "report.json"
    
    # The validate_checklist function calls sys.exit, so we need to catch that.
    with pytest.raises(SystemExit) as excinfo:
        validate_checklist(CHECKLIST_JSON_PATH, str(temp_evidence_pack), str(report_path))

    assert excinfo.value.code == 0 # Expect success exit code

    # Verify report content
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["summary"]["status"] == "SUCCESS"
    assert report["summary"]["passed_steps"] == report["summary"]["total_steps"]

    # Check console output for color codes (basic check)
    captured = capsys.readouterr()
    assert "SUCCESS: All evidence checklist items passed." in captured.stdout


def test_validate_ndaa_evidence_missing_artifact(temp_evidence_pack, monkeypatch, capsys):
    """Tests validation with a missing artifact and ensures it reports failure without crashing."""
    # Simulate a missing file by mocking os.path.exists
    original_exists = os.path.exists
    def mock_exists(path):
        if "first_light_red_flag_matrix.json" in path:
            return False # This file will be 'missing'
        return original_exists(path)

    monkeypatch.setattr(os, 'path.exists', mock_exists)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout='mocked success', stderr=''))

    report_path = temp_evidence_pack / "report_missing.json"

    with pytest.raises(SystemExit) as excinfo:
        validate_checklist(CHECKLIST_JSON_PATH, str(temp_evidence_pack), str(report_path))

    assert excinfo.value.code == 1 # Expect failure exit code

    # Verify report content
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["summary"]["status"] == "FAILURE"
    assert report["summary"]["failed_steps"] > 0
    
    # Find the specific step that should have failed due to missing artifact
    failed_step = next( (s for s in report["results"] if s["step_id"] == "1.1"), None)
    assert failed_step is not None
    assert failed_step["status"] == "FAILURE"
    assert any("Artifact NOT found." in c["details"] for c in failed_step["checks"] if c["check"] == "artifact_existence")

    captured = capsys.readouterr()
    assert "FAILURE: One or more evidence checklist items failed." in captured.stdout


def test_validate_ndaa_evidence_command_failure(temp_evidence_pack, monkeypatch, capsys):
    """Tests validation where a verification command fails."""
    def mock_run_fail(command, **kwargs):
        if "--check-field-equals" in command: # Mock a specific verification command to fail
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="mocked failure")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="mocked success", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run_fail)
    monkeypatch.setattr(os, 'path.exists', lambda path: True)

    report_path = temp_evidence_pack / "report_command_fail.json"

    with pytest.raises(SystemExit) as excinfo:
        validate_checklist(CHECKLIST_JSON_PATH, str(temp_evidence_pack), str(report_path))

    assert excinfo.value.code == 1 # Expect failure exit code

    # Verify report content
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["summary"]["status"] == "FAILURE"
    assert report["summary"]["failed_steps"] > 0

    # Find the specific step that should have failed due to command failure (step 1.3)
    failed_step = next( (s for s in report["results"] if s["step_id"] == "1.3"), None)
    assert failed_step is not None
    assert failed_step["status"] == "FAILURE"
    assert any("Command failed" in c["details"] for c in failed_step["checks"] if c["check"] == "verification_command")

    captured = capsys.readouterr()
    assert "FAILURE: One or more evidence checklist items failed." in captured.stdout


def test_validate_ndaa_evidence_with_manifest_arg(temp_evidence_pack, monkeypatch):
    """Tests the --manifest argument path."""
    manifest_file = temp_evidence_pack / "test_manifest.json"
    manifest_file.write_text('{}') # Content doesn't matter for this test

    # Mock the main function to capture sys.exit
    monkeypatch.setattr(sys, "exit", lambda x: None) 
    
    # Mock all subprocess calls to succeed for this path test
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout='mocked success', stderr=''))
    monkeypatch.setattr(os, 'path.exists', lambda path: True) # All artifacts exist

    # Simulate command-line arguments
    monkeypatch.setattr(sys, 'argv', [
        "validate_ndaa_evidence.py", 
        "--manifest", str(manifest_file),
        "--json-report-out", str(temp_evidence_pack / "report_manifest.json")
    ])

    main()
    # No SystemExit is raised due to mocking, but the function should have run
    # We can check the report file to confirm execution
    report_path = temp_evidence_pack / "report_manifest.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["summary"]["status"] == "SUCCESS"
    assert report["summary"]["passed_steps"] > 0


