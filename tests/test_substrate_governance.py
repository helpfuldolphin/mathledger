import subprocess
import sys

import pytest

SCRIPT_PATH = "scripts/substrate_governance_check.py"
SCHEMA_PATH = "docs/governance/substrate/substrate_schema_draft07.json"

def run_script(json_path, shadow_only=True):
    """Helper to run the script and return its result."""
    command = [sys.executable, SCRIPT_PATH, json_path]
    if not shadow_only:
        command.append("--no-shadow-only")
    command.append(f"--schema-path={SCHEMA_PATH}")
    return subprocess.run(command, capture_output=True, text=True)

def test_status_green_ok():
    result = run_script("tests/fixtures/health_green.json")
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""

def test_status_yellow_ok():
    result = run_script("tests/fixtures/health_yellow.json")
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""

def test_red_shadow_mode_advisory():
    result = run_script("tests/fixtures/health_red.json", shadow_only=True)
    assert result.returncode == 0
    assert "SHADOW-ONLY MODE ENABLED" in result.stderr
    assert "SUBSTRATE GOVERNANCE CHECK: FAILED (RED)" in result.stderr
    assert "EXITING 0 DUE TO SHADOW-ONLY MODE" in result.stderr

def test_red_no_shadow_mode_fail():
    result = run_script("tests/fixtures/health_red.json", shadow_only=False)
    assert result.returncode == 64
    assert "SUBSTRATE GOVERNANCE CHECK: FAILED (RED)" in result.stderr
    assert "SHADOW-ONLY" not in result.stderr

def test_block_shadow_mode_advisory():
    result = run_script("tests/fixtures/health_block.json", shadow_only=True)
    assert result.returncode == 0
    assert "SHADOW-ONLY MODE ENABLED" in result.stderr
    assert "SUBSTRATE GOVERNANCE CHECK: FAILED (BLOCK)" in result.stderr
    assert "EXITING 0 DUE TO SHADOW-ONLY MODE" in result.stderr

def test_block_no_shadow_mode_fail():
    result = run_script("tests/fixtures/health_block.json", shadow_only=False)
    assert result.returncode == 65
    assert "SUBSTRATE GOVERNANCE CHECK: FAILED (BLOCK)" in result.stderr
    assert "SHADOW-ONLY" not in result.stderr

def test_invalid_schema_fails():
    result = run_script("tests/fixtures/health_invalid_schema.json")
    assert result.returncode == 1
    assert "Schema validation failed" in result.stderr

def test_no_substrate_key_fails():
    result = run_script("tests/fixtures/health_no_substrate.json")
    assert result.returncode == 1
    assert "Schema validation failed" in result.stderr

def test_file_not_found_fails():
    result = run_script("tests/fixtures/non_existent_file.json")
    assert result.returncode == 1
    assert "Error: File not found" in result.stderr

def test_invalid_json_fails():
    with open("tests/fixtures/invalid.json", "w") as f:
        f.write("{ not valid json }")
    result = run_script("tests/fixtures/invalid.json")
    assert result.returncode == 1
    assert "Error: Invalid JSON" in result.stderr

