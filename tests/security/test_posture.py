# tests/security/test_posture.py

import pytest
import subprocess
import sys
import json
from backend.security.posture import (
    build_security_posture,
    is_security_ok,
    get_security_level,
    summarize_security_for_governance,
    merge_into_global_health,
    SecurityPosture,
    SecurityLevel,
)

# --- Test Cases for different postures ---

# The ideal, perfect state
POSTURE_GREEN = build_security_posture(is_replay_ok=True, did_seed_drift=False, last_mile_passed=True)

# A critical failure of determinism
POSTURE_RED_REPLAY = build_security_posture(is_replay_ok=False, did_seed_drift=False, last_mile_passed=True)
POSTURE_RED_SEED = build_security_posture(is_replay_ok=True, did_seed_drift=True, last_mile_passed=True)

# A warning state, determinism holds but downstream tasks might fail
POSTURE_AMBER_VALIDATION = build_security_posture(is_replay_ok=True, did_seed_drift=False, last_mile_passed=False)

# --- Unit Tests for Governance Logic ---

def test_build_security_posture():
    assert POSTURE_GREEN["replay_status"] == "OK"
    assert POSTURE_GREEN["seed_classification"] == "PURE"
    assert POSTURE_RED_SEED["seed_classification"] == "DRIFT"
    assert POSTURE_AMBER_VALIDATION["last_mile_validation"] == "FAIL"

def test_is_security_ok():
    assert is_security_ok(POSTURE_GREEN) is True
    assert is_security_ok(POSTURE_RED_REPLAY) is False
    assert is_security_ok(POSTURE_AMBER_VALIDATION) is False

@pytest.mark.parametrize("posture, expected_level", [
    (POSTURE_GREEN, "GREEN"),
    (POSTURE_RED_REPLAY, "RED"),
    (POSTURE_RED_SEED, "RED"),
    (POSTURE_AMBER_VALIDATION, "AMBER"),
])
def test_get_security_level(posture: SecurityPosture, expected_level: SecurityLevel):
    assert get_security_level(posture) == expected_level

def test_summarize_security_for_governance():
    summary = summarize_security_for_governance(POSTURE_RED_REPLAY)
    assert summary["security_level"] == "RED"
    assert summary["is_ok"] is False
    assert "CRITICAL" in summary["narrative"]
    assert summary["components"]["replay_status"] == "FAIL"

def test_merge_into_global_health():
    global_health = {"traffic_light": "GREEN", "subsystems": {}}
    
    # Merge an AMBER status
    amber_summary = summarize_security_for_governance(POSTURE_AMBER_VALIDATION)
    global_health = merge_into_global_health(global_health, amber_summary)
    assert global_health["traffic_light"] == "AMBER"
    assert "security_determinism" in global_health["subsystems"]
    
    # Merge a GREEN status (should not demote from AMBER)
    green_summary = summarize_security_for_governance(POSTURE_GREEN)
    global_health = merge_into_global_health(global_health, green_summary)
    assert global_health["traffic_light"] == "AMBER"

    # Merge a RED status (should override AMBER)
    red_summary = summarize_security_for_governance(POSTURE_RED_SEED)
    global_health = merge_into_global_health(global_health, red_summary)
    assert global_health["traffic_light"] == "RED"

# --- Integration Tests for CLI ---

def run_cli(args: list) -> subprocess.CompletedProcess:
    """Helper to run the CLI script as a subprocess."""
    base_command = [sys.executable, "scripts/security_posture_check.py"]
    return subprocess.run(base_command + args, capture_output=True, text=True)

def test_cli_exit_code_green():
    result = run_cli(["--replay-ok", "--seed-pure", "--last-mile-pass"])
    assert result.returncode == 0
    assert "Security Level: GREEN" in result.stdout

def test_cli_exit_code_amber():
    result = run_cli(["--replay-ok", "--seed-pure"]) # Missing last-mile-pass
    assert result.returncode == 1
    assert "Security Level: AMBER" in result.stdout

def test_cli_exit_code_red():
    result = run_cli(["--seed-pure", "--last-mile-pass"]) # Missing replay-ok
    assert result.returncode == 2
    assert "Security Level: RED" in result.stdout

def test_cli_json_output():
    result = run_cli(["--replay-ok", "--seed-pure", "--json-output"])
    assert result.returncode == 1 # Amber because last-mile-pass is missing
    
    try:
        data = json.loads(result.stdout)
        assert data["security_level"] == "AMBER"
        assert data["is_ok"] is False
        assert data["components"]["replay_status"] == "OK"
    except json.JSONDecodeError:
        pytest.fail("CLI did not produce valid JSON when --json-output was used.")

# --- Scenario Classification Tests ---

from backend.security.posture import classify_security_scenario

@pytest.mark.parametrize("posture, expected_scenario", [
    (POSTURE_GREEN, "HEALTHY_GREEN"),
    (POSTURE_RED_REPLAY, "REPLAY_FAILURE_RED"),
    (POSTURE_RED_SEED, "SEED_DRIFT_RED"),
    (POSTURE_AMBER_VALIDATION, "VALIDATION_FAIL_AMBER"),
    # Test a non-canonical posture that still matches a RED pattern
    (build_security_posture(False, True, False), "REPLAY_FAILURE_RED"),
])
def test_classify_security_scenario(posture: SecurityPosture, expected_scenario: str):
    assert classify_security_scenario(posture) == expected_scenario

def test_classify_unknown_scenario():
    # An unusual posture that doesn't match any primary scenario pattern
    unknown_posture = SecurityPosture(
        replay_status="OK",
        seed_classification="UNKNOWN",
        last_mile_validation="PASS"
    )
    assert classify_security_scenario(unknown_posture) == "unknown"

# --- MAAS Adapter Tests ---

from backend.security.posture import summarize_security_for_maas_tile

def test_maas_summary_for_green_posture():
    summary = summarize_security_for_maas_tile(POSTURE_GREEN)
    assert summary == {
        "dominant_scenario_id": "HEALTHY_GREEN",
        "security_level": "GREEN",
        "is_binding_constraint": False,
    }

def test_maas_summary_for_red_posture():
    summary = summarize_security_for_maas_tile(POSTURE_RED_REPLAY)
    assert summary == {
        "dominant_scenario_id": "REPLAY_FAILURE_RED",
        "security_level": "RED",
        "is_binding_constraint": True,
    }

def test_maas_summary_for_amber_posture():
    summary = summarize_security_for_maas_tile(POSTURE_AMBER_VALIDATION)
    assert summary == {
        "dominant_scenario_id": "VALIDATION_FAIL_AMBER",
        "security_level": "AMBER",
        "is_binding_constraint": True,
    }
