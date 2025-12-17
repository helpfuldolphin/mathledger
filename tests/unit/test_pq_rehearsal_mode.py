#!/usr/bin/env python3
"""
# REAL-READY

Test Suite for PQ Activation Rehearsal Mode

This test suite validates that the rehearsal mode produces deterministic outputs
for all failure scenarios. Each scenario has 2 tests: one for exit code and one
for JSON report structure.

Author: Manus-H
Date: 2025-12-11
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Test configuration
SCRIPT_PATH = "scripts/pq_activation_dryrun.py"
FIXTURE_PATH = "tests/fixtures/pq_rehearsal_scenarios.json"

def run_rehearsal(scenario: str, output_file: str) -> tuple[int, Dict]:
    """
    Run the dry-run script in rehearsal mode and return exit code and report.
    
    Args:
        scenario: Scenario name (success, missing_module, etc.)
        output_file: Path to output JSON file
    
    Returns:
        Tuple of (exit_code, report_dict)
    """
    cmd = [
        sys.executable,
        SCRIPT_PATH,
        "--rehearsal",
        "--scenario", scenario,
        "--output", output_file,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Load the generated report
    with open(output_file, 'r') as f:
        report = json.load(f)
    
    return result.returncode, report

def test_success_scenario_exit_code():
    """Test 1: Success scenario returns exit code 0."""
    print("Running Test 1: Success scenario exit code...")
    
    exit_code, report = run_rehearsal("success", "/tmp/test_success.json")
    
    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    print("✓ Test 1 PASSED: Exit code is 0")

def test_success_scenario_report():
    """Test 2: Success scenario report has correct structure."""
    print("Running Test 2: Success scenario report structure...")
    
    exit_code, report = run_rehearsal("success", "/tmp/test_success.json")
    
    # Validate report structure
    assert report["checks_passed"] == 6, f"Expected 6 checks passed, got {report['checks_passed']}"
    assert report["checks_failed"] == 0, f"Expected 0 checks failed, got {report['checks_failed']}"
    assert report["warnings"] == 0, f"Expected 0 warnings, got {report['warnings']}"
    assert report["ready_for_activation"] is True, "Expected ready_for_activation=True"
    assert "failure_reason" not in report, "Success scenario should not have failure_reason"
    
    print("✓ Test 2 PASSED: Report structure is correct")

def test_missing_module_exit_code():
    """Test 3: Missing module scenario returns exit code 1."""
    print("Running Test 3: Missing module scenario exit code...")
    
    exit_code, report = run_rehearsal("missing_module", "/tmp/test_missing_module.json")
    
    assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
    print("✓ Test 3 PASSED: Exit code is 1")

def test_missing_module_report():
    """Test 4: Missing module scenario report has correct structure."""
    print("Running Test 4: Missing module scenario report structure...")
    
    exit_code, report = run_rehearsal("missing_module", "/tmp/test_missing_module.json")
    
    # Validate report structure
    assert report["checks_passed"] == 5, f"Expected 5 checks passed, got {report['checks_passed']}"
    assert report["checks_failed"] == 1, f"Expected 1 check failed, got {report['checks_failed']}"
    assert report["warnings"] == 0, f"Expected 0 warnings, got {report['warnings']}"
    assert report["ready_for_activation"] is False, "Expected ready_for_activation=False"
    assert report["failure_reason"] == "missing_pq_modules", f"Expected failure_reason='missing_pq_modules', got '{report.get('failure_reason')}'"
    
    print("✓ Test 4 PASSED: Report structure is correct")

def test_drift_radar_disabled_exit_code():
    """Test 5: Drift radar disabled scenario returns exit code 1."""
    print("Running Test 5: Drift radar disabled scenario exit code...")
    
    exit_code, report = run_rehearsal("drift_radar_disabled", "/tmp/test_drift_radar.json")
    
    assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
    print("✓ Test 5 PASSED: Exit code is 1")

def test_drift_radar_disabled_report():
    """Test 6: Drift radar disabled scenario report has correct structure."""
    print("Running Test 6: Drift radar disabled scenario report structure...")
    
    exit_code, report = run_rehearsal("drift_radar_disabled", "/tmp/test_drift_radar.json")
    
    # Validate report structure
    assert report["checks_passed"] == 5, f"Expected 5 checks passed, got {report['checks_passed']}"
    assert report["checks_failed"] == 1, f"Expected 1 check failed, got {report['checks_failed']}"
    assert report["warnings"] == 1, f"Expected 1 warning, got {report['warnings']}"
    assert report["ready_for_activation"] is False, "Expected ready_for_activation=False"
    assert report["failure_reason"] == "drift_radar_disabled", f"Expected failure_reason='drift_radar_disabled', got '{report.get('failure_reason')}'"
    
    print("✓ Test 6 PASSED: Report structure is correct")

def test_low_disk_space_exit_code():
    """Test 7: Low disk space scenario returns exit code 1."""
    print("Running Test 7: Low disk space scenario exit code...")
    
    exit_code, report = run_rehearsal("low_disk_space", "/tmp/test_low_disk.json")
    
    assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
    print("✓ Test 7 PASSED: Exit code is 1")

def test_low_disk_space_report():
    """Test 8: Low disk space scenario report has correct structure."""
    print("Running Test 8: Low disk space scenario report structure...")
    
    exit_code, report = run_rehearsal("low_disk_space", "/tmp/test_low_disk.json")
    
    # Validate report structure
    assert report["checks_passed"] == 5, f"Expected 5 checks passed, got {report['checks_passed']}"
    assert report["checks_failed"] == 1, f"Expected 1 check failed, got {report['checks_failed']}"
    assert report["warnings"] == 0, f"Expected 0 warnings, got {report['warnings']}"
    assert report["ready_for_activation"] is False, "Expected ready_for_activation=False"
    assert report["failure_reason"] == "insufficient_disk_space", f"Expected failure_reason='insufficient_disk_space', got '{report.get('failure_reason')}'"
    
    print("✓ Test 8 PASSED: Report structure is correct")

def main():
    """Run all tests."""
    print("=" * 70)
    print("PQ REHEARSAL MODE TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_success_scenario_exit_code,
        test_success_scenario_report,
        test_missing_module_exit_code,
        test_missing_module_report,
        test_drift_radar_disabled_exit_code,
        test_drift_radar_disabled_report,
        test_low_disk_space_exit_code,
        test_low_disk_space_report,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ TEST FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"✗ TEST ERROR: {e}")
            failed += 1
            print()
    
    print("=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
