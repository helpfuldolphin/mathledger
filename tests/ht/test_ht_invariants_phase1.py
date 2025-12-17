"""
Tests for Phase 1 of the HT-Series Invariant Implementation.

This module focuses on the foundational Ht computation invariants.
"""

import pytest
import json
from pathlib import Path
from backend.ht.ht_invariant_checker import HtInvariantChecker

# Mark all tests in this file as 'ht_spec'
pytestmark = pytest.mark.ht_spec

# --- Fixtures ---

@pytest.fixture
def load_test_data():
    """Fixture factory to load test data from JSON files."""
    def _loader(filename: str):
        path = Path(__file__).parent.parent.parent / "backend" / "ht" / "test_data" / filename
        with open(path, 'r') as f:
            return json.load(f)
    return _loader

# --- TDD: Failing Tests First ---

def test_verify_ht_computation_fails_on_mismatched_hash(load_test_data):
    """
    Tests that the invariant checker fails when H_t does not match the hash of R_t and U_t.
    Corresponds to a violation of the core Ht computation rule.
    """
    # This test should fail initially because the verification logic is not implemented.
    mismatched_data = load_test_data("phase1_mismatched_ht_series.json")
    
    # We pass dummy dicts for prereg and full ht_series, as they are not needed for this specific check
    checker = HtInvariantChecker(manifest={}, prereg={}, ht_series={})
    
    # This method doesn't exist yet, so this test will fail until we implement it.
    failures = checker.verify_ht_computation(mismatched_data)
    
    assert len(failures) > 0, "Checker should have detected a failure"
    assert "HT-COMP-01" in failures[0], "Failure message should contain the correct invariant ID"
    assert "cycle 1" in failures[0], "Failure message should specify the correct cycle"

def test_verify_ht_computation_fails_on_missing_data(load_test_data):
    """
    Tests that the invariant checker fails if a cycle is missing a required root hash (R_t).
    Corresponds to a violation of data presence invariants.
    """
    # This test should also fail initially.
    missing_data = load_test_data("phase1_missing_rt_series.json")
    checker = HtInvariantChecker(manifest={}, prereg={}, ht_series={})
    
    failures = checker.verify_ht_computation(missing_data)
    
    assert len(failures) > 0, "Checker should have detected a failure"
    assert "HT-COMP-02" in failures[0], "Failure message should contain the correct invariant ID"
    assert "Missing R_t in cycle 1" in failures[0], "Failure message should be specific"

# --- Passing Test (to be written after implementation) ---

# @pytest.mark.skip(reason="Implementation not yet complete")
# def test_verify_ht_computation_passes_on_valid_data(load_test_data):
#     """
#     Tests that the invariant checker passes for a well-formed, valid series.
#     """
#     valid_data = load_test_data("phase1_valid_series.json")
#     checker = HtInvariantChecker(manifest={}, prereg={}, ht_series={})
#     
#     failures = checker.verify_ht_computation(valid_data)
#     
#     assert len(failures) == 0, f"Checker found unexpected failures: {failures}"

