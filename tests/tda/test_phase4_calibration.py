"""
Phase IV Calibration Tests

Operation CORTEX: Phase IV Validation
======================================

Tests for golden set calibration regression, ensuring the hard gate
doesn't silently over- or under-block.

Test Categories:
1. Perfect calibration (no errors)
2. Over-blocking (too many false blocks)
3. Under-blocking (too many false passes)
4. Deterministic behavior
5. Edge cases
"""

import pytest
from typing import List
from unittest.mock import Mock

from backend.tda.governance import (
    LabeledTDAResult,
    CalibrationResult,
    evaluate_hard_gate_calibration,
    TDA_CALIBRATION_SCHEMA_VERSION,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def perfect_golden_set() -> List[LabeledTDAResult]:
    """
    Golden set with perfect calibration.

    All OK runs have HSS >= 0.2 (won't be blocked)
    All BLOCK runs have HSS < 0.2 (will be blocked)
    """
    return [
        # OK runs (HSS >= 0.2)
        LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
        LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
        LabeledTDAResult(hss=0.6, sns=0.5, pcs=0.4, drs=0.3, expected_label="OK"),
        LabeledTDAResult(hss=0.5, sns=0.4, pcs=0.3, drs=0.4, expected_label="OK"),
        LabeledTDAResult(hss=0.3, sns=0.3, pcs=0.2, drs=0.5, expected_label="OK"),
        # BLOCK runs (HSS < 0.2)
        LabeledTDAResult(hss=0.15, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.1, sns=0.15, pcs=0.05, drs=0.7, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.05, sns=0.1, pcs=0.02, drs=0.8, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.08, sns=0.12, pcs=0.03, drs=0.75, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.12, sns=0.18, pcs=0.08, drs=0.65, expected_label="BLOCK"),
    ]


@pytest.fixture
def over_blocking_golden_set() -> List[LabeledTDAResult]:
    """
    Golden set where hard gate would over-block (false blocks).

    Some OK runs have HSS < 0.2 (would be incorrectly blocked)
    """
    return [
        # OK runs that would be incorrectly blocked (HSS < 0.2)
        LabeledTDAResult(hss=0.15, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
        LabeledTDAResult(hss=0.18, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
        LabeledTDAResult(hss=0.12, sns=0.5, pcs=0.4, drs=0.3, expected_label="OK"),
        # OK runs that pass correctly
        LabeledTDAResult(hss=0.5, sns=0.4, pcs=0.3, drs=0.4, expected_label="OK"),
        LabeledTDAResult(hss=0.6, sns=0.3, pcs=0.2, drs=0.5, expected_label="OK"),
        # BLOCK runs (correctly blocked)
        LabeledTDAResult(hss=0.1, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.05, sns=0.15, pcs=0.05, drs=0.7, expected_label="BLOCK"),
    ]


@pytest.fixture
def under_blocking_golden_set() -> List[LabeledTDAResult]:
    """
    Golden set where hard gate would under-block (false passes).

    Some BLOCK runs have HSS >= 0.2 (would be incorrectly allowed)
    """
    return [
        # OK runs (correctly allowed)
        LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
        LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
        # BLOCK runs that would be incorrectly allowed (HSS >= 0.2)
        LabeledTDAResult(hss=0.25, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.22, sns=0.15, pcs=0.05, drs=0.7, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.3, sns=0.1, pcs=0.02, drs=0.8, expected_label="BLOCK"),
        # BLOCK runs correctly blocked
        LabeledTDAResult(hss=0.1, sns=0.12, pcs=0.03, drs=0.75, expected_label="BLOCK"),
        LabeledTDAResult(hss=0.08, sns=0.18, pcs=0.08, drs=0.65, expected_label="BLOCK"),
    ]


# ============================================================================
# 1. Perfect Calibration Tests
# ============================================================================

class TestPerfectCalibration:
    """Tests for perfectly calibrated golden sets."""

    def test_perfect_calibration_returns_ok_status(self, perfect_golden_set):
        """Perfect calibration should return OK status."""
        result = evaluate_hard_gate_calibration(perfect_golden_set)

        assert result.calibration_status == "OK"
        assert result.false_block_rate == 0.0
        assert result.false_pass_rate == 0.0

    def test_perfect_calibration_counts_are_correct(self, perfect_golden_set):
        """Perfect calibration should have correct counts."""
        result = evaluate_hard_gate_calibration(perfect_golden_set)

        assert result.n_runs == 10
        assert result.n_expected_ok == 5
        assert result.n_expected_block == 5
        assert result.actual_ok == 5
        assert result.actual_block == 5
        assert result.false_block_count == 0
        assert result.false_pass_count == 0

    def test_perfect_calibration_has_schema_version(self, perfect_golden_set):
        """Calibration result should include schema version."""
        result = evaluate_hard_gate_calibration(perfect_golden_set)

        assert result.schema_version == TDA_CALIBRATION_SCHEMA_VERSION

    def test_perfect_calibration_has_notes(self, perfect_golden_set):
        """Calibration result should include explanatory notes."""
        result = evaluate_hard_gate_calibration(perfect_golden_set)

        assert len(result.notes) > 0
        assert any("acceptable" in note.lower() for note in result.notes)


# ============================================================================
# 2. Over-Blocking Tests
# ============================================================================

class TestOverBlocking:
    """Tests for over-blocking scenarios (false blocks)."""

    def test_high_false_block_rate_returns_broken(self, over_blocking_golden_set):
        """High false block rate should return BROKEN status."""
        result = evaluate_hard_gate_calibration(over_blocking_golden_set)

        # 3 out of 5 OK runs would be blocked = 60% false block rate
        assert result.calibration_status == "BROKEN"
        assert result.false_block_rate > 0.15

    def test_moderate_false_block_rate_returns_drifting(self):
        """Moderate false block rate should return DRIFTING status."""
        golden = [
            # 1 out of 10 OK runs incorrectly blocked = 10%
            LabeledTDAResult(hss=0.15, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
            LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
            LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
            LabeledTDAResult(hss=0.6, sns=0.5, pcs=0.4, drs=0.3, expected_label="OK"),
            LabeledTDAResult(hss=0.5, sns=0.4, pcs=0.3, drs=0.4, expected_label="OK"),
            LabeledTDAResult(hss=0.4, sns=0.3, pcs=0.2, drs=0.5, expected_label="OK"),
            LabeledTDAResult(hss=0.3, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.25, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.22, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.21, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            # BLOCK runs
            LabeledTDAResult(hss=0.1, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
        ]

        result = evaluate_hard_gate_calibration(golden)

        assert result.calibration_status == "DRIFTING"
        assert 0.05 < result.false_block_rate <= 0.15

    def test_false_block_count_tracked(self, over_blocking_golden_set):
        """False block count should be tracked correctly."""
        result = evaluate_hard_gate_calibration(over_blocking_golden_set)

        assert result.false_block_count == 3  # 3 OK runs incorrectly blocked


# ============================================================================
# 3. Under-Blocking Tests
# ============================================================================

class TestUnderBlocking:
    """Tests for under-blocking scenarios (false passes)."""

    def test_high_false_pass_rate_returns_broken(self, under_blocking_golden_set):
        """High false pass rate should return BROKEN status."""
        result = evaluate_hard_gate_calibration(under_blocking_golden_set)

        # 3 out of 5 BLOCK runs would be allowed = 60% false pass rate
        assert result.calibration_status == "BROKEN"
        assert result.false_pass_rate > 0.15

    def test_moderate_false_pass_rate_returns_drifting(self):
        """Moderate false pass rate should return DRIFTING status."""
        golden = [
            # OK runs
            LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
            LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
            # 1 out of 10 BLOCK runs incorrectly allowed = 10%
            LabeledTDAResult(hss=0.25, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.1, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.08, sns=0.15, pcs=0.05, drs=0.7, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.05, sns=0.1, pcs=0.02, drs=0.8, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.12, sns=0.12, pcs=0.03, drs=0.75, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.15, sns=0.18, pcs=0.08, drs=0.65, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.11, sns=0.14, pcs=0.04, drs=0.72, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.09, sns=0.11, pcs=0.03, drs=0.78, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.07, sns=0.09, pcs=0.02, drs=0.82, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.06, sns=0.08, pcs=0.01, drs=0.85, expected_label="BLOCK"),
        ]

        result = evaluate_hard_gate_calibration(golden)

        assert result.calibration_status == "DRIFTING"
        assert 0.05 < result.false_pass_rate <= 0.15

    def test_false_pass_count_tracked(self, under_blocking_golden_set):
        """False pass count should be tracked correctly."""
        result = evaluate_hard_gate_calibration(under_blocking_golden_set)

        assert result.false_pass_count == 3  # 3 BLOCK runs incorrectly allowed


# ============================================================================
# 4. Deterministic Behavior Tests
# ============================================================================

class TestDeterministicBehavior:
    """Tests for deterministic calibration behavior."""

    def test_same_input_produces_same_result(self, perfect_golden_set):
        """Same input should always produce same result."""
        result1 = evaluate_hard_gate_calibration(perfect_golden_set)
        result2 = evaluate_hard_gate_calibration(perfect_golden_set)

        assert result1.calibration_status == result2.calibration_status
        assert result1.false_block_rate == result2.false_block_rate
        assert result1.false_pass_rate == result2.false_pass_rate
        assert result1.n_runs == result2.n_runs

    def test_order_does_not_affect_result(self, perfect_golden_set):
        """Order of golden runs should not affect result."""
        reversed_set = list(reversed(perfect_golden_set))

        result1 = evaluate_hard_gate_calibration(perfect_golden_set)
        result2 = evaluate_hard_gate_calibration(reversed_set)

        assert result1.calibration_status == result2.calibration_status
        assert result1.false_block_rate == result2.false_block_rate
        assert result1.false_pass_rate == result2.false_pass_rate

    def test_threshold_affects_classification(self, perfect_golden_set):
        """Different thresholds should affect classification."""
        result_default = evaluate_hard_gate_calibration(
            perfect_golden_set, block_threshold=0.2
        )
        result_higher = evaluate_hard_gate_calibration(
            perfect_golden_set, block_threshold=0.4
        )

        # Higher threshold means more would be blocked
        assert result_higher.actual_block >= result_default.actual_block


# ============================================================================
# 5. Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_golden_set_returns_ok(self):
        """Empty golden set should return OK with zero counts."""
        result = evaluate_hard_gate_calibration([])

        assert result.calibration_status == "OK"
        assert result.n_runs == 0
        assert result.false_block_rate == 0.0
        assert result.false_pass_rate == 0.0

    def test_all_ok_runs(self):
        """Golden set with only OK runs should work correctly."""
        golden = [
            LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
            LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
        ]

        result = evaluate_hard_gate_calibration(golden)

        assert result.n_expected_ok == 2
        assert result.n_expected_block == 0
        assert result.false_pass_rate == 0.0  # No BLOCK runs to miss

    def test_all_block_runs(self):
        """Golden set with only BLOCK runs should work correctly."""
        golden = [
            LabeledTDAResult(hss=0.1, sns=0.2, pcs=0.1, drs=0.6, expected_label="BLOCK"),
            LabeledTDAResult(hss=0.05, sns=0.15, pcs=0.05, drs=0.7, expected_label="BLOCK"),
        ]

        result = evaluate_hard_gate_calibration(golden)

        assert result.n_expected_ok == 0
        assert result.n_expected_block == 2
        assert result.false_block_rate == 0.0  # No OK runs to incorrectly block

    def test_hss_at_exact_threshold(self):
        """HSS exactly at threshold should not be blocked."""
        golden = [
            LabeledTDAResult(hss=0.2, sns=0.5, pcs=0.4, drs=0.3, expected_label="OK"),
        ]

        result = evaluate_hard_gate_calibration(golden, block_threshold=0.2)

        # HSS = 0.2 is NOT < 0.2, so should not be blocked
        assert result.actual_ok == 1
        assert result.actual_block == 0

    def test_single_run(self):
        """Single run should be evaluated correctly."""
        golden = [
            LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
        ]

        result = evaluate_hard_gate_calibration(golden)

        assert result.n_runs == 1
        assert result.calibration_status == "OK"

    def test_to_dict_produces_valid_output(self, perfect_golden_set):
        """to_dict should produce serializable output."""
        result = evaluate_hard_gate_calibration(perfect_golden_set)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "schema_version" in result_dict
        assert "calibration_status" in result_dict
        assert "false_block_rate" in result_dict
        assert "false_pass_rate" in result_dict
        assert "notes" in result_dict


# ============================================================================
# 6. Custom Threshold Tests
# ============================================================================

class TestCustomThresholds:
    """Tests for custom calibration thresholds."""

    def test_custom_ok_thresholds(self, perfect_golden_set):
        """Custom OK thresholds should be respected."""
        result = evaluate_hard_gate_calibration(
            perfect_golden_set,
            false_block_threshold_ok=0.01,  # Very strict
            false_pass_threshold_ok=0.01,
        )

        # Perfect set should still pass even with strict thresholds
        assert result.calibration_status == "OK"

    def test_custom_drifting_thresholds(self):
        """Custom DRIFTING thresholds should be respected."""
        # Create a set with 10% false block rate
        golden = [
            LabeledTDAResult(hss=0.15, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),  # False block
            LabeledTDAResult(hss=0.8, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
            LabeledTDAResult(hss=0.7, sns=0.6, pcs=0.5, drs=0.2, expected_label="OK"),
            LabeledTDAResult(hss=0.6, sns=0.5, pcs=0.4, drs=0.3, expected_label="OK"),
            LabeledTDAResult(hss=0.5, sns=0.4, pcs=0.3, drs=0.4, expected_label="OK"),
            LabeledTDAResult(hss=0.4, sns=0.3, pcs=0.2, drs=0.5, expected_label="OK"),
            LabeledTDAResult(hss=0.3, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.25, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.22, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
            LabeledTDAResult(hss=0.21, sns=0.2, pcs=0.1, drs=0.6, expected_label="OK"),
        ]

        # With default thresholds, 10% should be DRIFTING
        result_default = evaluate_hard_gate_calibration(golden)
        assert result_default.calibration_status == "DRIFTING"

        # With higher DRIFTING threshold, should still be OK
        result_lenient = evaluate_hard_gate_calibration(
            golden,
            false_block_threshold_ok=0.12,  # 12% is OK
        )
        assert result_lenient.calibration_status == "OK"
