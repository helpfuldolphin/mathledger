"""
Tests for MathLedger Statistical Metrics Library.

This test suite verifies:
  - Determinism across all functions
  - Correct Wilson CI behavior (edge cases, known values)
  - Drift detection reliability
  - Resilience to malformed input

All tests are marked as unit tests and do not require external dependencies.
"""

import math
import pytest

from backend.metrics.statistical import (
    compute_wilson_ci,
    detect_temporal_success_drift,
    detect_policy_weight_drift,
    detect_metric_instability,
    DriftResult,
    rolling_average,
    rolling_variance,
    stability_index,
    abstention_trend,
    validate_series,
)


# ===========================================================================
# WILSON CI TESTS (15 tests)
# ===========================================================================

class TestWilsonCI:
    """Tests for compute_wilson_ci function."""

    # --- Determinism Tests ---

    @pytest.mark.unit
    def test_wilson_ci_determinism_basic(self):
        """Wilson CI returns identical results across multiple calls."""
        for _ in range(100):
            result = compute_wilson_ci(80, 100, 0.95)
            assert result == compute_wilson_ci(80, 100, 0.95)

    @pytest.mark.unit
    def test_wilson_ci_determinism_various_inputs(self):
        """Wilson CI is deterministic for various input combinations."""
        test_cases = [
            (0, 100, 0.95),
            (100, 100, 0.95),
            (50, 100, 0.90),
            (1, 1000, 0.99),
            (999, 1000, 0.80),
        ]
        for successes, trials, conf in test_cases:
            result1 = compute_wilson_ci(successes, trials, conf)
            result2 = compute_wilson_ci(successes, trials, conf)
            assert result1 == result2, f"Non-deterministic for ({successes}, {trials}, {conf})"

    # --- Known Value Tests ---

    @pytest.mark.unit
    def test_wilson_ci_80_of_100(self):
        """Wilson CI for 80/100 at 95% confidence matches expected range."""
        ci_low, ci_high = compute_wilson_ci(80, 100, 0.95)
        # Expected: approximately (0.71, 0.87) based on Wilson formula
        assert 0.70 < ci_low < 0.72, f"ci_low={ci_low}"
        assert 0.86 < ci_high < 0.88, f"ci_high={ci_high}"
        assert ci_low < 0.80 < ci_high

    @pytest.mark.unit
    def test_wilson_ci_50_50(self):
        """Wilson CI for 50/100 is symmetric around 0.5."""
        ci_low, ci_high = compute_wilson_ci(50, 100, 0.95)
        # Should be roughly symmetric
        assert abs((0.5 - ci_low) - (ci_high - 0.5)) < 0.01
        assert 0.39 < ci_low < 0.42
        assert 0.58 < ci_high < 0.61

    @pytest.mark.unit
    def test_wilson_ci_different_confidence_levels(self):
        """Higher confidence produces wider intervals."""
        ci_90 = compute_wilson_ci(75, 100, 0.90)
        ci_95 = compute_wilson_ci(75, 100, 0.95)
        ci_99 = compute_wilson_ci(75, 100, 0.99)

        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_90 < width_95 < width_99

    # --- Edge Cases ---

    @pytest.mark.unit
    def test_wilson_ci_zero_trials(self):
        """Wilson CI for n=0 returns uninformative interval [0, 1]."""
        ci_low, ci_high = compute_wilson_ci(0, 0, 0.95)
        assert ci_low == 0.0
        assert ci_high == 1.0

    @pytest.mark.unit
    def test_wilson_ci_zero_successes(self):
        """Wilson CI for 0 successes has lower bound at 0."""
        ci_low, ci_high = compute_wilson_ci(0, 100, 0.95)
        assert ci_low == 0.0
        assert 0.0 < ci_high < 0.05  # Should be small but nonzero

    @pytest.mark.unit
    def test_wilson_ci_all_successes(self):
        """Wilson CI for 100% success has upper bound at 1."""
        ci_low, ci_high = compute_wilson_ci(100, 100, 0.95)
        assert ci_high == 1.0
        assert 0.95 < ci_low < 1.0  # Should be close to 1

    @pytest.mark.unit
    def test_wilson_ci_small_sample(self):
        """Wilson CI handles small samples correctly."""
        ci_low, ci_high = compute_wilson_ci(1, 3, 0.95)
        assert 0.0 < ci_low < 0.20
        assert 0.70 < ci_high < 1.0
        # With Wilson, interval should contain the true proportion

    @pytest.mark.unit
    def test_wilson_ci_single_success(self):
        """Wilson CI for 1/1 doesn't collapse to a point."""
        ci_low, ci_high = compute_wilson_ci(1, 1, 0.95)
        assert ci_low > 0.0
        assert ci_high == 1.0
        assert ci_high > ci_low

    # --- Error Handling ---

    @pytest.mark.unit
    def test_wilson_ci_negative_successes_raises(self):
        """Wilson CI raises for negative successes."""
        with pytest.raises(ValueError, match="successes must be >= 0"):
            compute_wilson_ci(-1, 100, 0.95)

    @pytest.mark.unit
    def test_wilson_ci_successes_exceed_trials_raises(self):
        """Wilson CI raises when successes > trials."""
        with pytest.raises(ValueError, match="cannot exceed trials"):
            compute_wilson_ci(101, 100, 0.95)

    @pytest.mark.unit
    def test_wilson_ci_invalid_confidence_raises(self):
        """Wilson CI raises for invalid confidence levels."""
        with pytest.raises(ValueError, match="confidence must be in"):
            compute_wilson_ci(50, 100, 0.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            compute_wilson_ci(50, 100, 1.0)
        with pytest.raises(ValueError, match="confidence must be in"):
            compute_wilson_ci(50, 100, 1.5)

    @pytest.mark.unit
    def test_wilson_ci_bounds_always_valid(self):
        """Wilson CI bounds are always in [0, 1] and low <= high."""
        test_cases = [
            (0, 1), (1, 1), (0, 10), (5, 10), (10, 10),
            (0, 100), (1, 100), (50, 100), (99, 100), (100, 100),
            (0, 1000), (500, 1000), (1000, 1000),
        ]
        for successes, trials in test_cases:
            ci_low, ci_high = compute_wilson_ci(successes, trials, 0.95)
            assert 0.0 <= ci_low <= 1.0
            assert 0.0 <= ci_high <= 1.0
            assert ci_low <= ci_high

    @pytest.mark.unit
    def test_wilson_ci_nonstandard_confidence(self):
        """Wilson CI handles non-standard confidence levels."""
        # Test confidence levels not in precomputed table
        ci_85 = compute_wilson_ci(50, 100, 0.85)
        ci_93 = compute_wilson_ci(50, 100, 0.93)

        assert ci_85[0] < ci_85[1]
        assert ci_93[0] < ci_93[1]
        # 93% should be between 90% and 95%
        ci_90 = compute_wilson_ci(50, 100, 0.90)
        ci_95 = compute_wilson_ci(50, 100, 0.95)
        width_90 = ci_90[1] - ci_90[0]
        width_93 = ci_93[1] - ci_93[0]
        width_95 = ci_95[1] - ci_95[0]
        assert width_90 < width_93 < width_95


# ===========================================================================
# TEMPORAL SUCCESS DRIFT TESTS (8 tests)
# ===========================================================================

class TestTemporalSuccessDrift:
    """Tests for detect_temporal_success_drift function."""

    @pytest.mark.unit
    def test_drift_detection_determinism(self):
        """Drift detection is deterministic."""
        series = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        for _ in range(50):
            result = detect_temporal_success_drift(series)
            assert result == detect_temporal_success_drift(series)

    @pytest.mark.unit
    def test_drift_detection_clear_decrease(self):
        """Detects clear decreasing drift."""
        series = [1] * 20 + [0] * 20
        # With window_size=10, the max change between adjacent windows is 0.1
        # Use a lower threshold to detect this drift
        result = detect_temporal_success_drift(series, window_size=10, threshold=0.05)
        assert result.detected is True
        assert result.direction == "decreasing"
        assert result.score >= 0.05

    @pytest.mark.unit
    def test_drift_detection_clear_increase(self):
        """Detects clear increasing drift."""
        series = [0] * 20 + [1] * 20
        # With window_size=10, the max change between adjacent windows is 0.1
        # Use a lower threshold to detect this drift
        result = detect_temporal_success_drift(series, window_size=10, threshold=0.05)
        assert result.detected is True
        assert result.direction == "increasing"
        assert result.score >= 0.05

    @pytest.mark.unit
    def test_drift_detection_stable_series(self):
        """No drift detected in stable series."""
        series = [0.5] * 50
        result = detect_temporal_success_drift(series, window_size=10, threshold=0.15)
        assert result.detected is False
        assert result.direction == "stable"

    @pytest.mark.unit
    def test_drift_detection_empty_series(self):
        """Handles empty series gracefully."""
        result = detect_temporal_success_drift([])
        assert result.detected is False
        assert "Empty" in result.details

    @pytest.mark.unit
    def test_drift_detection_short_series(self):
        """Handles series shorter than window."""
        result = detect_temporal_success_drift([1, 0, 1], window_size=10)
        assert result.detected is False
        assert "too short" in result.details

    @pytest.mark.unit
    def test_drift_detection_binary_series(self):
        """Works with binary 0/1 series."""
        series = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
        result = detect_temporal_success_drift(series, window_size=3, threshold=0.3)
        assert isinstance(result, DriftResult)
        assert isinstance(result.score, float)

    @pytest.mark.unit
    def test_drift_detection_float_series(self):
        """Works with float rate series."""
        series = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        # With window_size=3 and this gradual decline, max change is ~0.05
        result = detect_temporal_success_drift(series, window_size=3, threshold=0.04)
        assert result.detected is True
        assert result.direction == "decreasing"


# ===========================================================================
# POLICY WEIGHT DRIFT TESTS (6 tests)
# ===========================================================================

class TestPolicyWeightDrift:
    """Tests for detect_policy_weight_drift function."""

    @pytest.mark.unit
    def test_policy_drift_determinism(self):
        """Policy drift detection is deterministic."""
        weights = [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.5, 0.2]]
        for _ in range(50):
            result = detect_policy_weight_drift(weights)
            assert result == detect_policy_weight_drift(weights)

    @pytest.mark.unit
    def test_policy_drift_stable_weights(self):
        """No drift for constant weights."""
        weights = [[0.5, 0.3, 0.2]] * 10
        result = detect_policy_weight_drift(weights, threshold=0.1)
        assert result.detected is False
        assert result.direction == "stable"

    @pytest.mark.unit
    def test_policy_drift_large_change(self):
        """Detects large weight shift."""
        weights = [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.2, 0.4, 0.4],
        ]
        result = detect_policy_weight_drift(weights, threshold=0.3)
        assert result.detected is True

    @pytest.mark.unit
    def test_policy_drift_empty_list(self):
        """Handles empty weight list."""
        result = detect_policy_weight_drift([])
        assert result.detected is False
        assert "Empty" in result.details

    @pytest.mark.unit
    def test_policy_drift_single_snapshot(self):
        """Handles single weight snapshot."""
        result = detect_policy_weight_drift([[0.5, 0.3, 0.2]])
        assert result.detected is False
        assert "at least 2" in result.details

    @pytest.mark.unit
    def test_policy_drift_dimension_mismatch(self):
        """Handles weight vectors of different sizes."""
        weights = [
            [0.5, 0.5],
            [0.4, 0.4, 0.2],
            [0.3, 0.3, 0.2, 0.2],
        ]
        result = detect_policy_weight_drift(weights, threshold=0.1)
        # Should not raise, should handle gracefully
        assert isinstance(result, DriftResult)


# ===========================================================================
# METRIC INSTABILITY TESTS (5 tests)
# ===========================================================================

class TestMetricInstability:
    """Tests for detect_metric_instability function."""

    @pytest.mark.unit
    def test_instability_determinism(self):
        """Instability detection is deterministic."""
        series = [1.0, 1.5, 0.8, 2.0, 0.5]
        for _ in range(50):
            result = detect_metric_instability(series)
            assert result == detect_metric_instability(series)

    @pytest.mark.unit
    def test_instability_stable_series(self):
        """Low instability for consistent series."""
        series = [1.0, 1.01, 0.99, 1.0, 0.98, 1.02]
        result = detect_metric_instability(series, threshold=0.5)
        assert result.detected is False
        assert result.direction == "stable"
        assert result.score < 0.5

    @pytest.mark.unit
    def test_instability_volatile_series(self):
        """High instability for volatile series."""
        series = [1.0, 10.0, 0.5, 8.0, 2.0, 15.0]
        result = detect_metric_instability(series, threshold=0.5)
        assert result.detected is True
        assert result.direction == "volatile"

    @pytest.mark.unit
    def test_instability_empty_series(self):
        """Handles empty series."""
        result = detect_metric_instability([])
        assert result.detected is False
        assert "Empty" in result.details

    @pytest.mark.unit
    def test_instability_single_value(self):
        """Handles single value series."""
        result = detect_metric_instability([5.0])
        assert result.detected is False
        assert "at least 2" in result.details


# ===========================================================================
# ROLLING AVERAGE TESTS (4 tests)
# ===========================================================================

class TestRollingAverage:
    """Tests for rolling_average function."""

    @pytest.mark.unit
    def test_rolling_average_determinism(self):
        """Rolling average is deterministic."""
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _ in range(50):
            result = rolling_average(series, window=3)
            assert result == rolling_average(series, window=3)

    @pytest.mark.unit
    def test_rolling_average_known_values(self):
        """Rolling average produces expected values."""
        series = [1, 2, 3, 4, 5]
        result = rolling_average(series, window=3)
        # Expected: [1, 1.5, 2, 3, 4]
        assert result[0] == 1.0  # First value
        assert result[1] == 1.5  # Average of [1, 2]
        assert result[2] == 2.0  # Average of [1, 2, 3]
        assert result[3] == 3.0  # Average of [2, 3, 4]
        assert result[4] == 4.0  # Average of [3, 4, 5]

    @pytest.mark.unit
    def test_rolling_average_empty(self):
        """Rolling average of empty series is empty."""
        assert rolling_average([], window=3) == []

    @pytest.mark.unit
    def test_rolling_average_invalid_window(self):
        """Rolling average raises for invalid window."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_average([1, 2, 3], window=0)


# ===========================================================================
# ROLLING VARIANCE TESTS (4 tests)
# ===========================================================================

class TestRollingVariance:
    """Tests for rolling_variance function."""

    @pytest.mark.unit
    def test_rolling_variance_determinism(self):
        """Rolling variance is deterministic."""
        series = [1, 2, 3, 4, 5]
        for _ in range(50):
            result = rolling_variance(series, window=3)
            assert result == rolling_variance(series, window=3)

    @pytest.mark.unit
    def test_rolling_variance_known_values(self):
        """Rolling variance produces expected values."""
        series = [1, 2, 3, 4, 5]
        result = rolling_variance(series, window=3)
        # First value: single point -> variance = 0
        assert result[0] == 0.0
        # Second value: [1, 2] -> variance = 0.5
        assert result[1] == 0.5
        # Third value: [1, 2, 3] -> variance = 1.0
        assert result[2] == 1.0
        # Fourth value: [2, 3, 4] -> variance = 1.0
        assert result[3] == 1.0

    @pytest.mark.unit
    def test_rolling_variance_constant_series(self):
        """Rolling variance of constant series is 0."""
        series = [5.0] * 10
        result = rolling_variance(series, window=3)
        # All values should be 0 (or very close due to floating point)
        for v in result[1:]:  # First value is 0 by definition
            assert abs(v) < 1e-10

    @pytest.mark.unit
    def test_rolling_variance_invalid_window(self):
        """Rolling variance raises for invalid window."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_variance([1, 2, 3], window=0)


# ===========================================================================
# STABILITY INDEX TESTS (4 tests)
# ===========================================================================

class TestStabilityIndex:
    """Tests for stability_index function."""

    @pytest.mark.unit
    def test_stability_index_determinism(self):
        """Stability index is deterministic."""
        series = [1.0, 1.1, 0.9, 1.0, 1.05]
        for _ in range(50):
            result = stability_index(series)
            assert result == stability_index(series)

    @pytest.mark.unit
    def test_stability_index_constant_series(self):
        """Constant series has stability index of 1.0."""
        series = [5.0] * 10
        assert stability_index(series) == 1.0

    @pytest.mark.unit
    def test_stability_index_variable_series(self):
        """Variable series has lower stability index."""
        stable = [1.0, 1.01, 0.99, 1.0]
        variable = [1.0, 5.0, 0.5, 3.0]

        si_stable = stability_index(stable)
        si_variable = stability_index(variable)

        assert si_stable > si_variable
        assert 0.0 < si_variable < si_stable <= 1.0

    @pytest.mark.unit
    def test_stability_index_edge_cases(self):
        """Stability index handles edge cases."""
        assert stability_index([]) == 1.0  # Empty
        assert stability_index([5.0]) == 1.0  # Single value
        assert stability_index([0.0, 0.0, 0.0]) == 1.0  # All zeros


# ===========================================================================
# ABSTENTION TREND TESTS (4 tests)
# ===========================================================================

class TestAbstentionTrend:
    """Tests for abstention_trend function."""

    @pytest.mark.unit
    def test_abstention_trend_determinism(self):
        """Abstention trend is deterministic."""
        series = [1, 2, 3, 4, 5]
        for _ in range(50):
            result = abstention_trend(series)
            assert result == abstention_trend(series)

    @pytest.mark.unit
    def test_abstention_trend_increasing(self):
        """Increasing series has positive trend."""
        series = [1, 2, 3, 4, 5]
        trend = abstention_trend(series)
        assert trend > 0.0
        assert abs(trend - 1.0) < 0.01  # Should be close to 1.0

    @pytest.mark.unit
    def test_abstention_trend_decreasing(self):
        """Decreasing series has negative trend."""
        series = [5, 4, 3, 2, 1]
        trend = abstention_trend(series)
        assert trend < 0.0
        assert abs(trend - (-1.0)) < 0.01  # Should be close to -1.0

    @pytest.mark.unit
    def test_abstention_trend_stable(self):
        """Constant series has zero trend."""
        series = [3, 3, 3, 3, 3]
        trend = abstention_trend(series)
        assert trend == 0.0


# ===========================================================================
# VALIDATE SERIES TESTS (4 tests)
# ===========================================================================

class TestValidateSeries:
    """Tests for validate_series utility function."""

    @pytest.mark.unit
    def test_validate_series_converts_to_float(self):
        """validate_series converts integers to floats."""
        result = validate_series([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.unit
    def test_validate_series_rejects_none(self):
        """validate_series raises for None input."""
        with pytest.raises(TypeError, match="cannot be None"):
            validate_series(None)

    @pytest.mark.unit
    def test_validate_series_rejects_nan(self):
        """validate_series raises for NaN values."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            validate_series([1.0, float('nan'), 3.0])

    @pytest.mark.unit
    def test_validate_series_rejects_infinity(self):
        """validate_series raises for infinite values."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            validate_series([1.0, float('inf'), 3.0])


# ===========================================================================
# COMPREHENSIVE INTEGRATION TESTS (2 tests)
# ===========================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.unit
    def test_full_analysis_pipeline(self):
        """Full analysis pipeline with all functions."""
        # Simulated success series over 100 cycles
        # First 50: high success (~80%), Last 50: lower success (~60%)
        series = [1] * 40 + [0] * 10 + [1] * 30 + [0] * 20

        # Wilson CI at different points
        ci_early = compute_wilson_ci(40, 50, 0.95)
        ci_late = compute_wilson_ci(30, 50, 0.95)

        assert ci_early[0] > ci_late[0]  # Early was better

        # Drift detection (use lower threshold for gradual changes)
        drift = detect_temporal_success_drift(series, window_size=10, threshold=0.05)
        assert drift.detected is True
        assert drift.direction == "decreasing"

        # Rolling stats
        ra = rolling_average(series, window=10)
        rv = rolling_variance(series, window=10)
        assert len(ra) == len(series)
        assert len(rv) == len(series)

        # Stability
        si = stability_index(series)
        assert 0.0 < si < 1.0  # Should be moderate

        # Abstention trend (inverse of success)
        abstentions = [1 - s for s in series]
        at = abstention_trend(abstentions)
        assert at > 0  # Abstentions increasing

    @pytest.mark.unit
    def test_all_functions_return_correct_types(self):
        """All functions return expected types."""
        series = [0.8, 0.75, 0.82, 0.78, 0.80]

        # Wilson CI
        ci = compute_wilson_ci(80, 100, 0.95)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert all(isinstance(v, float) for v in ci)

        # Drift detectors
        assert isinstance(detect_temporal_success_drift(series), DriftResult)
        assert isinstance(detect_policy_weight_drift([[0.5, 0.5]]), DriftResult)
        assert isinstance(detect_metric_instability(series), DriftResult)

        # Rolling stats
        assert isinstance(rolling_average(series, 3), list)
        assert isinstance(rolling_variance(series, 3), list)

        # Scalars
        assert isinstance(stability_index(series), float)
        assert isinstance(abstention_trend(series), float)


# ===========================================================================
# MALFORMED INPUT RESILIENCE TESTS (4 tests)
# ===========================================================================

class TestMalformedInputResilience:
    """Tests for resilience to malformed or edge-case inputs."""

    @pytest.mark.unit
    def test_large_numbers(self):
        """Functions handle large numbers without overflow."""
        # Large success count
        ci = compute_wilson_ci(999999, 1000000, 0.95)
        assert 0.999 < ci[0] <= 1.0
        # Upper bound should be very close to 1.0 (allowing floating point tolerance)
        assert ci[1] > 0.9999

        # Large series
        large_series = [1.0] * 10000
        ra = rolling_average(large_series, 100)
        assert len(ra) == 10000
        assert all(abs(v - 1.0) < 1e-10 for v in ra)

    @pytest.mark.unit
    def test_very_small_values(self):
        """Functions handle very small values."""
        # Use slightly larger values to avoid floating point edge cases
        small_series = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        trend = abstention_trend(small_series)
        # Should detect increasing trend (positive value)
        assert trend > 0.5  # Strong positive trend

        si = stability_index(small_series)
        assert 0.0 < si < 1.0

    @pytest.mark.unit
    def test_negative_values_in_series(self):
        """Rolling functions handle negative values."""
        series = [-5, -3, -1, 1, 3, 5]
        ra = rolling_average(series, 3)
        rv = rolling_variance(series, 3)

        assert ra[0] == -5.0
        assert len(ra) == len(series)
        assert all(v >= 0 for v in rv)  # Variance is always non-negative

    @pytest.mark.unit
    def test_mixed_int_float_input(self):
        """Functions accept mixed int/float inputs."""
        series = [1, 2.0, 3, 4.5, 5]
        ra = rolling_average(series, 2)
        assert ra == [1.0, 1.5, 2.5, 3.75, 4.75]

        trend = abstention_trend(series)
        assert trend > 0  # Increasing


# ===========================================================================
# Total: 40+ tests covering all requirements
# ===========================================================================

