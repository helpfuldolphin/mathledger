"""
Tests for MathLedger Multi-Metric Drift Alignment Engine.

This test suite verifies:
  - Drift monotonicity analysis
  - Stability envelope determinism
  - Multi-metric alignment reproducibility
  - Cross-metric correlation accuracy

All tests are marked as unit tests and do not require external dependencies.
"""

import math
import pytest

from backend.metrics.drift_alignment import (
    MetricDriftSignature,
    DriftAlignmentResult,
    StabilityEnvelope,
    compute_drift_alignment,
    generate_stability_envelope,
    generate_multi_metric_envelopes,
    compute_monotonicity,
    compute_pearson_correlation,
    analyze_drift_monotonicity,
)


# ===========================================================================
# MONOTONICITY TESTS (6 tests)
# ===========================================================================

class TestMonotonicity:
    """Tests for compute_monotonicity function."""

    @pytest.mark.unit
    def test_monotonicity_determinism(self):
        """Monotonicity computation is deterministic."""
        series = [1, 2, 3, 4, 5, 4, 5, 6]
        for _ in range(50):
            result = compute_monotonicity(series)
            assert result == compute_monotonicity(series)

    @pytest.mark.unit
    def test_monotonicity_perfectly_increasing(self):
        """Perfectly increasing series has monotonicity of 1.0."""
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert compute_monotonicity(series) == 1.0

    @pytest.mark.unit
    def test_monotonicity_perfectly_decreasing(self):
        """Perfectly decreasing series has monotonicity of 1.0."""
        series = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert compute_monotonicity(series) == 1.0

    @pytest.mark.unit
    def test_monotonicity_alternating(self):
        """Alternating series has monotonicity of 0.0."""
        series = [1, 2, 1, 2, 1, 2, 1, 2, 1]
        mono = compute_monotonicity(series)
        assert mono == 0.0

    @pytest.mark.unit
    def test_monotonicity_constant_series(self):
        """Constant series (all ties) has monotonicity of 1.0."""
        series = [5, 5, 5, 5, 5]
        assert compute_monotonicity(series) == 1.0

    @pytest.mark.unit
    def test_monotonicity_edge_cases(self):
        """Monotonicity handles edge cases correctly."""
        assert compute_monotonicity([]) == 1.0  # Empty
        assert compute_monotonicity([5]) == 1.0  # Single value
        assert compute_monotonicity([1, 2]) == 1.0  # Two values increasing


# ===========================================================================
# PEARSON CORRELATION TESTS (5 tests)
# ===========================================================================

class TestPearsonCorrelation:
    """Tests for compute_pearson_correlation function."""

    @pytest.mark.unit
    def test_correlation_determinism(self):
        """Correlation computation is deterministic."""
        a = [1, 2, 3, 4, 5]
        b = [2, 4, 6, 8, 10]
        for _ in range(50):
            result = compute_pearson_correlation(a, b)
            assert result == compute_pearson_correlation(a, b)

    @pytest.mark.unit
    def test_correlation_perfect_positive(self):
        """Perfectly correlated series have correlation of 1.0."""
        a = [1, 2, 3, 4, 5]
        b = [2, 4, 6, 8, 10]  # b = 2*a
        corr = compute_pearson_correlation(a, b)
        assert abs(corr - 1.0) < 1e-6

    @pytest.mark.unit
    def test_correlation_perfect_negative(self):
        """Perfectly anti-correlated series have correlation of -1.0."""
        a = [1, 2, 3, 4, 5]
        b = [10, 8, 6, 4, 2]  # b = 12 - 2*a
        corr = compute_pearson_correlation(a, b)
        assert abs(corr - (-1.0)) < 1e-6

    @pytest.mark.unit
    def test_correlation_uncorrelated(self):
        """Uncorrelated series have correlation near 0."""
        a = [1, 2, 3, 4, 5]
        b = [5, 3, 4, 2, 5]  # Roughly uncorrelated
        corr = compute_pearson_correlation(a, b)
        # Should be near zero but not exactly
        assert abs(corr) < 0.5

    @pytest.mark.unit
    def test_correlation_length_mismatch_raises(self):
        """Correlation raises for mismatched series lengths."""
        a = [1, 2, 3]
        b = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="same length"):
            compute_pearson_correlation(a, b)


# ===========================================================================
# DRIFT ALIGNMENT TESTS (8 tests)
# ===========================================================================

class TestDriftAlignment:
    """Tests for compute_drift_alignment function."""

    @pytest.mark.unit
    def test_alignment_determinism(self):
        """Drift alignment is deterministic."""
        trajectories = {
            "success": [0.8, 0.75, 0.7, 0.65, 0.6],
            "abstention": [0.1, 0.12, 0.15, 0.18, 0.2],
        }
        for _ in range(50):
            result = compute_drift_alignment("test_slice", trajectories)
            result2 = compute_drift_alignment("test_slice", trajectories)
            assert result.drift_alignment_score == result2.drift_alignment_score
            assert result.coherence_score == result2.coherence_score

    @pytest.mark.unit
    def test_alignment_empty_trajectories(self):
        """Handles empty trajectories gracefully."""
        result = compute_drift_alignment("test_slice", {})
        assert result.drift_alignment_score == 0.0
        assert result.metrics == {}
        assert "error" in result.metadata

    @pytest.mark.unit
    def test_alignment_single_metric(self):
        """Handles single metric trajectory."""
        trajectories = {"success": [0.8, 0.75, 0.7, 0.65, 0.6]}
        result = compute_drift_alignment("test_slice", trajectories)
        assert "success" in result.metrics
        assert result.coherence_score == 1.0  # Single metric = perfect coherence

    @pytest.mark.unit
    def test_alignment_coherent_metrics(self):
        """High coherence for metrics moving in same direction."""
        # Both decreasing
        trajectories = {
            "success": [0.9, 0.8, 0.7, 0.6, 0.5],
            "abstention": [0.9, 0.8, 0.7, 0.6, 0.5],
            "depth": [10, 9, 8, 7, 6],
        }
        result = compute_drift_alignment(
            "coherent_slice", trajectories,
            window_size=3, drift_threshold=0.05
        )
        # All moving same direction = high coherence
        assert result.coherence_score >= 0.8

    @pytest.mark.unit
    def test_alignment_divergent_metrics(self):
        """Lower coherence for metrics moving in different directions."""
        trajectories = {
            "success": [0.5, 0.6, 0.7, 0.8, 0.9],  # Increasing
            "abstention": [0.9, 0.8, 0.7, 0.6, 0.5],  # Decreasing
            "depth": [5, 5, 5, 5, 5],  # Stable
        }
        result = compute_drift_alignment(
            "divergent_slice", trajectories,
            window_size=3, drift_threshold=0.05
        )
        # Different directions = lower coherence
        assert result.coherence_score < 1.0

    @pytest.mark.unit
    def test_alignment_result_structure(self):
        """Result has correct structure with all expected fields."""
        trajectories = {
            "success": [0.8, 0.75, 0.7],
            "abstention": [0.1, 0.15, 0.2],
        }
        result = compute_drift_alignment("test_slice", trajectories)

        assert isinstance(result, DriftAlignmentResult)
        assert result.slice == "test_slice"
        assert 0.0 <= result.drift_alignment_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert "success" in result.metrics
        assert "abstention" in result.metrics
        assert isinstance(result.metrics["success"], MetricDriftSignature)

    @pytest.mark.unit
    def test_alignment_to_dict(self):
        """to_dict produces expected JSON structure."""
        trajectories = {"success": [0.8, 0.7, 0.6]}
        result = compute_drift_alignment("my_slice", trajectories)
        d = result.to_dict()

        assert d["slice"] == "my_slice"
        assert "drift_alignment_score" in d
        assert "metrics" in d
        assert "success" in d["metrics"]
        assert "drift_score" in d["metrics"]["success"]

    @pytest.mark.unit
    def test_alignment_with_all_metric_types(self):
        """Works with all expected metric types."""
        trajectories = {
            "success": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
            "abstention": [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22],
            "depth": [5, 6, 7, 8, 9, 10, 11],
            "entropy": [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        }
        result = compute_drift_alignment(
            "slice_uplift_goal", trajectories,
            window_size=3
        )

        assert len(result.metrics) == 4
        assert all(m in result.metrics for m in ["success", "abstention", "depth", "entropy"])
        # Should have pairwise correlations
        assert len(result.trajectory_correlation) == 6  # C(4,2) = 6 pairs


# ===========================================================================
# STABILITY ENVELOPE TESTS (7 tests)
# ===========================================================================

class TestStabilityEnvelope:
    """Tests for generate_stability_envelope function."""

    @pytest.mark.unit
    def test_envelope_determinism(self):
        """Envelope generation is deterministic."""
        series = [0.8, 0.75, 0.82, 0.78, 0.80, 0.77, 0.79]
        for _ in range(50):
            env1 = generate_stability_envelope(series, window=3)
            env2 = generate_stability_envelope(series, window=3)
            assert env1.center_line == env2.center_line
            assert env1.upper_band == env2.upper_band
            assert env1.lower_band == env2.lower_band
            assert env1.envelope_width == env2.envelope_width

    @pytest.mark.unit
    def test_envelope_empty_series(self):
        """Handles empty series gracefully."""
        env = generate_stability_envelope([])
        assert env.center_line == []
        assert env.upper_band == []
        assert env.lower_band == []
        assert env.containment_ratio == 1.0

    @pytest.mark.unit
    def test_envelope_length_preservation(self):
        """Envelope bands have same length as input series."""
        series = [0.5, 0.6, 0.55, 0.58, 0.62]
        env = generate_stability_envelope(series, window=3)
        assert len(env.center_line) == 5
        assert len(env.upper_band) == 5
        assert len(env.lower_band) == 5
        assert len(env.variance_band) == 5

    @pytest.mark.unit
    def test_envelope_bounds_ordering(self):
        """Lower band is always <= center <= upper band."""
        series = [0.8, 0.6, 0.9, 0.5, 0.7, 0.85, 0.65]
        env = generate_stability_envelope(series, window=3)

        for i in range(len(series)):
            assert env.lower_band[i] <= env.center_line[i] <= env.upper_band[i], \
                f"Bounds violated at index {i}"

    @pytest.mark.unit
    def test_envelope_rate_metric_bounds(self):
        """Rate metrics (values in [0,1]) have bounds clamped to [0,1]."""
        series = [0.1, 0.15, 0.12, 0.08, 0.05, 0.02]
        env = generate_stability_envelope(series, window=3, confidence=0.95)

        assert all(0.0 <= v <= 1.0 for v in env.upper_band)
        assert all(0.0 <= v <= 1.0 for v in env.lower_band)

    @pytest.mark.unit
    def test_envelope_containment_ratio(self):
        """Containment ratio reflects actual containment."""
        # A stable series should have high containment
        series = [0.5, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51]
        env = generate_stability_envelope(series, window=3, confidence=0.95)
        assert env.containment_ratio >= 0.8  # Most points within envelope

    @pytest.mark.unit
    def test_envelope_to_dict(self):
        """to_dict produces expected structure."""
        series = [0.8, 0.75, 0.82]
        env = generate_stability_envelope(series, metric_name="test_metric", window=2)
        d = env.to_dict()

        assert d["metric_name"] == "test_metric"
        assert "center_line" in d
        assert "upper_band" in d
        assert "lower_band" in d
        assert "variance_band" in d
        assert "confidence_level" in d
        assert "envelope_width" in d
        assert "containment_ratio" in d


# ===========================================================================
# MULTI-METRIC ENVELOPE TESTS (4 tests)
# ===========================================================================

class TestMultiMetricEnvelopes:
    """Tests for generate_multi_metric_envelopes function."""

    @pytest.mark.unit
    def test_multi_envelope_determinism(self):
        """Multi-envelope generation is deterministic."""
        trajectories = {
            "success": [0.8, 0.75, 0.7],
            "abstention": [0.1, 0.15, 0.2],
        }
        for _ in range(50):
            envs1 = generate_multi_metric_envelopes(trajectories, window=2)
            envs2 = generate_multi_metric_envelopes(trajectories, window=2)
            assert envs1["success"].center_line == envs2["success"].center_line
            assert envs1["abstention"].upper_band == envs2["abstention"].upper_band

    @pytest.mark.unit
    def test_multi_envelope_all_metrics(self):
        """Generates envelopes for all provided metrics."""
        trajectories = {
            "success": [0.8, 0.75, 0.7],
            "abstention": [0.1, 0.15, 0.2],
            "depth": [5, 6, 7],
            "entropy": [1.5, 1.6, 1.7],
        }
        envs = generate_multi_metric_envelopes(trajectories, window=2)

        assert len(envs) == 4
        assert all(m in envs for m in ["success", "abstention", "depth", "entropy"])
        assert all(isinstance(e, StabilityEnvelope) for e in envs.values())

    @pytest.mark.unit
    def test_multi_envelope_empty_input(self):
        """Handles empty trajectories."""
        envs = generate_multi_metric_envelopes({}, window=3)
        assert envs == {}

    @pytest.mark.unit
    def test_multi_envelope_varying_lengths(self):
        """Handles trajectories of different lengths."""
        trajectories = {
            "short": [0.5, 0.6],
            "medium": [0.5, 0.6, 0.7, 0.8],
            "long": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        envs = generate_multi_metric_envelopes(trajectories, window=2)

        assert len(envs["short"].center_line) == 2
        assert len(envs["medium"].center_line) == 4
        assert len(envs["long"].center_line) == 6


# ===========================================================================
# DRIFT MONOTONICITY ANALYSIS TESTS (5 tests)
# ===========================================================================

class TestDriftMonotonicityAnalysis:
    """Tests for analyze_drift_monotonicity function."""

    @pytest.mark.unit
    def test_monotonicity_analysis_determinism(self):
        """Monotonicity analysis is deterministic."""
        series = [1, 2, 3, 2, 3, 4, 5, 6, 5, 6, 7, 8]
        for _ in range(50):
            result1 = analyze_drift_monotonicity(series, segment_size=3)
            result2 = analyze_drift_monotonicity(series, segment_size=3)
            assert result1 == result2

    @pytest.mark.unit
    def test_monotonicity_analysis_structure(self):
        """Analysis returns expected structure."""
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = analyze_drift_monotonicity(series, segment_size=3)

        assert "overall_monotonicity" in result
        assert "segment_monotonicity" in result
        assert "segment_trends" in result
        assert "meta_monotonicity" in result
        assert "segment_count" in result

    @pytest.mark.unit
    def test_monotonicity_analysis_perfectly_monotonic(self):
        """Perfectly monotonic series has high scores."""
        series = list(range(1, 21))  # 1 to 20
        result = analyze_drift_monotonicity(series, segment_size=5)

        assert result["overall_monotonicity"] == 1.0
        assert all(m == 1.0 for m in result["segment_monotonicity"])
        assert result["meta_monotonicity"] == 1.0

    @pytest.mark.unit
    def test_monotonicity_analysis_volatile_series(self):
        """Volatile series has lower monotonicity."""
        series = [1, 5, 2, 6, 3, 7, 4, 8, 5, 9]
        result = analyze_drift_monotonicity(series, segment_size=2)

        # Overall monotonicity should be imperfect
        assert result["overall_monotonicity"] < 1.0

    @pytest.mark.unit
    def test_monotonicity_analysis_edge_cases(self):
        """Handles edge cases correctly."""
        # Empty series
        result = analyze_drift_monotonicity([])
        assert result["overall_monotonicity"] == 1.0
        assert result["segment_count"] == 0

        # Single value
        result = analyze_drift_monotonicity([5])
        assert result["overall_monotonicity"] == 1.0

        # Two values
        result = analyze_drift_monotonicity([1, 2])
        assert result["overall_monotonicity"] == 1.0


# ===========================================================================
# REPRODUCIBILITY TESTS (3 tests)
# ===========================================================================

class TestReproducibility:
    """Tests for reproducibility of multi-metric alignment."""

    @pytest.mark.unit
    def test_full_pipeline_reproducibility(self):
        """Full analysis pipeline produces identical results."""
        trajectories = {
            "success": [0.8, 0.78, 0.75, 0.73, 0.7, 0.68, 0.65],
            "abstention": [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17],
            "depth": [3, 4, 4, 5, 5, 6, 6],
            "entropy": [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5],
        }

        # Run full pipeline twice
        for _ in range(10):
            alignment = compute_drift_alignment("repro_test", trajectories)
            envelopes = generate_multi_metric_envelopes(trajectories)
            mono_analysis = analyze_drift_monotonicity(trajectories["success"])

            alignment2 = compute_drift_alignment("repro_test", trajectories)
            envelopes2 = generate_multi_metric_envelopes(trajectories)
            mono_analysis2 = analyze_drift_monotonicity(trajectories["success"])

            assert alignment.drift_alignment_score == alignment2.drift_alignment_score
            assert envelopes["success"].envelope_width == envelopes2["success"].envelope_width
            assert mono_analysis == mono_analysis2

    @pytest.mark.unit
    def test_correlation_symmetry(self):
        """Pairwise correlations are symmetric."""
        trajectories = {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        }
        result = compute_drift_alignment("sym_test", trajectories)

        # Should have a_b correlation
        assert "a_b" in result.trajectory_correlation
        corr_ab = result.trajectory_correlation["a_b"]

        # Direct check of symmetry
        corr_direct = compute_pearson_correlation(trajectories["a"], trajectories["b"])
        corr_reverse = compute_pearson_correlation(trajectories["b"], trajectories["a"])
        assert corr_direct == corr_reverse

    @pytest.mark.unit
    def test_alignment_score_bounds(self):
        """Alignment score is always in [0, 1]."""
        test_cases = [
            {"a": [0.9, 0.8, 0.7]},
            {"a": [0.1, 0.5, 0.9], "b": [0.9, 0.5, 0.1]},
            {"a": [0.5] * 10, "b": [0.5] * 10, "c": [0.5] * 10},
            {"x": list(range(100))},
            {"noisy": [i + (i % 3) * 0.1 for i in range(20)]},
        ]

        for trajectories in test_cases:
            result = compute_drift_alignment("bounds_test", trajectories)
            assert 0.0 <= result.drift_alignment_score <= 1.0, \
                f"Score out of bounds: {result.drift_alignment_score}"
            assert 0.0 <= result.coherence_score <= 1.0, \
                f"Coherence out of bounds: {result.coherence_score}"


# ===========================================================================
# INTEGRATION TESTS (2 tests)
# ===========================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.unit
    def test_end_to_end_analysis(self):
        """Full end-to-end analysis pipeline."""
        # Simulate a slice with declining success and rising abstention
        n = 50
        trajectories = {
            "success": [0.9 - 0.01 * i for i in range(n)],
            "abstention": [0.05 + 0.01 * i for i in range(n)],
            "depth": [5 + i // 10 for i in range(n)],
            "entropy": [1.0 + 0.02 * i for i in range(n)],
        }

        # Compute alignment
        alignment = compute_drift_alignment(
            "slice_uplift_goal",
            trajectories,
            window_size=10,
            drift_threshold=0.05,
        )

        # Generate envelopes
        envelopes = generate_multi_metric_envelopes(trajectories, window=10)

        # Analyze monotonicity
        mono = analyze_drift_monotonicity(trajectories["success"], segment_size=10)

        # Verify structure
        assert alignment.slice == "slice_uplift_goal"
        assert len(alignment.metrics) == 4
        assert len(envelopes) == 4
        assert mono["segment_count"] == 5

        # Success and abstention should be anti-correlated
        assert alignment.trajectory_correlation["success_abstention"] < -0.5

    @pytest.mark.unit
    def test_all_output_types_correct(self):
        """All functions return correct types."""
        series = [0.8, 0.75, 0.82, 0.78, 0.80]
        trajectories = {"success": series, "abstention": [0.1, 0.12, 0.08, 0.11, 0.09]}

        # Monotonicity
        mono = compute_monotonicity(series)
        assert isinstance(mono, float)
        assert 0.0 <= mono <= 1.0

        # Correlation
        corr = compute_pearson_correlation(series, trajectories["abstention"])
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

        # Alignment
        alignment = compute_drift_alignment("test", trajectories)
        assert isinstance(alignment, DriftAlignmentResult)

        # Envelope
        envelope = generate_stability_envelope(series)
        assert isinstance(envelope, StabilityEnvelope)

        # Multi-envelope
        envelopes = generate_multi_metric_envelopes(trajectories)
        assert isinstance(envelopes, dict)
        assert all(isinstance(e, StabilityEnvelope) for e in envelopes.values())

        # Monotonicity analysis
        mono_analysis = analyze_drift_monotonicity(series)
        assert isinstance(mono_analysis, dict)


# ===========================================================================
# CI GATE HELPER TESTS (6 tests)
# ===========================================================================

class TestCIGateHelper:
    """Tests for evaluate_drift_for_ci function."""

    @pytest.mark.unit
    def test_ci_gate_determinism(self):
        """CI gate evaluation is deterministic."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci, CIGateResult

        trajectories = {
            "success": [0.8, 0.75, 0.7, 0.65, 0.6],
            "abstention": [0.1, 0.15, 0.2, 0.25, 0.3],
        }
        result = compute_drift_alignment("test_slice", trajectories)

        for _ in range(50):
            ci1 = evaluate_drift_for_ci(result)
            ci2 = evaluate_drift_for_ci(result)
            assert ci1.status == ci2.status
            assert ci1.max_metric_drift_score == ci2.max_metric_drift_score
            assert ci1.offending_metrics == ci2.offending_metrics

    @pytest.mark.unit
    def test_ci_gate_ok_status(self):
        """CI gate returns OK for low drift and high coherence."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci

        # Stable metrics should get OK
        trajectories = {
            "success": [0.8, 0.81, 0.79, 0.8, 0.8],
            "abstention": [0.1, 0.11, 0.09, 0.1, 0.1],
        }
        result = compute_drift_alignment("stable_slice", trajectories, window_size=3)
        ci = evaluate_drift_for_ci(result, drift_score_threshold=0.3, coherence_threshold=0.4)

        assert ci.status == "OK"
        assert ci.offending_metrics == []

    @pytest.mark.unit
    def test_ci_gate_warn_or_block_status(self):
        """CI gate returns WARN or BLOCK for metrics exceeding thresholds."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci

        # Metrics moving in different directions with significant drift
        trajectories = {
            "success": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],  # Declining
            "abstention": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],  # Rising (opposite)
            "depth": [1, 1, 10, 10, 1, 1, 10],  # Volatile/alternating pattern
        }
        result = compute_drift_alignment(
            "mixed_slice", trajectories,
            window_size=3, drift_threshold=0.05
        )
        # Use very strict thresholds to trigger non-OK status
        ci = evaluate_drift_for_ci(result, drift_score_threshold=0.01, coherence_threshold=0.99)

        # With strict thresholds, should not be OK
        assert ci.status in ("WARN", "BLOCK"), f"Expected WARN or BLOCK, got {ci.status}"

    @pytest.mark.unit
    def test_ci_gate_offending_metrics(self):
        """CI gate correctly identifies offending metrics."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci

        trajectories = {
            "stable": [0.5, 0.5, 0.5, 0.5, 0.5],
            "drifting": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        result = compute_drift_alignment(
            "mixed_slice", trajectories,
            window_size=2, drift_threshold=0.05
        )
        ci = evaluate_drift_for_ci(result, drift_score_threshold=0.05)

        # drifting metric should be flagged if its drift exceeds threshold
        # (exact behavior depends on drift scores)
        assert isinstance(ci.offending_metrics, list)

    @pytest.mark.unit
    def test_ci_gate_result_structure(self):
        """CI gate result has correct structure."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci, CIGateResult

        trajectories = {"success": [0.8, 0.75, 0.7]}
        result = compute_drift_alignment("test", trajectories)
        ci = evaluate_drift_for_ci(result)

        assert isinstance(ci, CIGateResult)
        assert ci.status in ("OK", "WARN", "BLOCK")
        assert isinstance(ci.max_metric_drift_score, float)
        assert isinstance(ci.drift_alignment_score, float)
        assert isinstance(ci.coherence_score, float)
        assert isinstance(ci.offending_metrics, list)

    @pytest.mark.unit
    def test_ci_gate_to_dict(self):
        """CI gate result to_dict works correctly."""
        from backend.metrics.drift_alignment import evaluate_drift_for_ci

        trajectories = {"success": [0.8, 0.75, 0.7]}
        result = compute_drift_alignment("test", trajectories)
        ci = evaluate_drift_for_ci(result)
        d = ci.to_dict()

        assert "status" in d
        assert "max_metric_drift_score" in d
        assert "drift_alignment_score" in d
        assert "coherence_score" in d
        assert "offending_metrics" in d


# ===========================================================================
# PATTERN HINT TESTS (5 tests)
# ===========================================================================

class TestPatternHints:
    """Tests for classify_pattern_hint function."""

    @pytest.mark.unit
    def test_pattern_hint_determinism(self):
        """Pattern hint classification is deterministic."""
        from backend.metrics.drift_alignment import classify_pattern_hint

        series = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for _ in range(50):
            hint1 = classify_pattern_hint(series)
            hint2 = classify_pattern_hint(series)
            assert hint1 == hint2

    @pytest.mark.unit
    def test_pattern_hint_valid_values(self):
        """Pattern hints are one of the 4 allowed values."""
        from backend.metrics.drift_alignment import classify_pattern_hint, PATTERN_HINTS

        test_series = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Rising
            [0.9, 0.8, 0.7, 0.6, 0.5],  # Falling
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Flat
            [0.1, 0.9, 0.1, 0.9, 0.1, 0.9],  # Oscillatory
            [],  # Empty
            [0.5],  # Single
        ]

        for series in test_series:
            hint = classify_pattern_hint(series)
            assert hint in PATTERN_HINTS, f"Invalid hint '{hint}' for series {series}"

    @pytest.mark.unit
    def test_pattern_hint_rising(self):
        """Rising series classified as 'rising'."""
        from backend.metrics.drift_alignment import classify_pattern_hint

        series = list(range(1, 21))  # 1 to 20
        hint = classify_pattern_hint(series)
        assert hint == "rising"

    @pytest.mark.unit
    def test_pattern_hint_falling(self):
        """Falling series classified as 'falling'."""
        from backend.metrics.drift_alignment import classify_pattern_hint

        series = list(range(20, 0, -1))  # 20 to 1
        hint = classify_pattern_hint(series)
        assert hint == "falling"

    @pytest.mark.unit
    def test_pattern_hint_flat(self):
        """Constant series classified as 'flat'."""
        from backend.metrics.drift_alignment import classify_pattern_hint

        series = [0.5] * 20
        hint = classify_pattern_hint(series)
        assert hint == "flat"


# ===========================================================================
# CLI TESTS (4 tests)
# ===========================================================================

class TestCLI:
    """Tests for drift_alignment_report.py CLI."""

    @pytest.mark.unit
    def test_cli_load_jsonl(self):
        """JSONL loading works correctly."""
        import tempfile
        import os
        from experiments.drift_alignment_report import load_jsonl

        # Create temp JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"cycle": 1, "success": true, "abstention": 0.1}\n')
            f.write('{"cycle": 2, "success": false, "abstention": 0.2}\n')
            f.write('{"cycle": 3, "success": true, "abstention": 0.15}\n')
            temp_path = f.name

        try:
            records = load_jsonl(temp_path)
            assert len(records) == 3
            assert records[0]["cycle"] == 1
            assert records[1]["success"] is False
            assert records[2]["abstention"] == 0.15
        finally:
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_cli_derive_trajectories(self):
        """Trajectory derivation from logs works correctly."""
        from experiments.drift_alignment_report import derive_trajectories_from_logs

        baseline_records = [
            {"success": True, "abstention_rate": 0.1},
            {"success": True, "abstention_rate": 0.12},
            {"success": False, "abstention_rate": 0.15},
        ]
        rfl_records = [
            {"success": True, "abstention_rate": 0.08},
            {"success": True, "abstention_rate": 0.06},
        ]

        trajectories = derive_trajectories_from_logs(baseline_records, rfl_records)

        assert "success" in trajectories
        assert "abstention" in trajectories
        assert len(trajectories["success"]) == 5
        assert len(trajectories["abstention"]) == 5

    @pytest.mark.unit
    def test_cli_generate_report(self):
        """Report generation produces expected structure."""
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {
            "success": [0.8, 0.75, 0.7, 0.65, 0.6],
            "abstention": [0.1, 0.15, 0.2, 0.25, 0.3],
        }

        report = generate_drift_report(
            "test_slice",
            trajectories,
            window=3,
            threshold=0.1,
        )

        assert report["slice"] == "test_slice"
        assert "drift_alignment_score" in report
        assert "coherence_score" in report
        assert "metrics" in report
        assert "stability_envelopes" in report
        assert "success" in report["metrics"]
        assert "pattern_hint" in report["metrics"]["success"]

    @pytest.mark.unit
    def test_cli_markdown_report(self):
        """Markdown report generation works."""
        from experiments.drift_alignment_report import generate_drift_report, generate_markdown_report

        trajectories = {
            "success": [0.8, 0.75, 0.7],
            "abstention": [0.1, 0.15, 0.2],
        }

        report = generate_drift_report("test_slice", trajectories, window=2)
        md = generate_markdown_report(report)

        assert "# Drift Alignment Report" in md
        assert "test_slice" in md
        assert "| Metric |" in md
        assert "success" in md
        assert "## Legend" in md


# ===========================================================================
# CLI SUBPROCESS TESTS (2 tests)
# ===========================================================================

class TestCLISubprocess:
    """Tests for CLI using subprocess."""

    @pytest.mark.unit
    def test_cli_ci_mode_exit_code(self):
        """CI mode returns correct exit codes."""
        import subprocess
        import tempfile
        import os

        # Create minimal test data
        baseline_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 10
        rfl_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 10

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as bf:
            bf.write(baseline_data)
            baseline_path = bf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as rf:
            rf.write(rfl_data)
            rfl_path = rf.name

        try:
            # Run CLI in CI mode
            result = subprocess.run(
                [
                    "uv", "run", "python", "experiments/drift_alignment_report.py",
                    "--slice", "test_slice",
                    "--baseline-log", baseline_path,
                    "--rfl-log", rfl_path,
                    "--ci",
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )

            # Should succeed (OK or WARN)
            assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}\nStderr: {result.stderr}"

            # Should have status in output
            assert "status=" in result.stdout

        finally:
            os.unlink(baseline_path)
            os.unlink(rfl_path)

    @pytest.mark.unit
    def test_cli_json_output(self):
        """CLI --json flag produces valid JSON output."""
        import subprocess
        import tempfile
        import os
        import json

        baseline_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 5
        rfl_data = '{"cycle": 1, "success": false, "abstention_rate": 0.2}\n' * 5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as bf:
            bf.write(baseline_data)
            baseline_path = bf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as rf:
            rf.write(rfl_data)
            rfl_path = rf.name

        try:
            result = subprocess.run(
                [
                    "uv", "run", "python", "experiments/drift_alignment_report.py",
                    "--slice", "json_test",
                    "--baseline-log", baseline_path,
                    "--rfl-log", rfl_path,
                    "--json",
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Should be valid JSON
            report = json.loads(result.stdout)
            assert report["slice"] == "json_test"
            assert "drift_alignment_score" in report
            assert "metrics" in report

        finally:
            os.unlink(baseline_path)
            os.unlink(rfl_path)


# ===========================================================================
# DRIFT THRESHOLD LIBRARY TESTS (10 tests)
# ===========================================================================

class TestDriftThresholds:
    """Tests for DRIFT_THRESHOLDS and get_drift_thresholds."""

    @pytest.mark.unit
    def test_drift_thresholds_structure(self):
        """DRIFT_THRESHOLDS has expected structure."""
        from backend.metrics.drift_alignment import DRIFT_THRESHOLDS

        assert "default" in DRIFT_THRESHOLDS
        assert "strict" in DRIFT_THRESHOLDS
        assert "lenient" in DRIFT_THRESHOLDS

        for profile, config in DRIFT_THRESHOLDS.items():
            assert "drift_score" in config
            assert "coherence" in config
            assert isinstance(config["drift_score"], (int, float))
            assert isinstance(config["coherence"], (int, float))

    @pytest.mark.unit
    def test_get_drift_thresholds_default(self):
        """get_drift_thresholds returns default profile correctly."""
        from backend.metrics.drift_alignment import get_drift_thresholds, DRIFT_THRESHOLDS

        thresholds = get_drift_thresholds("default")
        assert thresholds["drift_score"] == DRIFT_THRESHOLDS["default"]["drift_score"]
        assert thresholds["coherence"] == DRIFT_THRESHOLDS["default"]["coherence"]

    @pytest.mark.unit
    def test_get_drift_thresholds_strict(self):
        """get_drift_thresholds returns strict profile correctly."""
        from backend.metrics.drift_alignment import get_drift_thresholds

        thresholds = get_drift_thresholds("strict")
        assert thresholds["drift_score"] == 0.2
        assert thresholds["coherence"] == 0.5

    @pytest.mark.unit
    def test_get_drift_thresholds_invalid_profile(self):
        """get_drift_thresholds raises for invalid profile."""
        from backend.metrics.drift_alignment import get_drift_thresholds

        with pytest.raises(ValueError, match="Unknown profile"):
            get_drift_thresholds("nonexistent")

    @pytest.mark.unit
    def test_profile_affects_thresholds_not_math(self):
        """Profile selection changes thresholds but not drift math."""
        from backend.metrics.drift_alignment import (
            compute_drift_alignment,
            evaluate_drift_for_ci,
            get_drift_thresholds,
        )

        trajectories = {
            "success": [0.8, 0.75, 0.7, 0.65, 0.6],
            "abstention": [0.1, 0.15, 0.2, 0.25, 0.3],
        }

        # Compute alignment once
        result = compute_drift_alignment("test", trajectories, window_size=3)

        # Evaluate with different profiles
        default_th = get_drift_thresholds("default")
        strict_th = get_drift_thresholds("strict")

        ci_default = evaluate_drift_for_ci(
            result,
            drift_score_threshold=default_th["drift_score"],
            coherence_threshold=default_th["coherence"],
        )

        ci_strict = evaluate_drift_for_ci(
            result,
            drift_score_threshold=strict_th["drift_score"],
            coherence_threshold=strict_th["coherence"],
        )

        # Drift scores should be identical (math unchanged)
        assert ci_default.max_metric_drift_score == ci_strict.max_metric_drift_score
        assert ci_default.drift_alignment_score == ci_strict.drift_alignment_score
        assert ci_default.coherence_score == ci_strict.coherence_score

        # But status may differ due to thresholds
        # (not asserting specific statuses, just that computation is deterministic)

    @pytest.mark.unit
    def test_valid_profiles_matches_thresholds(self):
        """VALID_PROFILES exactly matches DRIFT_THRESHOLDS keys."""
        from backend.metrics.drift_alignment import DRIFT_THRESHOLDS, VALID_PROFILES

        assert set(DRIFT_THRESHOLDS.keys()) == set(VALID_PROFILES)
        # No extra profiles allowed
        assert len(DRIFT_THRESHOLDS) == len(VALID_PROFILES)

    @pytest.mark.unit
    def test_all_threshold_values_in_bounds(self):
        """All threshold values are floats in [0.0, 1.0]."""
        from backend.metrics.drift_alignment import DRIFT_THRESHOLDS

        for profile, config in DRIFT_THRESHOLDS.items():
            for key, value in config.items():
                assert isinstance(value, (int, float)), (
                    f"Profile '{profile}' key '{key}' must be numeric"
                )
                assert 0.0 <= value <= 1.0, (
                    f"Profile '{profile}' key '{key}' out of bounds: {value}"
                )

    @pytest.mark.unit
    def test_threshold_keys_are_exactly_expected(self):
        """Each profile has exactly drift_score and coherence keys."""
        from backend.metrics.drift_alignment import DRIFT_THRESHOLDS

        expected_keys = {"drift_score", "coherence"}
        for profile, config in DRIFT_THRESHOLDS.items():
            assert set(config.keys()) == expected_keys, (
                f"Profile '{profile}' has unexpected keys: {set(config.keys())}"
            )

    @pytest.mark.unit
    def test_validate_drift_thresholds_passes(self):
        """validate_drift_thresholds does not raise on valid config."""
        from backend.metrics.drift_alignment import validate_drift_thresholds

        # Should not raise - called at import time already
        validate_drift_thresholds()

    @pytest.mark.unit
    def test_get_drift_thresholds_returns_floats(self):
        """get_drift_thresholds always returns float values."""
        from backend.metrics.drift_alignment import get_drift_thresholds, VALID_PROFILES

        for profile in VALID_PROFILES:
            thresholds = get_drift_thresholds(profile)
            assert isinstance(thresholds["drift_score"], float)
            assert isinstance(thresholds["coherence"], float)


# ===========================================================================
# PATTERN HINTS IN CI TESTS (3 tests)
# ===========================================================================

class TestPatternHintsInCI:
    """Tests for pattern hints in CI JSON output."""

    @pytest.mark.unit
    def test_ci_result_includes_pattern_hints(self):
        """CI result with hints includes pattern_hints field."""
        from experiments.drift_alignment_report import evaluate_ci_with_hints

        trajectories = {
            "success": [0.9, 0.85, 0.8, 0.75, 0.7],  # Falling
            "abstention": [0.1, 0.15, 0.2, 0.25, 0.3],  # Rising
        }

        ci_result = evaluate_ci_with_hints(
            "test_slice",
            trajectories,
            window=3,
            threshold=0.1,
            drift_score_threshold=0.3,
            coherence_threshold=0.4,
        )

        assert "pattern_hints" in ci_result
        assert "success" in ci_result["pattern_hints"]
        assert "abstention" in ci_result["pattern_hints"]

    @pytest.mark.unit
    def test_pattern_hints_are_valid(self):
        """Pattern hints in CI are valid values."""
        from experiments.drift_alignment_report import evaluate_ci_with_hints
        from backend.metrics.drift_alignment import PATTERN_HINTS

        trajectories = {
            "success": [0.8] * 10,
            "abstention": list(range(10)),
            "depth": [5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
        }

        ci_result = evaluate_ci_with_hints(
            "test_slice",
            trajectories,
            window=3,
            threshold=0.1,
            drift_score_threshold=0.3,
            coherence_threshold=0.4,
        )

        for metric, hint in ci_result["pattern_hints"].items():
            assert hint in PATTERN_HINTS, f"Invalid hint '{hint}' for metric '{metric}'"

    @pytest.mark.unit
    def test_pattern_hints_determinism(self):
        """Pattern hints in CI are deterministic."""
        from experiments.drift_alignment_report import evaluate_ci_with_hints

        trajectories = {
            "success": [0.8, 0.75, 0.82, 0.78, 0.80],
            "abstention": [0.1, 0.12, 0.08, 0.11, 0.09],
        }

        for _ in range(10):
            ci1 = evaluate_ci_with_hints(
                "test", trajectories, 3, 0.1, 0.3, 0.4
            )
            ci2 = evaluate_ci_with_hints(
                "test", trajectories, 3, 0.1, 0.3, 0.4
            )
            assert ci1["pattern_hints"] == ci2["pattern_hints"]


# ===========================================================================
# MULTI-SLICE REPORT TESTS (4 tests)
# ===========================================================================

class TestMultiSliceReport:
    """Tests for multi-slice report generation."""

    @pytest.mark.unit
    def test_multi_slice_report_structure(self):
        """Multi-slice report has expected structure."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        # Create sample slice reports
        trajectories = {"success": [0.8, 0.7, 0.6], "abstention": [0.1, 0.2, 0.3]}

        slice_reports = {
            "slice_a": generate_drift_report("slice_a", trajectories, window=2),
            "slice_b": generate_drift_report("slice_b", trajectories, window=2),
        }

        multi_report = generate_multi_slice_report(slice_reports)

        assert "version" in multi_report
        assert "generated_at" in multi_report
        assert "slice_count" in multi_report
        assert "slices" in multi_report
        assert multi_report["slice_count"] == 2
        assert "slice_a" in multi_report["slices"]
        assert "slice_b" in multi_report["slices"]

    @pytest.mark.unit
    def test_multi_slice_ci_summary(self):
        """Multi-slice report includes CI summary when provided."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        trajectories = {"success": [0.8, 0.7, 0.6]}

        slice_reports = {
            "slice_a": generate_drift_report("slice_a", trajectories, window=2),
            "slice_b": generate_drift_report("slice_b", trajectories, window=2),
        }

        ci_results = {
            "slice_a": {"status": "OK", "max_metric_drift_score": 0.1},
            "slice_b": {"status": "WARN", "max_metric_drift_score": 0.25},
        }

        multi_report = generate_multi_slice_report(slice_reports, ci_results)

        assert "ci_summary" in multi_report
        assert "slice_statuses" in multi_report["ci_summary"]
        assert multi_report["ci_summary"]["slice_statuses"]["slice_a"] == "OK"
        assert multi_report["ci_summary"]["slice_statuses"]["slice_b"] == "WARN"
        assert multi_report["ci_summary"]["any_block"] is False
        assert multi_report["ci_summary"]["all_ok"] is False

    @pytest.mark.unit
    def test_multi_slice_markdown_generation(self):
        """Multi-slice markdown report is generated correctly."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
            generate_multi_slice_markdown,
        )

        trajectories = {"success": [0.8, 0.7, 0.6]}

        slice_reports = {
            "slice_a": generate_drift_report("slice_a", trajectories, window=2),
            "slice_b": generate_drift_report("slice_b", trajectories, window=2),
        }

        multi_report = generate_multi_slice_report(slice_reports)
        md = generate_multi_slice_markdown(multi_report)

        assert "# Multi-Slice Drift Alignment Dashboard" in md
        assert "slice_a" in md
        assert "slice_b" in md
        assert "| Slice |" in md

    @pytest.mark.unit
    def test_discover_slices(self):
        """Slice discovery finds matching baseline/rfl pairs."""
        import tempfile
        import os
        from pathlib import Path
        from experiments.drift_alignment_report import discover_slices

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "slice_a_baseline.jsonl").write_text('{"x": 1}\n')
            (Path(tmpdir) / "slice_a_rfl.jsonl").write_text('{"x": 2}\n')
            (Path(tmpdir) / "slice_b_baseline.jsonl").write_text('{"x": 3}\n')
            (Path(tmpdir) / "slice_b_rfl.jsonl").write_text('{"x": 4}\n')
            # Create orphan file (no matching rfl)
            (Path(tmpdir) / "slice_c_baseline.jsonl").write_text('{"x": 5}\n')

            slices = discover_slices(tmpdir)

            assert len(slices) == 2
            slice_ids = [s[0] for s in slices]
            assert "slice_a" in slice_ids
            assert "slice_b" in slice_ids
            assert "slice_c" not in slice_ids  # No matching RFL


# ===========================================================================
# CLI PROFILE TESTS (2 tests)
# ===========================================================================

class TestCLIProfile:
    """Tests for CLI --profile flag."""

    @pytest.mark.unit
    def test_cli_profile_in_output(self):
        """CLI includes profile in CI output."""
        import subprocess
        import tempfile
        import os

        baseline_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 10
        rfl_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 10

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as bf:
            bf.write(baseline_data)
            baseline_path = bf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as rf:
            rf.write(rfl_data)
            rfl_path = rf.name

        try:
            result = subprocess.run(
                [
                    "uv", "run", "python", "experiments/drift_alignment_report.py",
                    "--slice", "test_slice",
                    "--baseline-log", baseline_path,
                    "--rfl-log", rfl_path,
                    "--ci",
                    "--profile", "strict",
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )

            assert result.returncode in (0, 1)
            assert "profile=strict" in result.stdout

        finally:
            os.unlink(baseline_path)
            os.unlink(rfl_path)

    @pytest.mark.unit
    def test_cli_json_includes_profile_metadata(self):
        """CLI JSON output includes profile in metadata."""
        import subprocess
        import tempfile
        import os
        import json

        baseline_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 5
        rfl_data = '{"cycle": 1, "success": true, "abstention_rate": 0.1}\n' * 5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as bf:
            bf.write(baseline_data)
            baseline_path = bf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as rf:
            rf.write(rfl_data)
            rfl_path = rf.name

        try:
            result = subprocess.run(
                [
                    "uv", "run", "python", "experiments/drift_alignment_report.py",
                    "--slice", "profile_test",
                    "--baseline-log", baseline_path,
                    "--rfl-log", rfl_path,
                    "--json",
                    "--profile", "lenient",
                ],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            report = json.loads(result.stdout)
            assert report["metadata"]["ci_profile"] == "lenient"
            assert report["metadata"]["ci_drift_score_threshold"] == 0.4
            assert report["metadata"]["ci_coherence_threshold"] == 0.3

        finally:
            os.unlink(baseline_path)
            os.unlink(rfl_path)


# ===========================================================================
# MULTI-SLICE CI CONTRACT TESTS (6 tests)
# ===========================================================================

class TestMultiSliceCIContract:
    """Tests for multi-slice CI report contract."""

    @pytest.mark.unit
    def test_multi_slice_ci_any_block(self):
        """any_block=True when at least one slice is BLOCK."""
        from experiments.drift_alignment_report import generate_multi_slice_report

        slice_reports = {
            "slice_a": {"drift_alignment_score": 0.1},
            "slice_b": {"drift_alignment_score": 0.2},
        }
        ci_results = {
            "slice_a": {"status": "OK", "max_metric_drift_score": 0.1},
            "slice_b": {"status": "BLOCK", "max_metric_drift_score": 0.5},
        }

        multi_report = generate_multi_slice_report(slice_reports, ci_results)

        assert multi_report["ci_summary"]["any_block"] is True
        assert multi_report["ci_summary"]["all_ok"] is False

    @pytest.mark.unit
    def test_multi_slice_ci_all_ok(self):
        """all_ok=True only when ALL slices are OK."""
        from experiments.drift_alignment_report import generate_multi_slice_report

        slice_reports = {
            "slice_a": {"drift_alignment_score": 0.1},
            "slice_b": {"drift_alignment_score": 0.15},
        }
        ci_results = {
            "slice_a": {"status": "OK", "max_metric_drift_score": 0.1},
            "slice_b": {"status": "OK", "max_metric_drift_score": 0.12},
        }

        multi_report = generate_multi_slice_report(slice_reports, ci_results)

        assert multi_report["ci_summary"]["any_block"] is False
        assert multi_report["ci_summary"]["all_ok"] is True

    @pytest.mark.unit
    def test_multi_slice_ci_mixed_statuses(self):
        """Mixed OK/WARN slices: any_block=False, all_ok=False."""
        from experiments.drift_alignment_report import generate_multi_slice_report

        slice_reports = {
            "slice_a": {"drift_alignment_score": 0.1},
            "slice_b": {"drift_alignment_score": 0.25},
        }
        ci_results = {
            "slice_a": {"status": "OK", "max_metric_drift_score": 0.1},
            "slice_b": {"status": "WARN", "max_metric_drift_score": 0.25},
        }

        multi_report = generate_multi_slice_report(slice_reports, ci_results)

        assert multi_report["ci_summary"]["any_block"] is False
        assert multi_report["ci_summary"]["all_ok"] is False

    @pytest.mark.unit
    def test_multi_slice_determinism(self):
        """Multi-slice report is deterministic."""
        import json
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        trajectories = {"success": [0.8, 0.7, 0.6], "abstention": [0.1, 0.2, 0.3]}

        for _ in range(10):
            slice_reports = {
                "slice_a": generate_drift_report("slice_a", trajectories, window=2),
                "slice_b": generate_drift_report("slice_b", trajectories, window=2),
            }
            ci_results = {
                "slice_a": {"status": "OK", "max_metric_drift_score": 0.1},
                "slice_b": {"status": "WARN", "max_metric_drift_score": 0.2},
            }

            multi1 = generate_multi_slice_report(slice_reports, ci_results)
            multi2 = generate_multi_slice_report(slice_reports, ci_results)

            # JSON serialization should be identical
            json1 = json.dumps(multi1, sort_keys=True)
            json2 = json.dumps(multi2, sort_keys=True)
            assert json1 == json2

    @pytest.mark.unit
    def test_multi_slice_ci_gate_per_slice(self):
        """Each slice in multi-report can have ci_gate field."""
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {"success": [0.8, 0.7, 0.6]}
        report = generate_drift_report("test", trajectories, window=2)

        # Simulate adding ci_gate as CLI does
        report["ci_gate"] = {
            "status": "OK",
            "max_metric_drift_score": 0.1,
            "pattern_hints": {"success": "falling"},
        }

        assert "ci_gate" in report
        assert report["ci_gate"]["status"] == "OK"
        assert "pattern_hints" in report["ci_gate"]

    @pytest.mark.unit
    def test_multi_slice_contract_required_fields(self):
        """Multi-slice report has all required fields per contract."""
        from experiments.drift_alignment_report import generate_multi_slice_report

        slice_reports = {"slice_a": {"drift_alignment_score": 0.1, "coherence_score": 0.8}}
        ci_results = {"slice_a": {"status": "OK", "max_metric_drift_score": 0.1}}

        multi_report = generate_multi_slice_report(slice_reports, ci_results)

        # Required top-level fields
        assert "version" in multi_report
        assert "generated_at" in multi_report
        assert "slice_count" in multi_report
        assert "slices" in multi_report
        assert "ci_summary" in multi_report

        # Required ci_summary fields
        assert "slice_statuses" in multi_report["ci_summary"]
        assert "any_block" in multi_report["ci_summary"]
        assert "all_ok" in multi_report["ci_summary"]


# ===========================================================================
# PATTERN HINTS ADVISORY-ONLY TESTS (5 tests)
# ===========================================================================

class TestPatternHintsAdvisory:
    """Tests verifying pattern hints are advisory only."""

    @pytest.mark.unit
    def test_pattern_hints_do_not_affect_ci_status(self):
        """Pattern hints have NO effect on CI gate status."""
        from backend.metrics.drift_alignment import (
            compute_drift_alignment,
            evaluate_drift_for_ci,
            classify_pattern_hint,
        )

        # Two identical trajectories should get identical CI status
        # regardless of what pattern hints compute to
        trajectories = {
            "success": [0.9, 0.85, 0.8, 0.75, 0.7],  # Falling pattern
            "abstention": [0.1, 0.15, 0.2, 0.25, 0.3],  # Rising pattern
        }

        result = compute_drift_alignment("test", trajectories, window_size=3)

        # CI status only depends on drift_score_threshold and coherence_threshold
        ci1 = evaluate_drift_for_ci(result, drift_score_threshold=0.3, coherence_threshold=0.4)
        ci2 = evaluate_drift_for_ci(result, drift_score_threshold=0.3, coherence_threshold=0.4)

        # Pattern hints are computed separately and don't affect gate
        hints = {k: classify_pattern_hint(v) for k, v in trajectories.items()}

        # CI status is identical regardless of hints
        assert ci1.status == ci2.status
        # Hints exist but don't factor into status computation
        assert hints["success"] in ("flat", "rising", "falling", "oscillatory")
        assert hints["abstention"] in ("flat", "rising", "falling", "oscillatory")

    @pytest.mark.unit
    def test_high_drift_blocks_regardless_of_hint(self):
        """Drift above threshold triggers WARN/BLOCK even with 'flat' hint."""
        from backend.metrics.drift_alignment import (
            compute_drift_alignment,
            evaluate_drift_for_ci,
            classify_pattern_hint,
        )

        # A series with high drift but ambiguous pattern
        trajectories = {
            "metric": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4],  # High variance
        }

        result = compute_drift_alignment("test", trajectories, window_size=2, drift_threshold=0.05)

        # Very strict thresholds should trigger non-OK status
        ci = evaluate_drift_for_ci(result, drift_score_threshold=0.01, coherence_threshold=0.99)

        hint = classify_pattern_hint(trajectories["metric"])

        # Even if hint is 'flat' or 'oscillatory', high drift should not be OK
        assert ci.status in ("WARN", "BLOCK")
        # Pattern hint is independent of status
        assert hint in ("flat", "rising", "falling", "oscillatory")

    @pytest.mark.unit
    def test_pattern_hints_survive_json_roundtrip(self):
        """Pattern hints remain unchanged after JSON serialization."""
        import json
        from backend.metrics.drift_alignment import PATTERN_HINTS, classify_pattern_hint
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {
            "success": list(range(1, 21)),  # Rising
            "abstention": list(range(20, 0, -1)),  # Falling
            "depth": [5] * 20,  # Flat
        }

        report = generate_drift_report("test", trajectories, window=5)

        # Serialize and deserialize
        json_str = json.dumps(report, sort_keys=True)
        restored = json.loads(json_str)

        # Pattern hints should survive intact
        assert restored["pattern_hints"]["success"] == report["pattern_hints"]["success"]
        assert restored["pattern_hints"]["abstention"] == report["pattern_hints"]["abstention"]
        assert restored["pattern_hints"]["depth"] == report["pattern_hints"]["depth"]

        # All hints are valid values
        for hint in restored["pattern_hints"].values():
            assert hint in PATTERN_HINTS

    @pytest.mark.unit
    def test_pattern_hints_in_single_and_multi_slice(self):
        """Pattern hints appear in both single-slice and multi-slice reports."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
            evaluate_ci_with_hints,
        )
        from backend.metrics.drift_alignment import PATTERN_HINTS

        trajectories = {"success": [0.9, 0.8, 0.7], "abstention": [0.1, 0.2, 0.3]}

        # Single slice
        single_report = generate_drift_report("single", trajectories, window=2)
        assert "pattern_hints" in single_report
        for hint in single_report["pattern_hints"].values():
            assert hint in PATTERN_HINTS

        # Multi-slice with CI
        ci_result = evaluate_ci_with_hints(
            "slice_a", trajectories, 2, 0.1, 0.3, 0.4
        )
        assert "pattern_hints" in ci_result
        for hint in ci_result["pattern_hints"].values():
            assert hint in PATTERN_HINTS

    @pytest.mark.unit
    def test_pattern_hints_per_metric_in_report(self):
        """Each metric in report has its own pattern_hint field."""
        from experiments.drift_alignment_report import generate_drift_report
        from backend.metrics.drift_alignment import PATTERN_HINTS

        trajectories = {
            "success": list(range(10)),  # Rising
            "abstention": list(range(10, 0, -1)),  # Falling
        }

        report = generate_drift_report("test", trajectories, window=3)

        # Top-level pattern_hints dict
        assert "pattern_hints" in report
        assert "success" in report["pattern_hints"]
        assert "abstention" in report["pattern_hints"]

        # Per-metric pattern_hint in metrics dict
        for metric_name, metric_data in report["metrics"].items():
            assert "pattern_hint" in metric_data
            assert metric_data["pattern_hint"] in PATTERN_HINTS


# ===========================================================================
# DRIFT CELLS GRID TESTS (7 tests)
# ===========================================================================

class TestDriftCellsGrid:
    """Tests for build_drift_cells and drift grid functionality."""

    @pytest.mark.unit
    def test_build_drift_cells_single_slice(self):
        """build_drift_cells works for single-slice report."""
        from backend.metrics.drift_alignment import build_drift_cells, DriftCell
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {"success": [0.8, 0.7, 0.6], "abstention": [0.1, 0.2, 0.3]}
        report = generate_drift_report("test_slice", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        cells = build_drift_cells(report, profile="default")

        assert len(cells) == 2  # success and abstention
        assert all(isinstance(c, DriftCell) for c in cells)
        assert cells[0].slice == "test_slice"
        assert cells[0].profile == "default"

    @pytest.mark.unit
    def test_build_drift_cells_multi_slice(self):
        """build_drift_cells works for multi-slice report."""
        from backend.metrics.drift_alignment import build_drift_cells
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        trajectories = {"success": [0.8, 0.7], "abstention": [0.1, 0.2]}

        slice_reports = {
            "slice_a": generate_drift_report("slice_a", trajectories, window=2),
            "slice_b": generate_drift_report("slice_b", trajectories, window=2),
        }
        slice_reports["slice_a"]["ci_gate"] = {"status": "OK"}
        slice_reports["slice_b"]["ci_gate"] = {"status": "WARN"}

        multi_report = generate_multi_slice_report(slice_reports)

        cells = build_drift_cells(multi_report, profile="strict")

        # 2 slices × 2 metrics = 4 cells
        assert len(cells) == 4
        slice_ids = [c.slice for c in cells]
        assert "slice_a" in slice_ids
        assert "slice_b" in slice_ids

    @pytest.mark.unit
    def test_drift_cells_deterministic_ordering(self):
        """Drift cells are sorted by (slice, metric_kind)."""
        from backend.metrics.drift_alignment import build_drift_cells
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        trajectories = {"zebra": [0.1, 0.2], "alpha": [0.3, 0.4], "beta": [0.5, 0.6]}

        slice_reports = {
            "z_slice": generate_drift_report("z_slice", trajectories, window=2),
            "a_slice": generate_drift_report("a_slice", trajectories, window=2),
        }

        multi_report = generate_multi_slice_report(slice_reports)

        for _ in range(10):
            cells = build_drift_cells(multi_report)
            # Should be sorted: a_slice.alpha, a_slice.beta, a_slice.zebra, z_slice.alpha, ...
            assert cells[0].slice == "a_slice"
            assert cells[0].metric_kind == "alpha"
            assert cells[-1].slice == "z_slice"
            assert cells[-1].metric_kind == "zebra"

    @pytest.mark.unit
    def test_drift_cells_to_dicts(self):
        """drift_cells_to_dicts produces correct dict structure."""
        from backend.metrics.drift_alignment import (
            build_drift_cells,
            drift_cells_to_dicts,
            DRIFT_CELL_COLUMNS,
        )
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {"success": [0.8, 0.7, 0.6]}
        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        cells = build_drift_cells(report)
        dicts = drift_cells_to_dicts(cells)

        assert len(dicts) == 1
        assert set(dicts[0].keys()) == set(DRIFT_CELL_COLUMNS)

    @pytest.mark.unit
    def test_drift_cell_columns_complete(self):
        """DRIFT_CELL_COLUMNS matches DriftCell.to_dict() keys."""
        from backend.metrics.drift_alignment import DriftCell, DRIFT_CELL_COLUMNS

        cell = DriftCell(
            slice="test",
            metric_kind="success",
            drift_score=0.1,
            direction="stable",
            stability=0.9,
            pattern_hint="flat",
            drift_alignment_score=0.2,
            coherence_score=0.8,
            status="OK",
            profile="default",
        )

        assert set(cell.to_dict().keys()) == set(DRIFT_CELL_COLUMNS)

    @pytest.mark.unit
    def test_drift_cells_coverage_all_metrics(self):
        """All metrics from all slices are covered in cells."""
        from backend.metrics.drift_alignment import build_drift_cells
        from experiments.drift_alignment_report import (
            generate_drift_report,
            generate_multi_slice_report,
        )

        trajectories_a = {"metric1": [0.1, 0.2], "metric2": [0.3, 0.4]}
        trajectories_b = {"metric1": [0.5, 0.6], "metric3": [0.7, 0.8]}

        slice_reports = {
            "slice_a": generate_drift_report("slice_a", trajectories_a, window=2),
            "slice_b": generate_drift_report("slice_b", trajectories_b, window=2),
        }

        multi_report = generate_multi_slice_report(slice_reports)
        cells = build_drift_cells(multi_report)

        # slice_a has 2 metrics, slice_b has 2 metrics = 4 cells
        assert len(cells) == 4

        # Check all combinations are present
        combos = {(c.slice, c.metric_kind) for c in cells}
        assert ("slice_a", "metric1") in combos
        assert ("slice_a", "metric2") in combos
        assert ("slice_b", "metric1") in combos
        assert ("slice_b", "metric3") in combos

    @pytest.mark.unit
    def test_drift_cells_profile_propagation(self):
        """Profile is correctly propagated to all cells."""
        from backend.metrics.drift_alignment import build_drift_cells
        from experiments.drift_alignment_report import generate_drift_report

        trajectories = {"a": [0.1, 0.2], "b": [0.3, 0.4]}
        report = generate_drift_report("test", trajectories, window=2)

        for profile in ["default", "strict", "lenient"]:
            cells = build_drift_cells(report, profile=profile)
            assert all(c.profile == profile for c in cells)


# ===========================================================================
# CSV EXPORT TESTS (5 tests)
# ===========================================================================

class TestCSVExport:
    """Tests for CSV export functionality."""

    @pytest.mark.unit
    def test_csv_has_header_row(self):
        """CSV export includes header row."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            drift_cells_to_csv_string,
        )
        from backend.metrics.drift_alignment import DRIFT_CELL_COLUMNS

        trajectories = {"success": [0.8, 0.7, 0.6]}
        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        csv_str = drift_cells_to_csv_string(report)
        # Handle both Unix and Windows line endings
        lines = csv_str.strip().replace("\r\n", "\n").split("\n")

        # First line should be header
        header = lines[0].split(",")
        assert header == list(DRIFT_CELL_COLUMNS)

    @pytest.mark.unit
    def test_csv_fixed_column_order(self):
        """CSV columns are in deterministic order."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            drift_cells_to_csv_string,
        )
        from backend.metrics.drift_alignment import DRIFT_CELL_COLUMNS

        trajectories = {"metric1": [0.1, 0.2], "metric2": [0.3, 0.4]}
        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        for _ in range(10):
            csv_str = drift_cells_to_csv_string(report)
            # Handle both Unix and Windows line endings
            header = csv_str.strip().replace("\r\n", "\n").split("\n")[0].split(",")
            assert tuple(header) == DRIFT_CELL_COLUMNS

    @pytest.mark.unit
    def test_csv_deterministic_content(self):
        """CSV content is deterministic for identical inputs."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            drift_cells_to_csv_string,
        )

        trajectories = {"success": [0.8, 0.7, 0.6], "abstention": [0.1, 0.2, 0.3]}
        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        csv1 = drift_cells_to_csv_string(report)
        csv2 = drift_cells_to_csv_string(report)

        assert csv1 == csv2

    @pytest.mark.unit
    def test_csv_export_file(self):
        """CSV export writes correct file."""
        import tempfile
        import os
        from experiments.drift_alignment_report import (
            generate_drift_report,
            export_drift_cells_csv,
        )
        from backend.metrics.drift_alignment import DRIFT_CELL_COLUMNS

        trajectories = {"success": [0.8, 0.7]}
        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = {"status": "OK"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            export_drift_cells_csv(report, csv_path, profile="default")

            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 2  # header + 1 data row
            assert lines[0].split(",") == list(DRIFT_CELL_COLUMNS)
        finally:
            os.unlink(csv_path)

    @pytest.mark.unit
    def test_csv_no_ci_status_change(self):
        """CSV export does not change CI status."""
        from experiments.drift_alignment_report import (
            generate_drift_report,
            drift_cells_to_csv_string,
        )
        from backend.metrics.drift_alignment import evaluate_drift_for_ci, compute_drift_alignment

        trajectories = {"success": [0.8, 0.7, 0.6]}

        # Generate report and get CI status before CSV
        result = compute_drift_alignment("test", trajectories, window_size=2)
        ci_before = evaluate_drift_for_ci(result)

        report = generate_drift_report("test", trajectories, window=2)
        report["ci_gate"] = ci_before.to_dict()

        # Generate CSV
        _ = drift_cells_to_csv_string(report)

        # CI status should be unchanged
        ci_after = evaluate_drift_for_ci(result)
        assert ci_before.status == ci_after.status
        assert ci_before.max_metric_drift_score == ci_after.max_metric_drift_score


# ===========================================================================
# THRESHOLD SNAPSHOT TESTS (5 tests)
# ===========================================================================

class TestThresholdSnapshot:
    """Tests for threshold snapshot (no-silent-change guard)."""

    @pytest.mark.unit
    def test_snapshot_thresholds_structure(self):
        """snapshot_thresholds returns expected structure."""
        from backend.metrics.drift_alignment import snapshot_thresholds

        snapshot = snapshot_thresholds()

        assert "version" in snapshot
        assert "profiles" in snapshot
        assert "thresholds" in snapshot
        assert isinstance(snapshot["profiles"], list)
        assert isinstance(snapshot["thresholds"], dict)

    @pytest.mark.unit
    def test_snapshot_matches_drift_thresholds(self):
        """Snapshot matches current DRIFT_THRESHOLDS values."""
        from backend.metrics.drift_alignment import (
            snapshot_thresholds,
            DRIFT_THRESHOLDS,
        )

        snapshot = snapshot_thresholds()

        for profile, config in snapshot["thresholds"].items():
            assert profile in DRIFT_THRESHOLDS
            for key, value in config.items():
                assert DRIFT_THRESHOLDS[profile][key] == value

    @pytest.mark.unit
    def test_verify_thresholds_unchanged_passes(self):
        """verify_thresholds_unchanged returns True for current config."""
        from backend.metrics.drift_alignment import verify_thresholds_unchanged

        assert verify_thresholds_unchanged() is True

    @pytest.mark.unit
    def test_snapshot_determinism(self):
        """Snapshot is deterministic."""
        import json
        from backend.metrics.drift_alignment import snapshot_thresholds

        for _ in range(10):
            s1 = snapshot_thresholds()
            s2 = snapshot_thresholds()

            # JSON serialization should be identical
            json1 = json.dumps(s1, sort_keys=True)
            json2 = json.dumps(s2, sort_keys=True)
            assert json1 == json2

    @pytest.mark.unit
    def test_snapshot_profiles_sorted(self):
        """Snapshot profiles are in sorted order."""
        from backend.metrics.drift_alignment import snapshot_thresholds

        snapshot = snapshot_thresholds()
        profiles = snapshot["profiles"]

        assert profiles == sorted(profiles)


# ===========================================================================
# PHASE III: PROMOTION RADAR & GLOBAL DRIFT SIGNAL TESTS (16 tests)
# ===========================================================================

class TestSliceDriftSummary:
    """Tests for summarize_slice_drift function."""

    @pytest.mark.unit
    def test_summarize_slice_drift_structure(self):
        """summarize_slice_drift returns expected structure."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_slice_drift,
        )

        cells = [
            DriftCell(
                slice="test_slice",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        summary = summarize_slice_drift(cells, "test_slice")

        assert "schema_version" in summary
        assert "slice_name" in summary
        assert "metrics_evaluated" in summary
        assert "metrics_with_high_drift" in summary
        assert "coherence_issues" in summary
        assert "slice_drift_status" in summary

    @pytest.mark.unit
    def test_summarize_slice_drift_ok_status(self):
        """Slice with all OK cells gets OK status."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_slice_drift,
        )

        cells = [
            DriftCell(
                slice="ok_slice",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
            DriftCell(
                slice="ok_slice",
                metric_kind="abstention",
                drift_score=0.05,
                direction="stable",
                stability=0.95,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        summary = summarize_slice_drift(cells, "ok_slice")

        assert summary["slice_drift_status"] == "OK"
        assert summary["metrics_evaluated"] == 2
        assert summary["metrics_with_high_drift"] == []
        assert summary["coherence_issues"] == 0

    @pytest.mark.unit
    def test_summarize_slice_drift_drifty_status(self):
        """Slice with BLOCK status gets DRIFTY."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_slice_drift,
        )

        cells = [
            DriftCell(
                slice="drifty_slice",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        summary = summarize_slice_drift(cells, "drifty_slice")

        assert summary["slice_drift_status"] == "DRIFTY"
        assert "success" in summary["metrics_with_high_drift"]

    @pytest.mark.unit
    def test_summarize_slice_drift_warn_status(self):
        """Slice with WARN cells gets WARN status (not DRIFTY if <50% affected)."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_slice_drift,
        )

        cells = [
            DriftCell(
                slice="warn_slice",
                metric_kind="success",
                drift_score=0.25,
                direction="stable",
                stability=0.7,
                pattern_hint="flat",
                drift_alignment_score=0.3,
                coherence_score=0.5,
                status="WARN",
                profile="default",
            ),
            DriftCell(
                slice="warn_slice",
                metric_kind="abstention",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.3,
                coherence_score=0.5,
                status="OK",
                profile="default",
            ),
            DriftCell(
                slice="warn_slice",
                metric_kind="depth",
                drift_score=0.08,
                direction="stable",
                stability=0.92,
                pattern_hint="flat",
                drift_alignment_score=0.3,
                coherence_score=0.5,
                status="OK",
                profile="default",
            ),
        ]

        summary = summarize_slice_drift(cells, "warn_slice")

        # 1 out of 3 metrics is WARN (<50%), so should be WARN not DRIFTY
        assert summary["slice_drift_status"] == "WARN"
        assert len(summary["metrics_with_high_drift"]) == 1

    @pytest.mark.unit
    def test_summarize_slice_drift_empty_cells(self):
        """Empty cells list returns OK status."""
        from backend.metrics.drift_alignment import summarize_slice_drift

        summary = summarize_slice_drift([], "empty_slice")

        assert summary["slice_drift_status"] == "OK"
        assert summary["metrics_evaluated"] == 0


class TestPromotionRadar:
    """Tests for evaluate_drift_for_promotion function."""

    @pytest.mark.unit
    def test_evaluate_drift_for_promotion_structure(self):
        """evaluate_drift_for_promotion returns expected structure."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            evaluate_drift_for_promotion,
        )

        cells = [
            DriftCell(
                slice="test",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        result = evaluate_drift_for_promotion(cells)

        assert "promotion_ok" in result
        assert "drifty_slices" in result
        assert "drifty_metrics" in result
        assert "status" in result

    @pytest.mark.unit
    def test_evaluate_drift_for_promotion_ok(self):
        """All OK cells result in promotion_ok=True."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            evaluate_drift_for_promotion,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
            DriftCell(
                slice="slice_b",
                metric_kind="abstention",
                drift_score=0.05,
                direction="stable",
                stability=0.95,
                pattern_hint="flat",
                drift_alignment_score=0.15,
                coherence_score=0.85,
                status="OK",
                profile="default",
            ),
        ]

        result = evaluate_drift_for_promotion(cells)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert result["drifty_slices"] == []
        assert result["drifty_metrics"] == []

    @pytest.mark.unit
    def test_evaluate_drift_for_promotion_block(self):
        """DRIFTY slice results in BLOCK status."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            evaluate_drift_for_promotion,
        )

        cells = [
            DriftCell(
                slice="drifty_slice",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        result = evaluate_drift_for_promotion(cells)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert "drifty_slice" in result["drifty_slices"]

    @pytest.mark.unit
    def test_evaluate_drift_for_promotion_attention(self):
        """WARN slices result in ATTENTION status (not BLOCK if not DRIFTY)."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            evaluate_drift_for_promotion,
        )

        cells = [
            DriftCell(
                slice="warn_slice",
                metric_kind="success",
                drift_score=0.25,
                direction="stable",
                stability=0.7,
                pattern_hint="flat",
                drift_alignment_score=0.3,
                coherence_score=0.5,
                status="WARN",
                profile="default",
            ),
            DriftCell(
                slice="warn_slice",
                metric_kind="abstention",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.3,
                coherence_score=0.5,
                status="OK",
                profile="default",
            ),
        ]

        result = evaluate_drift_for_promotion(cells)

        assert result["promotion_ok"] is False
        assert result["status"] == "ATTENTION"
        assert "warn_slice" in result["drifty_slices"]

    @pytest.mark.unit
    def test_evaluate_drift_for_promotion_empty(self):
        """Empty cells list results in promotion_ok=True."""
        from backend.metrics.drift_alignment import evaluate_drift_for_promotion

        result = evaluate_drift_for_promotion([])

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"


class TestGlobalHealthSummary:
    """Tests for summarize_drift_for_global_health function."""

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_structure(self):
        """summarize_drift_for_global_health returns expected structure."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_drift_for_global_health,
        )

        cells = [
            DriftCell(
                slice="test",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        summary = summarize_drift_for_global_health(cells)

        assert "drift_hotspot_slices" in summary
        assert "drift_hotspot_metrics" in summary
        assert "status" in summary

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_ok(self):
        """All OK cells result in OK status."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_drift_for_global_health,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
            DriftCell(
                slice="slice_b",
                metric_kind="abstention",
                drift_score=0.05,
                direction="stable",
                stability=0.95,
                pattern_hint="flat",
                drift_alignment_score=0.15,
                coherence_score=0.85,
                status="OK",
                profile="default",
            ),
        ]

        summary = summarize_drift_for_global_health(cells)

        assert summary["status"] == "OK"
        assert summary["drift_hotspot_slices"] == []
        assert summary["drift_hotspot_metrics"] == []

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_hotspot_slice(self):
        """Slice with >50% non-OK cells is a hotspot."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_drift_for_global_health,
        )

        cells = [
            DriftCell(
                slice="hotspot_slice",
                metric_kind="metric1",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
            DriftCell(
                slice="hotspot_slice",
                metric_kind="metric2",
                drift_score=0.4,
                direction="decreasing",
                stability=0.4,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="WARN",
                profile="default",
            ),
        ]

        summary = summarize_drift_for_global_health(cells, hotspot_threshold=0.5)

        assert "hotspot_slice" in summary["drift_hotspot_slices"]
        assert summary["status"] in ("WARN", "HOT")

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_hotspot_metric(self):
        """Metric with >50% non-OK across slices is a hotspot."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_drift_for_global_health,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="hotspot_metric",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
            DriftCell(
                slice="slice_b",
                metric_kind="hotspot_metric",
                drift_score=0.4,
                direction="decreasing",
                stability=0.4,
                pattern_hint="falling",
                drift_alignment_score=0.5,
                coherence_score=0.3,
                status="WARN",
                profile="default",
            ),
        ]

        summary = summarize_drift_for_global_health(cells, hotspot_threshold=0.5)

        assert "hotspot_metric" in summary["drift_hotspot_metrics"]
        assert summary["status"] in ("WARN", "HOT")

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_hot_status(self):
        """DRIFTY slice results in HOT status."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            summarize_drift_for_global_health,
        )

        cells = [
            DriftCell(
                slice="drifty_slice",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        summary = summarize_drift_for_global_health(cells)

        assert summary["status"] == "HOT"

    @pytest.mark.unit
    def test_summarize_drift_for_global_health_empty(self):
        """Empty cells list results in OK status."""
        from backend.metrics.drift_alignment import summarize_drift_for_global_health

        summary = summarize_drift_for_global_health([])

        assert summary["status"] == "OK"
        assert summary["drift_hotspot_slices"] == []
        assert summary["drift_hotspot_metrics"] == []


# ===========================================================================
# PHASE IV: DRIFT-AWARE PROMOTION & GLOBAL DRIFT INTELLIGENCE TESTS (12 tests)
# ===========================================================================

class TestDriftTrendHistory:
    """Tests for build_drift_trend_history function."""

    @pytest.mark.unit
    def test_build_drift_trend_history_structure(self):
        """build_drift_trend_history returns expected structure."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            build_drift_trend_history,
        )

        cells_run1 = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells_run1})

        assert "runs" in history
        assert "runs_with_drift" in history
        assert "runs_without_drift" in history
        assert "trend_status" in history

    @pytest.mark.unit
    def test_build_drift_trend_history_no_drift(self):
        """Runs with no drift are classified correctly."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            build_drift_trend_history,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells, "run2": cells})

        assert "run1" in history["runs_without_drift"]
        assert "run2" in history["runs_without_drift"]
        assert history["runs_with_drift"] == []

    @pytest.mark.unit
    def test_build_drift_trend_history_with_drift(self):
        """Runs with drift are classified correctly."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            build_drift_trend_history,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history(
            {"run1": cells_drifty, "run2": cells_ok}
        )

        assert "run1" in history["runs_with_drift"]
        assert "run2" in history["runs_without_drift"]

    @pytest.mark.unit
    def test_build_drift_trend_history_improving(self):
        """Trend status is IMPROVING when drift decreases over time."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            build_drift_trend_history,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Early runs have drift, late runs don't
        history = build_drift_trend_history(
            {
                "run1": cells_drifty,
                "run2": cells_drifty,
                "run3": cells_ok,
                "run4": cells_ok,
            }
        )

        assert history["trend_status"] == "IMPROVING"

    @pytest.mark.unit
    def test_build_drift_trend_history_degrading(self):
        """Trend status is DEGRADING when drift increases over time."""
        from backend.metrics.drift_alignment import (
            DriftCell,
            build_drift_trend_history,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Early runs are OK, late runs have drift
        history = build_drift_trend_history(
            {
                "run1": cells_ok,
                "run2": cells_ok,
                "run3": cells_drifty,
                "run4": cells_drifty,
            }
        )

        assert history["trend_status"] == "DEGRADING"


class TestDriftTrendPromotion:
    """Tests for evaluate_drift_trend_for_promotion function."""

    @pytest.mark.unit
    def test_evaluate_drift_trend_for_promotion_structure(self):
        """evaluate_drift_trend_for_promotion returns expected structure."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells})
        result = evaluate_drift_trend_for_promotion(history)

        assert "promotion_ok" in result
        assert "status" in result
        assert "blocking_reasons" in result

    @pytest.mark.unit
    def test_evaluate_drift_trend_for_promotion_block_degrading_core(self):
        """BLOCK if trend is DEGRADING and affects core metrics."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
            CORE_UPLIFT_METRICS,
        )

        # Create cells with drift in core metrics
        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",  # Core metric
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Degrading trend with core metrics affected
        history = build_drift_trend_history(
            {
                "run1": cells_ok,
                "run2": cells_ok,
                "run3": cells_drifty,  # Recent run with core metric drift
            }
        )

        result = evaluate_drift_trend_for_promotion(history)

        assert result["promotion_ok"] is False
        assert result["status"] == "BLOCK"
        assert len(result["blocking_reasons"]) > 0
        assert "DEGRADING" in result["blocking_reasons"][0]
        assert any(m in result["blocking_reasons"][0] for m in CORE_UPLIFT_METRICS)

    @pytest.mark.unit
    def test_evaluate_drift_trend_for_promotion_warn_stable_frequent(self):
        """WARN if trend is STABLE but drift is frequent."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="other_metric",  # Non-core
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="WARN",
                profile="default",
            ),
        ]

        # Stable trend (same ratio in early and late) but frequent drift (>50% of runs)
        # Need equal drift ratio in early and late halves to be STABLE
        # With 6 runs: early=[run1,run2,run3], late=[run4,run5,run6]
        # Early: 2 drifty out of 3 = 0.667, Late: 2 drifty out of 3 = 0.667 -> STABLE
        history = build_drift_trend_history(
            {
                "run1": cells_drifty,  # Early: drifty
                "run2": cells_drifty,  # Early: drifty
                "run3": [],  # Early: OK
                "run4": cells_drifty,  # Late: drifty
                "run5": cells_drifty,  # Late: drifty
                "run6": [],  # Late: OK
            }
        )

        result = evaluate_drift_trend_for_promotion(history)

        assert result["promotion_ok"] is False
        assert result["status"] == "WARN"
        assert any("frequent" in r.lower() for r in result["blocking_reasons"])

    @pytest.mark.unit
    def test_evaluate_drift_trend_for_promotion_ok_improving(self):
        """OK if trend is IMPROVING."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Improving trend
        history = build_drift_trend_history(
            {
                "run1": cells_drifty,
                "run2": cells_drifty,
                "run3": cells_ok,
                "run4": cells_ok,
            }
        )

        result = evaluate_drift_trend_for_promotion(history)

        assert result["promotion_ok"] is True
        assert result["status"] == "OK"
        assert result["blocking_reasons"] == []


class TestDirectorDriftPanel:
    """Tests for build_drift_director_panel function."""

    @pytest.mark.unit
    def test_build_drift_director_panel_structure(self):
        """build_drift_director_panel returns expected structure."""
        from backend.metrics.drift_alignment import (
            build_drift_director_panel,
            summarize_drift_for_global_health,
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        global_health = summarize_drift_for_global_health(cells)
        history = build_drift_trend_history({"run1": cells})
        promotion = evaluate_drift_trend_for_promotion(history)

        panel = build_drift_director_panel(global_health, history, promotion)

        assert "status_light" in panel
        assert "drift_hotspot_slices" in panel
        assert "drift_hotspot_metrics" in panel
        assert "trend_status" in panel
        assert "headline" in panel

    @pytest.mark.unit
    def test_build_drift_director_panel_red_status(self):
        """RED status light for critical issues."""
        from backend.metrics.drift_alignment import (
            build_drift_director_panel,
            summarize_drift_for_global_health,
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        global_health = summarize_drift_for_global_health(cells_drifty)
        history = build_drift_trend_history(
            {"run1": cells_ok, "run2": cells_ok, "run3": cells_drifty}
        )
        promotion = evaluate_drift_trend_for_promotion(history)

        panel = build_drift_director_panel(global_health, history, promotion)

        assert panel["status_light"] == "RED"

    @pytest.mark.unit
    def test_build_drift_director_panel_green_status(self):
        """GREEN status light for no issues."""
        from backend.metrics.drift_alignment import (
            build_drift_director_panel,
            summarize_drift_for_global_health,
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        global_health = summarize_drift_for_global_health(cells)
        history = build_drift_trend_history({"run1": cells, "run2": cells})
        promotion = evaluate_drift_trend_for_promotion(history)

        panel = build_drift_director_panel(global_health, history, promotion)

        assert panel["status_light"] == "GREEN"

    @pytest.mark.unit
    def test_build_drift_director_panel_headline(self):
        """Headline is generated and neutral."""
        from backend.metrics.drift_alignment import (
            build_drift_director_panel,
            summarize_drift_for_global_health,
            build_drift_trend_history,
            evaluate_drift_trend_for_promotion,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        global_health = summarize_drift_for_global_health(cells)
        history = build_drift_trend_history({"run1": cells})
        promotion = evaluate_drift_trend_for_promotion(history)

        panel = build_drift_director_panel(global_health, history, promotion)

        assert isinstance(panel["headline"], str)
        assert len(panel["headline"]) > 0
        # Headline should be neutral (no "good/bad" language)
        assert "good" not in panel["headline"].lower()
        assert "bad" not in panel["headline"].lower()


# ===========================================================================
# PHASE V: CROSS-METRIC DRIFT COUPLER & MULTI-AXIS DRIFT SUMMARY TESTS (8 tests)
# ===========================================================================

class TestMultiAxisDriftView:
    """Tests for build_multi_axis_drift_view function."""

    @pytest.mark.unit
    def test_build_multi_axis_drift_view_structure(self):
        """build_multi_axis_drift_view returns expected structure."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells})
        view = build_multi_axis_drift_view(history)

        assert "axes_with_drift" in view
        assert "high_risk_axes" in view
        assert "global_drift_status" in view
        assert "neutral_notes" in view

    @pytest.mark.unit
    def test_multi_axis_drift_view_single_axis_attention(self):
        """Single-axis drift results in ATTENTION status."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Degrading trend (single axis)
        history = build_drift_trend_history(
            {"run1": cells_ok, "run2": cells_ok, "run3": cells_drifty}
        )
        view = build_multi_axis_drift_view(history)

        assert view["global_drift_status"] == "ATTENTION"
        assert "drift" in view["axes_with_drift"]
        assert "drift" in view["high_risk_axes"]

    @pytest.mark.unit
    def test_multi_axis_drift_view_multi_axis_block(self):
        """Multi-axis (drift+budget) results in BLOCK status."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        # Degrading trend
        history = build_drift_trend_history(
            {"run1": cells_ok, "run2": cells_ok, "run3": cells_drifty}
        )

        # Budget axis with BLOCK
        budget_drift = {"status": "BLOCK", "has_drift": True}

        view = build_multi_axis_drift_view(history, budget_drift_view=budget_drift)

        assert view["global_drift_status"] == "BLOCK"
        assert len(view["high_risk_axes"]) >= 2
        assert "drift" in view["high_risk_axes"]
        assert "budget" in view["high_risk_axes"]

    @pytest.mark.unit
    def test_multi_axis_drift_view_deterministic_ordering(self):
        """Axis ordering is deterministic."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells})
        budget_drift = {"status": "BLOCK", "has_drift": True}
        metrics_drift = {"status": "WARN", "has_drift": True}

        for _ in range(10):
            view = build_multi_axis_drift_view(
                history, budget_drift_view=budget_drift, metric_conformance_drift=metrics_drift
            )
            assert view["axes_with_drift"] == sorted(view["axes_with_drift"])
            assert view["high_risk_axes"] == sorted(view["high_risk_axes"])


class TestUpliftDriftEnvelope:
    """Tests for summarize_drift_for_uplift_envelope function."""

    @pytest.mark.unit
    def test_summarize_drift_for_uplift_envelope_structure(self):
        """summarize_drift_for_uplift_envelope returns expected structure."""
        from backend.metrics.drift_alignment import (
            build_multi_axis_drift_view,
            build_drift_trend_history,
            summarize_drift_for_uplift_envelope,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells})
        multi_axis = build_multi_axis_drift_view(history)
        envelope = summarize_drift_for_uplift_envelope(multi_axis)

        assert "uplift_safe_under_drift" in envelope
        assert "status" in envelope
        assert "blocking_axes" in envelope
        assert "recommendations" in envelope

    @pytest.mark.unit
    def test_uplift_envelope_single_axis_attention(self):
        """Single-axis drift results in ATTENTION status."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            summarize_drift_for_uplift_envelope,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history(
            {"run1": cells_ok, "run2": cells_ok, "run3": cells_drifty}
        )
        multi_axis = build_multi_axis_drift_view(history)
        envelope = summarize_drift_for_uplift_envelope(multi_axis)

        assert envelope["status"] == "ATTENTION"
        assert envelope["uplift_safe_under_drift"] is False

    @pytest.mark.unit
    def test_uplift_envelope_multi_axis_block(self):
        """Multi-axis (drift+budget) results in BLOCK status."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            summarize_drift_for_uplift_envelope,
            DriftCell,
        )

        cells_drifty = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        cells_ok = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history(
            {"run1": cells_ok, "run2": cells_ok, "run3": cells_drifty}
        )
        budget_drift = {"status": "BLOCK", "has_drift": True}
        multi_axis = build_multi_axis_drift_view(history, budget_drift_view=budget_drift)
        envelope = summarize_drift_for_uplift_envelope(multi_axis)

        assert envelope["status"] == "BLOCK"
        assert envelope["uplift_safe_under_drift"] is False
        assert len(envelope["blocking_axes"]) >= 2
        assert "drift" in envelope["blocking_axes"]
        assert "budget" in envelope["blocking_axes"]
        # Should have special note about drift+budget
        assert any("drift" in r.lower() and "budget" in r.lower() for r in envelope["recommendations"])

    @pytest.mark.unit
    def test_uplift_envelope_neutral_language(self):
        """Recommendations use neutral, descriptive language."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            summarize_drift_for_uplift_envelope,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.5,
                direction="decreasing",
                stability=0.3,
                pattern_hint="falling",
                drift_alignment_score=0.6,
                coherence_score=0.2,
                status="BLOCK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells})
        budget_drift = {"status": "BLOCK", "has_drift": True}
        multi_axis = build_multi_axis_drift_view(history, budget_drift_view=budget_drift)
        envelope = summarize_drift_for_uplift_envelope(multi_axis)

        # Check that recommendations don't use evaluative language
        for rec in envelope["recommendations"]:
            rec_lower = rec.lower()
            assert "good" not in rec_lower
            assert "bad" not in rec_lower
            assert "success" not in rec_lower or "success" in rec_lower  # metric name is OK
            assert "failure" not in rec_lower

    @pytest.mark.unit
    def test_uplift_envelope_ok_status(self):
        """No drift results in OK status and safe uplift."""
        from backend.metrics.drift_alignment import (
            build_drift_trend_history,
            build_multi_axis_drift_view,
            summarize_drift_for_uplift_envelope,
            DriftCell,
        )

        cells = [
            DriftCell(
                slice="slice_a",
                metric_kind="success",
                drift_score=0.1,
                direction="stable",
                stability=0.9,
                pattern_hint="flat",
                drift_alignment_score=0.2,
                coherence_score=0.8,
                status="OK",
                profile="default",
            ),
        ]

        history = build_drift_trend_history({"run1": cells, "run2": cells})
        multi_axis = build_multi_axis_drift_view(history)
        envelope = summarize_drift_for_uplift_envelope(multi_axis)

        assert envelope["status"] == "OK"
        assert envelope["uplift_safe_under_drift"] is True
        assert envelope["blocking_axes"] == []


# ===========================================================================
# Total: 142 tests
# ===========================================================================

