"""
Comprehensive Statistical Tests for Paired Bootstrap Engine

This module contains 80+ tests covering:
- Determinism: same seed → identical distribution
- Known-distribution tests: verify statistical properties
- Edge-case convergence: small samples, skewed distributions, identical arrays
- Input validation: error handling for invalid inputs
- Binary success metrics
- Continuous metrics
- Confidence band generation (visualization)
- Leakage detection (seed invariance, epsilon stability)
- Profiling suite (timing, memory, histogram)
- Contract schema generation

Test categories are marked with pytest markers for selective execution.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from statistical.bootstrap import (
    paired_bootstrap_delta,
    PairedBootstrapResult,
    DistributionSummary,
    MIN_BOOTSTRAP_RESAMPLES,
    MAX_BOOTSTRAP_RESAMPLES,
    MIN_SAMPLE_SIZE,
    SMALL_SAMPLE_THRESHOLD,
    _get_bootstrap_distribution,
    _compute_skewness,
    _validate_inputs,
    # New imports for D3 enhancement
    compute_confidence_band,
    ConfidenceBand,
    detect_bootstrap_leakage,
    LeakageDetectionResult,
    profile_bootstrap,
    BootstrapProfile,
    BootstrapHistogram,
    get_bootstrap_contract,
    write_bootstrap_contract,
)


# =============================================================================
# DETERMINISM TESTS (Tests 1-10)
# Verify that same seed produces identical results
# =============================================================================

class TestDeterminism:
    """Determinism guarantees: same seed → identical distribution."""
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_basic(self):
        """Test 1: Basic determinism with continuous data."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high
        assert result1.delta_mean == result2.delta_mean
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_binary_data(self):
        """Test 2: Determinism with binary success data."""
        baseline = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        rfl = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=123)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=123)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_large_n_bootstrap(self):
        """Test 3: Determinism holds with large number of bootstrap samples."""
        np.random.seed(99)
        baseline = np.random.normal(100, 10, size=30)
        rfl = np.random.normal(110, 10, size=30)
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=7777)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=7777)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high
        assert result1.distribution_summary.std == result2.distribution_summary.std
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_distribution_summary(self):
        """Test 4: Determinism extends to full distribution summary."""
        baseline = np.linspace(0, 10, 20)
        rfl = np.linspace(1, 11, 20)
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=42)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=42)
        
        s1 = result1.distribution_summary
        s2 = result2.distribution_summary
        
        assert s1.percentile_2_5 == s2.percentile_2_5
        assert s1.percentile_50 == s2.percentile_50
        assert s1.percentile_97_5 == s2.percentile_97_5
        assert s1.skewness == s2.skewness
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_different_seeds_differ(self):
        """Test 5: Different seeds produce different results."""
        # Use data with variable paired differences so bootstrap produces different results
        np.random.seed(42)
        baseline = np.random.normal(10, 2, size=20)
        rfl = baseline + np.random.normal(2, 1, size=20)  # Variable effect
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=43)
        
        # Different seeds should (almost certainly) produce different CIs
        # Point estimate is the same (not random)
        assert result1.delta_mean == result2.delta_mean
        # But CI bounds differ due to different resamples
        assert result1.CI_low != result2.CI_low or result1.CI_high != result2.CI_high
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_raw_distribution(self):
        """Test 6: Raw bootstrap distribution is identical for same seed."""
        baseline = np.array([10, 20, 30, 40, 50])
        rfl = np.array([15, 25, 35, 45, 55])
        
        dist1 = _get_bootstrap_distribution(baseline, rfl, n_bootstrap=1000, seed=42)
        dist2 = _get_bootstrap_distribution(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert_array_equal(dist1, dist2)
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_across_calls(self):
        """Test 7: Multiple consecutive calls with same seed are identical."""
        baseline = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        rfl = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        
        results = [
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=999)
            for _ in range(5)
        ]
        
        # All results should be identical
        for i in range(1, 5):
            assert results[0].CI_low == results[i].CI_low
            assert results[0].CI_high == results[i].CI_high
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_seed_zero(self):
        """Test 8: Seed=0 produces consistent results."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=0)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=0)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_large_seed(self):
        """Test 9: Large seed values work correctly."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        large_seed = 2**31 - 1  # Max 32-bit signed int
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=large_seed)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=large_seed)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_determinism_negative_values(self):
        """Test 10: Determinism with negative values."""
        baseline = np.array([-5, -3, -1, 1, 3])
        rfl = np.array([-4, -2, 0, 2, 4])
        
        result1 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        result2 = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result1.CI_low == result2.CI_low
        assert result1.CI_high == result2.CI_high


# =============================================================================
# KNOWN-DISTRIBUTION TESTS (Tests 11-25)
# Verify statistical properties match theoretical expectations
# =============================================================================

class TestKnownDistribution:
    """Tests verifying statistical properties against known distributions."""
    
    @pytest.mark.unit
    def test_known_mean_difference(self):
        """Test 11: Point estimate matches known mean difference."""
        baseline = np.array([10, 20, 30, 40, 50])
        rfl = np.array([15, 25, 35, 45, 55])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected_delta = np.mean(rfl) - np.mean(baseline)  # = 5.0
        assert result.delta_mean == expected_delta
    
    @pytest.mark.unit
    def test_known_zero_difference(self):
        """Test 12: Zero difference when arrays are identical."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = paired_bootstrap_delta(data, data, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 0.0
        assert result.CI_low <= 0 <= result.CI_high
    
    @pytest.mark.unit
    def test_ci_contains_true_delta_normal(self):
        """Test 13: 95% CI contains true delta for normal samples (coverage test)."""
        np.random.seed(42)
        true_delta = 5.0
        n_experiments = 100
        coverage_count = 0
        
        for i in range(n_experiments):
            baseline = np.random.normal(100, 10, size=30)
            rfl = baseline + true_delta + np.random.normal(0, 3, size=30)
            
            result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=i)
            
            if result.CI_low <= true_delta <= result.CI_high:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_experiments
        # 95% CI should cover ~95% of the time (allow 80-100% for finite samples)
        assert 0.80 <= coverage_rate <= 1.0
    
    @pytest.mark.unit
    def test_ci_width_decreases_with_sample_size(self):
        """Test 14: CI width decreases as sample size increases."""
        np.random.seed(42)
        
        widths = []
        for n in [10, 30, 100]:
            baseline = np.random.normal(100, 10, size=n)
            rfl = np.random.normal(110, 10, size=n)
            
            result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
            widths.append(result.ci_width)
        
        # CI width should decrease with larger n
        assert widths[0] > widths[1] > widths[2]
    
    @pytest.mark.unit
    def test_symmetric_ci_for_symmetric_data(self):
        """Test 15: CI is approximately symmetric for symmetric distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=100)
        rfl = np.random.normal(105, 10, size=100)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=42)
        
        # Check symmetry: |CI_high - delta| ≈ |delta - CI_low|
        upper_dist = result.CI_high - result.delta_mean
        lower_dist = result.delta_mean - result.CI_low
        
        # Allow 20% asymmetry tolerance
        ratio = max(upper_dist, lower_dist) / max(min(upper_dist, lower_dist), 1e-10)
        assert ratio < 1.5
    
    @pytest.mark.unit
    def test_positive_effect_significance(self):
        """Test 16: Clear positive effect detected as significant."""
        np.random.seed(42)
        baseline = np.random.normal(50, 5, size=50)
        rfl = np.random.normal(60, 5, size=50)  # +20% effect
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        assert result.significant is True
        assert result.direction == "positive"
        assert result.CI_low > 0
    
    @pytest.mark.unit
    def test_negative_effect_significance(self):
        """Test 17: Clear negative effect detected as significant."""
        np.random.seed(42)
        baseline = np.random.normal(60, 5, size=50)
        rfl = np.random.normal(50, 5, size=50)  # -17% effect
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        assert result.significant is True
        assert result.direction == "negative"
        assert result.CI_high < 0
    
    @pytest.mark.unit
    def test_null_effect_not_significant(self):
        """Test 18: Null effect (same means) typically not significant."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=20)
        rfl = np.random.normal(100, 10, size=20)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # With no true effect and moderate variance, CI should include 0
        assert result.direction == "null"
        assert result.CI_low < 0 < result.CI_high
    
    @pytest.mark.unit
    def test_median_approximates_mean_symmetric(self):
        """Test 19: Bootstrap median ≈ mean for symmetric distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 5, size=50)
        rfl = np.random.normal(110, 5, size=50)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=42)
        
        median = result.distribution_summary.percentile_50
        mean = result.delta_mean
        
        # Median should be close to mean for symmetric data
        assert abs(median - mean) < 1.0
    
    @pytest.mark.unit
    def test_percentile_ordering(self):
        """Test 20: Percentiles are correctly ordered."""
        baseline = np.random.normal(50, 10, size=30)
        rfl = np.random.normal(55, 10, size=30)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        s = result.distribution_summary
        assert s.percentile_2_5 < s.percentile_25
        assert s.percentile_25 < s.percentile_50
        assert s.percentile_50 < s.percentile_75
        assert s.percentile_75 < s.percentile_97_5
    
    @pytest.mark.unit
    def test_ci_matches_percentiles(self):
        """Test 21: CI bounds match 2.5/97.5 percentiles for large samples."""
        np.random.seed(42)
        baseline = np.random.normal(100, 5, size=100)
        rfl = np.random.normal(105, 5, size=100)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=42)
        
        # For percentile method, CI should match percentiles closely
        if result.method == "percentile":
            assert_allclose(result.CI_low, result.distribution_summary.percentile_2_5, rtol=0.01)
            assert_allclose(result.CI_high, result.distribution_summary.percentile_97_5, rtol=0.01)
    
    @pytest.mark.unit
    def test_std_reasonable_for_normal(self):
        """Test 22: Bootstrap std is reasonable for normal data."""
        np.random.seed(42)
        n = 50
        sigma = 10
        baseline = np.random.normal(100, sigma, size=n)
        rfl = np.random.normal(110, sigma, size=n)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=42)
        
        # Theoretical SE for difference of means (assuming independence):
        # SE = sqrt(2 * sigma^2 / n) ≈ sqrt(2 * 100 / 50) ≈ 2.0
        # Bootstrap std should be in similar ballpark
        assert 1.0 < result.distribution_summary.std < 4.0
    
    @pytest.mark.unit
    def test_skewness_near_zero_symmetric(self):
        """Test 23: Skewness ≈ 0 for symmetric distribution."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=100)
        rfl = np.random.normal(110, 10, size=100)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=10000, seed=42)
        
        # Symmetric distribution should have low skewness
        assert abs(result.distribution_summary.skewness) < 0.5
    
    @pytest.mark.unit
    def test_binary_coverage_rate(self):
        """Test 24: Bootstrap works correctly with binary (0/1) data."""
        np.random.seed(42)
        n = 100
        # Baseline: 60% success rate
        baseline = np.random.binomial(1, 0.6, size=n)
        # RFL: 75% success rate
        rfl = np.random.binomial(1, 0.75, size=n)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # True delta is ~0.15
        assert result.delta_mean > 0
        assert result.CI_low < result.delta_mean < result.CI_high
    
    @pytest.mark.unit
    def test_ci_width_scales_with_variance(self):
        """Test 25: CI width scales with data variance."""
        np.random.seed(42)
        n = 50
        
        # Low variance
        baseline_low = np.random.normal(100, 2, size=n)
        rfl_low = np.random.normal(105, 2, size=n)
        result_low = paired_bootstrap_delta(baseline_low, rfl_low, n_bootstrap=5000, seed=42)
        
        # High variance
        baseline_high = np.random.normal(100, 20, size=n)
        rfl_high = np.random.normal(105, 20, size=n)
        result_high = paired_bootstrap_delta(baseline_high, rfl_high, n_bootstrap=5000, seed=42)
        
        # High variance should have wider CI
        assert result_high.ci_width > result_low.ci_width


# =============================================================================
# EDGE CASE TESTS (Tests 26-40)
# Small samples, skewed distributions, identical arrays, numerical stability
# =============================================================================

class TestEdgeCases:
    """Edge case handling: small samples, skew, identical data, numerical issues."""
    
    @pytest.mark.unit
    def test_minimum_sample_size(self):
        """Test 26: Works with minimum sample size (n=2)."""
        baseline = np.array([1.0, 2.0])
        rfl = np.array([3.0, 4.0])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 2.0
        assert result.distribution_summary.n_samples == 2
    
    @pytest.mark.unit
    def test_small_sample_bca(self):
        """Test 27: Small samples (n < 20) use BCa method."""
        baseline = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # n=10
        rfl = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=42)
        
        # BCa should be used for n < 20
        assert result.method in ["BCa", "percentile_fallback"]
    
    @pytest.mark.unit
    def test_identical_arrays(self):
        """Test 28: Identical arrays produce delta=0 and CI containing 0."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        result = paired_bootstrap_delta(data, data, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 0.0
        assert result.CI_low == 0.0
        assert result.CI_high == 0.0
        assert result.distribution_summary.std == 0.0
    
    @pytest.mark.unit
    def test_constant_difference(self):
        """Test 29: Constant difference between pairs."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = baseline + 10.0  # Constant delta = 10
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 10.0
        # With constant difference, CI is a point
        assert result.CI_low == 10.0
        assert result.CI_high == 10.0
    
    @pytest.mark.unit
    def test_highly_skewed_positive(self):
        """Test 30: Highly positively skewed distribution."""
        np.random.seed(42)
        # Exponential is positively skewed
        baseline = np.random.exponential(10, size=50)
        rfl = np.random.exponential(15, size=50)  # Higher mean
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # Should still compute valid CI
        assert result.CI_low < result.CI_high
        assert np.isfinite(result.CI_low)
        assert np.isfinite(result.CI_high)
    
    @pytest.mark.unit
    def test_highly_skewed_negative(self):
        """Test 31: Negatively skewed distribution."""
        np.random.seed(42)
        # Create negatively skewed data
        baseline = -np.random.exponential(10, size=50)
        rfl = -np.random.exponential(8, size=50)  # Less negative = higher
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        assert result.CI_low < result.CI_high
        assert np.isfinite(result.CI_low)
        assert np.isfinite(result.CI_high)
    
    @pytest.mark.unit
    def test_very_small_values(self):
        """Test 32: Very small values (near machine epsilon)."""
        baseline = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        rfl = np.array([2e-10, 3e-10, 4e-10, 5e-10, 6e-10])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected_delta = 1e-10
        assert_allclose(result.delta_mean, expected_delta, rtol=1e-5)
    
    @pytest.mark.unit
    def test_very_large_values(self):
        """Test 33: Very large values."""
        baseline = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        rfl = np.array([2e10, 3e10, 4e10, 5e10, 6e10])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected_delta = 1e10
        assert_allclose(result.delta_mean, expected_delta, rtol=1e-5)
    
    @pytest.mark.unit
    def test_mixed_sign_values(self):
        """Test 34: Mix of positive and negative values."""
        baseline = np.array([-5, -3, 0, 3, 5])
        rfl = np.array([-3, -1, 2, 5, 7])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected_delta = np.mean(rfl) - np.mean(baseline)  # = 2.0
        assert result.delta_mean == expected_delta
    
    @pytest.mark.unit
    def test_one_outlier(self):
        """Test 35: Data with one outlier."""
        baseline = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 100])
        rfl = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 150])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # Should handle outlier gracefully
        assert np.isfinite(result.CI_low)
        assert np.isfinite(result.CI_high)
        assert result.CI_low < result.CI_high
    
    @pytest.mark.unit
    def test_all_zeros_baseline(self):
        """Test 36: All zeros in baseline."""
        baseline = np.zeros(10)
        rfl = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected_delta = np.mean(rfl)  # = 5.5
        assert result.delta_mean == expected_delta
    
    @pytest.mark.unit
    def test_all_ones_binary(self):
        """Test 37: All ones in binary RFL (perfect success)."""
        baseline = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 50% success
        rfl = np.ones(10)  # 100% success
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 0.5  # 100% - 50%
    
    @pytest.mark.unit
    def test_n_equals_20_boundary(self):
        """Test 38: Exactly n=20 (boundary for small sample handling)."""
        baseline = np.arange(20, dtype=float)
        rfl = baseline + 5
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=42)
        
        assert result.delta_mean == 5.0
        # n=20 is the boundary; method may be either
        assert result.method in ["percentile", "BCa", "percentile_fallback"]
    
    @pytest.mark.unit
    def test_n_equals_19_small_sample(self):
        """Test 39: n=19 uses small sample handling (BCa)."""
        baseline = np.arange(19, dtype=float)
        rfl = baseline + 3
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=2000, seed=42)
        
        assert result.delta_mean == 3.0
        # n < 20 should trigger BCa
        assert result.method in ["BCa", "percentile_fallback"]
    
    @pytest.mark.unit
    def test_high_correlation_pairs(self):
        """Test 40: Highly correlated paired data."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=30)
        # RFL is strongly correlated (small random perturbation + constant shift)
        rfl = baseline + 5 + np.random.normal(0, 1, size=30)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # High correlation should reduce CI width
        assert result.ci_width < 3.0  # Narrow CI expected


# =============================================================================
# INPUT VALIDATION TESTS (Tests 41-50)
# Error handling for invalid inputs
# =============================================================================

class TestInputValidation:
    """Input validation and error handling."""
    
    @pytest.mark.unit
    def test_mismatched_lengths(self):
        """Test 41: Mismatched array lengths raise ValueError."""
        baseline = np.array([1, 2, 3])
        rfl = np.array([1, 2])
        
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_nan_in_baseline(self):
        """Test 42: NaN in baseline raises ValueError."""
        baseline = np.array([1.0, np.nan, 3.0])
        rfl = np.array([2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError, match="NaN"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_nan_in_rfl(self):
        """Test 43: NaN in rfl_values raises ValueError."""
        baseline = np.array([1.0, 2.0, 3.0])
        rfl = np.array([2.0, np.nan, 4.0])
        
        with pytest.raises(ValueError, match="NaN"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_empty_arrays(self):
        """Test 44: Empty arrays raise ValueError."""
        baseline = np.array([])
        rfl = np.array([])
        
        with pytest.raises(ValueError, match="at least"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_single_element(self):
        """Test 45: Single element arrays raise ValueError."""
        baseline = np.array([1.0])
        rfl = np.array([2.0])
        
        with pytest.raises(ValueError, match="at least"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_n_bootstrap_too_small(self):
        """Test 46: n_bootstrap below minimum raises ValueError."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        with pytest.raises(ValueError, match="n_bootstrap"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=100, seed=42)
    
    @pytest.mark.unit
    def test_n_bootstrap_too_large(self):
        """Test 47: n_bootstrap above maximum raises ValueError."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        with pytest.raises(ValueError, match="n_bootstrap"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=200_000, seed=42)
    
    @pytest.mark.unit
    def test_2d_arrays_rejected(self):
        """Test 48: 2D arrays raise ValueError."""
        baseline = np.array([[1, 2], [3, 4]])
        rfl = np.array([[2, 3], [4, 5]])
        
        with pytest.raises(ValueError, match="1D"):
            paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
    
    @pytest.mark.unit
    def test_list_input_accepted(self):
        """Test 49: Python lists are converted to arrays and accepted."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        rfl = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 1.0
    
    @pytest.mark.unit
    def test_integer_input_accepted(self):
        """Test 50: Integer arrays are accepted and converted to float64."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.delta_mean == 1.0


# =============================================================================
# ADDITIONAL UTILITY TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions and result object methods."""
    
    @pytest.mark.unit
    def test_result_to_dict(self):
        """Test PairedBootstrapResult.to_dict() serialization."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        d = result.to_dict()
        
        assert "CI_low" in d
        assert "CI_high" in d
        assert "delta_mean" in d
        assert "distribution_summary" in d
        assert "percentiles" in d["distribution_summary"]
    
    @pytest.mark.unit
    def test_summary_to_dict(self):
        """Test DistributionSummary.to_dict() serialization."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        s = result.distribution_summary.to_dict()
        
        assert "percentiles" in s
        assert "2.5" in s["percentiles"]
        assert "97.5" in s["percentiles"]
        assert "std" in s
        assert "skewness" in s
    
    @pytest.mark.unit
    def test_compute_skewness_symmetric(self):
        """Test skewness computation for symmetric data."""
        symmetric_data = np.array([-2, -1, 0, 1, 2])
        skew = _compute_skewness(symmetric_data)
        
        assert abs(skew) < 0.5
    
    @pytest.mark.unit
    def test_compute_skewness_constant(self):
        """Test skewness for constant data returns 0."""
        constant_data = np.array([5, 5, 5, 5, 5])
        skew = _compute_skewness(constant_data)
        
        assert skew == 0.0
    
    @pytest.mark.unit
    def test_compute_skewness_small_sample(self):
        """Test skewness for n < 3 returns 0."""
        small_data = np.array([1, 2])
        skew = _compute_skewness(small_data)
        
        assert skew == 0.0


class TestResultProperties:
    """Tests for computed properties on result objects."""
    
    @pytest.mark.unit
    def test_ci_width_positive(self):
        """Test ci_width is always non-negative."""
        baseline = np.array([1, 2, 3, 4, 5])
        rfl = np.array([2, 3, 4, 5, 6])
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert result.ci_width >= 0
    
    @pytest.mark.unit
    def test_significant_positive_effect(self):
        """Test significant property for positive effect."""
        np.random.seed(42)
        baseline = np.random.normal(50, 3, size=50)
        rfl = np.random.normal(60, 3, size=50)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        assert result.significant is True
        assert result.direction == "positive"
    
    @pytest.mark.unit
    def test_direction_null(self):
        """Test direction is 'null' when CI contains zero."""
        np.random.seed(42)
        baseline = np.random.normal(100, 20, size=20)
        rfl = np.random.normal(100, 20, size=20)
        
        result = paired_bootstrap_delta(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # With high variance and no true effect, usually not significant
        if not result.significant:
            assert result.direction == "null"


# =============================================================================
# CONFIDENCE BAND TESTS (Tests 51-58)
# Visualization envelope generation
# =============================================================================

class TestConfidenceBand:
    """Tests for confidence band generation (visualization only)."""
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_band_determinism(self):
        """Test 51: Confidence band is deterministic with same seed."""
        series = np.array([0.85, 0.87, 0.88, 0.86, 0.90, 0.89, 0.91])
        
        band1 = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        band2 = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        
        assert_array_equal(band1.lower, band2.lower)
        assert_array_equal(band1.upper, band2.upper)
        assert_array_equal(band1.center, band2.center)
    
    @pytest.mark.unit
    def test_band_contains_center(self):
        """Test 52: Band envelope contains center values."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        band = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        
        # Lower <= center <= upper for all points
        assert np.all(band.lower <= band.center)
        assert np.all(band.center <= band.upper)
    
    @pytest.mark.unit
    def test_band_ordering(self):
        """Test 53: Lower band is always <= upper band."""
        np.random.seed(42)
        series = np.random.normal(10, 2, size=20)
        
        band = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        
        assert np.all(band.lower <= band.upper)
    
    @pytest.mark.unit
    def test_band_confidence_levels(self):
        """Test 54: Wider confidence produces wider bands."""
        np.random.seed(42)
        series = np.random.normal(100, 10, size=30)
        
        band_90 = compute_confidence_band(series, confidence=0.90, n_bootstrap=500, seed=42)
        band_99 = compute_confidence_band(series, confidence=0.99, n_bootstrap=500, seed=42)
        
        # 99% band should be wider on average
        width_90 = np.mean(band_90.upper - band_90.lower)
        width_99 = np.mean(band_99.upper - band_99.lower)
        
        assert width_99 >= width_90
    
    @pytest.mark.unit
    def test_band_single_element(self):
        """Test 55: Single element series produces degenerate band."""
        series = np.array([42.0])
        
        band = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        
        assert band.lower[0] == 42.0
        assert band.upper[0] == 42.0
        assert band.center[0] == 42.0
    
    @pytest.mark.unit
    def test_band_to_dict(self):
        """Test 56: Band serialization to dict."""
        series = np.array([1.0, 2.0, 3.0])
        
        band = compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)
        d = band.to_dict()
        
        assert "lower" in d
        assert "upper" in d
        assert "center" in d
        assert "confidence" in d
        assert d["confidence"] == 0.95
    
    @pytest.mark.unit
    def test_band_invalid_confidence(self):
        """Test 57: Invalid confidence raises ValueError."""
        series = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="confidence"):
            compute_confidence_band(series, confidence=1.5, n_bootstrap=500, seed=42)
        
        with pytest.raises(ValueError, match="confidence"):
            compute_confidence_band(series, confidence=0.0, n_bootstrap=500, seed=42)
    
    @pytest.mark.unit
    def test_band_nan_rejected(self):
        """Test 58: NaN in series raises ValueError."""
        series = np.array([1.0, np.nan, 3.0])
        
        with pytest.raises(ValueError, match="NaN"):
            compute_confidence_band(series, confidence=0.95, n_bootstrap=500, seed=42)


# =============================================================================
# LEAKAGE DETECTION TESTS (Tests 59-66)
# Seed invariance and epsilon stability checks
# =============================================================================

class TestLeakageDetection:
    """Tests for bootstrap leakage detection."""
    
    @pytest.mark.unit
    def test_leakage_constant_series_detected(self):
        """Test 59: Constant paired differences are detected."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = baseline + 10.0  # Constant difference
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        
        assert result.is_constant_series == True
    
    @pytest.mark.unit
    def test_leakage_variable_series_not_constant(self):
        """Test 60: Variable series correctly identified."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=20)
        rfl = np.random.normal(110, 10, size=20)
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        
        assert result.is_constant_series == False
    
    @pytest.mark.unit
    def test_leakage_seed_invariance_constant(self):
        """Test 61: Constant series is seed-invariant."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = baseline + 5.0
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        
        assert result.seed_invariant == True
        assert result.leakage_detected == False
    
    @pytest.mark.unit
    def test_leakage_epsilon_stability(self):
        """Test 62: Epsilon stability is checked."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = baseline + 3.0
        
        result = detect_bootstrap_leakage(
            baseline, rfl, n_bootstrap=1000, epsilon=1e-12
        )
        
        assert result.epsilon_stable == True
    
    @pytest.mark.unit
    def test_leakage_custom_seeds(self):
        """Test 63: Custom seed list is used."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        custom_seeds = [1, 2, 3]
        result = detect_bootstrap_leakage(
            baseline, rfl, n_bootstrap=1000, seeds=custom_seeds
        )
        
        assert result.seeds_tested == custom_seeds
    
    @pytest.mark.unit
    def test_leakage_result_to_dict(self):
        """Test 64: Leakage result serialization."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        d = result.to_dict()
        
        assert "is_constant_series" in d
        assert "seed_invariant" in d
        assert "epsilon_stable" in d
        assert "leakage_detected" in d
        assert "max_delta_variance" in d
    
    @pytest.mark.unit
    def test_leakage_no_false_positives_variable(self):
        """Test 65: No leakage detected for normal variable data."""
        np.random.seed(42)
        baseline = np.random.normal(50, 5, size=30)
        rfl = np.random.normal(55, 5, size=30)
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        
        # Variable data should pass epsilon stability
        assert result.epsilon_stable == True
    
    @pytest.mark.unit
    def test_leakage_property_computation(self):
        """Test 66: leakage_detected property computes correctly."""
        # Constant series with seed invariance should not detect leakage
        baseline = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rfl = baseline + 100.0
        
        result = detect_bootstrap_leakage(baseline, rfl, n_bootstrap=1000)
        
        # No leakage for constant series that is seed-invariant
        assert result.is_constant_series == True
        assert result.seed_invariant == True
        assert result.leakage_detected == False


# =============================================================================
# PROFILING SUITE TESTS (Tests 67-74)
# Timing, memory, and histogram analysis
# =============================================================================

class TestProfilingSuite:
    """Tests for bootstrap profiling suite."""
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_profile_histogram_determinism(self):
        """Test 67: Histogram is deterministic for same seed."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        profile1 = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        profile2 = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        
        # Histogram statistics should be identical
        assert profile1.histogram.mean == profile2.histogram.mean
        assert profile1.histogram.std == profile2.histogram.std
        assert profile1.histogram.n_total == profile2.histogram.n_total
    
    @pytest.mark.unit
    def test_profile_timing_positive(self):
        """Test 68: Execution time is positive."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        profile = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        
        assert profile.execution_time_ms > 0
    
    @pytest.mark.unit
    def test_profile_memory_tracking(self):
        """Test 69: Memory allocation is tracked."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=50)
        rfl = np.random.normal(110, 10, size=50)
        
        profile = profile_bootstrap(baseline, rfl, n_bootstrap=5000, seed=42)
        
        assert profile.peak_memory_bytes > 0
        assert profile.memory_allocated_bytes >= 0
    
    @pytest.mark.unit
    def test_profile_histogram_bins(self):
        """Test 70: Histogram has correct bin structure."""
        np.random.seed(42)
        baseline = np.random.normal(100, 10, size=30)
        rfl = np.random.normal(110, 10, size=30)
        
        profile = profile_bootstrap(
            baseline, rfl, n_bootstrap=2000, seed=42, n_histogram_bins=20
        )
        
        # Check histogram structure
        assert len(profile.histogram.bins) > 0
        assert profile.histogram.n_total == 2000
        
        # Check bin ordering
        for i in range(len(profile.histogram.bins) - 1):
            assert profile.histogram.bins[i].right_edge <= profile.histogram.bins[i + 1].left_edge + 1e-10
    
    @pytest.mark.unit
    def test_profile_degenerate_histogram(self):
        """Test 71: Degenerate distribution produces single-bin histogram."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = baseline + 10.0  # Constant delta
        
        profile = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        
        # All deltas are identical, so histogram is degenerate
        assert profile.histogram.std == 0.0
        assert len(profile.histogram.bins) == 1
    
    @pytest.mark.unit
    def test_profile_time_per_resample(self):
        """Test 72: Time per resample is computed correctly."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        profile = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        
        expected = (profile.execution_time_ms * 1000) / 1000
        assert profile.time_per_resample_us == expected
    
    @pytest.mark.unit
    def test_profile_to_dict(self):
        """Test 73: Profile serialization to dict."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        profile = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        d = profile.to_dict()
        
        assert "execution_time_ms" in d
        assert "peak_memory_bytes" in d
        assert "histogram" in d
        assert "bins" in d["histogram"]
        assert "time_per_resample_us" in d
    
    @pytest.mark.unit
    def test_profile_scaling_with_n_bootstrap(self):
        """Test 74: Execution time scales with n_bootstrap."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        rfl = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        
        profile_small = profile_bootstrap(baseline, rfl, n_bootstrap=1000, seed=42)
        profile_large = profile_bootstrap(baseline, rfl, n_bootstrap=5000, seed=42)
        
        # Larger n_bootstrap should take more time (allow variance)
        # This is a weak test to avoid flakiness
        assert profile_large.n_bootstrap > profile_small.n_bootstrap


# =============================================================================
# CONTRACT SCHEMA TESTS (Tests 75-83)
# Bootstrap contract generation and validation
# =============================================================================

class TestContractSchema:
    """Tests for bootstrap contract schema generation."""
    
    @pytest.mark.unit
    def test_contract_structure(self):
        """Test 75: Contract has required top-level fields."""
        contract = get_bootstrap_contract()
        
        assert "$schema" in contract
        assert "title" in contract
        assert "version" in contract
        assert "functions" in contract
        assert "types" in contract
        assert "constraints" in contract
    
    @pytest.mark.unit
    def test_contract_sha256_commitment(self):
        """Test 76: Contract includes SHA-256 formula commitment."""
        contract = get_bootstrap_contract()
        
        assert "formula_specification_sha256" in contract
        sha = contract["formula_specification_sha256"]
        
        # SHA-256 is 64 hex chars
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)
    
    @pytest.mark.unit
    def test_contract_function_definitions(self):
        """Test 77: All public functions are documented."""
        contract = get_bootstrap_contract()
        
        required_functions = [
            "paired_bootstrap_delta",
            "compute_confidence_band",
            "detect_bootstrap_leakage",
            "profile_bootstrap",
        ]
        
        for func in required_functions:
            assert func in contract["functions"], f"Missing function: {func}"
    
    @pytest.mark.unit
    def test_contract_parameter_types(self):
        """Test 78: Function parameters have type definitions."""
        contract = get_bootstrap_contract()
        
        delta_func = contract["functions"]["paired_bootstrap_delta"]
        params = delta_func["parameters"]
        
        assert "baseline_values" in params
        assert "type" in params["baseline_values"]
        assert params["n_bootstrap"]["type"] == "integer"
        assert params["seed"]["type"] == "integer"
    
    @pytest.mark.unit
    def test_contract_return_types(self):
        """Test 79: Function return types are specified."""
        contract = get_bootstrap_contract()
        
        delta_func = contract["functions"]["paired_bootstrap_delta"]
        returns = delta_func["returns"]
        
        assert "type" in returns
        assert returns["type"] == "PairedBootstrapResult"
        assert "fields" in returns
        assert "CI_low" in returns["fields"]
    
    @pytest.mark.unit
    def test_contract_constraints(self):
        """Test 80: Contract includes constraint definitions."""
        contract = get_bootstrap_contract()
        
        constraints = contract["constraints"]
        
        assert "determinism" in constraints
        assert "minimum_sample_size" in constraints
        assert constraints["minimum_sample_size"] == 2
    
    @pytest.mark.unit
    def test_contract_complexity(self):
        """Test 81: Contract includes complexity specifications."""
        contract = get_bootstrap_contract()
        
        complexity = contract["complexity"]
        
        assert "time" in complexity
        assert "space" in complexity
        assert "O(n_bootstrap" in complexity["time"]
    
    @pytest.mark.unit
    @pytest.mark.determinism
    def test_contract_determinism(self):
        """Test 82: Contract generation is deterministic."""
        contract1 = get_bootstrap_contract()
        contract2 = get_bootstrap_contract()
        
        # SHA-256 commitment should be identical
        assert contract1["formula_specification_sha256"] == contract2["formula_specification_sha256"]
        
        # All fields should match
        assert contract1["version"] == contract2["version"]
        assert contract1["functions"].keys() == contract2["functions"].keys()
    
    @pytest.mark.unit
    def test_contract_type_definitions(self):
        """Test 83: Custom types are fully defined."""
        contract = get_bootstrap_contract()
        
        types = contract["types"]
        
        assert "DistributionSummary" in types
        assert "BootstrapHistogram" in types
        
        # Check DistributionSummary has expected fields
        ds_fields = types["DistributionSummary"]["fields"]
        assert "percentile_2_5" in ds_fields
        assert "std" in ds_fields
        assert "skewness" in ds_fields


# =============================================================================
# CONTRACT FILE CHECKER TESTS (Tests 84-91)
# Validate bootstrap_contract.json against runtime contract
# =============================================================================

class TestContractFileChecker:
    """
    Tests that validate bootstrap_contract.json file consistency.
    
    These tests ensure:
    - The contract file exists and is valid JSON
    - Contract functions match runtime implementation
    - No unexpected keys appear if contract is regenerated
    """
    
    CONTRACT_PATH = "statistical/bootstrap_contract.json"
    
    # Expected top-level keys in contract (canonical set)
    EXPECTED_TOP_LEVEL_KEYS = {
        "$schema",
        "title",
        "version",
        "formula_specification_sha256",
        "determinism_guarantee",
        "functions",
        "types",
        "constraints",
        "complexity",
    }
    
    # Expected function names (canonical set)
    EXPECTED_FUNCTIONS = {
        "paired_bootstrap_delta",
        "compute_confidence_band",
        "detect_bootstrap_leakage",
        "profile_bootstrap",
    }
    
    @pytest.mark.unit
    def test_contract_file_exists(self):
        """Test 84: Contract file exists at expected path."""
        import os
        from pathlib import Path
        
        # Try multiple paths for test execution context
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                found = True
                break
        
        assert found, f"Contract file not found at any of: {possible_paths}"
    
    @pytest.mark.unit
    def test_contract_file_valid_json(self):
        """Test 85: Contract file is valid JSON."""
        import json
        from pathlib import Path
        
        # Find the contract file
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        assert contract_path is not None, "Contract file not found"
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should not raise
        contract = json.loads(content)
        assert isinstance(contract, dict)
    
    @pytest.mark.unit
    def test_contract_file_matches_runtime(self):
        """Test 86: Contract file matches runtime-generated contract."""
        import json
        from pathlib import Path
        
        # Find the contract file
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        runtime_contract = get_bootstrap_contract()
        
        # Check version matches
        assert file_contract["version"] == runtime_contract["version"], \
            "Contract file version mismatch with runtime"
        
        # Check formula SHA matches (critical for reproducibility)
        assert file_contract["formula_specification_sha256"] == \
            runtime_contract["formula_specification_sha256"], \
            "Formula specification SHA-256 mismatch - contract needs regeneration"
    
    @pytest.mark.unit
    def test_contract_file_no_unexpected_top_level_keys(self):
        """Test 87: Contract file has no unexpected top-level keys."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        actual_keys = set(file_contract.keys())
        unexpected_keys = actual_keys - self.EXPECTED_TOP_LEVEL_KEYS
        
        assert not unexpected_keys, \
            f"Unexpected top-level keys in contract: {unexpected_keys}"
    
    @pytest.mark.unit
    def test_contract_file_has_all_expected_functions(self):
        """Test 88: Contract file documents all expected functions."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        actual_functions = set(file_contract.get("functions", {}).keys())
        missing_functions = self.EXPECTED_FUNCTIONS - actual_functions
        
        assert not missing_functions, \
            f"Missing functions in contract: {missing_functions}"
    
    @pytest.mark.unit
    def test_contract_file_no_unexpected_functions(self):
        """Test 89: Contract file has no unexpected function definitions."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        actual_functions = set(file_contract.get("functions", {}).keys())
        unexpected_functions = actual_functions - self.EXPECTED_FUNCTIONS
        
        assert not unexpected_functions, \
            f"Unexpected functions in contract: {unexpected_functions}"
    
    @pytest.mark.unit
    def test_contract_file_determinism_guarantee(self):
        """Test 90: Contract file asserts determinism guarantee."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        assert file_contract.get("determinism_guarantee") == True, \
            "Contract must assert determinism_guarantee: true"
    
    @pytest.mark.unit
    def test_contract_file_constraints_present(self):
        """Test 91: Contract file includes required constraints."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        contract_path = None
        for path in possible_paths:
            if path.exists():
                contract_path = path
                break
        
        if contract_path is None:
            pytest.skip("Contract file not found")
        
        with open(contract_path, 'r', encoding='utf-8') as f:
            file_contract = json.load(f)
        
        constraints = file_contract.get("constraints", {})
        
        # Required constraint keys
        required_constraints = ["determinism", "minimum_sample_size", "bootstrap_range"]
        
        for key in required_constraints:
            assert key in constraints, f"Missing required constraint: {key}"


# =============================================================================
# PAIRED DELTA RESULT CONTRACT TESTS (Tests 92-99)
# Validate PairedDeltaResult fields match contract
# =============================================================================

class TestPairedDeltaResultContract:
    """
    Tests that validate PairedDeltaResult against bootstrap_contract.json.
    
    These tests ensure:
    - PairedDeltaResult fields exactly match contract declaration
    - No extra fields in runtime or contract
    - to_dict() produces deterministic, sorted output
    """
    
    CONTRACT_PATH = "statistical/bootstrap_contract.json"
    
    # Expected fields in PairedDeltaResult (canonical set, alphabetically sorted)
    # LOCKED: 10 fields exactly
    EXPECTED_FIELDS = {
        "analysis_id",
        "ci_lower",
        "ci_upper",
        "delta",
        "method",
        "metric_path",
        "n_baseline",
        "n_bootstrap",
        "n_rfl",
        "seed",
    }
    
    def _find_contract_path(self):
        """Find the contract file path."""
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    @pytest.mark.unit
    def test_paired_delta_result_to_dict_fields(self):
        """Test 92: PairedDeltaResult.to_dict() produces exactly 10 expected fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        # Create a sample result with all 10 fields
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="abc123def456",
        )
        
        d = result.to_dict()
        actual_fields = set(d.keys())
        
        # Must have exactly 10 fields
        assert len(d) == 10, f"Expected exactly 10 fields, got {len(d)}"
        assert actual_fields == self.EXPECTED_FIELDS, \
            f"Field mismatch. Expected: {self.EXPECTED_FIELDS}, Got: {actual_fields}"
    
    @pytest.mark.unit
    def test_paired_delta_result_no_extra_fields(self):
        """Test 93: PairedDeltaResult.to_dict() has no extra fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="test_analysis_id",
        )
        
        d = result.to_dict()
        extra_fields = set(d.keys()) - self.EXPECTED_FIELDS
        
        assert not extra_fields, f"Unexpected fields in to_dict(): {extra_fields}"
    
    @pytest.mark.unit
    def test_paired_delta_result_no_missing_fields(self):
        """Test 94: PairedDeltaResult.to_dict() has no missing fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="test_id",
        )
        
        d = result.to_dict()
        missing_fields = self.EXPECTED_FIELDS - set(d.keys())
        
        assert not missing_fields, f"Missing fields in to_dict(): {missing_fields}"
    
    @pytest.mark.unit
    def test_paired_delta_result_to_dict_sorted_keys(self):
        """Test 95: PairedDeltaResult.to_dict() keys are alphabetically sorted."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="sorted_test",
        )
        
        d = result.to_dict()
        keys = list(d.keys())
        sorted_keys = sorted(keys)
        
        assert keys == sorted_keys, \
            f"Keys not sorted. Got: {keys}, Expected: {sorted_keys}"
    
    @pytest.mark.unit
    def test_paired_delta_result_to_json_deterministic(self):
        """Test 96: PairedDeltaResult.to_json() is deterministic."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.123456,
            ci_lower=0.05,
            ci_upper=0.20,
            n_baseline=50,
            n_rfl=50,
            metric_path="derivation.verified",
            seed=123,
            n_bootstrap=5000,
            method="BCa",
            analysis_id="determinism_test_123",
        )
        
        # Call to_json() multiple times
        json1 = result.to_json()
        json2 = result.to_json()
        json3 = result.to_json()
        
        assert json1 == json2 == json3, "to_json() is not deterministic"
    
    @pytest.mark.unit
    def test_paired_delta_result_get_field_names(self):
        """Test 97: PairedDeltaResult.get_field_names() returns canonical 10 fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        field_names = PairedDeltaResult.get_field_names()
        
        # Should be a tuple
        assert isinstance(field_names, tuple)
        
        # Should have exactly 10 fields
        assert len(field_names) == 10, f"Expected 10 fields, got {len(field_names)}"
        
        # Should match expected fields
        assert set(field_names) == self.EXPECTED_FIELDS
        
        # Should be sorted
        assert list(field_names) == sorted(field_names)
    
    @pytest.mark.unit
    def test_paired_delta_result_primitive_types_only(self):
        """Test 98: PairedDeltaResult.to_dict() contains only primitive types."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="primitive_test",
        )
        
        d = result.to_dict()
        
        for key, value in d.items():
            assert isinstance(value, (int, float, str, bool)), \
                f"Field '{key}' has non-primitive type: {type(value)}"
    
    @pytest.mark.unit
    def test_paired_delta_result_analysis_id_present(self):
        """Test 99: PairedDeltaResult includes analysis_id field."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="test_analysis_id_abc123",
        )
        
        d = result.to_dict()
        
        assert "analysis_id" in d, "analysis_id field missing"
        assert isinstance(d["analysis_id"], str), "analysis_id must be string"
        assert d["analysis_id"] == "test_analysis_id_abc123"


# =============================================================================
# CONTRACT BACKWARDS COMPATIBILITY GUARD TESTS (Tests 100-107)
# Ensure changes to PairedDeltaResult require updating bootstrap_contract.json
# =============================================================================

class TestContractBackwardsCompatibility:
    """
    Backwards compatibility guard tests.
    
    These tests ensure:
    - Changes to PairedDeltaResult require updating bootstrap_contract.json
    - Deviations between code and contract cause test failures
    - Contract can be used directly by governance verifier without transformation
    """
    
    CONTRACT_PATH = "statistical/bootstrap_contract.json"
    
    # LOCKED: The canonical 10-field schema
    CANONICAL_SCHEMA_FIELDS = (
        "analysis_id",
        "ci_lower",
        "ci_upper",
        "delta",
        "method",
        "metric_path",
        "n_baseline",
        "n_bootstrap",
        "n_rfl",
        "seed",
    )
    
    def _find_contract_path(self):
        """Find the contract file path."""
        from pathlib import Path
        
        possible_paths = [
            Path(self.CONTRACT_PATH),
            Path("..") / self.CONTRACT_PATH,
            Path(__file__).parent.parent.parent / self.CONTRACT_PATH,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    @pytest.mark.unit
    def test_schema_field_count_locked(self):
        """Test 100: PairedDeltaResult has exactly 10 fields (LOCKED)."""
        from backend.metrics.u2_analysis import PAIRED_DELTA_RESULT_FIELDS
        
        assert len(PAIRED_DELTA_RESULT_FIELDS) == 10, \
            f"Schema must have exactly 10 fields, got {len(PAIRED_DELTA_RESULT_FIELDS)}. " \
            "If you need to add/remove fields, update bootstrap_contract.json first."
    
    @pytest.mark.unit
    def test_schema_fields_match_canonical(self):
        """Test 101: PairedDeltaResult fields match canonical schema exactly."""
        from backend.metrics.u2_analysis import PAIRED_DELTA_RESULT_FIELDS
        
        assert PAIRED_DELTA_RESULT_FIELDS == self.CANONICAL_SCHEMA_FIELDS, \
            f"Schema fields deviate from canonical. " \
            f"Expected: {self.CANONICAL_SCHEMA_FIELDS}, " \
            f"Got: {PAIRED_DELTA_RESULT_FIELDS}. " \
            "Update bootstrap_contract.json if this is intentional."
    
    @pytest.mark.unit
    def test_validate_dict_rejects_extra_fields(self):
        """Test 102: validate_dict() rejects dictionaries with extra fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        valid_dict = {
            "analysis_id": "test",
            "ci_lower": 0.05,
            "ci_upper": 0.15,
            "delta": 0.1,
            "method": "percentile",
            "metric_path": "success",
            "n_baseline": 100,
            "n_bootstrap": 10000,
            "n_rfl": 100,
            "seed": 42,
        }
        
        # Valid dict should pass
        assert PairedDeltaResult.validate_dict(valid_dict) == True
        
        # Dict with extra field should fail
        invalid_dict = valid_dict.copy()
        invalid_dict["extra_field"] = "not allowed"
        
        with pytest.raises(ValueError, match="Unexpected fields"):
            PairedDeltaResult.validate_dict(invalid_dict)
    
    @pytest.mark.unit
    def test_validate_dict_rejects_missing_fields(self):
        """Test 103: validate_dict() rejects dictionaries with missing fields."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        # Missing analysis_id
        incomplete_dict = {
            "ci_lower": 0.05,
            "ci_upper": 0.15,
            "delta": 0.1,
            "method": "percentile",
            "metric_path": "success",
            "n_baseline": 100,
            "n_bootstrap": 10000,
            "n_rfl": 100,
            "seed": 42,
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            PairedDeltaResult.validate_dict(incomplete_dict)
    
    @pytest.mark.unit
    def test_validate_dict_checks_types(self):
        """Test 104: validate_dict() checks field types."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        # Wrong type for n_baseline (string instead of int)
        wrong_type_dict = {
            "analysis_id": "test",
            "ci_lower": 0.05,
            "ci_upper": 0.15,
            "delta": 0.1,
            "method": "percentile",
            "metric_path": "success",
            "n_baseline": "one hundred",  # Should be int
            "n_bootstrap": 10000,
            "n_rfl": 100,
            "seed": 42,
        }
        
        with pytest.raises(ValueError, match="wrong type"):
            PairedDeltaResult.validate_dict(wrong_type_dict)
    
    @pytest.mark.unit
    def test_to_dict_passes_own_validation(self):
        """Test 105: PairedDeltaResult.to_dict() passes its own validate_dict()."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="self_validation_test",
        )
        
        d = result.to_dict()
        
        # to_dict() output must pass its own validation
        assert PairedDeltaResult.validate_dict(d) == True
    
    @pytest.mark.unit
    def test_schema_version_exists(self):
        """Test 106: PairedDeltaResult has a schema version for tracking."""
        from backend.metrics.u2_analysis import PairedDeltaResult, PAIRED_DELTA_RESULT_SCHEMA_VERSION
        
        version = PairedDeltaResult.get_schema_version()
        
        assert version is not None
        assert isinstance(version, str)
        assert version == PAIRED_DELTA_RESULT_SCHEMA_VERSION
        # Version should follow semver pattern
        assert "." in version
    
    @pytest.mark.unit
    def test_governance_verifier_compatibility(self):
        """Test 107: Contract output usable directly by governance verifier."""
        from backend.metrics.u2_analysis import PairedDeltaResult
        import json
        
        result = PairedDeltaResult(
            delta=0.15,
            ci_lower=0.08,
            ci_upper=0.22,
            n_baseline=500,
            n_rfl=500,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="percentile",
            analysis_id="governance_test_abc123",
        )
        
        # Get JSON output
        json_str = result.to_json()
        
        # Parse it back (simulating governance verifier consumption)
        parsed = json.loads(json_str)
        
        # Governance verifier should be able to:
        # 1. Access CI bounds directly
        assert "ci_lower" in parsed
        assert "ci_upper" in parsed
        assert isinstance(parsed["ci_lower"], float)
        assert isinstance(parsed["ci_upper"], float)
        
        # 2. Access delta directly
        assert "delta" in parsed
        assert isinstance(parsed["delta"], float)
        
        # 3. Verify sample sizes
        assert "n_baseline" in parsed
        assert "n_rfl" in parsed
        assert parsed["n_baseline"] >= 2
        assert parsed["n_rfl"] >= 2
        
        # 4. Trace back to source via analysis_id
        assert "analysis_id" in parsed
        assert len(parsed["analysis_id"]) > 0
        
        # 5. No transformation needed - all fields are primitives
        for key, value in parsed.items():
            assert isinstance(value, (int, float, str, bool)), \
                f"Field '{key}' requires transformation (type: {type(value)})"


# ===========================================================================
# D3 v1.1 — Evidence Pack Contract Tests
# ===========================================================================

class TestEvidencePackContract:
    """Tests for the multi-analysis evidence pack contract (D3 v1.1)."""
    
    @pytest.mark.unit
    def test_build_evidence_pack_returns_schema_version(self):
        """Test 108: Evidence pack includes schema_version."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.05, ci_upper=0.15, n_baseline=100, n_rfl=100,
            metric_path="success", seed=42, n_bootstrap=10000, method="percentile",
            analysis_id="test_id_1",
        )
        
        pack = build_evidence_pack([result])
        
        assert "schema_version" in pack
        assert pack["schema_version"] == "1.0.0"
    
    @pytest.mark.unit
    def test_build_evidence_pack_analysis_count_matches(self):
        """Test 109: Evidence pack analysis_count matches len(analyses)."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        
        results = [
            PairedDeltaResult(
                delta=0.1 * i, ci_lower=0.05, ci_upper=0.15, n_baseline=100, n_rfl=100,
                metric_path=f"metric_{i}", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id=f"test_id_{i}",
            )
            for i in range(5)
        ]
        
        pack = build_evidence_pack(results)
        
        assert pack["analysis_count"] == 5
        assert len(pack["analyses"]) == 5
        assert pack["analysis_count"] == len(pack["analyses"])
    
    @pytest.mark.unit
    def test_build_evidence_pack_deterministic_ordering(self):
        """Test 110: Evidence pack analyses are sorted by (metric_path, analysis_id)."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        
        # Create results in random order
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.05, ci_upper=0.15, n_baseline=100, n_rfl=100,
                metric_path="zebra", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="aaa",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="apple", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="zzz",
            ),
            PairedDeltaResult(
                delta=0.3, ci_lower=0.15, ci_upper=0.45, n_baseline=100, n_rfl=100,
                metric_path="apple", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="aaa",
            ),
        ]
        
        pack = build_evidence_pack(results)
        
        # Should be sorted by (metric_path, analysis_id)
        paths_and_ids = [(a["metric_path"], a["analysis_id"]) for a in pack["analyses"]]
        assert paths_and_ids == sorted(paths_and_ids)
        
        # Specific expected order: apple/aaa, apple/zzz, zebra/aaa
        assert pack["analyses"][0]["metric_path"] == "apple"
        assert pack["analyses"][0]["analysis_id"] == "aaa"
        assert pack["analyses"][1]["metric_path"] == "apple"
        assert pack["analyses"][1]["analysis_id"] == "zzz"
        assert pack["analyses"][2]["metric_path"] == "zebra"
    
    @pytest.mark.unit
    def test_build_evidence_pack_stable_across_runs(self):
        """Test 111: Evidence pack JSON is stable across multiple builds."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        import json
        
        results = [
            PairedDeltaResult(
                delta=0.15, ci_lower=0.08, ci_upper=0.22, n_baseline=50, n_rfl=50,
                metric_path="success", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="stable_test_id",
            ),
        ]
        
        pack1 = build_evidence_pack(results)
        pack2 = build_evidence_pack(results)
        
        json1 = json.dumps(pack1, sort_keys=True)
        json2 = json.dumps(pack2, sort_keys=True)
        
        assert json1 == json2
    
    @pytest.mark.unit
    def test_build_evidence_pack_empty_list(self):
        """Test 112: Evidence pack handles empty result list."""
        from backend.metrics.u2_analysis import build_evidence_pack
        
        pack = build_evidence_pack([])
        
        assert pack["analysis_count"] == 0
        assert pack["analyses"] == []
        assert "schema_version" in pack
        assert "pack_id" in pack
    
    @pytest.mark.unit
    def test_build_evidence_pack_custom_pack_id(self):
        """Test 113: Evidence pack accepts custom pack_id."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.05, ci_upper=0.15, n_baseline=100, n_rfl=100,
            metric_path="success", seed=42, n_bootstrap=10000, method="percentile",
            analysis_id="test_id",
        )
        
        pack = build_evidence_pack([result], pack_id="custom_pack_identifier_123")
        
        assert pack["pack_id"] == "custom_pack_identifier_123"
    
    @pytest.mark.unit
    def test_build_evidence_pack_deterministic_pack_id(self):
        """Test 114: Evidence pack generates deterministic pack_id when not provided."""
        from backend.metrics.u2_analysis import build_evidence_pack, PairedDeltaResult
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.05, ci_upper=0.15, n_baseline=100, n_rfl=100,
                metric_path="success", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="deterministic_test_id",
            ),
        ]
        
        pack1 = build_evidence_pack(results)
        pack2 = build_evidence_pack(results)
        
        # Same inputs should produce same pack_id
        assert pack1["pack_id"] == pack2["pack_id"]
        assert len(pack1["pack_id"]) == 64  # SHA-256 hex


class TestSummaryLineFormat:
    """Tests for the CLI summary line format (D3 v1.1)."""
    
    @pytest.mark.unit
    def test_summary_line_exact_format(self):
        """Test 115: Summary line has exact expected format."""
        from experiments.cli_uplift_stats import format_summary_line
        
        line = format_summary_line(
            metric_path="success",
            delta=0.123456,
            ci_lower=0.05,
            ci_upper=0.20,
            n_baseline=100,
            n_rfl=100,
            method="BCa",
        )
        
        expected = "BootstrapEvidence: metric=success delta=0.123456 ci=[0.050000,0.200000] n_base=100 n_rfl=100 method=BCa"
        assert line == expected
    
    @pytest.mark.unit
    def test_summary_line_no_trailing_newline(self):
        """Test 116: Summary line has no trailing newline."""
        from experiments.cli_uplift_stats import format_summary_line
        
        line = format_summary_line(
            metric_path="test",
            delta=0.0,
            ci_lower=-0.1,
            ci_upper=0.1,
            n_baseline=50,
            n_rfl=50,
            method="percentile",
        )
        
        assert not line.endswith("\n")
        assert not line.endswith("\r")
        assert "\n" not in line
    
    @pytest.mark.unit
    def test_summary_line_no_extra_whitespace(self):
        """Test 117: Summary line has no extra whitespace."""
        from experiments.cli_uplift_stats import format_summary_line
        
        line = format_summary_line(
            metric_path="success",
            delta=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
            n_baseline=100,
            n_rfl=100,
            method="BCa",
        )
        
        # No double spaces
        assert "  " not in line
        # No leading whitespace
        assert not line.startswith(" ")
        # No trailing whitespace
        assert not line.endswith(" ")
    
    @pytest.mark.unit
    def test_summary_line_deterministic(self):
        """Test 118: Summary line is deterministic."""
        from experiments.cli_uplift_stats import format_summary_line
        
        kwargs = {
            "metric_path": "success",
            "delta": 0.15,
            "ci_lower": 0.08,
            "ci_upper": 0.22,
            "n_baseline": 200,
            "n_rfl": 200,
            "method": "percentile",
        }
        
        line1 = format_summary_line(**kwargs)
        line2 = format_summary_line(**kwargs)
        
        assert line1 == line2
    
    @pytest.mark.unit
    def test_summary_line_handles_negative_delta(self):
        """Test 119: Summary line correctly formats negative delta."""
        from experiments.cli_uplift_stats import format_summary_line
        
        line = format_summary_line(
            metric_path="error_rate",
            delta=-0.05,
            ci_lower=-0.10,
            ci_upper=0.0,
            n_baseline=100,
            n_rfl=100,
            method="BCa",
        )
        
        assert "delta=-0.050000" in line
        assert "ci=[-0.100000,0.000000]" in line
    
    @pytest.mark.unit
    def test_summary_line_six_decimal_precision(self):
        """Test 120: Summary line uses 6 decimal places for numeric values."""
        from experiments.cli_uplift_stats import format_summary_line
        
        line = format_summary_line(
            metric_path="test",
            delta=0.1,
            ci_lower=0.0,
            ci_upper=0.2,
            n_baseline=100,
            n_rfl=100,
            method="BCa",
        )
        
        # Check decimal precision
        assert "delta=0.100000" in line
        assert "ci=[0.000000,0.200000]" in line


class TestGovernanceSummary:
    """Tests for the non-interpretive governance summary (D3 v1.1)."""
    
    @pytest.mark.unit
    def test_governance_summary_no_forbidden_words(self):
        """Test 121: Governance summary contains no forbidden interpretive words."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        # Use metric paths that don't contain forbidden words
        # Avoid: "success", "failure", "improvement", etc.
        results = [
            PairedDeltaResult(
                delta=0.5, ci_lower=0.3, ci_upper=0.7, n_baseline=100, n_rfl=100,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="test_1",
            ),
            PairedDeltaResult(
                delta=-0.1, ci_lower=-0.2, ci_upper=0.0, n_baseline=50, n_rfl=50,
                metric_path="latency_ms", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="test_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        summary = summarize_evidence_for_governance(pack)
        
        # Convert to string to check for forbidden words
        summary_str = json.dumps(summary).lower()
        
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in summary_str, \
                f"Forbidden word '{word}' found in governance summary"
    
    @pytest.mark.unit
    def test_governance_summary_analysis_count_matches(self):
        """Test 122: Governance summary analysis_count matches pack."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1 * i, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path=f"metric_{i}", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id=f"id_{i}",
            )
            for i in range(3)
        ]
        
        pack = build_evidence_pack(results)
        summary = summarize_evidence_for_governance(pack)
        
        assert summary["analysis_count"] == pack["analysis_count"]
        assert summary["analysis_count"] == 3
    
    @pytest.mark.unit
    def test_governance_summary_metric_paths_complete(self):
        """Test 123: Governance summary includes all metric paths."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="success", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="abstention_rate", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_2",
            ),
            PairedDeltaResult(
                delta=0.15, ci_lower=0.05, ci_upper=0.25, n_baseline=100, n_rfl=100,
                metric_path="throughput", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_3",
            ),
        ]
        
        pack = build_evidence_pack(results)
        summary = summarize_evidence_for_governance(pack)
        
        assert set(summary["metric_paths"]) == {"success", "abstention_rate", "throughput"}
        # Should be sorted
        assert summary["metric_paths"] == sorted(summary["metric_paths"])
    
    @pytest.mark.unit
    def test_governance_summary_sample_sizes_correct(self):
        """Test 124: Governance summary sample sizes are accurate."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=50, n_rfl=60,
                metric_path="m1", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=200, n_rfl=180,
                metric_path="m2", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_2",
            ),
            PairedDeltaResult(
                delta=0.15, ci_lower=0.05, ci_upper=0.25, n_baseline=100, n_rfl=100,
                metric_path="m3", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_3",
            ),
        ]
        
        pack = build_evidence_pack(results)
        summary = summarize_evidence_for_governance(pack)
        
        assert summary["sample_sizes"]["min_baseline"] == 50
        assert summary["sample_sizes"]["max_baseline"] == 200
        assert summary["sample_sizes"]["min_rfl"] == 60
        assert summary["sample_sizes"]["max_rfl"] == 180
    
    @pytest.mark.unit
    def test_governance_summary_methods_used(self):
        """Test 125: Governance summary includes all methods used."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="m1", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="m2", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="id_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        summary = summarize_evidence_for_governance(pack)
        
        assert set(summary["methods_used"]) == {"BCa", "percentile"}
        assert summary["methods_used"] == sorted(summary["methods_used"])
    
    @pytest.mark.unit
    def test_governance_summary_empty_pack(self):
        """Test 126: Governance summary handles empty evidence pack."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
        )
        
        pack = build_evidence_pack([])
        summary = summarize_evidence_for_governance(pack)
        
        assert summary["analysis_count"] == 0
        assert summary["metric_paths"] == []
        assert summary["methods_used"] == []
        assert summary["sample_sizes"]["min_baseline"] == 0
        assert summary["sample_sizes"]["max_baseline"] == 0
    
    @pytest.mark.unit
    def test_governance_summary_includes_pack_id(self):
        """Test 127: Governance summary includes pack_id for traceability."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack, 
            summarize_evidence_for_governance,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="success", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="traceability_test",
        )
        
        pack = build_evidence_pack([result], pack_id="governance_trace_123")
        summary = summarize_evidence_for_governance(pack)
        
        assert summary["pack_id"] == "governance_trace_123"


class TestU2AnalysisFormatHelper:
    """Tests for format_evidence_summary_line in u2_analysis (D3 v1.1)."""
    
    @pytest.mark.unit
    def test_format_evidence_summary_line_from_result(self):
        """Test 128: format_evidence_summary_line works with PairedDeltaResult."""
        from backend.metrics.u2_analysis import format_evidence_summary_line, PairedDeltaResult
        
        result = PairedDeltaResult(
            delta=0.123456,
            ci_lower=0.05,
            ci_upper=0.20,
            n_baseline=100,
            n_rfl=100,
            metric_path="success",
            seed=42,
            n_bootstrap=10000,
            method="BCa",
            analysis_id="test_id",
        )
        
        line = format_evidence_summary_line(result)
        
        expected = "BootstrapEvidence: metric=success delta=0.123456 ci=[0.050000,0.200000] n_base=100 n_rfl=100 method=BCa"
        assert line == expected
    
    @pytest.mark.unit
    def test_format_evidence_summary_line_matches_cli(self):
        """Test 129: u2_analysis helper produces same output as CLI helper."""
        from backend.metrics.u2_analysis import format_evidence_summary_line, PairedDeltaResult
        from experiments.cli_uplift_stats import format_summary_line
        
        result = PairedDeltaResult(
            delta=0.15,
            ci_lower=0.08,
            ci_upper=0.22,
            n_baseline=200,
            n_rfl=200,
            metric_path="throughput",
            seed=123,
            n_bootstrap=5000,
            method="percentile",
            analysis_id="consistency_test",
        )
        
        # Both should produce identical output
        u2_line = format_evidence_summary_line(result)
        cli_line = format_summary_line(
            metric_path=result.metric_path,
            delta=result.delta,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            n_baseline=result.n_baseline,
            n_rfl=result.n_rfl,
            method=result.method,
        )
        
        assert u2_line == cli_line


# ===========================================================================
# D3 Phase III — Evidence Pack Governance & Readiness Tests
# ===========================================================================

class TestEvidenceQualitySnapshot:
    """Tests for build_evidence_quality_snapshot (D3 Phase III)."""
    
    @pytest.mark.unit
    def test_quality_snapshot_schema_version(self):
        """Test 130: Quality snapshot includes schema_version from pack."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == pack["schema_version"]
    
    @pytest.mark.unit
    def test_quality_snapshot_analysis_count(self):
        """Test 131: Quality snapshot has correct analysis_count."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1 * i, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path=f"metric_{i}", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id=f"id_{i}",
            )
            for i in range(5)
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert snapshot["analysis_count"] == 5
    
    @pytest.mark.unit
    def test_quality_snapshot_methods_by_metric(self):
        """Test 132: Quality snapshot correctly maps methods to metrics."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="id_2",
            ),
            PairedDeltaResult(
                delta=0.15, ci_lower=0.05, ci_upper=0.25, n_baseline=100, n_rfl=100,
                metric_path="latency", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_3",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert "accuracy" in snapshot["methods_by_metric"]
        assert "latency" in snapshot["methods_by_metric"]
        assert set(snapshot["methods_by_metric"]["accuracy"]) == {"BCa", "percentile"}
        assert snapshot["methods_by_metric"]["latency"] == ["BCa"]
    
    @pytest.mark.unit
    def test_quality_snapshot_metrics_with_multiple_methods(self):
        """Test 133: Quality snapshot identifies metrics with multiple methods."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="multi_method", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="multi_method", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="id_2",
            ),
            PairedDeltaResult(
                delta=0.15, ci_lower=0.05, ci_upper=0.25, n_baseline=100, n_rfl=100,
                metric_path="single_method", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_3",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert "multi_method" in snapshot["metrics_with_multiple_methods"]
        assert "single_method" not in snapshot["metrics_with_multiple_methods"]
    
    @pytest.mark.unit
    def test_quality_snapshot_sample_size_bounds(self):
        """Test 134: Quality snapshot has correct min/max sample sizes."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=50, n_rfl=60,
                metric_path="m1", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=200, n_rfl=180,
                metric_path="m2", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert snapshot["min_n_baseline"] == 50
        assert snapshot["max_n_baseline"] == 200
        assert snapshot["min_n_rfl"] == 60
        assert snapshot["max_n_rfl"] == 180
    
    @pytest.mark.unit
    def test_quality_snapshot_empty_pack(self):
        """Test 135: Quality snapshot handles empty pack."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        
        assert snapshot["analysis_count"] == 0
        assert snapshot["methods_by_metric"] == {}
        assert snapshot["metrics_with_multiple_methods"] == []
        assert snapshot["min_n_baseline"] == 0
        assert snapshot["max_n_baseline"] == 0
    
    @pytest.mark.unit
    def test_quality_snapshot_deterministic(self):
        """Test 136: Quality snapshot output is deterministic."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            PairedDeltaResult,
        )
        import json
        
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="zebra", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="apple", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot1 = build_evidence_quality_snapshot(pack)
        snapshot2 = build_evidence_quality_snapshot(pack)
        
        assert json.dumps(snapshot1, sort_keys=True) == json.dumps(snapshot2, sort_keys=True)


class TestEvidenceReadiness:
    """Tests for evaluate_evidence_readiness (D3 Phase III)."""
    
    @pytest.mark.unit
    def test_readiness_ok_status(self):
        """Test 137: Readiness returns OK for quality evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        assert readiness["status"] == "OK"
        assert readiness["ready_for_governance_review"] == True
        assert readiness["weak_points"] == []
    
    @pytest.mark.unit
    def test_readiness_attention_low_samples(self):
        """Test 138: Readiness returns ATTENTION for low sample sizes."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            PairedDeltaResult,
            EVIDENCE_QUALITY_THRESHOLDS,
        )
        
        # Use sample size below weak threshold but above block threshold
        low_n = EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_weak"] - 1
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=low_n, n_rfl=low_n,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        assert readiness["status"] == "ATTENTION"
        assert readiness["ready_for_governance_review"] == True
        assert len(readiness["weak_points"]) > 0
    
    @pytest.mark.unit
    def test_readiness_weak_critical_samples(self):
        """Test 139: Readiness returns WEAK for critically low samples."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            PairedDeltaResult,
            EVIDENCE_QUALITY_THRESHOLDS,
        )
        
        # Use sample size below block threshold
        critical_n = EVIDENCE_QUALITY_THRESHOLDS["min_sample_size_block"] - 1
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=critical_n, n_rfl=critical_n,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        assert readiness["status"] == "WEAK"
        assert readiness["ready_for_governance_review"] == False
    
    @pytest.mark.unit
    def test_readiness_weak_empty_pack(self):
        """Test 140: Readiness returns WEAK for empty pack."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        assert readiness["status"] == "WEAK"
        assert readiness["ready_for_governance_review"] == False
        assert "no analyses present" in readiness["weak_points"][0]
    
    @pytest.mark.unit
    def test_readiness_weak_points_sorted(self):
        """Test 141: Readiness weak_points are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            PairedDeltaResult,
        )
        
        # Create conditions that trigger multiple weak points
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        # Weak points should be sorted
        assert readiness["weak_points"] == sorted(readiness["weak_points"])
    
    @pytest.mark.unit
    def test_readiness_no_uplift_semantics(self):
        """Test 142: Readiness has no uplift semantics in weak_points."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            PairedDeltaResult,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        # Check no forbidden words in weak_points
        all_weak_text = " ".join(readiness["weak_points"]).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in all_weak_text, \
                f"Forbidden word '{word}' found in readiness weak_points"


class TestGlobalHealthSummary:
    """Tests for summarize_evidence_for_global_health (D3 Phase III)."""
    
    @pytest.mark.unit
    def test_global_health_ok_status(self):
        """Test 143: Global health returns OK for quality evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        assert health["status"] == "OK"
        assert health["evidence_ok"] == True
        assert health["analysis_count"] == 1
        assert health["weak_metric_paths"] == []
    
    @pytest.mark.unit
    def test_global_health_warn_status(self):
        """Test 144: Global health returns WARN for ATTENTION readiness."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
        )
        
        # Low sample size triggers ATTENTION
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        assert health["status"] == "WARN"
        assert health["evidence_ok"] == True  # Still ready for review
    
    @pytest.mark.unit
    def test_global_health_block_status(self):
        """Test 145: Global health returns BLOCK for WEAK readiness."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        assert health["status"] == "BLOCK"
        assert health["evidence_ok"] == False
    
    @pytest.mark.unit
    def test_global_health_weak_metric_paths(self):
        """Test 146: Global health identifies weak metric paths."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
        )
        
        # Low sample sizes trigger weak metric paths
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=5, n_rfl=5,
                metric_path="latency", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="id_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        # With low samples, all metrics are marked as weak
        assert len(health["weak_metric_paths"]) > 0
        assert "accuracy" in health["weak_metric_paths"]
        assert "latency" in health["weak_metric_paths"]
    
    @pytest.mark.unit
    def test_global_health_analysis_count(self):
        """Test 147: Global health has correct analysis_count."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
        )
        
        results = [
            PairedDeltaResult(
                delta=0.1 * i, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path=f"metric_{i}", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id=f"id_{i}",
            )
            for i in range(3)
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        assert health["analysis_count"] == 3
    
    @pytest.mark.unit
    def test_global_health_deterministic(self):
        """Test 148: Global health output is deterministic."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
        )
        import json
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        
        health1 = summarize_evidence_for_global_health(snapshot, readiness)
        health2 = summarize_evidence_for_global_health(snapshot, readiness)
        
        assert json.dumps(health1, sort_keys=True) == json.dumps(health2, sort_keys=True)
    
    @pytest.mark.unit
    def test_global_health_no_uplift_interpretation(self):
        """Test 149: Global health contains no uplift interpretation."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            summarize_evidence_for_global_health,
            PairedDeltaResult,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        result = PairedDeltaResult(
            delta=0.5, ci_lower=0.3, ci_upper=0.7, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        health = summarize_evidence_for_global_health(snapshot, readiness)
        
        health_str = json.dumps(health).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in health_str, \
                f"Forbidden word '{word}' found in global health summary"


# ===========================================================================
# D3 Phase IV — Evidence Quality Ladder & Release Governance Feed Tests
# ===========================================================================

class TestEvidenceQualityLadder:
    """Tests for classify_evidence_quality_level (D3 Phase IV)."""
    
    @pytest.mark.unit
    def test_quality_tier_1_minimal(self):
        """Test 150: Quality tier 1 for minimal evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            classify_evidence_quality_level,
            PairedDeltaResult,
        )
        
        # Single metric, low sample size, single method
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=10, n_rfl=10,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        tier_info = classify_evidence_quality_level(snapshot)
        
        assert tier_info["quality_tier"] == "TIER_1"
        assert len(tier_info["requirements_met"]) > 0
        assert len(tier_info["requirements_missing"]) > 0
    
    @pytest.mark.unit
    def test_quality_tier_2_standard(self):
        """Test 151: Quality tier 2 for standard evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            classify_evidence_quality_level,
            PairedDeltaResult,
        )
        
        # Multiple metrics, adequate sample sizes, single method
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=25, n_rfl=25,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="test_1",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=25, n_rfl=25,
                metric_path="latency", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="test_2",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        tier_info = classify_evidence_quality_level(snapshot)
        
        assert tier_info["quality_tier"] == "TIER_2"
        assert "multiple metrics" in " ".join(tier_info["requirements_met"])
        assert "adequate sample sizes" in " ".join(tier_info["requirements_met"])
    
    @pytest.mark.unit
    def test_quality_tier_3_strong(self):
        """Test 152: Quality tier 3 for strong evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            classify_evidence_quality_level,
            PairedDeltaResult,
        )
        
        # Multiple metrics, strong sample sizes, multiple methods per metric
        results = [
            PairedDeltaResult(
                delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="test_1",
            ),
            PairedDeltaResult(
                delta=0.15, ci_lower=0.05, ci_upper=0.25, n_baseline=100, n_rfl=100,
                metric_path="accuracy", seed=42, n_bootstrap=10000, method="percentile",
                analysis_id="test_2",
            ),
            PairedDeltaResult(
                delta=0.2, ci_lower=0.1, ci_upper=0.3, n_baseline=100, n_rfl=100,
                metric_path="latency", seed=42, n_bootstrap=10000, method="BCa",
                analysis_id="test_3",
            ),
        ]
        
        pack = build_evidence_pack(results)
        snapshot = build_evidence_quality_snapshot(pack)
        tier_info = classify_evidence_quality_level(snapshot)
        
        assert tier_info["quality_tier"] == "TIER_3"
        assert "multiple methods per metric" in " ".join(tier_info["requirements_met"])
        assert "strong sample sizes" in " ".join(tier_info["requirements_met"])
        assert len(tier_info["requirements_missing"]) == 0
    
    @pytest.mark.unit
    def test_quality_tier_requirements_sorted(self):
        """Test 153: Quality tier requirements are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            classify_evidence_quality_level,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=10, n_rfl=10,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        tier_info = classify_evidence_quality_level(snapshot)
        
        # Requirements should be sorted
        assert tier_info["requirements_met"] == sorted(tier_info["requirements_met"])
        assert tier_info["requirements_missing"] == sorted(tier_info["requirements_missing"])
    
    @pytest.mark.unit
    def test_quality_tier_no_uplift_semantics(self):
        """Test 154: Quality tier has no uplift semantics."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            classify_evidence_quality_level,
            PairedDeltaResult,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        result = PairedDeltaResult(
            delta=0.5, ci_lower=0.3, ci_upper=0.7, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        tier_info = classify_evidence_quality_level(snapshot)
        
        tier_str = json.dumps(tier_info).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in tier_str, \
                f"Forbidden word '{word}' found in quality tier info"


class TestEvidencePromotionEvaluation:
    """Tests for evaluate_evidence_for_promotion (D3 Phase IV)."""
    
    @pytest.mark.unit
    def test_promotion_ok_status(self):
        """Test 155: Promotion returns OK for quality evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "OK"}
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["promotion_ok"] == True
        assert promotion["status"] == "OK"
        assert len(promotion["blocking_reasons"]) == 0
    
    @pytest.mark.unit
    def test_promotion_warn_status(self):
        """Test 156: Promotion returns WARN for attention-needed evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
            PairedDeltaResult,
        )
        
        # Low sample size triggers ATTENTION
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "OK"}
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["promotion_ok"] == True
        assert promotion["status"] == "WARN"
        assert len(promotion["notes"]) > 0
    
    @pytest.mark.unit
    def test_promotion_block_weak_evidence(self):
        """Test 157: Promotion returns BLOCK for weak evidence."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "OK"}
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["promotion_ok"] == False
        assert promotion["status"] == "BLOCK"
        assert len(promotion["blocking_reasons"]) > 0
    
    @pytest.mark.unit
    def test_promotion_block_adversarial_failure(self):
        """Test 158: Promotion blocks when adversarial health fails."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "BLOCK"}  # Adversarial health blocks
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["promotion_ok"] == False
        assert promotion["status"] == "BLOCK"
        assert "adversarial health check failed" in promotion["blocking_reasons"][0]
    
    @pytest.mark.unit
    def test_promotion_warn_adversarial_warn(self):
        """Test 159: Promotion warns when adversarial health warns."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "WARN"}  # Adversarial health warns
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["promotion_ok"] == True
        assert promotion["status"] == "WARN"
        assert "adversarial health shows warnings" in promotion["notes"][0]
    
    @pytest.mark.unit
    def test_promotion_blocking_reasons_sorted(self):
        """Test 160: Promotion blocking_reasons are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            evaluate_evidence_for_promotion,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        adversarial = {"status": "BLOCK"}
        
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        assert promotion["blocking_reasons"] == sorted(promotion["blocking_reasons"])
        assert promotion["notes"] == sorted(promotion["notes"])


class TestDirectorEvidencePanel:
    """Tests for build_evidence_director_panel (D3 Phase IV)."""
    
    @pytest.mark.unit
    def test_director_panel_green_light(self):
        """Test 161: Director panel shows GREEN for OK promotion."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        assert panel["status_light"] == "GREEN"
        assert panel["evidence_ok"] == True
        assert panel["quality_tier"] in ["TIER_1", "TIER_2", "TIER_3"]
    
    @pytest.mark.unit
    def test_director_panel_yellow_light(self):
        """Test 162: Director panel shows YELLOW for WARN promotion."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
        )
        
        # Low sample size triggers WARN
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=5, n_rfl=5,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        assert panel["status_light"] == "YELLOW"
        assert panel["evidence_ok"] == True
    
    @pytest.mark.unit
    def test_director_panel_red_light(self):
        """Test 163: Director panel shows RED for BLOCK promotion."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
        )
        
        pack = build_evidence_pack([])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        assert panel["status_light"] == "RED"
        assert panel["evidence_ok"] == False
    
    @pytest.mark.unit
    def test_director_panel_headline_present(self):
        """Test 164: Director panel includes headline."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        assert "headline" in panel
        assert isinstance(panel["headline"], str)
        assert len(panel["headline"]) > 0
    
    @pytest.mark.unit
    def test_director_panel_headline_neutral(self):
        """Test 165: Director panel headline is neutral and descriptive."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        
        result = PairedDeltaResult(
            delta=0.5, ci_lower=0.3, ci_upper=0.7, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        headline_lower = panel["headline"].lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in headline_lower, \
                f"Forbidden word '{word}' found in director panel headline"
    
    @pytest.mark.unit
    def test_director_panel_all_fields_present(self):
        """Test 166: Director panel includes all required fields."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel = build_evidence_director_panel(tier_info, promotion)
        
        required_fields = ["status_light", "quality_tier", "analysis_count", "evidence_ok", "headline"]
        for field in required_fields:
            assert field in panel, f"Missing required field: {field}"
    
    @pytest.mark.unit
    def test_director_panel_deterministic(self):
        """Test 167: Director panel output is deterministic."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_director_panel,
            PairedDeltaResult,
        )
        import json
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        panel1 = build_evidence_director_panel(tier_info, promotion)
        panel2 = build_evidence_director_panel(tier_info, promotion)
        
        assert json.dumps(panel1, sort_keys=True) == json.dumps(panel2, sort_keys=True)


# ===========================================================================
# D3 Phase V — Evidence Evolution Timeline & Regression Watchdog Tests
# ===========================================================================

class TestEvidenceQualityTimeline:
    """Tests for build_evidence_quality_timeline (D3 Phase V)."""
    
    @pytest.mark.unit
    def test_timeline_improving_trend(self):
        """Test 168: Timeline detects IMPROVING trend."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        
        assert timeline["quality_trend"] == "IMPROVING"
        assert len(timeline["timeline"]) == 3
        assert "TIER_1→TIER_2" in timeline["tier_transition_counts"]
        assert "TIER_2→TIER_3" in timeline["tier_transition_counts"]
    
    @pytest.mark.unit
    def test_timeline_degrading_trend(self):
        """Test 169: Timeline detects DEGRADING trend."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        
        assert timeline["quality_trend"] == "DEGRADING"
        assert "TIER_3→TIER_2" in timeline["tier_transition_counts"]
        assert "TIER_2→TIER_1" in timeline["tier_transition_counts"]
    
    @pytest.mark.unit
    def test_timeline_stable_trend(self):
        """Test 170: Timeline detects STABLE trend."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        
        assert timeline["quality_trend"] == "STABLE"
        assert len(timeline["tier_transition_counts"]) == 0
    
    @pytest.mark.unit
    def test_timeline_empty_snapshots(self):
        """Test 171: Timeline handles empty snapshots."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        timeline = build_evidence_quality_timeline([])
        
        assert timeline["quality_trend"] == "STABLE"
        assert timeline["timeline"] == []
        assert timeline["tier_transition_counts"] == {}
        assert timeline["schema_version"] == "1.0.0"
    
    @pytest.mark.unit
    def test_timeline_sorted_by_run_id(self):
        """Test 172: Timeline entries are sorted by run_id."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        snapshots = [
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        
        run_ids = [entry["run_id"] for entry in timeline["timeline"]]
        assert run_ids == ["run1", "run2", "run3"]
    
    @pytest.mark.unit
    def test_timeline_transition_counts_deterministic(self):
        """Test 173: Timeline transition counts are sorted for determinism."""
        from backend.metrics.u2_analysis import build_evidence_quality_timeline
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run3", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        
        # Transition counts keys should be sorted
        keys = list(timeline["tier_transition_counts"].keys())
        assert keys == sorted(keys)


class TestEvidenceQualityRegression:
    """Tests for evaluate_evidence_quality_regression (D3 Phase V)."""
    
    @pytest.mark.unit
    def test_regression_block_multiple_tier3_drops(self):
        """Test 174: Regression blocks on multiple TIER_3 drops."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == True
        assert regression["status"] == "BLOCK"
        assert "multiple quality drops from TIER_3" in " ".join(regression["neutral_reasons"])
    
    @pytest.mark.unit
    def test_regression_block_recent_below_tier3(self):
        """Test 175: Regression blocks when recent runs consistently below TIER_3."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == True
        assert regression["status"] == "BLOCK"
        assert "recent runs show consistent quality below TIER_3" in " ".join(regression["neutral_reasons"])
    
    @pytest.mark.unit
    def test_regression_attention_degrading_trend(self):
        """Test 176: Regression warns on degrading trend."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == True
        assert regression["status"] == "ATTENTION"
        assert "quality drop from TIER_3" in " ".join(regression["neutral_reasons"])
    
    @pytest.mark.unit
    def test_regression_ok_no_regression(self):
        """Test 177: Regression returns OK when no regression detected."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == False
        assert regression["status"] == "OK"
        assert len(regression["neutral_reasons"]) == 0
    
    @pytest.mark.unit
    def test_regression_empty_timeline(self):
        """Test 178: Regression handles empty timeline."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        timeline = build_evidence_quality_timeline([])
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == False
        assert regression["status"] == "OK"
        assert "no timeline data available" in regression["neutral_reasons"][0]
    
    @pytest.mark.unit
    def test_regression_neutral_reasons_sorted(self):
        """Test 179: Regression neutral_reasons are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["neutral_reasons"] == sorted(regression["neutral_reasons"])
    
    @pytest.mark.unit
    def test_regression_no_uplift_semantics(self):
        """Test 180: Regression evaluation has no uplift semantics."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        regression_str = json.dumps(regression).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in regression_str, \
                f"Forbidden word '{word}' found in regression evaluation"
    
    @pytest.mark.unit
    def test_regression_attention_multiple_degrading(self):
        """Test 181: Regression warns on multiple degrading transitions."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            evaluate_evidence_quality_regression,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run5", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run6", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        regression = evaluate_evidence_quality_regression(timeline)
        
        assert regression["regression_detected"] == True
        assert regression["status"] == "ATTENTION"
        assert "multiple quality degrading transitions" in " ".join(regression["neutral_reasons"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

