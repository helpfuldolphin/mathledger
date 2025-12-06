"""
Tests for RFL Bootstrap Statistics

Validates BCa confidence intervals, uplift computation, and metabolism verification.
"""

import pytest
import numpy as np
from backend.rfl.bootstrap_stats import (
    bootstrap_bca,
    bootstrap_percentile,
    compute_uplift_ci,
    compute_coverage_ci,
    verify_metabolism,
    BootstrapResult
)


class TestBootstrapBCA:
    """Tests for BCa bootstrap confidence intervals."""

    def test_bca_basic(self):
        """Test basic BCa computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)

        result = bootstrap_bca(
            data,
            statistic=np.mean,
            num_replicates=1000,
            confidence_level=0.95,
            random_state=42
        )

        assert isinstance(result, BootstrapResult)
        assert 9 < result.point_estimate < 11  # ~10
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.ci_width > 0
        assert result.method.startswith("BCa_")
        assert result.num_replicates == 1000

    def test_bca_median(self):
        """Test BCa with median (non-mean statistic)."""
        np.random.seed(42)
        data = np.random.exponential(5, size=50)

        result = bootstrap_bca(
            data,
            statistic=np.median,
            num_replicates=1000,
            random_state=42
        )

        assert result.ci_lower < result.point_estimate < result.ci_upper

    def test_bca_ratio_statistic(self):
        """Test BCa with ratio statistic (biased estimator)."""
        np.random.seed(42)

        # Paired data: [numerator, denominator]
        data = np.column_stack([
            np.random.poisson(100, size=40),
            np.random.poisson(80, size=40)
        ])

        def ratio_statistic(paired_sample):
            num_mean = np.mean(paired_sample[:, 0])
            den_mean = np.mean(paired_sample[:, 1])
            return num_mean / den_mean if den_mean > 0 else np.nan

        result = bootstrap_bca(
            data,
            statistic=ratio_statistic,
            num_replicates=1000,
            random_state=42
        )

        assert result.point_estimate > 1.0  # Expect ~1.25
        assert result.ci_lower > 0

    def test_bca_insufficient_data(self):
        """Test BCa with insufficient data."""
        data = np.array([5.0])

        with pytest.raises(ValueError, match="at least 2 observations"):
            bootstrap_bca(data, np.mean, num_replicates=100)

    def test_bca_determinism(self):
        """Test BCa determinism with same random seed."""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=50)

        result1 = bootstrap_bca(data, np.mean, num_replicates=500, random_state=42)
        result2 = bootstrap_bca(data, np.mean, num_replicates=500, random_state=42)

        assert result1.point_estimate == result2.point_estimate
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper


class TestBootstrapPercentile:
    """Tests for percentile bootstrap (fallback method)."""

    def test_percentile_basic(self):
        """Test basic percentile bootstrap."""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)

        result = bootstrap_percentile(
            data,
            statistic=np.mean,
            num_replicates=1000,
            random_state=42
        )

        assert isinstance(result, BootstrapResult)
        assert result.method.startswith("Percentile_")
        assert result.ci_lower < result.point_estimate < result.ci_upper


class TestUpliftCI:
    """Tests for uplift confidence intervals."""

    def test_uplift_basic(self):
        """Test basic uplift CI computation."""
        np.random.seed(42)

        # Baseline: mean=100, Treatment: mean=130 (30% uplift)
        baseline = np.random.poisson(100, size=40)
        treatment = np.random.poisson(130, size=40)

        result = compute_uplift_ci(
            baseline,
            treatment,
            num_replicates=1000,
            random_state=42
        )

        assert result.point_estimate > 1.0  # Positive uplift
        assert result.ci_lower > 0
        assert 1.1 < result.point_estimate < 1.5  # ~30% uplift

    def test_uplift_no_change(self):
        """Test uplift when baseline == treatment."""
        np.random.seed(42)

        baseline = np.random.poisson(100, size=40)
        treatment = np.random.poisson(100, size=40)

        result = compute_uplift_ci(baseline, treatment, num_replicates=1000, random_state=42)

        # Should be close to 1.0
        assert 0.8 < result.point_estimate < 1.2
        assert result.ci_lower < 1.0 < result.ci_upper  # CI should include 1.0

    def test_uplift_regression(self):
        """Test uplift with regression (treatment < baseline)."""
        np.random.seed(42)

        baseline = np.random.poisson(100, size=40)
        treatment = np.random.poisson(70, size=40)  # 30% regression

        result = compute_uplift_ci(baseline, treatment, num_replicates=1000, random_state=42)

        assert result.point_estimate < 1.0  # Negative uplift

    def test_uplift_mismatched_length(self):
        """Test uplift with mismatched array lengths."""
        baseline = np.array([1, 2, 3])
        treatment = np.array([1, 2])

        with pytest.raises(ValueError, match="same length"):
            compute_uplift_ci(baseline, treatment, num_replicates=100)

    def test_uplift_zero_baseline_abstain(self):
        """Test uplift abstains when baseline is zero."""
        baseline = np.zeros(40)
        treatment = np.random.poisson(100, size=40)

        result = compute_uplift_ci(baseline, treatment, num_replicates=1000, random_state=42)

        assert result.method == "ABSTAIN"
        assert np.isnan(result.point_estimate)


class TestCoverageCI:
    """Tests for coverage confidence intervals."""

    def test_coverage_basic(self):
        """Test basic coverage CI computation."""
        np.random.seed(42)

        # Simulate coverage rates around 0.92
        coverage_rates = np.random.beta(20, 2, size=40)  # Mean ~0.909

        result = compute_coverage_ci(
            coverage_rates,
            num_replicates=1000,
            random_state=42
        )

        assert 0.8 < result.point_estimate < 1.0
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert 0 <= result.ci_lower <= 1
        assert 0 <= result.ci_upper <= 1

    def test_coverage_high(self):
        """Test coverage CI with high coverage rates."""
        coverage_rates = np.random.beta(50, 2, size=40)  # Mean ~0.96

        result = compute_coverage_ci(coverage_rates, num_replicates=1000, random_state=42)

        assert result.point_estimate > 0.92

    def test_coverage_invalid_range(self):
        """Test coverage CI rejects invalid rates."""
        invalid_rates = np.array([0.5, 0.8, 1.2])  # 1.2 is invalid

        with pytest.raises(ValueError, match="must be in"):
            compute_coverage_ci(invalid_rates, num_replicates=100)


class TestMetabolismVerification:
    """Tests for metabolism acceptance criteria verification."""

    def test_metabolism_pass(self):
        """Test metabolism verification passes with good metrics."""
        coverage_ci = BootstrapResult(
            point_estimate=0.95,
            ci_lower=0.93,
            ci_upper=0.97,
            std_error=0.01,
            num_replicates=10000,
            method="BCa_95%"
        )

        uplift_ci = BootstrapResult(
            point_estimate=1.30,
            ci_lower=1.15,
            ci_upper=1.45,
            std_error=0.08,
            num_replicates=10000,
            method="BCa_95%"
        )

        passed, message = verify_metabolism(
            coverage_ci,
            uplift_ci,
            coverage_threshold=0.92,
            uplift_threshold=1.0
        )

        assert passed is True
        assert "[PASS]" in message
        assert "coverage=" in message
        assert "uplift=" in message

    def test_metabolism_fail_coverage(self):
        """Test metabolism verification fails on low coverage."""
        coverage_ci = BootstrapResult(
            point_estimate=0.88,
            ci_lower=0.85,
            ci_upper=0.91,
            std_error=0.015,
            num_replicates=10000,
            method="BCa_95%"
        )

        uplift_ci = BootstrapResult(
            point_estimate=1.20,
            ci_lower=1.10,
            ci_upper=1.30,
            std_error=0.05,
            num_replicates=10000,
            method="BCa_95%"
        )

        passed, message = verify_metabolism(coverage_ci, uplift_ci)

        assert passed is False
        assert "[FAIL]" in message
        assert "coverage CI lower" in message

    def test_metabolism_fail_uplift(self):
        """Test metabolism verification fails on low uplift."""
        coverage_ci = BootstrapResult(
            point_estimate=0.95,
            ci_lower=0.93,
            ci_upper=0.97,
            std_error=0.01,
            num_replicates=10000,
            method="BCa_95%"
        )

        uplift_ci = BootstrapResult(
            point_estimate=1.05,
            ci_lower=0.95,
            ci_upper=1.15,
            std_error=0.05,
            num_replicates=10000,
            method="BCa_95%"
        )

        passed, message = verify_metabolism(coverage_ci, uplift_ci)

        assert passed is False
        assert "[FAIL]" in message
        assert "uplift CI lower" in message

    def test_metabolism_abstain_coverage(self):
        """Test metabolism verification abstains on coverage failure."""
        coverage_ci = BootstrapResult(
            point_estimate=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std_error=np.nan,
            num_replicates=0,
            method="ABSTAIN"
        )

        uplift_ci = BootstrapResult(
            point_estimate=1.30,
            ci_lower=1.15,
            ci_upper=1.45,
            std_error=0.08,
            num_replicates=10000,
            method="BCa_95%"
        )

        passed, message = verify_metabolism(coverage_ci, uplift_ci)

        assert passed is False
        assert "[ABSTAIN]" in message
        assert "Coverage computation failed" in message

    def test_metabolism_abstain_uplift(self):
        """Test metabolism verification abstains on uplift failure."""
        coverage_ci = BootstrapResult(
            point_estimate=0.95,
            ci_lower=0.93,
            ci_upper=0.97,
            std_error=0.01,
            num_replicates=10000,
            method="BCa_95%"
        )

        uplift_ci = BootstrapResult(
            point_estimate=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std_error=np.nan,
            num_replicates=0,
            method="ABSTAIN"
        )

        passed, message = verify_metabolism(coverage_ci, uplift_ci)

        assert passed is False
        assert "[ABSTAIN]" in message
        assert "Uplift computation failed" in message


class TestBootstrapResultMethods:
    """Tests for BootstrapResult utility methods."""

    def test_ci_width(self):
        """Test CI width calculation."""
        result = BootstrapResult(
            point_estimate=1.0,
            ci_lower=0.8,
            ci_upper=1.2,
            std_error=0.1,
            num_replicates=1000,
            method="BCa_95%"
        )

        assert result.ci_width == pytest.approx(0.4)

    def test_relative_width(self):
        """Test relative CI width calculation."""
        result = BootstrapResult(
            point_estimate=1.0,
            ci_lower=0.8,
            ci_upper=1.2,
            std_error=0.1,
            num_replicates=1000,
            method="BCa_95%"
        )

        assert result.relative_width == pytest.approx(0.4)

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = BootstrapResult(
            point_estimate=1.0,
            ci_lower=0.8,
            ci_upper=1.2,
            std_error=0.1,
            num_replicates=1000,
            method="BCa_95%"
        )

        d = result.to_dict()

        assert d["point_estimate"] == 1.0
        assert d["ci_lower"] == 0.8
        assert d["ci_upper"] == 1.2
        assert d["method"] == "BCa_95%"
        assert "ci_width" in d
        assert "relative_width" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
