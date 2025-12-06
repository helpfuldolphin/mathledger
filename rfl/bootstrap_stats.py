"""
Bootstrap Statistical Methods for RFL Confidence Intervals

Implements bias-corrected and accelerated (BCa) bootstrap for:
- Coverage confidence intervals
- Uplift confidence intervals
- Multi-run metric aggregation

References:
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- DiCiccio & Efron (1996) "Bootstrap Confidence Intervals"
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval calculation."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    num_replicates: int
    method: str

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_width(self) -> float:
        """CI width as fraction of point estimate."""
        if self.point_estimate == 0:
            return np.inf
        return self.ci_width / abs(self.point_estimate)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "point_estimate": float(self.point_estimate),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "std_error": float(self.std_error),
            "num_replicates": self.num_replicates,
            "method": self.method,
            "ci_width": float(self.ci_width),
            "relative_width": float(self.relative_width) if not np.isinf(self.relative_width) else "inf"
        }


def bootstrap_bca(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    num_replicates: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    More accurate than percentile bootstrap when:
    - Statistic is biased (e.g., ratio estimators)
    - Variance is non-constant across parameter space
    - Transformation-respecting intervals needed

    Args:
        data: 1D array of observations (e.g., per-run metrics)
        statistic: Function computing statistic from data
        num_replicates: Number of bootstrap samples (≥1000 recommended)
        confidence_level: CI level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with point estimate and BCa confidence interval

    Algorithm:
        1. Compute point estimate θ̂ = statistic(data)
        2. Generate B bootstrap samples, compute θ̂*_b for each
        3. Compute bias correction z₀ = Φ⁻¹(#{θ̂*_b < θ̂} / B)
        4. Compute acceleration â via jackknife
        5. Adjust percentiles: α₁ = Φ(z₀ + (z₀+z_α)/(1-â(z₀+z_α)))
        6. Return (θ̂*_{α₁}, θ̂*_{α₂}) as CI
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)

    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # Step 1: Point estimate
    theta_hat = statistic(data)

    # Step 2: Bootstrap replicates
    # DETERMINISM: Use seeded RNG state for reproducibility
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    bootstrap_estimates = np.zeros(num_replicates)
    for b in range(num_replicates):
        # Resample with replacement using seeded RNG
        sample_indices = rng.randint(0, n, size=n)
        bootstrap_sample = data[sample_indices]
        bootstrap_estimates[b] = statistic(bootstrap_sample)

    # Check for NaN in bootstrap distribution (e.g., from zero denominators)
    if np.any(np.isnan(bootstrap_estimates)):
        warnings.warn("ABSTAIN: Bootstrap distribution contains NaN values")
        return BootstrapResult(
            point_estimate=theta_hat,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std_error=np.nan,
            num_replicates=0,
            method="ABSTAIN"
        )

    # Standard error from bootstrap distribution
    std_error = np.std(bootstrap_estimates, ddof=1)

    # Step 3: Bias correction z₀
    # Proportion of bootstrap estimates less than point estimate
    prop_less = np.mean(bootstrap_estimates < theta_hat)

    # Handle edge cases (all bootstrap estimates ≥ or ≤ point estimate)
    if prop_less == 0:
        prop_less = 1 / (2 * num_replicates)  # Continuity correction
    elif prop_less == 1:
        prop_less = 1 - 1 / (2 * num_replicates)

    # z₀ = Φ⁻¹(prop_less)
    from scipy.stats import norm
    z0 = norm.ppf(prop_less)

    # Step 4: Acceleration via jackknife
    jackknife_estimates = np.zeros(n)
    for i in range(n):
        # Leave-one-out sample
        # For multi-dimensional data, delete along axis 0 (rows)
        if data.ndim > 1:
            jackknife_sample = np.delete(data, i, axis=0)
        else:
            jackknife_sample = np.delete(data, i)
        jackknife_estimates[i] = statistic(jackknife_sample)

    jackknife_mean = np.mean(jackknife_estimates)
    numerator = np.sum((jackknife_mean - jackknife_estimates) ** 3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_estimates) ** 2) ** 1.5)

    if denominator == 0:
        # No acceleration needed (constant statistic across jackknife)
        acceleration = 0.0
    else:
        acceleration = numerator / denominator

    # Step 5: Adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = norm.ppf(alpha / 2)
    z_alpha_upper = norm.ppf(1 - alpha / 2)

    # BCa-adjusted percentiles
    def bca_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - acceleration * (z0 + z_alpha)

        if abs(denominator) < 1e-10:
            # Acceleration too extreme, fall back to percentile method
            return norm.cdf(z_alpha)

        adjusted_z = z0 + numerator / denominator
        return norm.cdf(adjusted_z)

    p_lower = bca_percentile(z_alpha_lower)
    p_upper = bca_percentile(z_alpha_upper)

    # Clamp to [0, 1]
    p_lower = np.clip(p_lower, 0, 1)
    p_upper = np.clip(p_upper, 0, 1)

    # Step 6: Compute CI from bootstrap distribution
    ci_lower = np.percentile(bootstrap_estimates, 100 * p_lower)
    ci_upper = np.percentile(bootstrap_estimates, 100 * p_upper)

    return BootstrapResult(
        point_estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        num_replicates=num_replicates,
        method=f"BCa_{int(confidence_level*100)}%"
    )


def bootstrap_percentile(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    num_replicates: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Simple percentile bootstrap (fallback for BCa failures).

    Faster but less accurate than BCa. Use when:
    - Statistic is unbiased and symmetric
    - Quick rough estimate needed
    - BCa fails due to numerical issues
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)

    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # Point estimate
    theta_hat = statistic(data)

    # Bootstrap replicates
    # DETERMINISM: Use seeded RNG state for reproducibility
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    bootstrap_estimates = np.zeros(num_replicates)
    for b in range(num_replicates):
        sample_indices = rng.randint(0, n, size=n)
        bootstrap_sample = data[sample_indices]
        bootstrap_estimates[b] = statistic(bootstrap_sample)

    # Standard error
    std_error = np.std(bootstrap_estimates, ddof=1)

    # Percentile CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        num_replicates=num_replicates,
        method=f"Percentile_{int(confidence_level*100)}%"
    )


def compute_uplift_ci(
    baseline_rates: np.ndarray,
    treatment_rates: np.ndarray,
    num_replicates: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap CI for uplift ratio (treatment/baseline).

    Args:
        baseline_rates: Array of baseline metric values (e.g., proofs/hour per run)
        treatment_rates: Array of treatment metric values (same length as baseline)
        num_replicates: Bootstrap samples
        confidence_level: CI level
        random_state: Random seed

    Returns:
        BootstrapResult for uplift ratio with BCa CI
    """
    if len(baseline_rates) != len(treatment_rates):
        raise ValueError(f"Baseline and treatment must have same length: {len(baseline_rates)} vs {len(treatment_rates)}")

    # Paired data: each index is one experimental pair
    n = len(baseline_rates)
    paired_data = np.column_stack([baseline_rates, treatment_rates])

    def uplift_statistic(paired_sample):
        baseline_mean = np.mean(paired_sample[:, 0])
        treatment_mean = np.mean(paired_sample[:, 1])

        if baseline_mean == 0:
            # ABSTAIN: Cannot compute uplift with zero baseline
            return np.nan

        return treatment_mean / baseline_mean

    # Use BCa bootstrap for ratio estimator (biased statistic)
    result = bootstrap_bca(
        paired_data,
        uplift_statistic,
        num_replicates=num_replicates,
        confidence_level=confidence_level,
        random_state=random_state
    )

    # Check for ABSTAIN condition
    if np.isnan(result.point_estimate):
        warnings.warn("ABSTAIN: Zero baseline mean, cannot compute uplift")
        return BootstrapResult(
            point_estimate=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std_error=np.nan,
            num_replicates=0,
            method="ABSTAIN"
        )

    return result


def compute_coverage_ci(
    coverage_rates: np.ndarray,
    num_replicates: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap CI for coverage rate (proportion metric).

    Args:
        coverage_rates: Array of coverage values (0.0-1.0) per run
        num_replicates: Bootstrap samples
        confidence_level: CI level
        random_state: Random seed

    Returns:
        BootstrapResult for mean coverage with BCa CI
    """
    coverage_rates = np.asarray(coverage_rates)

    # Validate coverage in [0, 1]
    if np.any((coverage_rates < 0) | (coverage_rates > 1)):
        raise ValueError(f"Coverage rates must be in [0, 1], got range [{coverage_rates.min()}, {coverage_rates.max()}]")

    def coverage_statistic(data):
        return np.mean(data)

    return bootstrap_bca(
        coverage_rates,
        coverage_statistic,
        num_replicates=num_replicates,
        confidence_level=confidence_level,
        random_state=random_state
    )


def verify_metabolism(
    coverage_result: BootstrapResult,
    uplift_result: BootstrapResult,
    coverage_threshold: float = 0.92,
    uplift_threshold: float = 1.0
) -> Tuple[bool, str]:
    """
    Verify reflexive metabolism passes acceptance criteria.

    Acceptance Criteria:
    - Coverage CI lower bound ≥ 0.92
    - Uplift CI lower bound > 1.0

    Args:
        coverage_result: Bootstrap result for coverage
        uplift_result: Bootstrap result for uplift
        coverage_threshold: Minimum acceptable coverage (default 0.92)
        uplift_threshold: Minimum acceptable uplift (default 1.0)

    Returns:
        (pass: bool, message: str)
    """
    # Check for abstentions
    if coverage_result.method == "ABSTAIN":
        return False, "[ABSTAIN] Coverage computation failed - insufficient data"

    if uplift_result.method == "ABSTAIN":
        return False, "[ABSTAIN] Uplift computation failed - zero baseline"

    # Verify coverage
    coverage_pass = coverage_result.ci_lower >= coverage_threshold
    coverage_msg = (
        f"coverage={coverage_result.point_estimate:.4f} "
        f"CI=[{coverage_result.ci_lower:.4f}, {coverage_result.ci_upper:.4f}]"
    )

    # Verify uplift
    uplift_pass = uplift_result.ci_lower > uplift_threshold
    uplift_msg = (
        f"uplift={uplift_result.point_estimate:.4f} "
        f"CI=[{uplift_result.ci_lower:.4f}, {uplift_result.ci_upper:.4f}]"
    )

    # Overall verdict
    if coverage_pass and uplift_pass:
        return True, f"[PASS] Reflexive Metabolism Verified {coverage_msg} {uplift_msg}"
    else:
        failures = []
        if not coverage_pass:
            failures.append(f"coverage CI lower {coverage_result.ci_lower:.4f} < {coverage_threshold}")
        if not uplift_pass:
            failures.append(f"uplift CI lower {uplift_result.ci_lower:.4f} ≤ {uplift_threshold}")

        return False, f"[FAIL] {'; '.join(failures)}. Observed: {coverage_msg} {uplift_msg}"


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate 40 runs with coverage rates
    coverage_data = np.random.beta(20, 2, size=40)  # Mean ~0.909
    coverage_ci = compute_coverage_ci(coverage_data, random_state=42)

    print("Coverage Bootstrap CI:")
    print(f"  Point estimate: {coverage_ci.point_estimate:.4f}")
    print(f"  95% CI: [{coverage_ci.ci_lower:.4f}, {coverage_ci.ci_upper:.4f}]")
    print(f"  Std error: {coverage_ci.std_error:.4f}")
    print()

    # Simulate uplift (baseline vs treatment)
    baseline_throughput = np.random.poisson(100, size=40)
    treatment_throughput = np.random.poisson(130, size=40)  # 30% uplift
    uplift_ci = compute_uplift_ci(baseline_throughput, treatment_throughput, random_state=42)

    print("Uplift Bootstrap CI:")
    print(f"  Point estimate: {uplift_ci.point_estimate:.4f}")
    print(f"  95% CI: [{uplift_ci.ci_lower:.4f}, {uplift_ci.ci_upper:.4f}]")
    print(f"  Std error: {uplift_ci.std_error:.4f}")
    print()

    # Verify metabolism
    passed, message = verify_metabolism(coverage_ci, uplift_ci)
    print(message)
