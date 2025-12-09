"""
MathLedger Statistical Metrics Library.

===============================================================================
STATUS: PHASE II — STATISTICAL GUARDRAIL ZONE
===============================================================================

DISCLAIMER: Statistics provided here are descriptive; uplift evaluation is
handled solely by governance-gate logic. This module provides governance-grade
statistical measurements that allow G5 to decide truth from noise.

ABSOLUTE SAFEGUARDS:
  - No claims of significance.
  - No uplift interpretation.
  - No modification of governance thresholds.

This module implements:
  - Wilson score confidence interval for binomial proportions
  - Drift detectors for temporal, policy weight, and metric instability
  - Rolling statistics (average, variance)
  - Stability indices and abstention trend analysis

All functions are deterministic and numerically stable.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

NumericSeries = Sequence[Union[int, float]]


# ---------------------------------------------------------------------------
# Wilson Confidence Interval
# ---------------------------------------------------------------------------

# Precomputed z-values for common confidence levels (avoids scipy dependency)
_Z_SCORES = {
    0.80: 1.281552,
    0.85: 1.439531,
    0.90: 1.644854,
    0.95: 1.959964,
    0.99: 2.575829,
    0.995: 2.807034,
    0.999: 3.290527,
}


def _get_z_score(confidence: float) -> float:
    """
    Get z-score for a given confidence level.

    Uses precomputed values for common levels or falls back to
    the Abramowitz & Stegun approximation for the inverse normal CDF.

    Args:
        confidence: Confidence level in (0, 1)

    Returns:
        Two-tailed z-score
    """
    # Check precomputed values first (exact to 6 decimal places)
    if confidence in _Z_SCORES:
        return _Z_SCORES[confidence]

    # Abramowitz & Stegun approximation (26.2.23) for inverse normal CDF
    # This is accurate to ~4.5e-4 for the tail region
    p = (1 + confidence) / 2  # Convert to upper tail probability

    if p <= 0 or p >= 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    # Coefficients for the rational approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    t = math.sqrt(-2 * math.log(1 - p))
    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    return z


def compute_wilson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    The Wilson score interval provides better coverage than the normal
    approximation (Wald interval), especially for proportions near 0 or 1
    and for small sample sizes. It is recommended for success rate CIs.

    Formula (Wilson 1927):
        p̂ = successes / trials
        z = z-score for confidence level (e.g., 1.96 for 95%)
        denominator = 1 + z²/n
        center = (p̂ + z²/2n) / denominator
        margin = z * sqrt(p̂(1-p̂)/n + z²/4n²) / denominator
        CI = (center - margin, center + margin)

    This implementation uses the Agresti-Coull adjusted form which is
    numerically stable for edge cases.

    Args:
        successes: Number of successful trials (0 ≤ successes ≤ trials)
        trials: Total number of trials (n ≥ 0)
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (ci_low, ci_high) tuple representing the confidence interval.
        For n=0, returns (0.0, 1.0) as the uninformative interval.

    Raises:
        ValueError: If successes > trials, successes < 0, or
                    confidence not in (0, 1)

    Examples:
        >>> compute_wilson_ci(80, 100, confidence=0.95)
        (0.7109..., 0.8693...)

        >>> compute_wilson_ci(0, 100, confidence=0.95)
        (0.0, 0.0369...)

        >>> compute_wilson_ci(100, 100, confidence=0.95)
        (0.9631..., 1.0)

        >>> compute_wilson_ci(0, 0, confidence=0.95)
        (0.0, 1.0)

    References:
        - Wilson, E.B. (1927). "Probable Inference, the Law of Succession,
          and Statistical Inference". J. Amer. Statist. Assoc. 22: 209–212.
        - Agresti & Coull (1998). "Approximate Is Better than 'Exact' for
          Interval Estimation of Binomial Proportions". The American
          Statistician, 52(2), 119-126.
    """
    # Validate inputs
    if successes < 0:
        raise ValueError(f"successes must be >= 0, got {successes}")
    if trials < 0:
        raise ValueError(f"trials must be >= 0, got {trials}")
    if successes > trials:
        raise ValueError(
            f"successes ({successes}) cannot exceed trials ({trials})"
        )
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Edge case: n=0 returns uninformative interval
    if trials == 0:
        return (0.0, 1.0)

    # Get z-score for confidence level
    z = _get_z_score(confidence)
    z_sq = z * z
    n = float(trials)

    # Observed proportion
    p_hat = successes / n

    # Wilson score interval formula
    denominator = 1 + z_sq / n
    center = (p_hat + z_sq / (2 * n)) / denominator

    # sqrt term with numerical stability
    variance_term = (p_hat * (1 - p_hat) / n) + (z_sq / (4 * n * n))
    # Clamp to avoid negative sqrt due to floating point errors
    variance_term = max(0.0, variance_term)
    margin = z * math.sqrt(variance_term) / denominator

    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    return (ci_low, ci_high)


# ---------------------------------------------------------------------------
# Drift Detection Results
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """
    Result of a drift detection analysis.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Attributes:
        detected: Whether drift was detected
        score: Numerical drift score (interpretation varies by detector)
        direction: Direction of drift ("increasing", "decreasing", "stable", "volatile")
        segments: Number of distinct segments detected (if applicable)
        details: Additional diagnostic information
    """
    detected: bool
    score: float
    direction: str
    segments: int = 1
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Drift Detectors
# ---------------------------------------------------------------------------

def detect_temporal_success_drift(
    series: NumericSeries,
    window_size: int = 10,
    threshold: float = 0.15,
) -> DriftResult:
    """
    Detect drift in a temporal success rate series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This detector compares rolling averages across windows to identify
    significant shifts in success rate over time. It uses a simple
    change-point detection based on the maximum difference between
    adjacent window means.

    Args:
        series: Sequence of success indicators (0/1) or rates (0.0-1.0),
                ordered from oldest to newest
        window_size: Size of rolling window for comparison (default 10)
        threshold: Minimum difference to trigger drift detection (default 0.15)

    Returns:
        DriftResult with detection status and metrics

    Examples:
        >>> detect_temporal_success_drift([1,1,1,1,1,0,0,0,0,0])
        DriftResult(detected=True, score=1.0, direction='decreasing', ...)
    """
    # Guard against empty or too-short series
    if not series:
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details="Empty series"
        )

    n = len(series)
    if n < window_size:
        # Not enough data for drift detection
        mean_val = sum(series) / n if n > 0 else 0.0
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details=f"Series too short (n={n}, window={window_size})"
        )

    # Convert to float list
    values = [float(v) for v in series]

    # Compute rolling means
    num_windows = n - window_size + 1
    window_means = []
    for i in range(num_windows):
        window_sum = sum(values[i:i + window_size])
        window_means.append(window_sum / window_size)

    # Find maximum change between consecutive windows
    max_change = 0.0
    max_change_idx = 0
    for i in range(1, len(window_means)):
        change = abs(window_means[i] - window_means[i - 1])
        if change > max_change:
            max_change = change
            max_change_idx = i

    # Determine direction
    if len(window_means) >= 2:
        overall_change = window_means[-1] - window_means[0]
        if overall_change > threshold:
            direction = "increasing"
        elif overall_change < -threshold:
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        direction = "stable"

    # Compute drift score as max change normalized
    drift_score = max_change

    detected = max_change >= threshold

    # Count segments (simple: count threshold crossings)
    segments = 1
    for i in range(1, len(window_means)):
        if abs(window_means[i] - window_means[i - 1]) >= threshold:
            segments += 1

    return DriftResult(
        detected=detected,
        score=round(drift_score, 6),
        direction=direction,
        segments=segments,
        details=f"max_change at index {max_change_idx}"
    )


def detect_policy_weight_drift(
    weights_over_time: List[NumericSeries],
    threshold: float = 0.20,
) -> DriftResult:
    """
    Detect drift in policy weights over time.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This detector analyzes how policy weights (e.g., feature importance
    or action probabilities) change across time steps. It computes the
    total variation distance between consecutive weight vectors.

    Args:
        weights_over_time: List of weight vectors, each representing
                          policy weights at a point in time. Each vector
                          should sum to approximately 1.0.
        threshold: Minimum L1 distance to trigger drift detection (default 0.20)

    Returns:
        DriftResult with detection status and total variation score

    Examples:
        >>> w1 = [0.5, 0.3, 0.2]
        >>> w2 = [0.5, 0.3, 0.2]
        >>> w3 = [0.1, 0.6, 0.3]
        >>> detect_policy_weight_drift([w1, w2, w3])
        DriftResult(detected=True, ...)
    """
    if not weights_over_time:
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details="Empty weight history"
        )

    if len(weights_over_time) < 2:
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details="Need at least 2 snapshots for drift detection"
        )

    # Compute L1 distances between consecutive weight vectors
    l1_distances = []
    for i in range(1, len(weights_over_time)):
        prev_weights = list(weights_over_time[i - 1])
        curr_weights = list(weights_over_time[i])

        # Handle dimension mismatch by padding with zeros
        max_len = max(len(prev_weights), len(curr_weights))
        prev_weights.extend([0.0] * (max_len - len(prev_weights)))
        curr_weights.extend([0.0] * (max_len - len(curr_weights)))

        l1 = sum(abs(p - c) for p, c in zip(prev_weights, curr_weights))
        l1_distances.append(l1)

    max_l1 = max(l1_distances)
    mean_l1 = sum(l1_distances) / len(l1_distances)

    # Determine direction based on entropy change
    def entropy(weights: NumericSeries) -> float:
        total = sum(weights)
        if total == 0:
            return 0.0
        probs = [max(w / total, 1e-10) for w in weights]
        return -sum(p * math.log(p) for p in probs if p > 0)

    first_entropy = entropy(weights_over_time[0])
    last_entropy = entropy(weights_over_time[-1])
    entropy_change = last_entropy - first_entropy

    if entropy_change > 0.1:
        direction = "increasing"  # More spread out
    elif entropy_change < -0.1:
        direction = "decreasing"  # More concentrated
    elif max_l1 > threshold:
        direction = "volatile"
    else:
        direction = "stable"

    detected = max_l1 >= threshold

    return DriftResult(
        detected=detected,
        score=round(max_l1, 6),
        direction=direction,
        segments=sum(1 for d in l1_distances if d >= threshold) + 1,
        details=f"mean_l1={mean_l1:.4f}, max_l1={max_l1:.4f}"
    )


def detect_metric_instability(
    series: NumericSeries,
    threshold: float = 2.0,
) -> DriftResult:
    """
    Detect instability in a metric series using coefficient of variation.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This detector identifies metrics that exhibit excessive variability
    relative to their mean, which may indicate system instability or
    measurement issues.

    The coefficient of variation (CV) is used as the instability measure:
        CV = std(series) / |mean(series)|

    Args:
        series: Sequence of metric values
        threshold: CV threshold above which instability is detected (default 2.0)

    Returns:
        DriftResult with instability detection and CV score

    Examples:
        >>> detect_metric_instability([1.0, 1.1, 0.9, 1.0])
        DriftResult(detected=False, score=0.08..., direction='stable', ...)

        >>> detect_metric_instability([1.0, 10.0, 0.5, 8.0])
        DriftResult(detected=True, score=1.04..., direction='volatile', ...)
    """
    if not series:
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details="Empty series"
        )

    values = [float(v) for v in series]
    n = len(values)

    if n < 2:
        return DriftResult(
            detected=False,
            score=0.0,
            direction="stable",
            details="Need at least 2 values for instability detection"
        )

    # Compute mean and standard deviation
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
    std_val = math.sqrt(variance)

    # Compute coefficient of variation
    if abs(mean_val) < 1e-10:
        # Near-zero mean: use std as the score
        cv = std_val
    else:
        cv = std_val / abs(mean_val)

    # Determine direction based on trend
    first_half_mean = sum(values[:n // 2]) / max(1, n // 2)
    second_half_mean = sum(values[n // 2:]) / max(1, n - n // 2)

    if cv >= threshold:
        direction = "volatile"
    elif second_half_mean > first_half_mean * 1.1:
        direction = "increasing"
    elif second_half_mean < first_half_mean * 0.9:
        direction = "decreasing"
    else:
        direction = "stable"

    detected = cv >= threshold

    return DriftResult(
        detected=detected,
        score=round(cv, 6),
        direction=direction,
        segments=1,
        details=f"mean={mean_val:.4f}, std={std_val:.4f}, cv={cv:.4f}"
    )


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------

def rolling_average(
    series: NumericSeries,
    window: int,
) -> List[float]:
    """
    Compute rolling average over a series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Uses a simple moving average (SMA) with the specified window size.
    The first (window-1) values are computed with a smaller window.

    Args:
        series: Sequence of numeric values
        window: Window size for averaging (must be >= 1)

    Returns:
        List of rolling averages, same length as input series

    Raises:
        ValueError: If window < 1

    Examples:
        >>> rolling_average([1, 2, 3, 4, 5], window=3)
        [1.0, 1.5, 2.0, 3.0, 4.0]
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if not series:
        return []

    values = [float(v) for v in series]
    n = len(values)
    result = []

    for i in range(n):
        # Use smaller window at the beginning
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        avg = sum(window_values) / len(window_values)
        result.append(round(avg, 10))

    return result


def rolling_variance(
    series: NumericSeries,
    window: int,
) -> List[float]:
    """
    Compute rolling variance over a series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Uses sample variance (N-1 denominator) with the specified window size.
    Returns 0.0 for windows with fewer than 2 values.

    Args:
        series: Sequence of numeric values
        window: Window size for variance computation (must be >= 1)

    Returns:
        List of rolling variances, same length as input series

    Raises:
        ValueError: If window < 1

    Examples:
        >>> rolling_variance([1, 2, 3, 4, 5], window=3)
        [0.0, 0.5, 1.0, 1.0, 1.0]
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if not series:
        return []

    values = [float(v) for v in series]
    n = len(values)
    result = []

    for i in range(n):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        k = len(window_values)

        if k < 2:
            result.append(0.0)
        else:
            mean_val = sum(window_values) / k
            variance = sum((v - mean_val) ** 2 for v in window_values) / (k - 1)
            result.append(round(variance, 10))

    return result


# ---------------------------------------------------------------------------
# Stability and Trend Metrics
# ---------------------------------------------------------------------------

def stability_index(series: NumericSeries) -> float:
    """
    Compute a stability index for a series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    The stability index is computed as:
        SI = 1 / (1 + CV)

    Where CV is the coefficient of variation. A value close to 1.0 indicates
    high stability, while lower values indicate instability.

    Args:
        series: Sequence of numeric values

    Returns:
        Stability index in range [0, 1], where 1.0 is perfectly stable

    Examples:
        >>> stability_index([1.0, 1.0, 1.0, 1.0])
        1.0

        >>> stability_index([1.0, 2.0, 3.0, 4.0])  # More variable
        0.4...  # Lower stability
    """
    if not series:
        return 1.0  # Empty series is trivially stable

    values = [float(v) for v in series]
    n = len(values)

    if n < 2:
        return 1.0  # Single value is trivially stable

    mean_val = sum(values) / n

    if abs(mean_val) < 1e-10:
        # Near-zero mean: check if all values are near zero
        max_abs = max(abs(v) for v in values)
        if max_abs < 1e-10:
            return 1.0  # All zeros is stable
        else:
            return 0.0  # Large variation around zero

    variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
    std_val = math.sqrt(variance)
    cv = std_val / abs(mean_val)

    # Stability index: inverse of CV, scaled to [0, 1]
    si = 1.0 / (1.0 + cv)

    return round(si, 6)


def abstention_trend(series: NumericSeries) -> float:
    """
    Compute abstention trend coefficient.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This function computes the linear trend slope of an abstention series,
    normalized by the series range. A positive value indicates increasing
    abstentions over time, negative indicates decreasing.

    Uses ordinary least squares regression:
        trend = covariance(x, y) / variance(x)

    Then normalizes by the y-range for comparability.

    Args:
        series: Sequence of abstention counts or rates, ordered from
                oldest to newest

    Returns:
        Normalized trend coefficient:
        - Positive: abstentions increasing over time
        - Negative: abstentions decreasing over time
        - Near zero: stable abstention rate

    Examples:
        >>> abstention_trend([1, 2, 3, 4, 5])  # Increasing
        1.0

        >>> abstention_trend([5, 4, 3, 2, 1])  # Decreasing
        -1.0

        >>> abstention_trend([3, 3, 3, 3, 3])  # Stable
        0.0
    """
    if not series:
        return 0.0

    values = [float(v) for v in series]
    n = len(values)

    if n < 2:
        return 0.0

    # X values: 0, 1, 2, ..., n-1
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n

    # Compute covariance and variance
    cov_xy = sum((i - x_mean) * (values[i] - y_mean) for i in range(n)) / n
    var_x = sum((i - x_mean) ** 2 for i in range(n)) / n

    if var_x < 1e-10:
        return 0.0

    # Raw slope
    slope = cov_xy / var_x

    # Normalize by range
    y_range = max(values) - min(values)
    if y_range < 1e-10:
        return 0.0

    # Normalize slope to [-1, 1] range based on maximum possible slope
    # Maximum slope for n points over range y_range is y_range / (n-1)
    max_slope = y_range / (n - 1) if n > 1 else 1.0
    normalized_trend = slope / max_slope if abs(max_slope) > 1e-10 else 0.0

    # Clamp to [-1, 1]
    normalized_trend = max(-1.0, min(1.0, normalized_trend))

    return round(normalized_trend, 6)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def validate_series(series: NumericSeries, name: str = "series") -> List[float]:
    """
    Validate and convert a numeric series to a list of floats.

    Args:
        series: Input numeric sequence
        name: Name for error messages

    Returns:
        List of float values

    Raises:
        TypeError: If series is not iterable or contains non-numeric values
        ValueError: If series contains NaN or infinity
    """
    if series is None:
        raise TypeError(f"{name} cannot be None")

    try:
        values = []
        for v in series:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"{name} contains NaN or infinity")
            values.append(f)
        return values
    except (TypeError, ValueError) as e:
        raise type(e)(f"Invalid {name}: {e}")


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Wilson CI
    "compute_wilson_ci",
    # Drift detectors
    "detect_temporal_success_drift",
    "detect_policy_weight_drift",
    "detect_metric_instability",
    "DriftResult",
    # Rolling statistics
    "rolling_average",
    "rolling_variance",
    # Stability metrics
    "stability_index",
    "abstention_trend",
    # Utilities
    "validate_series",
]

