"""
MathLedger Multi-Metric Drift Alignment Engine.

===============================================================================
STATUS: PHASE II — METRICS BUREAU (D5) — DRIFT ENGINE CONSOLIDATION
===============================================================================

DISCLAIMER: Statistics provided here are descriptive; uplift evaluation is
handled solely by governance-gate logic. This module computes structural
drift metrics across multiple measurement dimensions.

ABSOLUTE SAFEGUARDS:
  - No governance/promotion logic.
  - No significance testing.
  - No uplift computation.
  - No interpretation of drift.

This module implements:
  - Multi-metric drift alignment signatures
  - Stability envelope generation using rolling variance + Wilson CI
  - Cross-metric trajectory analysis

All functions are deterministic and produce only descriptive structural metrics.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from backend.metrics.statistical import (
    NumericSeries,
    DriftResult,
    compute_wilson_ci,
    detect_temporal_success_drift,
    detect_metric_instability,
    rolling_average,
    rolling_variance,
    stability_index,
    abstention_trend,
)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

MetricTrajectory = Dict[str, NumericSeries]


# ---------------------------------------------------------------------------
# Drift Alignment Result Structures
# ---------------------------------------------------------------------------

@dataclass
class MetricDriftSignature:
    """
    Drift signature for a single metric.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Attributes:
        metric_name: Name of the metric
        drift_score: Raw drift score from detector
        direction: Direction of drift ("increasing", "decreasing", "stable", "volatile")
        stability: Stability index [0, 1]
        trend: Normalized trend coefficient [-1, 1]
        monotonicity: Monotonicity score [0, 1] where 1 is perfectly monotonic
    """
    metric_name: str
    drift_score: float
    direction: str
    stability: float
    trend: float
    monotonicity: float


@dataclass
class DriftAlignmentResult:
    """
    Result of multi-metric drift alignment analysis.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Attributes:
        slice: Identifier for the slice being analyzed
        drift_alignment_score: Aggregate alignment score across all metrics [0, 1]
        metrics: Individual metric signatures
        coherence_score: How aligned the metrics are in direction [0, 1]
        trajectory_correlation: Pairwise correlation summary
        metadata: Additional diagnostic information
    """
    slice: str
    drift_alignment_score: float
    metrics: Dict[str, MetricDriftSignature]
    coherence_score: float = 0.0
    trajectory_correlation: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "slice": self.slice,
            "drift_alignment_score": self.drift_alignment_score,
            "metrics": {
                name: {
                    "drift_score": sig.drift_score,
                    "direction": sig.direction,
                    "stability": sig.stability,
                    "trend": sig.trend,
                    "monotonicity": sig.monotonicity,
                }
                for name, sig in self.metrics.items()
            },
            "coherence_score": self.coherence_score,
            "trajectory_correlation": self.trajectory_correlation,
            "metadata": self.metadata,
        }


@dataclass
class StabilityEnvelope:
    """
    Stability envelope with confidence bands.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Attributes:
        metric_name: Name of the metric
        center_line: Rolling average (center of envelope)
        upper_band: Upper confidence band
        lower_band: Lower confidence band
        variance_band: Rolling variance values
        confidence_level: Confidence level used for bands
        envelope_width: Mean width of envelope
        containment_ratio: Proportion of points within envelope
    """
    metric_name: str
    center_line: List[float]
    upper_band: List[float]
    lower_band: List[float]
    variance_band: List[float]
    confidence_level: float
    envelope_width: float
    containment_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "center_line": self.center_line,
            "upper_band": self.upper_band,
            "lower_band": self.lower_band,
            "variance_band": self.variance_band,
            "confidence_level": self.confidence_level,
            "envelope_width": self.envelope_width,
            "containment_ratio": self.containment_ratio,
        }


# ---------------------------------------------------------------------------
# Monotonicity Computation
# ---------------------------------------------------------------------------

def compute_monotonicity(series: NumericSeries) -> float:
    """
    Compute monotonicity score for a series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Monotonicity measures how consistently a series moves in one direction.
    A score of 1.0 means perfectly monotonic (all increasing or all decreasing).
    A score of 0.0 means no consistent direction.

    Formula:
        monotonicity = |increasing_pairs - decreasing_pairs| / total_pairs

    Args:
        series: Sequence of numeric values

    Returns:
        Monotonicity score in [0, 1]

    Examples:
        >>> compute_monotonicity([1, 2, 3, 4, 5])  # Perfectly increasing
        1.0
        >>> compute_monotonicity([5, 4, 3, 2, 1])  # Perfectly decreasing
        1.0
        >>> compute_monotonicity([1, 2, 1, 2, 1])  # Alternating
        0.0
    """
    if not series or len(series) < 2:
        return 1.0  # Trivially monotonic

    values = [float(v) for v in series]
    n = len(values)

    increasing = 0
    decreasing = 0
    ties = 0

    for i in range(1, n):
        diff = values[i] - values[i - 1]
        if diff > 1e-10:
            increasing += 1
        elif diff < -1e-10:
            decreasing += 1
        else:
            ties += 1

    total_comparisons = n - 1 - ties
    if total_comparisons == 0:
        return 1.0  # All ties means constant series

    monotonicity = abs(increasing - decreasing) / total_comparisons
    return round(monotonicity, 6)


# ---------------------------------------------------------------------------
# Cross-Metric Correlation
# ---------------------------------------------------------------------------

def compute_pearson_correlation(
    series_a: NumericSeries,
    series_b: NumericSeries,
) -> float:
    """
    Compute Pearson correlation coefficient between two series.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        series_a: First numeric series
        series_b: Second numeric series (must be same length)

    Returns:
        Correlation coefficient in [-1, 1], or 0.0 if undefined

    Raises:
        ValueError: If series have different lengths
    """
    if len(series_a) != len(series_b):
        raise ValueError(
            f"Series must have same length: {len(series_a)} vs {len(series_b)}"
        )

    if len(series_a) < 2:
        return 0.0

    a = [float(v) for v in series_a]
    b = [float(v) for v in series_b]
    n = len(a)

    mean_a = sum(a) / n
    mean_b = sum(b) / n

    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    var_a = sum((v - mean_a) ** 2 for v in a) / n
    var_b = sum((v - mean_b) ** 2 for v in b) / n

    if var_a < 1e-10 or var_b < 1e-10:
        return 0.0  # Undefined for constant series

    std_a = math.sqrt(var_a)
    std_b = math.sqrt(var_b)

    correlation = cov / (std_a * std_b)
    return round(max(-1.0, min(1.0, correlation)), 6)


# ---------------------------------------------------------------------------
# Multi-Metric Drift Alignment Engine
# ---------------------------------------------------------------------------

def compute_drift_alignment(
    slice_id: str,
    trajectories: MetricTrajectory,
    window_size: int = 10,
    drift_threshold: float = 0.10,
) -> DriftAlignmentResult:
    """
    Compute drift alignment signature across multiple metrics.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This function analyzes drift patterns across multiple metric trajectories
    and computes an aggregate alignment score indicating how coherently the
    metrics are moving together.

    Expected metric keys (any subset supported):
        - "success": Success rate series
        - "abstention": Abstention rate series
        - "depth": Chain depth series
        - "entropy": Candidate entropy series
        - Any custom metric_value trajectories

    Args:
        slice_id: Identifier for the slice being analyzed
        trajectories: Dict mapping metric names to their time series
        window_size: Rolling window size for drift detection (default 10)
        drift_threshold: Threshold for drift detection (default 0.10)

    Returns:
        DriftAlignmentResult with alignment score and per-metric signatures

    Examples:
        >>> trajectories = {
        ...     "success": [0.8, 0.75, 0.7, 0.65, 0.6],
        ...     "abstention": [0.1, 0.12, 0.15, 0.18, 0.2],
        ... }
        >>> result = compute_drift_alignment("slice_uplift_goal", trajectories)
        >>> result.drift_alignment_score
        0.731...
    """
    if not trajectories:
        return DriftAlignmentResult(
            slice=slice_id,
            drift_alignment_score=0.0,
            metrics={},
            coherence_score=0.0,
            metadata={"error": "No trajectories provided"},
        )

    # Compute individual metric signatures
    metric_signatures: Dict[str, MetricDriftSignature] = {}
    directions: List[str] = []
    trends: List[float] = []
    stabilities: List[float] = []

    for metric_name, series in trajectories.items():
        if not series or len(series) < 2:
            # Insufficient data for this metric
            metric_signatures[metric_name] = MetricDriftSignature(
                metric_name=metric_name,
                drift_score=0.0,
                direction="stable",
                stability=1.0,
                trend=0.0,
                monotonicity=1.0,
            )
            continue

        # Detect drift
        drift_result = detect_temporal_success_drift(
            series,
            window_size=min(window_size, len(series)),
            threshold=drift_threshold,
        )

        # Compute stability
        stab = stability_index(series)

        # Compute trend
        trend_val = abstention_trend(series)

        # Compute monotonicity
        mono = compute_monotonicity(series)

        signature = MetricDriftSignature(
            metric_name=metric_name,
            drift_score=drift_result.score,
            direction=drift_result.direction,
            stability=stab,
            trend=trend_val,
            monotonicity=mono,
        )
        metric_signatures[metric_name] = signature

        directions.append(drift_result.direction)
        trends.append(trend_val)
        stabilities.append(stab)

    # Compute coherence score (how aligned are the directions?)
    coherence_score = _compute_direction_coherence(directions)

    # Compute trajectory correlations (pairwise)
    trajectory_correlation = _compute_pairwise_correlations(trajectories)

    # Compute aggregate drift alignment score
    drift_alignment_score = _compute_alignment_score(
        stabilities, trends, coherence_score, trajectory_correlation
    )

    return DriftAlignmentResult(
        slice=slice_id,
        drift_alignment_score=drift_alignment_score,
        metrics=metric_signatures,
        coherence_score=coherence_score,
        trajectory_correlation=trajectory_correlation,
        metadata={
            "window_size": window_size,
            "drift_threshold": drift_threshold,
            "metric_count": len(metric_signatures),
        },
    )


def _compute_direction_coherence(directions: List[str]) -> float:
    """
    Compute coherence score based on direction alignment.

    Returns 1.0 if all directions agree, 0.0 if maximally divergent.
    """
    if not directions:
        return 0.0

    # Count direction frequencies
    direction_counts: Dict[str, int] = {}
    for d in directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1

    # Coherence is the proportion of the dominant direction
    max_count = max(direction_counts.values())
    coherence = max_count / len(directions)

    return round(coherence, 6)


def _compute_pairwise_correlations(
    trajectories: MetricTrajectory,
) -> Dict[str, float]:
    """
    Compute pairwise Pearson correlations between trajectories.

    Returns dict with keys like "success_abstention" -> correlation value.
    """
    correlations: Dict[str, float] = {}
    metric_names = list(trajectories.keys())

    for i, name_a in enumerate(metric_names):
        for name_b in metric_names[i + 1:]:
            series_a = trajectories[name_a]
            series_b = trajectories[name_b]

            # Align lengths by truncation
            min_len = min(len(series_a), len(series_b))
            if min_len < 2:
                correlations[f"{name_a}_{name_b}"] = 0.0
                continue

            corr = compute_pearson_correlation(
                list(series_a)[:min_len],
                list(series_b)[:min_len],
            )
            correlations[f"{name_a}_{name_b}"] = corr

    return correlations


def _compute_alignment_score(
    stabilities: List[float],
    trends: List[float],
    coherence: float,
    correlations: Dict[str, float],
) -> float:
    """
    Compute aggregate drift alignment score.

    Score components:
        - Mean stability (weight 0.3)
        - Trend alignment (weight 0.2)
        - Direction coherence (weight 0.3)
        - Mean absolute correlation (weight 0.2)

    Returns score in [0, 1].
    """
    if not stabilities:
        return 0.0

    # Mean stability
    mean_stability = sum(stabilities) / len(stabilities)

    # Trend alignment: how similar are the trend directions?
    if len(trends) >= 2:
        # Standard deviation of trends normalized
        trend_mean = sum(trends) / len(trends)
        trend_var = sum((t - trend_mean) ** 2 for t in trends) / len(trends)
        trend_std = math.sqrt(trend_var)
        # Convert to alignment score (lower variance = higher alignment)
        trend_alignment = 1.0 / (1.0 + trend_std)
    else:
        trend_alignment = 1.0

    # Mean absolute correlation
    if correlations:
        mean_abs_corr = sum(abs(c) for c in correlations.values()) / len(correlations)
    else:
        mean_abs_corr = 0.0

    # Weighted aggregate
    score = (
        0.3 * mean_stability +
        0.2 * trend_alignment +
        0.3 * coherence +
        0.2 * mean_abs_corr
    )

    return round(score, 6)


# ---------------------------------------------------------------------------
# Stability Envelope Generator
# ---------------------------------------------------------------------------

def generate_stability_envelope(
    series: NumericSeries,
    metric_name: str = "metric",
    window: int = 10,
    confidence: float = 0.95,
) -> StabilityEnvelope:
    """
    Generate stability envelope with confidence bands.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This function combines rolling variance with Wilson CI to create
    descriptive envelope bands around a metric trajectory.

    The envelope is constructed as:
        - Center line: Rolling average
        - Variance band: Rolling variance
        - Upper/Lower bands: Center ± z * sqrt(variance)

    For rate metrics (values in [0, 1]), Wilson CI bounds are also
    computed and the tighter bound is used.

    Args:
        series: Sequence of metric values
        metric_name: Name for the metric
        window: Rolling window size (default 10)
        confidence: Confidence level for bands (default 0.95)

    Returns:
        StabilityEnvelope with center line and confidence bands

    Examples:
        >>> envelope = generate_stability_envelope(
        ...     [0.8, 0.75, 0.82, 0.78, 0.80],
        ...     metric_name="success_rate",
        ...     window=3,
        ... )
        >>> len(envelope.center_line)
        5
    """
    if not series:
        return StabilityEnvelope(
            metric_name=metric_name,
            center_line=[],
            upper_band=[],
            lower_band=[],
            variance_band=[],
            confidence_level=confidence,
            envelope_width=0.0,
            containment_ratio=1.0,
        )

    values = [float(v) for v in series]
    n = len(values)

    # Compute rolling statistics
    center_line = rolling_average(values, window)
    variance_band = rolling_variance(values, window)

    # Get z-score for confidence level
    from backend.metrics.statistical import _get_z_score
    z = _get_z_score(confidence)

    # Compute confidence bands
    upper_band: List[float] = []
    lower_band: List[float] = []

    # Check if this looks like a rate metric (values in [0, 1])
    is_rate_metric = all(0.0 <= v <= 1.0 for v in values)

    for i in range(n):
        center = center_line[i]
        var = variance_band[i]
        std = math.sqrt(var) if var > 0 else 0.0

        # Standard confidence band
        margin = z * std
        upper = center + margin
        lower = center - margin

        # For rate metrics, also consider Wilson CI
        if is_rate_metric and i >= window - 1:
            # Estimate successes from rate and window
            window_start = max(0, i - window + 1)
            window_values = values[window_start:i + 1]
            k = len(window_values)
            if k > 0:
                successes = int(round(sum(window_values)))
                trials = k
                wilson_low, wilson_high = compute_wilson_ci(
                    min(successes, trials), trials, confidence
                )
                # Use tighter bounds
                upper = min(upper, wilson_high)
                lower = max(lower, wilson_low)

        # Clamp to valid range if rate metric
        if is_rate_metric:
            upper = min(1.0, upper)
            lower = max(0.0, lower)

        upper_band.append(round(upper, 10))
        lower_band.append(round(lower, 10))

    # Compute envelope width (mean width)
    widths = [upper_band[i] - lower_band[i] for i in range(n)]
    envelope_width = sum(widths) / n if n > 0 else 0.0

    # Compute containment ratio (proportion of points within envelope)
    contained = sum(
        1 for i in range(n)
        if lower_band[i] <= values[i] <= upper_band[i]
    )
    containment_ratio = contained / n if n > 0 else 1.0

    return StabilityEnvelope(
        metric_name=metric_name,
        center_line=center_line,
        upper_band=upper_band,
        lower_band=lower_band,
        variance_band=variance_band,
        confidence_level=confidence,
        envelope_width=round(envelope_width, 6),
        containment_ratio=round(containment_ratio, 6),
    )


def generate_multi_metric_envelopes(
    trajectories: MetricTrajectory,
    window: int = 10,
    confidence: float = 0.95,
) -> Dict[str, StabilityEnvelope]:
    """
    Generate stability envelopes for multiple metrics.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        trajectories: Dict mapping metric names to their time series
        window: Rolling window size (default 10)
        confidence: Confidence level for bands (default 0.95)

    Returns:
        Dict mapping metric names to their StabilityEnvelopes

    Examples:
        >>> trajectories = {
        ...     "success": [0.8, 0.75, 0.82],
        ...     "abstention": [0.1, 0.12, 0.15],
        ... }
        >>> envelopes = generate_multi_metric_envelopes(trajectories, window=2)
        >>> "success" in envelopes
        True
    """
    envelopes: Dict[str, StabilityEnvelope] = {}

    for metric_name, series in trajectories.items():
        envelope = generate_stability_envelope(
            series,
            metric_name=metric_name,
            window=window,
            confidence=confidence,
        )
        envelopes[metric_name] = envelope

    return envelopes


# ---------------------------------------------------------------------------
# Drift Monotonicity Analysis
# ---------------------------------------------------------------------------

def analyze_drift_monotonicity(
    series: NumericSeries,
    segment_size: int = 5,
) -> Dict[str, Any]:
    """
    Analyze monotonicity of drift across segments.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This function breaks the series into segments and computes monotonicity
    for each segment, as well as whether segment trends are themselves monotonic.

    Args:
        series: Sequence of metric values
        segment_size: Size of each segment (default 5)

    Returns:
        Dict with:
            - overall_monotonicity: Monotonicity of full series
            - segment_monotonicity: List of per-segment monotonicity scores
            - segment_trends: List of per-segment trend values
            - meta_monotonicity: Monotonicity of segment trends
            - segment_count: Number of segments
    """
    if not series or len(series) < 2:
        return {
            "overall_monotonicity": 1.0,
            "segment_monotonicity": [],
            "segment_trends": [],
            "meta_monotonicity": 1.0,
            "segment_count": 0,
        }

    values = [float(v) for v in series]
    n = len(values)

    # Overall monotonicity
    overall_mono = compute_monotonicity(values)

    # Segment analysis
    segment_monotonicity: List[float] = []
    segment_trends: List[float] = []

    for i in range(0, n, segment_size):
        segment = values[i:i + segment_size]
        if len(segment) >= 2:
            segment_monotonicity.append(compute_monotonicity(segment))
            segment_trends.append(abstention_trend(segment))

    # Meta-monotonicity: are the segment trends themselves monotonic?
    meta_mono = compute_monotonicity(segment_trends) if len(segment_trends) >= 2 else 1.0

    return {
        "overall_monotonicity": overall_mono,
        "segment_monotonicity": segment_monotonicity,
        "segment_trends": segment_trends,
        "meta_monotonicity": meta_mono,
        "segment_count": len(segment_monotonicity),
    }


# ---------------------------------------------------------------------------
# Drift Threshold Library
# ---------------------------------------------------------------------------

# Predefined threshold profiles for CI gating
# These are purely descriptive thresholds for CI tooling.
# They do NOT determine whether drift is good or bad.
#
# STABILITY CONTRACT:
#   - All profiles must have exactly two numeric keys: "drift_score" and "coherence"
#   - All values must be floats in [0.0, 1.0]
#   - Only these profiles are permitted: "default", "strict", "lenient"
#   - Adding/removing profiles requires updating tests
#
DRIFT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "default": {
        "drift_score": 0.3,
        "coherence": 0.4,
    },
    "strict": {
        "drift_score": 0.2,
        "coherence": 0.5,
    },
    "lenient": {
        "drift_score": 0.4,
        "coherence": 0.3,
    },
}

# Canonical list of valid profile names (for validation)
VALID_PROFILES: Tuple[str, ...] = ("default", "strict", "lenient")


def validate_drift_thresholds() -> None:
    """
    Validate DRIFT_THRESHOLDS at import time.

    Raises:
        AssertionError: If thresholds violate contract
    """
    # Check profile set matches canonical list
    assert set(DRIFT_THRESHOLDS.keys()) == set(VALID_PROFILES), (
        f"Profile mismatch: got {set(DRIFT_THRESHOLDS.keys())}, expected {set(VALID_PROFILES)}"
    )

    for profile, thresholds in DRIFT_THRESHOLDS.items():
        # Check required keys
        assert set(thresholds.keys()) == {"drift_score", "coherence"}, (
            f"Profile '{profile}' must have exactly 'drift_score' and 'coherence' keys"
        )

        # Check types and bounds
        for key, value in thresholds.items():
            assert isinstance(value, (int, float)), (
                f"Profile '{profile}' key '{key}' must be numeric, got {type(value)}"
            )
            assert 0.0 <= value <= 1.0, (
                f"Profile '{profile}' key '{key}' must be in [0.0, 1.0], got {value}"
            )


# Run validation at import time
validate_drift_thresholds()


def get_drift_thresholds(profile: str = "default") -> Dict[str, float]:
    """
    Get drift thresholds for a named profile.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        profile: Profile name ("default", "strict", "lenient")

    Returns:
        Dict with "drift_score" and "coherence" thresholds (floats in [0, 1])

    Raises:
        ValueError: If profile is unknown
    """
    if profile not in DRIFT_THRESHOLDS:
        valid = ", ".join(sorted(DRIFT_THRESHOLDS.keys()))
        raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {valid}")

    thresholds = DRIFT_THRESHOLDS[profile]
    return {
        "drift_score": float(thresholds["drift_score"]),
        "coherence": float(thresholds["coherence"]),
    }


# ---------------------------------------------------------------------------
# Pattern Hint Classification
# ---------------------------------------------------------------------------

# Valid pattern hints (for visualization/intuition only, not decision logic)
PATTERN_HINTS = ("flat", "rising", "falling", "oscillatory")


def classify_pattern_hint(
    series: NumericSeries,
    monotonicity_threshold: float = 0.8,
    segment_variance_threshold: float = 0.3,
) -> str:
    """
    Classify a series into a pattern hint for visualization.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This is meant for visualization/intuition, not for any decision logic.

    Classification rules (heuristic):
        - Monotonicity >= threshold and positive slope → "rising"
        - Monotonicity >= threshold and negative slope → "falling"
        - High variance between segments → "oscillatory"
        - Otherwise → "flat"

    Args:
        series: Sequence of metric values
        monotonicity_threshold: Threshold for monotonic classification (default 0.8)
        segment_variance_threshold: Threshold for oscillatory detection (default 0.3)

    Returns:
        One of: "flat", "rising", "falling", "oscillatory"
    """
    if not series or len(series) < 2:
        return "flat"

    values = [float(v) for v in series]
    n = len(values)

    # Compute monotonicity
    mono = compute_monotonicity(values)

    # Compute slope direction
    trend = abstention_trend(values)

    # Check for high monotonicity with clear direction
    if mono >= monotonicity_threshold:
        if trend > 0.1:
            return "rising"
        elif trend < -0.1:
            return "falling"

    # Check for oscillatory behavior using segment analysis
    if n >= 4:
        segment_size = max(2, n // 4)
        analysis = analyze_drift_monotonicity(values, segment_size=segment_size)
        segment_trends = analysis.get("segment_trends", [])

        if len(segment_trends) >= 2:
            # Check variance of segment trends
            trend_mean = sum(segment_trends) / len(segment_trends)
            trend_var = sum((t - trend_mean) ** 2 for t in segment_trends) / len(segment_trends)

            if trend_var >= segment_variance_threshold:
                return "oscillatory"

    return "flat"


# ---------------------------------------------------------------------------
# CI Gate Helper
# ---------------------------------------------------------------------------

@dataclass
class CIGateResult:
    """
    Result of CI gate evaluation.

    This is for CI/tooling only. Must NOT be used by promotion or governance code.
    """
    status: str  # "OK", "WARN", "BLOCK"
    max_metric_drift_score: float
    drift_alignment_score: float
    coherence_score: float
    offending_metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "max_metric_drift_score": self.max_metric_drift_score,
            "drift_alignment_score": self.drift_alignment_score,
            "coherence_score": self.coherence_score,
            "offending_metrics": self.offending_metrics,
        }


def evaluate_drift_for_ci(
    result: DriftAlignmentResult,
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
) -> CIGateResult:
    """
    Evaluate drift alignment result for CI gating.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    This function is for CI/tooling ONLY. It must NOT be called by
    promotion or governance code. It produces no side effects.

    Status logic:
        - OK: max_metric_drift_score <= drift_score_threshold AND
              coherence_score >= coherence_threshold
        - WARN: Drifts are moderate but coherent, or coherence is marginal
        - BLOCK: High drift with low coherence

    Args:
        result: DriftAlignmentResult from compute_drift_alignment
        drift_score_threshold: Maximum acceptable drift score (default 0.3)
        coherence_threshold: Minimum acceptable coherence (default 0.4)

    Returns:
        CIGateResult with status and diagnostic info
    """
    # Compute max drift score across all metrics
    drift_scores = [sig.drift_score for sig in result.metrics.values()]
    max_drift = max(drift_scores) if drift_scores else 0.0

    # Find metrics exceeding threshold
    offending = [
        name for name, sig in result.metrics.items()
        if sig.drift_score > drift_score_threshold
    ]

    coherence = result.coherence_score
    alignment = result.drift_alignment_score

    # Determine status
    if max_drift <= drift_score_threshold and coherence >= coherence_threshold:
        status = "OK"
    elif max_drift <= drift_score_threshold * 1.5 and coherence >= coherence_threshold * 0.75:
        # Moderate drift but reasonable coherence
        status = "WARN"
    elif max_drift > drift_score_threshold and coherence >= coherence_threshold:
        # High drift but coherent (metrics moving together)
        status = "WARN"
    else:
        # High drift with low coherence
        status = "BLOCK"

    return CIGateResult(
        status=status,
        max_metric_drift_score=round(max_drift, 6),
        drift_alignment_score=alignment,
        coherence_score=coherence,
        offending_metrics=sorted(offending),
    )


# ---------------------------------------------------------------------------
# Drift Cells Grid (metric×slice table)
# ---------------------------------------------------------------------------

@dataclass
class DriftCell:
    """
    A single cell in the drift grid (one metric in one slice).

    This is a convenience structure for building drift tables/CSVs.
    """
    slice: str
    metric_kind: str
    drift_score: float
    direction: str
    stability: float
    pattern_hint: str
    drift_alignment_score: float
    coherence_score: float
    status: str
    profile: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/CSV export."""
        return {
            "slice": self.slice,
            "metric_kind": self.metric_kind,
            "drift_score": self.drift_score,
            "direction": self.direction,
            "stability": self.stability,
            "pattern_hint": self.pattern_hint,
            "drift_alignment_score": self.drift_alignment_score,
            "coherence_score": self.coherence_score,
            "status": self.status,
            "profile": self.profile,
        }


# Column ordering for CSV export (deterministic)
DRIFT_CELL_COLUMNS: Tuple[str, ...] = (
    "slice",
    "metric_kind",
    "drift_score",
    "direction",
    "stability",
    "pattern_hint",
    "drift_alignment_score",
    "coherence_score",
    "status",
    "profile",
)


def build_drift_cells(
    report: Dict[str, Any],
    *,
    profile: str = "default",
) -> List[DriftCell]:
    """
    Build a list of drift cells from a report or multi-slice report.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        report: Either a single-slice report dict or multi-slice report dict.
        profile: Threshold profile used for the report.

    Returns:
        List of DriftCell objects, sorted by (slice, metric_kind).
    """
    cells: List[DriftCell] = []

    # Check if this is a multi-slice report
    if "slices" in report:
        # Multi-slice report
        for slice_id in sorted(report["slices"].keys()):
            slice_report = report["slices"][slice_id]
            cells.extend(_extract_cells_from_slice(slice_id, slice_report, profile))
    else:
        # Single-slice report
        slice_id = report.get("slice", "unknown")
        cells.extend(_extract_cells_from_slice(slice_id, report, profile))

    # Sort by slice then metric_kind for deterministic ordering
    cells.sort(key=lambda c: (c.slice, c.metric_kind))

    return cells


def _extract_cells_from_slice(
    slice_id: str,
    slice_report: Dict[str, Any],
    profile: str,
) -> List[DriftCell]:
    """Extract drift cells from a single slice report."""
    cells: List[DriftCell] = []

    drift_alignment_score = slice_report.get("drift_alignment_score", 0.0)
    coherence_score = slice_report.get("coherence_score", 0.0)

    # Get CI gate status if available
    ci_gate = slice_report.get("ci_gate", {})
    status = ci_gate.get("status", "UNKNOWN")

    # Get pattern hints
    pattern_hints = slice_report.get("pattern_hints", {})

    # Iterate over metrics
    metrics = slice_report.get("metrics", {})
    for metric_name in sorted(metrics.keys()):
        metric_data = metrics[metric_name]

        cell = DriftCell(
            slice=slice_id,
            metric_kind=metric_name,
            drift_score=metric_data.get("drift_score", 0.0),
            direction=metric_data.get("direction", "stable"),
            stability=metric_data.get("stability", 1.0),
            pattern_hint=pattern_hints.get(metric_name, metric_data.get("pattern_hint", "flat")),
            drift_alignment_score=drift_alignment_score,
            coherence_score=coherence_score,
            status=status,
            profile=profile,
        )
        cells.append(cell)

    return cells


def drift_cells_to_dicts(cells: List[DriftCell]) -> List[Dict[str, Any]]:
    """
    Convert a list of DriftCell objects to list of dicts.

    Args:
        cells: List of DriftCell objects.

    Returns:
        List of dicts with deterministic key ordering.
    """
    return [cell.to_dict() for cell in cells]


# ---------------------------------------------------------------------------
# Phase III: Promotion Radar & Global Drift Signal
# ---------------------------------------------------------------------------

def summarize_slice_drift(
    cells: Sequence[DriftCell],
    slice_name: str,
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Summarize drift at the slice level.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        cells: Sequence of DriftCell objects (filtered to one slice or all).
        slice_name: Name of the slice to summarize.
        drift_score_threshold: Threshold for high drift detection (default 0.3).
        coherence_threshold: Threshold for coherence issues (default 0.4).

    Returns:
        Dict with slice-level drift summary:
        - schema_version: "1.0"
        - slice_name: str
        - metrics_evaluated: int
        - metrics_with_high_drift: List[str]
        - coherence_issues: int
        - slice_drift_status: "OK" | "WARN" | "DRIFTY"
    """
    # Filter cells for this slice
    slice_cells = [c for c in cells if c.slice == slice_name]

    if not slice_cells:
        return {
            "schema_version": "1.0",
            "slice_name": slice_name,
            "metrics_evaluated": 0,
            "metrics_with_high_drift": [],
            "coherence_issues": 0,
            "slice_drift_status": "OK",
        }

    metrics_evaluated = len(slice_cells)
    metrics_with_high_drift = [
        c.metric_kind for c in slice_cells
        if c.drift_score > drift_score_threshold or c.status in ("WARN", "BLOCK")
    ]
    coherence_issues = sum(
        1 for c in slice_cells
        if c.coherence_score < coherence_threshold
    )

    # Determine slice status
    if any(c.status == "BLOCK" for c in slice_cells):
        slice_status = "DRIFTY"
    elif len(metrics_with_high_drift) > metrics_evaluated * 0.5:
        # More than 50% of metrics have high drift
        slice_status = "DRIFTY"
    elif metrics_with_high_drift or coherence_issues > 0:
        slice_status = "WARN"
    else:
        slice_status = "OK"

    return {
        "schema_version": "1.0",
        "slice_name": slice_name,
        "metrics_evaluated": metrics_evaluated,
        "metrics_with_high_drift": sorted(metrics_with_high_drift),
        "coherence_issues": coherence_issues,
        "slice_drift_status": slice_status,
    }


def evaluate_drift_for_promotion(
    cells: Sequence[DriftCell],
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Evaluate drift cells for promotion readiness.

    This is a pure advisory signal that MAAS / CI can choose to turn into
    a hard gate. Statistics provided here are descriptive; uplift evaluation
    is handled solely by governance-gate logic.

    Args:
        cells: Sequence of DriftCell objects (all slices/metrics).
        drift_score_threshold: Threshold for high drift detection (default 0.3).
        coherence_threshold: Threshold for coherence issues (default 0.4).

    Returns:
        Dict with promotion evaluation:
        - promotion_ok: bool
        - drifty_slices: List[str]
        - drifty_metrics: List[str]
        - status: "OK" | "ATTENTION" | "BLOCK"
    """
    if not cells:
        return {
            "promotion_ok": True,
            "drifty_slices": [],
            "drifty_metrics": [],
            "status": "OK",
        }

    # Group by slice
    slices = {}
    for cell in cells:
        if cell.slice not in slices:
            slices[cell.slice] = []
        slices[cell.slice].append(cell)

    drifty_slices = []
    drifty_metrics = set()

    for slice_name, slice_cells in slices.items():
        summary = summarize_slice_drift(
            cells,
            slice_name,
            drift_score_threshold=drift_score_threshold,
            coherence_threshold=coherence_threshold,
        )

        if summary["slice_drift_status"] in ("WARN", "DRIFTY"):
            drifty_slices.append(slice_name)
            drifty_metrics.update(summary["metrics_with_high_drift"])

    # Determine overall status
    if any(
        summarize_slice_drift(
            cells,
            s,
            drift_score_threshold=drift_score_threshold,
            coherence_threshold=coherence_threshold,
        )["slice_drift_status"] == "DRIFTY"
        for s in slices.keys()
    ):
        status = "BLOCK"
        promotion_ok = False
    elif drifty_slices:
        status = "ATTENTION"
        promotion_ok = False
    else:
        status = "OK"
        promotion_ok = True

    return {
        "promotion_ok": promotion_ok,
        "drifty_slices": sorted(drifty_slices),
        "drifty_metrics": sorted(drifty_metrics),
        "status": status,
    }


def summarize_drift_for_global_health(
    cells: Sequence[DriftCell],
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
    hotspot_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Summarize drift across all slices for global health monitoring.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        cells: Sequence of DriftCell objects (all slices/metrics).
        drift_score_threshold: Threshold for high drift detection (default 0.3).
        coherence_threshold: Threshold for coherence issues (default 0.4).
        hotspot_threshold: Fraction of non-OK cells to be a hotspot (default 0.5).

    Returns:
        Dict with global health summary:
        - drift_hotspot_slices: List[str] (slices with many non-OK cells)
        - drift_hotspot_metrics: List[str] (metrics with frequent drift)
        - status: "OK" | "WARN" | "HOT"
    """
    if not cells:
        return {
            "drift_hotspot_slices": [],
            "drift_hotspot_metrics": [],
            "status": "OK",
        }

    # Group by slice
    slices = {}
    for cell in cells:
        if cell.slice not in slices:
            slices[cell.slice] = []
        slices[cell.slice].append(cell)

    # Group by metric
    metrics = {}
    for cell in cells:
        if cell.metric_kind not in metrics:
            metrics[cell.metric_kind] = []
        metrics[cell.metric_kind].append(cell)

    # Find hotspot slices (slices with >threshold non-OK cells)
    hotspot_slices = []
    for slice_name, slice_cells in slices.items():
        non_ok_count = sum(1 for c in slice_cells if c.status != "OK")
        if non_ok_count / len(slice_cells) >= hotspot_threshold:
            hotspot_slices.append(slice_name)

    # Find hotspot metrics (metrics with >threshold non-OK across slices)
    hotspot_metrics = []
    for metric_name, metric_cells in metrics.items():
        non_ok_count = sum(1 for c in metric_cells if c.status != "OK")
        if non_ok_count / len(metric_cells) >= hotspot_threshold:
            hotspot_metrics.append(metric_name)

    # Determine global status
    if hotspot_slices or hotspot_metrics:
        # Check if any slice is DRIFTY
        has_drifty = any(
            summarize_slice_drift(
                cells,
                s,
                drift_score_threshold=drift_score_threshold,
                coherence_threshold=coherence_threshold,
            )["slice_drift_status"] == "DRIFTY"
            for s in slices.keys()
        )
        if has_drifty:
            status = "HOT"
        else:
            status = "WARN"
    else:
        status = "OK"

    return {
        "drift_hotspot_slices": sorted(hotspot_slices),
        "drift_hotspot_metrics": sorted(hotspot_metrics),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Phase IV: Drift-Aware Promotion & Global Drift Intelligence
# ---------------------------------------------------------------------------

# Core uplift metrics that are critical for promotion decisions
CORE_UPLIFT_METRICS: Tuple[str, ...] = ("success", "abstention", "depth")


def build_drift_trend_history(
    drift_cells_by_run: Dict[str, Sequence[DriftCell]],
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Build drift trend history across multiple runs.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_cells_by_run: Dict mapping run_id to Sequence of DriftCell objects.
        drift_score_threshold: Threshold for high drift detection (default 0.3).
        coherence_threshold: Threshold for coherence issues (default 0.4).

    Returns:
        Dict with trend history including:
        - runs: Dict mapping run_id to per-run summary
        - runs_with_drift: List of run_ids with any drift
        - runs_without_drift: List of run_ids with no drift
        - trend_status: "IMPROVING" | "STABLE" | "DEGRADING"
    """
    runs: Dict[str, Dict[str, Any]] = {}
    runs_with_drift: List[str] = []
    runs_without_drift: List[str] = []

    # Process each run
    for run_id, cells in sorted(drift_cells_by_run.items()):
        if not cells:
            runs[run_id] = {
                "drifty_slices": [],
                "metrics_with_high_drift": [],
                "has_drift": False,
            }
            runs_without_drift.append(run_id)
            continue

        # Get unique slices
        slices = sorted(set(c.slice for c in cells))

        drifty_slices: List[str] = []
        metrics_with_high_drift: set = set()

        for slice_name in slices:
            slice_summary = summarize_slice_drift(
                cells,
                slice_name,
                drift_score_threshold=drift_score_threshold,
                coherence_threshold=coherence_threshold,
            )

            if slice_summary["slice_drift_status"] in ("WARN", "DRIFTY"):
                drifty_slices.append(slice_name)
                metrics_with_high_drift.update(slice_summary["metrics_with_high_drift"])

        has_drift = bool(drifty_slices or metrics_with_high_drift)

        runs[run_id] = {
            "drifty_slices": sorted(drifty_slices),
            "metrics_with_high_drift": sorted(metrics_with_high_drift),
            "has_drift": has_drift,
        }

        if has_drift:
            runs_with_drift.append(run_id)
        else:
            runs_without_drift.append(run_id)

    # Determine trend status
    trend_status = _compute_trend_status(runs_with_drift, runs_without_drift)

    return {
        "runs": runs,
        "runs_with_drift": sorted(runs_with_drift),
        "runs_without_drift": sorted(runs_without_drift),
        "trend_status": trend_status,
    }


def _compute_trend_status(
    runs_with_drift: List[str],
    runs_without_drift: List[str],
) -> str:
    """
    Compute trend status based on drift frequency over time.

    Args:
        runs_with_drift: List of run_ids with drift (should be sorted chronologically).
        runs_without_drift: List of run_ids without drift (should be sorted chronologically).

    Returns:
        "IMPROVING" | "STABLE" | "DEGRADING"
    """
    total_runs = len(runs_with_drift) + len(runs_without_drift)

    if total_runs == 0:
        return "STABLE"

    if total_runs < 3:
        # Not enough data for trend analysis
        drift_ratio = len(runs_with_drift) / total_runs
        if drift_ratio == 0.0:
            return "STABLE"
        elif drift_ratio < 0.5:
            return "STABLE"
        else:
            return "DEGRADING"

    # Split runs into early and late halves
    all_runs = sorted(runs_with_drift + runs_without_drift)
    mid_point = len(all_runs) // 2
    early_runs = all_runs[:mid_point]
    late_runs = all_runs[mid_point:]

    early_drift_count = sum(1 for r in early_runs if r in runs_with_drift)
    late_drift_count = sum(1 for r in late_runs if r in runs_with_drift)

    early_ratio = early_drift_count / len(early_runs) if early_runs else 0.0
    late_ratio = late_drift_count / len(late_runs) if late_runs else 0.0

    # Determine trend
    if late_ratio < early_ratio - 0.1:  # At least 10% improvement
        return "IMPROVING"
    elif late_ratio > early_ratio + 0.1:  # At least 10% degradation
        return "DEGRADING"
    else:
        return "STABLE"


def evaluate_drift_trend_for_promotion(
    trend_history: Dict[str, Any],
    *,
    core_metrics: Sequence[str] = CORE_UPLIFT_METRICS,
) -> Dict[str, Any]:
    """
    Evaluate drift trend for promotion readiness.

    This is a pure advisory signal that MAAS / CI can choose to turn into
    a hard gate. Statistics provided here are descriptive; uplift evaluation
    is handled solely by governance-gate logic.

    Args:
        trend_history: Dict from build_drift_trend_history.
        core_metrics: Sequence of metric names considered critical (default: success, abstention, depth).

    Returns:
        Dict with promotion evaluation:
        - promotion_ok: bool
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: List[str]
    """
    trend_status = trend_history.get("trend_status", "STABLE")
    runs = trend_history.get("runs", {})
    runs_with_drift = trend_history.get("runs_with_drift", [])

    blocking_reasons: List[str] = []
    promotion_ok = True
    status = "OK"

    # Check if trend is DEGRADING
    if trend_status == "DEGRADING":
        # Check if core metrics are affected in recent runs
        recent_runs = sorted(runs.keys())[-3:] if len(runs) >= 3 else sorted(runs.keys())
        core_metrics_affected = False

        for run_id in recent_runs:
            run_data = runs.get(run_id, {})
            metrics_with_drift = set(run_data.get("metrics_with_high_drift", []))
            if any(m in metrics_with_drift for m in core_metrics):
                core_metrics_affected = True
                break

        if core_metrics_affected:
            promotion_ok = False
            status = "BLOCK"
            blocking_reasons.append(
                f"Drift trend is DEGRADING and affects core metrics: {', '.join(core_metrics)}"
            )
        else:
            promotion_ok = False
            status = "WARN"
            blocking_reasons.append("Drift trend is DEGRADING")

    # Check if drift is STABLE but frequent
    elif trend_status == "STABLE":
        drift_frequency = len(runs_with_drift) / len(runs) if runs else 0.0

        if drift_frequency > 0.5:  # More than 50% of runs have drift
            promotion_ok = False
            status = "WARN"
            blocking_reasons.append(
                f"Drift is STABLE but frequent ({drift_frequency:.1%} of runs)"
            )

    # IMPROVING trend or rare drift -> OK
    # (trend_status == "IMPROVING" or drift is rare)

    return {
        "promotion_ok": promotion_ok,
        "status": status,
        "blocking_reasons": sorted(blocking_reasons),
    }


def build_drift_director_panel(
    global_health_drift: Dict[str, Any],
    trend_history: Dict[str, Any],
    promotion_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Director Console drift panel combining all drift intelligence.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        global_health_drift: Dict from summarize_drift_for_global_health.
        trend_history: Dict from build_drift_trend_history.
        promotion_eval: Dict from evaluate_drift_trend_for_promotion.

    Returns:
        Dict with director panel including:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - drift_hotspot_slices: List[str]
        - drift_hotspot_metrics: List[str]
        - trend_status: str
        - headline: str (neutral summary)
    """
    # Extract hotspots first (needed for status light logic)
    drift_hotspot_slices = global_health_drift.get("drift_hotspot_slices", [])
    drift_hotspot_metrics = global_health_drift.get("drift_hotspot_metrics", [])

    # Determine status light
    global_status = global_health_drift.get("status", "OK")
    promotion_status = promotion_eval.get("status", "OK")
    trend_status = trend_history.get("trend_status", "STABLE")

    # RED if any critical issue
    if (
        global_status == "HOT"
        or promotion_status == "BLOCK"
        or trend_status == "DEGRADING"
    ):
        status_light = "RED"
    # YELLOW if warnings (but not if everything is OK)
    elif (
        global_status == "WARN"
        or promotion_status == "WARN"
        or (trend_status == "STABLE" and (drift_hotspot_slices or drift_hotspot_metrics))
    ):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Generate neutral headline
    headline = _generate_drift_headline(
        global_status, trend_status, promotion_status, drift_hotspot_slices
    )

    return {
        "status_light": status_light,
        "drift_hotspot_slices": sorted(drift_hotspot_slices),
        "drift_hotspot_metrics": sorted(drift_hotspot_metrics),
        "trend_status": trend_status,
        "headline": headline,
    }


def _generate_drift_headline(
    global_status: str,
    trend_status: str,
    promotion_status: str,
    hotspot_slices: List[str],
) -> str:
    """
    Generate a neutral, descriptive headline about drift posture.

    Args:
        global_status: Status from global health summary.
        trend_status: Status from trend history.
        promotion_status: Status from promotion evaluation.
        hotspot_slices: List of hotspot slice names.

    Returns:
        Neutral headline string.
    """
    parts: List[str] = []

    # Trend information
    if trend_status == "IMPROVING":
        parts.append("Drift trend improving")
    elif trend_status == "DEGRADING":
        parts.append("Drift trend degrading")
    else:
        parts.append("Drift trend stable")

    # Global status
    if global_status == "HOT":
        parts.append("hotspots detected")
    elif global_status == "WARN":
        parts.append("warnings present")

    # Hotspot slices
    if hotspot_slices:
        if len(hotspot_slices) == 1:
            parts.append(f"hotspot in {hotspot_slices[0]}")
        elif len(hotspot_slices) <= 3:
            parts.append(f"hotspots in {', '.join(hotspot_slices)}")
        else:
            parts.append(f"{len(hotspot_slices)} hotspot slices")

    # Promotion status
    if promotion_status == "BLOCK":
        parts.append("promotion blocked")
    elif promotion_status == "WARN":
        parts.append("promotion attention required")

    if not parts:
        return "No drift issues detected"

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Phase V: Cross-Metric Drift Coupler & Multi-Axis Drift Summary
# ---------------------------------------------------------------------------

# Valid axis names for multi-axis drift view
DRIFT_AXES: Tuple[str, ...] = ("drift", "budget", "metrics")


def build_multi_axis_drift_view(
    drift_trend_history: Dict[str, Any],
    budget_drift_view: Optional[Dict[str, Any]] = None,
    metric_conformance_drift: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build multi-axis drift view combining drift, budget, and metrics axes.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_trend_history: Dict from build_drift_trend_history.
        budget_drift_view: Optional dict with budget drift information.
            Expected keys: "status" ("OK" | "WARN" | "BLOCK"), "has_drift" (bool).
        metric_conformance_drift: Optional dict with metric conformance drift.
            Expected keys: "status" ("OK" | "WARN" | "BLOCK"), "has_drift" (bool).

    Returns:
        Dict with multi-axis view:
        - axes_with_drift: List[str] (from DRIFT_AXES)
        - high_risk_axes: List[str] (axes with BLOCK status or severe drift)
        - global_drift_status: "OK" | "ATTENTION" | "BLOCK"
        - neutral_notes: List[str] (descriptive notes about each axis)
    """
    axes_with_drift: List[str] = []
    high_risk_axes: List[str] = []
    neutral_notes: List[str] = []

    # Check drift axis
    drift_trend_status = drift_trend_history.get("trend_status", "STABLE")
    runs_with_drift = drift_trend_history.get("runs_with_drift", [])
    total_runs = len(drift_trend_history.get("runs", {}))

    if runs_with_drift or drift_trend_status != "STABLE":
        axes_with_drift.append("drift")
        if drift_trend_status == "DEGRADING" or (
            total_runs > 0 and len(runs_with_drift) / total_runs > 0.5
        ):
            high_risk_axes.append("drift")
            neutral_notes.append("Drift axis: degrading trend or frequent drift detected")
        else:
            neutral_notes.append("Drift axis: drift present but within tolerance")

    # Check budget axis
    if budget_drift_view:
        budget_status = budget_drift_view.get("status", "OK")
        budget_has_drift = budget_drift_view.get("has_drift", False)

        if budget_has_drift or budget_status != "OK":
            axes_with_drift.append("budget")
            if budget_status == "BLOCK" or budget_has_drift:
                high_risk_axes.append("budget")
                neutral_notes.append("Budget axis: drift or constraint violations detected")
            else:
                neutral_notes.append("Budget axis: warnings present")

    # Check metrics axis
    if metric_conformance_drift:
        metrics_status = metric_conformance_drift.get("status", "OK")
        metrics_has_drift = metric_conformance_drift.get("has_drift", False)

        if metrics_has_drift or metrics_status != "OK":
            axes_with_drift.append("metrics")
            if metrics_status == "BLOCK" or metrics_has_drift:
                high_risk_axes.append("metrics")
                neutral_notes.append("Metrics axis: conformance drift detected")
            else:
                neutral_notes.append("Metrics axis: warnings present")

    # Determine global status
    if len(high_risk_axes) >= 2:
        # Multiple high-risk axes -> BLOCK
        global_status = "BLOCK"
    elif len(high_risk_axes) == 1 or len(axes_with_drift) >= 2:
        # Single high-risk or multiple axes with drift -> ATTENTION
        global_status = "ATTENTION"
    elif axes_with_drift:
        # Single axis with drift -> ATTENTION
        global_status = "ATTENTION"
    else:
        global_status = "OK"

    return {
        "axes_with_drift": sorted(axes_with_drift),
        "high_risk_axes": sorted(high_risk_axes),
        "global_drift_status": global_status,
        "neutral_notes": sorted(neutral_notes),
    }


def summarize_drift_for_uplift_envelope(
    multi_axis_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize drift conditions for uplift envelope safety assessment.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        multi_axis_view: Dict from build_multi_axis_drift_view.

    Returns:
        Dict with uplift envelope summary:
        - uplift_safe_under_drift: bool
        - status: "OK" | "ATTENTION" | "BLOCK"
        - blocking_axes: List[str]
        - recommendations: List[str] (neutral, descriptive recommendations)
    """
    high_risk_axes = set(multi_axis_view.get("high_risk_axes", []))
    axes_with_drift = set(multi_axis_view.get("axes_with_drift", []))
    global_status = multi_axis_view.get("global_drift_status", "OK")

    blocking_axes: List[str] = []
    recommendations: List[str] = []
    uplift_safe = True
    status = "OK"

    # BLOCK if multiple high-risk axes, especially drift+budget
    if len(high_risk_axes) >= 2:
        uplift_safe = False
        status = "BLOCK"
        blocking_axes = sorted(high_risk_axes)
        recommendations.append(
            f"Multiple high-risk axes detected: {', '.join(sorted(high_risk_axes))}"
        )

        # Special case: drift+budget combination
        if "drift" in high_risk_axes and "budget" in high_risk_axes:
            recommendations.append(
                "Drift and budget axes both show high risk; uplift may be constrained"
            )
    elif len(high_risk_axes) == 1:
        # Single high-risk axis
        uplift_safe = False
        status = "ATTENTION"
        blocking_axes = sorted(high_risk_axes)
        recommendations.append(
            f"High-risk axis detected: {', '.join(sorted(high_risk_axes))}"
        )
    elif len(axes_with_drift) >= 2:
        # Multiple axes with drift (but not high-risk)
        uplift_safe = False
        status = "ATTENTION"
        recommendations.append(
            f"Multiple axes show drift: {', '.join(sorted(axes_with_drift))}"
        )
    elif axes_with_drift:
        # Single axis with drift
        uplift_safe = True
        status = "ATTENTION"
        recommendations.append(
            f"Drift detected in {', '.join(sorted(axes_with_drift))} axis"
        )
    else:
        # No drift
        uplift_safe = True
        status = "OK"
        recommendations.append("No drift detected across monitored axes")

    return {
        "uplift_safe_under_drift": uplift_safe,
        "status": status,
        "blocking_axes": sorted(blocking_axes),
        "recommendations": sorted(recommendations),
    }


# ---------------------------------------------------------------------------
# Phase VI: Drift Tensor & Poly-Cause Analysis
# ---------------------------------------------------------------------------

DRIFT_TENSOR_SCHEMA_VERSION = "1.0.0"


def build_drift_tensor(
    drift_axis_view: Dict[str, Any],
    budget_axis: Optional[Dict[str, Any]] = None,
    metric_axis: Optional[Dict[str, Any]] = None,
    semantic_axis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build drift tensor representing multi-dimensional drift across slices and axes.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_axis_view: Dict from build_multi_axis_drift_view OR drift_trend_history.
            If from build_multi_axis_drift_view, should contain "drift_trend_history" key.
            If drift_trend_history directly, should have "runs" key with per-run data.
        budget_axis: Optional dict with budget drift per slice.
            Expected: {"<slice>": {"drift_score": float}, ...}
        metric_axis: Optional dict with metric drift per slice.
            Expected: {"<slice>": {"drift_score": float}, ...}
        semantic_axis: Optional dict with semantic drift per slice.
            Expected: {"<slice>": {"drift_score": float}, ...}

    Returns:
        Dict with drift tensor:
        - tensor: Dict mapping slice to axis drift scores
        - global_tensor_norm: float (L2 norm of all drift scores)
        - ranked_slices: List[str] (slices sorted by total drift magnitude)
        - schema_version: "1.0.0"
    """
    tensor: Dict[str, Dict[str, float]] = {}

    # Collect all slices from all available axes
    all_slices: set = set()

    # Extract slices from budget_axis
    if budget_axis:
        all_slices.update(budget_axis.keys())

    # Extract slices from metric_axis
    if metric_axis:
        all_slices.update(metric_axis.keys())

    # Extract slices from semantic_axis
    if semantic_axis:
        all_slices.update(semantic_axis.keys())

    # Extract slices from drift_axis_view (from trend history)
    # Handle both cases: drift_axis_view is drift_trend_history directly,
    # or it's from build_multi_axis_drift_view with nested drift_trend_history
    if "drift_trend_history" in drift_axis_view:
        drift_trend_history = drift_axis_view["drift_trend_history"]
    else:
        # Assume drift_axis_view is drift_trend_history directly
        drift_trend_history = drift_axis_view
    
    runs = drift_trend_history.get("runs", {})
    for run_data in runs.values():
        drifty_slices = run_data.get("drifty_slices", [])
        all_slices.update(drifty_slices)

    # If no slices found, return empty tensor
    if not all_slices:
        return {
            "tensor": {},
            "global_tensor_norm": 0.0,
            "ranked_slices": [],
            "schema_version": DRIFT_TENSOR_SCHEMA_VERSION,
        }

    # Build tensor for each slice
    for slice_name in sorted(all_slices):
        slice_tensor: Dict[str, float] = {}

        # Drift axis score (from trend history)
        # Compute drift frequency for this slice across runs
        drift_count = 0
        total_runs = len(runs)
        for run_data in runs.values():
            if slice_name in run_data.get("drifty_slices", []):
                drift_count += 1
        
        # Use drift frequency as drift score (0.0 to 1.0)
        slice_tensor["drift"] = drift_count / total_runs if total_runs > 0 else 0.0

        # Budget axis score
        if budget_axis and slice_name in budget_axis:
            budget_slice_data = budget_axis[slice_name]
            if isinstance(budget_slice_data, dict):
                slice_tensor["budget"] = budget_slice_data.get("drift_score", 0.0)
            else:
                slice_tensor["budget"] = float(budget_slice_data) if isinstance(budget_slice_data, (int, float)) else 0.0
        else:
            slice_tensor["budget"] = 0.0

        # Metric axis score
        if metric_axis and slice_name in metric_axis:
            metric_slice_data = metric_axis[slice_name]
            if isinstance(metric_slice_data, dict):
                slice_tensor["metric"] = metric_slice_data.get("drift_score", 0.0)
            else:
                slice_tensor["metric"] = float(metric_slice_data) if isinstance(metric_slice_data, (int, float)) else 0.0
        else:
            slice_tensor["metric"] = 0.0

        # Semantic axis score
        if semantic_axis and slice_name in semantic_axis:
            semantic_slice_data = semantic_axis[slice_name]
            if isinstance(semantic_slice_data, dict):
                slice_tensor["semantic"] = semantic_slice_data.get("drift_score", 0.0)
            else:
                slice_tensor["semantic"] = float(semantic_slice_data) if isinstance(semantic_slice_data, (int, float)) else 0.0
        else:
            slice_tensor["semantic"] = 0.0

        tensor[slice_name] = slice_tensor

    # Compute global tensor norm (L2 norm of all drift scores)
    all_scores: List[float] = []
    for slice_tensor in tensor.values():
        all_scores.extend(slice_tensor.values())
    
    global_tensor_norm = math.sqrt(sum(s * s for s in all_scores)) if all_scores else 0.0

    # Rank slices by total drift magnitude (sum of all axis scores)
    slice_totals: List[Tuple[str, float]] = [
        (slice_name, sum(slice_tensor.values()))
        for slice_name, slice_tensor in tensor.items()
    ]
    slice_totals.sort(key=lambda x: x[1], reverse=True)
    ranked_slices = [slice_name for slice_name, _ in slice_totals]

    return {
        "tensor": tensor,
        "global_tensor_norm": round(global_tensor_norm, 6),
        "ranked_slices": ranked_slices,
        "schema_version": DRIFT_TENSOR_SCHEMA_VERSION,
    }


def build_drift_poly_cause_analyzer(
    drift_tensor: Dict[str, Any],
    multi_axis_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect cross-axis causal drift patterns (poly-cause analysis).

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_tensor: Dict from build_drift_tensor.
        multi_axis_view: Dict from build_multi_axis_drift_view.

    Returns:
        Dict with poly-cause analysis:
        - poly_cause_detected: bool
        - cause_vectors: List[Dict] (axis combinations showing correlated drift)
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - notes: List[str] (descriptive notes about detected patterns)
    """
    tensor_data = drift_tensor.get("tensor", {})
    high_risk_axes = set(multi_axis_view.get("high_risk_axes", []))
    axes_with_drift = set(multi_axis_view.get("axes_with_drift", []))

    poly_cause_detected = False
    cause_vectors: List[Dict[str, Any]] = []
    notes: List[str] = []

    # Check for multi-axis drift in same slices
    for slice_name, slice_tensor in tensor_data.items():
        # Count axes with non-zero drift for this slice
        axes_with_drift_in_slice = [
            axis for axis, score in slice_tensor.items()
            if score > 0.0
        ]

        if len(axes_with_drift_in_slice) >= 2:
            poly_cause_detected = True
            cause_vectors.append({
                "slice": slice_name,
                "axes": sorted(axes_with_drift_in_slice),
                "drift_scores": {axis: slice_tensor[axis] for axis in axes_with_drift_in_slice},
            })
            notes.append(
                f"Slice {slice_name}: multiple axes show drift ({', '.join(sorted(axes_with_drift_in_slice))})"
            )

    # Determine risk band
    if len(high_risk_axes) >= 2:
        risk_band = "HIGH"
    elif len(high_risk_axes) == 1 or len(axes_with_drift) >= 2:
        risk_band = "MEDIUM"
    elif poly_cause_detected:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"

    return {
        "poly_cause_detected": poly_cause_detected,
        "cause_vectors": sorted(cause_vectors, key=lambda x: x["slice"]),
        "risk_band": risk_band,
        "notes": sorted(notes),
    }


def build_drift_director_tile_v2(
    global_health_drift: Dict[str, Any],
    trend_history: Dict[str, Any],
    promotion_eval: Dict[str, Any],
    multi_axis_view: Dict[str, Any],
    drift_tensor: Dict[str, Any],
    poly_cause_analysis: Dict[str, Any],
    uplift_envelope: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build enhanced Director Console tile v2 with tensor norms and poly-cause status.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        global_health_drift: Dict from summarize_drift_for_global_health.
        trend_history: Dict from build_drift_trend_history.
        promotion_eval: Dict from evaluate_drift_trend_for_promotion.
        multi_axis_view: Dict from build_multi_axis_drift_view.
        drift_tensor: Dict from build_drift_tensor.
        poly_cause_analysis: Dict from build_drift_poly_cause_analyzer.
        uplift_envelope: Dict from summarize_drift_for_uplift_envelope.

    Returns:
        Dict with enhanced director tile:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - tensor_norm: float
        - poly_cause_status: str
        - risk_band: str
        - uplift_envelope_impact: Dict
        - headline: str
    """
    # Extract key metrics
    global_status = global_health_drift.get("status", "OK")
    promotion_status = promotion_eval.get("status", "OK")
    trend_status = trend_history.get("trend_status", "STABLE")
    tensor_norm = drift_tensor.get("global_tensor_norm", 0.0)
    poly_cause_detected = poly_cause_analysis.get("poly_cause_detected", False)
    risk_band = poly_cause_analysis.get("risk_band", "LOW")
    uplift_status = uplift_envelope.get("status", "OK")

    # Determine status light (enhanced logic)
    if (
        global_status == "HOT"
        or promotion_status == "BLOCK"
        or trend_status == "DEGRADING"
        or risk_band == "HIGH"
        or uplift_status == "BLOCK"
    ):
        status_light = "RED"
    elif (
        global_status == "WARN"
        or promotion_status == "WARN"
        or poly_cause_detected
        or risk_band == "MEDIUM"
        or uplift_status == "ATTENTION"
    ):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Generate headline
    headline_parts: List[str] = []
    
    if tensor_norm > 0.5:
        headline_parts.append(f"Tensor norm: {tensor_norm:.3f}")
    
    if poly_cause_detected:
        headline_parts.append("Poly-cause patterns detected")
    
    if risk_band != "LOW":
        headline_parts.append(f"Risk band: {risk_band}")
    
    if uplift_status != "OK":
        headline_parts.append(f"Uplift envelope: {uplift_status}")

    headline = ". ".join(headline_parts) + "." if headline_parts else "No drift issues detected."

    return {
        "status_light": status_light,
        "tensor_norm": round(tensor_norm, 6),
        "poly_cause_status": "DETECTED" if poly_cause_detected else "NONE",
        "risk_band": risk_band,
        "uplift_envelope_impact": {
            "status": uplift_status,
            "uplift_safe": uplift_envelope.get("uplift_safe_under_drift", True),
            "blocking_axes": uplift_envelope.get("blocking_axes", []),
        },
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# Threshold Snapshot (No-Silent-Change Guard)
# ---------------------------------------------------------------------------

# Frozen snapshot of DRIFT_THRESHOLDS for change detection.
# If you intentionally change DRIFT_THRESHOLDS:
#   1. Update the values in DRIFT_THRESHOLDS above
#   2. Update this snapshot to match
#   3. Run tests to verify the change is intentional
#
# This prevents accidental/silent changes to threshold values.
#
_DRIFT_THRESHOLDS_SNAPSHOT: Dict[str, Dict[str, float]] = {
    "default": {"drift_score": 0.3, "coherence": 0.4},
    "strict": {"drift_score": 0.2, "coherence": 0.5},
    "lenient": {"drift_score": 0.4, "coherence": 0.3},
}


def snapshot_thresholds() -> Dict[str, Any]:
    """
    Return a frozen snapshot of the current DRIFT_THRESHOLDS.

    This is used for no-silent-change testing. The returned snapshot
    should match _DRIFT_THRESHOLDS_SNAPSHOT exactly.

    Returns:
        Dict with current threshold values and metadata.
    """
    return {
        "version": "1.0",
        "profiles": list(sorted(DRIFT_THRESHOLDS.keys())),
        "thresholds": {
            profile: dict(sorted(config.items()))
            for profile, config in sorted(DRIFT_THRESHOLDS.items())
        },
    }


def verify_thresholds_unchanged() -> bool:
    """
    Verify that DRIFT_THRESHOLDS matches the frozen snapshot.

    Returns:
        True if thresholds match snapshot, False otherwise.
    """
    for profile, expected in _DRIFT_THRESHOLDS_SNAPSHOT.items():
        if profile not in DRIFT_THRESHOLDS:
            return False
        current = DRIFT_THRESHOLDS[profile]
        for key, value in expected.items():
            if key not in current or current[key] != value:
                return False
    # Also check for unexpected profiles
    if set(DRIFT_THRESHOLDS.keys()) != set(_DRIFT_THRESHOLDS_SNAPSHOT.keys()):
        return False
    return True


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Result structures
    "MetricDriftSignature",
    "DriftAlignmentResult",
    "StabilityEnvelope",
    "CIGateResult",
    "DriftCell",
    # Core functions
    "compute_drift_alignment",
    "generate_stability_envelope",
    "generate_multi_metric_envelopes",
    # Analysis functions
    "compute_monotonicity",
    "compute_pearson_correlation",
    "analyze_drift_monotonicity",
    # CI/Tooling helpers
    "evaluate_drift_for_ci",
    "classify_pattern_hint",
    "get_drift_thresholds",
    "validate_drift_thresholds",
    # Drift cells grid
    "build_drift_cells",
    "drift_cells_to_dicts",
    "DRIFT_CELL_COLUMNS",
    # Phase III: Promotion Radar & Global Drift Signal
    "summarize_slice_drift",
    "evaluate_drift_for_promotion",
    "summarize_drift_for_global_health",
    # Phase IV: Drift-Aware Promotion & Global Drift Intelligence
    "build_drift_trend_history",
    "evaluate_drift_trend_for_promotion",
    "build_drift_director_panel",
    "CORE_UPLIFT_METRICS",
    # Phase V: Cross-Metric Drift Coupler & Multi-Axis Drift Summary
    "build_multi_axis_drift_view",
    "summarize_drift_for_uplift_envelope",
    "DRIFT_AXES",
    # Phase VI: Drift Tensor & Poly-Cause Analysis
    "build_drift_tensor",
    "build_drift_poly_cause_analyzer",
    "build_drift_director_tile_v2",
    "DRIFT_TENSOR_SCHEMA_VERSION",
    # Threshold snapshot
    "snapshot_thresholds",
    "verify_thresholds_unchanged",
    # Constants
    "PATTERN_HINTS",
    "DRIFT_THRESHOLDS",
    "VALID_PROFILES",
]


def summarize_drift_for_global_health(
    cells: Sequence[DriftCell],
    *,
    drift_score_threshold: float = 0.3,
    coherence_threshold: float = 0.4,
    hotspot_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Summarize drift across all slices for global health monitoring.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        cells: Sequence of DriftCell objects (all slices).
        drift_score_threshold: Threshold for high drift detection (default 0.3).
        coherence_threshold: Threshold for coherence issues (default 0.4).
        hotspot_threshold: Fraction of slices/metrics needed to be a hotspot (default 0.5).

    Returns:
        Dict with global health summary including:
        - drift_hotspot_slices (slices with many non-OK cells)
        - drift_hotspot_metrics (metrics with frequent drift across slices)
        - status: "OK" | "WARN" | "HOT"
    """
    if not cells:
        return {
            "drift_hotspot_slices": [],
            "drift_hotspot_metrics": [],
            "status": "OK",
        }

    # Group by slice and metric
    slices = sorted(set(c.slice for c in cells))
    metrics = sorted(set(c.metric_kind for c in cells))

    # Count non-OK cells per slice
    slice_non_ok_counts: Dict[str, int] = {}
    slice_total_counts: Dict[str, int] = {}

    for slice_name in slices:
        slice_cells = [c for c in cells if c.slice == slice_name]
        slice_total_counts[slice_name] = len(slice_cells)
        slice_non_ok_counts[slice_name] = sum(
            1 for c in slice_cells
            if c.status in ("WARN", "BLOCK") or c.drift_score > drift_score_threshold
        )

    # Count drifty occurrences per metric across slices
    metric_drifty_counts: Dict[str, int] = {}
    metric_total_counts: Dict[str, int] = {}

    for metric_name in metrics:
        metric_cells = [c for c in cells if c.metric_kind == metric_name]
        metric_total_counts[metric_name] = len(metric_cells)
        metric_drifty_counts[metric_name] = sum(
            1 for c in metric_cells
            if c.status in ("WARN", "BLOCK") or c.drift_score > drift_score_threshold
        )

    # Identify hotspot slices (>= hotspot_threshold fraction of cells are non-OK)
    drift_hotspot_slices: List[str] = []
    for slice_name in slices:
        total = slice_total_counts[slice_name]
        non_ok = slice_non_ok_counts[slice_name]
        if total > 0 and (non_ok / total) >= hotspot_threshold:
            drift_hotspot_slices.append(slice_name)

    # Identify hotspot metrics (>= hotspot_threshold fraction of occurrences are drifty)
    drift_hotspot_metrics: List[str] = []
    for metric_name in metrics:
        total = metric_total_counts[metric_name]
        drifty = metric_drifty_counts[metric_name]
        if total > 0 and (drifty / total) >= hotspot_threshold:
            drift_hotspot_metrics.append(metric_name)

    # Determine global status
    has_block = any(c.status == "BLOCK" for c in cells)
    has_hotspots = bool(drift_hotspot_slices or drift_hotspot_metrics)
    has_warn = any(c.status == "WARN" for c in cells)

    if has_block or has_hotspots:
        status = "HOT"
    elif has_warn:
        status = "WARN"
    else:
        status = "OK"

    return {
        "drift_hotspot_slices": sorted(drift_hotspot_slices),
        "drift_hotspot_metrics": sorted(drift_hotspot_metrics),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Threshold Snapshot (No-Silent-Change Guard)
# ---------------------------------------------------------------------------

# Frozen snapshot of DRIFT_THRESHOLDS for change detection.
# If you intentionally change DRIFT_THRESHOLDS:
#   1. Update the values in DRIFT_THRESHOLDS above
#   2. Update this snapshot to match
#   3. Run tests to verify the change is intentional
#
# This prevents accidental/silent changes to threshold values.
#
_DRIFT_THRESHOLDS_SNAPSHOT: Dict[str, Dict[str, float]] = {
    "default": {"drift_score": 0.3, "coherence": 0.4},
    "strict": {"drift_score": 0.2, "coherence": 0.5},
    "lenient": {"drift_score": 0.4, "coherence": 0.3},
}


def snapshot_thresholds() -> Dict[str, Any]:
    """
    Return a frozen snapshot of the current DRIFT_THRESHOLDS.

    This is used for no-silent-change testing. The returned snapshot
    should match _DRIFT_THRESHOLDS_SNAPSHOT exactly.

    Returns:
        Dict with current threshold values and metadata.
    """
    return {
        "version": "1.0",
        "profiles": list(sorted(DRIFT_THRESHOLDS.keys())),
        "thresholds": {
            profile: dict(sorted(config.items()))
            for profile, config in sorted(DRIFT_THRESHOLDS.items())
        },
    }


def verify_thresholds_unchanged() -> bool:
    """
    Verify that DRIFT_THRESHOLDS matches the frozen snapshot.

    Returns:
        True if thresholds match snapshot, False otherwise.
    """
    for profile, expected in _DRIFT_THRESHOLDS_SNAPSHOT.items():
        if profile not in DRIFT_THRESHOLDS:
            return False
        current = DRIFT_THRESHOLDS[profile]
        for key, value in expected.items():
            if key not in current or current[key] != value:
                return False
    # Also check for unexpected profiles
    if set(DRIFT_THRESHOLDS.keys()) != set(_DRIFT_THRESHOLDS_SNAPSHOT.keys()):
        return False
    return True


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Result structures
    "MetricDriftSignature",
    "DriftAlignmentResult",
    "StabilityEnvelope",
    "CIGateResult",
    "DriftCell",
    # Core functions
    "compute_drift_alignment",
    "generate_stability_envelope",
    "generate_multi_metric_envelopes",
    # Analysis functions
    "compute_monotonicity",
    "compute_pearson_correlation",
    "analyze_drift_monotonicity",
    # CI/Tooling helpers
    "evaluate_drift_for_ci",
    "classify_pattern_hint",
    "get_drift_thresholds",
    "validate_drift_thresholds",
    # Drift cells grid
    "build_drift_cells",
    "drift_cells_to_dicts",
    "DRIFT_CELL_COLUMNS",
    # Phase III: Promotion Radar & Global Drift Signal
    "summarize_slice_drift",
    "evaluate_drift_for_promotion",
    "summarize_drift_for_global_health",
    # Phase IV: Drift-Aware Promotion & Global Drift Intelligence
    "build_drift_trend_history",
    "evaluate_drift_trend_for_promotion",
    "build_drift_director_panel",
    "CORE_UPLIFT_METRICS",
    # Phase V: Cross-Metric Drift Coupler & Multi-Axis Drift Summary
    "build_multi_axis_drift_view",
    "summarize_drift_for_uplift_envelope",
    "DRIFT_AXES",
    # Phase VI: Drift Tensor & Poly-Cause Analysis
    "build_drift_tensor",
    "build_drift_poly_cause_analyzer",
    "build_drift_director_tile_v2",
    "DRIFT_TENSOR_SCHEMA_VERSION",
    # Threshold snapshot
    "snapshot_thresholds",
    "verify_thresholds_unchanged",
    # Constants
    "PATTERN_HINTS",
    "DRIFT_THRESHOLDS",
    "VALID_PROFILES",
]

