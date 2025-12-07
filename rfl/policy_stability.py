"""
RFL Policy Stability Module
============================

Monitors policy evolution for stability and drift, providing non-normative
metadata for governance decisions. Detects:
- Long-range oscillation patterns
- Directional divergence from stable trajectories
- Slice-transition drift spikes
- Cross-slice feature flips
- Policy toxicity indicators (concentration, diversity collapse)

All outputs are neutral metadata only - no normative judgments.

Usage:
    from rfl.policy_stability import (
        evaluate_policy_stability,
        StabilityScore,
        detect_slice_coupled_drift,
        detect_policy_toxicity,
        summarize_policy_stability_for_global_health,
    )

    # Evaluate stability across policy evolution
    stability = evaluate_policy_stability(policy_snapshots)
    
    # Detect drift at slice boundaries
    drift_events = detect_slice_coupled_drift(policy_snapshots, curriculum_slices)
    
    # Check for toxicity indicators
    toxicity = detect_policy_toxicity(policy_state)
    
    # Generate governance summary
    summary = summarize_policy_stability_for_global_health(stability, drift_events, toxicity)
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .update_algebra import PolicyState
from .config import CurriculumSlice


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Numerical epsilon for variance and zero comparisons
# Value chosen to be small enough to detect meaningful differences while
# avoiding false positives from floating-point precision issues
EPSILON = 1e-9

# Threshold for negative weight growth to flag divergence
# A 50% increase in negative weight norm is considered significant
NEGATIVE_DIVERGENCE_THRESHOLD = 0.5

# Thresholds for governance red flag counting
# These values balance sensitivity with false positive rate
MAX_SLICE_BOUNDARY_DRIFTS = 5  # Typical curriculum has ~5-10 slices
MAX_FEATURE_FLIPS = 3  # Feature sign flips are rare in stable policies


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

class HealthStatus(Enum):
    """Policy stability health status for governance."""
    OK = "OK"
    WARN = "WARN"
    HOT = "HOT"
    DEGRADED = "DEGRADED"


@dataclass
class OscillationMetrics:
    """Metrics for detecting long-range oscillation."""
    frequency: float  # Dominant oscillation frequency
    amplitude: float  # Oscillation amplitude
    trend_direction: float  # Overall trend direction (-1 to 1)
    autocorrelation: float  # Temporal autocorrelation
    is_oscillating: bool  # True if significant oscillation detected


@dataclass
class DivergenceMetrics:
    """Metrics for detecting directional divergence."""
    drift_rate: float  # Rate of drift per epoch
    cumulative_drift: float  # Total cumulative drift
    acceleration: float  # Second derivative of drift
    is_diverging: bool  # True if divergence exceeds threshold


@dataclass
class StabilityScore:
    """
    Overall policy stability assessment.
    
    Stability score is a value in [0, 1] where:
    - 1.0 = Perfectly stable (no oscillation or drift)
    - 0.0 = Highly unstable (severe oscillation and/or divergence)
    
    Components:
    - oscillation_penalty: Reduction due to oscillation
    - divergence_penalty: Reduction due to directional drift
    - base_score: Starting score before penalties
    """
    score: float  # Overall stability [0, 1]
    oscillation: OscillationMetrics
    divergence: DivergenceMetrics
    base_score: float = 1.0
    oscillation_penalty: float = 0.0
    divergence_penalty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate score is in [0, 1]."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Stability score must be in [0, 1], got {self.score}")


@dataclass
class DriftEvent:
    """
    Drift event detected at slice boundary or within slice.
    
    A drift event represents a significant change in policy behavior,
    potentially correlated with curriculum slice transitions.
    """
    epoch: int
    slice_name: Optional[str]  # Slice where drift occurred (if mapped)
    drift_magnitude: float  # L2 norm of weight change
    is_slice_boundary: bool  # True if at slice transition
    is_feature_flip: bool  # True if any feature changed sign
    flipped_features: List[str]  # Features that flipped sign
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToxicityIndicators:
    """
    Non-normative toxicity indicators for policy health monitoring.
    
    These indicators flag potential issues but do NOT make normative
    judgments. Governance must interpret these signals in context.
    """
    weight_concentration: float  # Gini coefficient of weight distribution [0, 1]
    diversity_score: float  # Effective number of active features
    negative_norm_divergence: float  # Rate of negative weight growth
    variance_ratio: float  # Ratio of recent to historical variance
    has_extreme_concentration: bool  # True if concentration > threshold
    has_diversity_collapse: bool  # True if diversity < threshold
    has_negative_divergence: bool  # True if negative weights growing
    has_high_variance: bool  # True if variance spike detected
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyStabilitySummary:
    """
    Complete policy stability summary for governance.
    
    Aggregates stability, drift, and toxicity indicators into a
    single governance-ready report with health status.
    """
    health_status: HealthStatus
    stability_score: StabilityScore
    drift_events: List[DriftEvent]
    toxicity: ToxicityIndicators
    epoch_range: Tuple[int, int]  # (start_epoch, end_epoch)
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "health_status": self.health_status.value,
            "stability_score": {
                "score": self.stability_score.score,
                "oscillation": asdict(self.stability_score.oscillation),
                "divergence": asdict(self.stability_score.divergence),
                "base_score": self.stability_score.base_score,
                "oscillation_penalty": self.stability_score.oscillation_penalty,
                "divergence_penalty": self.stability_score.divergence_penalty,
                "metadata": self.stability_score.metadata,
            },
            "drift_events": [asdict(e) for e in self.drift_events],
            "toxicity": asdict(self.toxicity),
            "epoch_range": self.epoch_range,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# -----------------------------------------------------------------------------
# Stability Evaluation
# -----------------------------------------------------------------------------

def evaluate_policy_stability(
    snapshot_series: List[PolicyState],
    oscillation_threshold: float = 0.3,
    divergence_threshold: float = 0.5,
) -> StabilityScore:
    """
    Evaluate policy stability across a time series of policy snapshots.
    
    Detects:
    - Long-range oscillation patterns (frequency, amplitude)
    - Directional divergence from stable trajectory
    - Overall stability score [0, 1]
    
    Args:
        snapshot_series: Ordered list of PolicyState snapshots
        oscillation_threshold: Threshold for oscillation detection
        divergence_threshold: Threshold for divergence detection
    
    Returns:
        StabilityScore with oscillation/divergence metrics
    
    Raises:
        ValueError: If snapshot_series is empty or has < 2 snapshots
    """
    if not snapshot_series:
        raise ValueError("snapshot_series cannot be empty")
    
    if len(snapshot_series) < 2:
        # Trivially stable with single snapshot
        return StabilityScore(
            score=1.0,
            oscillation=OscillationMetrics(
                frequency=0.0,
                amplitude=0.0,
                trend_direction=0.0,
                autocorrelation=0.0,
                is_oscillating=False,
            ),
            divergence=DivergenceMetrics(
                drift_rate=0.0,
                cumulative_drift=0.0,
                acceleration=0.0,
                is_diverging=False,
            ),
        )
    
    # Extract weight vectors for all features
    all_features = sorted(snapshot_series[0].weights.keys())
    weight_matrix = np.array([
        [state.weights.get(f, 0.0) for f in all_features]
        for state in snapshot_series
    ])
    
    # Compute oscillation metrics
    oscillation = _compute_oscillation_metrics(weight_matrix, oscillation_threshold)
    
    # Compute divergence metrics
    divergence = _compute_divergence_metrics(weight_matrix, divergence_threshold)
    
    # Compute stability score with penalties
    base_score = 1.0
    oscillation_penalty = oscillation.amplitude * 0.5  # Scale by amplitude
    divergence_penalty = min(abs(divergence.drift_rate) * 2.0, 0.5)  # Cap at 0.5
    
    stability_score = max(0.0, base_score - oscillation_penalty - divergence_penalty)
    
    return StabilityScore(
        score=stability_score,
        oscillation=oscillation,
        divergence=divergence,
        base_score=base_score,
        oscillation_penalty=oscillation_penalty,
        divergence_penalty=divergence_penalty,
        metadata={
            "num_snapshots": len(snapshot_series),
            "num_features": len(all_features),
            "epoch_range": (snapshot_series[0].epoch, snapshot_series[-1].epoch),
        },
    )


def _compute_oscillation_metrics(
    weight_matrix: np.ndarray,
    threshold: float,
) -> OscillationMetrics:
    """
    Compute oscillation metrics from weight time series.
    
    Uses FFT to detect dominant frequencies and autocorrelation
    to measure temporal structure.
    
    Args:
        weight_matrix: (T, F) array of weights over time
        threshold: Threshold for oscillation detection
    
    Returns:
        OscillationMetrics
    """
    T, F = weight_matrix.shape
    
    # Compute L2 norm trajectory over time
    trajectory = np.linalg.norm(weight_matrix, axis=1)
    
    # Detrend by removing linear trend
    time_idx = np.arange(T)
    coeffs = np.polyfit(time_idx, trajectory, deg=1)
    trend_line = np.polyval(coeffs, time_idx)
    detrended = trajectory - trend_line
    
    # Compute FFT to detect dominant frequency
    fft = np.fft.fft(detrended)
    power = np.abs(fft[:T // 2]) ** 2
    
    # Find dominant frequency (excluding DC component)
    if len(power) > 1:
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = dominant_idx / T
    else:
        dominant_freq = 0.0
    
    # Compute amplitude as standard deviation of detrended signal
    amplitude = np.std(detrended)
    
    # Compute trend direction
    trend_direction = np.sign(coeffs[0])  # Slope of trend line
    
    # Compute autocorrelation at lag 1
    if T > 1:
        autocorr = np.corrcoef(detrended[:-1], detrended[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Detect oscillation
    is_oscillating = amplitude > threshold
    
    return OscillationMetrics(
        frequency=float(dominant_freq),
        amplitude=float(amplitude),
        trend_direction=float(trend_direction),
        autocorrelation=float(autocorr),
        is_oscillating=bool(is_oscillating),
    )


def _compute_divergence_metrics(
    weight_matrix: np.ndarray,
    threshold: float,
) -> DivergenceMetrics:
    """
    Compute directional divergence metrics.
    
    Measures the rate at which policy weights drift from initial state.
    
    Args:
        weight_matrix: (T, F) array of weights over time
        threshold: Threshold for divergence detection
    
    Returns:
        DivergenceMetrics
    """
    T, F = weight_matrix.shape
    
    # Compute cumulative drift from initial state
    initial_weights = weight_matrix[0]
    drift_norms = np.linalg.norm(weight_matrix - initial_weights, axis=1)
    
    # Compute drift rate via linear regression
    time_idx = np.arange(T)
    if T > 1:
        coeffs = np.polyfit(time_idx, drift_norms, deg=1)
        drift_rate = coeffs[0]
    else:
        drift_rate = 0.0
    
    # Cumulative drift is final distance from initial state
    cumulative_drift = float(drift_norms[-1])
    
    # Compute acceleration (second derivative)
    if T > 2:
        # Fit quadratic and extract second-order coefficient
        coeffs_quad = np.polyfit(time_idx, drift_norms, deg=2)
        acceleration = 2 * coeffs_quad[0]  # Second derivative
    else:
        acceleration = 0.0
    
    # Detect divergence
    is_diverging = abs(drift_rate) > threshold
    
    return DivergenceMetrics(
        drift_rate=float(drift_rate),
        cumulative_drift=cumulative_drift,
        acceleration=float(acceleration),
        is_diverging=bool(is_diverging),
    )


# -----------------------------------------------------------------------------
# Slice-Coupled Drift Detection
# -----------------------------------------------------------------------------

def detect_slice_coupled_drift(
    snapshot_series: List[PolicyState],
    curriculum_slices: List[CurriculumSlice],
    drift_threshold: float = 0.1,
) -> List[DriftEvent]:
    """
    Detect drift events and correlate with curriculum slice boundaries.
    
    Maps drift events to curriculum slices and identifies:
    - Slice-transition drift spikes
    - Cross-slice feature flips (sign changes)
    
    Args:
        snapshot_series: Ordered list of PolicyState snapshots
        curriculum_slices: Curriculum slice definitions
        drift_threshold: Minimum magnitude for drift detection
    
    Returns:
        List of DriftEvent objects
    """
    if not snapshot_series or len(snapshot_series) < 2:
        return []
    
    drift_events = []
    
    # Build slice map: epoch -> slice_name
    slice_map = {}
    for slice_obj in curriculum_slices:
        for run_idx in range(slice_obj.start_run, slice_obj.end_run + 1):
            # Assuming 1 epoch per run (adjust if needed)
            slice_map[run_idx - 1] = slice_obj.name
    
    # Detect slice boundaries
    slice_boundaries = set()
    for i, slice_obj in enumerate(curriculum_slices):
        if i > 0:
            # Boundary between previous slice end and this slice start
            slice_boundaries.add(slice_obj.start_run - 1)
    
    # Compute drift between consecutive snapshots
    for i in range(1, len(snapshot_series)):
        prev_state = snapshot_series[i - 1]
        curr_state = snapshot_series[i]
        
        # Compute weight deltas
        all_features = sorted(prev_state.weights.keys())
        deltas = {
            f: curr_state.weights.get(f, 0.0) - prev_state.weights.get(f, 0.0)
            for f in all_features
        }
        
        # Compute drift magnitude (L2 norm)
        drift_magnitude = np.sqrt(sum(d ** 2 for d in deltas.values()))
        
        # Check for feature flips (sign changes)
        flipped_features = []
        for f in all_features:
            prev_val = prev_state.weights.get(f, 0.0)
            curr_val = curr_state.weights.get(f, 0.0)
            if prev_val * curr_val < 0:  # Sign flip
                flipped_features.append(f)
        
        is_feature_flip = len(flipped_features) > 0
        
        # Map to slice
        epoch = curr_state.epoch
        slice_name = slice_map.get(epoch)
        
        # Check if at slice boundary
        is_slice_boundary = epoch in slice_boundaries
        
        # Record drift event if magnitude exceeds threshold
        if drift_magnitude >= drift_threshold or is_feature_flip or is_slice_boundary:
            drift_events.append(DriftEvent(
                epoch=epoch,
                slice_name=slice_name,
                drift_magnitude=float(drift_magnitude),
                is_slice_boundary=is_slice_boundary,
                is_feature_flip=is_feature_flip,
                flipped_features=flipped_features,
                metadata={
                    "prev_epoch": prev_state.epoch,
                    "deltas": {f: float(d) for f, d in deltas.items()},
                },
            ))
    
    return drift_events


# -----------------------------------------------------------------------------
# Policy Toxicity Detection
# -----------------------------------------------------------------------------

def detect_policy_toxicity(
    policy_state: PolicyState,
    historical_snapshots: Optional[List[PolicyState]] = None,
    concentration_threshold: float = 0.8,
    diversity_threshold: float = 2.0,
    variance_spike_threshold: float = 3.0,
) -> ToxicityIndicators:
    """
    Detect policy toxicity indicators (non-normative metadata only).
    
    Checks for:
    - Extreme weight concentration (Gini coefficient)
    - Collapse of diversity (effective number of features)
    - Negative norm divergence (negative weights growing)
    - High-variance transitions (variance spike)
    
    Args:
        policy_state: Current policy state to evaluate
        historical_snapshots: Optional historical snapshots for variance comparison
        concentration_threshold: Threshold for extreme concentration
        diversity_threshold: Threshold for diversity collapse
        variance_spike_threshold: Threshold for variance spike detection
    
    Returns:
        ToxicityIndicators with neutral metadata
    """
    weights = np.array(list(policy_state.weights.values()))
    
    # Compute weight concentration (Gini coefficient)
    weight_concentration = _compute_gini_coefficient(np.abs(weights))
    
    # Compute diversity score (effective number of features)
    diversity_score = _compute_effective_features(weights)
    
    # Compute negative norm divergence
    negative_weights = weights[weights < 0]
    if len(negative_weights) > 0:
        negative_norm = np.linalg.norm(negative_weights)
    else:
        negative_norm = 0.0
    
    # Compute variance ratio if historical snapshots provided
    if historical_snapshots and len(historical_snapshots) > 1:
        # Compute variance within each snapshot
        historical_vars = [np.var(list(s.weights.values())) for s in historical_snapshots]
        historical_var = np.mean(historical_vars)
        current_var = np.var(weights)
        
        # If historical variance is near zero, use epsilon to avoid division by zero
        if historical_var > EPSILON:
            variance_ratio = current_var / historical_var
        else:
            # If historical has no variance but current does, ratio is large
            if current_var > EPSILON:
                variance_ratio = current_var / EPSILON  # Effectively infinite ratio
            else:
                variance_ratio = 1.0
    else:
        variance_ratio = 1.0
    
    # Compute rate of negative norm growth (requires historical data)
    if historical_snapshots and len(historical_snapshots) > 1:
        prev_weights = np.array(list(historical_snapshots[-1].weights.values()))
        prev_negative = prev_weights[prev_weights < 0]
        prev_negative_norm = np.linalg.norm(prev_negative) if len(prev_negative) > 0 else 0.0
        
        if prev_negative_norm > EPSILON:
            negative_norm_divergence = (negative_norm - prev_negative_norm) / prev_negative_norm
        else:
            negative_norm_divergence = 0.0
    else:
        negative_norm_divergence = 0.0
    
    # Flag indicators
    has_extreme_concentration = weight_concentration > concentration_threshold
    has_diversity_collapse = diversity_score < diversity_threshold
    has_negative_divergence = negative_norm_divergence > NEGATIVE_DIVERGENCE_THRESHOLD
    has_high_variance = variance_ratio > variance_spike_threshold
    
    return ToxicityIndicators(
        weight_concentration=float(weight_concentration),
        diversity_score=float(diversity_score),
        negative_norm_divergence=float(negative_norm_divergence),
        variance_ratio=float(variance_ratio),
        has_extreme_concentration=has_extreme_concentration,
        has_diversity_collapse=has_diversity_collapse,
        has_negative_divergence=has_negative_divergence,
        has_high_variance=has_high_variance,
        metadata={
            "num_features": len(weights),
            "num_negative": int(np.sum(weights < 0)),
            "num_positive": int(np.sum(weights > 0)),
            "negative_norm": float(negative_norm),
        },
    )


def _compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for weight distribution.
    
    Gini coefficient measures inequality in distribution:
    - 0.0 = Perfect equality (all weights equal)
    - 1.0 = Perfect inequality (one weight has everything)
    
    Args:
        values: Array of non-negative values
    
    Returns:
        Gini coefficient in [0, 1]
    """
    if len(values) == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_values)
    sum_of_values = cumsum[-1]
    
    if sum_of_values == 0:
        return 0.0
    
    # Gini = (2 * sum of (i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    weighted_sum = np.sum((np.arange(n) + 1) * sorted_values)
    gini = (2 * weighted_sum) / (n * sum_of_values) - (n + 1) / n
    
    return float(gini)


def _compute_effective_features(weights: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute effective number of features (diversity score).
    
    Uses exponential of Shannon entropy to measure diversity:
    - High value = Many features contributing
    - Low value = Few features dominating
    
    Args:
        weights: Array of weight values
        epsilon: Regularization for log (default: EPSILON)
    
    Returns:
        Effective number of features
    """
    if len(weights) == 0:
        return 0.0
    
    # Normalize weights to probabilities
    abs_weights = np.abs(weights)
    total = np.sum(abs_weights)
    
    if total < epsilon:
        return 0.0
    
    probs = abs_weights / total
    
    # Compute Shannon entropy
    log_probs = np.log(probs + epsilon)
    entropy = -np.sum(probs * log_probs)
    
    # Effective number of features = exp(entropy)
    effective_n = np.exp(entropy)
    
    return float(effective_n)


# -----------------------------------------------------------------------------
# Governance Hook
# -----------------------------------------------------------------------------

def summarize_policy_stability_for_global_health(
    stability_score: StabilityScore,
    drift_events: List[DriftEvent],
    toxicity: ToxicityIndicators,
    timestamp: str = "",
) -> PolicyStabilitySummary:
    """
    Generate governance-ready summary of policy stability.
    
    Maps stability metrics to health status:
    - OK: Stable policy, no concerning indicators
    - WARN: Minor instability or drift detected
    - HOT: Significant instability or toxicity indicators
    - DEGRADED: Severe instability or multiple red flags
    
    Args:
        stability_score: StabilityScore from evaluate_policy_stability
        drift_events: List of DriftEvent from detect_slice_coupled_drift
        toxicity: ToxicityIndicators from detect_policy_toxicity
        timestamp: Timestamp of summary generation
    
    Returns:
        PolicyStabilitySummary with health status
    """
    # Determine health status based on multiple signals
    red_flags = 0
    
    # Check stability score
    if stability_score.score < 0.3:
        red_flags += 2  # Severe instability
    elif stability_score.score < 0.6:
        red_flags += 1  # Moderate instability
    
    # Check oscillation
    if stability_score.oscillation.is_oscillating:
        red_flags += 1
    
    # Check divergence
    if stability_score.divergence.is_diverging:
        red_flags += 1
    
    # Check drift events
    slice_boundary_drifts = sum(1 for e in drift_events if e.is_slice_boundary)
    feature_flips = sum(1 for e in drift_events if e.is_feature_flip)
    
    if slice_boundary_drifts > MAX_SLICE_BOUNDARY_DRIFTS:
        red_flags += 1
    if feature_flips > MAX_FEATURE_FLIPS:
        red_flags += 1
    
    # Check toxicity indicators
    if toxicity.has_extreme_concentration:
        red_flags += 1
    if toxicity.has_diversity_collapse:
        red_flags += 1
    if toxicity.has_negative_divergence:
        red_flags += 1
    if toxicity.has_high_variance:
        red_flags += 1
    
    # Map red flags to health status
    if red_flags == 0:
        health_status = HealthStatus.OK
    elif red_flags <= 2:
        health_status = HealthStatus.WARN
    elif red_flags <= 4:
        health_status = HealthStatus.HOT
    else:
        health_status = HealthStatus.DEGRADED
    
    # Determine epoch range
    if hasattr(stability_score, 'metadata') and 'epoch_range' in stability_score.metadata:
        epoch_range = stability_score.metadata['epoch_range']
    else:
        epoch_range = (0, 0)
    
    return PolicyStabilitySummary(
        health_status=health_status,
        stability_score=stability_score,
        drift_events=drift_events,
        toxicity=toxicity,
        epoch_range=epoch_range,
        timestamp=timestamp,
        metadata={
            "red_flags": red_flags,
            "slice_boundary_drifts": slice_boundary_drifts,
            "feature_flips": feature_flips,
        },
    )
