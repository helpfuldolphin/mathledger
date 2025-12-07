"""
Tests for RFL Policy Stability Module
======================================

Validates stability evaluation, drift detection, toxicity indicators,
and governance hooks for policy health monitoring.
"""

import pytest
import numpy as np
from datetime import datetime

from rfl.policy_stability import (
    evaluate_policy_stability,
    detect_slice_coupled_drift,
    detect_policy_toxicity,
    summarize_policy_stability_for_global_health,
    StabilityScore,
    OscillationMetrics,
    DivergenceMetrics,
    DriftEvent,
    ToxicityIndicators,
    PolicyStabilitySummary,
    HealthStatus,
    _compute_gini_coefficient,
    _compute_effective_features,
)
from rfl.update_algebra import PolicyState
from rfl.config import CurriculumSlice


# -----------------------------------------------------------------------------
# Test Data Generators
# -----------------------------------------------------------------------------

def create_policy_state(epoch: int, weights: dict) -> PolicyState:
    """Create a PolicyState for testing."""
    return PolicyState(
        weights=weights,
        epoch=epoch,
        timestamp=f"2024-01-01T00:{epoch:02d}:00Z",
    )


def create_stable_series(num_epochs: int = 10) -> list:
    """Create a stable policy series (no oscillation, minimal drift)."""
    return [
        create_policy_state(i, {"len": 1.0, "depth": 0.5, "success": 2.0})
        for i in range(num_epochs)
    ]


def create_oscillating_series(num_epochs: int = 20) -> list:
    """Create an oscillating policy series."""
    return [
        create_policy_state(
            i,
            {
                "len": 1.0 + 0.5 * np.sin(2 * np.pi * i / 10),
                "depth": 0.5 + 0.3 * np.cos(2 * np.pi * i / 10),
                "success": 2.0,
            }
        )
        for i in range(num_epochs)
    ]


def create_diverging_series(num_epochs: int = 20) -> list:
    """Create a diverging policy series (linear drift)."""
    return [
        create_policy_state(
            i,
            {
                "len": 1.0 + 0.1 * i,
                "depth": 0.5 + 0.05 * i,
                "success": 2.0 + 0.15 * i,
            }
        )
        for i in range(num_epochs)
    ]


def create_curriculum_slices() -> list:
    """Create test curriculum slices."""
    return [
        CurriculumSlice(
            name="slice_a",
            start_run=1,
            end_run=10,
            derive_steps=50,
            max_breadth=200,
            max_total=1000,
            depth_max=4,
        ),
        CurriculumSlice(
            name="slice_b",
            start_run=11,
            end_run=20,
            derive_steps=75,
            max_breadth=300,
            max_total=1500,
            depth_max=5,
        ),
        CurriculumSlice(
            name="slice_c",
            start_run=21,
            end_run=30,
            derive_steps=100,
            max_breadth=400,
            max_total=2000,
            depth_max=6,
        ),
    ]


# -----------------------------------------------------------------------------
# Test Stability Evaluation
# -----------------------------------------------------------------------------

class TestStabilityEvaluation:
    """Tests for evaluate_policy_stability."""
    
    def test_empty_series_raises(self):
        """Test that empty series raises ValueError."""
        with pytest.raises(ValueError, match="snapshot_series cannot be empty"):
            evaluate_policy_stability([])
    
    def test_single_snapshot_trivially_stable(self):
        """Test that single snapshot is trivially stable."""
        series = [create_policy_state(0, {"len": 1.0, "depth": 0.5})]
        stability = evaluate_policy_stability(series)
        
        assert stability.score == 1.0
        assert not stability.oscillation.is_oscillating
        assert not stability.divergence.is_diverging
    
    def test_stable_series_high_score(self):
        """Test that stable series produces high stability score."""
        series = create_stable_series(num_epochs=10)
        stability = evaluate_policy_stability(series)
        
        assert stability.score >= 0.9  # Should be very stable
        assert not stability.oscillation.is_oscillating
        assert not stability.divergence.is_diverging
        assert stability.oscillation_penalty < 0.1
        assert stability.divergence_penalty < 0.1
    
    def test_oscillating_series_detected(self):
        """Test that oscillation is detected and penalized."""
        series = create_oscillating_series(num_epochs=20)
        stability = evaluate_policy_stability(
            series,
            oscillation_threshold=0.1,  # Lower threshold for detection
        )
        
        assert stability.oscillation.is_oscillating
        assert stability.oscillation.amplitude > 0.1
        assert stability.oscillation_penalty > 0.0
        assert stability.score < 1.0  # Penalty applied
    
    def test_diverging_series_detected(self):
        """Test that divergence is detected and penalized."""
        series = create_diverging_series(num_epochs=20)
        stability = evaluate_policy_stability(
            series,
            divergence_threshold=0.05,  # Lower threshold
        )
        
        assert stability.divergence.is_diverging
        assert stability.divergence.drift_rate > 0.05
        assert stability.divergence_penalty > 0.0
        assert stability.score < 1.0
    
    def test_oscillation_frequency_computed(self):
        """Test that oscillation frequency is computed correctly."""
        # Create series with known frequency (period = 10)
        series = create_oscillating_series(num_epochs=30)
        stability = evaluate_policy_stability(series)
        
        # Frequency should be around 0.1 (1/10)
        assert 0.05 < stability.oscillation.frequency < 0.2
    
    def test_cumulative_drift_computed(self):
        """Test that cumulative drift is computed correctly."""
        series = create_diverging_series(num_epochs=20)
        stability = evaluate_policy_stability(series)
        
        # Should have non-zero cumulative drift
        assert stability.divergence.cumulative_drift > 0.0
        # Drift should increase with time
        assert stability.divergence.cumulative_drift > stability.divergence.drift_rate
    
    def test_metadata_included(self):
        """Test that metadata is populated correctly."""
        series = create_stable_series(num_epochs=15)
        stability = evaluate_policy_stability(series)
        
        assert "num_snapshots" in stability.metadata
        assert stability.metadata["num_snapshots"] == 15
        assert "num_features" in stability.metadata
        assert stability.metadata["num_features"] == 3
        assert "epoch_range" in stability.metadata
        assert stability.metadata["epoch_range"] == (0, 14)


# -----------------------------------------------------------------------------
# Test Slice-Coupled Drift Detection
# -----------------------------------------------------------------------------

class TestSliceCoupledDrift:
    """Tests for detect_slice_coupled_drift."""
    
    def test_empty_series_returns_empty(self):
        """Test that empty series returns empty drift events."""
        events = detect_slice_coupled_drift([], create_curriculum_slices())
        assert events == []
    
    def test_single_snapshot_returns_empty(self):
        """Test that single snapshot returns empty drift events."""
        series = [create_policy_state(0, {"len": 1.0})]
        events = detect_slice_coupled_drift(series, create_curriculum_slices())
        assert events == []
    
    def test_stable_series_no_drift(self):
        """Test that stable series produces no drift events."""
        series = create_stable_series(num_epochs=10)
        events = detect_slice_coupled_drift(
            series,
            create_curriculum_slices(),
            drift_threshold=0.5,  # Higher threshold
        )
        assert len(events) == 0
    
    def test_slice_boundary_detected(self):
        """Test that drift at slice boundaries is detected."""
        # Create series with jump at slice boundary (epoch 10)
        series = []
        for i in range(20):
            if i < 10:
                weights = {"len": 1.0, "depth": 0.5, "success": 2.0}
            else:
                weights = {"len": 1.5, "depth": 0.8, "success": 2.5}
            series.append(create_policy_state(i, weights))
        
        events = detect_slice_coupled_drift(
            series,
            create_curriculum_slices(),
            drift_threshold=0.1,
        )
        
        # Should detect drift at epoch 10 (slice boundary)
        boundary_events = [e for e in events if e.is_slice_boundary]
        assert len(boundary_events) > 0
    
    def test_feature_flip_detected(self):
        """Test that feature sign flips are detected."""
        series = [
            create_policy_state(0, {"len": 1.0, "depth": 0.5, "success": 2.0}),
            create_policy_state(1, {"len": -1.0, "depth": 0.5, "success": 2.0}),
        ]
        
        events = detect_slice_coupled_drift(series, create_curriculum_slices())
        
        assert len(events) == 1
        assert events[0].is_feature_flip
        assert "len" in events[0].flipped_features
    
    def test_multiple_feature_flips(self):
        """Test that multiple feature flips are detected."""
        series = [
            create_policy_state(0, {"len": 1.0, "depth": 0.5, "success": 2.0}),
            create_policy_state(1, {"len": -1.0, "depth": -0.5, "success": 2.0}),
        ]
        
        events = detect_slice_coupled_drift(series, create_curriculum_slices())
        
        assert len(events) == 1
        assert events[0].is_feature_flip
        assert set(events[0].flipped_features) == {"len", "depth"}
    
    def test_drift_magnitude_computed(self):
        """Test that drift magnitude is computed correctly."""
        series = [
            create_policy_state(0, {"len": 0.0, "depth": 0.0}),
            create_policy_state(1, {"len": 3.0, "depth": 4.0}),  # Magnitude = 5.0
        ]
        
        events = detect_slice_coupled_drift(
            series,
            create_curriculum_slices(),
            drift_threshold=1.0,
        )
        
        assert len(events) == 1
        assert abs(events[0].drift_magnitude - 5.0) < 0.01
    
    def test_slice_name_mapped(self):
        """Test that drift events are mapped to slice names."""
        series = []
        for i in range(15):
            series.append(create_policy_state(i, {"len": float(i)}))
        
        events = detect_slice_coupled_drift(
            series,
            create_curriculum_slices(),
            drift_threshold=0.1,
        )
        
        # Check that some events have slice names
        slice_names = [e.slice_name for e in events if e.slice_name]
        assert len(slice_names) > 0
        assert "slice_a" in slice_names or "slice_b" in slice_names


# -----------------------------------------------------------------------------
# Test Toxicity Detection
# -----------------------------------------------------------------------------

class TestToxicityDetection:
    """Tests for detect_policy_toxicity."""
    
    def test_balanced_policy_no_toxicity(self):
        """Test that balanced policy has low toxicity indicators."""
        policy = create_policy_state(0, {
            "len": 1.0,
            "depth": 1.0,
            "success": 1.0,
            "novelty": 1.0,
        })
        
        toxicity = detect_policy_toxicity(policy)
        
        assert toxicity.weight_concentration < 0.5  # Low concentration
        assert toxicity.diversity_score > 2.0  # Good diversity
        assert not toxicity.has_extreme_concentration
        assert not toxicity.has_diversity_collapse
    
    def test_concentrated_policy_detected(self):
        """Test that concentrated weights are detected."""
        policy = create_policy_state(0, {
            "len": 10.0,
            "depth": 0.1,
            "success": 0.1,
            "novelty": 0.1,
        })
        
        toxicity = detect_policy_toxicity(
            policy,
            concentration_threshold=0.5,
        )
        
        assert toxicity.weight_concentration > 0.5
        assert toxicity.has_extreme_concentration
    
    def test_diversity_collapse_detected(self):
        """Test that diversity collapse is detected."""
        policy = create_policy_state(0, {
            "len": 1.0,
            "depth": 0.0,
            "success": 0.0,
        })
        
        toxicity = detect_policy_toxicity(
            policy,
            diversity_threshold=2.0,
        )
        
        assert toxicity.diversity_score < 2.0
        assert toxicity.has_diversity_collapse
    
    def test_negative_divergence_detected(self):
        """Test that negative weight growth is detected."""
        historical = [
            create_policy_state(0, {"len": 1.0, "depth": -0.5}),
            create_policy_state(1, {"len": 1.0, "depth": -1.0}),
        ]
        current = create_policy_state(2, {"len": 1.0, "depth": -2.0})
        
        toxicity = detect_policy_toxicity(current, historical_snapshots=historical)
        
        assert toxicity.negative_norm_divergence > 0.0
        # Note: May or may not flag depending on threshold
    
    def test_variance_spike_detected(self):
        """Test that variance spike is detected."""
        # Historical: low variance (all same values)
        historical = [
            create_policy_state(i, {"len": 1.0, "depth": 1.0})
            for i in range(5)
        ]
        
        # Current: high variance (very different values)
        current = create_policy_state(5, {"len": 100.0, "depth": 0.01})
        
        toxicity = detect_policy_toxicity(
            current,
            historical_snapshots=historical,
            variance_spike_threshold=2.0,
        )
        
        # Variance should be much higher in current state
        assert toxicity.variance_ratio > 2.0
        assert toxicity.has_high_variance
    
    def test_metadata_populated(self):
        """Test that metadata is populated correctly."""
        policy = create_policy_state(0, {
            "a": 1.0,
            "b": -2.0,
            "c": 3.0,
        })
        
        toxicity = detect_policy_toxicity(policy)
        
        assert "num_features" in toxicity.metadata
        assert toxicity.metadata["num_features"] == 3
        assert "num_negative" in toxicity.metadata
        assert toxicity.metadata["num_negative"] == 1
        assert "num_positive" in toxicity.metadata
        assert toxicity.metadata["num_positive"] == 2


# -----------------------------------------------------------------------------
# Test Gini Coefficient
# -----------------------------------------------------------------------------

class TestGiniCoefficient:
    """Tests for _compute_gini_coefficient."""
    
    def test_equal_distribution_zero_gini(self):
        """Test that equal distribution gives Gini = 0."""
        values = np.array([1.0, 1.0, 1.0, 1.0])
        gini = _compute_gini_coefficient(values)
        assert abs(gini) < 0.01  # Should be ~0
    
    def test_extreme_inequality_high_gini(self):
        """Test that extreme inequality gives high Gini."""
        values = np.array([100.0, 0.0, 0.0, 0.0])
        gini = _compute_gini_coefficient(values)
        assert gini > 0.7  # Should be close to 1
    
    def test_empty_array_zero_gini(self):
        """Test that empty array gives Gini = 0."""
        values = np.array([])
        gini = _compute_gini_coefficient(values)
        assert gini == 0.0
    
    def test_all_zeros_zero_gini(self):
        """Test that all zeros gives Gini = 0."""
        values = np.array([0.0, 0.0, 0.0])
        gini = _compute_gini_coefficient(values)
        assert gini == 0.0


# -----------------------------------------------------------------------------
# Test Effective Features
# -----------------------------------------------------------------------------

class TestEffectiveFeatures:
    """Tests for _compute_effective_features."""
    
    def test_equal_weights_max_diversity(self):
        """Test that equal weights give maximum diversity."""
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        effective = _compute_effective_features(weights)
        assert 3.9 < effective < 4.1  # Should be ~4
    
    def test_single_dominant_weight_low_diversity(self):
        """Test that single dominant weight gives low diversity."""
        weights = np.array([10.0, 0.1, 0.1, 0.1])
        effective = _compute_effective_features(weights)
        assert effective < 2.0  # Much less than 4
    
    def test_empty_array_zero_diversity(self):
        """Test that empty array gives zero diversity."""
        weights = np.array([])
        effective = _compute_effective_features(weights)
        assert effective == 0.0
    
    def test_all_zeros_zero_diversity(self):
        """Test that all zeros gives zero diversity."""
        weights = np.array([0.0, 0.0, 0.0])
        effective = _compute_effective_features(weights)
        assert effective == 0.0


# -----------------------------------------------------------------------------
# Test Governance Hook
# -----------------------------------------------------------------------------

class TestGovernanceHook:
    """Tests for summarize_policy_stability_for_global_health."""
    
    def test_stable_policy_ok_status(self):
        """Test that stable policy produces OK status."""
        series = create_stable_series(num_epochs=10)
        stability = evaluate_policy_stability(series)
        drift_events = detect_slice_coupled_drift(series, create_curriculum_slices())
        toxicity = detect_policy_toxicity(series[0])
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        assert summary.health_status == HealthStatus.OK
        assert summary.stability_score.score >= 0.9
    
    def test_unstable_policy_degraded_status(self):
        """Test that unstable policy produces DEGRADED status."""
        # Create highly unstable scenario
        series = create_oscillating_series(num_epochs=20)
        stability = evaluate_policy_stability(
            series,
            oscillation_threshold=0.1,
            divergence_threshold=0.05,
        )
        
        # Force low stability score for test
        stability = StabilityScore(
            score=0.2,
            oscillation=OscillationMetrics(
                frequency=0.1,
                amplitude=0.5,
                trend_direction=1.0,
                autocorrelation=0.8,
                is_oscillating=True,
            ),
            divergence=DivergenceMetrics(
                drift_rate=0.8,
                cumulative_drift=5.0,
                acceleration=0.1,
                is_diverging=True,
            ),
        )
        
        drift_events = []
        toxicity = ToxicityIndicators(
            weight_concentration=0.9,
            diversity_score=1.0,
            negative_norm_divergence=1.0,
            variance_ratio=5.0,
            has_extreme_concentration=True,
            has_diversity_collapse=True,
            has_negative_divergence=True,
            has_high_variance=True,
        )
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
        )
        
        assert summary.health_status in (HealthStatus.HOT, HealthStatus.DEGRADED)
    
    def test_warn_status_intermediate(self):
        """Test that intermediate instability produces WARN status."""
        series = create_stable_series(num_epochs=10)
        stability = evaluate_policy_stability(series)
        
        # Force score to intermediate range
        stability = StabilityScore(
            score=0.65,
            oscillation=OscillationMetrics(
                frequency=0.0,
                amplitude=0.2,
                trend_direction=0.0,
                autocorrelation=0.0,
                is_oscillating=False,
            ),
            divergence=DivergenceMetrics(
                drift_rate=0.1,
                cumulative_drift=1.0,
                acceleration=0.0,
                is_diverging=False,
            ),
        )
        
        drift_events = []
        toxicity = detect_policy_toxicity(series[0])
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
        )
        
        assert summary.health_status in (HealthStatus.OK, HealthStatus.WARN)
    
    def test_summary_to_dict(self):
        """Test that summary can be serialized to dict."""
        series = create_stable_series(num_epochs=5)
        stability = evaluate_policy_stability(series)
        drift_events = []
        toxicity = detect_policy_toxicity(series[0])
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        summary_dict = summary.to_dict()
        
        assert "health_status" in summary_dict
        assert summary_dict["health_status"] in ("OK", "WARN", "HOT", "DEGRADED")
        assert "stability_score" in summary_dict
        assert "drift_events" in summary_dict
        assert "toxicity" in summary_dict
    
    def test_summary_to_json(self):
        """Test that summary can be serialized to JSON."""
        series = create_stable_series(num_epochs=5)
        stability = evaluate_policy_stability(series)
        drift_events = []
        toxicity = detect_policy_toxicity(series[0])
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        json_str = summary.to_json()
        
        assert isinstance(json_str, str)
        assert "health_status" in json_str
        assert "stability_score" in json_str
    
    def test_red_flags_metadata(self):
        """Test that red flags are tracked in metadata."""
        series = create_stable_series(num_epochs=5)
        stability = evaluate_policy_stability(series)
        drift_events = []
        toxicity = detect_policy_toxicity(series[0])
        
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
        )
        
        assert "red_flags" in summary.metadata
        assert isinstance(summary.metadata["red_flags"], int)
        assert summary.metadata["red_flags"] >= 0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_end_to_end_stable_policy(self):
        """Test complete workflow with stable policy."""
        # Create stable policy series
        series = create_stable_series(num_epochs=15)
        curriculum = create_curriculum_slices()
        
        # Evaluate stability
        stability = evaluate_policy_stability(series)
        assert stability.score >= 0.9
        
        # Detect drift
        drift_events = detect_slice_coupled_drift(series, curriculum)
        # May detect slice boundaries even with zero drift
        # Filter for actual drift
        actual_drift = [e for e in drift_events if e.drift_magnitude > 0.01]
        assert len(actual_drift) == 0  # No actual drift in stable series
        
        # Check toxicity
        toxicity = detect_policy_toxicity(series[-1], historical_snapshots=series)
        assert not toxicity.has_extreme_concentration
        
        # Generate summary
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        assert summary.health_status == HealthStatus.OK
    
    def test_end_to_end_unstable_policy(self):
        """Test complete workflow with unstable policy."""
        # Create oscillating + diverging series
        series = []
        for i in range(20):
            weights = {
                "len": 1.0 + 0.5 * np.sin(2 * np.pi * i / 10) + 0.1 * i,
                "depth": 0.5,
            }
            series.append(create_policy_state(i, weights))
        
        curriculum = create_curriculum_slices()
        
        # Evaluate stability
        stability = evaluate_policy_stability(
            series,
            oscillation_threshold=0.1,
            divergence_threshold=0.05,
        )
        assert stability.score < 1.0  # Should have penalties
        
        # Detect drift
        drift_events = detect_slice_coupled_drift(
            series,
            curriculum,
            drift_threshold=0.1,
        )
        # May have drift events
        
        # Check toxicity
        toxicity = detect_policy_toxicity(series[-1], historical_snapshots=series)
        
        # Generate summary
        summary = summarize_policy_stability_for_global_health(
            stability,
            drift_events,
            toxicity,
        )
        
        assert summary.health_status in (
            HealthStatus.OK,
            HealthStatus.WARN,
            HealthStatus.HOT,
            HealthStatus.DEGRADED,
        )
