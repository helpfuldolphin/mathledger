"""
Tests for Curriculum Stability Envelope

Validates HSS variance tracking, slice stability detection, and
suitability scoring for curriculum transitions.
"""

import pytest
from datetime import datetime

from curriculum.stability_envelope import (
    CurriculumStabilityEnvelope,
    StabilityEnvelopeConfig,
    HSSMetrics,
    SliceStabilityMetrics,
)


class TestHSSMetrics:
    """Test HSS metrics data class."""
    
    def test_hss_metrics_creation(self):
        """Test creating HSS metrics."""
        metrics = HSSMetrics(
            cycle_id="cycle_001",
            hss_value=0.75,
            verified_count=10,
            timestamp="2025-01-01T00:00:00Z",
            slice_name="slice_a",
        )
        
        assert metrics.cycle_id == "cycle_001"
        assert metrics.hss_value == 0.75
        assert metrics.verified_count == 10
        assert metrics.slice_name == "slice_a"
    
    def test_hss_metrics_to_dict(self):
        """Test serialization to dictionary."""
        metrics = HSSMetrics(
            cycle_id="cycle_001",
            hss_value=0.75,
            verified_count=10,
            timestamp="2025-01-01T00:00:00Z",
            slice_name="slice_a",
        )
        
        data = metrics.to_dict()
        assert data["cycle_id"] == "cycle_001"
        assert data["hss_value"] == 0.75
        assert data["verified_count"] == 10


class TestStabilityEnvelopeConfig:
    """Test stability envelope configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StabilityEnvelopeConfig()
        
        assert config.max_hss_cv == 0.25
        assert config.min_hss_threshold == 0.3
        assert config.max_low_hss_ratio == 0.2
        assert config.min_cycles_for_stability == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StabilityEnvelopeConfig(
            max_hss_cv=0.3,
            min_hss_threshold=0.4,
            max_low_hss_ratio=0.15,
        )
        
        assert config.max_hss_cv == 0.3
        assert config.min_hss_threshold == 0.4
        assert config.max_low_hss_ratio == 0.15


class TestCurriculumStabilityEnvelope:
    """Test curriculum stability envelope."""
    
    def test_initialization(self):
        """Test envelope initialization."""
        envelope = CurriculumStabilityEnvelope()
        
        assert envelope.config is not None
        assert envelope.hss_history == {}
        assert envelope.baseline_variance == {}
    
    def test_record_single_cycle(self):
        """Test recording a single cycle."""
        envelope = CurriculumStabilityEnvelope()
        
        envelope.record_cycle(
            cycle_id="cycle_001",
            slice_name="slice_a",
            hss_value=0.8,
            verified_count=10,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        assert "slice_a" in envelope.hss_history
        assert len(envelope.hss_history["slice_a"]) == 1
        assert envelope.hss_history["slice_a"][0].hss_value == 0.8
    
    def test_record_multiple_cycles(self):
        """Test recording multiple cycles for same slice."""
        envelope = CurriculumStabilityEnvelope()
        
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_a",
                hss_value=0.7 + (i * 0.01),
                verified_count=10 + i,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        assert len(envelope.hss_history["slice_a"]) == 10
    
    def test_stable_slice_detection(self):
        """Test detection of stable slice."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record stable HSS values (low variance, high mean)
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_stable",
                hss_value=0.75 + (i % 3) * 0.02,  # Small variation
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_stable")
        
        assert stability.slice_name == "slice_stable"
        assert stability.is_stable
        assert stability.suitability_score > 0.7
        assert "high_variance" not in stability.flags
        assert "repeated_low_hss" not in stability.flags
    
    def test_unstable_slice_high_variance(self):
        """Test detection of unstable slice with high variance."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record HSS values with high variance
        hss_values = [0.8, 0.2, 0.9, 0.1, 0.85, 0.15, 0.95, 0.25, 0.9, 0.2]
        for i, hss in enumerate(hss_values):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_unstable",
                hss_value=hss,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_unstable")
        
        assert not stability.is_stable
        assert "high_variance" in stability.flags
        assert stability.hss_cv > 0.25  # Default max_hss_cv
    
    def test_unstable_slice_low_hss(self):
        """Test detection of slice with repeated low HSS."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record mostly low HSS values
        for i in range(10):
            hss_value = 0.25 if i < 7 else 0.8  # 70% low HSS
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_low_hss",
                hss_value=hss_value,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_low_hss")
        
        assert not stability.is_stable
        assert "repeated_low_hss" in stability.flags
        assert stability.low_hss_count >= 7
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record only 2 cycles (less than min_cycles_for_stability)
        for i in range(2):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_insufficient",
                hss_value=0.75,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_insufficient")
        
        assert not stability.is_stable
        assert "insufficient_cycles" in stability.flags
    
    def test_no_data(self):
        """Test handling of slice with no data."""
        envelope = CurriculumStabilityEnvelope()
        
        stability = envelope.compute_slice_stability("nonexistent_slice")
        
        assert not stability.is_stable
        assert "insufficient_data" in stability.flags
        assert stability.suitability_score == 0.0
    
    def test_suitability_score_calculation(self):
        """Test suitability score calculation."""
        envelope = CurriculumStabilityEnvelope()
        
        # High suitability: high mean, low variance, no low-HSS
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_high_suit",
                hss_value=0.85 + (i % 2) * 0.02,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_high_suit")
        assert stability.suitability_score > 0.8
        
        # Low suitability: low mean, high variance, many low-HSS
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_low_suit",
                hss_value=0.2 if i < 7 else 0.9,
                verified_count=5,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        stability = envelope.compute_slice_stability("slice_low_suit")
        assert stability.suitability_score < 0.5
    
    def test_variance_spike_detection(self):
        """Test detection of HSS variance spikes."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record baseline with low variance
        for i in range(20):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_spike",
                hss_value=0.75 + (i % 2) * 0.02,  # Small variation
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        # Add recent cycles with high variance
        recent_values = [0.9, 0.1, 0.95, 0.05, 0.85, 0.15, 0.9, 0.2, 0.95, 0.1]
        for i, hss in enumerate(recent_values):
            envelope.record_cycle(
                cycle_id=f"cycle_{20+i:03d}",
                slice_name="slice_spike",
                hss_value=hss,
                verified_count=10,
                timestamp=f"2025-01-{20+i+1:02d}T00:00:00Z",
            )
        
        spike_detected, current_var = envelope.detect_variance_spike("slice_spike")
        
        assert spike_detected
        assert current_var is not None
        assert current_var > envelope.baseline_variance["slice_spike"]
    
    def test_no_variance_spike(self):
        """Test no spike detection when variance is stable."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record cycles with consistent variance
        for i in range(30):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_no_spike",
                hss_value=0.75 + (i % 3) * 0.02,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        spike_detected, current_var = envelope.detect_variance_spike("slice_no_spike")
        
        assert not spike_detected
        assert current_var is not None
    
    def test_slice_transition_allowed(self):
        """Test allowing stable slice transition."""
        envelope = CurriculumStabilityEnvelope()
        
        # Create stable slice
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_from",
                hss_value=0.8 + (i % 2) * 0.02,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        allowed, reason, details = envelope.check_slice_transition_allowed(
            "slice_from", "slice_to"
        )
        
        assert allowed
        assert "stable" in reason.lower()
        assert details["from_slice"] == "slice_from"
        assert details["to_slice"] == "slice_to"
    
    def test_slice_transition_blocked_unstable(self):
        """Test blocking transition from unstable slice."""
        envelope = CurriculumStabilityEnvelope()
        
        # Create unstable slice (high variance)
        hss_values = [0.9, 0.1, 0.85, 0.15, 0.95, 0.2, 0.9, 0.25, 0.85, 0.1]
        for i, hss in enumerate(hss_values):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_unstable",
                hss_value=hss,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        allowed, reason, details = envelope.check_slice_transition_allowed(
            "slice_unstable", "slice_to"
        )
        
        assert not allowed
        assert "unstable" in reason.lower()
        assert len(details["from_stability"]["flags"]) > 0
    
    def test_slice_transition_blocked_spike(self):
        """Test blocking transition when variance spike detected."""
        envelope = CurriculumStabilityEnvelope()
        
        # Create baseline
        for i in range(20):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_spike",
                hss_value=0.75 + (i % 2) * 0.01,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        # Add spike
        recent_values = [0.9, 0.1, 0.95, 0.05, 0.85, 0.15, 0.9, 0.2, 0.95, 0.1]
        for i, hss in enumerate(recent_values):
            envelope.record_cycle(
                cycle_id=f"cycle_{20+i:03d}",
                slice_name="slice_spike",
                hss_value=hss,
                verified_count=10,
                timestamp=f"2025-01-{20+i+1:02d}T00:00:00Z",
            )
        
        allowed, reason, details = envelope.check_slice_transition_allowed(
            "slice_spike", "slice_to"
        )
        
        assert not allowed
        assert "spike" in reason.lower()
        assert details["variance_spike_detected"]
    
    def test_get_all_slice_suitability(self):
        """Test getting suitability scores for all slices."""
        envelope = CurriculumStabilityEnvelope()
        
        # Create multiple slices
        for slice_name in ["slice_a", "slice_b", "slice_c"]:
            for i in range(10):
                envelope.record_cycle(
                    cycle_id=f"{slice_name}_cycle_{i:03d}",
                    slice_name=slice_name,
                    hss_value=0.75 + (i % 2) * 0.02,
                    verified_count=10,
                    timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                )
        
        scores = envelope.get_all_slice_suitability()
        
        assert len(scores) == 3
        assert "slice_a" in scores
        assert "slice_b" in scores
        assert "slice_c" in scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    def test_export_stability_report(self):
        """Test exporting complete stability report."""
        envelope = CurriculumStabilityEnvelope()
        
        # Record data for two slices
        for i in range(10):
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_a",
                hss_value=0.8 + (i % 2) * 0.02,
                verified_count=10,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
            envelope.record_cycle(
                cycle_id=f"cycle_{i:03d}",
                slice_name="slice_b",
                hss_value=0.7 + (i % 3) * 0.03,
                verified_count=8,
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            )
        
        report = envelope.export_stability_report()
        
        assert "config" in report
        assert "slices" in report
        assert "slice_a" in report["slices"]
        assert "slice_b" in report["slices"]
        
        slice_a_report = report["slices"]["slice_a"]
        assert "hss_mean" in slice_a_report
        assert "hss_cv" in slice_a_report
        assert "suitability_score" in slice_a_report
        assert "is_stable" in slice_a_report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
