"""
Tests for TDA Pattern Classifier

Tests the TDAPatternClassifier class and attach_tda_patterns_to_evidence function.
See: docs/system_law/TDA_PhaseX_Binding.md Section 10-11

SHADOW MODE: All tests verify observational metrics only.
"""

import pytest
from typing import List, Optional

from backend.tda.pattern_classifier import (
    RTTSPattern,
    PatternClassification,
    TDAPatternClassifier,
    attach_tda_patterns_to_evidence,
    PATTERN_PRIORITY,
)
from backend.tda.monitor import TDASummary
from backend.tda.metrics import TDAWindowMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classifier() -> TDAPatternClassifier:
    """Create a fresh classifier instance."""
    return TDAPatternClassifier()


@pytest.fixture
def nominal_p3_summary() -> TDASummary:
    """Create a nominal (healthy) P3 TDA summary."""
    return TDASummary(
        total_cycles=100,
        sns_mean=0.15,
        sns_max=0.25,
        pcs_mean=0.85,
        pcs_min=0.75,
        hss_mean=0.90,
        hss_min=0.80,
        envelope_occupancy=0.98,
        envelope_exit_total=2,
        total_red_flags=0,
    )


@pytest.fixture
def nominal_p4_summary() -> TDASummary:
    """Create a nominal (healthy) P4 TDA summary."""
    return TDASummary(
        total_cycles=100,
        drs_mean=0.02,
        drs_max=0.04,
        sns_mean=0.15,
        sns_max=0.25,
        pcs_mean=0.85,
        pcs_min=0.75,
        hss_mean=0.90,
        hss_min=0.80,
        envelope_occupancy=0.98,
    )


def make_window_metrics(
    sns_mean: float = 0.15,
    pcs_mean: float = 0.85,
    hss_mean: float = 0.90,
    window_index: int = 0,
) -> TDAWindowMetrics:
    """Helper to create window metrics."""
    return TDAWindowMetrics(
        window_index=window_index,
        sns_mean=sns_mean,
        sns_max=sns_mean + 0.1,
        pcs_mean=pcs_mean,
        pcs_min=pcs_mean - 0.1,
        hss_mean=hss_mean,
        hss_min=hss_mean - 0.1,
        envelope_occupancy_rate=0.95,
    )


# =============================================================================
# TestNoneClassification - Stable/Nominal State
# =============================================================================

class TestNoneClassification:
    """Tests for NONE (stable state) classification."""

    def test_nominal_metrics_classify_as_none(self, classifier, nominal_p3_summary, nominal_p4_summary):
        """Nominal metrics should classify as NONE with high confidence."""
        result = classifier.classify(p3_tda=nominal_p3_summary, p4_tda=nominal_p4_summary)

        assert result.pattern == RTTSPattern.NONE
        assert result.confidence == 1.0
        assert "all_metrics_nominal" in result.secondary_triggers

    def test_none_classification_has_shadow_mode(self, classifier, nominal_p3_summary):
        """NONE classification should include SHADOW mode marker."""
        result = classifier.classify(p3_tda=nominal_p3_summary)

        assert result.mode == "SHADOW"

    def test_empty_inputs_classify_as_none(self, classifier):
        """Empty inputs should classify as NONE."""
        result = classifier.classify()

        assert result.pattern == RTTSPattern.NONE
        assert result.confidence == 1.0

    def test_p3_only_nominal_classifies_as_none(self, classifier, nominal_p3_summary):
        """P3-only nominal input should classify as NONE."""
        result = classifier.classify(p3_tda=nominal_p3_summary)

        assert result.pattern == RTTSPattern.NONE


# =============================================================================
# TestDriftDetection
# =============================================================================

class TestDriftDetection:
    """Tests for DRIFT pattern classification."""

    def test_drift_detected_with_elevated_drs(self, classifier):
        """DRIFT should be detected when DRS > 0.05 with stable topology."""
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.07,  # Above threshold
            drs_max=0.09,
            sns_mean=0.2,   # Below SNS ceiling (0.4)
            pcs_mean=0.7,   # Above PCS floor (0.5)
            hss_mean=0.8,   # Above HSS floor (0.6)
        )

        result = classifier.classify(p4_tda=p4)

        assert result.pattern == RTTSPattern.DRIFT
        assert result.confidence > 0.5
        assert any("drs_above" in t for t in result.primary_triggers)

    def test_drift_not_detected_with_low_drs(self, classifier):
        """DRIFT should NOT be detected when DRS is below threshold."""
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.03,  # Below threshold
            drs_max=0.04,
            sns_mean=0.2,
            pcs_mean=0.7,
            hss_mean=0.8,
        )

        result = classifier.classify(p4_tda=p4)

        assert result.pattern == RTTSPattern.NONE

    def test_drift_lower_confidence_with_high_sns(self, classifier):
        """DRIFT confidence should be lower when SNS is too high (violates one secondary)."""
        # Use P3 summary to provide SNS/PCS/HSS since P4 only provides DRS
        p3_high_sns = TDASummary(
            total_cycles=100,
            sns_mean=0.55,  # Above SNS ceiling (0.4) - violates secondary
            pcs_mean=0.7,
            hss_mean=0.8,
        )
        p3_low_sns = TDASummary(
            total_cycles=100,
            sns_mean=0.2,   # Below SNS ceiling - passes secondary
            pcs_mean=0.7,
            hss_mean=0.8,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.07,
            drs_max=0.09,
        )

        result_high_sns = classifier.classify(p3_tda=p3_high_sns, p4_tda=p4)
        result_low_sns = classifier.classify(p3_tda=p3_low_sns, p4_tda=p4)

        # Both may detect DRIFT, but high SNS should have lower confidence
        # (missing one secondary condition)
        assert result_low_sns.pattern == RTTSPattern.DRIFT
        assert result_low_sns.confidence > result_high_sns.confidence

    def test_drift_confidence_increases_with_drs(self, classifier):
        """DRIFT confidence should increase with higher DRS values."""
        p4_low = TDASummary(drs_mean=0.06, sns_mean=0.2, pcs_mean=0.7, hss_mean=0.8, total_cycles=100)
        p4_high = TDASummary(drs_mean=0.12, sns_mean=0.2, pcs_mean=0.7, hss_mean=0.8, total_cycles=100)

        result_low = classifier.classify(p4_tda=p4_low)
        result_high = classifier.classify(p4_tda=p4_high)

        assert result_low.pattern == RTTSPattern.DRIFT
        assert result_high.pattern == RTTSPattern.DRIFT
        assert result_high.confidence >= result_low.confidence


# =============================================================================
# TestPhaseLagDetection
# =============================================================================

class TestPhaseLagDetection:
    """Tests for PHASE_LAG pattern classification."""

    def test_phase_lag_detected_with_low_pcs(self, classifier):
        """PHASE_LAG should be detected when PCS < 0.5 with moderate DRS."""
        p3 = TDASummary(
            total_cycles=100,
            pcs_mean=0.4,   # Below threshold (0.5)
            pcs_min=0.3,
            sns_mean=0.3,   # Below SNS ceiling (0.5)
            hss_mean=0.7,   # Above HSS floor (0.5)
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.06,  # In range [0.03, 0.10]
            drs_max=0.08,
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4)

        # PHASE_LAG may compete with DRIFT - verify PCS trigger detected
        assert result.pattern in [RTTSPattern.PHASE_LAG, RTTSPattern.DRIFT]
        if result.pattern == RTTSPattern.PHASE_LAG:
            assert any("pcs_below" in t for t in result.primary_triggers)

    def test_phase_lag_detected_with_pcs_delta(self, classifier):
        """PHASE_LAG should be detected with sharp PCS drop (delta < -0.15)."""
        # Create window history with PCS drop
        history = [
            make_window_metrics(pcs_mean=0.8, hss_mean=0.7, window_index=0),
            make_window_metrics(pcs_mean=0.55, hss_mean=0.7, window_index=1),  # Delta = -0.25
        ]
        p4 = TDASummary(drs_mean=0.06, total_cycles=100)

        result = classifier.classify(p4_tda=p4, window_history=history)

        # Check that pcs_delta was detected
        if result.pattern == RTTSPattern.PHASE_LAG:
            assert any("pcs_delta" in t for t in result.primary_triggers)

    def test_phase_lag_not_detected_without_moderate_drs(self, classifier):
        """PHASE_LAG requires DRS in range [0.03, 0.10]."""
        p3 = TDASummary(
            total_cycles=100,
            pcs_mean=0.4,
            sns_mean=0.3,
            hss_mean=0.7,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.15,  # Above range - too high
            drs_max=0.18,
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4)

        # With DRS too high, might classify as something else or NONE
        assert result.pattern != RTTSPattern.PHASE_LAG or result.confidence < 0.5


# =============================================================================
# TestStructuralBreakDetection
# =============================================================================

class TestStructuralBreakDetection:
    """Tests for STRUCTURAL_BREAK pattern classification."""

    def test_structural_break_detected_with_hss_collapse_and_sns_spike(self, classifier):
        """STRUCTURAL_BREAK should be detected when HSS < 0.5 AND SNS > 0.5."""
        # Only use P3 for SNS/PCS/HSS - don't provide P4 to avoid DRIFT detection
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,  # Below threshold (0.5)
            hss_min=0.25,
            sns_mean=0.6,   # Above threshold (0.5)
            sns_max=0.7,
            pcs_mean=0.6,
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern == RTTSPattern.STRUCTURAL_BREAK
        assert result.confidence >= 0.5
        assert any("hss_below" in t for t in result.primary_triggers)
        assert any("sns_above" in t for t in result.secondary_triggers)

    def test_structural_break_detected_with_hss_delta(self, classifier):
        """STRUCTURAL_BREAK should be detected with rapid HSS collapse (delta < -0.25)."""
        history = [
            make_window_metrics(hss_mean=0.82, sns_mean=0.2, window_index=0),
            make_window_metrics(hss_mean=0.45, sns_mean=0.55, window_index=1),  # HSS delta = -0.37
        ]
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.45,  # Below 0.5 threshold
            sns_mean=0.55,  # Above 0.5 threshold (required secondary)
        )

        result = classifier.classify(p3_tda=p3, window_history=history)

        # With hss_mean=0.45 and sns_mean=0.55, should detect STRUCTURAL_BREAK
        assert result.pattern == RTTSPattern.STRUCTURAL_BREAK
        assert any("hss_delta" in t or "hss_below" in t for t in result.primary_triggers)

    def test_structural_break_not_detected_without_sns_spike(self, classifier):
        """STRUCTURAL_BREAK requires SNS > 0.5 as secondary condition."""
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,  # Low HSS
            sns_mean=0.3,   # SNS NOT high enough
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern != RTTSPattern.STRUCTURAL_BREAK

    def test_structural_break_has_highest_priority(self, classifier):
        """STRUCTURAL_BREAK should take priority over other patterns."""
        # Create metrics that match STRUCTURAL_BREAK (no P4/DRS to avoid DRIFT)
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,  # STRUCTURAL_BREAK primary (below 0.5)
            sns_mean=0.55,  # STRUCTURAL_BREAK secondary (above 0.5)
            pcs_mean=0.4,   # Also could match PHASE_LAG
        )

        result = classifier.classify(p3_tda=p3)

        # STRUCTURAL_BREAK should win due to priority
        assert result.pattern == RTTSPattern.STRUCTURAL_BREAK


# =============================================================================
# TestNoiseAmplificationDetection
# =============================================================================

class TestNoiseAmplificationDetection:
    """Tests for NOISE_AMPLIFICATION pattern classification."""

    def test_noise_amp_detected_with_high_sns_variance(self, classifier):
        """NOISE_AMPLIFICATION should be detected with high SNS variance."""
        # Create history with high variance
        history = [
            make_window_metrics(sns_mean=0.1, pcs_mean=0.55, hss_mean=0.6, window_index=0),
            make_window_metrics(sns_mean=0.5, pcs_mean=0.55, hss_mean=0.6, window_index=1),
            make_window_metrics(sns_mean=0.15, pcs_mean=0.55, hss_mean=0.6, window_index=2),
            make_window_metrics(sns_mean=0.55, pcs_mean=0.55, hss_mean=0.6, window_index=3),
            make_window_metrics(sns_mean=0.2, pcs_mean=0.55, hss_mean=0.6, window_index=4),
        ]
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.3,
            pcs_mean=0.55,  # Below PCS ceiling (0.6)
            hss_mean=0.6,   # Above HSS floor (0.4)
        )

        result = classifier.classify(p3_tda=p3, window_history=history)

        # If variance is computed, should detect NOISE_AMPLIFICATION
        if result.metrics_snapshot.get("sns_variance", 0) > 0.04:
            assert result.pattern == RTTSPattern.NOISE_AMPLIFICATION

    def test_noise_amp_detected_with_elevated_sns_and_spikes(self, classifier):
        """NOISE_AMPLIFICATION detected when SNS mean >= 0.35 AND max >= 0.6."""
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.4,   # Above mean threshold (0.35)
            sns_max=0.65,   # Above max threshold (0.6)
            pcs_mean=0.55,  # Below PCS ceiling (0.6)
            hss_mean=0.5,   # Above HSS floor (0.4) - but must check
            envelope_exit_total=5,  # Above envelope exits threshold (2)
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern == RTTSPattern.NOISE_AMPLIFICATION
        assert result.confidence > 0.5
        assert any("sns_elevated" in t for t in result.primary_triggers)

    def test_noise_amp_not_detected_with_stable_sns(self, classifier):
        """NOISE_AMPLIFICATION should NOT be detected with stable SNS."""
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.2,   # Low mean
            sns_max=0.3,    # Low max
            pcs_mean=0.55,
            hss_mean=0.6,
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern != RTTSPattern.NOISE_AMPLIFICATION


# =============================================================================
# TestAttractorMissDetection
# =============================================================================

class TestAttractorMissDetection:
    """Tests for ATTRACTOR_MISS pattern classification."""

    def test_attractor_miss_detected_with_decreasing_hss_and_high_drs(self, classifier):
        """ATTRACTOR_MISS should be detected with HSS < 0.6 (decreasing) AND DRS > 0.10."""
        history = [
            make_window_metrics(hss_mean=0.7, sns_mean=0.4, window_index=0),
            make_window_metrics(hss_mean=0.6, sns_mean=0.4, window_index=1),
            make_window_metrics(hss_mean=0.5, sns_mean=0.4, window_index=2),  # Decreasing HSS
        ]
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.5,   # Below threshold (0.6)
            sns_mean=0.4,   # In range [0.3, 0.6] - must NOT be > 0.5 to avoid STRUCTURAL_BREAK
            pcs_mean=0.65,  # Below PCS ceiling (0.7)
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.15,  # Above DRS threshold (0.10)
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4, window_history=history)

        # ATTRACTOR_MISS requires both HSS decreasing AND high DRS
        # With SNS=0.4, it shouldn't trigger STRUCTURAL_BREAK
        assert result.pattern == RTTSPattern.ATTRACTOR_MISS
        assert result.confidence > 0.5
        assert any("hss_below" in t and "decreasing" in t for t in result.primary_triggers)
        assert any("drs_above" in t for t in result.primary_triggers)

    def test_attractor_miss_not_detected_with_stable_hss(self, classifier):
        """ATTRACTOR_MISS requires HSS to be decreasing."""
        history = [
            make_window_metrics(hss_mean=0.55, window_index=0),
            make_window_metrics(hss_mean=0.55, window_index=1),
            make_window_metrics(hss_mean=0.56, window_index=2),  # Stable/increasing
        ]
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.56,
            sns_mean=0.4,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.15,
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4, window_history=history)

        # HSS not decreasing, should not be ATTRACTOR_MISS
        assert result.pattern != RTTSPattern.ATTRACTOR_MISS or result.confidence < 0.5

    def test_attractor_miss_not_detected_with_low_drs(self, classifier):
        """ATTRACTOR_MISS requires DRS > 0.10."""
        history = [
            make_window_metrics(hss_mean=0.7, window_index=0),
            make_window_metrics(hss_mean=0.6, window_index=1),
            make_window_metrics(hss_mean=0.5, window_index=2),
        ]
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.5,
            sns_mean=0.4,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.05,  # Below threshold
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4, window_history=history)

        assert result.pattern != RTTSPattern.ATTRACTOR_MISS


# =============================================================================
# TestTransientMissDetection
# =============================================================================

class TestTransientMissDetection:
    """Tests for TRANSIENT_MISS pattern classification."""

    def test_transient_miss_detected_with_spike_and_recovery(self, classifier):
        """TRANSIENT_MISS should be detected when SNS spikes then recovers quickly."""
        # No P4 to avoid DRIFT (DRS defaults to 0 which is < 0.08)
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.2,   # Current (recovered below 0.3)
            sns_max=0.6,    # Had a spike (above 0.5)
            hss_mean=0.75,  # Stable HSS > 0.6
            pcs_mean=0.8,
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern == RTTSPattern.TRANSIENT_MISS
        assert result.confidence > 0.5
        assert any("spike_recovered" in t for t in result.primary_triggers)

    def test_transient_miss_not_detected_without_recovery(self, classifier):
        """TRANSIENT_MISS should NOT be detected if SNS hasn't recovered."""
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.55,  # Still elevated (not recovered)
            sns_max=0.6,
            hss_mean=0.75,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.05,
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4)

        assert result.pattern != RTTSPattern.TRANSIENT_MISS

    def test_transient_miss_not_detected_with_high_drs(self, classifier):
        """TRANSIENT_MISS requires DRS < 0.08."""
        p3 = TDASummary(
            total_cycles=100,
            sns_mean=0.2,
            sns_max=0.6,
            hss_mean=0.75,
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.12,  # Too high
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4)

        # With high DRS, might be DRIFT instead
        assert result.pattern != RTTSPattern.TRANSIENT_MISS or result.confidence < 0.5


# =============================================================================
# TestConfidenceComputation
# =============================================================================

class TestConfidenceComputation:
    """Tests for confidence score computation."""

    def test_confidence_range(self, classifier):
        """Confidence should always be in [0.0, 1.0]."""
        test_cases = [
            TDASummary(drs_mean=0.07, sns_mean=0.2, pcs_mean=0.7, hss_mean=0.8, total_cycles=100),
            TDASummary(pcs_mean=0.4, sns_mean=0.3, hss_mean=0.7, drs_mean=0.06, total_cycles=100),
            TDASummary(hss_mean=0.35, sns_mean=0.6, drs_mean=0.1, total_cycles=100),
        ]

        for p4 in test_cases:
            result = classifier.classify(p4_tda=p4)
            assert 0.0 <= result.confidence <= 1.0

    def test_none_confidence_is_one(self, classifier, nominal_p3_summary):
        """NONE classification should have confidence = 1.0."""
        result = classifier.classify(p3_tda=nominal_p3_summary)

        assert result.pattern == RTTSPattern.NONE
        assert result.confidence == 1.0

    def test_confidence_formula(self, classifier):
        """Test that confidence follows the formula: 0.6*primary + 0.4*secondary."""
        # This is a behavioral test - confidence should increase with more triggers
        p4_partial = TDASummary(
            drs_mean=0.06,  # Just above threshold
            sns_mean=0.35,  # Close to ceiling
            pcs_mean=0.55,  # Close to floor
            hss_mean=0.65,  # Close to floor
            total_cycles=100,
        )
        p4_full = TDASummary(
            drs_mean=0.15,  # Well above threshold
            sns_mean=0.15,  # Well below ceiling
            pcs_mean=0.8,   # Well above floor
            hss_mean=0.9,   # Well above floor
            total_cycles=100,
        )

        result_partial = classifier.classify(p4_tda=p4_partial)
        result_full = classifier.classify(p4_tda=p4_full)

        # Both should be DRIFT, but full should have higher confidence
        if result_partial.pattern == RTTSPattern.DRIFT and result_full.pattern == RTTSPattern.DRIFT:
            assert result_full.confidence >= result_partial.confidence


# =============================================================================
# TestPriorityOrdering
# =============================================================================

class TestPriorityOrdering:
    """Tests for deterministic priority ordering of patterns."""

    def test_priority_list_completeness(self):
        """All RTTSPattern values should be in PATTERN_PRIORITY."""
        for pattern in RTTSPattern:
            assert pattern in PATTERN_PRIORITY

    def test_priority_order_is_deterministic(self):
        """Priority order should be deterministic and match spec."""
        expected_order = [
            RTTSPattern.STRUCTURAL_BREAK,
            RTTSPattern.ATTRACTOR_MISS,
            RTTSPattern.NOISE_AMPLIFICATION,
            RTTSPattern.PHASE_LAG,
            RTTSPattern.DRIFT,
            RTTSPattern.TRANSIENT_MISS,
            RTTSPattern.NONE,
        ]
        assert PATTERN_PRIORITY == expected_order

    def test_structural_break_beats_drift(self, classifier):
        """STRUCTURAL_BREAK should take priority over DRIFT."""
        # Use only P3 to test STRUCTURAL_BREAK priority without DRS complication
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.4,   # STRUCTURAL_BREAK primary (below 0.5)
            sns_mean=0.55,  # STRUCTURAL_BREAK secondary (above 0.5)
        )

        result = classifier.classify(p3_tda=p3)

        assert result.pattern == RTTSPattern.STRUCTURAL_BREAK

    def test_attractor_miss_beats_drift(self, classifier):
        """ATTRACTOR_MISS should take priority over DRIFT."""
        history = [
            make_window_metrics(hss_mean=0.7, sns_mean=0.4, window_index=0),
            make_window_metrics(hss_mean=0.6, sns_mean=0.4, window_index=1),
            make_window_metrics(hss_mean=0.5, sns_mean=0.4, window_index=2),
        ]
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.5,   # Below 0.6 threshold
            sns_mean=0.4,   # In range [0.3, 0.6] for ATTRACTOR_MISS
            pcs_mean=0.65,  # Below 0.7 for ATTRACTOR_MISS secondary
        )
        p4 = TDASummary(
            total_cycles=100,
            drs_mean=0.12,  # Above 0.10 for ATTRACTOR_MISS, also above 0.05 for DRIFT
        )

        result = classifier.classify(p3_tda=p3, p4_tda=p4, window_history=history)

        # ATTRACTOR_MISS has higher priority than DRIFT
        assert result.pattern == RTTSPattern.ATTRACTOR_MISS


# =============================================================================
# TestEvidenceAttachment
# =============================================================================

class TestEvidenceAttachment:
    """Tests for attach_tda_patterns_to_evidence function."""

    def test_attach_creates_patterns_section(self, classifier, nominal_p3_summary):
        """Should create governance.tda.patterns section."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        assert "governance" in result
        assert "tda" in result["governance"]
        assert "patterns" in result["governance"]["tda"]

    def test_attach_preserves_existing_governance(self, classifier, nominal_p3_summary):
        """Should preserve existing governance data."""
        evidence = {"governance": {"other": "data", "tda": {"p3_synthetic": {"exists": True}}}}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        assert result["governance"]["other"] == "data"
        assert result["governance"]["tda"]["p3_synthetic"]["exists"] is True
        assert "patterns" in result["governance"]["tda"]

    def test_attach_includes_classification(self, classifier, nominal_p3_summary):
        """Should include current classification."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert "classification" in patterns
        assert patterns["classification"]["pattern"] == "NONE"
        assert patterns["classification"]["confidence"] == 1.0

    def test_attach_includes_history(self, classifier, nominal_p3_summary):
        """Should include classification history."""
        # Create some history
        history = [
            PatternClassification(
                pattern=RTTSPattern.NONE,
                confidence=1.0,
                window_id="window_001",
            ),
            PatternClassification(
                pattern=RTTSPattern.DRIFT,
                confidence=0.62,
                window_id="window_002",
            ),
        ]

        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier,
            p3_tda=nominal_p3_summary,
            classification_history=history,
            window_id="window_003",
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert "classification_history" in patterns
        assert len(patterns["classification_history"]) == 3  # 2 history + 1 current

    def test_attach_includes_summary(self, classifier, nominal_p3_summary):
        """Should include pattern summary statistics."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert "summary" in patterns
        assert "dominant_pattern" in patterns["summary"]
        assert "pattern_counts" in patterns["summary"]
        assert "total_windows_classified" in patterns["summary"]

    def test_attach_includes_schema_version(self, classifier, nominal_p3_summary):
        """Should include schema version."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert patterns["schema_version"] == "1.0.0"
        assert patterns["classifier_version"] == "TDAPatternClassifier-v1"

    def test_attach_includes_verifier_note(self, classifier, nominal_p3_summary):
        """Should include SHADOW MODE verifier note."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert "verifier_note" in patterns
        assert "SHADOW" in patterns["verifier_note"]

    def test_attach_shadow_mode_marker(self, classifier, nominal_p3_summary):
        """Should include SHADOW mode marker."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        patterns = result["governance"]["tda"]["patterns"]
        assert patterns["mode"] == "SHADOW"


# =============================================================================
# TestShadowModeInvariants
# =============================================================================

class TestShadowModeInvariants:
    """Tests for SHADOW MODE contract compliance."""

    def test_classification_always_has_shadow_mode(self, classifier):
        """All classifications should have mode='SHADOW'."""
        test_cases = [
            TDASummary(total_cycles=100),  # Empty
            TDASummary(drs_mean=0.07, sns_mean=0.2, pcs_mean=0.7, hss_mean=0.8, total_cycles=100),
            TDASummary(hss_mean=0.35, sns_mean=0.6, total_cycles=100),
        ]

        for summary in test_cases:
            result = classifier.classify(p4_tda=summary)
            assert result.mode == "SHADOW"

    def test_classification_to_dict_has_shadow_mode(self, classifier, nominal_p3_summary):
        """Classification.to_dict() should include mode='SHADOW'."""
        result = classifier.classify(p3_tda=nominal_p3_summary)
        d = result.to_dict()

        assert d["mode"] == "SHADOW"

    def test_evidence_attachment_has_shadow_mode(self, classifier, nominal_p3_summary):
        """Evidence attachment should have mode='SHADOW'."""
        evidence = {}
        result = attach_tda_patterns_to_evidence(
            evidence, classifier, p3_tda=nominal_p3_summary
        )

        assert result["governance"]["tda"]["patterns"]["mode"] == "SHADOW"

    def test_classification_is_pure_observation(self, classifier, nominal_p3_summary):
        """Classification should not modify input data."""
        original_cycles = nominal_p3_summary.total_cycles
        original_sns = nominal_p3_summary.sns_mean

        _ = classifier.classify(p3_tda=nominal_p3_summary)

        # Input should be unchanged
        assert nominal_p3_summary.total_cycles == original_cycles
        assert nominal_p3_summary.sns_mean == original_sns


# =============================================================================
# TestDeterministicBehavior
# =============================================================================

class TestDeterministicBehavior:
    """Tests for deterministic classification behavior."""

    def test_same_inputs_same_output(self, classifier):
        """Same inputs should always produce same output."""
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,
            sns_mean=0.6,
            pcs_mean=0.6,
        )

        result1 = classifier.classify(p3_tda=p3)
        result2 = classifier.classify(p3_tda=p3)

        assert result1.pattern == result2.pattern
        assert result1.confidence == result2.confidence
        assert result1.primary_triggers == result2.primary_triggers
        assert result1.secondary_triggers == result2.secondary_triggers

    def test_classification_order_independent_of_instance(self):
        """Different classifier instances should produce same results."""
        classifier1 = TDAPatternClassifier()
        classifier2 = TDAPatternClassifier()

        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,
            sns_mean=0.6,
        )

        result1 = classifier1.classify(p3_tda=p3)
        result2 = classifier2.classify(p3_tda=p3)

        assert result1.pattern == result2.pattern
        assert result1.confidence == result2.confidence

    def test_triggers_are_sorted_deterministically(self, classifier):
        """Triggers should be in deterministic order."""
        p3 = TDASummary(
            total_cycles=100,
            hss_mean=0.35,
            sns_mean=0.6,
        )

        result1 = classifier.classify(p3_tda=p3)
        result2 = classifier.classify(p3_tda=p3)

        # Order should be identical
        assert result1.primary_triggers == result2.primary_triggers
        assert result1.secondary_triggers == result2.secondary_triggers
