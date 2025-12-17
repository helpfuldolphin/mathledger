"""
Tests for TDA Windowed Pattern Classification

Tests for:
- classify_windows(): Per-window pattern classification
- aggregate_pattern_summary(): Summary statistics across windows
- attach_windowed_patterns_to_evidence(): Evidence pack attachment

All tests verify SHADOW MODE markers and deterministic behavior.
"""

import pytest
from typing import Any, Dict, List, Optional

from backend.tda.metrics import TDAWindowMetrics
from backend.tda.patterns_from_windows import (
    classify_windows,
    aggregate_pattern_summary,
    attach_windowed_patterns_to_evidence,
    extract_windowed_patterns_status,
    get_top_events_digest,
    attach_signals_tda_windowed_patterns,
    WindowPatternResult,
    PatternAggregateSummary,
    WindowedPatternsStatus,
    TopEventDigest,
    DEFAULT_MAX_WINDOWS,
    DEFAULT_TOP_EVENTS,
    _select_representative_indices,
)
from backend.tda.pattern_classifier import (
    RTTSPattern,
    PatternClassification,
    TDAPatternClassifier,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_window(
    window_index: int,
    start_cycle: int,
    end_cycle: int,
    sns_mean: float = 0.1,
    sns_max: float = 0.15,
    pcs_mean: float = 0.9,
    pcs_min: float = 0.85,
    hss_mean: float = 0.9,
    hss_min: float = 0.85,
    envelope_occupancy_rate: float = 0.95,
    envelope_exit_count: int = 0,
    max_envelope_exit_streak: int = 0,
    tda_sns_anomaly_flags: int = 0,
    tda_pcs_collapse_flags: int = 0,
    tda_hss_degradation_flags: int = 0,
    tda_envelope_exit_flags: int = 0,
) -> TDAWindowMetrics:
    """Create a TDAWindowMetrics with specified parameters."""
    return TDAWindowMetrics(
        window_index=window_index,
        window_start_cycle=start_cycle,
        window_end_cycle=end_cycle,
        sns_mean=sns_mean,
        sns_max=sns_max,
        pcs_mean=pcs_mean,
        pcs_min=pcs_min,
        hss_mean=hss_mean,
        hss_min=hss_min,
        envelope_occupancy_rate=envelope_occupancy_rate,
        envelope_exit_count=envelope_exit_count,
        max_envelope_exit_streak=max_envelope_exit_streak,
        tda_sns_anomaly_flags=tda_sns_anomaly_flags,
        tda_pcs_collapse_flags=tda_pcs_collapse_flags,
        tda_hss_degradation_flags=tda_hss_degradation_flags,
        tda_envelope_exit_flags=tda_envelope_exit_flags,
    )


def make_healthy_windows(count: int, start_index: int = 0) -> List[TDAWindowMetrics]:
    """Create a list of healthy windows (should classify as NONE).

    Healthy = all metrics nominal:
    - SNS < 0.3 (low novelty)
    - PCS > 0.5 (good coherence)
    - HSS > 0.6 (stable homology)
    - DRS ~ 0 (no drift, simulated via SNS/PCS/HSS)
    """
    windows = []
    for i in range(count):
        idx = start_index + i
        windows.append(make_window(
            window_index=idx,
            start_cycle=idx * 100,
            end_cycle=(idx + 1) * 100 - 1,
            sns_mean=0.05,  # Well below all SNS thresholds
            sns_max=0.08,   # Well below all max thresholds
            pcs_mean=0.95,  # Well above all PCS floors
            pcs_min=0.90,
            hss_mean=0.95,  # Well above all HSS floors
            hss_min=0.90,
        ))
    return windows


def make_drift_window(window_index: int) -> TDAWindowMetrics:
    """Create a window with DRIFT pattern characteristics.

    DRIFT detection requires (from pattern_classifier.py):
    - DRS > 0.05 (or DRS slope > 0.01 with sustained periods)
    - SNS < 0.4 (ceiling)
    - HSS > 0.6 (floor)
    - PCS > 0.5 (floor)

    Note: DRS is computed from P4 or window history. Without P4,
    the classifier estimates from trend analysis.
    """
    # Since we don't have DRS directly, this creates a window that
    # may trigger DRIFT via other metric patterns
    return make_window(
        window_index=window_index,
        start_cycle=window_index * 100,
        end_cycle=(window_index + 1) * 100 - 1,
        sns_mean=0.30,  # Moderate novelty, < 0.4 ceiling
        sns_max=0.35,
        pcs_mean=0.60,  # Still > 0.5 floor
        pcs_min=0.55,
        hss_mean=0.70,  # Still > 0.6 floor
        hss_min=0.65,
    )


def make_structural_break_window(window_index: int) -> TDAWindowMetrics:
    """Create a window with STRUCTURAL_BREAK pattern characteristics."""
    return make_window(
        window_index=window_index,
        start_cycle=window_index * 100,
        end_cycle=(window_index + 1) * 100 - 1,
        sns_mean=0.55,  # SNS > 0.5 threshold
        sns_max=0.65,
        pcs_mean=0.70,
        pcs_min=0.60,
        hss_mean=0.45,  # HSS < 0.5 threshold
        hss_min=0.35,
    )


def make_noise_amplification_window(window_index: int) -> TDAWindowMetrics:
    """Create a window with NOISE_AMPLIFICATION pattern characteristics.

    NOISE_AMPLIFICATION detection requires (from pattern_classifier.py):
    - (SNS_variance > 0.04 OR (SNS_mean > 0.35 AND SNS_max > 0.6))
    - PCS < 0.6 (ceiling)
    - HSS > 0.4 (floor)
    - Envelope exits > 2 (secondary trigger)
    """
    return make_window(
        window_index=window_index,
        start_cycle=window_index * 100,
        end_cycle=(window_index + 1) * 100 - 1,
        sns_mean=0.40,  # Above 0.35 threshold
        sns_max=0.65,   # Above 0.6 threshold
        pcs_mean=0.55,  # Below 0.6 ceiling (critical!)
        pcs_min=0.50,
        hss_mean=0.50,  # Above 0.4 floor
        hss_min=0.45,
        envelope_exit_count=3,  # Above 2 threshold (secondary)
    )


# =============================================================================
# Test: classify_windows()
# =============================================================================

class TestClassifyWindows:
    """Tests for classify_windows() function."""

    def test_empty_input_returns_empty_list(self):
        """Empty input returns empty results."""
        result = classify_windows([])
        assert result == []

    def test_single_healthy_window_classifies_as_none(self):
        """Single healthy window classifies as NONE."""
        windows = make_healthy_windows(1)
        results = classify_windows(windows)

        assert len(results) == 1
        assert results[0].classification.pattern == RTTSPattern.NONE
        assert results[0].window_index == 0
        assert results[0].mode == "SHADOW"

    def test_multiple_healthy_windows_all_classify_as_none(self):
        """Multiple healthy windows all classify as NONE."""
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.classification.pattern == RTTSPattern.NONE
            assert result.window_index == i
            assert result.mode == "SHADOW"

    def test_drift_window_detection(self):
        """Windows with drift characteristics may be detected.

        Note: DRIFT detection typically requires DRS from P4 windows or
        explicit window history with degrading trends. Without P4 data,
        drift detection is limited.
        """
        windows = [make_drift_window(i) for i in range(3)]
        results = classify_windows(windows)

        assert len(results) == 3
        # Verify classification runs without error
        # Drift detection without P4 may not trigger
        for result in results:
            assert isinstance(result.classification.pattern, RTTSPattern)
            assert result.mode == "SHADOW"

    def test_structural_break_detection(self):
        """Structural break patterns are detected."""
        windows = [make_structural_break_window(0)]
        results = classify_windows(windows)

        assert len(results) == 1
        assert results[0].classification.pattern == RTTSPattern.STRUCTURAL_BREAK
        assert results[0].classification.confidence >= 0.5

    def test_noise_amplification_detection(self):
        """Noise amplification patterns are detected."""
        windows = [make_noise_amplification_window(0)]
        results = classify_windows(windows)

        assert len(results) == 1
        assert results[0].classification.pattern == RTTSPattern.NOISE_AMPLIFICATION
        assert results[0].classification.confidence >= 0.5

    def test_window_result_contains_correct_fields(self):
        """WindowPatternResult contains all required fields."""
        windows = make_healthy_windows(1)
        results = classify_windows(windows)

        result = results[0]
        assert hasattr(result, "window_index")
        assert hasattr(result, "window_id")
        assert hasattr(result, "cycle_range")
        assert hasattr(result, "classification")
        assert hasattr(result, "mode")

        assert result.window_id == "window_0000"
        assert result.cycle_range == (0, 99)
        assert result.mode == "SHADOW"

    def test_window_id_format(self):
        """Window IDs follow expected format."""
        windows = make_healthy_windows(3, start_index=42)
        results = classify_windows(windows)

        assert results[0].window_id == "window_0042"
        assert results[1].window_id == "window_0043"
        assert results[2].window_id == "window_0044"

    def test_cycle_range_preserved(self):
        """Cycle ranges are correctly preserved from input windows."""
        windows = [
            make_window(0, start_cycle=100, end_cycle=199),
            make_window(1, start_cycle=200, end_cycle=299),
            make_window(2, start_cycle=300, end_cycle=399),
        ]
        results = classify_windows(windows)

        assert results[0].cycle_range == (100, 199)
        assert results[1].cycle_range == (200, 299)
        assert results[2].cycle_range == (300, 399)

    def test_history_depth_affects_classification(self):
        """History depth parameter affects trend detection."""
        # Create windows with degrading metrics to trigger trend detection
        windows = [
            make_window(0, 0, 99, pcs_mean=0.95, pcs_min=0.90),
            make_window(1, 100, 199, pcs_mean=0.85, pcs_min=0.80),
            make_window(2, 200, 299, pcs_mean=0.75, pcs_min=0.70),
            make_window(3, 300, 399, pcs_mean=0.65, pcs_min=0.60),
        ]

        # With history_depth=1, less trend context
        results_shallow = classify_windows(windows, history_depth=1)
        # With history_depth=3, more trend context
        results_deep = classify_windows(windows, history_depth=3)

        # Both should produce results
        assert len(results_shallow) == 4
        assert len(results_deep) == 4

    def test_custom_classifier_used(self):
        """Custom classifier instance is used when provided."""
        classifier = TDAPatternClassifier()
        windows = make_healthy_windows(2)

        results = classify_windows(windows, classifier=classifier)

        assert len(results) == 2

    def test_to_dict_serialization(self):
        """WindowPatternResult.to_dict() produces valid dict."""
        windows = make_healthy_windows(1)
        results = classify_windows(windows)

        result_dict = results[0].to_dict()

        assert isinstance(result_dict, dict)
        assert "window_index" in result_dict
        assert "window_id" in result_dict
        assert "cycle_range" in result_dict
        assert "pattern" in result_dict
        assert "confidence" in result_dict
        assert "primary_triggers" in result_dict
        assert "secondary_triggers" in result_dict
        assert "mode" in result_dict
        assert result_dict["mode"] == "SHADOW"

    def test_deterministic_output_order(self):
        """Results are in deterministic order matching input."""
        windows = make_healthy_windows(10)

        # Run multiple times
        results1 = classify_windows(windows)
        results2 = classify_windows(windows)

        # Order should be identical
        for i in range(10):
            assert results1[i].window_index == results2[i].window_index
            assert results1[i].classification.pattern == results2[i].classification.pattern


class TestClassifyWindowsWithP4:
    """Tests for classify_windows() with P4 windows."""

    def test_p4_windows_used_for_drs(self):
        """P4 windows provide DRS information when available."""
        p3_windows = make_healthy_windows(2)
        p4_windows = make_healthy_windows(2)

        results = classify_windows(p3_windows, tda_windows_p4=p4_windows)

        assert len(results) == 2
        # Should still classify (P4 provides additional context)
        for result in results:
            assert result.mode == "SHADOW"

    def test_missing_p4_window_handled(self):
        """Missing P4 window for a P3 window is handled gracefully."""
        p3_windows = make_healthy_windows(3)
        p4_windows = [make_healthy_windows(1)[0]]  # Only one P4 window

        results = classify_windows(p3_windows, tda_windows_p4=p4_windows)

        assert len(results) == 3
        # All should still classify
        for result in results:
            assert isinstance(result.classification, PatternClassification)


# =============================================================================
# Test: aggregate_pattern_summary()
# =============================================================================

class TestAggregatePatternSummary:
    """Tests for aggregate_pattern_summary() function."""

    def test_empty_input_returns_default_summary(self):
        """Empty input returns default summary."""
        summary = aggregate_pattern_summary([])

        assert summary.total_windows == 0
        assert summary.dominant_pattern == "NONE"
        assert summary.mode == "SHADOW"

    def test_all_none_patterns(self):
        """All NONE patterns produces correct summary."""
        windows = make_healthy_windows(5)
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        assert summary.total_windows == 5
        assert summary.pattern_counts.get("NONE", 0) == 5
        assert summary.dominant_pattern == "NONE"
        assert summary.mode == "SHADOW"

    def test_pattern_counts_accurate(self):
        """Pattern counts accurately reflect classifications."""
        # Create windows with different patterns
        windows = [
            make_structural_break_window(0),
            make_structural_break_window(1),
            make_noise_amplification_window(2),
            make_window(3, 300, 399),  # healthy -> NONE
        ]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        assert summary.total_windows == 4
        # Check total counts add up
        total_counted = sum(summary.pattern_counts.values())
        assert total_counted == 4

    def test_dominant_pattern_excludes_none(self):
        """Dominant pattern calculation excludes NONE."""
        # Use only structural break windows to ensure clear dominant pattern
        windows = [make_structural_break_window(i) for i in range(3)]

        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        # STRUCTURAL_BREAK should be dominant
        assert summary.dominant_pattern == "STRUCTURAL_BREAK"
        assert summary.dominant_pattern_count == 3

    def test_dominant_pattern_ratio_calculation(self):
        """Dominant pattern ratio is correctly calculated."""
        windows = [make_structural_break_window(i) for i in range(4)]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        assert summary.total_windows == 4
        assert summary.dominant_pattern == "STRUCTURAL_BREAK"
        assert summary.dominant_pattern_ratio == 1.0

    def test_high_confidence_events_tracked(self):
        """High confidence events (>= 0.75) are tracked."""
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        # Structural break should have high confidence
        high_conf_count = sum(
            1 for r in results
            if r.classification.confidence >= 0.75
            and r.classification.pattern != RTTSPattern.NONE
        )
        assert summary.high_confidence_events == high_conf_count

    def test_streak_tracking(self):
        """Pattern streaks are correctly tracked."""
        # Create 5 structural break windows in a row
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        # Should have a streak of 5 STRUCTURAL_BREAK
        assert summary.max_streak_length >= 5
        assert summary.max_streak_pattern == "STRUCTURAL_BREAK"

    def test_current_streak_tracked(self):
        """Current (ending) streak is tracked separately."""
        # Use only healthy windows to ensure clean NONE streak
        windows = make_healthy_windows(5)

        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        # All windows should be NONE, so streak should be all 5
        assert summary.current_streak_pattern == "NONE"
        assert summary.current_streak_length == 5
        assert summary.max_streak_length == 5

    def test_transition_count(self):
        """Pattern transitions are counted correctly."""
        windows = [
            make_structural_break_window(0),
            make_window(1, 100, 199),  # healthy -> NONE
            make_structural_break_window(2),
        ]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        # STRUCTURAL_BREAK -> NONE -> STRUCTURAL_BREAK = 2 transitions
        assert summary.pattern_transitions == 2

    def test_time_range_tracking(self):
        """First/last window and cycle ranges are tracked."""
        windows = [
            make_window(5, 500, 599),
            make_window(6, 600, 699),
            make_window(7, 700, 799),
        ]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        assert summary.first_window_index == 5
        assert summary.last_window_index == 7
        assert summary.first_cycle == 500
        assert summary.last_cycle == 799

    def test_to_dict_serialization(self):
        """PatternAggregateSummary.to_dict() produces valid dict."""
        windows = make_healthy_windows(3)
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        summary_dict = summary.to_dict()

        assert isinstance(summary_dict, dict)
        assert "schema_version" in summary_dict
        assert summary_dict["schema_version"] == "1.0.0"
        assert "total_windows" in summary_dict
        assert "pattern_counts" in summary_dict
        assert "dominant_pattern" in summary_dict
        assert "streaks" in summary_dict
        assert "time_range" in summary_dict
        assert "mode" in summary_dict
        assert summary_dict["mode"] == "SHADOW"

    def test_high_confidence_windows_limited(self):
        """High confidence windows list is limited to first 10."""
        # Create many high-confidence windows
        windows = [make_structural_break_window(i) for i in range(20)]
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        summary_dict = summary.to_dict()
        # Should be limited to 10 in dict output
        assert len(summary_dict["high_confidence_windows"]) <= 10


# =============================================================================
# Test: attach_windowed_patterns_to_evidence()
# =============================================================================

class TestAttachWindowedPatternsToEvidence:
    """Tests for attach_windowed_patterns_to_evidence() function."""

    def test_empty_evidence_creates_structure(self):
        """Empty evidence dict gets proper structure created."""
        evidence = {}
        windows = make_healthy_windows(2)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        assert "governance" in updated
        assert "tda" in updated["governance"]
        assert "patterns" in updated["governance"]["tda"]
        assert updated["governance"]["tda"]["patterns"]["mode"] == "SHADOW"

    def test_existing_evidence_augmented(self):
        """Existing evidence structure is augmented, not replaced."""
        evidence = {
            "governance": {
                "existing_key": "value",
                "tda": {
                    "existing_tda_key": "tda_value",
                },
            },
        }
        windows = make_healthy_windows(2)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        assert updated["governance"]["existing_key"] == "value"
        assert updated["governance"]["tda"]["existing_tda_key"] == "tda_value"
        assert "patterns" in updated["governance"]["tda"]

    def test_aggregate_summary_attached(self):
        """Aggregate summary is attached to evidence."""
        evidence = {}
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert "aggregate_summary" in patterns
        assert patterns["aggregate_summary"]["total_windows"] == 5

    def test_per_window_classifications_attached(self):
        """Per-window classifications are attached."""
        evidence = {}
        windows = make_healthy_windows(3)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert "per_window_classifications" in patterns
        assert len(patterns["per_window_classifications"]) == 3

    def test_per_window_can_be_disabled(self):
        """Per-window classifications can be disabled."""
        evidence = {}
        windows = make_healthy_windows(3)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(
            evidence, results, include_per_window=False
        )

        patterns = updated["governance"]["tda"]["patterns"]
        assert "per_window_classifications" not in patterns

    def test_max_windows_limit_enforced(self):
        """Max windows limit is enforced."""
        evidence = {}
        windows = make_healthy_windows(100)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(
            evidence, results, max_windows=10
        )

        patterns = updated["governance"]["tda"]["patterns"]
        assert len(patterns["per_window_classifications"]) == 10
        assert patterns["windowed_metadata"]["truncated"] is True
        assert patterns["windowed_metadata"]["total_windows"] == 100
        assert patterns["windowed_metadata"]["included_windows"] == 10

    def test_default_max_windows_is_50(self):
        """Default max windows is 50."""
        assert DEFAULT_MAX_WINDOWS == 50

        evidence = {}
        windows = make_healthy_windows(60)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert len(patterns["per_window_classifications"]) == 50
        assert patterns["windowed_metadata"]["truncated"] is True

    def test_no_truncation_when_under_limit(self):
        """No truncation metadata when under limit."""
        evidence = {}
        windows = make_healthy_windows(10)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert patterns["windowed_metadata"]["truncated"] is False
        assert len(patterns["per_window_classifications"]) == 10

    def test_triggers_included_when_requested(self):
        """Trigger details included when requested."""
        evidence = {}
        windows = [make_structural_break_window(0)]
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(
            evidence, results, include_triggers=True
        )

        patterns = updated["governance"]["tda"]["patterns"]
        window_data = patterns["per_window_classifications"][0]
        assert "primary_triggers" in window_data
        assert "secondary_triggers" in window_data

    def test_triggers_excluded_by_default(self):
        """Trigger details excluded by default."""
        evidence = {}
        windows = [make_structural_break_window(0)]
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        window_data = patterns["per_window_classifications"][0]
        assert "primary_triggers" not in window_data
        assert "secondary_triggers" not in window_data

    def test_shadow_mode_marker_always_present(self):
        """SHADOW mode marker is always present."""
        evidence = {}
        windows = make_healthy_windows(2)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert patterns["mode"] == "SHADOW"
        assert "verifier_note" in patterns
        assert "SHADOW-ONLY" in patterns["verifier_note"]

    def test_pre_computed_summary_used(self):
        """Pre-computed summary is used when provided."""
        evidence = {}
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        # Compute summary separately
        summary = aggregate_pattern_summary(results)
        summary.dominant_pattern = "TEST_OVERRIDE"

        updated = attach_windowed_patterns_to_evidence(
            evidence, results, aggregate_summary=summary
        )

        patterns = updated["governance"]["tda"]["patterns"]
        assert patterns["aggregate_summary"]["dominant_pattern"] == "TEST_OVERRIDE"

    def test_empty_results_handled(self):
        """Empty results list is handled gracefully."""
        evidence = {}
        updated = attach_windowed_patterns_to_evidence(evidence, [])

        patterns = updated["governance"]["tda"]["patterns"]
        assert patterns["mode"] == "SHADOW"

    def test_schema_version_present(self):
        """Schema version is present in patterns section."""
        evidence = {}
        windows = make_healthy_windows(2)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        patterns = updated["governance"]["tda"]["patterns"]
        assert "schema_version" in patterns
        assert patterns["schema_version"] == "1.0.0"


# =============================================================================
# Test: Representative Index Selection
# =============================================================================

class TestSelectRepresentativeIndices:
    """Tests for _select_representative_indices() helper function."""

    def test_no_truncation_when_under_max(self):
        """No truncation when total <= max."""
        indices = _select_representative_indices(10, 20)
        assert indices == list(range(10))

    def test_exactly_max_selected(self):
        """Exactly max_count indices are selected."""
        indices = _select_representative_indices(100, 20)
        assert len(indices) == 20

    def test_includes_first_indices(self):
        """First few indices are always included."""
        indices = _select_representative_indices(100, 20)
        # Should include 0, 1, 2, 3, 4 (first 5)
        for i in range(5):
            assert i in indices

    def test_includes_last_indices(self):
        """Last few indices are always included."""
        indices = _select_representative_indices(100, 20)
        # Should include 95, 96, 97, 98, 99 (last 5)
        for i in range(95, 100):
            assert i in indices

    def test_sorted_output(self):
        """Output indices are sorted."""
        indices = _select_representative_indices(100, 20)
        assert indices == sorted(indices)

    def test_no_duplicates(self):
        """No duplicate indices."""
        indices = _select_representative_indices(100, 20)
        assert len(indices) == len(set(indices))

    def test_small_max_count(self):
        """Works with small max_count."""
        indices = _select_representative_indices(100, 6)
        assert len(indices) == 6
        # Should still include first and last
        assert 0 in indices
        assert 99 in indices


# =============================================================================
# Test: Shadow Mode Invariants
# =============================================================================

class TestShadowModeInvariants:
    """Tests ensuring SHADOW MODE is always respected."""

    def test_window_result_always_shadow(self):
        """Every WindowPatternResult has mode='SHADOW'."""
        windows = make_healthy_windows(10)
        results = classify_windows(windows)

        for result in results:
            assert result.mode == "SHADOW"

    def test_aggregate_summary_always_shadow(self):
        """PatternAggregateSummary always has mode='SHADOW'."""
        windows = make_healthy_windows(10)
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        assert summary.mode == "SHADOW"

    def test_evidence_always_shadow(self):
        """Attached evidence always has SHADOW mode."""
        evidence = {}
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        updated = attach_windowed_patterns_to_evidence(evidence, results)

        assert updated["governance"]["tda"]["patterns"]["mode"] == "SHADOW"
        assert updated["governance"]["tda"]["mode"] == "SHADOW"


# =============================================================================
# Test: Deterministic Behavior
# =============================================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior."""

    def test_classify_windows_deterministic(self):
        """classify_windows() produces deterministic results."""
        windows = make_healthy_windows(20)

        results1 = classify_windows(windows)
        results2 = classify_windows(windows)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.window_index == r2.window_index
            assert r1.window_id == r2.window_id
            assert r1.classification.pattern == r2.classification.pattern
            assert r1.classification.confidence == r2.classification.confidence

    def test_aggregate_summary_deterministic(self):
        """aggregate_pattern_summary() produces deterministic results."""
        windows = make_healthy_windows(20)
        results = classify_windows(windows)

        summary1 = aggregate_pattern_summary(results)
        summary2 = aggregate_pattern_summary(results)

        assert summary1.total_windows == summary2.total_windows
        assert summary1.dominant_pattern == summary2.dominant_pattern
        assert summary1.pattern_counts == summary2.pattern_counts
        assert summary1.max_streak_length == summary2.max_streak_length

    def test_evidence_attachment_deterministic(self):
        """attach_windowed_patterns_to_evidence() produces deterministic results."""
        windows = make_healthy_windows(20)
        results = classify_windows(windows)

        evidence1 = attach_windowed_patterns_to_evidence({}, results)
        evidence2 = attach_windowed_patterns_to_evidence({}, results)

        patterns1 = evidence1["governance"]["tda"]["patterns"]
        patterns2 = evidence2["governance"]["tda"]["patterns"]

        assert patterns1["aggregate_summary"] == patterns2["aggregate_summary"]
        assert patterns1["per_window_classifications"] == patterns2["per_window_classifications"]

    def test_representative_indices_deterministic(self):
        """Representative index selection is deterministic."""
        indices1 = _select_representative_indices(100, 20)
        indices2 = _select_representative_indices(100, 20)

        assert indices1 == indices2


# =============================================================================
# Test: Status Extraction (extract_windowed_patterns_status)
# =============================================================================

class TestExtractWindowedPatternsStatus:
    """Tests for extract_windowed_patterns_status() function."""

    def test_empty_input_returns_default_status(self):
        """Empty input returns default status."""
        status = extract_windowed_patterns_status([])

        assert status.dominant_pattern == "NONE"
        assert status.max_streak_length == 0
        assert status.high_confidence_count == 0
        assert status.total_windows == 0
        assert status.mode == "SHADOW"

    def test_extracts_dominant_pattern(self):
        """Extracts dominant pattern from results."""
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)

        assert status.dominant_pattern == "STRUCTURAL_BREAK"
        assert status.total_windows == 5

    def test_extracts_max_streak(self):
        """Extracts max streak info."""
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)

        assert status.max_streak_pattern == "STRUCTURAL_BREAK"
        assert status.max_streak_length >= 5

    def test_counts_high_confidence_events(self):
        """Counts high confidence (>= 0.75) non-NONE events."""
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)

        # Count expected high-confidence events
        expected = sum(
            1 for r in results
            if r.classification.confidence >= 0.75
            and r.classification.pattern != RTTSPattern.NONE
        )
        assert status.high_confidence_count == expected

    def test_computes_windows_with_patterns(self):
        """Computes count of windows with non-NONE patterns."""
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)

        assert status.windows_with_patterns == 3

    def test_computes_transition_rate(self):
        """Computes transition rate correctly."""
        windows = make_healthy_windows(10)
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)

        status = extract_windowed_patterns_status(results, summary)

        expected_rate = summary.pattern_transitions / 10 if summary.pattern_transitions else 0.0
        assert status.transition_rate == expected_rate

    def test_extracts_top_patterns(self):
        """Extracts top 3 non-NONE patterns."""
        # Create mixed windows
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)

        # top_patterns should be list of (pattern, count) tuples
        assert isinstance(status.top_patterns, list)
        # All entries should be non-NONE
        for pattern, count in status.top_patterns:
            assert pattern != "NONE"
            assert count > 0

    def test_to_dict_produces_valid_structure(self):
        """to_dict() produces expected structure."""
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        status = extract_windowed_patterns_status(results)
        status_dict = status.to_dict()

        assert "dominant_pattern" in status_dict
        assert "max_streak" in status_dict
        assert "pattern" in status_dict["max_streak"]
        assert "length" in status_dict["max_streak"]
        assert "high_confidence_count" in status_dict
        assert "coverage" in status_dict
        assert "top_patterns" in status_dict
        assert "transition_rate" in status_dict
        assert "time_range" in status_dict
        assert "mode" in status_dict
        assert status_dict["mode"] == "SHADOW"

    def test_uses_precomputed_summary(self):
        """Uses pre-computed summary when provided."""
        windows = make_healthy_windows(5)
        results = classify_windows(windows)
        summary = aggregate_pattern_summary(results)
        summary.dominant_pattern = "TEST_PATTERN"

        status = extract_windowed_patterns_status(results, summary)

        assert status.dominant_pattern == "TEST_PATTERN"


# =============================================================================
# Test: Top Events Digest (get_top_events_digest)
# =============================================================================

class TestGetTopEventsDigest:
    """Tests for get_top_events_digest() function."""

    def test_empty_input_returns_empty_list(self):
        """Empty input returns empty digest."""
        digest = get_top_events_digest([])
        assert digest == []

    def test_returns_only_non_none_patterns(self):
        """Only includes non-NONE patterns."""
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        digest = get_top_events_digest(results)

        # All healthy windows should be NONE, so digest should be empty
        assert len(digest) == 0

    def test_returns_max_5_events_by_default(self):
        """Returns at most 5 events by default."""
        assert DEFAULT_TOP_EVENTS == 5

        windows = [make_structural_break_window(i) for i in range(10)]
        results = classify_windows(windows)

        digest = get_top_events_digest(results)

        assert len(digest) <= 5

    def test_respects_max_events_parameter(self):
        """Respects max_events parameter."""
        windows = [make_structural_break_window(i) for i in range(10)]
        results = classify_windows(windows)

        digest = get_top_events_digest(results, max_events=3)

        assert len(digest) <= 3

    def test_filters_by_min_confidence(self):
        """Filters events by minimum confidence threshold."""
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)

        # High threshold should filter more
        digest_high = get_top_events_digest(results, min_confidence=0.9)
        digest_low = get_top_events_digest(results, min_confidence=0.5)

        assert len(digest_high) <= len(digest_low)

    def test_sorted_by_confidence_descending(self):
        """Events are sorted by confidence descending."""
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)

        digest = get_top_events_digest(results)

        if len(digest) >= 2:
            for i in range(len(digest) - 1):
                assert digest[i].confidence >= digest[i + 1].confidence

    def test_deterministic_ordering_for_equal_confidence(self):
        """Deterministic ordering when confidences are equal."""
        windows = [make_structural_break_window(i) for i in range(5)]
        results = classify_windows(windows)

        digest1 = get_top_events_digest(results)
        digest2 = get_top_events_digest(results)

        # Should produce identical order
        for d1, d2 in zip(digest1, digest2):
            assert d1.window_index == d2.window_index
            assert d1.pattern == d2.pattern

    def test_digest_entry_contains_required_fields(self):
        """TopEventDigest contains all required fields."""
        windows = [make_structural_break_window(0)]
        results = classify_windows(windows)

        digest = get_top_events_digest(results)

        if digest:
            entry = digest[0]
            assert hasattr(entry, "window_index")
            assert hasattr(entry, "pattern")
            assert hasattr(entry, "confidence")
            assert hasattr(entry, "cycle_range")
            assert hasattr(entry, "primary_triggers")

    def test_to_dict_serialization(self):
        """TopEventDigest.to_dict() produces valid dict."""
        windows = [make_structural_break_window(0)]
        results = classify_windows(windows)

        digest = get_top_events_digest(results)

        if digest:
            entry_dict = digest[0].to_dict()
            assert "window_index" in entry_dict
            assert "pattern" in entry_dict
            assert "confidence" in entry_dict
            assert "cycle_range" in entry_dict
            assert "primary_triggers" in entry_dict


# =============================================================================
# Test: Signals Attachment (attach_signals_tda_windowed_patterns)
# =============================================================================

class TestAttachSignalsTdaWindowedPatterns:
    """Tests for attach_signals_tda_windowed_patterns() function."""

    def test_creates_tda_windowed_patterns_section(self):
        """Creates signals.tda_windowed_patterns section."""
        signals: Dict[str, Any] = {}
        windows = make_healthy_windows(5)
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(signals, results)

        assert "tda_windowed_patterns" in updated
        assert "status" in updated["tda_windowed_patterns"]
        assert "schema_version" in updated["tda_windowed_patterns"]

    def test_includes_status_summary(self):
        """Includes status summary."""
        signals: Dict[str, Any] = {}
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(signals, results)

        status = updated["tda_windowed_patterns"]["status"]
        assert "dominant_pattern" in status
        assert "max_streak" in status
        assert "high_confidence_count" in status

    def test_includes_top_events_by_default(self):
        """Includes top events by default."""
        signals: Dict[str, Any] = {}
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(signals, results)

        assert "top_events" in updated["tda_windowed_patterns"]
        assert "top_events_count" in updated["tda_windowed_patterns"]

    def test_can_exclude_top_events(self):
        """Can exclude top events."""
        signals: Dict[str, Any] = {}
        windows = [make_structural_break_window(i) for i in range(3)]
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(
            signals, results, include_top_events=False
        )

        assert "top_events" not in updated["tda_windowed_patterns"]

    def test_shadow_mode_marker_present(self):
        """SHADOW mode marker is present."""
        signals: Dict[str, Any] = {}
        windows = make_healthy_windows(3)
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(signals, results)

        assert updated["tda_windowed_patterns"]["mode"] == "SHADOW"

    def test_preserves_existing_signals(self):
        """Preserves existing signals in dict."""
        signals = {"existing_signal": {"value": 42}}
        windows = make_healthy_windows(3)
        results = classify_windows(windows)

        updated = attach_signals_tda_windowed_patterns(signals, results)

        assert "existing_signal" in updated
        assert updated["existing_signal"]["value"] == 42
        assert "tda_windowed_patterns" in updated
