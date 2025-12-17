"""Tests for semantic-TDA correlation timeline extractor.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

Tests windowed correlation extraction, trend analysis, and visualization.
"""

import json

import pytest

from backend.health.semantic_tda_adapter import extract_correlation_for_pattern_classifier
from backend.health.semantic_tda_timeline import (
    DEFAULT_WINDOW_SIZE,
    compute_phase_lag_index,
    extract_correlation_timeline,
    extract_correlation_trends,
    visualize_correlation_trajectory,
    visualize_phase_lag,
)


class TestExtractCorrelationTimeline:
    """Tests for extract_correlation_timeline function."""

    def test_empty_history_returns_empty_timeline(self):
        """Empty history should return empty timeline."""
        result = extract_correlation_timeline([], [])

        assert result["total_windows"] == 0
        assert result["windows"] == []
        assert result["window_size"] == DEFAULT_WINDOW_SIZE

    def test_single_window_extraction(self):
        """Extract correlation for a single window."""
        semantic_history = [
            {
                "timeline": [{"run_id": "run_001", "status": "CRITICAL"}],
                "runs_with_critical_signals": ["run_001"],
                "node_disappearance_events": [{"run_id": "run_001", "term": "slice_alpha"}],
                "trend": "DRIFTING",
                "semantic_status_light": "RED",
            }
        ] * 5  # 5 cycles

        tda_history = [
            {
                "tda_status": "ALERT",
                "block_rate": 0.25,
                "hss_trend": "DEGRADING",
                "governance_signal": "BLOCK",
            }
        ] * 5  # 5 cycles

        result = extract_correlation_timeline(semantic_history, tda_history, window_size=10)

        assert result["total_windows"] == 1
        assert len(result["windows"]) == 1
        assert result["windows"][0]["window_index"] == 0
        assert result["windows"][0]["start_cycle"] == 0
        assert result["windows"][0]["end_cycle"] == 4
        assert result["windows"][0]["correlation_coefficient"] >= 0.8  # High correlation

    def test_multiple_windows_extraction(self):
        """Extract correlation for multiple windows."""
        # Create 25 cycles of data
        semantic_history = [
            {
                "timeline": [],
                "runs_with_critical_signals": [],
                "node_disappearance_events": [],
                "trend": "STABLE",
                "semantic_status_light": "GREEN",
            }
        ] * 25

        tda_history = [
            {
                "tda_status": "OK",
                "block_rate": 0.05,
                "hss_trend": "STABLE",
                "governance_signal": "OK",
            }
        ] * 25

        result = extract_correlation_timeline(semantic_history, tda_history, window_size=10)

        assert result["total_windows"] == 3  # 25 cycles / 10 = 3 windows (ceiling)
        assert len(result["windows"]) == 3
        assert result["windows"][0]["start_cycle"] == 0
        assert result["windows"][0]["end_cycle"] == 9
        assert result["windows"][1]["start_cycle"] == 10
        assert result["windows"][1]["end_cycle"] == 19
        assert result["windows"][2]["start_cycle"] == 20
        assert result["windows"][2]["end_cycle"] == 24  # Last window is partial

    def test_mismatched_history_lengths_raises_error(self):
        """Mismatched history lengths should raise ValueError."""
        semantic_history = [{"timeline": []}] * 10
        tda_history = [{"tda_status": "OK"}] * 5

        with pytest.raises(ValueError, match="History lengths must match"):
            extract_correlation_timeline(semantic_history, tda_history)

    def test_window_aggregation_correct(self):
        """Window aggregation should correctly combine multiple cycles."""
        # Create 2 windows with different statuses
        semantic_history = (
            [
                {
                    "timeline": [{"run_id": "run_001", "status": "CRITICAL"}],
                    "runs_with_critical_signals": ["run_001"],
                    "node_disappearance_events": [{"run_id": "run_001", "term": "slice_alpha"}],
                    "trend": "DRIFTING",
                    "semantic_status_light": "RED",
                }
            ]
            * 10  # First window: RED
            + [
                {
                    "timeline": [],
                    "runs_with_critical_signals": [],
                    "node_disappearance_events": [],
                    "trend": "STABLE",
                    "semantic_status_light": "GREEN",
                }
            ]
            * 10  # Second window: GREEN
        )

        tda_history = (
            [
                {
                    "tda_status": "ALERT",
                    "block_rate": 0.25,
                    "hss_trend": "DEGRADING",
                    "governance_signal": "BLOCK",
                }
            ]
            * 10  # First window: ALERT
            + [
                {
                    "tda_status": "OK",
                    "block_rate": 0.05,
                    "hss_trend": "STABLE",
                    "governance_signal": "OK",
                }
            ]
            * 10  # Second window: OK
        )

        result = extract_correlation_timeline(semantic_history, tda_history, window_size=10)

        assert result["total_windows"] == 2
        # First window should have high correlation (both RED/ALERT)
        assert result["windows"][0]["correlation_coefficient"] >= 0.8
        assert result["windows"][0]["semantic_status"] == "RED"
        assert result["windows"][0]["tda_status"] == "ALERT"
        # Second window should have neutral correlation (both GREEN/OK)
        assert abs(result["windows"][1]["correlation_coefficient"]) < 0.1
        assert result["windows"][1]["semantic_status"] == "GREEN"
        assert result["windows"][1]["tda_status"] == "OK"


class TestExtractCorrelationTrends:
    """Tests for extract_correlation_trends function."""

    def test_empty_timeline_returns_default_trends(self):
        """Empty timeline should return default trend values."""
        timeline = {"windows": []}
        trends = extract_correlation_trends(timeline)

        assert trends["correlation_mean"] == 0.0
        assert trends["correlation_variance"] == 0.0
        assert trends["correlation_slope"] == 0.0
        assert trends["correlation_regime"] == "STABLE"

    def test_trend_computation(self):
        """Trend computation should correctly analyze correlation patterns."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.0, "window_index": 0},
                {"correlation_coefficient": 0.5, "window_index": 1},
                {"correlation_coefficient": 1.0, "window_index": 2},
            ]
        }

        trends = extract_correlation_trends(timeline)

        assert trends["correlation_mean"] == pytest.approx(0.5, abs=0.01)
        assert trends["correlation_slope"] > 0  # Positive trend
        # High variance (0.0 to 1.0) should classify as VOLATILE
        assert trends["correlation_regime"] == "VOLATILE"
        assert trends["windows_with_high_correlation"] == 1  # One window with >= 0.7

    def test_regime_classification(self):
        """Regime classification should correctly identify patterns."""
        # ALIGNED regime
        aligned_timeline = {
            "windows": [{"correlation_coefficient": 0.8, "window_index": i} for i in range(5)]
        }
        trends = extract_correlation_trends(aligned_timeline)
        assert trends["correlation_regime"] == "ALIGNED"

        # MISALIGNED regime
        misaligned_timeline = {
            "windows": [{"correlation_coefficient": -0.5, "window_index": i} for i in range(5)]
        }
        trends = extract_correlation_trends(misaligned_timeline)
        assert trends["correlation_regime"] == "MISALIGNED"

        # VOLATILE regime (high variance)
        volatile_timeline = {
            "windows": [
                {"correlation_coefficient": 1.0, "window_index": 0},
                {"correlation_coefficient": -1.0, "window_index": 1},
                {"correlation_coefficient": 1.0, "window_index": 2},
                {"correlation_coefficient": -1.0, "window_index": 3},
            ]
        }
        trends = extract_correlation_trends(volatile_timeline)
        assert trends["correlation_regime"] == "VOLATILE"


class TestVisualizeCorrelationTrajectory:
    """Tests for visualize_correlation_trajectory function."""

    def test_ascii_visualization(self):
        """ASCII visualization should produce readable output."""
        timeline = {
            "windows": [
                {
                    "window_index": 0,
                    "start_cycle": 0,
                    "end_cycle": 9,
                    "correlation_coefficient": 0.8,
                    "num_key_slices": 2,
                },
                {
                    "window_index": 1,
                    "start_cycle": 10,
                    "end_cycle": 19,
                    "correlation_coefficient": 0.5,
                    "num_key_slices": 1,
                },
            ]
        }

        output = visualize_correlation_trajectory(timeline, format="ascii")

        assert "Semantic-TDA Correlation Trajectory" in output
        assert "Total Windows: 2" in output
        assert "Window | Cycles" in output
        assert "0.800" in output or "0.8" in output

    def test_json_visualization(self):
        """JSON visualization should produce valid JSON."""
        timeline = {
            "windows": [
                {
                    "window_index": 0,
                    "start_cycle": 0,
                    "end_cycle": 9,
                    "correlation_coefficient": 0.8,
                    "num_key_slices": 2,
                }
            ]
        }

        output = visualize_correlation_trajectory(timeline, format="json")

        # Should be valid JSON
        parsed = json.loads(output)
        assert "total_windows" in parsed
        assert "correlation_range" in parsed
        assert parsed["total_windows"] == 1

    def test_empty_timeline_visualization(self):
        """Empty timeline should produce appropriate output."""
        timeline = {"windows": []}

        ascii_output = visualize_correlation_trajectory(timeline, format="ascii")
        assert "No correlation data" in ascii_output

        json_output = visualize_correlation_trajectory(timeline, format="json")
        parsed = json.loads(json_output)
        assert "No windows" in parsed["summary"] or parsed.get("total_windows") == 0


class TestPatternClassifierIntegration:
    """Tests for pattern classifier integration."""

    def test_extract_correlation_for_pattern_classifier(self):
        """Extract correlation data formatted for pattern classifier."""
        semantic_history = [
            {
                "timeline": [{"run_id": f"run_{i:03d}", "status": "CRITICAL"}],
                "runs_with_critical_signals": [f"run_{i:03d}"],
                "node_disappearance_events": [{"run_id": f"run_{i:03d}", "term": "slice_alpha"}],
                "trend": "DRIFTING",
                "semantic_status_light": "RED",
            }
            for i in range(20)  # 20 cycles
        ]

        tda_history = [
            {
                "tda_status": "ALERT",
                "block_rate": 0.25,
                "hss_trend": "DEGRADING",
                "governance_signal": "BLOCK",
            }
        ] * 20

        result = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        assert "timeline" in result
        assert "trends" in result
        assert "pattern_classifier_input" in result

        # Check pattern classifier input format
        pci = result["pattern_classifier_input"]
        assert "correlation_mean" in pci
        assert "correlation_slope" in pci
        assert "correlation_regime" in pci
        assert "alignment_strength" in pci
        assert 0.0 <= pci["alignment_strength"] <= 1.0

        # Check timeline structure
        assert result["timeline"]["total_windows"] == 2
        assert len(result["timeline"]["windows"]) == 2

    def test_alignment_strength_normalization(self):
        """Alignment strength should be normalized to [0, 1]."""
        # High correlation (1.0) -> alignment_strength should be 1.0
        semantic_history = [
            {
                "timeline": [{"run_id": "run_001", "status": "CRITICAL"}],
                "runs_with_critical_signals": ["run_001"],
                "node_disappearance_events": [{"run_id": "run_001", "term": "slice_alpha"}],
                "trend": "DRIFTING",
                "semantic_status_light": "RED",
            }
        ] * 10

        tda_history = [
            {
                "tda_status": "ALERT",
                "block_rate": 0.25,
                "hss_trend": "DEGRADING",
                "governance_signal": "BLOCK",
            }
        ] * 10

        result = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        pci = result["pattern_classifier_input"]
        # High correlation should map to high alignment strength
        assert pci["alignment_strength"] >= 0.8

    def test_pattern_classifier_input_is_json_serializable(self):
        """Pattern classifier input must be JSON serializable."""
        semantic_history = [
            {
                "timeline": [],
                "runs_with_critical_signals": [],
                "node_disappearance_events": [],
                "trend": "STABLE",
                "semantic_status_light": "GREEN",
            }
        ] * 10

        tda_history = [
            {
                "tda_status": "OK",
                "block_rate": 0.05,
                "hss_trend": "STABLE",
                "governance_signal": "OK",
            }
        ] * 10

        result = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        # Should serialize without error
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["pattern_classifier_input"]["correlation_regime"] == result["pattern_classifier_input"]["correlation_regime"]


class TestPhaseLagIndex:
    """Tests for compute_phase_lag_index function."""

    def test_stable_timeline_has_low_phase_lag_index(self):
        """Stable correlation timeline should produce low phase lag index."""
        # Create stable timeline: all correlations high and positive
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.9, "window_index": i} for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Stable timeline should have low phase lag index (< 0.3)
        assert phase_lag["phase_lag_index"] < 0.3
        assert phase_lag["phase_lag_index"] >= 0.0

    def test_lagged_timeline_has_high_phase_lag_index(self):
        """Lagged correlation timeline (oscillating signs) should produce high phase lag index."""
        # Create lagged timeline: alternating positive/negative correlations
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.8 if i % 2 == 0 else -0.7, "window_index": i}
                for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Lagged timeline should have high phase lag index (> 0.5)
        assert phase_lag["phase_lag_index"] > 0.5
        assert phase_lag["phase_lag_index"] <= 1.0

    def test_negative_slope_increases_phase_lag_index(self):
        """Negative correlation slope should increase phase lag index."""
        # Create timeline with negative slope (deteriorating alignment)
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.9 - (i * 0.1), "window_index": i}
                for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Negative slope should contribute to phase lag
        assert phase_lag["correlation_slope_contribution"] > 0.0
        assert phase_lag["phase_lag_index"] > 0.2

    def test_sign_changes_increase_phase_lag_index(self):
        """Frequent sign changes should increase phase lag index."""
        # Create timeline with many sign changes
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.5, "window_index": i}
                for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Many sign changes should contribute to phase lag
        assert phase_lag["sign_change_contribution"] > 0.3
        assert phase_lag["phase_lag_index"] > 0.4

    def test_low_alignment_increases_phase_lag_index(self):
        """Low alignment strength should increase phase lag index."""
        # Create timeline with low correlations and negative slope
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.1 - (i * 0.02), "window_index": i} for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Low alignment should contribute to phase lag
        assert phase_lag["alignment_contribution"] > 0.4
        # With negative slope, overall index should be higher
        assert phase_lag["phase_lag_index"] > 0.2

    def test_phase_lag_index_is_deterministic(self):
        """Phase lag index computation should be deterministic."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.4, "window_index": i}
                for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag_1 = compute_phase_lag_index(timeline, trends=trends)
        phase_lag_2 = compute_phase_lag_index(timeline, trends=trends)

        # Results should be identical
        assert phase_lag_1["phase_lag_index"] == phase_lag_2["phase_lag_index"]
        assert phase_lag_1["correlation_slope_contribution"] == phase_lag_2["correlation_slope_contribution"]
        assert phase_lag_1["sign_change_contribution"] == phase_lag_2["sign_change_contribution"]
        assert phase_lag_1["alignment_contribution"] == phase_lag_2["alignment_contribution"]
        assert phase_lag_1["worst_windows"] == phase_lag_2["worst_windows"]

    def test_worst_windows_identification(self):
        """Worst windows should be identified correctly."""
        # Create timeline with some windows having low correlation
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.9 if i != 3 and i != 7 else 0.1, "window_index": i}
                for i in range(10)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Worst windows should include windows with low correlation
        assert len(phase_lag["worst_windows"]) > 0
        # Windows 3 and 7 should be in worst windows (low correlation)
        assert 3 in phase_lag["worst_windows"] or 7 in phase_lag["worst_windows"]

    def test_empty_timeline_returns_default_phase_lag(self):
        """Empty timeline should return default phase lag values."""
        timeline = {"windows": []}
        phase_lag = compute_phase_lag_index(timeline)

        assert phase_lag["phase_lag_index"] == 0.0
        assert phase_lag["correlation_slope_contribution"] == 0.0
        assert phase_lag["sign_change_contribution"] == 0.0
        assert phase_lag["alignment_contribution"] == 0.0
        assert phase_lag["worst_windows"] == []

    def test_phase_lag_index_is_in_valid_range(self):
        """Phase lag index should always be in [0, 1]."""
        # Test with various timelines
        test_cases = [
            # Stable
            [{"correlation_coefficient": 0.9, "window_index": i} for i in range(10)],
            # Lagged
            [{"correlation_coefficient": 0.5 if i % 2 == 0 else -0.5, "window_index": i} for i in range(10)],
            # Mixed
            [{"correlation_coefficient": 0.3 - (i * 0.05), "window_index": i} for i in range(10)],
        ]

        for windows in test_cases:
            timeline = {"windows": windows}
            trends = extract_correlation_trends(timeline)
            phase_lag = compute_phase_lag_index(timeline, trends=trends)

            assert 0.0 <= phase_lag["phase_lag_index"] <= 1.0
            assert 0.0 <= phase_lag["correlation_slope_contribution"] <= 1.0
            assert 0.0 <= phase_lag["sign_change_contribution"] <= 1.0
            assert 0.0 <= phase_lag["alignment_contribution"] <= 1.0


class TestPhaseLagVisualization:
    """Tests for visualize_phase_lag function."""

    def test_ascii_visualization(self):
        """ASCII visualization should produce readable output."""
        timeline = {
            "windows": [
                {
                    "window_index": i,
                    "start_cycle": i * 10,
                    "end_cycle": (i * 10) + 9,
                    "correlation_coefficient": 0.5 if i % 2 == 0 else -0.4,
                }
                for i in range(5)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        output = visualize_phase_lag(timeline, phase_lag_data=phase_lag, format="ascii")

        assert "Semantic-TDA Phase Lag Analysis" in output
        assert "Phase Lag Index:" in output
        assert "Contributions to Phase Lag:" in output
        assert "Worst Windows" in output or "worst_windows" in str(phase_lag)

    def test_json_visualization(self):
        """JSON visualization should produce valid JSON."""
        timeline = {
            "windows": [
                {
                    "window_index": i,
                    "start_cycle": i * 10,
                    "end_cycle": (i * 10) + 9,
                    "correlation_coefficient": 0.8,
                }
                for i in range(3)
            ]
        }

        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        output = visualize_phase_lag(timeline, phase_lag_data=phase_lag, format="json")

        # Should be valid JSON
        parsed = json.loads(output)
        assert "phase_lag_index" in parsed
        assert "worst_windows" in parsed
        assert isinstance(parsed["phase_lag_index"], (int, float))

    def test_empty_timeline_visualization(self):
        """Empty timeline should produce appropriate output."""
        timeline = {"windows": []}

        ascii_output = visualize_phase_lag(timeline, format="ascii")
        assert "No correlation data" in ascii_output

        json_output = visualize_phase_lag(timeline, format="json")
        parsed = json.loads(json_output)
        assert "No windows" in parsed["summary"] or parsed.get("phase_lag_index") is None


class TestPatternClassifierWithPhaseLag:
    """Tests for pattern classifier integration with phase lag."""

    def test_pattern_classifier_includes_phase_lag_index(self):
        """Pattern classifier input should include phase_lag_index."""
        semantic_history = [
            {
                "timeline": [],
                "runs_with_critical_signals": [],
                "node_disappearance_events": [],
                "trend": "STABLE",
                "semantic_status_light": "GREEN",
            }
        ] * 20

        tda_history = [
            {
                "tda_status": "OK",
                "block_rate": 0.05,
                "hss_trend": "STABLE",
                "governance_signal": "OK",
            }
        ] * 20

        result = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        # Should include phase_lag section
        assert "phase_lag" in result
        assert "phase_lag_index" in result["phase_lag"]

        # Should include phase_lag_index in pattern_classifier_input
        pci = result["pattern_classifier_input"]
        assert "phase_lag_index" in pci
        assert 0.0 <= pci["phase_lag_index"] <= 1.0

    def test_phase_lag_in_pattern_classifier_is_json_serializable(self):
        """Phase lag data in pattern classifier should be JSON serializable."""
        semantic_history = [
            {
                "timeline": [{"run_id": f"run_{i:03d}", "status": "CRITICAL"}],
                "runs_with_critical_signals": [f"run_{i:03d}"],
                "node_disappearance_events": [{"run_id": f"run_{i:03d}", "term": "slice_alpha"}],
                "trend": "DRIFTING",
                "semantic_status_light": "RED",
            }
            for i in range(20)
        ]

        tda_history = [
            {
                "tda_status": "ALERT",
                "block_rate": 0.25,
                "hss_trend": "DEGRADING",
                "governance_signal": "BLOCK",
            }
        ] * 20

        result = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        # Should serialize without error
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["pattern_classifier_input"]["phase_lag_index"] == result["pattern_classifier_input"]["phase_lag_index"]

