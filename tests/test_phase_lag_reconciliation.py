"""Tests for phase-lag vs divergence reconciliation.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

Tests the connection between phase-lag index and divergence decomposition
to distinguish state lag from outcome noise.
"""

import json

import pytest

from backend.health.semantic_tda_timeline import (
    compute_phase_lag_index,
    explain_phase_lag_vs_divergence,
    extract_correlation_trends,
)


class TestPhaseLagVsDivergence:
    """Tests for explain_phase_lag_vs_divergence function."""

    def test_state_lag_dominant_interpretation(self):
        """High phase lag + high state divergence → STATE_LAG_DOMINANT."""
        # Create phase lag data with high index
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.4, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # High state divergence, low outcome divergence
        divergence_decomp = {
            "state_divergence_rate": 0.6,  # High
            "success_divergence_rate": 0.1,  # Low
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "STATE_LAG_DOMINANT"
        assert result["phase_lag_index"] >= 0.4  # High phase lag
        assert result["state_divergence_rate"] == 0.6
        assert result["outcome_divergence_rate_success"] == 0.1
        assert len(result["notes"]) > 0
        assert "temporal misalignment" in " ".join(result["notes"]).lower()

    def test_outcome_noise_dominant_interpretation(self):
        """High phase lag + low state divergence + high outcome divergence → OUTCOME_NOISE_DOMINANT."""
        # Create phase lag data with high index
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.3 if i % 2 == 0 else -0.2, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Low state divergence, high outcome divergence
        divergence_decomp = {
            "state_divergence_rate": 0.15,  # Low
            "success_divergence_rate": 0.35,  # High
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "OUTCOME_NOISE_DOMINANT"
        assert result["phase_lag_index"] >= 0.4  # High phase lag
        assert result["state_divergence_rate"] == 0.15
        assert result["outcome_divergence_rate_success"] == 0.35
        assert len(result["notes"]) > 0
        assert "outcome prediction noise" in " ".join(result["notes"]).lower()

    def test_mixed_interpretation(self):
        """High phase lag + both high → MIXED."""
        # Create phase lag data with high index
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.4 if i % 2 == 0 else -0.3, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Both high
        divergence_decomp = {
            "state_divergence_rate": 0.5,  # High
            "success_divergence_rate": 0.3,  # High
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "MIXED"
        assert result["phase_lag_index"] >= 0.4  # High phase lag
        assert result["state_divergence_rate"] == 0.5
        assert result["outcome_divergence_rate_success"] == 0.3
        assert len(result["notes"]) > 0
        assert "both" in " ".join(result["notes"]).lower() or "temporal" in " ".join(result["notes"]).lower()

    def test_insufficient_data_interpretation(self):
        """Missing data → INSUFFICIENT_DATA."""
        phase_lag = {"phase_lag_index": 0.5}

        # Missing divergence rates
        divergence_decomp = {}

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "INSUFFICIENT_DATA"
        assert result["state_divergence_rate"] is None
        assert result["outcome_divergence_rate_success"] is None
        assert len(result["notes"]) > 0
        assert "missing" in " ".join(result["notes"]).lower()

    def test_low_phase_lag_with_high_state_divergence(self):
        """Low phase lag + high state divergence → MIXED (calibration issue)."""
        # Create phase lag data with low index
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.9, "window_index": i} for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # High state divergence despite low phase lag
        divergence_decomp = {
            "state_divergence_rate": 0.5,  # High
            "success_divergence_rate": 0.1,  # Low
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "MIXED"
        assert result["phase_lag_index"] < 0.4  # Low phase lag
        assert "calibration" in " ".join(result["notes"]).lower() or "state divergence" in " ".join(result["notes"]).lower()

    def test_low_phase_lag_with_low_divergence(self):
        """Low phase lag + low divergence → OUTCOME_NOISE_DOMINANT."""
        # Create phase lag data with low index
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.85, "window_index": i} for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Low divergence rates
        divergence_decomp = {
            "state_divergence_rate": 0.1,  # Low
            "success_divergence_rate": 0.15,  # Low
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "OUTCOME_NOISE_DOMINANT"
        assert result["phase_lag_index"] < 0.4  # Low phase lag
        assert "noise" in " ".join(result["notes"]).lower() or "aligned" in " ".join(result["notes"]).lower()

    def test_outcome_divergence_rate_success_key_variant(self):
        """Should handle outcome_divergence_rate_success key variant."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.4, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Use outcome_divergence_rate_success key
        divergence_decomp = {
            "state_divergence_rate": 0.6,
            "outcome_divergence_rate_success": 0.1,  # Different key
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "STATE_LAG_DOMINANT"
        assert result["outcome_divergence_rate_success"] == 0.1

    def test_result_is_json_serializable(self):
        """Result should be JSON serializable."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.4, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        divergence_decomp = {
            "state_divergence_rate": 0.6,
            "success_divergence_rate": 0.1,
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        # Should serialize without error
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["interpretation"] == result["interpretation"]
        assert parsed["phase_lag_index"] == result["phase_lag_index"]

    def test_result_is_deterministic(self):
        """Result should be deterministic for same inputs."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.5 if i % 2 == 0 else -0.4, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        divergence_decomp = {
            "state_divergence_rate": 0.6,
            "success_divergence_rate": 0.1,
        }

        result1 = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)
        result2 = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        # Results should be identical
        assert result1["interpretation"] == result2["interpretation"]
        assert result1["phase_lag_index"] == result2["phase_lag_index"]
        assert result1["state_divergence_rate"] == result2["state_divergence_rate"]
        assert result1["outcome_divergence_rate_success"] == result2["outcome_divergence_rate_success"]
        assert result1["notes"] == result2["notes"]

    def test_high_phase_lag_with_low_divergence_rates(self):
        """High phase lag but low divergence rates → MIXED (subtle misalignment)."""
        timeline = {
            "windows": [
                {"correlation_coefficient": 0.3 if i % 2 == 0 else -0.2, "window_index": i}
                for i in range(10)
            ]
        }
        trends = extract_correlation_trends(timeline)
        phase_lag = compute_phase_lag_index(timeline, trends=trends)

        # Low divergence rates despite high phase lag
        divergence_decomp = {
            "state_divergence_rate": 0.15,  # Low
            "success_divergence_rate": 0.1,  # Low
        }

        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        assert result["interpretation"] == "MIXED"
        assert result["phase_lag_index"] >= 0.4  # High phase lag
        assert "subtle" in " ".join(result["notes"]).lower() or "not yet" in " ".join(result["notes"]).lower()

    def test_integration_with_extract_correlation_for_pattern_classifier(self):
        """Integration test: real-ish phase_lag from extract_correlation_for_pattern_classifier + A1 annex divergence_decomp."""
        from backend.health.semantic_tda_adapter import extract_correlation_for_pattern_classifier

        # Create realistic semantic and TDA history
        semantic_history = [
            {
                "timeline": [{"run_id": f"run_{i:03d}", "status": "CRITICAL"}],
                "runs_with_critical_signals": [f"run_{i:03d}"],
                "node_disappearance_events": [{"run_id": f"run_{i:03d}", "term": "slice_alpha"}],
                "trend": "DRIFTING",
                "semantic_status_light": "RED",
            }
            for i in range(30)  # 30 cycles
        ]

        tda_history = [
            {
                "tda_status": "ALERT",
                "block_rate": 0.25,
                "hss_trend": "DEGRADING",
                "governance_signal": "BLOCK",
            }
        ] * 30

        # Extract correlation data (real-ish phase_lag)
        correlation_data = extract_correlation_for_pattern_classifier(
            semantic_history, tda_history, window_size=10
        )

        # Extract phase_lag from correlation data
        phase_lag = correlation_data["phase_lag"]

        # Create A1 annex shape divergence_decomp (from decompose_divergence_components)
        # This mimics the structure from experiments/u2/runtime/calibration_correlation.py
        divergence_decomp = {
            "state_divergence_rate": 0.6,  # From overall.state_divergence_rate
            "success_divergence_rate": 0.15,  # From overall.outcome_divergence_rate_success
        }

        # Run reconciliation
        result = explain_phase_lag_vs_divergence(phase_lag, divergence_decomp)

        # Verify structure
        assert "schema_version" in result
        assert "phase_lag_index" in result
        assert "state_divergence_rate" in result
        assert "outcome_divergence_rate_success" in result
        assert "interpretation" in result
        assert "notes" in result
        assert "thresholds" in result
        assert "basis" in result

        # Verify thresholds
        assert result["thresholds"]["high_phase_lag"] == 0.4
        assert result["thresholds"]["high_state_divergence"] == 0.3
        assert result["thresholds"]["high_outcome_divergence"] == 0.2

        # Verify basis
        assert result["basis"]["phase_lag"] == "semantic_tda_timeline"
        assert result["basis"]["divergence_decomp"] == "runtime_profile_calibration.decompose_divergence_components"

        # Verify interpretation is valid
        assert result["interpretation"] in [
            "STATE_LAG_DOMINANT",
            "OUTCOME_NOISE_DOMINANT",
            "MIXED",
            "INSUFFICIENT_DATA",
        ]

        # Verify JSON serialization
        import json

        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["interpretation"] == result["interpretation"]
        assert parsed["thresholds"] == result["thresholds"]
        assert parsed["basis"] == result["basis"]

