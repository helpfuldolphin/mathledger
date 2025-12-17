"""
Tests for Metrics Threshold Registry (P5 Migration).

SHADOW MODE: These tests verify threshold management only.
No governance decisions are enforced.

REAL-READY: Tests cover MOCK, HYBRID, REAL modes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest import mock

import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def clean_env():
    """Ensure clean environment for threshold mode tests."""
    original = os.environ.get("METRIC_THRESHOLDS_MODE")
    yield
    if original is None:
        os.environ.pop("METRIC_THRESHOLDS_MODE", None)
    else:
        os.environ["METRIC_THRESHOLDS_MODE"] = original


@pytest.fixture
def mock_mode(clean_env):
    """Set MOCK mode for tests."""
    os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
    yield


@pytest.fixture
def hybrid_mode(clean_env):
    """Set HYBRID mode for tests."""
    os.environ["METRIC_THRESHOLDS_MODE"] = "HYBRID"
    yield


@pytest.fixture
def real_mode(clean_env):
    """Set REAL mode for tests."""
    os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
    yield


@pytest.fixture
def healthy_metrics() -> Dict[str, float]:
    """Metrics that should be GREEN in all modes."""
    return {
        "drift_magnitude": 0.10,
        "success_rate": 95.0,
        "budget_utilization": 50.0,
        "abstention_rate": 2.0,
        "block_rate": 0.03,
    }


@pytest.fixture
def warn_boundary_metrics() -> Dict[str, float]:
    """Metrics at MOCK warn boundary (YELLOW in MOCK, GREEN in REAL)."""
    return {
        "drift_magnitude": 0.32,  # >= 0.30 (MOCK), < 0.35 (REAL)
        "success_rate": 78.0,     # < 80 (MOCK), >= 75 (REAL)
        "budget_utilization": 82.0,  # >= 80 (MOCK), < 85 (REAL)
        "abstention_rate": 6.0,   # >= 5 (MOCK), < 8 (REAL)
        "block_rate": 0.09,       # >= 0.08 (MOCK), < 0.12 (REAL)
    }


@pytest.fixture
def critical_metrics() -> Dict[str, float]:
    """Metrics that should be RED in all modes."""
    return {
        "drift_magnitude": 0.80,
        "success_rate": 30.0,
        "budget_utilization": 98.0,
        "abstention_rate": 20.0,
        "block_rate": 0.30,
    }


# ==============================================================================
# Test Class: Mode Management
# ==============================================================================


class TestModeManagement:
    """Tests for threshold mode management."""

    def test_default_mode_is_mock(self, clean_env):
        """Default mode should be MOCK when env not set."""
        os.environ.pop("METRIC_THRESHOLDS_MODE", None)

        from backend.health.metrics_thresholds import get_threshold_mode, MODE_MOCK

        assert get_threshold_mode() == MODE_MOCK

    def test_mock_mode_from_env(self, mock_mode):
        """MOCK mode should be read from environment."""
        from backend.health.metrics_thresholds import get_threshold_mode, MODE_MOCK

        assert get_threshold_mode() == MODE_MOCK

    def test_hybrid_mode_from_env(self, hybrid_mode):
        """HYBRID mode should be read from environment."""
        from backend.health.metrics_thresholds import get_threshold_mode, MODE_HYBRID

        assert get_threshold_mode() == MODE_HYBRID

    def test_real_mode_from_env(self, real_mode):
        """REAL mode should be read from environment."""
        from backend.health.metrics_thresholds import get_threshold_mode, MODE_REAL

        assert get_threshold_mode() == MODE_REAL

    def test_invalid_mode_falls_back_to_mock(self, clean_env):
        """Invalid mode should fall back to MOCK."""
        os.environ["METRIC_THRESHOLDS_MODE"] = "INVALID"

        from backend.health.metrics_thresholds import get_threshold_mode, MODE_MOCK

        assert get_threshold_mode() == MODE_MOCK

    def test_mode_is_case_insensitive(self, clean_env):
        """Mode should be case-insensitive."""
        from backend.health.metrics_thresholds import get_threshold_mode, MODE_REAL

        os.environ["METRIC_THRESHOLDS_MODE"] = "real"
        assert get_threshold_mode() == MODE_REAL

        os.environ["METRIC_THRESHOLDS_MODE"] = "Real"
        assert get_threshold_mode() == MODE_REAL


# ==============================================================================
# Test Class: Threshold Access
# ==============================================================================


class TestThresholdAccess:
    """Tests for threshold value access."""

    def test_get_threshold_mock_values(self, mock_mode):
        """MOCK thresholds should return P3/P4 values."""
        from backend.health.metrics_thresholds import get_threshold

        assert get_threshold("drift_warn") == 0.30
        assert get_threshold("drift_critical") == 0.70
        assert get_threshold("success_rate_warn") == 80.0
        assert get_threshold("success_rate_critical") == 50.0
        assert get_threshold("budget_warn") == 80.0
        assert get_threshold("budget_critical") == 95.0

    def test_get_threshold_real_values(self, real_mode):
        """REAL thresholds should return P5 values."""
        from backend.health.metrics_thresholds import get_threshold

        assert get_threshold("drift_warn") == 0.35
        assert get_threshold("drift_critical") == 0.75
        assert get_threshold("success_rate_warn") == 75.0
        assert get_threshold("success_rate_critical") == 45.0
        assert get_threshold("budget_warn") == 85.0
        assert get_threshold("budget_critical") == 92.0

    def test_get_threshold_hybrid_uses_mock(self, hybrid_mode):
        """HYBRID mode should use MOCK thresholds."""
        from backend.health.metrics_thresholds import get_threshold

        assert get_threshold("drift_warn") == 0.30
        assert get_threshold("success_rate_warn") == 80.0

    def test_get_threshold_with_explicit_mode(self, mock_mode):
        """Explicit mode should override env."""
        from backend.health.metrics_thresholds import get_threshold

        # Env is MOCK, but we request REAL
        assert get_threshold("drift_warn", "REAL") == 0.35
        assert get_threshold("drift_warn", "MOCK") == 0.30

    def test_get_all_thresholds_returns_copy(self, mock_mode):
        """get_all_thresholds should return a copy."""
        from backend.health.metrics_thresholds import get_all_thresholds

        thresholds1 = get_all_thresholds()
        thresholds2 = get_all_thresholds()

        assert thresholds1 == thresholds2
        assert thresholds1 is not thresholds2  # Different objects

        # Mutation should not affect original
        thresholds1["drift_warn"] = 999.0
        thresholds3 = get_all_thresholds()
        assert thresholds3["drift_warn"] == 0.30

    def test_get_threshold_pair(self, mock_mode):
        """get_threshold_pair should return both values."""
        from backend.health.metrics_thresholds import get_threshold_pair

        pair = get_threshold_pair("drift_warn")
        assert pair["mock"] == 0.30
        assert pair["real"] == 0.35

    def test_list_threshold_names(self, mock_mode):
        """list_threshold_names should return all names."""
        from backend.health.metrics_thresholds import list_threshold_names

        names = list_threshold_names()
        assert "drift_warn" in names
        assert "drift_critical" in names
        assert "success_rate_warn" in names
        assert "budget_critical" in names
        assert len(names) == 10  # 5 metrics * 2 levels

    def test_unknown_threshold_raises_keyerror(self, mock_mode):
        """Unknown threshold name should raise KeyError."""
        from backend.health.metrics_thresholds import get_threshold

        with pytest.raises(KeyError):
            get_threshold("nonexistent_threshold")


# ==============================================================================
# Test Class: Safe Comparison Bands
# ==============================================================================


class TestSafeComparisonBands:
    """Tests for safe comparison band functionality."""

    def test_get_safe_band_values(self):
        """Safe bands should match spec values."""
        from backend.health.metrics_thresholds import get_safe_band

        assert get_safe_band("success_rate") == 15.0
        assert get_safe_band("block_rate") == 0.08
        assert get_safe_band("abstention_rate") == 5.0
        assert get_safe_band("drift_magnitude") == 0.15
        assert get_safe_band("budget_utilization") == 10.0

    def test_get_all_safe_bands_returns_copy(self):
        """get_all_safe_bands should return a copy."""
        from backend.health.metrics_thresholds import get_all_safe_bands

        bands1 = get_all_safe_bands()
        bands2 = get_all_safe_bands()

        assert bands1 == bands2
        assert bands1 is not bands2

    def test_check_in_band_within_band(self):
        """Metrics within band should return in_band=True."""
        from backend.health.metrics_thresholds import check_in_band

        result = check_in_band("success_rate", 90.0, 82.0)
        assert result["in_band"] is True
        assert result["delta"] == 8.0
        assert result["band"] == 15.0

    def test_check_in_band_outside_band(self):
        """Metrics outside band should return in_band=False."""
        from backend.health.metrics_thresholds import check_in_band

        result = check_in_band("success_rate", 90.0, 70.0)
        assert result["in_band"] is False
        assert result["delta"] == 20.0

    def test_check_in_band_at_boundary(self):
        """Metrics at exact boundary should be in_band=True."""
        from backend.health.metrics_thresholds import check_in_band

        result = check_in_band("success_rate", 90.0, 75.0)  # delta = 15.0 = band
        assert result["in_band"] is True

    def test_check_in_band_unknown_metric(self):
        """Unknown metric should return error."""
        from backend.health.metrics_thresholds import check_in_band

        result = check_in_band("unknown_metric", 1.0, 2.0)
        assert result["in_band"] is None
        assert "error" in result

    def test_log_band_position_all_in_band(self):
        """All metrics in band should set all_in_band=True."""
        from backend.health.metrics_thresholds import log_band_position

        p3_metrics = {
            "success_rate": 90.0,
            "block_rate": 0.05,
            "abstention_rate": 3.0,
            "drift_magnitude": 0.20,
            "budget_utilization": 70.0,
        }
        p5_metrics = {
            "success_rate": 85.0,
            "block_rate": 0.08,
            "abstention_rate": 5.0,
            "drift_magnitude": 0.25,
            "budget_utilization": 75.0,
        }

        result = log_band_position(p3_metrics, p5_metrics)
        assert result["all_in_band"] is True
        assert result["out_of_band_count"] == 0
        assert result["out_of_band_metrics"] == []

    def test_log_band_position_some_outside(self):
        """Some metrics outside band should report correctly."""
        from backend.health.metrics_thresholds import log_band_position

        p3_metrics = {
            "success_rate": 90.0,
            "block_rate": 0.05,
        }
        p5_metrics = {
            "success_rate": 60.0,  # Outside band (delta=30, band=15)
            "block_rate": 0.05,
        }

        result = log_band_position(p3_metrics, p5_metrics)
        assert result["all_in_band"] is False
        assert result["out_of_band_count"] >= 1
        assert "success_rate" in result["out_of_band_metrics"]


# ==============================================================================
# Test Class: Dual Threshold Evaluation
# ==============================================================================


class TestDualThresholdEvaluation:
    """Tests for dual threshold evaluation."""

    def test_single_evaluation_mock_mode(self, mock_mode, healthy_metrics):
        """MOCK mode should do single evaluation."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(healthy_metrics)

        assert result["mode"] == "MOCK"
        assert result["dual_evaluation"] is False
        assert result["verdict"]["status"] == "GREEN"
        assert "p5_verdict" not in result

    def test_single_evaluation_real_mode(self, real_mode, healthy_metrics):
        """REAL mode should do single evaluation."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(healthy_metrics)

        assert result["mode"] == "REAL"
        assert result["dual_evaluation"] is False
        assert result["verdict"]["status"] == "GREEN"

    def test_dual_evaluation_hybrid_mode(self, hybrid_mode, healthy_metrics):
        """HYBRID mode should do dual evaluation."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(healthy_metrics)

        assert result["mode"] == "HYBRID"
        assert result["dual_evaluation"] is True
        assert "verdict" in result
        assert "p5_verdict" in result
        assert result["diverges"] is False

    def test_hybrid_divergence_detection(self, hybrid_mode, warn_boundary_metrics):
        """HYBRID mode should detect divergence at boundary."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        assert result["mode"] == "HYBRID"
        assert result["dual_evaluation"] is True
        assert result["diverges"] is True
        assert result["verdict"]["status"] == "YELLOW"  # MOCK (authoritative)
        assert result["p5_verdict"]["status"] == "GREEN"  # REAL
        assert "divergence_detail" in result

    def test_hybrid_mock_is_authoritative(self, hybrid_mode, warn_boundary_metrics):
        """HYBRID mode should use MOCK as authoritative verdict."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        # MOCK verdict (YELLOW) should be authoritative
        assert result["verdict"]["status"] == "YELLOW"
        assert result["verdict"]["mode"] == "MOCK"

    def test_critical_metrics_red_in_all_modes(
        self, mock_mode, hybrid_mode, real_mode, critical_metrics
    ):
        """Critical metrics should be RED in all modes."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        # Test each mode
        for mode_fixture in ["mock_mode", "hybrid_mode", "real_mode"]:
            # Need to re-import to pick up mode change
            import importlib
            import backend.health.metrics_thresholds as mt
            importlib.reload(mt)

            result = mt.evaluate_with_dual_thresholds(critical_metrics)
            assert result["verdict"]["status"] == "RED"

    def test_evaluation_includes_reasons(self, mock_mode, warn_boundary_metrics):
        """Evaluation should include reasons for status."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        assert "reasons" in result["verdict"]
        assert len(result["verdict"]["reasons"]) > 0

    def test_divergent_thresholds_identified(self, hybrid_mode, warn_boundary_metrics):
        """Divergent thresholds should be identified."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        assert result["diverges"] is True
        detail = result["divergence_detail"]
        assert "triggered_thresholds" in detail
        # At least one threshold should be divergent
        assert len(detail["triggered_thresholds"]) > 0


# ==============================================================================
# Test Class: Band Transitions
# ==============================================================================


class TestBandTransitions:
    """Tests for threshold band transitions between modes."""

    def test_drift_warn_transition(self, clean_env):
        """Drift at 0.32 should be YELLOW in MOCK, GREEN in REAL."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        metrics = {"drift_magnitude": 0.32}

        os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
        mock_result = evaluate_with_dual_thresholds(metrics)
        assert mock_result["verdict"]["status"] == "YELLOW"

        os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
        real_result = evaluate_with_dual_thresholds(metrics)
        assert real_result["verdict"]["status"] == "GREEN"

    def test_success_rate_warn_transition(self, clean_env):
        """Success rate at 78% should be YELLOW in MOCK, GREEN in REAL."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        metrics = {"success_rate": 78.0}

        os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
        mock_result = evaluate_with_dual_thresholds(metrics)
        assert mock_result["verdict"]["status"] == "YELLOW"

        os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
        real_result = evaluate_with_dual_thresholds(metrics)
        assert real_result["verdict"]["status"] == "GREEN"

    def test_budget_warn_transition(self, clean_env):
        """Budget at 82% should be YELLOW in MOCK, GREEN in REAL."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        metrics = {"budget_utilization": 82.0}

        os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
        mock_result = evaluate_with_dual_thresholds(metrics)
        assert mock_result["verdict"]["status"] == "YELLOW"

        os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
        real_result = evaluate_with_dual_thresholds(metrics)
        assert real_result["verdict"]["status"] == "GREEN"

    def test_abstention_warn_transition(self, clean_env):
        """Abstention at 6% should be YELLOW in MOCK, GREEN in REAL."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        metrics = {"abstention_rate": 6.0}

        os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
        mock_result = evaluate_with_dual_thresholds(metrics)
        assert mock_result["verdict"]["status"] == "YELLOW"

        os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
        real_result = evaluate_with_dual_thresholds(metrics)
        assert real_result["verdict"]["status"] == "GREEN"

    def test_block_rate_warn_transition(self, clean_env):
        """Block rate at 0.09 should be YELLOW in MOCK, GREEN in REAL."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        metrics = {"block_rate": 0.09}

        os.environ["METRIC_THRESHOLDS_MODE"] = "MOCK"
        mock_result = evaluate_with_dual_thresholds(metrics)
        assert mock_result["verdict"]["status"] == "YELLOW"

        os.environ["METRIC_THRESHOLDS_MODE"] = "REAL"
        real_result = evaluate_with_dual_thresholds(metrics)
        assert real_result["verdict"]["status"] == "GREEN"


# ==============================================================================
# Test Class: JSON Serialization
# ==============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization of threshold outputs."""

    def test_evaluation_result_is_json_serializable(self, hybrid_mode, warn_boundary_metrics):
        """Evaluation result should be JSON-serializable."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["mode"] == result["mode"]
        assert parsed["diverges"] == result["diverges"]

    def test_band_position_is_json_serializable(self):
        """Band position result should be JSON-serializable."""
        from backend.health.metrics_thresholds import log_band_position

        result = log_band_position(
            {"success_rate": 90.0, "block_rate": 0.05},
            {"success_rate": 85.0, "block_rate": 0.08},
        )

        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["all_in_band"] == result["all_in_band"]

    def test_all_thresholds_json_serializable(self, mock_mode):
        """All thresholds should be JSON-serializable."""
        from backend.health.metrics_thresholds import get_all_thresholds

        thresholds = get_all_thresholds()
        json_str = json.dumps(thresholds)
        assert isinstance(json_str, str)


# ==============================================================================
# Test Class: Divergence Detection Log-Only
# ==============================================================================


class TestDivergenceDetectionLogOnly:
    """Tests for log-only divergence detection."""

    def test_divergence_detection_logs_detail(self, hybrid_mode, warn_boundary_metrics):
        """Divergence detection should include detailed info for logging."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(warn_boundary_metrics)

        assert result["diverges"] is True
        detail = result["divergence_detail"]

        # Should have both statuses for logging
        assert "mock_status" in detail
        assert "real_status" in detail
        assert detail["mock_status"] == "YELLOW"
        assert detail["real_status"] == "GREEN"

        # Should identify triggered thresholds
        assert "triggered_thresholds" in detail

    def test_no_divergence_no_detail(self, hybrid_mode, healthy_metrics):
        """No divergence should not include detail."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds(healthy_metrics)

        assert result["diverges"] is False
        assert "divergence_detail" not in result or result.get("divergence_detail") is None

    def test_divergence_with_multiple_metrics(self, hybrid_mode):
        """Multiple divergent metrics should all be captured."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        # All metrics at MOCK boundary
        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
        }

        result = evaluate_with_dual_thresholds(metrics)

        assert result["diverges"] is True
        triggered = result["divergence_detail"]["triggered_thresholds"]
        # Should have multiple thresholds
        assert len(triggered) >= 1


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_metrics(self, mock_mode):
        """Empty metrics should evaluate to GREEN."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds({})
        assert result["verdict"]["status"] == "GREEN"

    def test_partial_metrics(self, mock_mode):
        """Partial metrics should evaluate correctly."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds({"drift_magnitude": 0.50})
        assert result["verdict"]["status"] == "YELLOW"

    def test_exact_boundary_mock_warn(self, mock_mode):
        """Metric at exact MOCK warn boundary should be YELLOW."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds({"drift_magnitude": 0.30})
        assert result["verdict"]["status"] == "YELLOW"

    def test_just_below_boundary_mock_warn(self, mock_mode):
        """Metric just below MOCK warn boundary should be GREEN."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        result = evaluate_with_dual_thresholds({"drift_magnitude": 0.29})
        assert result["verdict"]["status"] == "GREEN"

    def test_extreme_values(self, mock_mode):
        """Extreme metric values should not crash."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        # Very high values
        result = evaluate_with_dual_thresholds({
            "drift_magnitude": 100.0,
            "success_rate": -100.0,
            "budget_utilization": 1000.0,
        })
        assert result["verdict"]["status"] == "RED"

    def test_float_precision(self, mock_mode):
        """Float precision should not cause issues."""
        from backend.health.metrics_thresholds import evaluate_with_dual_thresholds

        # Very close to boundary
        result = evaluate_with_dual_thresholds({"drift_magnitude": 0.299999999})
        assert result["verdict"]["status"] == "GREEN"

        result = evaluate_with_dual_thresholds({"drift_magnitude": 0.300000001})
        assert result["verdict"]["status"] == "YELLOW"
