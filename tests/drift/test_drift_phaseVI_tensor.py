"""
Tests for Phase VI: Drift Tensor & Poly-Cause Analysis.

This test suite verifies:
  - Drift tensor construction and normalization
  - Poly-cause analyzer detection logic
  - Director tile v2 with tensor norms and poly-cause status
  - Deterministic outputs and edge case handling

All tests are marked as unit tests and do not require external dependencies.
"""

import math
import pytest

from backend.metrics.drift_alignment import (
    build_drift_tensor,
    build_drift_poly_cause_analyzer,
    build_drift_director_tile_v2,
    DRIFT_TENSOR_SCHEMA_VERSION,
    DriftCell,
)


# ===========================================================================
# DRIFT TENSOR TESTS (8 tests)
# ===========================================================================

class TestDriftTensor:
    """Tests for build_drift_tensor function."""

    @pytest.mark.unit
    def test_build_drift_tensor_structure(self):
        """Drift tensor has correct structure."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                    "run2": {"drifty_slices": ["slice_a", "slice_b"]},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.3},
            "slice_b": {"drift_score": 0.5},
        }
        metric_axis = {
            "slice_a": {"drift_score": 0.2},
        }

        result = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
            metric_axis=metric_axis,
        )

        assert "tensor" in result
        assert "global_tensor_norm" in result
        assert "ranked_slices" in result
        assert "schema_version" in result
        assert result["schema_version"] == DRIFT_TENSOR_SCHEMA_VERSION
        assert isinstance(result["tensor"], dict)
        assert isinstance(result["global_tensor_norm"], float)
        assert isinstance(result["ranked_slices"], list)

    @pytest.mark.unit
    def test_drift_tensor_slice_scores(self):
        """Drift tensor correctly maps slices to axis scores."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                    "run2": {"drifty_slices": ["slice_a"]},
                    "run3": {"drifty_slices": []},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.4},
            "slice_b": {"drift_score": 0.6},
        }
        metric_axis = {
            "slice_a": {"drift_score": 0.3},
        }

        result = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
            metric_axis=metric_axis,
        )

        tensor = result["tensor"]
        assert "slice_a" in tensor
        assert "slice_b" in tensor

        # slice_a: drift = 2/3 = 0.667, budget = 0.4, metric = 0.3, semantic = 0.0
        assert abs(tensor["slice_a"]["drift"] - 2.0 / 3.0) < 0.01
        assert abs(tensor["slice_a"]["budget"] - 0.4) < 0.01
        assert abs(tensor["slice_a"]["metric"] - 0.3) < 0.01
        assert tensor["slice_a"]["semantic"] == 0.0

        # slice_b: drift = 0/3 = 0.0, budget = 0.6, metric = 0.0, semantic = 0.0
        assert tensor["slice_b"]["drift"] == 0.0
        assert abs(tensor["slice_b"]["budget"] - 0.6) < 0.01
        assert tensor["slice_b"]["metric"] == 0.0
        assert tensor["slice_b"]["semantic"] == 0.0

    @pytest.mark.unit
    def test_drift_tensor_global_norm(self):
        """Global tensor norm is computed correctly (L2 norm)."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.3},
        }
        metric_axis = {
            "slice_a": {"drift_score": 0.4},
        }

        result = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
            metric_axis=metric_axis,
        )

        # Expected: sqrt(1.0^2 + 0.3^2 + 0.4^2 + 0.0^2) = sqrt(1.0 + 0.09 + 0.16) = sqrt(1.25) ≈ 1.118
        # But drift is frequency, so for 1 run with drift: 1/1 = 1.0
        expected_norm = math.sqrt(1.0**2 + 0.3**2 + 0.4**2 + 0.0**2)
        assert abs(result["global_tensor_norm"] - expected_norm) < 0.01

    @pytest.mark.unit
    def test_drift_tensor_ranked_slices(self):
        """Slices are ranked by total drift magnitude."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a", "slice_b"]},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.1},  # Total: 1.0 + 0.1 = 1.1
            "slice_b": {"drift_score": 0.5},  # Total: 1.0 + 0.5 = 1.5
        }

        result = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
        )

        # slice_b should rank higher (total = 1.5) than slice_a (total = 1.1)
        assert result["ranked_slices"][0] == "slice_b"
        assert result["ranked_slices"][1] == "slice_a"

    @pytest.mark.unit
    def test_drift_tensor_empty_inputs(self):
        """Drift tensor handles empty inputs gracefully."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {},
            },
        }

        result = build_drift_tensor(drift_axis_view)

        assert result["tensor"] == {}
        assert result["global_tensor_norm"] == 0.0
        assert result["ranked_slices"] == []

    @pytest.mark.unit
    def test_drift_tensor_deterministic(self):
        """Drift tensor output is deterministic."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                    "run2": {"drifty_slices": ["slice_b"]},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.3},
            "slice_b": {"drift_score": 0.5},
        }

        result1 = build_drift_tensor(drift_axis_view, budget_axis=budget_axis)
        result2 = build_drift_tensor(drift_axis_view, budget_axis=budget_axis)

        assert result1 == result2
        assert result1["ranked_slices"] == result2["ranked_slices"]

    @pytest.mark.unit
    def test_drift_tensor_semantic_axis(self):
        """Semantic axis is included when provided."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                },
            },
        }
        semantic_axis = {
            "slice_a": {"drift_score": 0.7},
        }

        result = build_drift_tensor(
            drift_axis_view,
            semantic_axis=semantic_axis,
        )

        assert "slice_a" in result["tensor"]
        assert abs(result["tensor"]["slice_a"]["semantic"] - 0.7) < 0.01

    @pytest.mark.unit
    def test_drift_tensor_numeric_scores(self):
        """Drift tensor handles numeric scores (not just dicts)."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                },
            },
        }
        budget_axis = {
            "slice_a": 0.5,  # Direct numeric value
        }

        result = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
        )

        assert abs(result["tensor"]["slice_a"]["budget"] - 0.5) < 0.01


# ===========================================================================
# POLY-CAUSE ANALYZER TESTS (6 tests)
# ===========================================================================

class TestPolyCauseAnalyzer:
    """Tests for build_drift_poly_cause_analyzer function."""

    @pytest.mark.unit
    def test_poly_cause_analyzer_structure(self):
        """Poly-cause analyzer has correct structure."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": ["drift"],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert "poly_cause_detected" in result
        assert "cause_vectors" in result
        assert "risk_band" in result
        assert "notes" in result
        assert isinstance(result["poly_cause_detected"], bool)
        assert isinstance(result["cause_vectors"], list)
        assert result["risk_band"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(result["notes"], list)

    @pytest.mark.unit
    def test_poly_cause_detection_multi_axis(self):
        """Poly-cause detected when multiple axes show drift in same slice."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.4, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": ["drift", "budget", "metric"],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert result["poly_cause_detected"] is True
        assert len(result["cause_vectors"]) > 0
        cause_vec = result["cause_vectors"][0]
        assert cause_vec["slice"] == "slice_a"
        assert len(cause_vec["axes"]) >= 2  # drift, budget, metric

    @pytest.mark.unit
    def test_poly_cause_no_detection_single_axis(self):
        """Poly-cause not detected when only one axis shows drift."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.0, "metric": 0.0, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": ["drift"],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert result["poly_cause_detected"] is False
        assert len(result["cause_vectors"]) == 0

    @pytest.mark.unit
    def test_poly_cause_risk_band_high(self):
        """Risk band is HIGH when multiple high-risk axes."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": ["drift", "budget"],
            "axes_with_drift": ["drift", "budget"],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert result["risk_band"] == "HIGH"

    @pytest.mark.unit
    def test_poly_cause_risk_band_medium(self):
        """Risk band is MEDIUM when single high-risk or multiple axes with drift."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.5, "budget": 0.3, "metric": 0.0, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": ["drift"],
            "axes_with_drift": ["drift", "budget"],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert result["risk_band"] == "MEDIUM"

    @pytest.mark.unit
    def test_poly_cause_risk_band_low(self):
        """Risk band is LOW when no high-risk axes and no poly-cause."""
        drift_tensor = {
            "tensor": {
                "slice_a": {"drift": 0.0, "budget": 0.0, "metric": 0.0, "semantic": 0.0},
            },
        }
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }

        result = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert result["risk_band"] == "LOW"
        assert result["poly_cause_detected"] is False


# ===========================================================================
# DIRECTOR TILE V2 TESTS (7 tests)
# ===========================================================================

class TestDirectorTileV2:
    """Tests for build_drift_director_tile_v2 function."""

    @pytest.mark.unit
    def test_director_tile_v2_structure(self):
        """Director tile v2 has correct structure."""
        global_health_drift = {"status": "OK"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }
        drift_tensor = {
            "global_tensor_norm": 0.5,
        }
        poly_cause_analysis = {
            "poly_cause_detected": False,
            "risk_band": "LOW",
        }
        uplift_envelope = {
            "status": "OK",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert "status_light" in result
        assert "tensor_norm" in result
        assert "poly_cause_status" in result
        assert "risk_band" in result
        assert "uplift_envelope_impact" in result
        assert "headline" in result
        assert result["status_light"] in ("GREEN", "YELLOW", "RED")
        assert result["poly_cause_status"] in ("DETECTED", "NONE")

    @pytest.mark.unit
    def test_director_tile_v2_green_status(self):
        """Status light is GREEN when all systems OK."""
        global_health_drift = {"status": "OK"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }
        drift_tensor = {"global_tensor_norm": 0.1}
        poly_cause_analysis = {
            "poly_cause_detected": False,
            "risk_band": "LOW",
        }
        uplift_envelope = {
            "status": "OK",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert result["status_light"] == "GREEN"

    @pytest.mark.unit
    def test_director_tile_v2_yellow_status(self):
        """Status light is YELLOW when moderate issues detected."""
        global_health_drift = {"status": "WARN"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "WARN"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": ["drift"],
        }
        drift_tensor = {"global_tensor_norm": 0.3}
        poly_cause_analysis = {
            "poly_cause_detected": True,
            "risk_band": "MEDIUM",
        }
        uplift_envelope = {
            "status": "ATTENTION",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert result["status_light"] == "YELLOW"

    @pytest.mark.unit
    def test_director_tile_v2_red_status(self):
        """Status light is RED when critical issues detected."""
        global_health_drift = {"status": "HOT"}
        trend_history = {"trend_status": "DEGRADING"}
        promotion_eval = {"status": "BLOCK"}
        multi_axis_view = {
            "high_risk_axes": ["drift", "budget"],
            "axes_with_drift": ["drift", "budget"],
        }
        drift_tensor = {"global_tensor_norm": 1.5}
        poly_cause_analysis = {
            "poly_cause_detected": True,
            "risk_band": "HIGH",
        }
        uplift_envelope = {
            "status": "BLOCK",
            "uplift_safe_under_drift": False,
            "blocking_axes": ["drift", "budget"],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert result["status_light"] == "RED"

    @pytest.mark.unit
    def test_director_tile_v2_tensor_norm(self):
        """Tensor norm is correctly extracted and rounded."""
        global_health_drift = {"status": "OK"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }
        drift_tensor = {"global_tensor_norm": 0.123456789}
        poly_cause_analysis = {
            "poly_cause_detected": False,
            "risk_band": "LOW",
        }
        uplift_envelope = {
            "status": "OK",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert result["tensor_norm"] == round(0.123456789, 6)

    @pytest.mark.unit
    def test_director_tile_v2_poly_cause_status(self):
        """Poly-cause status is correctly extracted."""
        global_health_drift = {"status": "OK"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }
        drift_tensor = {"global_tensor_norm": 0.1}
        poly_cause_analysis = {
            "poly_cause_detected": True,
            "risk_band": "MEDIUM",
        }
        uplift_envelope = {
            "status": "OK",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert result["poly_cause_status"] == "DETECTED"

    @pytest.mark.unit
    def test_director_tile_v2_headline_generation(self):
        """Headline includes relevant information."""
        global_health_drift = {"status": "OK"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "OK"}
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }
        drift_tensor = {"global_tensor_norm": 0.8}
        poly_cause_analysis = {
            "poly_cause_detected": True,
            "risk_band": "MEDIUM",
        }
        uplift_envelope = {
            "status": "ATTENTION",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        result = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        assert "tensor norm" in result["headline"].lower() or "poly-cause" in result["headline"].lower()
        assert result["headline"].endswith(".")


# ===========================================================================
# INTEGRATION TESTS (3 tests)
# ===========================================================================

class TestPhaseVIIntegration:
    """Integration tests for Phase VI components."""

    @pytest.mark.unit
    def test_full_workflow(self):
        """Full workflow: tensor → poly-cause → director tile."""
        # Build drift tensor
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                    "run2": {"drifty_slices": ["slice_a", "slice_b"]},
                },
            },
        }
        budget_axis = {
            "slice_a": {"drift_score": 0.4},
            "slice_b": {"drift_score": 0.3},
        }
        metric_axis = {
            "slice_a": {"drift_score": 0.5},
        }

        drift_tensor = build_drift_tensor(
            drift_axis_view,
            budget_axis=budget_axis,
            metric_axis=metric_axis,
        )

        # Build multi-axis view
        multi_axis_view = {
            "high_risk_axes": ["drift"],
            "axes_with_drift": ["drift", "budget", "metric"],
        }

        # Analyze poly-cause
        poly_cause_analysis = build_drift_poly_cause_analyzer(
            drift_tensor,
            multi_axis_view,
        )

        # Build director tile
        global_health_drift = {"status": "WARN"}
        trend_history = {"trend_status": "STABLE"}
        promotion_eval = {"status": "WARN"}
        uplift_envelope = {
            "status": "ATTENTION",
            "uplift_safe_under_drift": True,
            "blocking_axes": [],
        }

        director_tile = build_drift_director_tile_v2(
            global_health_drift,
            trend_history,
            promotion_eval,
            multi_axis_view,
            drift_tensor,
            poly_cause_analysis,
            uplift_envelope,
        )

        # Verify integration
        assert director_tile["tensor_norm"] == drift_tensor["global_tensor_norm"]
        assert director_tile["poly_cause_status"] == (
            "DETECTED" if poly_cause_analysis["poly_cause_detected"] else "NONE"
        )
        assert director_tile["risk_band"] == poly_cause_analysis["risk_band"]

    @pytest.mark.unit
    def test_deterministic_full_workflow(self):
        """Full workflow produces deterministic results."""
        drift_axis_view = {
            "drift_trend_history": {
                "runs": {
                    "run1": {"drifty_slices": ["slice_a"]},
                },
            },
        }
        budget_axis = {"slice_a": {"drift_score": 0.3}}

        tensor1 = build_drift_tensor(drift_axis_view, budget_axis=budget_axis)
        tensor2 = build_drift_tensor(drift_axis_view, budget_axis=budget_axis)

        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": ["drift", "budget"],
        }

        poly1 = build_drift_poly_cause_analyzer(tensor1, multi_axis_view)
        poly2 = build_drift_poly_cause_analyzer(tensor2, multi_axis_view)

        assert poly1 == poly2

    @pytest.mark.unit
    def test_edge_case_empty_tensor(self):
        """Workflow handles empty tensor gracefully."""
        drift_tensor = {
            "tensor": {},
            "global_tensor_norm": 0.0,
            "ranked_slices": [],
        }
        multi_axis_view = {
            "high_risk_axes": [],
            "axes_with_drift": [],
        }

        poly_cause = build_drift_poly_cause_analyzer(drift_tensor, multi_axis_view)

        assert poly_cause["poly_cause_detected"] is False
        assert poly_cause["risk_band"] == "LOW"

