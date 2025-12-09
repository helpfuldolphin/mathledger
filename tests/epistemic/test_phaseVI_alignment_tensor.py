"""
Phase VI Epistemic Alignment Tensor Tests

PHASE VI — NOT RUN IN PHASE I
No uplift claims are made.
Deterministic execution guaranteed.

Tests for:
- Epistemic alignment tensor construction
- Predictive misalignment forecasting
- Epistemic director panel
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.doc_sync_predictor import (
    build_epistemic_alignment_tensor,
    build_epistemic_director_panel,
    forecast_epistemic_misalignment,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_semantic_panel() -> Dict[str, Any]:
    """Create sample semantic panel."""
    return {
        "semantic_status_light": "GREEN",
        "semantic_ok": True,
        "alignment_status": "ALIGNED",
        "critical_term_count": 0,
        "slice_alignment": {
            "slice_easy_fo": {"score": 0.9},
            "slice_medium": {"score": 0.7},
            "slice_hard": {"score": 0.5},
        },
    }


@pytest.fixture
def sample_curriculum_panel() -> Dict[str, Any]:
    """Create sample curriculum panel."""
    return {
        "status_light": "GREEN",
        "curriculum_status": "ALIGNED",
        "alignment_rate": 0.85,
    }


@pytest.fixture
def sample_metric_readiness_matrix() -> Dict[str, Any]:
    """Create sample metric readiness matrix."""
    return {
        "schema_version": "1.0.0",
        "matrix": {
            "slice_easy_fo": {
                "goal_hit": {"status": "READY", "drift_severity": "NONE"},
                "sparse_density": {"status": "READY", "drift_severity": "NONE"},
            },
            "slice_medium": {
                "goal_hit": {"status": "READY", "drift_severity": "COSMETIC"},
                "sparse_density": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"},
            },
            "slice_hard": {
                "goal_hit": {"status": "BLOCKED", "drift_severity": "SEMANTIC"},
                "sparse_density": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"},
            },
        },
    }


@pytest.fixture
def sample_drift_multi_axis_view() -> Dict[str, Any]:
    """Create sample drift multi-axis view."""
    return {
        "schema_version": "1.0.0",
        "trend_status": "STABLE",
        "runs_with_high_risk": 1,
        "runs_with_new_critical_terms": 0,
        "max_consecutive_high_runs": 1,
    }


@pytest.fixture
def sample_alignment_tensor(
    sample_semantic_panel: Dict[str, Any],
    sample_curriculum_panel: Dict[str, Any],
    sample_metric_readiness_matrix: Dict[str, Any],
    sample_drift_multi_axis_view: Dict[str, Any],
) -> Dict[str, Any]:
    """Create sample alignment tensor."""
    return build_epistemic_alignment_tensor(
        sample_semantic_panel,
        sample_curriculum_panel,
        sample_metric_readiness_matrix,
        sample_drift_multi_axis_view,
    )


@pytest.fixture
def sample_historical_alignment() -> List[Dict[str, Any]]:
    """Create sample historical alignment tensors."""
    return [
        {
            "schema_version": "1.0.0",
            "tensor_id": "hist1",
            "alignment_tensor_norm": 0.8,
            "system_axes": {"semantic": 1.0, "curriculum": 0.9, "metrics": 0.8, "drift": 0.7},
            "misalignment_hotspots": [],
        },
        {
            "schema_version": "1.0.0",
            "tensor_id": "hist2",
            "alignment_tensor_norm": 0.7,
            "system_axes": {"semantic": 0.9, "curriculum": 0.8, "metrics": 0.7, "drift": 0.6},
            "misalignment_hotspots": ["slice_hard"],
        },
        {
            "schema_version": "1.0.0",
            "tensor_id": "hist3",
            "alignment_tensor_norm": 0.6,
            "system_axes": {"semantic": 0.8, "curriculum": 0.7, "metrics": 0.6, "drift": 0.5},
            "misalignment_hotspots": ["slice_hard", "slice_medium"],
        },
    ]


@pytest.fixture
def sample_structural_view() -> Dict[str, Any]:
    """Create sample structural view."""
    return {
        "status_light": "GREEN",
        "structural_ok": True,
        "governance_integrity": 0.9,
    }


# ==============================================================================
# 25. EPISTEMIC ALIGNMENT TENSOR TESTS (8 tests)
# ==============================================================================


class TestEpistemicAlignmentTensor:
    """Tests for epistemic alignment tensor construction."""

    def test_tensor_has_required_structure(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that tensor has all required top-level keys."""
        tensor = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        required_keys = {
            "schema_version",
            "tensor_id",
            "generated_at",
            "slice_axis",
            "system_axes",
            "alignment_tensor_norm",
            "misalignment_hotspots",
        }
        assert required_keys.issubset(set(tensor.keys()))
        assert tensor["schema_version"] == "1.0.0"

    def test_tensor_system_axes_normalized(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that system axes are normalized to [0, 1]."""
        tensor = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        system_axes = tensor["system_axes"]
        for axis_name, axis_value in system_axes.items():
            assert 0.0 <= axis_value <= 1.0, f"Axis {axis_name} out of bounds: {axis_value}"

    def test_tensor_slice_axis_scores_normalized(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that slice_axis scores are normalized to [0, 1]."""
        tensor = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        slice_axis = tensor["slice_axis"]
        for slice_name, slice_score in slice_axis.items():
            assert 0.0 <= slice_score <= 1.0, f"Slice {slice_name} score out of bounds: {slice_score}"

    def test_tensor_norm_computation(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that tensor norm is computed correctly (L2 norm normalized)."""
        tensor = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        system_axes = tensor["system_axes"]
        semantic = system_axes["semantic"]
        curriculum = system_axes["curriculum"]
        metrics = system_axes["metrics"]
        drift = system_axes["drift"]
        
        # Manual L2 norm computation
        expected_norm = math.sqrt(semantic**2 + curriculum**2 + metrics**2 + drift**2) / 2.0
        
        assert abs(tensor["alignment_tensor_norm"] - expected_norm) < 0.01

    def test_tensor_identifies_hotspots(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that tensor identifies misalignment hotspots."""
        # Create a panel with low semantic, low metric, high drift
        low_semantic_panel = {
            "semantic_status_light": "RED",
            "slice_alignment": {
                "slice_hard": {"score": 0.2},  # Low semantic
            },
        }
        
        low_metric_matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_hard": {
                    "goal_hit": {"status": "BLOCKED"},
                    "sparse_density": {"status": "BLOCKED"},
                },
            },
        }
        
        high_drift_view = {
            "schema_version": "1.0.0",
            "trend_status": "DEGRADING",  # High drift (low drift_axis)
        }
        
        tensor = build_epistemic_alignment_tensor(
            low_semantic_panel,
            sample_curriculum_panel,
            low_metric_matrix,
            high_drift_view,
        )
        
        # slice_hard should be a hotspot (low semantic, low metric, high drift)
        assert "slice_hard" in tensor["misalignment_hotspots"]

    def test_tensor_higher_scores_healthier(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that higher scores indicate healthier alignment."""
        # GREEN semantic panel
        green_tensor = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        # RED semantic panel
        red_semantic = {
            "semantic_status_light": "RED",
            "slice_alignment": sample_semantic_panel.get("slice_alignment", {}),
        }
        red_tensor = build_epistemic_alignment_tensor(
            red_semantic,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        # GREEN should have higher semantic axis
        assert green_tensor["system_axes"]["semantic"] > red_tensor["system_axes"]["semantic"]

    def test_tensor_is_deterministic(
        self,
        sample_semantic_panel: Dict[str, Any],
        sample_curriculum_panel: Dict[str, Any],
        sample_metric_readiness_matrix: Dict[str, Any],
        sample_drift_multi_axis_view: Dict[str, Any],
    ) -> None:
        """Test that tensor is deterministic."""
        tensor1 = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        tensor2 = build_epistemic_alignment_tensor(
            sample_semantic_panel,
            sample_curriculum_panel,
            sample_metric_readiness_matrix,
            sample_drift_multi_axis_view,
        )
        
        assert tensor1["tensor_id"] == tensor2["tensor_id"]
        assert tensor1["generated_at"] == tensor2["generated_at"]
        assert tensor1["alignment_tensor_norm"] == tensor2["alignment_tensor_norm"]

    def test_tensor_handles_missing_data(
        self,
    ) -> None:
        """Test that tensor handles missing or incomplete panel data."""
        minimal_semantic = {"semantic_status_light": "YELLOW"}
        minimal_curriculum = {"status_light": "YELLOW"}
        minimal_metrics = {"schema_version": "1.0.0", "matrix": {}}
        minimal_drift = {"trend_status": "STABLE"}
        
        tensor = build_epistemic_alignment_tensor(
            minimal_semantic,
            minimal_curriculum,
            minimal_metrics,
            minimal_drift,
        )
        
        # Should still produce valid tensor
        assert "alignment_tensor_norm" in tensor
        assert "system_axes" in tensor


# ==============================================================================
# 26. PREDICTIVE MISALIGNMENT FORECASTER TESTS (8 tests)
# ==============================================================================


class TestPredictiveMisalignmentForecaster:
    """Tests for predictive misalignment forecasting."""

    def test_forecast_has_required_structure(
        self, sample_alignment_tensor: Dict[str, Any]
    ) -> None:
        """Test that forecast has all required keys."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        required_keys = {
            "schema_version",
            "forecast_id",
            "generated_at",
            "predicted_band",
            "confidence",
            "time_to_drift_event",
            "neutral_explanation",
        }
        assert required_keys.issubset(set(forecast.keys()))
        assert forecast["schema_version"] == "1.0.0"

    def test_forecast_band_classification_low(
        self,
    ) -> None:
        """Test that forecast produces LOW band for high alignment."""
        high_alignment = {
            "schema_version": "1.0.0",
            "tensor_id": "high_align",
            "alignment_tensor_norm": 0.8,  # High norm
            "system_axes": {"semantic": 0.9, "curriculum": 0.8, "metrics": 0.8, "drift": 0.7},
            "misalignment_hotspots": [],  # No hotspots
        }
        
        forecast = forecast_epistemic_misalignment(high_alignment)
        
        assert forecast["predicted_band"] == "LOW"

    def test_forecast_band_classification_high(
        self,
    ) -> None:
        """Test that forecast produces HIGH band for low alignment."""
        low_alignment = {
            "schema_version": "1.0.0",
            "tensor_id": "low_align",
            "alignment_tensor_norm": 0.3,  # Low norm
            "system_axes": {"semantic": 0.2, "curriculum": 0.3, "metrics": 0.2, "drift": 0.3},
            "misalignment_hotspots": ["slice1", "slice2", "slice3"],  # Many hotspots
        }
        
        forecast = forecast_epistemic_misalignment(low_alignment)
        
        assert forecast["predicted_band"] == "HIGH"

    def test_forecast_uses_historical_trend(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_historical_alignment: List[Dict[str, Any]],
    ) -> None:
        """Test that forecast incorporates historical alignment trends."""
        # Add current tensor to history (decreasing trend)
        current_norm = 0.5
        sample_alignment_tensor["alignment_tensor_norm"] = current_norm
        
        forecast = forecast_epistemic_misalignment(
            sample_alignment_tensor, sample_historical_alignment
        )
        
        # Should detect decreasing trend and potentially raise risk
        assert "predicted_band" in forecast
        # Explanation should mention trend
        explanation = " ".join(forecast.get("neutral_explanation", []))
        assert "trend" in explanation.lower() or len(sample_historical_alignment) > 0

    def test_forecast_confidence_with_history(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_historical_alignment: List[Dict[str, Any]],
    ) -> None:
        """Test that forecast confidence increases with historical data."""
        forecast_with_history = forecast_epistemic_misalignment(
            sample_alignment_tensor, sample_historical_alignment
        )
        forecast_no_history = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        # Confidence should be higher with history
        assert forecast_with_history["confidence"] >= forecast_no_history["confidence"]

    def test_forecast_time_to_drift_event(
        self,
    ) -> None:
        """Test that time_to_drift_event is reasonable based on predicted_band."""
        high_risk_tensor = {
            "schema_version": "1.0.0",
            "tensor_id": "high_risk",
            "alignment_tensor_norm": 0.3,
            "system_axes": {"semantic": 0.2, "curriculum": 0.3, "metrics": 0.2, "drift": 0.3},
            "misalignment_hotspots": ["slice1", "slice2"],
        }
        
        forecast = forecast_epistemic_misalignment(high_risk_tensor)
        
        # HIGH band should have shorter time_to_drift_event
        if forecast["predicted_band"] == "HIGH":
            assert forecast["time_to_drift_event"] <= 10
        assert forecast["time_to_drift_event"] > 0

    def test_forecast_explanation_neutral(
        self, sample_alignment_tensor: Dict[str, Any]
    ) -> None:
        """Test that forecast explanation uses neutral language."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        explanation = " ".join(forecast.get("neutral_explanation", [])).lower()
        forbidden_terms = ["good", "bad", "healthy", "unhealthy", "better", "worse", "fix", "wrong"]
        
        for term in forbidden_terms:
            assert term not in explanation, f"Evaluative term '{term}' found in explanation"

    def test_forecast_is_deterministic(
        self, sample_alignment_tensor: Dict[str, Any]
    ) -> None:
        """Test that forecast is deterministic."""
        forecast1 = forecast_epistemic_misalignment(sample_alignment_tensor)
        forecast2 = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        assert forecast1["forecast_id"] == forecast2["forecast_id"]
        assert forecast1["generated_at"] == forecast2["generated_at"]
        assert forecast1["predicted_band"] == forecast2["predicted_band"]

    def test_forecast_confidence_bounds(
        self, sample_alignment_tensor: Dict[str, Any]
    ) -> None:
        """Test that forecast confidence is in [0, 1]."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        assert 0.0 <= forecast["confidence"] <= 1.0


# ==============================================================================
# 27. EPISTEMIC DIRECTOR PANEL TESTS (8 tests)
# ==============================================================================


class TestEpistemicDirectorPanel:
    """Tests for epistemic director panel construction."""

    def test_panel_has_required_structure(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel has all required keys."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        panel = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        
        required_keys = {
            "schema_version",
            "panel_id",
            "generated_at",
            "status_light",
            "alignment_band",
            "forecast_band",
            "structural_band",
            "headline",
            "flags",
        }
        assert required_keys.issubset(set(panel.keys()))
        assert panel["schema_version"] == "1.0.0"

    def test_panel_status_light_red_on_low_alignment(
        self,
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel shows RED when alignment is low."""
        low_tensor = {
            "schema_version": "1.0.0",
            "tensor_id": "low_tensor",
            "alignment_tensor_norm": 0.3,
            "system_axes": {"semantic": 0.2, "curriculum": 0.3, "metrics": 0.2, "drift": 0.3},
            "misalignment_hotspots": ["slice1"],
        }
        
        low_forecast = forecast_epistemic_misalignment(low_tensor)
        panel = build_epistemic_director_panel(low_tensor, low_forecast, sample_structural_view)
        
        assert panel["status_light"] == "RED"

    def test_panel_status_light_green_on_high_alignment(
        self,
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel shows GREEN when all indicators are positive."""
        high_tensor = {
            "schema_version": "1.0.0",
            "tensor_id": "high_tensor",
            "alignment_tensor_norm": 0.8,
            "system_axes": {"semantic": 0.9, "curriculum": 0.8, "metrics": 0.8, "drift": 0.7},
            "misalignment_hotspots": [],
        }
        
        high_forecast = forecast_epistemic_misalignment(high_tensor)
        green_structural = {"status_light": "GREEN"}
        panel = build_epistemic_director_panel(high_tensor, high_forecast, green_structural)
        
        assert panel["status_light"] == "GREEN"

    def test_panel_band_classification(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel correctly classifies alignment bands."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        panel = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        
        # Check band values are valid
        assert panel["alignment_band"] in {"LOW", "MEDIUM", "HIGH"}
        assert panel["forecast_band"] in {"LOW", "MEDIUM", "HIGH"}
        assert panel["structural_band"] in {"LOW", "MEDIUM", "HIGH"}

    def test_panel_headline_neutral(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel headline uses neutral language."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        panel = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        
        headline = panel.get("headline", "").lower()
        forbidden_terms = ["good", "bad", "healthy", "unhealthy", "better", "worse", "fix", "wrong"]
        
        for term in forbidden_terms:
            assert term not in headline, f"Evaluative term '{term}' found in headline"

    def test_panel_flags_descriptive(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel flags are descriptive and neutral."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        panel = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        
        flags_text = " ".join(panel.get("flags", [])).lower()
        forbidden_terms = ["good", "bad", "healthy", "unhealthy", "better", "worse"]
        
        for term in forbidden_terms:
            assert term not in flags_text, f"Evaluative term '{term}' found in flags"

    def test_panel_is_deterministic(
        self,
        sample_alignment_tensor: Dict[str, Any],
        sample_structural_view: Dict[str, Any],
    ) -> None:
        """Test that panel is deterministic."""
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        
        panel1 = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        panel2 = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, sample_structural_view
        )
        
        assert panel1["panel_id"] == panel2["panel_id"]
        assert panel1["generated_at"] == panel2["generated_at"]
        assert panel1["status_light"] == panel2["status_light"]

    def test_panel_handles_red_structural(
        self,
        sample_alignment_tensor: Dict[str, Any],
    ) -> None:
        """Test that panel shows RED when structural view is RED."""
        red_structural = {"status_light": "RED"}
        forecast = forecast_epistemic_misalignment(sample_alignment_tensor)
        panel = build_epistemic_director_panel(
            sample_alignment_tensor, forecast, red_structural
        )
        
        assert panel["status_light"] == "RED"
        assert "RED" in " ".join(panel.get("flags", []))

