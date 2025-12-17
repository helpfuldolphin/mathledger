"""
Tests for budget-aware modulation in calibration experiments.
"""

import pytest
from experiments.uplift_council import compute_budget_modulation_for_calibration_window
from experiments.budget_calibration_modulation import (
    annotate_calibration_windows_with_budget_modulation,
    demonstrate_budget_confounding_example,
)


class TestBudgetModulationForCalibrationWindow:
    """Tests for compute_budget_modulation_for_calibration_window."""
    
    def test_no_budget_data_returns_defaults(self):
        """Test that missing budget data returns safe defaults."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
            "divergence_rate": 0.12,
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=None,
            slice_name=None,
        )
        
        assert modulation["budget_confounded"] is False
        assert modulation["effective_lr_adjustment"] == 1.0
        assert modulation["drift_classification"] == "NONE"
        assert modulation["budget_health_during_window"] == "UNKNOWN"
    
    def test_safe_budget_no_confounding(self):
        """Test that SAFE budget results in no confounding."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
        }
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "SAFE",
                    "frequently_starved": False,
                    "budget_exhausted_pct": 0.0,
                }
            ]
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert modulation["budget_confounded"] is False
        assert modulation["effective_lr_adjustment"] == 1.0
        assert modulation["drift_classification"] == "NONE"
        assert modulation["budget_health_during_window"] == "SAFE"
    
    def test_starved_budget_transient_confounding(self):
        """Test that STARVED budget (not frequently) results in TRANSIENT confounding."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
        }
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": False,  # Transient
                    "budget_exhausted_pct": 15.0,
                }
            ]
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert modulation["budget_confounded"] is True
        assert modulation["effective_lr_adjustment"] == pytest.approx(0.85, abs=0.01)  # 1.0 - 0.15
        assert modulation["drift_classification"] == "TRANSIENT"
        assert modulation["budget_health_during_window"] == "STARVED"
    
    def test_starved_budget_persistent_confounding(self):
        """Test that frequently STARVED budget results in PERSISTENT confounding."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
        }
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": True,  # Persistent
                    "budget_exhausted_pct": 25.0,
                }
            ]
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert modulation["budget_confounded"] is True
        assert modulation["effective_lr_adjustment"] == pytest.approx(0.75, abs=0.01)  # 1.0 - 0.25
        assert modulation["drift_classification"] == "PERSISTENT"
        assert modulation["budget_health_during_window"] == "STARVED"
    
    def test_tight_budget_no_confounding(self):
        """Test that TIGHT budget (not STARVED) does not confound."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
        }
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "TIGHT",
                    "frequently_starved": False,
                    "budget_exhausted_pct": 5.0,
                }
            ]
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert modulation["budget_confounded"] is False
        assert modulation["effective_lr_adjustment"] == pytest.approx(0.95, abs=0.01)  # 1.0 - 0.05
        assert modulation["drift_classification"] == "NONE"
        assert modulation["budget_health_during_window"] == "TIGHT"
    
    def test_full_budget_exhaustion_zero_lr_adjustment(self):
        """Test that 100% budget exhaustion results in zero LR adjustment."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
        }
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": True,
                    "budget_exhausted_pct": 100.0,
                }
            ]
        }
        
        modulation = compute_budget_modulation_for_calibration_window(
            window=window,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert modulation["budget_confounded"] is True
        assert modulation["effective_lr_adjustment"] == 0.0
        assert modulation["drift_classification"] == "PERSISTENT"


class TestAnnotateCalibrationWindows:
    """Tests for annotate_calibration_windows_with_budget_modulation."""
    
    def test_annotate_single_window(self):
        """Test annotating a single calibration window."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "divergence_rate": 0.12,
            }
        ]
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": False,
                    "budget_exhausted_pct": 15.0,
                }
            ]
        }
        
        annotated = annotate_calibration_windows_with_budget_modulation(
            calibration_windows=windows,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert len(annotated) == 1
        assert annotated[0]["budget_confounded"] is True
        assert annotated[0]["effective_lr_adjustment"] == pytest.approx(0.85, abs=0.01)
        assert annotated[0]["drift_classification"] == "TRANSIENT"
        assert annotated[0]["budget_health_during_window"] == "STARVED"
        # Original fields preserved
        assert annotated[0]["start_cycle"] == 0
        assert annotated[0]["end_cycle"] == 50
        assert annotated[0]["divergence_rate"] == 0.12
    
    def test_annotate_multiple_windows(self):
        """Test annotating multiple calibration windows."""
        windows = [
            {"start_cycle": 0, "end_cycle": 50, "divergence_rate": 0.12},
            {"start_cycle": 50, "end_cycle": 100, "divergence_rate": 0.08},
        ]
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "SAFE",
                    "frequently_starved": False,
                    "budget_exhausted_pct": 0.0,
                }
            ]
        }
        
        annotated = annotate_calibration_windows_with_budget_modulation(
            calibration_windows=windows,
            budget_cross_view=budget_cross_view,
            slice_name="slice_uplift_goal",
        )
        
        assert len(annotated) == 2
        assert all(w["budget_confounded"] is False for w in annotated)
        assert all(w["effective_lr_adjustment"] == 1.0 for w in annotated)
    
    def test_annotate_without_slice_name(self):
        """Test that windows without slice name are preserved but not annotated."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "divergence_rate": 0.12,
            }
        ]
        
        budget_cross_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": False,
                    "budget_exhausted_pct": 15.0,
                }
            ]
        }
        
        annotated = annotate_calibration_windows_with_budget_modulation(
            calibration_windows=windows,
            budget_cross_view=budget_cross_view,
            slice_name=None,  # No slice name provided
        )
        
        assert len(annotated) == 1
        # Window preserved but not annotated
        assert "budget_confounded" not in annotated[0]
        assert annotated[0]["start_cycle"] == 0


class TestBudgetConfoundingExample:
    """Tests for demonstrate_budget_confounding_example."""
    
    def test_example_structure(self):
        """Test that the worked example has correct structure."""
        example = demonstrate_budget_confounding_example()
        
        assert "original_calibration_window" in example
        assert "budget_status" in example
        assert "budget_modulation" in example
        assert "divergence_interpretation" in example
        assert "calibration_recommendation" in example
    
    def test_example_shows_confounding(self):
        """Test that the example correctly identifies budget confounding."""
        example = demonstrate_budget_confounding_example()
        
        assert example["budget_modulation"]["budget_confounded"] is True
        assert example["budget_modulation"]["drift_classification"] == "TRANSIENT"
        assert example["calibration_recommendation"]["exclude_from_calibration"] is True
        assert example["calibration_recommendation"]["reason"] == "budget_confounded"
    
    def test_example_shows_adjusted_severity(self):
        """Test that the example shows severity adjustment."""
        example = demonstrate_budget_confounding_example()
        
        interpretation = example["divergence_interpretation"]
        assert interpretation["original_severity"] == "WARN"
        assert interpretation["adjusted_severity"] == "INFO"
        assert interpretation["original_delta_p"] == 0.12
        assert interpretation["adjusted_delta_p"] == pytest.approx(0.048, abs=0.01)  # 0.12 * 0.4



