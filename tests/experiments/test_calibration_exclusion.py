"""
Tests for calibration exclusion recommendations based on cross-signal checks.
"""

import pytest
import json
from experiments.budget_calibration_modulation import (
    compute_calibration_exclusion_recommendation,
    annotate_calibration_windows_with_exclusion_recommendations,
)
from experiments.uplift_council import annotate_calibration_window_with_exclusion


class TestCalibrationExclusionRecommendation:
    """Tests for compute_calibration_exclusion_recommendation."""
    
    def test_no_budget_confounding_no_exclusion(self):
        """Test that non-confounded budget does not recommend exclusion."""
        budget_modulation = {
            "budget_confounded": False,
            "drift_classification": "NONE",
        }
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is False
        assert exclusion["exclusion_reason"] is None
        assert exclusion["cross_signal_checks"]["budget_confounded"] is False
    
    def test_transient_budget_confounding_recommends_exclusion(self):
        """Test that transient budget confounding recommends exclusion when signals OK."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is True
        assert exclusion["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert exclusion["cross_signal_checks"]["budget_confounded"] is True
        assert exclusion["cross_signal_checks"]["prng_not_volatile"] is True
        assert exclusion["cross_signal_checks"]["topology_stable"] is True
    
    def test_persistent_budget_confounding_recommends_exclusion(self):
        """Test that persistent budget confounding recommends exclusion when signals OK."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "PERSISTENT",
        }
        
        prng_signal = {"drift_status": "DRIFTING"}  # Not volatile
        topology_signal = {"pressure_band": "MEDIUM"}  # Not HIGH
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is True
        assert exclusion["exclusion_reason"] == "BUDGET_CONFOUNDED_PERSISTENT"
    
    def test_budget_confounded_but_prng_volatile_no_exclusion(self):
        """Test that volatile PRNG prevents exclusion recommendation."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "VOLATILE"}  # Volatile!
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is False
        assert exclusion["exclusion_reason"] is None
        assert exclusion["cross_signal_checks"]["prng_not_volatile"] is False
    
    def test_budget_confounded_but_topology_high_no_exclusion(self):
        """Test that high topology pressure prevents exclusion recommendation."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "HIGH"}  # Not stable!
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is False
        assert exclusion["exclusion_reason"] is None
        assert exclusion["cross_signal_checks"]["topology_stable"] is False
    
    def test_missing_signals_default_to_safe(self):
        """Test that missing PRNG/topology signals default to safe (allow exclusion)."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        # No PRNG or topology signals provided
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        # Should recommend exclusion (missing signals default to safe)
        assert exclusion["calibration_exclusion_recommended"] is True
        assert exclusion["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert exclusion["cross_signal_checks"]["prng_not_volatile"] is True
        assert exclusion["cross_signal_checks"]["topology_stable"] is True
    
    def test_all_signals_volatile_no_exclusion(self):
        """Test that all signals being volatile prevents exclusion."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "PERSISTENT",
        }
        
        prng_signal = {"drift_status": "VOLATILE"}
        topology_signal = {"pressure_band": "HIGH"}
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert exclusion["calibration_exclusion_recommended"] is False
        assert exclusion["exclusion_reason"] is None


class TestAnnotateCalibrationWindowsWithExclusion:
    """Tests for annotate_calibration_windows_with_exclusion_recommendations."""
    
    def test_annotate_single_window_with_budget_modulation(self):
        """Test annotating a window that already has budget modulation."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "budget_confounded": True,
                "drift_classification": "TRANSIENT",
            }
        ]
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert len(annotated) == 1
        assert annotated[0]["calibration_exclusion_recommended"] is True
        assert annotated[0]["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert "cross_signal_checks" in annotated[0]
    
    def test_annotate_window_without_budget_modulation(self):
        """Test annotating a window without budget modulation."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "divergence_rate": 0.12,
            }
        ]
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
        )
        
        assert len(annotated) == 1
        assert annotated[0]["calibration_exclusion_recommended"] is False
        assert annotated[0]["exclusion_reason"] is None
        assert annotated[0]["cross_signal_checks"]["budget_confounded"] is False
    
    def test_annotate_multiple_windows(self):
        """Test annotating multiple windows with different conditions."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "budget_confounded": True,
                "drift_classification": "TRANSIENT",
            },
            {
                "start_cycle": 50,
                "end_cycle": 100,
                "budget_confounded": False,
            },
            {
                "start_cycle": 100,
                "end_cycle": 150,
                "budget_confounded": True,
                "drift_classification": "PERSISTENT",
            },
        ]
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert len(annotated) == 3
        assert annotated[0]["calibration_exclusion_recommended"] is True
        assert annotated[0]["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert annotated[1]["calibration_exclusion_recommended"] is False
        assert annotated[2]["calibration_exclusion_recommended"] is True
        assert annotated[2]["exclusion_reason"] == "BUDGET_CONFOUNDED_PERSISTENT"


class TestCalibrationExclusionTrace:
    """Tests for exclusion trace auditability."""
    
    def test_exclusion_trace_structure(self):
        """Test that exclusion trace has correct structure."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert "exclusion_trace" in exclusion
        trace = exclusion["exclusion_trace"]
        
        assert "checks" in trace
        assert "decision" in trace
        assert "reason" in trace
        assert "thresholds" in trace
        
        # Check structure of checks
        checks = trace["checks"]
        assert "budget_confounded" in checks
        assert "prng_not_volatile" in checks
        assert "topology_stable" in checks
        
        # Each check should have value, source, raw_value
        for check_name, check_data in checks.items():
            assert "value" in check_data
            assert "source" in check_data
            assert "raw_value" in check_data
    
    def test_exclusion_trace_with_missing_signals(self):
        """Test that missing signals are explicitly reflected in trace."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        # No PRNG or topology signals provided
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        trace = exclusion["exclusion_trace"]
        
        # Should have missing_signal_policy at trace root
        assert "missing_signal_policy" in trace
        assert trace["missing_signal_policy"] == "DEFAULT_TRUE_MISSING"
        
        # PRNG should show DEFAULT_TRUE_MISSING
        assert trace["checks"]["prng_not_volatile"]["source"] == "DEFAULT_TRUE_MISSING"
        assert trace["checks"]["prng_not_volatile"]["value"] is True
        assert trace["checks"]["prng_not_volatile"]["raw_value"] == "UNKNOWN"
        
        # Topology should show DEFAULT_TRUE_MISSING
        assert trace["checks"]["topology_stable"]["source"] == "DEFAULT_TRUE_MISSING"
        assert trace["checks"]["topology_stable"]["value"] is True
        assert trace["checks"]["topology_stable"]["raw_value"] == "UNKNOWN"
    
    def test_exclusion_trace_with_provided_signals(self):
        """Test that provided signals are correctly traced."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "DRIFTING"}
        topology_signal = {"pressure_band": "MEDIUM"}
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        trace = exclusion["exclusion_trace"]
        
        # PRNG should show source as prng_signal
        assert trace["checks"]["prng_not_volatile"]["source"] == "prng_signal"
        assert trace["checks"]["prng_not_volatile"]["raw_value"] == "DRIFTING"
        assert trace["checks"]["prng_not_volatile"]["value"] is True  # DRIFTING != VOLATILE
        
        # Topology should show source as topology_signal
        assert trace["checks"]["topology_stable"]["source"] == "topology_signal"
        assert trace["checks"]["topology_stable"]["raw_value"] == "MEDIUM"
        assert trace["checks"]["topology_stable"]["value"] is True  # MEDIUM != HIGH
    
    def test_exclusion_trace_decision_logic(self):
        """Test that trace decision matches recommendation."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        trace = exclusion["exclusion_trace"]
        
        # Decision should match recommendation
        assert trace["decision"] == exclusion["calibration_exclusion_recommended"]
        
        # Reason should match exclusion_reason
        assert trace["reason"] == exclusion["exclusion_reason"]
    
    def test_exclusion_trace_thresholds(self):
        """Test that thresholds are included in trace."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        trace = exclusion["exclusion_trace"]
        thresholds = trace["thresholds"]
        
        assert "prng_volatile_threshold" in thresholds
        assert thresholds["prng_volatile_threshold"] == "VOLATILE"
        
        assert "topology_high_pressure_threshold" in thresholds
        assert thresholds["topology_high_pressure_threshold"] == "HIGH"
    
    def test_exclusion_trace_missing_signal_policy(self):
        """Test that missing_signal_policy is included in trace."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        trace = exclusion["exclusion_trace"]
        
        assert "missing_signal_policy" in trace
        assert trace["missing_signal_policy"] == "DEFAULT_TRUE_MISSING"
    
    def test_exclusion_trace_deterministic(self):
        """Test that trace is deterministic (same inputs → same trace)."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion1 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        exclusion2 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        # Traces should be identical
        assert exclusion1["exclusion_trace"] == exclusion2["exclusion_trace"]
        
        # JSON serialization should also be identical
        json1 = json.dumps(exclusion1["exclusion_trace"], sort_keys=True)
        json2 = json.dumps(exclusion2["exclusion_trace"], sort_keys=True)
        assert json1 == json2


class TestCalibrationExclusionJSONDeterminism:
    """Tests for JSON serialization determinism of exclusion recommendations."""
    
    def test_exclusion_recommendation_json_serializable(self):
        """Test that exclusion recommendation is JSON serializable."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        exclusion = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
        )
        
        # Should serialize without error
        json_str = json.dumps(exclusion)
        assert json_str is not None
        
        # Should deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["calibration_exclusion_recommended"] is True
        assert deserialized["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert "exclusion_trace" in deserialized
    
    def test_annotated_windows_json_serializable(self):
        """Test that annotated windows are JSON serializable."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "budget_confounded": True,
                "drift_classification": "TRANSIENT",
            }
        ]
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
        )
        
        # Should serialize without error
        json_str = json.dumps(annotated)
        assert json_str is not None
        
        # Should deserialize back
        deserialized = json.loads(json_str)
        assert len(deserialized) == 1
        assert deserialized[0]["calibration_exclusion_recommended"] is True
    
    def test_exclusion_recommendation_deterministic(self):
        """Test that exclusion recommendation is deterministic (same inputs → same outputs)."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion1 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        exclusion2 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        # Should be identical
        assert exclusion1 == exclusion2
        
        # JSON serialization should also be identical
        json1 = json.dumps(exclusion1, sort_keys=True)
        json2 = json.dumps(exclusion2, sort_keys=True)
        assert json1 == json2


class TestAnnotateCalibrationWindowWithExclusion:
    """Tests for annotate_calibration_window_with_exclusion convenience function."""
    
    def test_annotate_window_with_budget_modulation(self):
        """Test annotating a window that already has budget modulation."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        annotated = annotate_calibration_window_with_exclusion(
            window=window,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        assert annotated["calibration_exclusion_recommended"] is True
        assert annotated["exclusion_reason"] == "BUDGET_CONFOUNDED_TRANSIENT"
        assert "cross_signal_checks" in annotated
        assert "exclusion_trace" in annotated
    
    def test_annotate_window_includes_trace(self):
        """Test that annotated windows include exclusion trace."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "budget_confounded": True,
                "drift_classification": "TRANSIENT",
            }
        ]
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
        )
        
        assert len(annotated) == 1
        assert "exclusion_trace" in annotated[0]
        assert "checks" in annotated[0]["exclusion_trace"]
        assert "decision" in annotated[0]["exclusion_trace"]
        assert "reason" in annotated[0]["exclusion_trace"]
        assert "thresholds" in annotated[0]["exclusion_trace"]
    
    def test_trace_present_when_budget_modulation_missing(self):
        """Test that trace is present even when budget modulation is missing."""
        windows = [
            {
                "start_cycle": 0,
                "end_cycle": 50,
                "divergence_rate": 0.12,
                # No budget_confounded field
            }
        ]
        
        annotated = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=windows,
        )
        
        assert len(annotated) == 1
        assert "exclusion_trace" in annotated[0]
        
        trace = annotated[0]["exclusion_trace"]
        
        # Should have missing_signal_policy
        assert "missing_signal_policy" in trace
        assert trace["missing_signal_policy"] == "DEFAULT_TRUE_MISSING"
        
        # Budget check should show source as budget_modulation (even though missing)
        assert trace["checks"]["budget_confounded"]["source"] == "budget_modulation"
        assert trace["checks"]["budget_confounded"]["value"] is False
        assert trace["checks"]["budget_confounded"]["raw_value"] == "false"
        
        # PRNG and topology should show DEFAULT_TRUE_MISSING
        assert trace["checks"]["prng_not_volatile"]["source"] == "DEFAULT_TRUE_MISSING"
        assert trace["checks"]["topology_stable"]["source"] == "DEFAULT_TRUE_MISSING"
    
    def test_trace_deterministic_dict_ordering(self):
        """Test that trace has deterministic dict ordering (sorted keys)."""
        budget_modulation = {
            "budget_confounded": True,
            "drift_classification": "TRANSIENT",
        }
        
        prng_signal = {"drift_status": "STABLE"}
        topology_signal = {"pressure_band": "LOW"}
        
        exclusion1 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        exclusion2 = compute_calibration_exclusion_recommendation(
            budget_modulation=budget_modulation,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        trace1 = exclusion1["exclusion_trace"]
        trace2 = exclusion2["exclusion_trace"]
        
        # Traces should be identical
        assert trace1 == trace2
        
        # JSON serialization should be identical (tests key ordering)
        json1 = json.dumps(trace1, sort_keys=True)
        json2 = json.dumps(trace2, sort_keys=True)
        assert json1 == json2
        
        # Verify checks dict keys are in consistent order
        checks1_keys = list(trace1["checks"].keys())
        checks2_keys = list(trace2["checks"].keys())
        assert checks1_keys == checks2_keys
        
        # Verify thresholds dict keys are in consistent order
        thresholds1_keys = list(trace1["thresholds"].keys())
        thresholds2_keys = list(trace2["thresholds"].keys())
        assert thresholds1_keys == thresholds2_keys
        
        # Verify top-level trace keys are in consistent order
        trace1_keys = list(trace1.keys())
        trace2_keys = list(trace2.keys())
        assert trace1_keys == trace2_keys
    
    def test_annotate_window_without_budget_modulation(self):
        """Test annotating a window without budget modulation."""
        window = {
            "start_cycle": 0,
            "end_cycle": 50,
            "divergence_rate": 0.12,
        }
        
        annotated = annotate_calibration_window_with_exclusion(
            window=window,
        )
        
        assert annotated["calibration_exclusion_recommended"] is False
        assert annotated["exclusion_reason"] is None

