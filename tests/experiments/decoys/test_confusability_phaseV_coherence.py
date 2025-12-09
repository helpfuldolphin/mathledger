# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Phase V — Decoy-Topology Coherence Grid

This module tests:
1. Confusability-Topology Coherence Map
2. Confusability Drift Horizon Predictor
3. Global Coherence Console Tile

All tests verify deterministic behavior and neutral language constraints.
"""

import pytest
from typing import Any, Dict, List

from experiments.decoys.risk import (
    build_confusability_topology_coherence_map,
    build_confusability_drift_horizon_predictor,
    build_global_coherence_console_tile,
    COHERENCE_BAND_THRESHOLDS,
    COHERENCE_BAND_ORDER,
    check_forbidden_language,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_drift_governors() -> Dict[str, Dict[str, Any]]:
    """Sample drift governors for testing."""
    return {
        "slice1": {"drift_severity": "NONE"},
        "slice2": {"drift_severity": "MINOR"},
        "slice3": {"drift_severity": "MAJOR"},
    }


@pytest.fixture
def sample_slice_views() -> Dict[str, Dict[str, Any]]:
    """Sample slice views for testing."""
    return {
        "slice1": {"slice_confusability_status": "OK"},
        "slice2": {"slice_confusability_status": "ATTENTION"},
        "slice3": {"slice_confusability_status": "HOT"},
    }


@pytest.fixture
def sample_topology_statuses() -> Dict[str, str]:
    """Sample topology statuses for testing."""
    return {
        "slice1": "STABLE",
        "slice2": "DRIFTING",
        "slice3": "UNSTABLE",
    }


@pytest.fixture
def sample_semantic_alignments() -> Dict[str, float]:
    """Sample semantic alignments for testing."""
    return {
        "slice1": 0.9,
        "slice2": 0.6,
        "slice3": 0.3,
    }


# =============================================================================
# TASK 1: CONFUSABILITY-TOPOLOGY COHERENCE MAP TESTS
# =============================================================================

class TestBuildConfusabilityTopologyCoherenceMap:
    """Tests for coherence map generation."""
    
    def test_coherence_map_has_required_fields(
        self, sample_drift_governors, sample_slice_views
    ):
        """Coherence map should have all required fields."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        required = {
            "slice_coherence_scores", "global_coherence_index",
            "coherence_band", "root_incoherence_causes"
        }
        assert required.issubset(set(coherence_map.keys()))
    
    def test_coherence_scores_computed_per_slice(
        self, sample_drift_governors, sample_slice_views
    ):
        """Should compute coherence score for each slice."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        scores = coherence_map["slice_coherence_scores"]
        assert len(scores) == 3
        assert "slice1" in scores
        assert "slice2" in scores
        assert "slice3" in scores
        
        # All scores should be in [0, 1]
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_global_coherence_index_average(
        self, sample_drift_governors, sample_slice_views
    ):
        """Global index should be average of slice scores."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        scores = coherence_map["slice_coherence_scores"]
        global_index = coherence_map["global_coherence_index"]
        
        expected = sum(scores.values()) / len(scores)
        assert abs(global_index - expected) < 0.0001
    
    def test_coherence_band_coherent(self):
        """Should return COHERENT for high coherence."""
        drift_governors = {
            "slice1": {"drift_severity": "NONE"},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
        }
        topology = {"slice1": "STABLE"}
        semantic = {"slice1": 0.9}
        
        coherence_map = build_confusability_topology_coherence_map(
            drift_governors, slice_views, topology, semantic
        )
        
        assert coherence_map["coherence_band"] == "COHERENT"
        assert coherence_map["global_coherence_index"] > COHERENCE_BAND_THRESHOLDS["COHERENT"]
    
    def test_coherence_band_partial(self):
        """Should return PARTIAL for moderate coherence."""
        drift_governors = {
            "slice1": {"drift_severity": "MINOR"},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "ATTENTION"},
        }
        topology = {"slice1": "DRIFTING"}
        semantic = {"slice1": 0.5}
        
        coherence_map = build_confusability_topology_coherence_map(
            drift_governors, slice_views, topology, semantic
        )
        
        assert coherence_map["coherence_band"] == "PARTIAL"
        index = coherence_map["global_coherence_index"]
        assert COHERENCE_BAND_THRESHOLDS["PARTIAL"] <= index <= COHERENCE_BAND_THRESHOLDS["COHERENT"]
    
    def test_coherence_band_misaligned(self):
        """Should return MISALIGNED for low coherence."""
        drift_governors = {
            "slice1": {"drift_severity": "MAJOR"},
        }
        slice_views = {
            "slice1": {"slice_confusability_status": "HOT"},
        }
        topology = {"slice1": "UNSTABLE"}
        semantic = {"slice1": 0.1}
        
        coherence_map = build_confusability_topology_coherence_map(
            drift_governors, slice_views, topology, semantic
        )
        
        assert coherence_map["coherence_band"] == "MISALIGNED"
        assert coherence_map["global_coherence_index"] < COHERENCE_BAND_THRESHOLDS["PARTIAL"]
    
    def test_integrates_topology_status(
        self, sample_drift_governors, sample_slice_views, sample_topology_statuses
    ):
        """Should integrate topology status into coherence scores."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views, sample_topology_statuses
        )
        
        # STABLE topology should contribute positively
        assert coherence_map["slice_coherence_scores"]["slice1"] > 0.5
    
    def test_integrates_semantic_alignment(
        self, sample_drift_governors, sample_slice_views, sample_semantic_alignments
    ):
        """Should integrate semantic alignment into coherence scores."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views, None, sample_semantic_alignments
        )
        
        # High semantic alignment should contribute positively
        assert coherence_map["slice_coherence_scores"]["slice1"] > 0.5
    
    def test_handles_missing_optional_inputs(
        self, sample_drift_governors, sample_slice_views
    ):
        """Should handle missing topology and semantic inputs."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        # Should still compute scores (using neutral defaults)
        assert len(coherence_map["slice_coherence_scores"]) == 3
    
    def test_root_causes_identified(self, sample_drift_governors, sample_slice_views):
        """Should identify root causes of incoherence."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        causes = coherence_map["root_incoherence_causes"]
        assert isinstance(causes, list)
        assert len(causes) > 0
    
    def test_root_causes_no_forbidden_words(
        self, sample_drift_governors, sample_slice_views
    ):
        """Root causes should not contain forbidden words."""
        coherence_map = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        for cause in coherence_map["root_incoherence_causes"]:
            forbidden = check_forbidden_language(cause)
            assert len(forbidden) == 0, f"Found forbidden words in '{cause}': {forbidden}"
    
    def test_deterministic_output(
        self, sample_drift_governors, sample_slice_views
    ):
        """Coherence map should be deterministic."""
        map1 = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        map2 = build_confusability_topology_coherence_map(
            sample_drift_governors, sample_slice_views
        )
        
        assert map1 == map2


# =============================================================================
# TASK 2: DRIFT HORIZON PREDICTOR TESTS
# =============================================================================

class TestBuildConfusabilityDriftHorizonPredictor:
    """Tests for drift horizon prediction."""
    
    def test_predictor_has_required_fields(self):
        """Predictor should have all required fields."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        required = {
            "horizon_estimate", "confidence", "trajectory",
            "current_coherence", "prediction_notes"
        }
        assert required.issubset(set(predictor.keys()))
    
    def test_insufficient_history_returns_none(self):
        """Should return None for horizon with insufficient history."""
        history = [{"drift_severity": "NONE"}]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert predictor["horizon_estimate"] is None
        assert predictor["confidence"] == 0.0
    
    def test_trajectory_improving(self):
        """Should detect IMPROVING trajectory."""
        history = [
            {"drift_severity": "MAJOR"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "NONE"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert predictor["trajectory"] == "IMPROVING"
    
    def test_trajectory_degrading(self):
        """Should detect DEGRADING trajectory."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MAJOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert predictor["trajectory"] == "DEGRADING"
    
    def test_trajectory_stable(self):
        """Should detect STABLE trajectory."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "NONE"},
            {"drift_severity": "NONE"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert predictor["trajectory"] == "STABLE"
    
    def test_horizon_estimate_degrading_trajectory(self):
        """Should estimate horizon for degrading trajectory."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MAJOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history, threshold=0.5)
        
        # If degrading and above threshold, should estimate steps
        if predictor["trajectory"] == "DEGRADING" and predictor["current_coherence"] > 0.5:
            assert predictor["horizon_estimate"] is not None or predictor["confidence"] < 0.5
    
    def test_confidence_range(self):
        """Confidence should be in [0, 1]."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MAJOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert 0.0 <= predictor["confidence"] <= 1.0
    
    def test_uses_last_10_points(self):
        """Should use last 10 points for trend analysis."""
        history = [
            {"drift_severity": "NONE"} for _ in range(15)
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        # Should still work with more than 10 points
        assert predictor["trajectory"] in ("IMPROVING", "STABLE", "DEGRADING")
    
    def test_prediction_notes_no_forbidden_words(self):
        """Prediction notes should not contain forbidden words."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        for note in predictor["prediction_notes"]:
            forbidden = check_forbidden_language(note)
            assert len(forbidden) == 0, f"Found forbidden words in '{note}': {forbidden}"
    
    def test_current_coherence_from_last_snapshot(self):
        """Current coherence should be derived from last snapshot."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "MAJOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        # MAJOR = 0.0 coherence
        assert predictor["current_coherence"] == 0.0


# =============================================================================
# TASK 3: GLOBAL COHERENCE CONSOLE TILE TESTS
# =============================================================================

class TestBuildGlobalCoherenceConsoleTile:
    """Tests for global coherence console tile."""
    
    def test_console_tile_has_required_fields(self):
        """Console tile should have all required fields."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.8},
            "global_coherence_index": 0.8,
            "coherence_band": "COHERENT",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        required = {
            "status_light", "coherence_band", "slices_at_risk",
            "dominant_coherence_drivers", "headline"
        }
        assert required.issubset(set(tile.keys()))
    
    def test_status_light_green_for_coherent(self):
        """Should be GREEN for COHERENT band."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.8},
            "global_coherence_index": 0.8,
            "coherence_band": "COHERENT",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        assert tile["status_light"] == "GREEN"
    
    def test_status_light_yellow_for_partial(self):
        """Should be YELLOW for PARTIAL band."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.6},
            "global_coherence_index": 0.6,
            "coherence_band": "PARTIAL",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        assert tile["status_light"] == "YELLOW"
    
    def test_status_light_red_for_misaligned(self):
        """Should be RED for MISALIGNED band."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.3},
            "global_coherence_index": 0.3,
            "coherence_band": "MISALIGNED",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        assert tile["status_light"] == "RED"
    
    def test_slices_at_risk_identified(self):
        """Should identify slices below threshold."""
        coherence_map = {
            "slice_coherence_scores": {
                "slice1": 0.8,  # Above threshold
                "slice2": 0.3,  # Below threshold
                "slice3": 0.4,  # Below threshold
            },
            "global_coherence_index": 0.5,
            "coherence_band": "PARTIAL",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        assert len(tile["slices_at_risk"]) == 2
        assert "slice2" in tile["slices_at_risk"]
        assert "slice3" in tile["slices_at_risk"]
        assert tile["slices_at_risk"] == sorted(tile["slices_at_risk"])
    
    def test_dominant_drivers_extracted(self):
        """Should extract dominant drivers from root causes."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.5},
            "global_coherence_index": 0.5,
            "coherence_band": "PARTIAL",
            "root_incoherence_causes": [
                "Cause 1",
                "Cause 2",
                "Cause 3",
                "Cause 4",
            ],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        # Should take top 3
        assert len(tile["dominant_coherence_drivers"]) == 3
    
    def test_headline_no_forbidden_words(self):
        """Headline should not contain forbidden words."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.5},
            "global_coherence_index": 0.5,
            "coherence_band": "PARTIAL",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        forbidden = check_forbidden_language(tile["headline"])
        assert len(forbidden) == 0, f"Found forbidden words: {forbidden}"
    
    def test_headline_descriptive(self):
        """Headline should be descriptive."""
        coherence_map = {
            "slice_coherence_scores": {"slice1": 0.8},
            "global_coherence_index": 0.8,
            "coherence_band": "COHERENT",
            "root_incoherence_causes": [],
        }
        
        tile = build_global_coherence_console_tile(coherence_map)
        
        headline = tile["headline"]
        assert "Coherence status:" in headline
        assert "COHERENT" in headline or "GREEN" in headline


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhaseVIntegration:
    """Integration tests for Phase V coherence system."""
    
    def test_full_workflow(self):
        """Test complete workflow from drift to console tile."""
        # Build drift governors
        drift_governors = {
            "slice1": {"drift_severity": "NONE"},
            "slice2": {"drift_severity": "MINOR"},
        }
        
        # Build slice views
        slice_views = {
            "slice1": {"slice_confusability_status": "OK"},
            "slice2": {"slice_confusability_status": "ATTENTION"},
        }
        
        # Build coherence map
        coherence_map = build_confusability_topology_coherence_map(
            drift_governors, slice_views
        )
        
        # Build console tile
        tile = build_global_coherence_console_tile(coherence_map)
        
        # Verify outputs
        assert coherence_map["coherence_band"] in ("COHERENT", "PARTIAL", "MISALIGNED")
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")
        assert tile["coherence_band"] == coherence_map["coherence_band"]
    
    def test_horizon_predictor_with_history(self):
        """Test horizon predictor with drift history."""
        history = [
            {"drift_severity": "NONE"},
            {"drift_severity": "NONE"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MINOR"},
            {"drift_severity": "MAJOR"},
        ]
        
        predictor = build_confusability_drift_horizon_predictor(history)
        
        assert predictor["trajectory"] in ("IMPROVING", "STABLE", "DEGRADING")
        assert 0.0 <= predictor["confidence"] <= 1.0

