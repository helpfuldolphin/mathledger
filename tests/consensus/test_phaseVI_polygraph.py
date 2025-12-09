"""
Consensus Polygraph v2 Tests

Tests for multi-agent consensus and conflict detection engine.

Author: Agent E4 (doc-ops-4) — Migration Council Orchestrator
Date: 2025-12-06
"""

import pytest
from typing import Any

# Add project root to path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.consensus_polygraph import (
    build_consensus_polygraph,
    detect_predictive_conflicts,
    build_consensus_director_panel,
    SystemConflict,
    ConsensusPolygraphResult,
    normalize_status,
    extract_slice_statuses,
    PANEL_STATUS_OK,
    PANEL_STATUS_ATTENTION,
    PANEL_STATUS_BLOCK,
)


# =============================================================================
# SECTION 1: Status Normalization Tests (5 tests)
# =============================================================================

class TestStatusNormalization:
    """Tests for normalize_status function."""
    
    def test_normalize_ok_variants(self):
        """Test that OK variants normalize correctly."""
        assert normalize_status("OK") == PANEL_STATUS_OK
        assert normalize_status("PASS") == PANEL_STATUS_OK
        assert normalize_status("GREEN") == PANEL_STATUS_OK
        assert normalize_status("SUCCESS") == PANEL_STATUS_OK
        assert normalize_status("VALID") == PANEL_STATUS_OK
    
    def test_normalize_attention_variants(self):
        """Test that ATTENTION variants normalize correctly."""
        assert normalize_status("ATTENTION") == PANEL_STATUS_ATTENTION
        assert normalize_status("WARN") == PANEL_STATUS_ATTENTION
        assert normalize_status("WARNING") == PANEL_STATUS_ATTENTION
        assert normalize_status("YELLOW") == PANEL_STATUS_ATTENTION
        assert normalize_status("CAUTION") == PANEL_STATUS_ATTENTION
    
    def test_normalize_block_variants(self):
        """Test that BLOCK variants normalize correctly."""
        assert normalize_status("BLOCK") == PANEL_STATUS_BLOCK
        assert normalize_status("FAIL") == PANEL_STATUS_BLOCK
        assert normalize_status("RED") == PANEL_STATUS_BLOCK
        assert normalize_status("ERROR") == PANEL_STATUS_BLOCK
        assert normalize_status("CRITICAL") == PANEL_STATUS_BLOCK
    
    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_status("ok") == PANEL_STATUS_OK
        assert normalize_status("Ok") == PANEL_STATUS_OK
        assert normalize_status("OK") == PANEL_STATUS_OK
        assert normalize_status("block") == PANEL_STATUS_BLOCK
        assert normalize_status("Block") == PANEL_STATUS_BLOCK
    
    def test_normalize_unknown_defaults_to_attention(self):
        """Test that unknown statuses default to ATTENTION."""
        assert normalize_status("UNKNOWN") == PANEL_STATUS_ATTENTION
        assert normalize_status("") == PANEL_STATUS_ATTENTION
        assert normalize_status("xyz") == PANEL_STATUS_ATTENTION


# =============================================================================
# SECTION 2: Slice Status Extraction Tests (6 tests)
# =============================================================================

class TestSliceStatusExtraction:
    """Tests for extract_slice_statuses function."""
    
    def test_extract_from_slices_array(self):
        """Test extraction from 'slices' array format."""
        panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "ATTENTION"},
                {"id": "slice_3", "signal": "BLOCK"},
            ]
        }
        
        statuses = extract_slice_statuses(panel, "test")
        
        assert "slice_1" in statuses
        assert statuses["slice_1"] == "OK"
        assert "slice_2" in statuses
        assert statuses["slice_2"] == "ATTENTION"
        assert "slice_3" in statuses
        assert statuses["slice_3"] == "BLOCK"
    
    def test_extract_from_components_dict(self):
        """Test extraction from 'components' dict format."""
        panel = {
            "components": {
                "slice_1": {"status": "OK"},
                "slice_2": {"signal": "ATTENTION"},
                "slice_3": "BLOCK",
            }
        }
        
        statuses = extract_slice_statuses(panel, "test")
        
        assert statuses["slice_1"] == "OK"
        assert statuses["slice_2"] == "ATTENTION"
        assert statuses["slice_3"] == "BLOCK"
    
    def test_extract_from_slice_statuses_dict(self):
        """Test extraction from 'slice_statuses' dict format."""
        panel = {
            "slice_statuses": {
                "slice_1": "OK",
                "slice_2": "ATTENTION",
                "slice_3": "BLOCK",
            }
        }
        
        statuses = extract_slice_statuses(panel, "test")
        
        assert statuses["slice_1"] == "OK"
        assert statuses["slice_2"] == "ATTENTION"
        assert statuses["slice_3"] == "BLOCK"
    
    def test_extract_from_nested_dict(self):
        """Test extraction from nested dict structure."""
        panel = {
            "slice_1": {"status": "OK"},
            "slice_2": {"signal": "ATTENTION"},
            "slice_3": {"verdict": "BLOCK"},
        }
        
        statuses = extract_slice_statuses(panel, "test")
        
        assert statuses["slice_1"] == "OK"
        assert statuses["slice_2"] == "ATTENTION"
        assert statuses["slice_3"] == "BLOCK"
    
    def test_extract_empty_panel(self):
        """Test extraction from empty panel."""
        panel = {}
        
        statuses = extract_slice_statuses(panel, "test")
        
        assert len(statuses) == 0
    
    def test_extract_missing_fields(self):
        """Test extraction handles missing fields gracefully."""
        panel = {
            "slices": [
                {"slice_id": "slice_1"},  # No status
                {"status": "OK"},  # No slice_id
                {"slice_id": "slice_2", "status": "OK"},
            ]
        }
        
        statuses = extract_slice_statuses(panel, "test")
        
        # Only slice_2 should be extracted
        assert "slice_2" in statuses
        assert len(statuses) == 1


# =============================================================================
# SECTION 3: Consensus Polygraph Tests (8 tests)
# =============================================================================

class TestConsensusPolygraph:
    """Tests for build_consensus_polygraph function."""
    
    def test_polygraph_no_panels(self):
        """Test polygraph with no panels."""
        result = build_consensus_polygraph()
        
        assert result.agreement_rate == 1.0
        assert result.consensus_band == "HIGH"
        assert len(result.system_conflicts) == 0
        assert "No panels provided" in " ".join(result.neutral_notes)
    
    def test_polygraph_full_agreement(self):
        """Test polygraph when all systems agree."""
        semantic_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "OK"},
            ]
        }
        metric_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "OK"},
            ]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert result.agreement_rate == 1.0
        assert result.consensus_band == "HIGH"
        assert len(result.system_conflicts) == 0
        assert result.agreeing_slices == 2
    
    def test_polygraph_conflict_detection(self):
        """Test polygraph detects conflicts when systems disagree."""
        semantic_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "BLOCK"},
            ]
        }
        metric_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "OK"},  # Disagrees with semantic
            ]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert len(result.system_conflicts) == 1
        assert result.system_conflicts[0].slice_id == "slice_2"
        assert "semantic" in result.system_conflicts[0].conflicting_systems
        assert "metric" in result.system_conflicts[0].conflicting_systems
        assert result.agreement_rate < 1.0
    
    def test_polygraph_conflict_severity(self):
        """Test that conflict severity is calculated correctly."""
        semantic_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "BLOCK"},
                {"slice_id": "slice_2", "status": "ATTENTION"},
            ]
        }
        metric_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},  # BLOCK vs OK = HIGH
                {"slice_id": "slice_2", "status": "OK"},  # ATTENTION vs OK = MEDIUM
            ]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert len(result.system_conflicts) == 2
        # slice_1 should be HIGH (BLOCK involved)
        high_conflicts = [c for c in result.system_conflicts if c.severity == "HIGH"]
        assert len(high_conflicts) == 1
        assert high_conflicts[0].slice_id == "slice_1"
    
    def test_polygraph_consensus_band_high(self):
        """Test consensus band HIGH (>=80% agreement)."""
        semantic_panel = {
            "slices": [{"slice_id": f"slice_{i}", "status": "OK"} for i in range(10)]
        }
        metric_panel = {
            "slices": [
                {"slice_id": f"slice_{i}", "status": "OK"} for i in range(9)
            ] + [{"slice_id": "slice_9", "status": "ATTENTION"}]  # 1 conflict out of 10
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert result.agreement_rate == 0.9
        assert result.consensus_band == "HIGH"
    
    def test_polygraph_consensus_band_medium(self):
        """Test consensus band MEDIUM (50-80% agreement)."""
        semantic_panel = {
            "slices": [{"slice_id": f"slice_{i}", "status": "OK"} for i in range(10)]
        }
        metric_panel = {
            "slices": [
                {"slice_id": f"slice_{i}", "status": "OK"} for i in range(6)
            ] + [{"slice_id": f"slice_{i}", "status": "ATTENTION"} for i in range(6, 10)]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert result.agreement_rate == 0.6
        assert result.consensus_band == "MEDIUM"
    
    def test_polygraph_consensus_band_low(self):
        """Test consensus band LOW (<50% agreement)."""
        semantic_panel = {
            "slices": [{"slice_id": f"slice_{i}", "status": "OK"} for i in range(10)]
        }
        metric_panel = {
            "slices": [
                {"slice_id": f"slice_{i}", "status": "OK"} for i in range(4)
            ] + [{"slice_id": f"slice_{i}", "status": "BLOCK"} for i in range(4, 10)]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
        )
        
        assert result.agreement_rate == 0.4
        assert result.consensus_band == "LOW"
    
    def test_polygraph_multiple_systems(self):
        """Test polygraph with multiple systems."""
        semantic_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "OK"},
            ]
        }
        metric_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "ATTENTION"},
            ]
        }
        topology_panel = {
            "slices": [
                {"slice_id": "slice_1", "status": "OK"},
                {"slice_id": "slice_2", "status": "OK"},
            ]
        }
        
        result = build_consensus_polygraph(
            semantic_panel=semantic_panel,
            metric_panel=metric_panel,
            topology_panel=topology_panel,
        )
        
        # slice_2: semantic=OK, metric=ATTENTION, topology=OK
        # Majority is OK, but there's still a conflict
        assert len(result.system_conflicts) == 1
        assert result.system_conflicts[0].slice_id == "slice_2"
        assert len(result.system_conflicts[0].conflicting_systems) == 3


# =============================================================================
# SECTION 4: Predictive Conflict Detection Tests (6 tests)
# =============================================================================

class TestPredictiveConflictDetection:
    """Tests for detect_predictive_conflicts function."""
    
    def test_predictive_high_severity_conflicts(self):
        """Test that HIGH severity conflicts generate predictions."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[
                SystemConflict(
                    slice_id="slice_1",
                    component=None,
                    conflicting_systems=["semantic", "metric"],
                    statuses={"semantic": "BLOCK", "metric": "OK"},
                    severity="HIGH",
                )
            ],
            agreement_rate=0.8,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=5,
            agreeing_slices=4,
        )
        
        predictions = detect_predictive_conflicts(polygraph)
        
        assert predictions["total_predictions"] > 0
        assert predictions["high_risk_predictions"] > 0
        assert any(p["slice_id"] == "slice_1" for p in predictions["predictive_conflicts"])
    
    def test_predictive_medium_severity_many_systems(self):
        """Test that MEDIUM conflicts with many systems generate predictions."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[
                SystemConflict(
                    slice_id="slice_1",
                    component=None,
                    conflicting_systems=["semantic", "metric", "topology", "drift"],
                    statuses={
                        "semantic": "ATTENTION",
                        "metric": "ATTENTION",
                        "topology": "OK",
                        "drift": "ATTENTION",
                    },
                    severity="MEDIUM",
                )
            ],
            agreement_rate=0.7,
            consensus_band="MEDIUM",
            neutral_notes=[],
            total_slices=5,
            agreeing_slices=4,
        )
        
        predictions = detect_predictive_conflicts(polygraph)
        
        assert predictions["total_predictions"] > 0
        assert any(p["slice_id"] == "slice_1" for p in predictions["predictive_conflicts"])
    
    def test_predictive_consensus_band_low(self):
        """Test that LOW consensus band generates global prediction."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.3,
            consensus_band="LOW",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=3,
        )
        
        predictions = detect_predictive_conflicts(polygraph)
        
        assert predictions["total_predictions"] > 0
        assert any(
            p["slice_id"] == "GLOBAL" and p["risk"] == "HIGH"
            for p in predictions["predictive_conflicts"]
        )
    
    def test_predictive_with_historical_trend(self):
        """Test predictive detection with historical comparison."""
        current = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.6,
            consensus_band="MEDIUM",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=6,
        )
        historical = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.9,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=9,
        )
        
        predictions = detect_predictive_conflicts(current, historical)
        
        # Should detect rate decrease
        assert predictions["total_predictions"] > 0
        assert any(
            "decreased" in p["reason"].lower()
            for p in predictions["predictive_conflicts"]
        )
    
    def test_predictive_new_conflicts(self):
        """Test detection of new conflicts."""
        current = ConsensusPolygraphResult(
            system_conflicts=[
                SystemConflict(
                    slice_id="slice_new",
                    component=None,
                    conflicting_systems=["semantic", "metric"],
                    statuses={"semantic": "OK", "metric": "ATTENTION"},
                    severity="MEDIUM",
                )
            ],
            agreement_rate=0.8,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=5,
            agreeing_slices=4,
        )
        historical = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=1.0,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=5,
            agreeing_slices=5,
        )
        
        predictions = detect_predictive_conflicts(current, historical)
        
        assert predictions["total_predictions"] > 0
        assert any(
            "new conflicts" in p["reason"].lower()
            for p in predictions["predictive_conflicts"]
        )
    
    def test_predictive_no_conflicts(self):
        """Test predictive detection with no conflicts."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=1.0,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=10,
        )
        
        predictions = detect_predictive_conflicts(polygraph)
        
        # Should have minimal or no predictions
        assert predictions["total_predictions"] >= 0
        assert predictions["high_risk_predictions"] == 0


# =============================================================================
# SECTION 5: Consensus Director Panel Tests (5 tests)
# =============================================================================

class TestConsensusDirectorPanel:
    """Tests for build_consensus_director_panel function."""
    
    def test_director_panel_structure(self):
        """Test director panel has expected structure."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.9,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=9,
        )
        
        panel = build_consensus_director_panel(polygraph)
        
        assert "status_light" in panel
        assert "consensus_band" in panel
        assert "agreement_rate" in panel
        assert "headline" in panel
        assert "total_slices" in panel
        assert "conflicts" in panel
    
    def test_director_panel_status_light_high(self):
        """Test status light for HIGH consensus band."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.9,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=9,
        )
        
        panel = build_consensus_director_panel(polygraph)
        
        assert panel["status_light"] == "GREEN"
    
    def test_director_panel_status_light_medium(self):
        """Test status light for MEDIUM consensus band."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.7,
            consensus_band="MEDIUM",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=7,
        )
        
        panel = build_consensus_director_panel(polygraph)
        
        assert panel["status_light"] == "YELLOW"
    
    def test_director_panel_status_light_low(self):
        """Test status light for LOW consensus band."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.3,
            consensus_band="LOW",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=3,
        )
        
        panel = build_consensus_director_panel(polygraph)
        
        assert panel["status_light"] == "RED"
    
    def test_director_panel_with_predictive_conflicts(self):
        """Test director panel includes predictive conflict info."""
        polygraph = ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=0.8,
            consensus_band="HIGH",
            neutral_notes=[],
            total_slices=10,
            agreeing_slices=8,
        )
        predictive = {
            "predictive_conflicts": [
                {"slice_id": "slice_1", "risk": "HIGH", "reason": "test", "systems_involved": []}
            ],
            "total_predictions": 1,
            "high_risk_predictions": 1,
        }
        
        panel = build_consensus_director_panel(polygraph, predictive)
        
        assert panel["predictive_risks"] == 1
        assert "high-risk" in panel["headline"].lower() or "predictions" in panel["headline"].lower()


# =============================================================================
# Pytest Markers
# =============================================================================

# Mark all tests as unit tests
pytestmark = pytest.mark.unit

