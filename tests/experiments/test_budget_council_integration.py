"""
Integration tests for budget council with global health and evidence.
"""

import json
import pytest
from experiments.budget_observability import BudgetSummary, BudgetHealthStatus
from experiments.uplift_council import (
    build_uplift_council_view,
    summarize_uplift_council_for_global_console,
    budget_summary_to_council_input,
    attach_budget_council_to_evidence,
    build_first_light_budget_summary,
    CRITICAL_UPLIFT_SLICES,
)
# Import with fallback for testing
try:
    from backend.health.global_surface import attach_uplift_council_tile
except ImportError:
    from backend.health.uplift_council_adapter import attach_uplift_council_tile


class TestBudgetSummaryToCouncilInput:
    """Tests for budget_summary_to_council_input helper."""
    
    def test_converts_safe_budget(self):
        """Test conversion of SAFE budget summary."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=0,
            max_candidates_hit_count=0,
            timeout_abstentions_total=5,
        )
        
        result = budget_summary_to_council_input(
            slice_name="slice_uplift_goal",
            budget_summary=summary,
            health_status="SAFE",
            frequently_starved=False,
        )
        
        assert result["slice_name"] == "slice_uplift_goal"
        assert result["health_status"] == "SAFE"
        assert result["frequently_starved"] is False
        assert result["budget_exhausted_pct"] == 0.0
        assert result["timeout_abstentions_avg"] == 0.05
    
    def test_converts_starved_budget(self):
        """Test conversion of STARVED budget summary."""
        summary = BudgetSummary(
            total_cycles=100,
            budget_exhausted_count=10,
            max_candidates_hit_count=5,
            timeout_abstentions_total=150,
        )
        
        result = budget_summary_to_council_input(
            slice_name="slice_uplift_sparse",
            budget_summary=summary,
            health_status="STARVED",
            frequently_starved=True,
        )
        
        assert result["slice_name"] == "slice_uplift_sparse"
        assert result["health_status"] == "STARVED"
        assert result["frequently_starved"] is True
        assert result["budget_exhausted_pct"] == 10.0
        assert result["timeout_abstentions_avg"] == 1.5


class TestGlobalHealthIntegration:
    """Tests for global health tile attachment."""
    
    def test_attach_council_tile_to_global_health(self):
        """Test attaching council tile to global health."""
        # Build council view with budget blocking critical slice
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": True,
                },
            ]
        }
        council_view = build_uplift_council_view(budget_view, None, None)
        council_tile = summarize_uplift_council_for_global_console(council_view)
        
        # Attach to global health
        global_health = {"schema_version": "1.0.0"}
        updated = attach_uplift_council_tile(global_health, council_tile)
        
        assert "uplift_council" in updated
        assert updated["uplift_council"]["status_light"] == "RED"
        assert updated["uplift_council"]["budget_status"] == "BLOCK"
        assert "slice_uplift_goal" in updated["uplift_council"]["critical_slices_blocked"]
    
    def test_global_health_green_when_all_ok(self):
        """Test global health shows GREEN when all dimensions OK."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "SAFE",
                    "frequently_starved": False,
                },
            ]
        }
        perf_trend = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "status": "OK"},
            ]
        }
        metric_conformance = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "ready": True},
            ]
        }
        
        council_view = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        council_tile = summarize_uplift_council_for_global_console(council_view)
        
        global_health = {}
        updated = attach_uplift_council_tile(global_health, council_tile)
        
        assert updated["uplift_council"]["status_light"] == "GREEN"
        assert updated["uplift_council"]["budget_status"] == "OK"
        assert len(updated["uplift_council"]["critical_slices_blocked"]) == 0
    
    def test_global_health_yellow_when_warn(self):
        """Test global health shows YELLOW when WARN."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "TIGHT",
                    "frequently_starved": False,
                },
            ]
        }
        
        council_view = build_uplift_council_view(budget_view, None, None)
        council_tile = summarize_uplift_council_for_global_console(council_view)
        
        global_health = {}
        updated = attach_uplift_council_tile(global_health, council_tile)
        
        assert updated["uplift_council"]["status_light"] == "YELLOW"
        assert updated["uplift_council"]["budget_status"] == "WARN"
    
    def test_council_tile_metadata_included(self):
        """Test that council tile includes all required metadata."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "SAFE",
                    "frequently_starved": False,
                },
            ]
        }
        
        council_view = build_uplift_council_view(budget_view, None, None)
        council_tile = summarize_uplift_council_for_global_console(council_view)
        
        # Check required metadata fields
        assert "critical_slices_blocked" in council_tile
        assert "blocked_slices" in council_tile
        assert "ready_slices" in council_tile
        assert "status_light" in council_tile
        assert "budget_status" in council_tile
        assert "headline" in council_tile
    
    def test_read_only_global_health(self):
        """Test that attach_uplift_council_tile doesn't mutate input."""
        council_tile = {
            "tile_type": "uplift_council",
            "status_light": "GREEN",
        }
        global_health = {"schema_version": "1.0.0"}
        global_health_copy = dict(global_health)
        
        updated = attach_uplift_council_tile(global_health, council_tile)
        
        # Input should be unchanged
        assert global_health == global_health_copy
        # Output should have tile
        assert "uplift_council" in updated


class TestEvidencePackIntegration:
    """Tests for evidence pack integration."""
    
    def test_attach_council_to_evidence(self):
        """Test attaching council view to evidence pack."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": True,
                },
            ]
        }
        council_view = build_uplift_council_view(budget_view, None, None)
        
        evidence = {
            "evidence_type": "uplift_experiment",
            "timestamp": "2025-12-10T00:00:00Z",
        }
        
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        assert "governance" in updated
        assert "budget_council" in updated["governance"]
        assert updated["governance"]["budget_council"]["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in updated["governance"]["budget_council"]["slices_blocked_by_budget"]
    
    def test_evidence_pack_read_only(self):
        """Test that attach_budget_council_to_evidence doesn't mutate input."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
        }
        evidence = {
            "evidence_type": "uplift_experiment",
        }
        evidence_copy = dict(evidence)
        
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        # Input should be unchanged
        assert evidence == evidence_copy
        # Output should have council
        assert "governance" in updated
        assert "budget_council" in updated["governance"]
    
    def test_evidence_pack_json_serializable(self):
        """Test that evidence pack with council is JSON-serializable."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "SAFE",
                    "frequently_starved": False,
                },
            ]
        }
        council_view = build_uplift_council_view(budget_view, None, None)
        
        evidence = {
            "evidence_type": "uplift_experiment",
        }
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        # Should serialize to JSON without error
        json_str = json.dumps(updated)
        assert json_str is not None
        
        # Should deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["governance"]["budget_council"]["council_status"] == "OK"
    
    def test_evidence_pack_preserves_existing_governance(self):
        """Test that evidence pack preserves existing governance data."""
        council_view = {
            "council_status": "OK",
        }
        evidence = {
            "governance": {
                "other_system": {
                    "status": "OK",
                },
            },
        }
        
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        # Should preserve existing governance
        assert "other_system" in updated["governance"]
        # Should add budget_council
        assert "budget_council" in updated["governance"]


class TestFirstLightBudgetSummary:
    """Tests for First Light budget summary."""
    
    def test_build_summary_from_council_view(self):
        """Test building First Light summary from council view."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": ["slice_uplift_sparse"],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        summary = build_first_light_budget_summary(council_view)
        
        assert summary["status"] == "BLOCK"
        assert "slice_uplift_goal" in summary["critical_slices_blocked"]
        assert "slice_uplift_goal" in summary["blocked_slices"]
        assert "slice_uplift_sparse" in summary["ready_slices"]
    
    def test_summary_includes_all_blocked_slices(self):
        """Test that summary includes slices blocked by any dimension."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": ["slice_uplift_sparse"],
            "slices_blocked_by_metrics": ["slice_uplift_tree"],
        }
        
        summary = build_first_light_budget_summary(council_view)
        
        assert len(summary["blocked_slices"]) == 3
        assert "slice_uplift_goal" in summary["blocked_slices"]
        assert "slice_uplift_sparse" in summary["blocked_slices"]
        assert "slice_uplift_tree" in summary["blocked_slices"]
    
    def test_summary_identifies_critical_blocked(self):
        """Test that summary correctly identifies critical slices blocked."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice_uplift_goal", "slice_non_critical"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        summary = build_first_light_budget_summary(council_view)
        
        # Only critical slices should be in critical_slices_blocked
        assert "slice_uplift_goal" in summary["critical_slices_blocked"]
        assert "slice_non_critical" not in summary["critical_slices_blocked"]
        # But both should be in blocked_slices
        assert "slice_uplift_goal" in summary["blocked_slices"]
        assert "slice_non_critical" in summary["blocked_slices"]
    
    def test_summary_ok_status(self):
        """Test summary with OK status."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        summary = build_first_light_budget_summary(council_view)
        
        assert summary["status"] == "OK"
        assert len(summary["critical_slices_blocked"]) == 0
        assert len(summary["blocked_slices"]) == 0
        assert len(summary["ready_slices"]) == 2


class TestFirstLightEvidenceIntegration:
    """Tests for First Light summary in evidence pack."""
    
    def test_evidence_includes_summary(self):
        """Test that evidence pack includes First Light summary."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": ["slice_uplift_sparse"],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        evidence = {
            "evidence_type": "uplift_experiment",
        }
        
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        # Should have both full council view and summary
        assert "budget_council" in updated["governance"]
        assert "budget_council_summary" in updated["governance"]
        
        summary = updated["governance"]["budget_council_summary"]
        assert summary["status"] == "BLOCK"
        assert "slice_uplift_goal" in summary["critical_slices_blocked"]
        assert "slice_uplift_goal" in summary["blocked_slices"]
        assert "slice_uplift_sparse" in summary["ready_slices"]
    
    def test_summary_structure_matches_expected(self):
        """Test that summary structure matches First Light expectations."""
        council_view = {
            "council_status": "WARN",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": ["slice_uplift_sparse"],
            "slices_blocked_by_metrics": [],
        }
        
        evidence = {}
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        summary = updated["governance"]["budget_council_summary"]
        
        # Verify structure
        assert "status" in summary
        assert "critical_slices_blocked" in summary
        assert "blocked_slices" in summary
        assert "ready_slices" in summary
        
        # Verify types
        assert isinstance(summary["status"], str)
        assert isinstance(summary["critical_slices_blocked"], list)
        assert isinstance(summary["blocked_slices"], list)
        assert isinstance(summary["ready_slices"], list)
    
    def test_summary_json_serializable(self):
        """Test that summary is JSON-serializable."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        evidence = {}
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        summary = updated["governance"]["budget_council_summary"]
        
        # Should serialize to JSON without error
        json_str = json.dumps(summary)
        assert json_str is not None
        
        # Should deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["status"] == "OK"
        assert "slice_uplift_goal" in deserialized["ready_slices"]
    
    def test_evidence_only_cross_check_scenario(self):
        """
        Scenario: Budget council BLOCK as evidence-only cross-check.
        
        In Phase X (SHADOW MODE), a BLOCK status in budget_council_summary
        does NOT abort or gate any decisions. It serves as a consistency check
        against other governance tiles (performance summary, budget storyline).
        
        In a future Phase Y, a BLOCK here could be used as an input to LastMile
        gating, but currently it is evidence-only.
        
        This test verifies that the summary is attached correctly and can be
        used for cross-checking, but does not implement any blocking logic.
        """
        # Simulate a BLOCK scenario
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": ["slice_uplift_sparse"],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        evidence = {
            "evidence_type": "first_light",
            "governance": {
                # Other governance tiles for cross-check
                "uplift_perf_summary": {
                    "perf_risk": "LOW",
                    "slices_with_regressions": [],
                },
            },
        }
        
        updated = attach_budget_council_to_evidence(evidence, council_view)
        
        # Verify summary is attached
        summary = updated["governance"]["budget_council_summary"]
        assert summary["status"] == "BLOCK"
        assert "slice_uplift_goal" in summary["critical_slices_blocked"]
        
        # Verify other governance tiles are preserved (cross-check capability)
        assert "uplift_perf_summary" in updated["governance"]
        
        # Note: In Phase X, this BLOCK does not prevent evidence pack generation
        # or abort any processes. It is purely advisory for external reviewers.

