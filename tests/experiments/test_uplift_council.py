"""
Minimal tests for uplift council.
"""

import pytest
from experiments.uplift_council import (
    build_uplift_council_view,
    summarize_uplift_council_for_global_console,
    CouncilStatus,
    DimensionStatus,
    CRITICAL_UPLIFT_SLICES,
)


class TestBuildUpliftCouncilView:
    """Tests for build_uplift_council_view."""
    
    def test_all_dimensions_ok(self):
        """Test council view when all dimensions are OK."""
        budget_view = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
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
        
        result = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        
        assert result["council_status"] == "OK"
        assert "slice_uplift_goal" in result["slices_ready_for_uplift"]
        assert len(result["slices_blocked_by_budget"]) == 0
    
    def test_blocked_by_budget(self):
        """Test council view when budget blocks a critical slice."""
        budget_view = {
            "slices": [
                {
                    "slice_name": "slice_uplift_goal",
                    "health_status": "STARVED",
                    "frequently_starved": True,
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
        
        result = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_budget"]
    
    def test_blocked_by_perf(self):
        """Test council view when performance blocks a critical slice."""
        budget_view = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
            ]
        }
        perf_trend = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "status": "BLOCK"},
            ]
        }
        metric_conformance = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "ready": True},
            ]
        }
        
        result = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_perf"]
    
    def test_blocked_by_metrics(self):
        """Test council view when metrics block a critical slice."""
        budget_view = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
            ]
        }
        perf_trend = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "status": "OK"},
            ]
        }
        metric_conformance = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "ready": False},
            ]
        }
        
        result = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_metrics"]
    
    def test_warn_for_non_critical(self):
        """Test council view with WARN for non-critical slice."""
        budget_view = {
            "slices": [
                {"slice_name": "slice_non_critical", "health_status": "TIGHT"},
            ]
        }
        perf_trend = {
            "slices": [
                {"slice_name": "slice_non_critical", "status": "OK"},
            ]
        }
        metric_conformance = {
            "slices": [
                {"slice_name": "slice_non_critical", "ready": True},
            ]
        }
        
        result = build_uplift_council_view(budget_view, perf_trend, metric_conformance)
        
        # Non-critical slice with WARN should result in WARN council status
        assert result["council_status"] == "WARN"
    
    def test_missing_dimensions_default_ok(self):
        """Test that missing dimensions default to OK."""
        # Only provide budget view
        budget_view = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
            ]
        }
        
        result = build_uplift_council_view(budget_view, None, None)
        
        # Should default perf and metrics to OK, so overall OK
        assert result["council_status"] == "OK"
        assert "slice_uplift_goal" in result["slices_ready_for_uplift"]


class TestSummarizeUpliftCouncilForGlobalConsole:
    """Tests for global console tile summarization."""
    
    def test_tile_green_status(self):
        """Test GREEN status light when all OK."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        assert tile["tile_type"] == "uplift_council"
        assert tile["status_light"] == "GREEN"
        assert tile["budget_status"] == "OK"
        assert len(tile["critical_slices_blocked"]) == 0
    
    def test_tile_red_status(self):
        """Test RED status light when critical slice blocked."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        assert tile["status_light"] == "RED"
        assert tile["budget_status"] == "BLOCK"
        assert "slice_uplift_goal" in tile["critical_slices_blocked"]
        assert "slice_uplift_goal" in tile["blocked_slices"]
    
    def test_tile_yellow_status(self):
        """Test YELLOW status light when WARN."""
        council_view = {
            "council_status": "WARN",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        assert tile["status_light"] == "YELLOW"
        assert tile["budget_status"] == "WARN"
    
    def test_tile_headline_with_blocked(self):
        """Test headline when slices are blocked."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": ["slice_uplift_sparse"],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        assert "1 slice(s) blocked" in tile["headline"]
        assert "1 ready" in tile["headline"]
    
    def test_tile_headline_all_ready(self):
        """Test headline when all slices ready."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        assert "2 slice(s) ready for uplift" in tile["headline"]


class TestReadOnlyInvariant:
    """Tests to ensure read-only behavior."""
    
    def test_council_view_read_only(self):
        """Test that council view doesn't modify input data."""
        budget_view = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
            ]
        }
        budget_view_copy = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "health_status": "SAFE"},
            ]
        }
        
        result = build_uplift_council_view(budget_view, None, None)
        
        # Input should be unchanged
        assert budget_view == budget_view_copy
    
    def test_console_tile_read_only(self):
        """Test that console tile doesn't modify council view."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
        }
        council_view_copy = dict(council_view)
        
        tile = summarize_uplift_council_for_global_console(council_view)
        
        # Input should be unchanged
        assert council_view == council_view_copy

