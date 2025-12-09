#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Test Suite for Synthetic vs Real Governance & RFL-Coupled Scenario Control
---------------------------------------------------------------------------

Tests for:
    - Synthetic vs Real Consistency View (Task 1)
    - RFL-Coupled Scenario Policy (Task 2)
    - Director Synthetic Panel (Task 3)

NOT derived from real derivations; NOT part of Evidence Pack.

==============================================================================
"""

import pytest
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.synthetic_real_governance import (
    SCHEMA_VERSION,
    ConsistencyStatus,
    PolicyStatus,
    StatusLight,
    build_synthetic_real_consistency_view,
    derive_synthetic_scenario_policy,
    build_synthetic_director_panel,
    build_complete_governance_view,
    format_director_panel,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_synthetic_timeline():
    """Mock synthetic timeline."""
    return {
        "per_scenario": {
            "synthetic_consistent": {
                "times_passed": 9,
                "times_failed": 1,
                "pass_rate": 0.9,
                "violated_checks_history": [],
            },
            "synthetic_aggressive": {
                "times_passed": 5,
                "times_failed": 5,
                "pass_rate": 0.5,
                "violated_checks_history": [],
            },
            "synthetic_under": {
                "times_passed": 10,
                "times_failed": 0,
                "pass_rate": 1.0,
                "violated_checks_history": [],
            },
        },
        "global": {
            "scenario_count": 3,
            "total_snapshots": 30,
            "envelope_breach_rate": 0.2,
            "scenarios_with_repeated_breaches": [],
        },
    }


@pytest.fixture
def mock_real_topology_health():
    """Mock real topology health."""
    return {
        "envelope_breach_rate": 0.1,
        "has_temporal_drift": True,
    }


@pytest.fixture
def mock_real_metric_health():
    """Mock real metric health."""
    return {
        "variance_level": "moderate",
        "correlation_level": "low",
    }


@pytest.fixture
def mock_realism_summary():
    """Mock realism summary."""
    return {
        "realism_ok": True,
        "status": "OK",
        "envelope_breach_rate": 0.1,
        "scenarios_needing_review": [],
    }


# ==============================================================================
# TASK 1: SYNTHETIC vs REAL CONSISTENCY VIEW TESTS
# ==============================================================================

class TestSyntheticRealConsistencyView:
    """Tests for build_synthetic_real_consistency_view."""
    
    def test_consistency_view_has_required_fields(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Consistency view should have all required fields."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        assert "schema_version" in view
        assert "scenarios_consistent_with_real" in view
        assert "scenarios_more_aggressive_than_real" in view
        assert "scenarios_under_exploring" in view
        assert "consistency_status" in view
    
    def test_consistency_view_schema_version(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Consistency view should have correct schema version."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        assert view["schema_version"] == SCHEMA_VERSION
    
    def test_classifies_consistent_scenarios(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should classify scenarios with similar breach rates as consistent."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        # synthetic_consistent has 10% breach rate, real has 10% -> consistent
        assert "synthetic_consistent" in view["scenarios_consistent_with_real"]
    
    def test_classifies_aggressive_scenarios(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should classify scenarios with higher breach rates as aggressive."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        # synthetic_aggressive has 50% breach rate, real has 10% -> aggressive
        assert "synthetic_aggressive" in view["scenarios_more_aggressive_than_real"]
    
    def test_classifies_under_exploring_scenarios(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should classify scenarios with much lower breach rates as under-exploring."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        # synthetic_under has 0% breach rate, real has 10% -> under-exploring
        assert "synthetic_under" in view["scenarios_under_exploring"]
    
    def test_consistency_status_aligned(self):
        """Should return ALIGNED when most scenarios are consistent."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},
                "s2": {"times_passed": 8, "times_failed": 2, "pass_rate": 0.8},
                "s3": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},
            },
            "global": {"envelope_breach_rate": 0.1},
        }
        
        real_topology = {"envelope_breach_rate": 0.1}
        real_metrics = {}
        
        view = build_synthetic_real_consistency_view(timeline, real_topology, real_metrics)
        
        assert view["consistency_status"] == ConsistencyStatus.ALIGNED.value
    
    def test_consistency_status_partial(self):
        """Should return PARTIAL when some scenarios are consistent."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},  # 10% breach, consistent (diff=0.0)
                "s2": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},  # 10% breach, consistent (diff=0.0)
                "s3": {"times_passed": 5, "times_failed": 5, "pass_rate": 0.5},  # 50% breach, aggressive (diff=0.4)
            },
            "global": {"envelope_breach_rate": 0.2},
        }
        
        real_topology = {"envelope_breach_rate": 0.1}
        real_metrics = {}
        
        view = build_synthetic_real_consistency_view(timeline, real_topology, real_metrics)
        
        # 2 out of 3 consistent = 0.67, which is >= 0.4 and < 0.7 -> PARTIAL
        assert view["consistency_status"] == ConsistencyStatus.PARTIAL.value
    
    def test_consistency_status_misaligned(self):
        """Should return MISALIGNED when few scenarios are consistent."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 5, "times_failed": 5, "pass_rate": 0.5},
                "s2": {"times_passed": 5, "times_failed": 5, "pass_rate": 0.5},
                "s3": {"times_passed": 10, "times_failed": 0, "pass_rate": 1.0},
            },
            "global": {"envelope_breach_rate": 0.3},
        }
        
        real_topology = {"envelope_breach_rate": 0.1}
        real_metrics = {}
        
        view = build_synthetic_real_consistency_view(timeline, real_topology, real_metrics)
        
        assert view["consistency_status"] == ConsistencyStatus.MISALIGNED.value
    
    def test_includes_safety_label(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Consistency view should include safety label."""
        view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        assert view["label"] == SAFETY_LABEL


# ==============================================================================
# TASK 2: RFL-COUPLED SCENARIO POLICY TESTS
# ==============================================================================

class TestRFLScenarioPolicy:
    """Tests for derive_synthetic_scenario_policy."""
    
    def test_policy_has_required_fields(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Policy should have all required fields."""
        consistency_view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        policy = derive_synthetic_scenario_policy(consistency_view, mock_synthetic_timeline)
        
        assert "scenarios_recommended_for_rfl_experiments" in policy
        assert "scenarios_needing_tuning" in policy
        assert "status" in policy
        assert "policy_notes" in policy
    
    def test_policy_recommends_consistent_scenarios(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should recommend consistent scenarios with good pass rates."""
        consistency_view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        policy = derive_synthetic_scenario_policy(consistency_view, mock_synthetic_timeline)
        
        # synthetic_consistent has 90% pass rate and is consistent
        assert "synthetic_consistent" in policy["scenarios_recommended_for_rfl_experiments"]
    
    def test_policy_identifies_tuning_needed(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should identify scenarios needing tuning."""
        consistency_view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        policy = derive_synthetic_scenario_policy(consistency_view, mock_synthetic_timeline)
        
        # synthetic_under is under-exploring -> needs tuning
        assert "synthetic_under" in policy["scenarios_needing_tuning"]
    
    def test_policy_status_ok(self):
        """Should return OK status when everything is good."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},
            },
            "global": {"envelope_breach_rate": 0.05, "scenarios_with_repeated_breaches": []},
        }
        
        consistency_view = {
            "scenarios_consistent_with_real": ["s1"],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": [],
            "consistency_status": "ALIGNED",
        }
        
        policy = derive_synthetic_scenario_policy(consistency_view, timeline)
        
        assert policy["status"] == PolicyStatus.OK.value
    
    def test_policy_status_attention(self):
        """Should return ATTENTION status when there are moderate issues."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 8, "times_failed": 2, "pass_rate": 0.8},
            },
            "global": {"envelope_breach_rate": 0.15, "scenarios_with_repeated_breaches": []},
        }
        
        consistency_view = {
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["s1"],
            "scenarios_under_exploring": [],
            "consistency_status": "PARTIAL",
        }
        
        policy = derive_synthetic_scenario_policy(consistency_view, timeline)
        
        assert policy["status"] == PolicyStatus.ATTENTION.value
    
    def test_policy_status_block(self):
        """Should return BLOCK status when there are serious issues."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 5, "times_failed": 5, "pass_rate": 0.5},
            },
            "global": {"envelope_breach_rate": 0.4, "scenarios_with_repeated_breaches": []},
        }
        
        consistency_view = {
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["s1"],
            "scenarios_under_exploring": [],
            "consistency_status": "MISALIGNED",
        }
        
        policy = derive_synthetic_scenario_policy(consistency_view, timeline)
        
        assert policy["status"] == PolicyStatus.BLOCK.value
    
    def test_policy_notes_present(self):
        """Policy should include neutral policy notes."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 9, "times_failed": 1, "pass_rate": 0.9},
            },
            "global": {"envelope_breach_rate": 0.05, "scenarios_with_repeated_breaches": []},
        }
        
        consistency_view = {
            "scenarios_consistent_with_real": ["s1"],
            "scenarios_more_aggressive_than_real": [],
            "scenarios_under_exploring": [],
            "consistency_status": "ALIGNED",
        }
        
        policy = derive_synthetic_scenario_policy(consistency_view, timeline)
        
        assert "policy_notes" in policy
        assert len(policy["policy_notes"]) > 0
        # Should not contain uplift claims
        assert "uplift" not in policy["policy_notes"].lower()
    
    def test_policy_includes_repeated_breaches_in_tuning(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Scenarios with repeated breaches should be in tuning list."""
        timeline = {
            "per_scenario": {
                "s1": {"times_passed": 5, "times_failed": 5, "pass_rate": 0.5},
            },
            "global": {
                "envelope_breach_rate": 0.5,
                "scenarios_with_repeated_breaches": ["s1"],
            },
        }
        
        consistency_view = {
            "scenarios_consistent_with_real": [],
            "scenarios_more_aggressive_than_real": ["s1"],
            "scenarios_under_exploring": [],
            "consistency_status": "PARTIAL",
        }
        
        policy = derive_synthetic_scenario_policy(consistency_view, timeline)
        
        assert "s1" in policy["scenarios_needing_tuning"]
    
    def test_policy_includes_safety_label(
        self,
        mock_synthetic_timeline,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Policy should include safety label."""
        consistency_view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        policy = derive_synthetic_scenario_policy(consistency_view, mock_synthetic_timeline)
        
        assert policy["label"] == SAFETY_LABEL


# ==============================================================================
# TASK 3: DIRECTOR SYNTHETIC PANEL TESTS
# ==============================================================================

class TestDirectorSyntheticPanel:
    """Tests for build_synthetic_director_panel."""
    
    def test_panel_has_required_fields(
        self,
        mock_synthetic_timeline,
        mock_realism_summary,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Panel should have all required fields."""
        consistency_view = build_synthetic_real_consistency_view(
            mock_synthetic_timeline,
            mock_real_topology_health,
            mock_real_metric_health,
        )
        
        policy = derive_synthetic_scenario_policy(consistency_view, mock_synthetic_timeline)
        
        panel = build_synthetic_director_panel(
            mock_realism_summary,
            consistency_view,
            policy,
        )
        
        assert "status_light" in panel
        assert "realism_ok" in panel
        assert "consistency_status" in panel
        assert "scenarios_needing_review" in panel
        assert "headline" in panel
    
    def test_status_light_green(self):
        """Should return GREEN when everything is OK."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["status_light"] == StatusLight.GREEN.value
    
    def test_status_light_yellow(self):
        """Should return YELLOW when there are warnings."""
        realism_summary = {
            "realism_ok": False,
            "status": "WARN",
        }
        
        consistency_view = {
            "consistency_status": "PARTIAL",
        }
        
        policy = {
            "status": "ATTENTION",
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["status_light"] == StatusLight.YELLOW.value
    
    def test_status_light_red_block(self):
        """Should return RED when there is a BLOCK status."""
        realism_summary = {
            "realism_ok": True,
            "status": "BLOCK",
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["status_light"] == StatusLight.RED.value
    
    def test_status_light_red_misaligned(self):
        """Should return RED when consistency is MISALIGNED."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
        }
        
        consistency_view = {
            "consistency_status": "MISALIGNED",
        }
        
        policy = {
            "status": "OK",
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["status_light"] == StatusLight.RED.value
    
    def test_panel_aggregates_scenarios_needing_review(self):
        """Should aggregate scenarios from all sources."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
            "scenarios_needing_review": ["s1", "s2"],
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": ["s2", "s3"],
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        # Should include all unique scenarios
        assert set(panel["scenarios_needing_review"]) == {"s1", "s2", "s3"}
    
    def test_headline_for_green(self):
        """Headline should reflect GREEN status."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
            "scenarios_needing_tuning": [],
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert "operational" in panel["headline"].lower()
    
    def test_headline_for_yellow(self):
        """Headline should reflect YELLOW status."""
        realism_summary = {
            "realism_ok": False,
            "status": "WARN",
        }
        
        consistency_view = {
            "consistency_status": "PARTIAL",
        }
        
        policy = {
            "status": "ATTENTION",
            "scenarios_needing_tuning": ["s1"],
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert "attention" in panel["headline"].lower()
    
    def test_headline_for_red(self):
        """Headline should reflect RED status."""
        realism_summary = {
            "realism_ok": True,
            "status": "BLOCK",
        }
        
        consistency_view = {
            "consistency_status": "MISALIGNED",
        }
        
        policy = {
            "status": "BLOCK",
            "scenarios_needing_tuning": ["s1"],
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert "blocked" in panel["headline"].lower()
    
    def test_panel_includes_recommended_scenarios(self):
        """Panel should include recommended scenarios from policy."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
            "scenarios_recommended_for_rfl_experiments": ["s1", "s2"],
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["recommended_scenarios"] == ["s1", "s2"]
    
    def test_panel_includes_safety_label(self):
        """Panel should include safety label."""
        realism_summary = {
            "realism_ok": True,
            "status": "OK",
        }
        
        consistency_view = {
            "consistency_status": "ALIGNED",
        }
        
        policy = {
            "status": "OK",
        }
        
        panel = build_synthetic_director_panel(realism_summary, consistency_view, policy)
        
        assert panel["label"] == SAFETY_LABEL


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests for complete governance view."""
    
    def test_build_complete_governance_view(
        self,
        mock_synthetic_timeline,
        mock_realism_summary,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Should build complete governance view with all components."""
        view = build_complete_governance_view(
            synthetic_timeline=mock_synthetic_timeline,
            realism_summary=mock_realism_summary,
            real_topology_health=mock_real_topology_health,
            real_metric_health=mock_real_metric_health,
        )
        
        assert "realism_summary" in view
        assert "consistency_view" in view
        assert "scenario_policy" in view
        assert "director_panel" in view
    
    def test_format_director_panel(self):
        """Should format director panel as readable text."""
        panel = {
            "status_light": "GREEN",
            "realism_ok": True,
            "consistency_status": "ALIGNED",
            "scenarios_needing_review": [],
            "headline": "Test headline.",
            "realism_status": "OK",
            "policy_status": "OK",
            "recommended_scenarios": ["s1"],
        }
        
        formatted = format_director_panel(panel)
        
        assert SAFETY_LABEL in formatted
        assert "SYNTHETIC UNIVERSE DIRECTOR PANEL" in formatted
        assert "GREEN" in formatted
        assert "Test headline" in formatted
    
    def test_complete_view_includes_safety_label(
        self,
        mock_synthetic_timeline,
        mock_realism_summary,
        mock_real_topology_health,
        mock_real_metric_health,
    ):
        """Complete view should include safety label."""
        view = build_complete_governance_view(
            synthetic_timeline=mock_synthetic_timeline,
            realism_summary=mock_realism_summary,
            real_topology_health=mock_real_topology_health,
            real_metric_health=mock_real_metric_health,
        )
        
        assert view["label"] == SAFETY_LABEL


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

