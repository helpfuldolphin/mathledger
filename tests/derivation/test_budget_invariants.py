"""
Tests for Phase III Budget Invariant Governance Layer.

Tests cover:
    1. Invariant snapshot contract (build_budget_invariant_snapshot)
    2. Cross-run timeline aggregation (build_budget_invariant_timeline)
    3. Global health / MAAS summary (summarize_budget_invariants_for_global_health)
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict

from derivation.budget_invariants import (
    SCHEMA_VERSION,
    REQUIRED_BUDGET_FIELDS,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    summarize_budget_invariants_for_global_health,
    build_budget_invariants_governance_view,
    evaluate_budget_release_readiness,
    build_budget_invariants_director_panel,
    build_budget_storyline,
    explain_budget_release_decision,
)


# ---------------------------------------------------------------------------
# Mock PipelineStats for testing
# ---------------------------------------------------------------------------

@dataclass
class MockPipelineStats:
    """Mock PipelineStats for testing invariant snapshot."""
    post_exhaustion_candidates: int = 0
    max_candidates_hit: bool = False
    candidates_considered: int = 10
    budget_remaining_s: float = 1.0
    timeout_abstentions: int = 0


# ---------------------------------------------------------------------------
# Test: build_budget_invariant_snapshot
# ---------------------------------------------------------------------------

class TestBuildBudgetInvariantSnapshot:
    """Tests for the invariant snapshot contract."""
    
    def test_snapshot_schema_version(self):
        """Snapshot includes correct schema version."""
        stats = MockPipelineStats()
        snapshot = build_budget_invariant_snapshot(stats)
        assert snapshot["schema_version"] == SCHEMA_VERSION
    
    def test_snapshot_all_invariants_ok(self):
        """All invariants pass with clean stats."""
        stats = MockPipelineStats()
        budget_section = {f: 0 for f in REQUIRED_BUDGET_FIELDS}
        
        snapshot = build_budget_invariant_snapshot(
            stats, max_candidates_limit=100, budget_section=budget_section
        )
        
        assert snapshot["inv_bud_1_ok"] is True
        assert snapshot["inv_bud_2_ok"] is True
        assert snapshot["inv_bud_3_ok"] is True
        assert snapshot["inv_bud_4_ok"] is True
        assert snapshot["inv_bud_5_ok"] is True
        assert snapshot["summary_status"] == "OK"
    
    def test_snapshot_inv_bud_1_failure(self):
        """INV-BUD-1 fails when post_exhaustion_candidates > 0."""
        stats = MockPipelineStats(post_exhaustion_candidates=5)
        snapshot = build_budget_invariant_snapshot(stats)
        
        assert snapshot["inv_bud_1_ok"] is False
        assert snapshot["summary_status"] == "FAIL"
        assert snapshot["post_exhaustion_candidates"] == 5
    
    def test_snapshot_inv_bud_2_failure(self):
        """INV-BUD-2 fails when candidates exceed limit."""
        stats = MockPipelineStats(
            max_candidates_hit=True,
            candidates_considered=15,
        )
        snapshot = build_budget_invariant_snapshot(stats, max_candidates_limit=10)
        
        assert snapshot["inv_bud_2_ok"] is False
        assert snapshot["summary_status"] == "FAIL"
    
    def test_snapshot_inv_bud_2_ok_when_within_limit(self):
        """INV-BUD-2 passes when candidates within limit."""
        stats = MockPipelineStats(
            max_candidates_hit=True,
            candidates_considered=10,
        )
        snapshot = build_budget_invariant_snapshot(stats, max_candidates_limit=10)
        
        assert snapshot["inv_bud_2_ok"] is True
    
    def test_snapshot_inv_bud_3_failure_negative(self):
        """INV-BUD-3 fails when remaining_budget_s is negative (not -1)."""
        stats = MockPipelineStats(budget_remaining_s=-0.5)
        snapshot = build_budget_invariant_snapshot(stats)
        
        assert snapshot["inv_bud_3_ok"] is False
        assert snapshot["summary_status"] == "FAIL"
    
    def test_snapshot_inv_bud_3_ok_minus_one(self):
        """INV-BUD-3 passes when remaining_budget_s is -1 (no budget)."""
        stats = MockPipelineStats(budget_remaining_s=-1.0)
        snapshot = build_budget_invariant_snapshot(stats)
        
        assert snapshot["inv_bud_3_ok"] is True
    
    def test_snapshot_inv_bud_4_failure_missing_fields(self):
        """INV-BUD-4 fails when budget section missing required fields."""
        stats = MockPipelineStats()
        incomplete_budget = {"cycle_budget_s": 5.0}  # Missing most fields
        
        snapshot = build_budget_invariant_snapshot(
            stats, budget_section=incomplete_budget
        )
        
        assert snapshot["inv_bud_4_ok"] is False
        assert snapshot["summary_status"] == "WARN"  # Soft invariant
    
    def test_snapshot_warn_on_timeout_abstentions(self):
        """Snapshot returns WARN when timeouts occurred."""
        stats = MockPipelineStats(timeout_abstentions=3)
        budget_section = {f: 0 for f in REQUIRED_BUDGET_FIELDS}
        
        snapshot = build_budget_invariant_snapshot(
            stats, budget_section=budget_section
        )
        
        assert snapshot["inv_bud_1_ok"] is True
        assert snapshot["summary_status"] == "WARN"
        assert snapshot["timeout_abstentions"] == 3
    
    def test_snapshot_is_json_serializable(self):
        """Snapshot can be serialized to JSON."""
        import json
        stats = MockPipelineStats()
        snapshot = build_budget_invariant_snapshot(stats)
        
        # Should not raise
        json_str = json.dumps(snapshot)
        assert isinstance(json_str, str)
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed == snapshot


# ---------------------------------------------------------------------------
# Test: build_budget_invariant_timeline
# ---------------------------------------------------------------------------

class TestBuildBudgetInvariantTimeline:
    """Tests for cross-run invariant timeline."""
    
    def test_timeline_empty_snapshots(self):
        """Empty snapshot list produces zero counts."""
        timeline = build_budget_invariant_timeline([])
        
        assert timeline["total_runs"] == 0
        assert timeline["ok_count"] == 0
        assert timeline["recent_status"] == "UNKNOWN"
        assert timeline["stability_index"] == 0.0
    
    def test_timeline_single_ok_snapshot(self):
        """Single OK snapshot produces correct timeline."""
        snapshots = [
            {"summary_status": "OK", "inv_bud_1_ok": True, "inv_bud_2_ok": True,
             "inv_bud_3_ok": True, "inv_bud_4_ok": True, "inv_bud_5_ok": True,
             "timeout_abstentions": 0}
        ]
        timeline = build_budget_invariant_timeline(snapshots)
        
        assert timeline["total_runs"] == 1
        assert timeline["ok_count"] == 1
        assert timeline["fail_count"] == 0
        assert timeline["recent_status"] == "OK"
        assert timeline["stability_index"] == 1.0
    
    def test_timeline_mixed_snapshots(self):
        """Mixed OK/WARN/FAIL snapshots aggregate correctly."""
        snapshots = [
            {"summary_status": "OK", "inv_bud_1_ok": True, "inv_bud_2_ok": True,
             "inv_bud_3_ok": True, "inv_bud_4_ok": True, "inv_bud_5_ok": True,
             "timeout_abstentions": 0},
            {"summary_status": "WARN", "inv_bud_1_ok": True, "inv_bud_2_ok": True,
             "inv_bud_3_ok": True, "inv_bud_4_ok": False, "inv_bud_5_ok": True,
             "timeout_abstentions": 5},
            {"summary_status": "FAIL", "inv_bud_1_ok": False, "inv_bud_2_ok": True,
             "inv_bud_3_ok": True, "inv_bud_4_ok": True, "inv_bud_5_ok": True,
             "timeout_abstentions": 0},
            {"summary_status": "OK", "inv_bud_1_ok": True, "inv_bud_2_ok": True,
             "inv_bud_3_ok": True, "inv_bud_4_ok": True, "inv_bud_5_ok": True,
             "timeout_abstentions": 0},
        ]
        timeline = build_budget_invariant_timeline(snapshots)
        
        assert timeline["total_runs"] == 4
        assert timeline["ok_count"] == 2
        assert timeline["warn_count"] == 1
        assert timeline["fail_count"] == 1
        assert timeline["inv_bud_1_failures"] == 1
        assert timeline["inv_bud_4_failures"] == 1
        assert timeline["recent_status"] == "OK"  # Last snapshot
        assert timeline["stability_index"] == 0.5  # 2/4
        assert timeline["timeout_abstention_runs"] == 1
    
    def test_timeline_tracks_all_invariant_failures(self):
        """Timeline correctly counts failures for each invariant."""
        snapshots = [
            {"summary_status": "FAIL", "inv_bud_1_ok": False, "inv_bud_2_ok": False,
             "inv_bud_3_ok": False, "inv_bud_4_ok": False, "inv_bud_5_ok": False,
             "timeout_abstentions": 0},
        ]
        timeline = build_budget_invariant_timeline(snapshots)
        
        assert timeline["inv_bud_1_failures"] == 1
        assert timeline["inv_bud_2_failures"] == 1
        assert timeline["inv_bud_3_failures"] == 1
        assert timeline["inv_bud_4_failures"] == 1
        assert timeline["inv_bud_5_failures"] == 1
    
    def test_timeline_recent_status_is_last_snapshot(self):
        """Recent status reflects the last snapshot in sequence."""
        snapshots = [
            {"summary_status": "OK", "inv_bud_1_ok": True, "timeout_abstentions": 0},
            {"summary_status": "FAIL", "inv_bud_1_ok": False, "timeout_abstentions": 0},
        ]
        timeline = build_budget_invariant_timeline(snapshots)
        
        assert timeline["recent_status"] == "FAIL"


# ---------------------------------------------------------------------------
# Test: summarize_budget_invariants_for_global_health
# ---------------------------------------------------------------------------

class TestSummarizeBudgetInvariantsForGlobalHealth:
    """Tests for MAAS / global health integration."""
    
    def test_global_health_all_ok(self):
        """All OK timeline produces OK status."""
        timeline = {
            "total_runs": 10,
            "ok_count": 10,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
            "stability_index": 1.0,
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["invariants_ok"] is True
        assert summary["status"] == "OK"
        assert summary["inv_bud_failures"] == []
        assert summary["recent_status"] == "OK"
    
    def test_global_health_hard_invariant_failure_blocks(self):
        """Hard invariant failure (1, 2, 3) produces BLOCK status."""
        timeline = {
            "total_runs": 10,
            "inv_bud_1_failures": 1,  # Hard invariant
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "FAIL",
            "stability_index": 0.9,
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["invariants_ok"] is False
        assert summary["status"] == "BLOCK"
        assert "INV-BUD-1" in summary["inv_bud_failures"]
    
    def test_global_health_soft_invariant_failure_warns(self):
        """Soft invariant failure (4, 5) produces WARN status."""
        timeline = {
            "total_runs": 10,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 2,  # Soft invariant
            "inv_bud_5_failures": 0,
            "recent_status": "WARN",
            "stability_index": 0.95,
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["invariants_ok"] is False
        assert summary["status"] == "WARN"
        assert "INV-BUD-4" in summary["inv_bud_failures"]
    
    def test_global_health_low_stability_warns(self):
        """Low stability (<0.9) produces WARN status."""
        timeline = {
            "total_runs": 10,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "WARN",
            "stability_index": 0.85,  # Below 0.9 threshold
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["invariants_ok"] is True  # No failures
        assert summary["status"] == "WARN"  # But low stability
    
    def test_global_health_lists_all_failures(self):
        """All failed invariants are listed."""
        timeline = {
            "total_runs": 5,
            "inv_bud_1_failures": 1,
            "inv_bud_2_failures": 2,
            "inv_bud_3_failures": 3,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 1,
            "recent_status": "FAIL",
            "stability_index": 0.0,
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert set(summary["inv_bud_failures"]) == {
            "INV-BUD-1", "INV-BUD-2", "INV-BUD-3", "INV-BUD-5"
        }
        assert summary["status"] == "BLOCK"  # Hard invariant failures
    
    def test_global_health_preserves_metadata(self):
        """Summary includes total_runs and stability_index."""
        timeline = {
            "total_runs": 42,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
            "stability_index": 0.95,
        }
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["total_runs"] == 42
        assert summary["stability_index"] == 0.95


# ---------------------------------------------------------------------------
# Test: Integration / End-to-End
# ---------------------------------------------------------------------------

class TestInvariantGovernanceIntegration:
    """Integration tests for the full governance pipeline."""
    
    def test_snapshot_to_timeline_to_global_health(self):
        """Full pipeline: snapshot → timeline → global health."""
        # Create several snapshots
        stats_ok = MockPipelineStats()
        stats_fail = MockPipelineStats(post_exhaustion_candidates=1)
        stats_warn = MockPipelineStats(timeout_abstentions=5)
        
        budget_section = {f: 0 for f in REQUIRED_BUDGET_FIELDS}
        
        snapshots = [
            build_budget_invariant_snapshot(stats_ok, budget_section=budget_section),
            build_budget_invariant_snapshot(stats_ok, budget_section=budget_section),
            build_budget_invariant_snapshot(stats_fail),
            build_budget_invariant_snapshot(stats_ok, budget_section=budget_section),
            build_budget_invariant_snapshot(stats_warn, budget_section=budget_section),
        ]
        
        # Build timeline
        timeline = build_budget_invariant_timeline(snapshots)
        
        assert timeline["total_runs"] == 5
        assert timeline["inv_bud_1_failures"] == 1
        
        # Build global health summary
        summary = summarize_budget_invariants_for_global_health(timeline)
        
        assert summary["status"] == "BLOCK"  # INV-BUD-1 is a hard invariant
        assert "INV-BUD-1" in summary["inv_bud_failures"]


# ---------------------------------------------------------------------------
# Test: Phase IV Cross-Layer Governance
# ---------------------------------------------------------------------------

class TestBuildBudgetInvariantsGovernanceView:
    """Tests for cross-layer governance view."""
    
    def test_governance_view_combines_invariants_and_health(self):
        """Governance view correctly combines invariant timeline and budget health."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.95,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        budget_health = {
            "health_score": 85.0,
            "trend_status": "STABLE",
        }
        
        view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert view["schema_version"] == SCHEMA_VERSION
        assert view["total_runs"] == 100
        assert view["stability_index"] == 0.95
        assert view["health_score"] == 85.0
        assert view["invariants_status"] == "OK"
        assert view["combined_status"] == "OK"
    
    def test_governance_view_block_on_invariant_block(self):
        """BLOCK status when invariants_status is BLOCK."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.90,
            "inv_bud_1_failures": 5,  # Hard invariant failure
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "FAIL",
        }
        
        budget_health = {
            "health_score": 90.0,
            "trend_status": "STABLE",
        }
        
        view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert view["invariants_status"] == "BLOCK"
        assert view["combined_status"] == "BLOCK"  # Invariant BLOCK takes precedence
    
    def test_governance_view_warn_on_degrading_trend(self):
        """WARN status when trend_status is DEGRADING."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.95,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        budget_health = {
            "health_score": 80.0,
            "trend_status": "DEGRADING",
        }
        
        view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert view["invariants_status"] == "OK"
        assert view["combined_status"] == "WARN"  # DEGRADING trend triggers WARN
    
    def test_governance_view_warn_on_invariant_warn(self):
        """WARN status when invariants_status is WARN."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.85,  # Below 0.9 threshold
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 2,  # Soft invariant failure
            "inv_bud_5_failures": 0,
            "recent_status": "WARN",
        }
        
        budget_health = {
            "health_score": 85.0,
            "trend_status": "STABLE",
        }
        
        view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert view["invariants_status"] == "WARN"
        assert view["combined_status"] == "WARN"
    
    def test_governance_view_includes_inv_bud_failures(self):
        """Governance view includes list of failed invariants."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.95,
            "inv_bud_1_failures": 3,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 1,
            "inv_bud_5_failures": 0,
            "recent_status": "FAIL",
        }
        
        budget_health = {
            "health_score": 85.0,
            "trend_status": "STABLE",
        }
        
        view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert "INV-BUD-1" in view["inv_bud_failures"]
        assert "INV-BUD-4" in view["inv_bud_failures"]


class TestEvaluateBudgetReleaseReadiness:
    """Tests for release readiness evaluation."""
    
    def test_release_readiness_ok(self):
        """Release OK when all conditions met."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.98,
            "health_score": 90.0,
            "inv_bud_failures": [],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is True
        assert readiness["status"] == "OK"
        assert len(readiness["blocking_reasons"]) == 0
    
    def test_release_readiness_block_on_block_status(self):
        """Release BLOCK when combined_status is BLOCK."""
        governance_view = {
            "combined_status": "BLOCK",
            "invariants_status": "BLOCK",
            "stability_index": 0.90,
            "health_score": 85.0,
            "inv_bud_failures": ["INV-BUD-1"],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is False
        assert readiness["status"] == "BLOCK"
        assert len(readiness["blocking_reasons"]) > 0
        assert any("BLOCK" in reason for reason in readiness["blocking_reasons"])
    
    def test_release_readiness_block_on_low_health_score(self):
        """Release BLOCK when health_score < 70."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.98,
            "health_score": 65.0,  # Below 70 threshold
            "inv_bud_failures": [],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is False
        assert readiness["status"] == "BLOCK"
        assert any("health_score < 70" in reason for reason in readiness["blocking_reasons"])
    
    def test_release_readiness_warn_on_low_stability(self):
        """Release WARN (not blocking) when stability_index < 0.95."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.92,  # Below 0.95 threshold
            "health_score": 85.0,
            "inv_bud_failures": [],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is True  # WARN doesn't block
        assert readiness["status"] == "WARN"
        assert any("stability_index < 0.95" in reason for reason in readiness["blocking_reasons"])
    
    def test_release_readiness_includes_invariant_failures(self):
        """Blocking reasons include invariant failure details."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.98,
            "health_score": 85.0,
            "inv_bud_failures": ["INV-BUD-2", "INV-BUD-4"],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        # Should include invariant failures in reasons (even if not blocking)
        assert any("INV-BUD-2" in reason for reason in readiness["blocking_reasons"])
        assert any("INV-BUD-4" in reason for reason in readiness["blocking_reasons"])
    
    def test_release_readiness_warn_status(self):
        """WARN status from governance view produces WARN readiness."""
        governance_view = {
            "combined_status": "WARN",
            "invariants_status": "WARN",
            "stability_index": 0.96,
            "health_score": 85.0,
            "inv_bud_failures": [],
        }
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is True  # WARN doesn't block
        assert readiness["status"] == "WARN"


class TestBuildBudgetInvariantsDirectorPanel:
    """Tests for Director Console panel."""
    
    def test_director_panel_green_light(self):
        """GREEN light when combined_status is OK."""
        governance_view = {
            "combined_status": "OK",
            "recent_status": "OK",
            "stability_index": 0.98,
            "health_score": 90.0,
            "inv_bud_failures": [],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert panel["status_light"] == "GREEN"
        assert panel["recent_status"] == "OK"
        assert panel["stability_index"] == 0.98
        assert panel["health_score"] == 90.0
    
    def test_director_panel_yellow_light(self):
        """YELLOW light when combined_status is WARN."""
        governance_view = {
            "combined_status": "WARN",
            "recent_status": "WARN",
            "stability_index": 0.92,
            "health_score": 85.0,
            "inv_bud_failures": [],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": ["stability_index < 0.95"],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert panel["status_light"] == "YELLOW"
        assert "stability" in panel["headline"].lower() or "warn" in panel["headline"].lower()
    
    def test_director_panel_red_light(self):
        """RED light when combined_status is BLOCK."""
        governance_view = {
            "combined_status": "BLOCK",
            "recent_status": "FAIL",
            "stability_index": 0.90,
            "health_score": 65.0,
            "inv_bud_failures": ["INV-BUD-1"],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK", "health_score < 70"],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert panel["status_light"] == "RED"
        assert "INV-BUD-1" in panel["headline"]
        assert "INV-BUD-1" in panel["key_invariants_with_failures"]
    
    def test_director_panel_headline_neutral_language(self):
        """Headline uses neutral, factual language (no good/bad/failure/success)."""
        governance_view = {
            "combined_status": "OK",
            "recent_status": "OK",
            "stability_index": 0.98,
            "health_score": 90.0,
            "inv_bud_failures": [],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        headline = panel["headline"].lower()
        # Check that neutral words are present, not judgmental words
        assert "good" not in headline
        assert "bad" not in headline
        assert "failure" not in headline
        assert "success" not in headline
        # Should contain factual information
        assert "invariants" in headline or "budget" in headline or "stability" in headline
    
    def test_director_panel_includes_invariant_failures(self):
        """Panel includes list of invariants with failures."""
        governance_view = {
            "combined_status": "BLOCK",
            "recent_status": "FAIL",
            "stability_index": 0.95,
            "health_score": 80.0,
            "inv_bud_failures": ["INV-BUD-1", "INV-BUD-3"],
            "total_runs": 50,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK"],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert "INV-BUD-1" in panel["key_invariants_with_failures"]
        assert "INV-BUD-3" in panel["key_invariants_with_failures"]
        assert len(panel["key_invariants_with_failures"]) == 2
    
    def test_director_panel_is_json_serializable(self):
        """Director panel can be serialized to JSON."""
        import json
        
        governance_view = {
            "combined_status": "OK",
            "recent_status": "OK",
            "stability_index": 0.98,
            "health_score": 90.0,
            "inv_bud_failures": [],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed == panel
    
    def test_director_panel_headline_for_low_health(self):
        """Headline mentions health score when below threshold."""
        governance_view = {
            "combined_status": "WARN",
            "recent_status": "OK",
            "stability_index": 0.98,
            "health_score": 75.0,  # Below 80 threshold
            "inv_bud_failures": [],
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": [],
        }
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        # Headline should mention health score
        assert "health" in panel["headline"].lower() or "75" in panel["headline"]


# ---------------------------------------------------------------------------
# Test: Phase IV Full Pipeline Integration
# ---------------------------------------------------------------------------

class TestPhaseIVFullPipeline:
    """Integration test for full Phase IV pipeline."""
    
    def test_full_pipeline_snapshot_to_director_panel(self):
        """Full pipeline: snapshot → timeline → governance → readiness → panel."""
        # Create snapshots
        stats_ok = MockPipelineStats()
        stats_warn = MockPipelineStats(timeout_abstentions=3)
        
        budget_section = {f: 0 for f in REQUIRED_BUDGET_FIELDS}
        
        snapshots = [
            build_budget_invariant_snapshot(stats_ok, budget_section=budget_section),
            build_budget_invariant_snapshot(stats_ok, budget_section=budget_section),
            build_budget_invariant_snapshot(stats_warn, budget_section=budget_section),
        ]
        
        # Build timeline
        timeline = build_budget_invariant_timeline(snapshots)
        
        # Create synthetic A5 budget_health
        budget_health = {
            "health_score": 85.0,
            "trend_status": "STABLE",
        }
        
        # Build governance view
        governance_view = build_budget_invariants_governance_view(timeline, budget_health)
        
        assert governance_view["combined_status"] in ["OK", "WARN"]
        assert governance_view["health_score"] == 85.0
        
        # Evaluate release readiness
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["status"] in ["OK", "WARN"]
        assert isinstance(readiness["release_ok"], bool)
        assert isinstance(readiness["blocking_reasons"], list)
        
        # Build director panel
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert panel["status_light"] in ["GREEN", "YELLOW", "RED"]
        assert isinstance(panel["headline"], str)
        assert len(panel["headline"]) > 0
        assert "good" not in panel["headline"].lower()
        assert "bad" not in panel["headline"].lower()
        assert "failure" not in panel["headline"].lower()
        assert "success" not in panel["headline"].lower()
    
    def test_full_pipeline_block_scenario(self):
        """Full pipeline handles BLOCK scenario correctly."""
        # Create snapshot with hard invariant failure
        stats_fail = MockPipelineStats(post_exhaustion_candidates=5)
        
        snapshots = [
            build_budget_invariant_snapshot(stats_fail),
        ]
        
        timeline = build_budget_invariant_timeline(snapshots)
        
        budget_health = {
            "health_score": 90.0,
            "trend_status": "STABLE",
        }
        
        governance_view = build_budget_invariants_governance_view(timeline, budget_health)
        
        # Should be BLOCK due to INV-BUD-1 failure
        assert governance_view["combined_status"] == "BLOCK"
        
        readiness = evaluate_budget_release_readiness(governance_view)
        
        assert readiness["release_ok"] is False
        assert readiness["status"] == "BLOCK"
        assert len(readiness["blocking_reasons"]) > 0
        
        panel = build_budget_invariants_director_panel(governance_view, readiness)
        
        assert panel["status_light"] == "RED"
        assert "INV-BUD-1" in panel["key_invariants_with_failures"]


# ---------------------------------------------------------------------------
# Test: Phase V Budget Storyline & Post-Mortem
# ---------------------------------------------------------------------------

class TestBuildBudgetStoryline:
    """Tests for budget storyline builder."""
    
    def test_storyline_stable_case(self):
        """STABLE stability class when all conditions met."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.98,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        health_history = [
            {"run_index": i, "health_score": 85.0, "trend_status": "STABLE"}
            for i in range(100)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        assert storyline["schema_version"] == "1.0.0"
        assert storyline["runs_analyzed"] == 100
        assert storyline["stability_class"] == "STABLE"
        assert len(storyline["episodes"]) > 0
        assert isinstance(storyline["summary"], str)
    
    def test_storyline_drifting_case(self):
        """DRIFTING stability class when stability_index < 0.95."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.90,  # Below 0.95
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 2,  # Soft invariant failure
            "inv_bud_5_failures": 0,
            "recent_status": "WARN",
        }
        
        health_history = [
            {"run_index": i, "health_score": 75.0, "trend_status": "STABLE"}
            for i in range(100)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        assert storyline["stability_class"] == "DRIFTING"
        assert any("0.90" in event or "0.95" in event for event in storyline["structural_events"])
    
    def test_storyline_volatile_case(self):
        """VOLATILE stability class with hard invariant failures."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.80,  # Below 0.85
            "inv_bud_1_failures": 5,  # Hard invariant failure
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "FAIL",
        }
        
        health_history = [
            {"run_index": i, "health_score": 65.0, "trend_status": "DEGRADING"}
            for i in range(100)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        assert storyline["stability_class"] == "VOLATILE"
        assert any("INV-BUD-1" in event for event in storyline["structural_events"])
        assert any("0.80" in event or "0.85" in event for event in storyline["structural_events"])
    
    def test_storyline_includes_structural_events(self):
        """Storyline includes structural events for violations."""
        timeline = {
            "total_runs": 50,
            "stability_index": 0.92,
            "inv_bud_1_failures": 3,
            "inv_bud_2_failures": 2,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 1,
            "inv_bud_5_failures": 0,
            "recent_status": "FAIL",
        }
        
        health_history = [
            {"run_index": i, "health_score": 80.0, "trend_status": "STABLE"}
            for i in range(50)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        # Should include events for violations
        assert any("INV-BUD-1" in event for event in storyline["structural_events"])
        assert any("INV-BUD-2" in event for event in storyline["structural_events"])
        assert any("INV-BUD-4" in event for event in storyline["structural_events"])
    
    def test_storyline_episodes_have_correct_structure(self):
        """Episodes have correct structure with run ranges."""
        timeline = {
            "total_runs": 20,
            "stability_index": 0.95,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        # Create history with changing status
        health_history = [
            {"run_index": i, "health_score": 85.0 if i < 10 else 75.0, "trend_status": "STABLE"}
            for i in range(20)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        assert len(storyline["episodes"]) > 0
        
        for episode in storyline["episodes"]:
            assert "run_range" in episode
            assert "status" in episode
            assert episode["status"] in ["OK", "WARN", "FAIL"]
            assert "invariants_affected" in episode
            assert isinstance(episode["invariants_affected"], list)
            assert "health_score_range" in episode
            assert len(episode["health_score_range"]) == 2
            assert "description" in episode
            assert isinstance(episode["description"], str)
    
    def test_storyline_is_json_serializable(self):
        """Storyline can be serialized to JSON."""
        import json
        
        timeline = {
            "total_runs": 10,
            "stability_index": 0.95,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        health_history = [
            {"run_index": i, "health_score": 85.0, "trend_status": "STABLE"}
            for i in range(10)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        # Should not raise
        json_str = json.dumps(storyline)
        assert isinstance(json_str, str)
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed == storyline
    
    def test_storyline_summary_is_neutral(self):
        """Summary uses neutral, factual language."""
        timeline = {
            "total_runs": 100,
            "stability_index": 0.95,
            "inv_bud_1_failures": 0,
            "inv_bud_2_failures": 0,
            "inv_bud_3_failures": 0,
            "inv_bud_4_failures": 0,
            "inv_bud_5_failures": 0,
            "recent_status": "OK",
        }
        
        health_history = [
            {"run_index": i, "health_score": 85.0, "trend_status": "STABLE"}
            for i in range(100)
        ]
        
        storyline = build_budget_storyline(timeline, health_history)
        
        summary_lower = storyline["summary"].lower()
        
        # Check for neutral language
        assert "good" not in summary_lower
        assert "bad" not in summary_lower
        assert "failure" not in summary_lower or "failures" in summary_lower  # Allow "violations"
        assert "success" not in summary_lower


class TestExplainBudgetReleaseDecision:
    """Tests for release post-mortem explanation."""
    
    def test_explain_ok_decision(self):
        """OK decision explanation has minimal causes."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.98,
            "health_score": 90.0,
            "inv_bud_failures": [],
            "recent_status": "OK",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        assert explanation["decision"] == "OK"
        assert len(explanation["primary_causes"]) == 0
        assert len(explanation["contributing_factors"]) == 0
        assert len(explanation["recommended_followups"]) == 0
    
    def test_explain_block_decision_hard_invariant(self):
        """BLOCK decision explanation includes hard invariant failures."""
        governance_view = {
            "combined_status": "BLOCK",
            "invariants_status": "BLOCK",
            "stability_index": 0.95,
            "health_score": 85.0,
            "inv_bud_failures": ["INV-BUD-1", "INV-BUD-2"],
            "recent_status": "FAIL",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        assert explanation["decision"] == "BLOCK"
        assert len(explanation["primary_causes"]) > 0
        assert any("INV-BUD-1" in cause or "INV-BUD-2" in cause for cause in explanation["primary_causes"])
        assert any("BLOCK" in cause for cause in explanation["primary_causes"])
        assert len(explanation["recommended_followups"]) > 0
    
    def test_explain_block_decision_low_health(self):
        """BLOCK decision explanation includes low health_score."""
        governance_view = {
            "combined_status": "OK",
            "invariants_status": "OK",
            "stability_index": 0.98,
            "health_score": 65.0,  # Below 70 threshold
            "inv_bud_failures": [],
            "recent_status": "OK",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["health_score < 70.0"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        assert explanation["decision"] == "BLOCK"
        assert any("health_score" in cause.lower() for cause in explanation["primary_causes"])
        assert any("70" in cause for cause in explanation["primary_causes"])
        assert any("health" in followup.lower() for followup in explanation["recommended_followups"])
    
    def test_explain_warn_decision_contributing_factors(self):
        """WARN decision explanation includes contributing factors."""
        governance_view = {
            "combined_status": "WARN",
            "invariants_status": "WARN",
            "stability_index": 0.92,  # Below 0.95
            "health_score": 85.0,
            "inv_bud_failures": ["INV-BUD-4"],  # Soft invariant
            "recent_status": "WARN",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": ["stability_index < 0.95"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        assert explanation["decision"] == "WARN"
        assert any("0.95" in factor or "stability" in factor.lower() for factor in explanation["contributing_factors"])
        assert any("INV-BUD-4" in factor for factor in explanation["contributing_factors"])
    
    def test_explain_includes_recommended_followups(self):
        """Explanation includes relevant recommended follow-ups."""
        governance_view = {
            "combined_status": "BLOCK",
            "invariants_status": "BLOCK",
            "stability_index": 0.90,
            "health_score": 65.0,
            "inv_bud_failures": ["INV-BUD-1"],
            "recent_status": "FAIL",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        assert len(explanation["recommended_followups"]) > 0
        assert any("INV-BUD-1" in followup for followup in explanation["recommended_followups"])
        assert any("health" in followup.lower() for followup in explanation["recommended_followups"])
        assert any("stability" in followup.lower() for followup in explanation["recommended_followups"])
    
    def test_explain_uses_neutral_language(self):
        """Explanation uses neutral, factual language."""
        governance_view = {
            "combined_status": "BLOCK",
            "invariants_status": "BLOCK",
            "stability_index": 0.95,
            "health_score": 80.0,
            "inv_bud_failures": ["INV-BUD-1"],
            "recent_status": "FAIL",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        # Check all text fields
        all_text = " ".join([
            " ".join(explanation["primary_causes"]),
            " ".join(explanation["contributing_factors"]),
            " ".join(explanation["recommended_followups"]),
        ]).lower()
        
        assert "good" not in all_text
        assert "bad" not in all_text
        assert "failure" not in all_text or "failures" in all_text  # Allow technical terms
        assert "success" not in all_text
    
    def test_explain_is_json_serializable(self):
        """Explanation can be serialized to JSON."""
        import json
        
        governance_view = {
            "combined_status": "WARN",
            "invariants_status": "WARN",
            "stability_index": 0.92,
            "health_score": 85.0,
            "inv_bud_failures": [],
            "recent_status": "WARN",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": [],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        # Should not raise
        json_str = json.dumps(explanation)
        assert isinstance(json_str, str)
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed == explanation
    
    def test_explain_prioritizes_hard_invariants(self):
        """Primary causes prioritize hard invariants over soft."""
        governance_view = {
            "combined_status": "BLOCK",
            "invariants_status": "BLOCK",
            "stability_index": 0.95,
            "health_score": 85.0,
            "inv_bud_failures": ["INV-BUD-1", "INV-BUD-4"],  # Hard + soft
            "recent_status": "FAIL",
            "total_runs": 100,
        }
        
        readiness = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["combined_status is BLOCK"],
        }
        
        explanation = explain_budget_release_decision(governance_view, readiness)
        
        # Hard invariants should be in primary_causes
        primary_text = " ".join(explanation["primary_causes"])
        assert "INV-BUD-1" in primary_text
        
        # Soft invariants should be in contributing_factors
        contributing_text = " ".join(explanation["contributing_factors"])
        assert "INV-BUD-4" in contributing_text

