"""
Minimal test suite for budget invariant governance core functions.

Tests snapshot contract, timeline aggregation, global health summary,
governance view, and release readiness evaluation.
"""

import json
from typing import Any, Dict

import pytest

from derivation.budget_invariants import (
    BUDGET_SCHEMA_VERSION,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    build_budget_invariants_governance_view,
    evaluate_budget_release_readiness,
    summarize_budget_invariants_for_global_health,
)


# Mock PipelineStats-like objects
class MockStats:
    def __init__(
        self,
        budget_exhausted: bool = False,
        max_candidates_hit: bool = False,
        timeout_abstentions: int = 0,
        statements_skipped: int = 0,
        candidates_considered: int = 0,
        budget_remaining_s: float | None = None,
        post_exhaustion_candidates: int = 0,
    ):
        self.budget_exhausted = budget_exhausted
        self.max_candidates_hit = max_candidates_hit
        self.timeout_abstentions = timeout_abstentions
        self.statements_skipped = statements_skipped
        self.candidates_considered = candidates_considered
        self.budget_remaining_s = budget_remaining_s
        self.post_exhaustion_candidates = post_exhaustion_candidates


# ---------------------------------------------------------------------------
# Snapshot Tests
# ---------------------------------------------------------------------------


def test_snapshot_contract_all_ok():
    """Test snapshot with all invariants OK."""
    stats = MockStats(
        budget_exhausted=False,
        timeout_abstentions=0,
        statements_skipped=0,
    )
    
    snapshot = build_budget_invariant_snapshot(stats)
    
    assert snapshot["schema_version"] == BUDGET_SCHEMA_VERSION
    assert snapshot["inv_bud_1_ok"] is True
    assert snapshot["inv_bud_2_ok"] is True
    assert snapshot["inv_bud_3_ok"] is True
    assert snapshot["inv_bud_4_ok"] is True
    assert snapshot["inv_bud_5_ok"] is True
    assert snapshot["summary_status"] == "OK"
    assert snapshot["timeout_abstentions"] == 0
    assert snapshot["post_exhaustion_candidates"] == 0
    assert snapshot["max_candidates_hit"] is False


def test_snapshot_contract_inv_bud_1_failure():
    """Test snapshot with INV-BUD-1 violation (post-exhaustion processing)."""
    stats = MockStats(
        budget_exhausted=True,
        post_exhaustion_candidates=5,
    )
    
    snapshot = build_budget_invariant_snapshot(stats)
    
    assert snapshot["inv_bud_1_ok"] is False
    assert snapshot["summary_status"] == "FAIL"


def test_snapshot_contract_inv_bud_2_failure():
    """Test snapshot with INV-BUD-2 violation (max candidates hit mismatch)."""
    # Test actual hit (candidates >= max when hit flag is True)
    stats = MockStats(
        max_candidates_hit=True,
        candidates_considered=40,
    )
    
    snapshot = build_budget_invariant_snapshot(stats, max_candidates_per_cycle=40)
    
    # Should be OK when candidates >= max and hit flag is True
    assert snapshot["inv_bud_2_ok"] is True
    
    # Test mismatch: hit flag True but candidates < max (violation)
    stats2 = MockStats(
        max_candidates_hit=True,
        candidates_considered=30,
    )
    snapshot2 = build_budget_invariant_snapshot(stats2, max_candidates_per_cycle=40)
    # This is a violation: hit flag set but not actually at limit
    assert snapshot2["inv_bud_2_ok"] is False


def test_snapshot_contract_inv_bud_3_failure():
    """Test snapshot with INV-BUD-3 violation (negative budget remaining)."""
    stats = MockStats(
        budget_remaining_s=-1.0,
    )
    
    snapshot = build_budget_invariant_snapshot(stats)
    
    assert snapshot["inv_bud_3_ok"] is False
    assert snapshot["summary_status"] == "FAIL"


def test_snapshot_contract_inv_bud_4_warn():
    """Test snapshot with INV-BUD-4 (timeout abstentions) triggering WARN."""
    stats = MockStats(
        timeout_abstentions=5,
    )
    
    snapshot = build_budget_invariant_snapshot(stats)
    
    assert snapshot["inv_bud_4_ok"] is True  # Non-negative is OK
    assert snapshot["summary_status"] == "WARN"  # But triggers WARN


def test_snapshot_contract_dict_input():
    """Test snapshot accepts dict input."""
    stats_dict = {
        "budget_exhausted": False,
        "max_candidates_hit": False,
        "timeout_abstentions": 0,
        "statements_skipped": 0,
        "candidates_considered": 10,
    }
    
    snapshot = build_budget_invariant_snapshot(stats_dict)
    
    assert snapshot["summary_status"] == "OK"


def test_snapshot_json_serializable():
    """Test snapshot is JSON-serializable."""
    stats = MockStats()
    snapshot = build_budget_invariant_snapshot(stats)
    
    # Should not raise
    json_str = json.dumps(snapshot)
    assert isinstance(json_str, str)
    
    # Should round-trip
    decoded = json.loads(json_str)
    assert decoded == snapshot


# ---------------------------------------------------------------------------
# Timeline Tests
# ---------------------------------------------------------------------------


def test_timeline_empty():
    """Test timeline with empty snapshots."""
    timeline = build_budget_invariant_timeline([])
    
    assert timeline["schema_version"] == BUDGET_SCHEMA_VERSION
    assert timeline["total_runs"] == 0
    assert timeline["ok_count"] == 0
    assert timeline["recent_status"] == "OK"
    assert timeline["stability_index"] == 1.0


def test_timeline_all_ok():
    """Test timeline with all OK snapshots."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats()),
    ]
    
    timeline = build_budget_invariant_timeline(snapshots)
    
    assert timeline["total_runs"] == 3
    assert timeline["ok_count"] == 3
    assert timeline["warn_count"] == 0
    assert timeline["fail_count"] == 0
    assert timeline["recent_status"] == "OK"
    assert timeline["stability_index"] == 1.0


def test_timeline_mixed_status():
    """Test timeline with mixed OK/WARN/FAIL."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=5)),
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
        build_budget_invariant_snapshot(MockStats()),
    ]
    
    timeline = build_budget_invariant_timeline(snapshots)
    
    assert timeline["total_runs"] == 4
    assert timeline["ok_count"] == 2
    assert timeline["warn_count"] == 1
    assert timeline["fail_count"] == 1
    assert timeline["inv_bud_1_failures"] == 1


def test_timeline_recent_status_preference():
    """Test recent status prefers most severe."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
        build_budget_invariant_snapshot(MockStats()),
    ]
    
    timeline = build_budget_invariant_timeline(snapshots)
    
    # Recent (last 5) includes the FAIL, so should prefer FAIL
    assert timeline["recent_status"] == "FAIL"


def test_timeline_stability_index():
    """Test stability index calculation."""
    # All same = high stability
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    assert timeline["stability_index"] == 1.0
    
    # Mixed = lower stability
    snapshots2 = [
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=1)),
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=1)),
        build_budget_invariant_snapshot(MockStats()),
    ]
    timeline2 = build_budget_invariant_timeline(snapshots2)
    assert timeline2["stability_index"] < 1.0


def test_timeline_json_serializable():
    """Test timeline is JSON-serializable."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    
    json_str = json.dumps(timeline)
    decoded = json.loads(json_str)
    assert decoded == timeline


# ---------------------------------------------------------------------------
# Global Health Summary Tests
# ---------------------------------------------------------------------------


def test_global_health_all_ok():
    """Test global health summary with all OK."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats()),
    ])
    
    health = summarize_budget_invariants_for_global_health(timeline)
    
    assert health["schema_version"] == BUDGET_SCHEMA_VERSION
    assert health["invariants_ok"] is True
    assert health["recent_status"] == "OK"
    assert health["status"] == "OK"
    assert health["inv_bud_failures"] == []
    assert health["stability_index"] == 1.0


def test_global_health_with_failures():
    """Test global health summary with failures."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    
    health = summarize_budget_invariants_for_global_health(timeline)
    
    assert health["invariants_ok"] is False
    assert "INV-BUD-1" in health["inv_bud_failures"]
    assert health["status"] == "BLOCK"  # FAIL -> BLOCK


def test_global_health_warn_status():
    """Test global health summary with WARN status."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=5)),
    ])
    
    health = summarize_budget_invariants_for_global_health(timeline)
    
    assert health["recent_status"] == "WARN"
    assert health["status"] == "WARN"


def test_global_health_json_serializable():
    """Test global health is JSON-serializable."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
    ])
    health = summarize_budget_invariants_for_global_health(timeline)
    
    json_str = json.dumps(health)
    decoded = json.loads(json_str)
    assert decoded == health


# ---------------------------------------------------------------------------
# Governance View Tests
# ---------------------------------------------------------------------------


def test_governance_view_ok():
    """Test governance view with OK status."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
    ])
    budget_health = {
        "health_score": 90.0,
        "trend_status": "STABLE",
    }
    
    view = build_budget_invariants_governance_view(timeline, budget_health)
    
    assert view["schema_version"] == BUDGET_SCHEMA_VERSION
    assert view["combined_status"] == "OK"
    assert view["health_score"] == 90.0
    assert view["stability_index"] == 1.0


def test_governance_view_block():
    """Test governance view with BLOCK status."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {
        "health_score": 90.0,
        "trend_status": "STABLE",
    }
    
    view = build_budget_invariants_governance_view(timeline, budget_health)
    
    assert view["combined_status"] == "BLOCK"


def test_governance_view_warn_degrading():
    """Test governance view with WARN due to degrading trend."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
    ])
    budget_health = {
        "health_score": 85.0,
        "trend_status": "DEGRADING",
    }
    
    view = build_budget_invariants_governance_view(timeline, budget_health)
    
    assert view["combined_status"] == "WARN"


# ---------------------------------------------------------------------------
# Release Readiness Tests
# ---------------------------------------------------------------------------


def test_release_readiness_ok():
    """Test release readiness with OK status."""
    view = {
        "combined_status": "OK",
        "stability_index": 1.0,
        "health_score": 90.0,
    }
    
    readiness = evaluate_budget_release_readiness(view)
    
    assert readiness["release_ok"] is True
    assert readiness["status"] == "OK"
    assert readiness["blocking_reasons"] == []


def test_release_readiness_block():
    """Test release readiness with BLOCK status."""
    view = {
        "combined_status": "BLOCK",
        "stability_index": 0.8,
        "health_score": 90.0,
    }
    
    readiness = evaluate_budget_release_readiness(view)
    
    assert readiness["release_ok"] is False
    assert readiness["status"] == "BLOCK"
    assert len(readiness["blocking_reasons"]) > 0


def test_release_readiness_block_low_health():
    """Test release readiness blocked by low health score."""
    view = {
        "combined_status": "OK",
        "stability_index": 1.0,
        "health_score": 65.0,
    }
    
    readiness = evaluate_budget_release_readiness(view)
    
    assert readiness["release_ok"] is False
    assert readiness["status"] == "BLOCK"
    assert any("health_score" in reason for reason in readiness["blocking_reasons"])


def test_release_readiness_warn():
    """Test release readiness with WARN status."""
    view = {
        "combined_status": "WARN",
        "stability_index": 0.9,
        "health_score": 85.0,
    }
    
    readiness = evaluate_budget_release_readiness(view)
    
    assert readiness["release_ok"] is True  # WARN doesn't block
    assert readiness["status"] == "WARN"


def test_release_readiness_json_serializable():
    """Test release readiness is JSON-serializable."""
    view = {
        "combined_status": "OK",
        "stability_index": 1.0,
        "health_score": 90.0,
    }
    readiness = evaluate_budget_release_readiness(view)
    
    json_str = json.dumps(readiness)
    decoded = json.loads(json_str)
    assert decoded == readiness

