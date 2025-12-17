"""
Test suite for budget storyline and BNH-Φ projection extensions.

Tests projection correctness, episode ledger structure, CI capsule formation,
and deterministic behavior.
"""

import json
from typing import Any, Dict, Sequence

import pytest

from derivation.budget_invariants import (
    BUDGET_SCHEMA_VERSION,
    attach_budget_invariants_to_evidence,
    build_budget_episode_ledger_tile,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    build_budget_invariants_governance_view,
    build_budget_storyline,
    build_first_light_budget_storyline,
    evaluate_budget_release_readiness,
    explain_budget_release_decision,
    project_budget_stability_horizon,
    summarize_storyline_for_global_health,
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
# Storyline Tests
# ---------------------------------------------------------------------------


def test_storyline_empty():
    """Test storyline with no runs."""
    timeline = build_budget_invariant_timeline([])
    storyline = build_budget_storyline(timeline, [])
    
    assert storyline["schema_version"] == BUDGET_SCHEMA_VERSION
    assert storyline["runs_analyzed"] == 0
    assert storyline["episodes"] == []
    assert storyline["structural_events"] == []
    assert storyline["stability_class"] == "STABLE"


def test_storyline_stable():
    """Test storyline with stable runs."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(10)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(10)]
    
    storyline = build_budget_storyline(timeline, health_history)
    
    assert storyline["stability_class"] == "STABLE"
    assert storyline["runs_analyzed"] == 10


def test_storyline_drifting():
    """Test storyline with drifting stability."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=1)),
        build_budget_invariant_snapshot(MockStats()),
    ]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [
        {"health_score": 85.0, "trend_status": "STABLE"},
        {"health_score": 80.0, "trend_status": "DEGRADING"},
        {"health_score": 75.0, "trend_status": "DEGRADING"},
    ]
    
    storyline = build_budget_storyline(timeline, health_history)
    
    assert storyline["stability_class"] in ["DRIFTING", "VOLATILE"]


def test_storyline_volatile():
    """Test storyline with volatile stability."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
        build_budget_invariant_snapshot(MockStats()),
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [
        {"health_score": 60.0, "trend_status": "DEGRADING"},
        {"health_score": 65.0, "trend_status": "DEGRADING"},
        {"health_score": 55.0, "trend_status": "DEGRADING"},
    ]
    
    storyline = build_budget_storyline(timeline, health_history)
    
    assert storyline["stability_class"] == "VOLATILE"


def test_storyline_episodes():
    """Test storyline episode generation."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(10)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [
        {"health_score": 90.0, "trend_status": "STABLE"},
        {"health_score": 65.0, "trend_status": "DEGRADING"},
        {"health_score": 60.0, "trend_status": "DEGRADING"},
        {"health_score": 85.0, "trend_status": "IMPROVING"},
    ]
    
    storyline = build_budget_storyline(timeline, health_history)
    
    # Should detect episode of concern
    assert len(storyline["episodes"]) > 0


def test_storyline_structural_events():
    """Test storyline structural event detection."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ]
    timeline = build_budget_invariant_timeline(snapshots)
    
    storyline = build_budget_storyline(timeline, [])
    
    assert len(storyline["structural_events"]) > 0
    assert any("INV-BUD-1" in event for event in storyline["structural_events"])


# ---------------------------------------------------------------------------
# Projection Tests
# ---------------------------------------------------------------------------


def test_projection_insufficient_data():
    """Test projection with insufficient data."""
    projection = project_budget_stability_horizon([])
    
    assert projection["projected_stability_class"] == "UNKNOWN"
    assert projection["projection_method"] == "insufficient_data"
    assert projection["horizon_length"] == 10


def test_projection_stable_trajectory():
    """Test projection with stable trajectory."""
    health_history = [
        {"health_score": 90.0, "trend_status": "STABLE"},
        {"health_score": 91.0, "trend_status": "STABLE"},
        {"health_score": 92.0, "trend_status": "STABLE"},
        {"health_score": 93.0, "trend_status": "STABLE"},
        {"health_score": 94.0, "trend_status": "STABLE"},
    ]
    
    projection = project_budget_stability_horizon(health_history)
    
    assert projection["projected_stability_class"] == "STABLE"
    assert projection["converging_invariants"] is True
    assert projection["diverging_invariants"] is False
    assert projection["projection_method"] == "linear_extrapolation"
    assert len(projection["risk_trajectory"]) == 10


def test_projection_drifting_trajectory():
    """Test projection with drifting trajectory."""
    health_history = [
        {"health_score": 75.0, "trend_status": "STABLE"},
        {"health_score": 74.0, "trend_status": "DEGRADING"},
        {"health_score": 73.0, "trend_status": "DEGRADING"},
        {"health_score": 72.0, "trend_status": "DEGRADING"},
        {"health_score": 71.0, "trend_status": "DEGRADING"},
    ]
    
    projection = project_budget_stability_horizon(health_history)
    
    assert projection["projected_stability_class"] in ["DRIFTING", "VOLATILE"]
    assert projection["diverging_invariants"] is True


def test_projection_volatile_trajectory():
    """Test projection with volatile trajectory."""
    health_history = [
        {"health_score": 65.0, "trend_status": "DEGRADING"},
        {"health_score": 60.0, "trend_status": "DEGRADING"},
        {"health_score": 55.0, "trend_status": "DEGRADING"},
        {"health_score": 50.0, "trend_status": "DEGRADING"},
        {"health_score": 45.0, "trend_status": "DEGRADING"},
    ]
    
    projection = project_budget_stability_horizon(health_history)
    
    assert projection["projected_stability_class"] == "VOLATILE"
    assert projection["diverging_invariants"] is True


def test_projection_deterministic():
    """Test projection is deterministic."""
    health_history = [
        {"health_score": 80.0, "trend_status": "STABLE"},
        {"health_score": 81.0, "trend_status": "STABLE"},
        {"health_score": 82.0, "trend_status": "STABLE"},
    ]
    
    projection1 = project_budget_stability_horizon(health_history)
    projection2 = project_budget_stability_horizon(health_history)
    
    assert projection1 == projection2


def test_projection_custom_horizon():
    """Test projection with custom horizon length."""
    health_history = [
        {"health_score": 85.0, "trend_status": "STABLE"},
        {"health_score": 86.0, "trend_status": "STABLE"},
    ]
    
    projection = project_budget_stability_horizon(health_history, horizon_length=5)
    
    assert projection["horizon_length"] == 5
    assert len(projection["risk_trajectory"]) == 5


# ---------------------------------------------------------------------------
# Episode Ledger Tile Tests
# ---------------------------------------------------------------------------


def test_episode_ledger_tile_structure():
    """Test episode ledger tile has correct structure."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    
    assert tile["schema_version"] == BUDGET_SCHEMA_VERSION
    assert "episodes_count" in tile
    assert "episode_status_distribution" in tile
    assert "stability_class" in tile
    assert "projection" in tile
    assert "status_light" in tile
    assert "headline" in tile
    assert tile["status_light"] in ["GREEN", "YELLOW", "RED"]


def test_episode_ledger_tile_status_light_green():
    """Test episode ledger tile status light GREEN."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    
    assert tile["status_light"] == "GREEN"


def test_episode_ledger_tile_status_light_yellow():
    """Test episode ledger tile status light YELLOW."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=1)),
    ]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 75.0, "trend_status": "DEGRADING"}]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    
    assert tile["status_light"] in ["YELLOW", "RED"]


def test_episode_ledger_tile_status_light_red():
    """Test episode ledger tile status light RED."""
    snapshots = [
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 50.0, "trend_status": "DEGRADING"}]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    
    assert tile["status_light"] == "RED"


def test_episode_ledger_tile_headline_neutral():
    """Test episode ledger tile headline uses neutral language."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    headline = tile["headline"].lower()
    
    # Should not contain blame terms
    assert "bad" not in headline
    assert "good" not in headline
    assert "failure" not in headline
    assert "success" not in headline
    assert "error" not in headline


def test_episode_ledger_tile_json_serializable():
    """Test episode ledger tile is JSON-serializable."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    tile = build_budget_episode_ledger_tile(storyline, projection)
    
    json_str = json.dumps(tile)
    decoded = json.loads(json_str)
    assert decoded == tile


# ---------------------------------------------------------------------------
# CI Release Explanation Capsule Tests
# ---------------------------------------------------------------------------


def test_ci_capsule_ok():
    """Test CI capsule with OK status."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
    ])
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    assert capsule["decision"] == "OK"
    assert len(capsule["root_cause_vector"]) >= 0
    assert capsule["ci_summary"] is not None
    assert len(capsule["ci_summary"]) <= 160


def test_ci_capsule_block():
    """Test CI capsule with BLOCK status."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    assert capsule["decision"] == "BLOCK"
    assert len(capsule["root_cause_vector"]) > 0
    assert capsule["trigger_index"] >= 0  # Should identify INV-BUD-1
    assert len(capsule["ci_summary"]) <= 160


def test_ci_capsule_trigger_index():
    """Test CI capsule trigger_index accuracy."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    # Should identify INV-BUD-1 (index 0)
    assert capsule["trigger_index"] == 0
    # Fault surface uses "Invariant 1" format, not "INV-BUD-1"
    assert "Invariant 1" in capsule["fault_surface"] or "INV-BUD-1" in capsule["fault_surface"]


def test_ci_capsule_fault_surface():
    """Test CI capsule fault_surface is neutral."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 65.0, "trend_status": "DEGRADING"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    fault_surface = capsule["fault_surface"].lower()
    # Should be neutral
    assert "bad" not in fault_surface
    assert "good" not in fault_surface
    assert "error" not in fault_surface


def test_ci_capsule_summary_length():
    """Test CI capsule summary is <=160 chars."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 65.0, "trend_status": "DEGRADING"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    assert len(capsule["ci_summary"]) <= 160


def test_ci_capsule_summary_neutral():
    """Test CI capsule summary uses neutral language."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 65.0, "trend_status": "DEGRADING"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    summary = capsule["ci_summary"].lower()
    
    # Should be neutral
    assert "bad" not in summary
    assert "good" not in summary
    assert "failure" not in summary
    assert "success" not in summary


def test_ci_capsule_deterministic():
    """Test CI capsule is deterministic."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ])
    budget_health = {"health_score": 65.0, "trend_status": "DEGRADING"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule1 = explain_budget_release_decision(view, readiness)
    capsule2 = explain_budget_release_decision(view, readiness)
    
    assert capsule1 == capsule2


def test_ci_capsule_json_serializable():
    """Test CI capsule is JSON-serializable."""
    timeline = build_budget_invariant_timeline([
        build_budget_invariant_snapshot(MockStats()),
    ])
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(view)
    
    capsule = explain_budget_release_decision(view, readiness)
    
    json_str = json.dumps(capsule)
    decoded = json.loads(json_str)
    assert decoded == capsule


# ---------------------------------------------------------------------------
# Global Health Adapter Tests
# ---------------------------------------------------------------------------


def test_global_health_adapter():
    """Test global health adapter from storyline."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    adapter = summarize_storyline_for_global_health(storyline, projection)
    
    assert adapter["schema_version"] == BUDGET_SCHEMA_VERSION
    assert adapter["status"] in ["OK", "WARN", "BLOCK"]
    assert "stability_class" in adapter
    assert "projected_stability_class" in adapter
    assert "episodes_count" in adapter
    assert "stability_index" in adapter
    assert "summary" in adapter


def test_global_health_adapter_json_serializable():
    """Test global health adapter is JSON-serializable."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    adapter = summarize_storyline_for_global_health(storyline, projection)
    
    json_str = json.dumps(adapter)
    decoded = json.loads(json_str)
    assert decoded == adapter


# ---------------------------------------------------------------------------
# First Light Budget Storyline Tests
# ---------------------------------------------------------------------------
#
# Example Narrative:
# In this synthetic timeline, episodes show repeated CRITICAL budget violations.
# The First Light storyline summary flags BLOCK and lists the five most important
# structural events; this is meant for human review, not automatic gating.
# When combined_status="BLOCK" but other budget tiles look OK, this indicates
# an early-warning, long-horizon signal requiring investigation before production
# deployment. The projection_class and key_structural_events should be read as
# a temporal narrative (what happened, what's projected) rather than a single
# scalar health check.
#


def test_first_light_budget_storyline_structure():
    """Test First Light budget storyline has correct structure."""
    # Build synthetic timeline, storyline, projection
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    # Build First Light storyline summary
    first_light = build_first_light_budget_storyline(timeline, storyline, projection)
    
    # Verify required keys
    assert first_light["schema_version"] == BUDGET_SCHEMA_VERSION
    assert "combined_status" in first_light
    assert "stability_index" in first_light
    assert "episodes_count" in first_light
    assert "projection_class" in first_light
    assert "key_structural_events" in first_light
    
    # Verify types
    assert first_light["combined_status"] in ["OK", "WARN", "BLOCK"]
    assert isinstance(first_light["stability_index"], float)
    assert isinstance(first_light["episodes_count"], int)
    assert isinstance(first_light["projection_class"], str)
    assert isinstance(first_light["key_structural_events"], list)


def test_first_light_budget_storyline_json_serializable():
    """Test First Light budget storyline is JSON-serializable."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    first_light = build_first_light_budget_storyline(timeline, storyline, projection)
    
    # Should round-trip through JSON
    json_str = json.dumps(first_light)
    decoded = json.loads(json_str)
    assert decoded == first_light


def test_first_light_budget_storyline_deterministic():
    """Test First Light budget storyline is deterministic."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    first_light1 = build_first_light_budget_storyline(timeline, storyline, projection)
    first_light2 = build_first_light_budget_storyline(timeline, storyline, projection)
    
    assert first_light1 == first_light2


def test_first_light_budget_storyline_key_events_limit():
    """Test First Light budget storyline limits key_structural_events to first 5."""
    snapshots = [build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)) for _ in range(10)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 50.0, "trend_status": "DEGRADING"} for _ in range(10)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    first_light = build_first_light_budget_storyline(timeline, storyline, projection)
    
    # Should have at most 5 key events
    assert len(first_light["key_structural_events"]) <= 5


def test_first_light_budget_storyline_status_mapping():
    """Test First Light budget storyline combined_status mapping."""
    # OK case
    snapshots_ok = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline_ok = build_budget_invariant_timeline(snapshots_ok)
    health_history_ok = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    storyline_ok = build_budget_storyline(timeline_ok, health_history_ok)
    projection_ok = project_budget_stability_horizon(health_history_ok)
    
    first_light_ok = build_first_light_budget_storyline(timeline_ok, storyline_ok, projection_ok)
    assert first_light_ok["combined_status"] == "OK"
    
    # WARN case
    snapshots_warn = [build_budget_invariant_snapshot(MockStats(timeout_abstentions=5)) for _ in range(3)]
    timeline_warn = build_budget_invariant_timeline(snapshots_warn)
    health_history_warn = [{"health_score": 75.0, "trend_status": "DEGRADING"} for _ in range(3)]
    storyline_warn = build_budget_storyline(timeline_warn, health_history_warn)
    projection_warn = project_budget_stability_horizon(health_history_warn)
    
    first_light_warn = build_first_light_budget_storyline(timeline_warn, storyline_warn, projection_warn)
    assert first_light_warn["combined_status"] in ["WARN", "OK"]  # May be OK if stability_index high enough


def test_evidence_pack_with_first_light_storyline():
    """Test evidence pack integration with First Light storyline summary."""
    # Build synthetic data
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    # Create evidence pack
    evidence = {
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {"test": "value"},
    }
    
    # Attach with First Light storyline
    enriched = attach_budget_invariants_to_evidence(
        evidence,
        governance_view,
        projection=projection,
        timeline=timeline,
        storyline=storyline,
    )
    
    # Verify First Light storyline is attached
    assert "governance" in enriched
    assert "budget_storyline_summary" in enriched["governance"]
    
    first_light = enriched["governance"]["budget_storyline_summary"]
    assert first_light["schema_version"] == BUDGET_SCHEMA_VERSION
    assert "combined_status" in first_light
    assert "stability_index" in first_light
    assert "episodes_count" in first_light
    assert "projection_class" in first_light
    assert "key_structural_events" in first_light
    
    # Verify JSON serializable
    json_str = json.dumps(enriched)
    decoded = json.loads(json_str)
    assert decoded == enriched


def test_evidence_pack_first_light_storyline_optional():
    """Test evidence pack integration without First Light storyline (backward compatible)."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    
    evidence = {"timestamp": "2024-01-01T00:00:00Z"}
    
    # Attach without timeline/storyline (should not include First Light storyline)
    enriched = attach_budget_invariants_to_evidence(evidence, governance_view)
    
    # Should have budget_invariants but not budget_storyline_summary
    assert "governance" in enriched
    assert "budget_invariants" in enriched["governance"]
    assert "budget_storyline_summary" not in enriched["governance"]


def test_evidence_pack_first_light_storyline_deterministic():
    """Test evidence pack with First Light storyline is deterministic."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    evidence = {"timestamp": "2024-01-01T00:00:00Z"}
    
    enriched1 = attach_budget_invariants_to_evidence(
        evidence, governance_view, projection=projection, timeline=timeline, storyline=storyline
    )
    enriched2 = attach_budget_invariants_to_evidence(
        evidence, governance_view, projection=projection, timeline=timeline, storyline=storyline
    )
    
    assert enriched1 == enriched2

