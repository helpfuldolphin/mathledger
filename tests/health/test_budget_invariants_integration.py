"""
Integration tests for budget invariants global health and evidence pack integration.
"""

import json
from typing import Any, Dict

import pytest

from backend.health.budget_invariants_adapter import (
    build_budget_invariants_tile_for_global_health,
)
from derivation.budget_invariants import (
    attach_budget_invariants_to_evidence,
    build_budget_invariant_snapshot,
    build_budget_invariant_timeline,
    build_budget_invariants_governance_view,
    evaluate_budget_release_readiness,
    explain_budget_release_decision,
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


def test_global_health_tile_integration():
    """Test budget invariants tile integration with global health."""
    # Build timeline
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    
    # Build tile
    tile = build_budget_invariants_tile_for_global_health(timeline)
    
    # Verify structure
    assert tile["schema_version"] == "1.0.0"
    assert tile["tile_type"] == "budget_invariants"
    assert tile["status_light"] in ["GREEN", "YELLOW", "RED"]
    assert "health_index" in tile
    assert "inv_bud_failures" in tile
    assert "stability_trend" in tile
    assert "headline" in tile
    assert "total_runs" in tile
    
    # Verify JSON serializability
    json_str = json.dumps(tile)
    assert isinstance(json_str, str)


def test_global_health_tile_status_light_mapping():
    """Test status_light mapping: BLOCK → RED, WARN → YELLOW, OK → GREEN."""
    # OK case
    snapshots_ok = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline_ok = build_budget_invariant_timeline(snapshots_ok)
    tile_ok = build_budget_invariants_tile_for_global_health(timeline_ok)
    assert tile_ok["status_light"] == "GREEN"
    
    # WARN case
    snapshots_warn = [
        build_budget_invariant_snapshot(MockStats(timeout_abstentions=5)),
    ]
    timeline_warn = build_budget_invariant_timeline(snapshots_warn)
    tile_warn = build_budget_invariants_tile_for_global_health(timeline_warn)
    assert tile_warn["status_light"] == "YELLOW"
    
    # BLOCK case
    snapshots_block = [
        build_budget_invariant_snapshot(MockStats(budget_exhausted=True, post_exhaustion_candidates=1)),
    ]
    timeline_block = build_budget_invariant_timeline(snapshots_block)
    tile_block = build_budget_invariants_tile_for_global_health(timeline_block)
    assert tile_block["status_light"] == "RED"


def test_evidence_pack_integration():
    """Test budget invariants evidence pack integration."""
    # Build governance view
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    
    # Create evidence pack
    evidence = {
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {"test": "value"},
    }
    
    # Attach budget invariants
    enriched = attach_budget_invariants_to_evidence(evidence, governance_view)
    
    # Verify structure
    assert "governance" in enriched
    assert "budget_invariants" in enriched["governance"]
    assert enriched["governance"]["budget_invariants"]["schema_version"] == "1.0.0"
    assert "invariant_failures" in enriched["governance"]["budget_invariants"]
    assert "stability_index" in enriched["governance"]["budget_invariants"]
    assert "combined_status" in enriched["governance"]["budget_invariants"]
    
    # Verify original evidence unchanged (non-mutating)
    assert "governance" not in evidence
    
    # Verify JSON serializability
    json_str = json.dumps(enriched)
    assert isinstance(json_str, str)


def test_evidence_pack_with_projection():
    """Test evidence pack integration with BNH-Φ projection."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    
    projection = {
        "projected_stability_class": "STABLE",
        "converging_invariants": True,
        "diverging_invariants": False,
        "horizon_length": 10,
    }
    
    evidence = {"timestamp": "2024-01-01T00:00:00Z"}
    enriched = attach_budget_invariants_to_evidence(
        evidence, governance_view, projection=projection
    )
    
    assert "projected_horizon" in enriched["governance"]["budget_invariants"]
    assert enriched["governance"]["budget_invariants"]["projected_horizon"]["projected_stability_class"] == "STABLE"


def test_evidence_pack_with_ci_capsule():
    """Test evidence pack integration with CI capsule."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    readiness = evaluate_budget_release_readiness(governance_view)
    ci_capsule = explain_budget_release_decision(governance_view, readiness)
    
    evidence = {"timestamp": "2024-01-01T00:00:00Z"}
    enriched = attach_budget_invariants_to_evidence(
        evidence, governance_view, ci_capsule=ci_capsule
    )
    
    assert "ci_trigger_index" in enriched["governance"]["budget_invariants"]
    assert "ci_fault_surface" in enriched["governance"]["budget_invariants"]
    assert "ci_summary" in enriched["governance"]["budget_invariants"]


def test_evidence_pack_deterministic():
    """Test evidence pack integration is deterministic."""
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    budget_health = {"health_score": 90.0, "trend_status": "STABLE"}
    governance_view = build_budget_invariants_governance_view(timeline, budget_health)
    
    evidence = {"timestamp": "2024-01-01T00:00:00Z"}
    enriched1 = attach_budget_invariants_to_evidence(evidence, governance_view)
    enriched2 = attach_budget_invariants_to_evidence(evidence, governance_view)
    
    assert enriched1 == enriched2


def test_global_health_tile_graceful_degradation():
    """Test global health tile degrades gracefully on invalid input."""
    # Empty timeline should not crash
    empty_timeline = {}
    tile = build_budget_invariants_tile_for_global_health(empty_timeline)
    
    assert tile["status_light"] in ["GREEN", "YELLOW", "RED"]
    assert "schema_version" in tile

