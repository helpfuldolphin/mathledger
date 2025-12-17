"""
Tests for Replay Safety integration with Global Governance Fusion Layer (GGFL).

Tests verify:
1. CONSISTENT replay signal → no escalation in fusion result
2. DIVERGENT replay signal → fusion result marks at least TENSION
3. Replay safety signal properly consumed by build_global_alignment_view
4. Conflict propagation from replay to fusion layer

SHADOW MODE CONTRACT:
- These tests validate observational behavior only
- No actual governance decisions are influenced
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

import pytest

from backend.governance.fusion import (
    build_global_alignment_view,
    EscalationLevel,
    GovernanceAction,
)
from experiments.u2.replay_safety import (
    PromotionStatus,
    GovernanceAlignment,
    to_governance_signal_for_replay_safety,
    replay_safety_for_alignment_view,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def healthy_replay_signal() -> Dict[str, Any]:
    """Healthy replay signal in GGFL format."""
    return {
        "replay_verified": True,
        "replay_divergence": 0.02,
        "replay_latency_ms": 50,
        "replay_hash_match": True,
        "replay_depth_valid": True,
    }


@pytest.fixture
def unhealthy_replay_signal() -> Dict[str, Any]:
    """Unhealthy replay signal triggering BLOCK."""
    return {
        "replay_verified": False,
        "replay_divergence": 0.5,
        "replay_latency_ms": 500,
        "replay_hash_match": True,  # Not a hard block
        "replay_depth_valid": False,
    }


@pytest.fixture
def hard_block_replay_signal() -> Dict[str, Any]:
    """Replay signal with hash mismatch (security critical HARD_BLOCK)."""
    return {
        "replay_verified": False,
        "replay_divergence": 1.0,
        "replay_latency_ms": 1000,
        "replay_hash_match": False,  # Security critical
        "replay_depth_valid": False,
    }


@pytest.fixture
def all_healthy_signals(healthy_replay_signal) -> Dict[str, Dict[str, Any]]:
    """All healthy signals for GGFL."""
    return {
        "topology": {
            "H": 0.85,
            "D": 5,
            "rho": 0.87,
            "within_omega": True,
            "C": 0,
            "active_cdis": [],
            "invariant_violations": [],
        },
        "replay": healthy_replay_signal,
        "metrics": {
            "success_rate": 0.85,
            "abstention_rate": 0.1,
            "block_rate": 0.15,
            "queue_depth": 50,
        },
        "budget": {
            "compute_budget_remaining": 0.75,
            "verification_quota_remaining": 1000,
            "budget_exhaustion_eta_cycles": 500,
        },
        "structure": {
            "dag_coherent": True,
            "orphan_count": 5,
            "cycle_detected": False,
            "min_cut_capacity": 0.5,
        },
        "telemetry": {
            "lean_healthy": True,
            "db_healthy": True,
            "redis_healthy": True,
            "worker_count": 4,
            "error_rate": 0.01,
        },
        "identity": {
            "block_hash_valid": True,
            "merkle_root_valid": True,
            "signature_valid": True,
            "chain_continuous": True,
            "pq_attestation_valid": True,
            "dual_root_consistent": True,
        },
        "narrative": {
            "current_slice": "propositional_tautology",
            "slice_progress": 0.6,
            "epoch": 5,
            "curriculum_health": "HEALTHY",
            "drift_detected": False,
            "narrative_coherence": 0.85,
        },
    }


# =============================================================================
# Test: CONSISTENT Replay → No Escalation
# =============================================================================

class TestConsistentReplayNoEscalation:
    """Test that consistent/healthy replay signal does not cause escalation."""

    def test_consistent_replay_l0_nominal(self, all_healthy_signals):
        """CONSISTENT replay with all healthy signals → L0 NOMINAL."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L0_NOMINAL
        assert result["fusion_result"]["decision"] == GovernanceAction.ALLOW

    def test_consistent_replay_signal_status_healthy(self, all_healthy_signals):
        """Replay signal marked as healthy in signal summary."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        assert result["signal_summary"]["replay"]["status"] == "healthy"
        assert result["signal_summary"]["replay"]["recommendations"] >= 1

    def test_consistent_replay_no_block_recommendations(self, all_healthy_signals):
        """Consistent replay produces no BLOCK recommendations."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        replay_recs = [
            r for r in result["recommendations"]
            if r["signal_id"] == "replay"
        ]

        block_recs = [
            r for r in replay_recs
            if r["action"] in (GovernanceAction.BLOCK, GovernanceAction.HARD_BLOCK)
        ]

        assert len(block_recs) == 0

    def test_consistent_replay_produces_allow_recommendation(self, all_healthy_signals):
        """Consistent replay produces ALLOW recommendation."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        replay_recs = [
            r for r in result["recommendations"]
            if r["signal_id"] == "replay"
        ]

        allow_recs = [r for r in replay_recs if r["action"] == GovernanceAction.ALLOW]
        assert len(allow_recs) == 1
        assert "passed" in allow_recs[0]["reason"].lower()


# =============================================================================
# Test: DIVERGENT Replay → At Least TENSION
# =============================================================================

class TestDivergentReplayTension:
    """Test that divergent replay signal causes at least TENSION in fusion."""

    def test_divergent_replay_causes_block_recommendation(
        self, all_healthy_signals, unhealthy_replay_signal
    ):
        """Unhealthy replay signal produces BLOCK recommendation."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["replay"] = unhealthy_replay_signal

        result = build_global_alignment_view(**signals, cycle=1)

        replay_recs = [
            r for r in result["recommendations"]
            if r["signal_id"] == "replay"
        ]

        block_recs = [
            r for r in replay_recs
            if r["action"] in (GovernanceAction.BLOCK, GovernanceAction.HARD_BLOCK)
        ]

        assert len(block_recs) >= 1
        assert "verification failed" in block_recs[0]["reason"].lower()

    def test_divergent_replay_escalation_at_least_l1(
        self, all_healthy_signals, unhealthy_replay_signal
    ):
        """Unhealthy replay causes escalation to at least L1 WARNING."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["replay"] = unhealthy_replay_signal

        result = build_global_alignment_view(**signals, cycle=1)

        # Should be at least L1 WARNING, possibly higher
        assert result["escalation"]["level"] >= EscalationLevel.L1_WARNING

    def test_divergent_replay_signal_status_unhealthy(
        self, all_healthy_signals, unhealthy_replay_signal
    ):
        """Divergent replay marked as unhealthy in signal summary."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["replay"] = unhealthy_replay_signal

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["signal_summary"]["replay"]["status"] == "unhealthy"

    def test_hard_block_replay_causes_l3_critical(
        self, all_healthy_signals, hard_block_replay_signal
    ):
        """Replay hash mismatch (HARD_BLOCK) causes L3 CRITICAL."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["replay"] = hard_block_replay_signal

        result = build_global_alignment_view(**signals, cycle=1)

        # Hash mismatch is security critical - should be L3+
        assert result["escalation"]["level"] >= EscalationLevel.L3_CRITICAL
        assert result["fusion_result"]["is_hard"] is True

    def test_hard_block_replay_decision_is_block(
        self, all_healthy_signals, hard_block_replay_signal
    ):
        """Replay hash mismatch results in BLOCK decision."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["replay"] = hard_block_replay_signal

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["fusion_result"]["decision"] == GovernanceAction.BLOCK


# =============================================================================
# Test: Replay Safety Signal Adapter Integration
# =============================================================================

class TestReplaySafetySignalAdapter:
    """Test replay_safety_for_alignment_view adapter integration."""

    def test_consistent_signal_safe_for_fusion(self):
        """Consistent replay signal is safe for fusion."""
        safety_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["All checks passed"],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
        }
        radar_eval = {
            "status": "OK",
            "governance_alignment": "aligned",
            "conflict": False,
            "reasons": ["No drift detected"],
        }

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
        alignment_view = replay_safety_for_alignment_view(signal)

        assert alignment_view["signal_type"] == "replay_safety"
        assert alignment_view["status"] == "ok"
        assert alignment_view["alignment"] == "aligned"
        assert alignment_view["conflict"] is False
        assert alignment_view["safe_for_fusion"] is True

    def test_divergent_signal_not_safe_for_fusion(self):
        """Divergent replay signal is not safe for fusion."""
        safety_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["Safety checks passed"],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
        }
        radar_eval = {
            "status": "BLOCK",
            "governance_alignment": "divergent",
            "conflict": True,
            "reasons": ["Governance drift critical"],
        }

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
        alignment_view = replay_safety_for_alignment_view(signal)

        assert alignment_view["status"] == "block"
        assert alignment_view["alignment"] == "divergent"
        assert alignment_view["conflict"] is True
        assert alignment_view["safe_for_fusion"] is False

    def test_tension_signal_not_safe_for_fusion(self):
        """TENSION alignment with WARN status is not safe for fusion."""
        safety_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["Safety OK"],
        }
        radar_eval = {
            "status": "WARN",
            "governance_alignment": "tension",
            "conflict": False,
            "reasons": ["Minor drift"],
        }

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
        alignment_view = replay_safety_for_alignment_view(signal)

        assert alignment_view["status"] == "warn"
        assert alignment_view["alignment"] == "tension"
        assert alignment_view["safe_for_fusion"] is False  # WARN is not safe

    def test_root_causes_populated(self):
        """Root causes are populated from signal reasons."""
        safety_eval = {
            "status": PromotionStatus.BLOCK,
            "reasons": ["Hash mismatch detected"],
        }
        radar_eval = {
            "status": "WARN",
            "governance_alignment": "tension",
            "conflict": False,
            "reasons": ["Drift threshold approaching"],
        }

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
        alignment_view = replay_safety_for_alignment_view(signal)

        assert len(alignment_view["root_causes"]) >= 2
        assert any("[Safety]" in r for r in alignment_view["root_causes"])
        assert any("[Radar]" in r for r in alignment_view["root_causes"])


# =============================================================================
# Test: Conflict Propagation to GGFL
# =============================================================================

class TestConflictPropagation:
    """Test that replay safety conflicts propagate to GGFL correctly."""

    def test_csc_003_conflict_detected(self, all_healthy_signals):
        """CSC-003: lean_healthy=true AND replay_verified=false → conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["telemetry"]["lean_healthy"] = True
        signals["replay"]["replay_verified"] = False

        result = build_global_alignment_view(**signals, cycle=1)

        # Should detect CSC-003 conflict
        conflict_rules = [c["rule_id"] for c in result["conflict_detections"]]
        assert "CSC-003" in conflict_rules

        # Should escalate to at least L4 CONFLICT
        assert result["escalation"]["level"] >= EscalationLevel.L4_CONFLICT

    def test_replay_conflict_with_telemetry(self, all_healthy_signals):
        """Replay failure with healthy telemetry creates cross-signal conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["telemetry"]["lean_healthy"] = True  # Lean is healthy
        signals["replay"]["replay_verified"] = False  # But replay failed

        result = build_global_alignment_view(**signals, cycle=1)

        # Conflict detection should fire
        assert len(result["conflict_detections"]) > 0

        # Headline should mention conflict
        assert "CONFLICT" in result["headline"] or "CRITICAL" in result["headline"]


# =============================================================================
# Test: Determinism
# =============================================================================

class TestReplaySignalDeterminism:
    """Test that replay signal integration is deterministic."""

    def test_deterministic_with_replay_signal(self, all_healthy_signals):
        """Same replay signal produces identical fusion results."""
        result1 = build_global_alignment_view(**all_healthy_signals, cycle=100)
        result2 = build_global_alignment_view(**all_healthy_signals, cycle=100)

        # Remove timestamps for comparison
        for r in [result1, result2]:
            del r["timestamp"]
            for sig in r["signals"].values():
                if "timestamp" in sig:
                    del sig["timestamp"]

        assert result1 == result2

    def test_deterministic_adapter_output(self):
        """replay_safety_for_alignment_view produces deterministic output."""
        safety_eval = {
            "status": PromotionStatus.OK,
            "reasons": ["Check 1", "Check 2"],
        }
        radar_eval = {
            "status": "OK",
            "governance_alignment": "aligned",
            "conflict": False,
            "reasons": ["Radar OK"],
        }

        results = []
        for _ in range(10):
            signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
            view = replay_safety_for_alignment_view(signal)
            results.append(json.dumps(view, sort_keys=True))

        # All results should be identical
        assert len(set(results)) == 1


# =============================================================================
# Test: JSON Serializability
# =============================================================================

class TestJSONSerializability:
    """Test that all outputs are JSON serializable."""

    def test_fusion_result_json_serializable(self, all_healthy_signals):
        """GGFL fusion result is JSON serializable."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["cycle"] == 1

    def test_alignment_view_json_serializable(self):
        """replay_safety_for_alignment_view output is JSON serializable."""
        safety_eval = {"status": PromotionStatus.OK, "reasons": []}
        radar_eval = {"status": "OK", "governance_alignment": "aligned", "conflict": False}

        signal = to_governance_signal_for_replay_safety(safety_eval, radar_eval)
        view = replay_safety_for_alignment_view(signal)

        # Should not raise
        json_str = json.dumps(view)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "replay_safety"
