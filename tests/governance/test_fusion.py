"""
Tests for Global Governance Fusion Layer (GGFL).

Tests cover:
1. Pairwise conflict detection (CSC-001 through CSC-004)
2. Correct escalation levels (L0-L5)
3. Determinism (same inputs → same outputs)
4. Signal precedence ordering
5. Weighted voting mechanics
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
    SIGNAL_PRECEDENCE,
    _detect_cross_signal_conflicts,
    _compute_escalation_level,
    _extract_recommendations,
    _validate_signal,
)


# =============================================================================
# Test Fixtures: Healthy Signals
# =============================================================================

@pytest.fixture
def healthy_topology() -> Dict[str, Any]:
    """Healthy topology signal."""
    return {
        "H": 0.85,
        "D": 5,
        "D_dot": 0.5,
        "B": 2.5,
        "S": 0.15,
        "C": 0,  # CONVERGING
        "rho": 0.87,
        "tau": 0.21,
        "J": 2.5,
        "within_omega": True,
        "active_cdis": [],
        "invariant_violations": [],
    }


@pytest.fixture
def healthy_replay() -> Dict[str, Any]:
    """Healthy replay signal."""
    return {
        "replay_verified": True,
        "replay_divergence": 0.02,
        "replay_latency_ms": 50,
        "replay_hash_match": True,
        "replay_depth_valid": True,
    }


@pytest.fixture
def healthy_metrics() -> Dict[str, Any]:
    """Healthy metrics signal."""
    return {
        "success_rate": 0.85,
        "abstention_rate": 0.1,
        "block_rate": 0.15,
        "throughput": 25.0,
        "latency_p50_ms": 100,
        "latency_p99_ms": 500,
        "queue_depth": 50,
    }


@pytest.fixture
def healthy_budget() -> Dict[str, Any]:
    """Healthy budget signal."""
    return {
        "compute_budget_remaining": 0.75,
        "memory_utilization": 0.4,
        "storage_headroom_gb": 100.0,
        "verification_quota_remaining": 1000,
        "budget_exhaustion_eta_cycles": 500,
    }


@pytest.fixture
def healthy_structure() -> Dict[str, Any]:
    """Healthy structure signal."""
    return {
        "dag_coherent": True,
        "orphan_count": 5,
        "max_fanout": 10,
        "depth_distribution": {"1": 100, "2": 200, "3": 150},
        "cycle_detected": False,
        "min_cut_capacity": 0.5,
    }


@pytest.fixture
def healthy_telemetry() -> Dict[str, Any]:
    """Healthy telemetry signal."""
    return {
        "lean_healthy": True,
        "db_healthy": True,
        "redis_healthy": True,
        "worker_count": 4,
        "error_rate": 0.01,
        "last_error": None,
        "uptime_seconds": 86400,
    }


@pytest.fixture
def healthy_identity() -> Dict[str, Any]:
    """Healthy identity signal."""
    return {
        "block_hash_valid": True,
        "merkle_root_valid": True,
        "signature_valid": True,
        "chain_continuous": True,
        "pq_attestation_valid": True,
        "dual_root_consistent": True,
    }


@pytest.fixture
def healthy_narrative() -> Dict[str, Any]:
    """Healthy narrative signal."""
    return {
        "current_slice": "propositional_tautology",
        "slice_progress": 0.6,
        "epoch": 5,
        "curriculum_health": "HEALTHY",
        "drift_detected": False,
        "narrative_coherence": 0.85,
    }


@pytest.fixture
def healthy_p5_patterns() -> Dict[str, Any]:
    """Healthy P5 patterns signal (SHADOW MODE - advisory only)."""
    return {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "final_pattern": "NOMINAL",
        "final_streak": 0,
        "cycles_analyzed": 100,
        "recalibration_triggered": False,
        "shadow_mode_invariants_ok": True,
        "pattern_history": [],
    }


@pytest.fixture
def healthy_risk() -> Dict[str, Any]:
    """Healthy risk signal (SHADOW MODE - advisory only)."""
    return {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "risk_level": "LOW",
        "risk_score": 0.1,
        "risk_factors": [],
    }


@pytest.fixture
def all_healthy_signals(
    healthy_topology,
    healthy_replay,
    healthy_metrics,
    healthy_budget,
    healthy_structure,
    healthy_telemetry,
    healthy_identity,
    healthy_narrative,
    healthy_p5_patterns,
    healthy_risk,
) -> Dict[str, Dict[str, Any]]:
    """All healthy signals."""
    return {
        "topology": healthy_topology,
        "replay": healthy_replay,
        "metrics": healthy_metrics,
        "budget": healthy_budget,
        "structure": healthy_structure,
        "telemetry": healthy_telemetry,
        "identity": healthy_identity,
        "narrative": healthy_narrative,
        "p5_patterns": healthy_p5_patterns,
        "risk": healthy_risk,
    }


# =============================================================================
# Test: Pairwise Conflict Detection (CSC-001 through CSC-004)
# =============================================================================

class TestPairwiseConflictDetection:
    """Test cross-signal conflict detection rules."""

    def test_csc_001_chain_vs_dag(self, all_healthy_signals):
        """CSC-001: chain_continuous=false AND dag_coherent=true → conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["identity"]["chain_continuous"] = False
        signals["structure"]["dag_coherent"] = True

        conflicts = _detect_cross_signal_conflicts(signals)

        assert len(conflicts) == 1
        assert conflicts[0].rule_id == "CSC-001"
        assert "identity" in conflicts[0].signals_involved
        assert "structure" in conflicts[0].signals_involved
        assert conflicts[0].severity == "HIGH"

    def test_csc_002_topology_vs_metrics(self, all_healthy_signals):
        """CSC-002: within_omega=true AND block_rate>0.5 → conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["topology"]["within_omega"] = True
        signals["metrics"]["block_rate"] = 0.6

        conflicts = _detect_cross_signal_conflicts(signals)

        assert len(conflicts) == 1
        assert conflicts[0].rule_id == "CSC-002"
        assert "topology" in conflicts[0].signals_involved
        assert "metrics" in conflicts[0].signals_involved
        assert conflicts[0].severity == "MEDIUM"

    def test_csc_003_telemetry_vs_replay(self, all_healthy_signals):
        """CSC-003: lean_healthy=true AND replay_verified=false → conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["telemetry"]["lean_healthy"] = True
        signals["replay"]["replay_verified"] = False

        conflicts = _detect_cross_signal_conflicts(signals)

        assert len(conflicts) == 1
        assert conflicts[0].rule_id == "CSC-003"
        assert "telemetry" in conflicts[0].signals_involved
        assert "replay" in conflicts[0].signals_involved
        assert conflicts[0].severity == "HIGH"

    def test_csc_004_budget_vs_telemetry(self, all_healthy_signals):
        """CSC-004: verification_quota_remaining>0 AND worker_count=0 → conflict."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["budget"]["verification_quota_remaining"] = 100
        signals["telemetry"]["worker_count"] = 0

        conflicts = _detect_cross_signal_conflicts(signals)

        # Should have both CSC-004 and potentially others due to worker_count=0
        csc_004 = [c for c in conflicts if c.rule_id == "CSC-004"]
        assert len(csc_004) == 1
        assert "budget" in csc_004[0].signals_involved
        assert "telemetry" in csc_004[0].signals_involved

    def test_no_conflicts_when_healthy(self, all_healthy_signals):
        """No conflicts when all signals are healthy."""
        conflicts = _detect_cross_signal_conflicts(all_healthy_signals)
        assert len(conflicts) == 0

    def test_multiple_conflicts(self, all_healthy_signals):
        """Multiple conflicts can be detected simultaneously."""
        signals = copy.deepcopy(all_healthy_signals)
        # Trigger CSC-001
        signals["identity"]["chain_continuous"] = False
        signals["structure"]["dag_coherent"] = True
        # Trigger CSC-002
        signals["topology"]["within_omega"] = True
        signals["metrics"]["block_rate"] = 0.6

        conflicts = _detect_cross_signal_conflicts(signals)

        rule_ids = {c.rule_id for c in conflicts}
        assert "CSC-001" in rule_ids
        assert "CSC-002" in rule_ids


# =============================================================================
# Test: Escalation Levels (L0-L5)
# =============================================================================

class TestEscalationLevels:
    """Test escalation level computation."""

    def test_l0_nominal_all_healthy(self, all_healthy_signals):
        """L0 NOMINAL: All signals healthy."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L0_NOMINAL
        assert result["escalation"]["level_name"] == "L0_NOMINAL"

    def test_l1_warning_from_warning_recommendation(self, all_healthy_signals):
        """L1 WARNING: Any WARNING recommendation."""
        signals = copy.deepcopy(all_healthy_signals)
        # High abstention rate triggers warning
        signals["metrics"]["abstention_rate"] = 0.4

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L1_WARNING
        assert result["escalation"]["level_name"] == "L1_WARNING"

    def test_l2_degraded_multiple_soft_blocks(self, all_healthy_signals):
        """L2 DEGRADED: Multiple soft BLOCK recommendations."""
        signals = copy.deepcopy(all_healthy_signals)
        # Multiple conditions that trigger soft BLOCK
        signals["topology"]["within_omega"] = False
        signals["structure"]["min_cut_capacity"] = 0.05

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["escalation"]["level"] >= EscalationLevel.L2_DEGRADED

    def test_l3_critical_hard_block(self, all_healthy_signals):
        """L3 CRITICAL: Any HARD_BLOCK recommendation."""
        signals = copy.deepcopy(all_healthy_signals)
        # Verification quota exhausted triggers HARD_BLOCK
        signals["budget"]["verification_quota_remaining"] = 0

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L3_CRITICAL
        assert result["escalation"]["level_name"] == "L3_CRITICAL"

    def test_l4_conflict_cross_signal(self, all_healthy_signals):
        """L4 CONFLICT: Cross-signal consistency failure."""
        signals = copy.deepcopy(all_healthy_signals)
        # Trigger CSC-003 conflict
        signals["telemetry"]["lean_healthy"] = True
        signals["replay"]["replay_verified"] = False

        result = build_global_alignment_view(**signals, cycle=1)

        # Should be at least L4 due to conflict
        assert result["escalation"]["level"] >= EscalationLevel.L4_CONFLICT
        assert len(result["conflict_detections"]) > 0

    def test_l5_emergency_identity_failure(self, all_healthy_signals):
        """L5 EMERGENCY: Identity signal failure."""
        signals = copy.deepcopy(all_healthy_signals)
        # Block hash invalid triggers HARD_BLOCK from identity
        signals["identity"]["block_hash_valid"] = False

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L5_EMERGENCY
        assert result["escalation"]["level_name"] == "L5_EMERGENCY"

    def test_l5_emergency_structure_failure(self, all_healthy_signals):
        """L5 EMERGENCY: Structure signal failure."""
        signals = copy.deepcopy(all_healthy_signals)
        # Cycle detected triggers HARD_BLOCK from structure
        signals["structure"]["cycle_detected"] = True

        result = build_global_alignment_view(**signals, cycle=1)

        assert result["escalation"]["level"] == EscalationLevel.L5_EMERGENCY

    def test_l5_emergency_multiple_hard_blocks(self, all_healthy_signals):
        """L5 EMERGENCY: Multiple HARD_BLOCK conditions."""
        signals = copy.deepcopy(all_healthy_signals)
        # Multiple HARD_BLOCK triggers (not from identity/structure)
        signals["telemetry"]["lean_healthy"] = False
        signals["telemetry"]["db_healthy"] = False

        result = build_global_alignment_view(**signals, cycle=1)

        # Multiple telemetry HARD_BLOCKs should trigger L5
        assert result["escalation"]["level"] == EscalationLevel.L5_EMERGENCY


# =============================================================================
# Test: Determinism
# =============================================================================

class TestDeterminism:
    """Test that same inputs produce same outputs."""

    def test_deterministic_output(self, all_healthy_signals):
        """Same inputs should produce identical outputs (except timestamp)."""
        result1 = build_global_alignment_view(**all_healthy_signals, cycle=100)
        result2 = build_global_alignment_view(**all_healthy_signals, cycle=100)

        # Remove timestamps for comparison
        for r in [result1, result2]:
            del r["timestamp"]
            for sig in r["signals"].values():
                if "timestamp" in sig:
                    del sig["timestamp"]

        assert result1 == result2

    def test_deterministic_with_unhealthy_signals(self, all_healthy_signals):
        """Determinism holds with unhealthy signals."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["topology"]["within_omega"] = False
        signals["metrics"]["block_rate"] = 0.6

        result1 = build_global_alignment_view(**signals, cycle=50)
        result2 = build_global_alignment_view(**signals, cycle=50)

        # Remove timestamps for comparison
        for r in [result1, result2]:
            del r["timestamp"]
            for sig in r["signals"].values():
                if "timestamp" in sig:
                    del sig["timestamp"]

        assert result1 == result2

    def test_deterministic_fusion_decision(self, all_healthy_signals):
        """Fusion decision is deterministic."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["topology"]["rho"] = 0.35  # Below threshold

        decisions = []
        for _ in range(10):
            result = build_global_alignment_view(**signals, cycle=1)
            decisions.append(result["fusion_result"]["decision"])

        # All decisions should be the same
        assert len(set(decisions)) == 1

    def test_deterministic_escalation(self, all_healthy_signals):
        """Escalation level is deterministic."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["telemetry"]["lean_healthy"] = False

        levels = []
        for _ in range(10):
            result = build_global_alignment_view(**signals, cycle=1)
            levels.append(result["escalation"]["level"])

        # All levels should be the same
        assert len(set(levels)) == 1


# =============================================================================
# Test: Signal Precedence
# =============================================================================

class TestSignalPrecedence:
    """Test signal precedence ordering."""

    def test_precedence_order(self):
        """Verify precedence order matches specification."""
        assert SIGNAL_PRECEDENCE["identity"] < SIGNAL_PRECEDENCE["structure"]
        assert SIGNAL_PRECEDENCE["structure"] < SIGNAL_PRECEDENCE["telemetry"]
        assert SIGNAL_PRECEDENCE["telemetry"] < SIGNAL_PRECEDENCE["replay"]
        assert SIGNAL_PRECEDENCE["replay"] < SIGNAL_PRECEDENCE["topology"]
        assert SIGNAL_PRECEDENCE["topology"] < SIGNAL_PRECEDENCE["budget"]
        assert SIGNAL_PRECEDENCE["budget"] < SIGNAL_PRECEDENCE["metrics"]
        assert SIGNAL_PRECEDENCE["metrics"] < SIGNAL_PRECEDENCE["narrative"]
        assert SIGNAL_PRECEDENCE["narrative"] < SIGNAL_PRECEDENCE["p5_patterns"]

    def test_identity_highest_precedence(self, all_healthy_signals):
        """Identity signal has highest precedence for HARD_BLOCK."""
        signals = copy.deepcopy(all_healthy_signals)
        # Trigger HARD_BLOCK from multiple signals
        signals["identity"]["block_hash_valid"] = False
        signals["structure"]["cycle_detected"] = True
        signals["telemetry"]["lean_healthy"] = False

        result = build_global_alignment_view(**signals, cycle=1)

        # Identity should be the determining signal
        assert result["fusion_result"]["determining_signal"] == "identity"


# =============================================================================
# Test: Weighted Voting
# =============================================================================

class TestWeightedVoting:
    """Test weighted voting mechanics."""

    def test_allow_bias_effect(self, all_healthy_signals):
        """Allow bias shifts decision toward ALLOW."""
        signals = copy.deepcopy(all_healthy_signals)
        # Borderline case
        signals["topology"]["rho"] = 0.38  # Just below threshold

        # Low bias - more likely to BLOCK
        result_low_bias = build_global_alignment_view(**signals, cycle=1, allow_bias=0.0)

        # High bias - more likely to ALLOW
        result_high_bias = build_global_alignment_view(**signals, cycle=1, allow_bias=100.0)

        # The high bias should have higher allow_score
        assert result_high_bias["fusion_result"]["allow_score"] > result_low_bias["fusion_result"]["allow_score"]

    def test_hard_block_ignores_voting(self, all_healthy_signals):
        """HARD_BLOCK bypasses weighted voting."""
        signals = copy.deepcopy(all_healthy_signals)
        signals["identity"]["block_hash_valid"] = False

        # Even with high allow bias, HARD_BLOCK wins
        result = build_global_alignment_view(**signals, cycle=1, allow_bias=1000.0)

        assert result["fusion_result"]["decision"] == GovernanceAction.BLOCK
        assert result["fusion_result"]["is_hard"] is True


# =============================================================================
# Test: Signal Validation
# =============================================================================

class TestSignalValidation:
    """Test signal validation logic."""

    def test_missing_signal(self):
        """Missing signal is invalid."""
        validation = _validate_signal("topology", None)
        assert not validation.valid
        assert "missing" in validation.reason

    def test_invalid_type(self):
        """Non-dict signal is invalid."""
        validation = _validate_signal("topology", "not a dict")
        assert not validation.valid
        assert "not a dict" in validation.reason

    def test_explicitly_invalid(self):
        """Signal with valid=False is invalid."""
        validation = _validate_signal("topology", {"valid": False})
        assert not validation.valid
        assert "marked invalid" in validation.reason

    def test_valid_signal(self, healthy_topology):
        """Valid signal passes validation."""
        validation = _validate_signal("topology", healthy_topology)
        assert validation.valid


# =============================================================================
# Test: Output Schema Compliance
# =============================================================================

class TestOutputSchema:
    """Test output conforms to expected schema."""

    def test_required_fields_present(self, all_healthy_signals):
        """All required fields are present in output."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        required_fields = [
            "schema_version",
            "timestamp",
            "cycle",
            "mode",
            "signals",
            "fusion_result",
            "escalation",
            "conflict_detections",
            "recommendations",
            "metadata",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_fusion_result_structure(self, all_healthy_signals):
        """Fusion result has correct structure."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        fusion = result["fusion_result"]
        assert "decision" in fusion
        assert "is_hard" in fusion
        assert "primary_reason" in fusion
        assert "block_score" in fusion
        assert "allow_score" in fusion

    def test_escalation_structure(self, all_healthy_signals):
        """Escalation state has correct structure."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        escalation = result["escalation"]
        assert "level" in escalation
        assert "level_name" in escalation
        assert "trigger_reason" in escalation

    def test_json_serializable(self, all_healthy_signals):
        """Output is JSON serializable."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["cycle"] == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_signals(self):
        """Handle case with no signals provided."""
        result = build_global_alignment_view(cycle=1)

        # Should still produce valid output with warnings
        assert result["escalation"]["level"] >= EscalationLevel.L1_WARNING
        assert result["metadata"]["signals_received"] == 0

    def test_partial_signals(self, healthy_topology, healthy_identity):
        """Handle case with only some signals provided."""
        result = build_global_alignment_view(
            topology=healthy_topology,
            identity=healthy_identity,
            cycle=1,
        )

        assert result["metadata"]["signals_received"] == 2
        assert result["metadata"]["signals_valid"] == 2

    def test_cycle_zero(self, all_healthy_signals):
        """Cycle 0 is valid."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=0)
        assert result["cycle"] == 0

    def test_large_cycle_number(self, all_healthy_signals):
        """Large cycle numbers work correctly."""
        result = build_global_alignment_view(**all_healthy_signals, cycle=1_000_000)
        assert result["cycle"] == 1_000_000
