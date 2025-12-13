"""
Smoke tests for P5 Divergence Diagnostic Panel.

Tests the 8-rule deterministic rule engine specified in
docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md

Test Cases:
    1. test_smoke_nominal_case - All systems healthy -> NOMINAL
    2. test_smoke_replay_divergent_case - Replay WARN, stable env -> REPLAY_FAILURE
    3. test_smoke_topology_critical_case - Replay OK, Topology TURBULENT -> STRUCTURAL_BREAK
    4. test_smoke_identity_violation_case - Identity signal BLOCK -> IDENTITY_VIOLATION
    5. test_smoke_cascading_failure_case - Multiple BLOCK/CRITICAL -> CASCADING_FAILURE

SHADOW MODE CONTRACT:
    - These tests validate observational behavior only
    - No actual governance decisions are influenced
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def nominal_divergence_snapshot() -> Dict[str, Any]:
    """Nominal divergence snapshot - no issues."""
    return {
        "cycle": 100,
        "severity": "NONE",
        "type": "NONE",
        "divergence_pct": 0.005,
        "H_diff": 0.01,
        "rho_diff": 0.005,
    }


@pytest.fixture
def warn_divergence_snapshot() -> Dict[str, Any]:
    """WARN-level divergence snapshot."""
    return {
        "cycle": 200,
        "severity": "WARN",
        "type": "STATE",
        "divergence_pct": 0.12,
        "H_diff": 0.08,
        "rho_diff": 0.05,
    }


@pytest.fixture
def critical_divergence_snapshot() -> Dict[str, Any]:
    """CRITICAL-level divergence snapshot."""
    return {
        "cycle": 300,
        "severity": "CRITICAL",
        "type": "COMBINED",
        "divergence_pct": 0.35,
        "H_diff": 0.25,
        "rho_diff": 0.20,
    }


@pytest.fixture
def healthy_replay_signal() -> Dict[str, Any]:
    """Healthy replay signal - OK status."""
    return {
        "status": "OK",
        "governance_alignment": "aligned",
        "conflict": False,
        "reasons": ["[Safety] All checks passed"],
    }


@pytest.fixture
def warn_replay_signal() -> Dict[str, Any]:
    """WARN-level replay signal."""
    return {
        "status": "WARN",
        "governance_alignment": "tension",
        "conflict": False,
        "reasons": ["[Safety] Minor drift detected"],
    }


@pytest.fixture
def block_replay_signal() -> Dict[str, Any]:
    """BLOCK-level replay signal."""
    return {
        "status": "BLOCK",
        "governance_alignment": "divergent",
        "conflict": True,
        "reasons": ["[Safety] Hash mismatch detected", "[Radar] Governance drift critical"],
    }


@pytest.fixture
def stable_topology_signal() -> Dict[str, Any]:
    """Stable topology signal."""
    return {
        "mode": "STABLE",
        "persistence_drift": 0.02,
        "betti_0": 1,
        "betti_1": 0,
        "within_omega": True,
    }


@pytest.fixture
def turbulent_topology_signal() -> Dict[str, Any]:
    """Turbulent topology signal."""
    return {
        "mode": "TURBULENT",
        "persistence_drift": 0.18,
        "betti_0": 1,
        "betti_1": 2,
        "within_omega": True,
    }


@pytest.fixture
def critical_topology_signal() -> Dict[str, Any]:
    """Critical topology signal."""
    return {
        "mode": "CRITICAL",
        "persistence_drift": 0.35,
        "betti_0": 3,
        "betti_1": 4,
        "within_omega": False,
    }


@pytest.fixture
def stable_budget_signal() -> Dict[str, Any]:
    """Stable budget signal."""
    return {
        "stability_class": "STABLE",
        "health_score": 95,
        "stability_index": 0.98,
    }


@pytest.fixture
def volatile_budget_signal() -> Dict[str, Any]:
    """Volatile budget signal."""
    return {
        "stability_class": "VOLATILE",
        "health_score": 45,
        "stability_index": 0.35,
    }


@pytest.fixture
def healthy_identity_signal() -> Dict[str, Any]:
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
def failed_identity_signal() -> Dict[str, Any]:
    """Failed identity signal - hash invalid."""
    return {
        "block_hash_valid": False,
        "merkle_root_valid": True,
        "signature_valid": True,
        "chain_continuous": True,
        "pq_attestation_valid": True,
        "dual_root_consistent": True,
    }


@pytest.fixture
def healthy_structure_signal() -> Dict[str, Any]:
    """Healthy structure signal."""
    return {
        "dag_coherent": True,
        "cycle_detected": False,
        "orphan_count": 0,
    }


# =============================================================================
# Test 1: Nominal Case
# =============================================================================


class TestSmokeNominalCase:
    """Smoke test: All systems nominal produces NOMINAL diagnosis."""

    def test_nominal_hypothesis(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """All healthy signals -> NOMINAL hypothesis."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "NOMINAL"
        assert result["root_cause_confidence"] == 1.0

    def test_nominal_action_logged_only(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """NOMINAL case -> LOGGED_ONLY action."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["action"] == "LOGGED_ONLY"
        assert "healthy" in result["headline"].lower()

    def test_nominal_schema_version(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Output includes schema version."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["schema_version"] == "1.0.0"


# =============================================================================
# Test 2: Replay Divergent Case (Rule 7 - soft REPLAY_FAILURE)
# =============================================================================


class TestSmokeReplayDivergentCase:
    """Smoke test: Replay WARN + stable env produces REPLAY_FAILURE."""

    def test_replay_warn_stable_env(
        self,
        nominal_divergence_snapshot,
        warn_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Replay WARN in stable environment -> REPLAY_FAILURE."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=warn_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "REPLAY_FAILURE"
        assert result["root_cause_confidence"] == 0.70

    def test_replay_warn_review_recommended(
        self,
        nominal_divergence_snapshot,
        warn_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Replay WARN -> REVIEW_RECOMMENDED (not INVESTIGATION_REQUIRED)."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=warn_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["action"] == "REVIEW_RECOMMENDED"

    def test_replay_block_investigation_required(
        self,
        nominal_divergence_snapshot,
        block_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Replay BLOCK in stable environment -> INVESTIGATION_REQUIRED."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "REPLAY_FAILURE"
        assert result["action"] == "INVESTIGATION_REQUIRED"
        assert result["root_cause_confidence"] == 0.95


# =============================================================================
# Test 3: Topology Critical Case (Rule 3 - STRUCTURAL_BREAK)
# =============================================================================


class TestSmokeTopologyCriticalCase:
    """Smoke test: Replay OK + Topology TURBULENT produces STRUCTURAL_BREAK."""

    def test_structural_break_hypothesis(
        self,
        warn_divergence_snapshot,
        healthy_replay_signal,
        turbulent_topology_signal,
        stable_budget_signal,
    ):
        """Replay OK + Topology TURBULENT + STATE divergence -> STRUCTURAL_BREAK."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=warn_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=turbulent_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "STRUCTURAL_BREAK"
        assert result["root_cause_confidence"] == 0.85

    def test_structural_break_action(
        self,
        warn_divergence_snapshot,
        healthy_replay_signal,
        turbulent_topology_signal,
        stable_budget_signal,
    ):
        """STRUCTURAL_BREAK -> REVIEW_RECOMMENDED."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=warn_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=turbulent_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["action"] == "REVIEW_RECOMMENDED"
        assert "TURBULENT" in result["headline"]

    def test_structural_break_with_critical_topology(
        self,
        warn_divergence_snapshot,
        healthy_replay_signal,
        critical_topology_signal,
        stable_budget_signal,
    ):
        """Topology CRITICAL also triggers STRUCTURAL_BREAK."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=warn_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "STRUCTURAL_BREAK"
        assert "CRITICAL" in result["headline"]


# =============================================================================
# Test 4: Identity Violation Case (Priority 1 - short circuits)
# =============================================================================


class TestSmokeIdentityViolationCase:
    """Smoke test: Identity signal BLOCK produces IDENTITY_VIOLATION."""

    def test_identity_violation_hypothesis(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
        failed_identity_signal,
    ):
        """Identity BLOCK -> IDENTITY_VIOLATION (priority 1)."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
            identity_signal=failed_identity_signal,
        )

        assert result["root_cause_hypothesis"] == "IDENTITY_VIOLATION"
        assert result["root_cause_confidence"] == 1.0

    def test_identity_violation_investigation_required(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
        failed_identity_signal,
    ):
        """IDENTITY_VIOLATION -> INVESTIGATION_REQUIRED."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
            identity_signal=failed_identity_signal,
        )

        assert result["action"] == "INVESTIGATION_REQUIRED"
        assert "cryptographic" in result["headline"].lower()

    def test_identity_short_circuits_other_rules(
        self,
        critical_divergence_snapshot,
        block_replay_signal,
        critical_topology_signal,
        volatile_budget_signal,
        failed_identity_signal,
    ):
        """Identity violation short-circuits all other rules."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        # Even with all other signals in critical state,
        # identity violation takes precedence
        result = interpret_p5_divergence(
            divergence_snapshot=critical_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=volatile_budget_signal,
            identity_signal=failed_identity_signal,
        )

        assert result["root_cause_hypothesis"] == "IDENTITY_VIOLATION"


# =============================================================================
# Test 5: Cascading Failure Case (Rule 6)
# =============================================================================


class TestSmokeCascadingFailureCase:
    """Smoke test: Multiple BLOCK/CRITICAL signals produces CASCADING_FAILURE."""

    def test_cascading_failure_hypothesis(
        self,
        critical_divergence_snapshot,
        block_replay_signal,
        critical_topology_signal,
        stable_budget_signal,
    ):
        """2+ BLOCK/CRITICAL signals -> CASCADING_FAILURE."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=critical_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "CASCADING_FAILURE"
        assert result["root_cause_confidence"] == 0.75

    def test_cascading_failure_investigation_required(
        self,
        critical_divergence_snapshot,
        block_replay_signal,
        critical_topology_signal,
        stable_budget_signal,
    ):
        """CASCADING_FAILURE -> INVESTIGATION_REQUIRED."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=critical_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["action"] == "INVESTIGATION_REQUIRED"
        assert "multiple" in result["headline"].lower()

    def test_cascading_with_confounding_factors(
        self,
        critical_divergence_snapshot,
        block_replay_signal,
        critical_topology_signal,
        stable_budget_signal,
    ):
        """Cascading failure includes confounding factors from topology."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=critical_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=stable_budget_signal,
        )

        assert result["root_cause_hypothesis"] == "CASCADING_FAILURE"
        # Topology CRITICAL adds "topology_unstable" to confounding factors
        assert "topology_unstable" in result["confounding_factors"]

    def test_budget_confound_takes_precedence_over_cascading(
        self,
        critical_divergence_snapshot,
        block_replay_signal,
        critical_topology_signal,
        volatile_budget_signal,
    ):
        """BUDGET_CONFOUND (Rule 5) takes precedence over CASCADING_FAILURE (Rule 6).

        Per spec Section 3.2, Rule 5 is evaluated before Rule 6.
        When budget is VOLATILE and divergence is WARN/CRITICAL, BUDGET_CONFOUND fires.
        """
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=critical_divergence_snapshot,
            replay_signal=block_replay_signal,
            topology_signal=critical_topology_signal,
            budget_signal=volatile_budget_signal,
        )

        # BUDGET_CONFOUND fires before CASCADING_FAILURE per rule order
        assert result["root_cause_hypothesis"] == "BUDGET_CONFOUND"
        assert "budget_volatile" in result["confounding_factors"]


# =============================================================================
# JSON Serializability Tests
# =============================================================================


class TestJSONSerializability:
    """Test that all outputs are JSON serializable."""

    def test_output_json_serializable(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Diagnostic output is JSON serializable."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        result = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"

    def test_all_hypotheses_json_serializable(
        self,
        nominal_divergence_snapshot,
        warn_divergence_snapshot,
        critical_divergence_snapshot,
        healthy_replay_signal,
        warn_replay_signal,
        block_replay_signal,
        stable_topology_signal,
        turbulent_topology_signal,
        critical_topology_signal,
        stable_budget_signal,
        volatile_budget_signal,
        failed_identity_signal,
    ):
        """All hypothesis types produce JSON-serializable output."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        test_cases = [
            # NOMINAL
            (
                nominal_divergence_snapshot,
                healthy_replay_signal,
                stable_topology_signal,
                stable_budget_signal,
                None,
            ),
            # REPLAY_FAILURE (soft)
            (
                nominal_divergence_snapshot,
                warn_replay_signal,
                stable_topology_signal,
                stable_budget_signal,
                None,
            ),
            # STRUCTURAL_BREAK
            (
                warn_divergence_snapshot,
                healthy_replay_signal,
                turbulent_topology_signal,
                stable_budget_signal,
                None,
            ),
            # IDENTITY_VIOLATION
            (
                nominal_divergence_snapshot,
                healthy_replay_signal,
                stable_topology_signal,
                stable_budget_signal,
                failed_identity_signal,
            ),
            # CASCADING_FAILURE
            (
                critical_divergence_snapshot,
                block_replay_signal,
                critical_topology_signal,
                stable_budget_signal,
                None,
            ),
        ]

        for div, rep, topo, budget, identity in test_cases:
            kwargs = {
                "divergence_snapshot": div,
                "replay_signal": rep,
                "topology_signal": topo,
                "budget_signal": budget,
            }
            if identity:
                kwargs["identity_signal"] = identity

            result = interpret_p5_divergence(**kwargs)
            json_str = json.dumps(result)
            assert isinstance(json_str, str), f"Failed for {result['root_cause_hypothesis']}"


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Test that the rule engine is deterministic."""

    def test_deterministic_output(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """Same inputs produce identical outputs (except timestamp)."""
        from backend.health.p5_divergence_interpreter import interpret_p5_divergence

        results = []
        for _ in range(5):
            result = interpret_p5_divergence(
                divergence_snapshot=nominal_divergence_snapshot,
                replay_signal=healthy_replay_signal,
                topology_signal=stable_topology_signal,
                budget_signal=stable_budget_signal,
                cycle=100,
            )
            # Remove timestamp for comparison
            result_copy = dict(result)
            del result_copy["timestamp"]
            results.append(json.dumps(result_copy, sort_keys=True))

        # All results should be identical
        assert len(set(results)) == 1


# =============================================================================
# Director Tile Integration Tests
# =============================================================================


class TestDirectorTileIntegration:
    """Test director tile and evidence attachment functions."""

    def test_build_p5_diagnostic_tile(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """build_p5_diagnostic_tile produces valid tile."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            build_p5_diagnostic_tile,
        )

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        tile = build_p5_diagnostic_tile(diagnostic)

        assert tile["tile_type"] == "p5_diagnostic"
        assert tile["schema_version"] == "1.0.0"
        assert tile["summary"]["hypothesis"] == "NOMINAL"
        assert tile["summary"]["severity_badge"] == "OK"
        assert "shadow_mode_notice" in tile

    def test_attach_p5_diagnostic_to_evidence(
        self,
        nominal_divergence_snapshot,
        healthy_replay_signal,
        stable_topology_signal,
        stable_budget_signal,
    ):
        """attach_p5_diagnostic_to_evidence adds p5_diagnostic key."""
        from backend.health.p5_divergence_interpreter import (
            interpret_p5_divergence,
            attach_p5_diagnostic_to_evidence,
        )

        diagnostic = interpret_p5_divergence(
            divergence_snapshot=nominal_divergence_snapshot,
            replay_signal=healthy_replay_signal,
            topology_signal=stable_topology_signal,
            budget_signal=stable_budget_signal,
        )

        evidence = {"existing_key": "existing_value"}
        updated = attach_p5_diagnostic_to_evidence(evidence, diagnostic)

        assert "p5_diagnostic" in updated
        assert updated["p5_diagnostic"]["shadow_mode"] is True
        assert updated["p5_diagnostic"]["root_cause_hypothesis"] == "NOMINAL"
        assert updated["existing_key"] == "existing_value"
