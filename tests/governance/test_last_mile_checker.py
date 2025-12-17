"""
Comprehensive tests for Last-Mile Governance Checker (CLAUDE K).

Test Categories:
1. Gate Combination Tests - All 6 gates in various states
2. Waiver/Override Handling Tests - Bypass mechanisms
3. Determinism Tests - Same inputs produce same outputs

SHADOW MODE CONTRACT:
- All tests verify that checker computes but does NOT enforce
- Mode is always tagged in results
"""

import json
import hashlib
import pytest
from datetime import datetime, timezone, timedelta
from typing import List

from backend.governance.last_mile_checker import (
    GovernanceFinalChecker,
    GovernanceFinalCheckInput,
    GovernanceFinalCheckResult,
    GovernanceFinalCheckConfig,
    GateResult,
    GateEvaluations,
    GovernanceWaiver,
    GovernanceOverride,
    TDAMetrics,
    GateId,
    GateStatus,
    Severity,
    Verdict,
    run_governance_final_check,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> GovernanceFinalCheckConfig:
    """Default configuration for tests."""
    return GovernanceFinalCheckConfig.default()


@pytest.fixture
def checker(default_config) -> GovernanceFinalChecker:
    """Fresh checker instance for each test."""
    return GovernanceFinalChecker(config=default_config)


@pytest.fixture
def healthy_input() -> GovernanceFinalCheckInput:
    """Input representing a healthy system state."""
    return GovernanceFinalCheckInput(
        cycle=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        H=0.8,
        D=5,
        D_dot=0.5,
        B=2.0,
        S=0.1,
        C=0,  # CONVERGING
        rho=0.8,
        tau=0.2,
        J=2.0,
        W=False,
        beta=0.1,
        kappa=0.8,
        nu=0.001,
        delta=0,
        Gamma=0.9,
        hard_ok=True,
        hard_fail_streak=0,
        in_omega=True,
        omega_exit_streak=0,
        invariant_violations=[],
        invariant_all_pass=True,
        active_cdis=[],
        cdi_010_active=False,
        rho_collapse_streak=0,
        beta_explosion_streak=0,
    )


@pytest.fixture
def timestamp_now() -> str:
    """Current timestamp."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 1. GATE COMBINATION TESTS
# =============================================================================

class TestG0CatastrophicGate:
    """Tests for G0: Catastrophic Gate (CDI-010 detection)."""

    def test_g0_pass_no_cdi010(self, checker, healthy_input):
        """G0 passes when CDI-010 is not active."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g0_catastrophic.status == GateStatus.PASS
        assert result.verdict == Verdict.ALLOW

    def test_g0_fail_cdi010_active(self, checker, healthy_input):
        """G0 fails immediately when CDI-010 is active."""
        healthy_input.cdi_010_active = True
        healthy_input.active_cdis = ["CDI-010"]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g0_catastrophic.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G0_CATASTROPHIC"
        assert "CDI-010" in result.blocking_reason

    def test_g0_fail_blocks_even_with_healthy_metrics(self, checker, healthy_input):
        """G0 failure blocks regardless of all other healthy metrics."""
        # Everything else is perfect
        healthy_input.H = 1.0
        healthy_input.rho = 1.0
        healthy_input.beta = 0.0
        healthy_input.hard_ok = True
        healthy_input.in_omega = True

        # But CDI-010 is active
        healthy_input.cdi_010_active = True

        result = checker.run_governance_final_check(healthy_input)

        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G0_CATASTROPHIC"


class TestG1HardGate:
    """Tests for G1: Hard Gate (HARD_OK failure streak)."""

    def test_g1_pass_hard_ok_true(self, checker, healthy_input):
        """G1 passes when HARD mode is OK."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g1_hard.status == GateStatus.PASS

    def test_g1_pass_streak_below_threshold(self, checker, healthy_input):
        """G1 passes when streak is below threshold."""
        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 49  # Below default 50

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g1_hard.status == GateStatus.PASS
        assert result.verdict == Verdict.ALLOW

    def test_g1_fail_streak_exceeds_threshold(self, checker, healthy_input):
        """G1 fails when streak exceeds threshold."""
        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 51  # Above default 50

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g1_hard.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G1_HARD"

    def test_g1_custom_threshold(self, healthy_input):
        """G1 respects custom threshold configuration."""
        config = GovernanceFinalCheckConfig(hard_fail_threshold=10)
        checker = GovernanceFinalChecker(config=config)

        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 11

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g1_hard.status == GateStatus.FAIL


class TestG2InvariantGate:
    """Tests for G2: Invariant Gate (INV-001 through INV-008)."""

    def test_g2_pass_all_invariants_satisfied(self, checker, healthy_input):
        """G2 passes when all invariants are satisfied."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g2_invariant.status == GateStatus.PASS

    def test_g2_fail_invariant_violations(self, checker, healthy_input):
        """G2 fails when invariants are violated."""
        healthy_input.invariant_violations = ["INV-001", "INV-003"]
        healthy_input.invariant_all_pass = False

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g2_invariant.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G2_INVARIANT"

    def test_g2_tolerance_allows_some_violations(self, healthy_input):
        """G2 passes when violations are within tolerance."""
        config = GovernanceFinalCheckConfig(invariant_tolerance=2)
        checker = GovernanceFinalChecker(config=config)

        healthy_input.invariant_violations = ["INV-001", "INV-002"]  # 2 violations
        healthy_input.invariant_all_pass = False

        result = checker.run_governance_final_check(healthy_input)

        # 2 violations == tolerance, so should still pass (> tolerance triggers fail)
        assert result.gates.g2_invariant.status == GateStatus.PASS

    def test_g2_waiver_bypasses_failure(self, checker, healthy_input, timestamp_now):
        """G2 can be waived."""
        healthy_input.invariant_violations = ["INV-001"]
        healthy_input.invariant_all_pass = False

        # Add waiver
        waiver = GovernanceWaiver(
            waiver_id="WAIVER-001",
            gate_id="G2_INVARIANT",
            issued_by="test_operator",
            issued_at=timestamp_now,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            justification="Test waiver",
        )
        healthy_input.waivers = [waiver]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g2_invariant.status == GateStatus.WAIVED
        assert result.verdict == Verdict.ALLOW
        assert "WAIVER-001" in result.waivers_applied


class TestG3SafeRegionGate:
    """Tests for G3: Safe Region Gate (Ω membership)."""

    def test_g3_pass_in_omega(self, checker, healthy_input):
        """G3 passes when state is in safe region."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g3_safe_region.status == GateStatus.PASS

    def test_g3_pass_exit_streak_below_threshold(self, checker, healthy_input):
        """G3 passes when exit streak is below threshold."""
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 99  # Below default 100

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g3_safe_region.status == GateStatus.PASS

    def test_g3_fail_exit_streak_exceeds_threshold(self, checker, healthy_input):
        """G3 fails when exit streak exceeds threshold."""
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 101  # Above default 100

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g3_safe_region.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G3_SAFE_REGION"

    def test_g3_waiver_bypasses_failure(self, checker, healthy_input, timestamp_now):
        """G3 can be waived."""
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 150

        waiver = GovernanceWaiver(
            waiver_id="WAIVER-002",
            gate_id="G3_SAFE_REGION",
            issued_by="test_operator",
            issued_at=timestamp_now,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            justification="Safe region waiver",
        )
        healthy_input.waivers = [waiver]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g3_safe_region.status == GateStatus.WAIVED
        assert result.verdict == Verdict.ALLOW


class TestG4SoftGate:
    """Tests for G4: Soft Gate (RSI collapse or block rate explosion)."""

    def test_g4_pass_healthy_metrics(self, checker, healthy_input):
        """G4 passes with healthy RSI and block rate."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g4_soft.status == GateStatus.PASS

    def test_g4_fail_rho_collapse(self, checker, healthy_input):
        """G4 fails on RSI collapse."""
        healthy_input.rho = 0.3  # Below default 0.4
        healthy_input.rho_collapse_streak = 15  # Above default 10

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g4_soft.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert result.blocking_gate == "G4_SOFT"
        assert "RSI collapse" in result.blocking_reason

    def test_g4_fail_beta_explosion(self, checker, healthy_input):
        """G4 fails on block rate explosion."""
        healthy_input.beta = 0.7  # Above default 0.6
        healthy_input.beta_explosion_streak = 25  # Above default 20

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g4_soft.status == GateStatus.FAIL
        assert result.verdict == Verdict.BLOCK
        assert "Block rate explosion" in result.blocking_reason

    def test_g4_override_bypasses_failure(self, checker, healthy_input, timestamp_now):
        """G4 can be overridden."""
        healthy_input.rho = 0.2
        healthy_input.rho_collapse_streak = 20

        override = GovernanceOverride(
            override_id="OVERRIDE-001",
            gate_id="G4_SOFT",
            issued_by="test_operator",
            issued_at=timestamp_now,
            reason="Test override",
            valid_for_cycles=50,
        )
        healthy_input.overrides = [override]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g4_soft.status == GateStatus.OVERRIDDEN
        assert result.verdict == Verdict.ALLOW
        assert "OVERRIDE-001" in result.overrides_applied


class TestG5AdvisoryGate:
    """Tests for G5: Advisory Gate (TDA metrics degradation)."""

    def test_g5_pass_no_tda_metrics(self, checker, healthy_input):
        """G5 passes when TDA metrics are not available."""
        healthy_input.tda_metrics = None

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g5_advisory.status == GateStatus.PASS

    def test_g5_pass_healthy_tda_metrics(self, checker, healthy_input):
        """G5 passes with healthy TDA metrics."""
        healthy_input.tda_metrics = TDAMetrics(
            sns=0.7,
            pcs=0.8,
            drs=0.1,
            hss=0.7,
        )

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g5_advisory.status == GateStatus.PASS
        assert "ADVISORY" not in (result.gates.g5_advisory.details or "")

    def test_g5_advisory_degraded_metrics(self, checker, healthy_input):
        """G5 generates advisory for degraded TDA metrics but never blocks."""
        healthy_input.tda_metrics = TDAMetrics(
            sns=0.3,  # Below threshold
            pcs=0.4,  # Below threshold
            drs=0.5,  # Above threshold
            hss=0.3,  # Below threshold
        )

        result = checker.run_governance_final_check(healthy_input)

        # G5 never blocks - always PASS
        assert result.gates.g5_advisory.status == GateStatus.PASS
        assert result.verdict == Verdict.ALLOW
        # But should have advisory in details
        assert "ADVISORY" in result.gates.g5_advisory.details


class TestGatePrecedence:
    """Tests for gate evaluation precedence."""

    def test_g0_blocks_before_g1(self, checker, healthy_input):
        """G0 failure takes precedence over G1 failure."""
        healthy_input.cdi_010_active = True
        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 100

        result = checker.run_governance_final_check(healthy_input)

        assert result.blocking_gate == "G0_CATASTROPHIC"

    def test_g1_blocks_before_g2(self, checker, healthy_input):
        """G1 failure takes precedence over G2 failure."""
        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 100
        healthy_input.invariant_violations = ["INV-001"]

        result = checker.run_governance_final_check(healthy_input)

        assert result.blocking_gate == "G1_HARD"

    def test_g2_blocks_before_g3(self, checker, healthy_input):
        """G2 failure takes precedence over G3 failure."""
        healthy_input.invariant_violations = ["INV-001"]
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 200

        result = checker.run_governance_final_check(healthy_input)

        assert result.blocking_gate == "G2_INVARIANT"

    def test_g3_blocks_before_g4(self, checker, healthy_input):
        """G3 failure takes precedence over G4 failure."""
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 200
        healthy_input.rho = 0.1
        healthy_input.rho_collapse_streak = 50

        result = checker.run_governance_final_check(healthy_input)

        assert result.blocking_gate == "G3_SAFE_REGION"


# =============================================================================
# 2. WAIVER/OVERRIDE HANDLING TESTS
# =============================================================================

class TestWaiverExpiration:
    """Tests for waiver expiration handling."""

    def test_expired_waiver_not_applied(self, checker, healthy_input):
        """Expired waivers are not applied."""
        healthy_input.invariant_violations = ["INV-001"]

        # Expired waiver
        waiver = GovernanceWaiver(
            waiver_id="EXPIRED-WAIVER",
            gate_id="G2_INVARIANT",
            issued_by="test",
            issued_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            expires_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            justification="Expired",
        )
        healthy_input.waivers = [waiver]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g2_invariant.status == GateStatus.FAIL
        assert len(result.waivers_applied) == 0

    def test_waiver_cycle_limit(self, healthy_input):
        """Waivers respect max_cycles limit."""
        checker = GovernanceFinalChecker()

        healthy_input.invariant_violations = ["INV-001"]

        waiver = GovernanceWaiver(
            waiver_id="LIMITED-WAIVER",
            gate_id="G2_INVARIANT",
            issued_by="test",
            issued_at=datetime.now(timezone.utc).isoformat(),
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            justification="Limited cycles",
            max_cycles=5,
        )

        # First check - waiver should apply
        healthy_input.cycle = 1
        healthy_input.waivers = [waiver]
        result1 = checker.run_governance_final_check(healthy_input)
        assert result1.gates.g2_invariant.status == GateStatus.WAIVED

        # Sixth check - waiver should NOT apply (exceeded max_cycles)
        healthy_input.cycle = 7
        result2 = checker.run_governance_final_check(healthy_input)
        assert result2.gates.g2_invariant.status == GateStatus.FAIL


class TestOverrideValidity:
    """Tests for override validity handling."""

    def test_override_valid_within_cycles(self, healthy_input):
        """Override valid within specified cycles."""
        checker = GovernanceFinalChecker()

        healthy_input.rho = 0.2
        healthy_input.rho_collapse_streak = 20

        override = GovernanceOverride(
            override_id="VALID-OVERRIDE",
            gate_id="G4_SOFT",
            issued_by="test",
            issued_at=datetime.now(timezone.utc).isoformat(),
            reason="Test",
            valid_for_cycles=10,
        )
        healthy_input.overrides = [override]

        # First check
        healthy_input.cycle = 1
        result1 = checker.run_governance_final_check(healthy_input)
        assert result1.gates.g4_soft.status == GateStatus.OVERRIDDEN

        # 5th check - still valid
        healthy_input.cycle = 5
        result2 = checker.run_governance_final_check(healthy_input)
        assert result2.gates.g4_soft.status == GateStatus.OVERRIDDEN

    def test_override_expires_after_cycles(self, healthy_input):
        """Override expires after specified cycles."""
        checker = GovernanceFinalChecker()

        healthy_input.rho = 0.2
        healthy_input.rho_collapse_streak = 20

        override = GovernanceOverride(
            override_id="EXPIRING-OVERRIDE",
            gate_id="G4_SOFT",
            issued_by="test",
            issued_at=datetime.now(timezone.utc).isoformat(),
            reason="Test",
            valid_for_cycles=5,
        )
        healthy_input.overrides = [override]

        # First check - starts tracking
        healthy_input.cycle = 1
        result1 = checker.run_governance_final_check(healthy_input)
        assert result1.gates.g4_soft.status == GateStatus.OVERRIDDEN

        # 10th check - override should be expired
        healthy_input.cycle = 10
        result2 = checker.run_governance_final_check(healthy_input)
        assert result2.gates.g4_soft.status == GateStatus.FAIL


class TestWrongGateWaiverOverride:
    """Tests that waivers/overrides only apply to correct gates."""

    def test_g2_waiver_doesnt_affect_g3(self, checker, healthy_input, timestamp_now):
        """G2 waiver doesn't bypass G3 failure."""
        healthy_input.in_omega = False
        healthy_input.omega_exit_streak = 200

        # G2 waiver (wrong gate)
        waiver = GovernanceWaiver(
            waiver_id="WRONG-GATE-WAIVER",
            gate_id="G2_INVARIANT",
            issued_by="test",
            issued_at=timestamp_now,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            justification="Wrong gate",
        )
        healthy_input.waivers = [waiver]

        result = checker.run_governance_final_check(healthy_input)

        assert result.gates.g3_safe_region.status == GateStatus.FAIL
        assert result.blocking_gate == "G3_SAFE_REGION"

    def test_override_only_applies_to_g4(self, checker, healthy_input, timestamp_now):
        """Overrides only apply to G4."""
        healthy_input.invariant_violations = ["INV-001"]

        # Override (only valid for G4)
        override = GovernanceOverride(
            override_id="G4-ONLY-OVERRIDE",
            gate_id="G4_SOFT",
            issued_by="test",
            issued_at=timestamp_now,
            reason="G4 only",
        )
        healthy_input.overrides = [override]

        result = checker.run_governance_final_check(healthy_input)

        # G2 should still fail
        assert result.gates.g2_invariant.status == GateStatus.FAIL


# =============================================================================
# 3. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests ensuring deterministic behavior."""

    def test_same_input_same_output(self, default_config, healthy_input):
        """Same input produces same output."""
        checker1 = GovernanceFinalChecker(config=default_config)
        checker2 = GovernanceFinalChecker(config=default_config)

        result1 = checker1.run_governance_final_check(healthy_input)
        result2 = checker2.run_governance_final_check(healthy_input)

        assert result1.verdict == result2.verdict
        assert result1.blocking_gate == result2.blocking_gate
        assert result1.gates.g0_catastrophic.status == result2.gates.g0_catastrophic.status
        assert result1.gates.g1_hard.status == result2.gates.g1_hard.status
        assert result1.gates.g2_invariant.status == result2.gates.g2_invariant.status
        assert result1.gates.g3_safe_region.status == result2.gates.g3_safe_region.status
        assert result1.gates.g4_soft.status == result2.gates.g4_soft.status
        assert result1.gates.g5_advisory.status == result2.gates.g5_advisory.status

    def test_input_hash_determinism(self, healthy_input):
        """Input hash is deterministic."""
        hash1 = healthy_input.compute_hash()
        hash2 = healthy_input.compute_hash()

        assert hash1 == hash2
        assert hash1.startswith("sha256:")

    def test_output_hash_determinism(self, checker, healthy_input):
        """Output hash is deterministic for same result."""
        result = checker.run_governance_final_check(healthy_input)

        hash1 = result.compute_output_hash()
        hash2 = result.compute_output_hash()

        assert hash1 == hash2

    def test_different_inputs_different_hashes(self, healthy_input):
        """Different inputs produce different hashes."""
        input1 = healthy_input

        # Create slightly different input
        input2 = GovernanceFinalCheckInput(
            cycle=healthy_input.cycle,
            timestamp=healthy_input.timestamp,
            H=0.9,  # Different
            D=healthy_input.D,
            hard_ok=healthy_input.hard_ok,
            hard_fail_streak=healthy_input.hard_fail_streak,
            in_omega=healthy_input.in_omega,
            omega_exit_streak=healthy_input.omega_exit_streak,
        )

        hash1 = input1.compute_hash()
        hash2 = input2.compute_hash()

        assert hash1 != hash2


class TestAuditChain:
    """Tests for audit chain integrity."""

    def test_chain_height_increments(self, checker, healthy_input):
        """Chain height increments with each check."""
        result1 = checker.run_governance_final_check(healthy_input)
        assert result1.chain_height == 1

        healthy_input.cycle = 2
        result2 = checker.run_governance_final_check(healthy_input)
        assert result2.chain_height == 2

        healthy_input.cycle = 3
        result3 = checker.run_governance_final_check(healthy_input)
        assert result3.chain_height == 3

    def test_previous_hash_links(self, checker, healthy_input):
        """Previous hash links correctly."""
        result1 = checker.run_governance_final_check(healthy_input)
        assert result1.previous_check_hash is None  # First check

        healthy_input.cycle = 2
        result2 = checker.run_governance_final_check(healthy_input)
        assert result2.previous_check_hash == result1.output_hash

    def test_reset_clears_chain(self, checker, healthy_input):
        """Reset clears audit chain state."""
        checker.run_governance_final_check(healthy_input)
        checker.run_governance_final_check(healthy_input)

        chain_info = checker.get_chain_info()
        assert chain_info["chain_height"] == 2

        checker.reset()

        chain_info = checker.get_chain_info()
        assert chain_info["chain_height"] == 0
        assert chain_info["previous_check_hash"] is None


# =============================================================================
# 4. SHADOW MODE TESTS
# =============================================================================

class TestShadowMode:
    """Tests for SHADOW MODE contract compliance."""

    def test_shadow_mode_tagged_in_result(self, checker, healthy_input):
        """Results are tagged with SHADOW mode."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.mode == "SHADOW"

    def test_active_mode_when_configured(self, healthy_input):
        """Results are tagged with ACTIVE mode when configured."""
        config = GovernanceFinalCheckConfig(shadow_mode=False)
        checker = GovernanceFinalChecker(config=config)

        result = checker.run_governance_final_check(healthy_input)

        assert result.mode == "ACTIVE"

    def test_verdict_computed_regardless_of_mode(self, healthy_input):
        """Verdict is computed in both modes."""
        shadow_config = GovernanceFinalCheckConfig(shadow_mode=True)
        active_config = GovernanceFinalCheckConfig(shadow_mode=False)

        shadow_checker = GovernanceFinalChecker(config=shadow_config)
        active_checker = GovernanceFinalChecker(config=active_config)

        shadow_result = shadow_checker.run_governance_final_check(healthy_input)
        active_result = active_checker.run_governance_final_check(healthy_input)

        # Same verdict regardless of mode
        assert shadow_result.verdict == active_result.verdict


# =============================================================================
# 5. CONFIDENCE SCORE TESTS
# =============================================================================

class TestConfidenceScore:
    """Tests for verdict confidence scoring."""

    def test_high_confidence_clean_pass(self, checker, healthy_input):
        """High confidence for clean pass with no issues."""
        result = checker.run_governance_final_check(healthy_input)

        assert result.verdict_confidence >= 0.9

    def test_reduced_confidence_with_waiver(self, checker, healthy_input, timestamp_now):
        """Reduced confidence when waiver is applied."""
        healthy_input.invariant_violations = ["INV-001"]

        waiver = GovernanceWaiver(
            waiver_id="WAIVER",
            gate_id="G2_INVARIANT",
            issued_by="test",
            issued_at=timestamp_now,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            justification="Test",
        )
        healthy_input.waivers = [waiver]

        result = checker.run_governance_final_check(healthy_input)

        assert result.verdict == Verdict.ALLOW
        assert result.verdict_confidence < 1.0

    def test_reduced_confidence_with_override(self, checker, healthy_input, timestamp_now):
        """Reduced confidence when override is applied."""
        healthy_input.rho = 0.2
        healthy_input.rho_collapse_streak = 20

        override = GovernanceOverride(
            override_id="OVERRIDE",
            gate_id="G4_SOFT",
            issued_by="test",
            issued_at=timestamp_now,
            reason="Test",
        )
        healthy_input.overrides = [override]

        result = checker.run_governance_final_check(healthy_input)

        assert result.verdict == Verdict.ALLOW
        assert result.verdict_confidence < 0.9


# =============================================================================
# 6. CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Tests for run_governance_final_check convenience function."""

    def test_convenience_function_works(self, healthy_input):
        """Convenience function produces valid result."""
        result = run_governance_final_check(healthy_input)

        assert isinstance(result, GovernanceFinalCheckResult)
        assert result.verdict in (Verdict.ALLOW, Verdict.BLOCK)
        assert result.gates is not None

    def test_convenience_function_accepts_config(self, healthy_input):
        """Convenience function accepts custom config."""
        config = GovernanceFinalCheckConfig(hard_fail_threshold=5)

        healthy_input.hard_ok = False
        healthy_input.hard_fail_streak = 10

        result = run_governance_final_check(healthy_input, config=config)

        assert result.gates.g1_hard.status == GateStatus.FAIL
