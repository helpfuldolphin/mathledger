"""
RFL Law Determinism Tests
=========================

These tests demonstrate that given a specific H_t with a known abstention profile,
the exact RFL ledger update is uniquely determined.

Test Protocol:
    1. Construct AttestedRunContext with specific H_t and abstention profile
    2. Invoke RFLRunner.run_with_attestation()
    3. Assert ledger entry matches expected deterministic values
    4. Repeat N times to verify stability

Per RFL Law (docs/RFL_LAW.md):
    - step_id = SHA256(experiment_id | slice_name | policy_id | H_t)
    - ∇_sym = -(α_rate - τ)
    - reward = max(0, 1 - α_rate)
    - Δα = α_mass - (τ × attempt_mass)
"""

from __future__ import annotations

import hashlib
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rfl.config import RFLConfig
from rfl.runner import RFLRunner, RflResult, RunLedgerEntry
from rfl.audit import (
    SymbolicDescentGradient,
    StepIdComputation,
    AuditEntry,
    RFLAuditLog,
    verify_rfl_transformation,
)

# Import from the same location as the runner uses
try:
    from substrate.bridge.context import AttestedRunContext
except ImportError:
    from backend.bridge.context import AttestedRunContext


# ---------------------------------------------------------------------------
# Test Data: Canonical H_t Scenarios
# ---------------------------------------------------------------------------

@dataclass
class RFLTestCase:
    """Test case for RFL Law verification."""
    name: str
    description: str

    # Input H_t and abstention profile
    h_t: str
    r_t: str
    u_t: str
    alpha_rate: float      # abstention rate [0,1]
    alpha_mass: float      # abstention mass (count)
    attempt_mass: float    # total attempts
    tolerance: float       # τ

    # Expected outputs (computed per RFL Law)
    @property
    def expected_mass_delta(self) -> float:
        """Δα = α_mass - (τ × attempt_mass)"""
        return self.alpha_mass - (self.tolerance * self.attempt_mass)

    @property
    def expected_rate_delta(self) -> float:
        """α_rate - τ"""
        return self.alpha_rate - self.tolerance

    @property
    def expected_symbolic_descent(self) -> float:
        """∇_sym = -(α_rate - τ)"""
        return -self.expected_rate_delta

    @property
    def expected_policy_reward(self) -> float:
        """r = max(0, 1 - α_rate)"""
        return max(0.0, 1.0 - max(self.alpha_rate, 0.0))

    @property
    def expected_policy_update(self) -> bool:
        """Update triggered if |Δα| > ε or |α_rate - τ| > ε"""
        return abs(self.expected_mass_delta) > 1e-9 or abs(self.expected_rate_delta) > 1e-9


# Generate deterministic H_t values for tests
def _make_h_t(seed: str) -> Tuple[str, str, str]:
    """Generate deterministic (H_t, R_t, U_t) from seed."""
    r_t = hashlib.sha256(f"{seed}:reasoning".encode()).hexdigest()
    u_t = hashlib.sha256(f"{seed}:ui".encode()).hexdigest()
    h_t = hashlib.sha256((r_t + u_t).encode()).hexdigest()
    return h_t, r_t, u_t


# Test cases covering the full spectrum of abstention profiles
TEST_CASES = [
    # Case 1: High abstention (35% > 10% tolerance)
    RFLTestCase(
        name="high_abstention",
        description="Abstention rate 35% exceeds 10% tolerance → negative descent",
        h_t=_make_h_t("high_abstention")[0],
        r_t=_make_h_t("high_abstention")[1],
        u_t=_make_h_t("high_abstention")[2],
        alpha_rate=0.35,
        alpha_mass=7.0,
        attempt_mass=20.0,
        tolerance=0.10,
    ),

    # Case 2: Zero abstention (perfect)
    RFLTestCase(
        name="zero_abstention",
        description="Zero abstention → positive descent, full reward",
        h_t=_make_h_t("zero_abstention")[0],
        r_t=_make_h_t("zero_abstention")[1],
        u_t=_make_h_t("zero_abstention")[2],
        alpha_rate=0.0,
        alpha_mass=0.0,
        attempt_mass=20.0,
        tolerance=0.10,
    ),

    # Case 3: At tolerance boundary (exactly 10%)
    RFLTestCase(
        name="at_tolerance",
        description="Abstention exactly at tolerance → zero descent",
        h_t=_make_h_t("at_tolerance")[0],
        r_t=_make_h_t("at_tolerance")[1],
        u_t=_make_h_t("at_tolerance")[2],
        alpha_rate=0.10,
        alpha_mass=2.0,
        attempt_mass=20.0,
        tolerance=0.10,
    ),

    # Case 4: Below tolerance (5% < 10%)
    RFLTestCase(
        name="below_tolerance",
        description="Abstention below tolerance → positive descent",
        h_t=_make_h_t("below_tolerance")[0],
        r_t=_make_h_t("below_tolerance")[1],
        u_t=_make_h_t("below_tolerance")[2],
        alpha_rate=0.05,
        alpha_mass=1.0,
        attempt_mass=20.0,
        tolerance=0.10,
    ),

    # Case 5: Very high abstention (80%)
    RFLTestCase(
        name="very_high_abstention",
        description="Very high abstention → large negative descent, low reward",
        h_t=_make_h_t("very_high_abstention")[0],
        r_t=_make_h_t("very_high_abstention")[1],
        u_t=_make_h_t("very_high_abstention")[2],
        alpha_rate=0.80,
        alpha_mass=16.0,
        attempt_mass=20.0,
        tolerance=0.10,
    ),

    # Case 6: Large scale (1000 attempts)
    RFLTestCase(
        name="large_scale",
        description="Large scale with 15% abstention",
        h_t=_make_h_t("large_scale")[0],
        r_t=_make_h_t("large_scale")[1],
        u_t=_make_h_t("large_scale")[2],
        alpha_rate=0.15,
        alpha_mass=150.0,
        attempt_mass=1000.0,
        tolerance=0.10,
    ),

    # Case 7: Tight tolerance (1%)
    RFLTestCase(
        name="tight_tolerance",
        description="Tight 1% tolerance with 2% abstention",
        h_t=_make_h_t("tight_tolerance")[0],
        r_t=_make_h_t("tight_tolerance")[1],
        u_t=_make_h_t("tight_tolerance")[2],
        alpha_rate=0.02,
        alpha_mass=2.0,
        attempt_mass=100.0,
        tolerance=0.01,
    ),

    # Case 8: Fractional abstention
    RFLTestCase(
        name="fractional",
        description="Fractional abstention values",
        h_t=_make_h_t("fractional")[0],
        r_t=_make_h_t("fractional")[1],
        u_t=_make_h_t("fractional")[2],
        alpha_rate=0.123456,
        alpha_mass=12.3456,
        attempt_mass=100.0,
        tolerance=0.10,
    ),
]


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> RFLConfig:
    """Standard RFL config for testing."""
    return RFLConfig(
        experiment_id="rfl_law_test",
        num_runs=5,
        abstention_tolerance=0.10,
    )


@pytest.fixture
def audit_log() -> RFLAuditLog:
    """Fresh audit log for testing."""
    return RFLAuditLog(seed=42)


def make_attested_context(case: RFLTestCase, config: RFLConfig) -> AttestedRunContext:
    """Create AttestedRunContext from test case."""
    return AttestedRunContext(
        slice_id="test-slice",
        statement_hash=hashlib.sha256(case.name.encode()).hexdigest(),
        proof_status="success" if case.alpha_rate == 0 else "failure",
        block_id=1,
        composite_root=case.h_t,
        reasoning_root=case.r_t,
        ui_root=case.u_t,
        abstention_metrics={
            "rate": case.alpha_rate,
            "mass": case.alpha_mass,
        },
        policy_id="test-policy",
        metadata={
            "attempt_mass": case.attempt_mass,
            "abstention_breakdown": {"test": int(case.alpha_mass)},
        },
    )


# ---------------------------------------------------------------------------
# Core Determinism Tests
# ---------------------------------------------------------------------------

class TestSymbolicDescentGradient:
    """Tests for SymbolicDescentGradient computation."""

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_gradient_computation_deterministic(self, case: RFLTestCase) -> None:
        """
        Verify that gradient computation is deterministic.

        Given: H_t with abstention profile (α_rate, α_mass)
        When: SymbolicDescentGradient.compute() is called
        Then: ∇_sym = -(α_rate - τ) exactly
        """
        gradient = SymbolicDescentGradient.compute(
            abstention_rate=case.alpha_rate,
            abstention_mass=case.alpha_mass,
            tolerance=case.tolerance,
            attempt_mass=case.attempt_mass,
        )

        # Verify symbolic descent per RFL Law
        assert gradient.symbolic_descent == pytest.approx(
            case.expected_symbolic_descent
        ), f"∇_sym should be {case.expected_symbolic_descent}"

        # Verify policy reward per RFL Law
        assert gradient.policy_reward == pytest.approx(
            case.expected_policy_reward
        ), f"reward should be {case.expected_policy_reward}"

        # Verify mass delta per RFL Law
        assert gradient.mass_delta == pytest.approx(
            case.expected_mass_delta
        ), f"Δα should be {case.expected_mass_delta}"

    def test_gradient_triggers_update_correctly(self) -> None:
        """Verify triggers_update() matches expected behavior."""
        # At tolerance → no update
        at_tolerance = SymbolicDescentGradient.compute(
            abstention_rate=0.10,
            abstention_mass=2.0,
            tolerance=0.10,
            attempt_mass=20.0,
        )
        assert not at_tolerance.triggers_update()

        # Above tolerance → update
        above_tolerance = SymbolicDescentGradient.compute(
            abstention_rate=0.15,
            abstention_mass=3.0,
            tolerance=0.10,
            attempt_mass=20.0,
        )
        assert above_tolerance.triggers_update()


class TestStepIdComputation:
    """Tests for step_id determinism."""

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_step_id_deterministic(self, case: RFLTestCase) -> None:
        """
        Verify step_id computation is deterministic.

        Given: experiment_id, slice_name, policy_id, H_t
        When: StepIdComputation.compute() is called
        Then: step_id = SHA256(experiment_id | slice_name | policy_id | H_t)
        """
        comp = StepIdComputation.compute(
            experiment_id="test_exp",
            slice_name="test_slice",
            policy_id="test_policy",
            composite_root=case.h_t,
        )

        # Manually compute expected step_id
        expected = hashlib.sha256(
            f"test_exp|test_slice|test_policy|{case.h_t}".encode()
        ).hexdigest()

        assert comp.step_id == expected
        assert comp.verify()

    def test_step_id_stable_across_invocations(self) -> None:
        """Verify step_id is identical across 100 invocations."""
        h_t = _make_h_t("stability_test")[0]
        step_ids = []

        for _ in range(100):
            comp = StepIdComputation.compute(
                experiment_id="stable_exp",
                slice_name="stable_slice",
                policy_id="stable_policy",
                composite_root=h_t,
            )
            step_ids.append(comp.step_id)

        # All 100 step_ids must be identical
        assert len(set(step_ids)) == 1, "step_id not stable across invocations"

    def test_different_h_t_produces_different_step_id(self) -> None:
        """Verify different H_t produces different step_id."""
        h_t_1 = _make_h_t("test_1")[0]
        h_t_2 = _make_h_t("test_2")[0]

        comp_1 = StepIdComputation.compute(
            experiment_id="exp",
            slice_name="slice",
            policy_id="policy",
            composite_root=h_t_1,
        )
        comp_2 = StepIdComputation.compute(
            experiment_id="exp",
            slice_name="slice",
            policy_id="policy",
            composite_root=h_t_2,
        )

        assert comp_1.step_id != comp_2.step_id


class TestRFLRunnerDeterminism:
    """Tests for RFLRunner.run_with_attestation() determinism."""

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_run_with_attestation_deterministic(
        self, case: RFLTestCase, config: RFLConfig
    ) -> None:
        """
        Verify that run_with_attestation produces deterministic output.

        Given: H_t with specific abstention profile
        When: run_with_attestation() is called
        Then: The exact RFL ledger update is produced
        """
        # Override config tolerance to match test case
        config = RFLConfig(
            experiment_id="rfl_law_test",
            num_runs=5,
            abstention_tolerance=case.tolerance,
        )

        runner = RFLRunner(config)
        ctx = make_attested_context(case, config)

        result = runner.run_with_attestation(ctx)

        # Verify step_id is deterministic
        resolved_slice = runner._resolve_slice(ctx.slice_id)
        expected_step = hashlib.sha256(
            f"{config.experiment_id}|{resolved_slice.name}|test-policy|{case.h_t}".encode()
        ).hexdigest()
        assert result.step_id == expected_step

        # Verify abstention_mass_delta
        assert result.abstention_mass_delta == pytest.approx(
            case.expected_mass_delta
        ), f"Δα should be {case.expected_mass_delta}"

        # Verify policy_update_applied
        assert result.policy_update_applied == case.expected_policy_update

        # Verify ledger entry
        entry = result.ledger_entry
        assert entry is not None
        assert entry.symbolic_descent == pytest.approx(case.expected_symbolic_descent)
        assert entry.policy_reward == pytest.approx(case.expected_policy_reward)
        assert entry.abstention_fraction == pytest.approx(case.alpha_rate)

    def test_multiple_invocations_identical(self, config: RFLConfig) -> None:
        """
        Verify that N invocations with same input produce identical output.

        This is the core determinism guarantee of the RFL Law.
        """
        case = TEST_CASES[0]  # high_abstention case

        results: List[Dict[str, Any]] = []

        for i in range(10):
            runner = RFLRunner(config)
            ctx = make_attested_context(case, config)
            result = runner.run_with_attestation(ctx)

            results.append({
                "step_id": result.step_id,
                "policy_update_applied": result.policy_update_applied,
                "abstention_mass_delta": result.abstention_mass_delta,
                "symbolic_descent": result.ledger_entry.symbolic_descent,
                "policy_reward": result.ledger_entry.policy_reward,
            })

        # All 10 results must be identical
        for i in range(1, 10):
            assert results[i]["step_id"] == results[0]["step_id"]
            assert results[i]["policy_update_applied"] == results[0]["policy_update_applied"]
            assert results[i]["abstention_mass_delta"] == pytest.approx(
                results[0]["abstention_mass_delta"]
            )
            assert results[i]["symbolic_descent"] == pytest.approx(
                results[0]["symbolic_descent"]
            )
            assert results[i]["policy_reward"] == pytest.approx(
                results[0]["policy_reward"]
            )


class TestVerifyRFLTransformation:
    """Tests for the verify_rfl_transformation audit function."""

    @pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
    def test_verify_transformation_passes(self, case: RFLTestCase) -> None:
        """
        Verify that verify_rfl_transformation correctly validates
        a properly computed transformation.
        """
        # Compute expected step_id
        expected_step_id = hashlib.sha256(
            f"test_exp|test_slice|test_policy|{case.h_t}".encode()
        ).hexdigest()

        is_valid, report = verify_rfl_transformation(
            h_t=case.h_t,
            r_t=case.r_t,
            u_t=case.u_t,
            abstention_rate=case.alpha_rate,
            abstention_mass=case.alpha_mass,
            attempt_mass=case.attempt_mass,
            experiment_id="test_exp",
            slice_name="test_slice",
            policy_id="test_policy",
            tolerance=case.tolerance,
            expected_step_id=expected_step_id,
            expected_symbolic_descent=case.expected_symbolic_descent,
            expected_policy_reward=case.expected_policy_reward,
        )

        assert is_valid, f"Verification failed: {report['checks']}"
        assert report["checks"]["step_id_match"]
        assert report["checks"]["symbolic_descent_match"]
        assert report["checks"]["policy_reward_match"]

    def test_verify_transformation_fails_on_wrong_step_id(self) -> None:
        """Verify that wrong step_id is detected."""
        case = TEST_CASES[0]

        is_valid, report = verify_rfl_transformation(
            h_t=case.h_t,
            r_t=case.r_t,
            u_t=case.u_t,
            abstention_rate=case.alpha_rate,
            abstention_mass=case.alpha_mass,
            attempt_mass=case.attempt_mass,
            experiment_id="test_exp",
            slice_name="test_slice",
            policy_id="test_policy",
            tolerance=case.tolerance,
            expected_step_id="wrong_step_id",  # Wrong!
            expected_symbolic_descent=case.expected_symbolic_descent,
            expected_policy_reward=case.expected_policy_reward,
        )

        assert not is_valid
        assert not report["checks"]["step_id_match"]


class TestAuditEntry:
    """Tests for AuditEntry verification."""

    def test_audit_entry_verifies_determinism(
        self, config: RFLConfig, audit_log: RFLAuditLog
    ) -> None:
        """
        Verify that AuditEntry correctly verifies determinism.
        """
        case = TEST_CASES[0]
        runner = RFLRunner(config)
        ctx = make_attested_context(case, config)

        result = runner.run_with_attestation(ctx)
        resolved_slice = runner._resolve_slice(ctx.slice_id)

        entry = audit_log.record_transformation(
            attestation=ctx,
            result=result,
            config=config,
            resolved_slice_name=resolved_slice.name,
        )

        is_valid, errors = entry.verify_determinism()
        assert is_valid, f"Audit entry verification failed: {errors}"

    def test_audit_log_verifies_all_entries(
        self, config: RFLConfig, audit_log: RFLAuditLog
    ) -> None:
        """
        Verify that audit log can verify all recorded entries.
        """
        # Record multiple transformations
        for case in TEST_CASES[:3]:
            runner = RFLRunner(config)
            ctx = make_attested_context(case, config)
            result = runner.run_with_attestation(ctx)
            resolved_slice = runner._resolve_slice(ctx.slice_id)

            audit_log.record_transformation(
                attestation=ctx,
                result=result,
                config=config,
                resolved_slice_name=resolved_slice.name,
            )

        # Verify all entries
        all_valid, invalid = audit_log.verify_all()
        assert all_valid, f"Audit log verification failed: {invalid}"
        assert len(audit_log.entries) == 3


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_attempt_mass(self, config: RFLConfig) -> None:
        """
        Verify behavior when attempt_mass defaults to abstention_mass.
        """
        ctx = AttestedRunContext(
            slice_id="test-slice",
            statement_hash="a" * 64,
            proof_status="failure",
            block_id=1,
            composite_root="b" * 64,
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            abstention_metrics={"rate": 0.5, "mass": 5.0},
            metadata={},  # No attempt_mass provided
        )

        runner = RFLRunner(config)
        result = runner.run_with_attestation(ctx)

        # attempt_mass should default to max(abstention_mass, 1.0) = 5.0
        # expected_mass = tolerance * 5.0 = 0.1 * 5.0 = 0.5
        # mass_delta = 5.0 - 0.5 = 4.5
        assert result.abstention_mass_delta == pytest.approx(4.5)

    def test_negative_abstention_rate_rejected(self, config: RFLConfig) -> None:
        """Verify that negative abstention rate is rejected."""
        ctx = AttestedRunContext(
            slice_id="test-slice",
            statement_hash="a" * 64,
            proof_status="failure",
            block_id=1,
            composite_root="b" * 64,
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            abstention_metrics={"rate": -0.1, "mass": 5.0},  # Negative rate
            metadata={"attempt_mass": 20.0},
        )

        runner = RFLRunner(config)
        with pytest.raises(ValueError, match="abstention_rate must be non-negative"):
            runner.run_with_attestation(ctx)

    def test_invalid_h_t_rejected(self, config: RFLConfig) -> None:
        """Verify that invalid H_t (wrong length) is rejected."""
        ctx = AttestedRunContext(
            slice_id="test-slice",
            statement_hash="a" * 64,
            proof_status="failure",
            block_id=1,
            composite_root="short",  # Invalid: not 64 chars
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            abstention_metrics={"rate": 0.1, "mass": 2.0},
            metadata={"attempt_mass": 20.0},
        )

        runner = RFLRunner(config)
        with pytest.raises(ValueError, match="must be 64 hex chars"):
            runner.run_with_attestation(ctx)


class TestRFLLawDocumentation:
    """
    Tests that verify the examples in docs/RFL_LAW.md are accurate.

    These serve as executable documentation of the RFL Law.
    """

    def test_high_abstention_example(self) -> None:
        """
        Verify Example 5.1 from RFL_LAW.md:

        Input:
            H_t = "6a006e789be39105..."
            α_rate = 0.35
            α_mass = 7.0
            attempt_mass = 20.0
            τ = 0.10

        Expected:
            Δα = 7.0 - (0.10 × 20.0) = 5.0
            ∇_sym = -(0.35 - 0.10) = -0.25
            reward = max(0, 1.0 - 0.35) = 0.65
        """
        gradient = SymbolicDescentGradient.compute(
            abstention_rate=0.35,
            abstention_mass=7.0,
            tolerance=0.10,
            attempt_mass=20.0,
        )

        assert gradient.mass_delta == pytest.approx(5.0)
        assert gradient.symbolic_descent == pytest.approx(-0.25)
        assert gradient.policy_reward == pytest.approx(0.65)
        assert gradient.triggers_update()

    def test_zero_abstention_example(self) -> None:
        """
        Verify Example 5.2 from RFL_LAW.md:

        Input:
            α_rate = 0.0
            α_mass = 0.0
            attempt_mass = 20.0
            τ = 0.10

        Expected:
            Δα = 0.0 - (0.10 × 20.0) = -2.0
            ∇_sym = -(0.0 - 0.10) = 0.10
            reward = max(0, 1.0 - 0.0) = 1.0
        """
        gradient = SymbolicDescentGradient.compute(
            abstention_rate=0.0,
            abstention_mass=0.0,
            tolerance=0.10,
            attempt_mass=20.0,
        )

        assert gradient.mass_delta == pytest.approx(-2.0)
        assert gradient.symbolic_descent == pytest.approx(0.10)
        assert gradient.policy_reward == pytest.approx(1.0)
        assert gradient.triggers_update()  # mass_delta != 0

    def test_at_tolerance_boundary_example(self) -> None:
        """
        Verify Example 5.3 from RFL_LAW.md:

        Input:
            α_rate = 0.10
            α_mass = 2.0
            attempt_mass = 20.0
            τ = 0.10

        Expected:
            Δα = 2.0 - (0.10 × 20.0) = 0.0
            ∇_sym = -(0.10 - 0.10) = 0.0
            reward = max(0, 1.0 - 0.10) = 0.90
            policy_update_applied = False
        """
        gradient = SymbolicDescentGradient.compute(
            abstention_rate=0.10,
            abstention_mass=2.0,
            tolerance=0.10,
            attempt_mass=20.0,
        )

        assert gradient.mass_delta == pytest.approx(0.0)
        assert gradient.symbolic_descent == pytest.approx(0.0)
        assert gradient.policy_reward == pytest.approx(0.90)
        assert not gradient.triggers_update()  # Both deltas are 0
