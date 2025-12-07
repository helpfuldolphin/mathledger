"""
RFL Engine Tests
================

Comprehensive tests for RFL engine components connected to epistemic risk functional J(π).

Tests cover:
1. Update algebra ⊕ operator
2. Symbolic delta tracking and policy serialization
3. Deterministic policy evolution
4. Step-size schedules
5. Dual-attested event verification
6. Integration with epistemic risk functional

The epistemic risk functional J(π) measures the expected abstention rate:
    J(π) = E[α(π)]
where α(π) is the abstention rate under policy π.

RFL aims to minimize J(π) through gradient descent:
    π_{t+1} = π_t ⊕ η_t Φ(V(e_t), π_t)
where Φ is the gradient of J with respect to π.
"""

import pytest
import hashlib
import json
from datetime import datetime
from typing import Dict, Any

from rfl.update_algebra import (
    PolicyState,
    PolicyUpdate,
    apply_update,
    compute_gradient_norm,
    zero_update,
    is_zero_update,
    PolicyEvolutionChain,
)
from rfl.policy_serialization import (
    PolicyCheckpoint,
    DeltaLog,
    DeltaLogEntry,
    save_checkpoint,
    load_checkpoint,
    replay_from_deltas,
    compute_config_hash,
)
from rfl.step_size_schedules import (
    ConstantSchedule,
    ExponentialDecaySchedule,
    AdaptiveSchedule,
    create_schedule,
    load_schedule,
)
from rfl.event_verification import (
    AttestedEvent,
    EventVerifier,
    verify_dual_attestation,
    filter_attested_events,
    compute_composite_root,
    RFLEventGate,
    VerificationStatus,
)


# -----------------------------------------------------------------------------
# Test Update Algebra ⊕
# -----------------------------------------------------------------------------

class TestUpdateAlgebra:
    """Tests for update algebra operator."""
    
    def test_policy_state_creation(self):
        """Test PolicyState creation and validation."""
        policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0, "success": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        assert policy.epoch == 0
        assert policy.weights["len"] == 0.0
        assert policy.parent_hash is None
        assert len(policy.hash()) == 64  # SHA256 hex digest
    
    def test_policy_state_hash_determinism(self):
        """Test that policy hash is deterministic."""
        policy1 = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        policy2 = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        assert policy1.hash() == policy2.hash()
    
    def test_policy_update_creation(self):
        """Test PolicyUpdate creation and validation."""
        update = PolicyUpdate(
            deltas={"len": -0.1, "depth": 0.05},
            step_size=0.1,
            gradient_norm=0.15,
        )
        
        assert update.step_size == 0.1
        assert update.deltas["len"] == -0.1
        
        scaled = update.scaled_deltas()
        assert scaled["len"] == pytest.approx(-0.01)
        assert scaled["depth"] == pytest.approx(0.005)
    
    def test_apply_update_basic(self):
        """Test basic update application: π_{t+1} = π_t ⊕ Δπ."""
        policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        update = PolicyUpdate(
            deltas={"len": -0.1, "depth": 0.05},
            step_size=0.1,
        )
        
        new_policy = apply_update(policy, update, "2025-01-01T01:00:00Z")
        
        assert new_policy.epoch == 1
        assert new_policy.weights["len"] == pytest.approx(-0.01)
        assert new_policy.weights["depth"] == pytest.approx(0.005)
        assert new_policy.parent_hash == policy.hash()
    
    def test_apply_update_with_constraints(self):
        """Test update application with weight constraints."""
        policy = PolicyState(
            weights={"success": 5.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Large negative update that would violate constraint
        update = PolicyUpdate(
            deltas={"success": -100.0},
            step_size=1.0,
        )
        
        # Apply with constraint: success >= 0
        new_policy = apply_update(
            policy,
            update,
            "2025-01-01T01:00:00Z",
            constraints={"success": (0.0, 10.0)},
        )
        
        # Should be clamped to lower bound
        assert new_policy.weights["success"] == 0.0
    
    def test_zero_update_identity(self):
        """Test that zero update is identity: π ⊕ 0 = π."""
        policy = PolicyState(
            weights={"len": 0.5, "depth": 0.3},
            epoch=5,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Zero update with matching features
        update = PolicyUpdate(deltas={"len": 0.0, "depth": 0.0}, step_size=0.0)
        new_policy = apply_update(policy, update, "2025-01-01T01:00:00Z")
        
        # Weights unchanged (except epoch increments)
        assert new_policy.weights == policy.weights
        assert new_policy.epoch == policy.epoch + 1
    
    def test_gradient_norm_computation(self):
        """Test L2 norm computation: ||Φ|| = sqrt(Σ Δ_i^2)."""
        deltas = {"len": 0.3, "depth": 0.4}
        norm = compute_gradient_norm(deltas)
        
        # ||[0.3, 0.4]|| = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5
        assert norm == pytest.approx(0.5)


class TestPolicyEvolutionChain:
    """Tests for policy evolution chain."""
    
    def test_chain_creation(self):
        """Test PolicyEvolutionChain creation."""
        initial_policy = PolicyState(
            weights={"len": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        chain = PolicyEvolutionChain(
            states=[initial_policy],
            updates=[],
            metadata={"experiment_id": "test_001"},
        )
        
        assert len(chain.states) == 1
        assert len(chain.updates) == 0
    
    def test_chain_append(self):
        """Test appending updates to chain."""
        initial_policy = PolicyState(
            weights={"len": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        chain = PolicyEvolutionChain(states=[initial_policy])
        
        update = PolicyUpdate(deltas={"len": -0.1}, step_size=0.1)
        new_policy = chain.append(update, "2025-01-01T01:00:00Z")
        
        assert len(chain.states) == 2
        assert len(chain.updates) == 1
        assert new_policy.epoch == 1
    
    def test_chain_verification(self):
        """Test chain integrity verification."""
        initial_policy = PolicyState(
            weights={"len": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        chain = PolicyEvolutionChain(states=[initial_policy])
        
        # Add several updates
        for i in range(5):
            update = PolicyUpdate(deltas={"len": -0.01}, step_size=0.1)
            chain.append(update, f"2025-01-01T0{i+1}:00:00Z")
        
        # Verify chain
        is_valid, errors = chain.verify_chain()
        assert is_valid
        assert len(errors) == 0


# -----------------------------------------------------------------------------
# Test Policy Serialization
# -----------------------------------------------------------------------------

class TestPolicySerialization:
    """Tests for policy serialization and checkpointing."""
    
    def test_checkpoint_creation(self):
        """Test PolicyCheckpoint creation."""
        policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=10,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        checkpoint = PolicyCheckpoint.from_policy_state(
            policy,
            experiment_id="test_001",
            config_hash="abc123",
        )
        
        assert checkpoint.policy_state.epoch == 10
        assert checkpoint.experiment_id == "test_001"
        assert checkpoint.verify_integrity()
    
    def test_checkpoint_serialization(self):
        """Test checkpoint JSON serialization."""
        policy = PolicyState(
            weights={"len": 0.0},
            epoch=5,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        checkpoint = PolicyCheckpoint.from_policy_state(
            policy,
            experiment_id="test_001",
            config_hash="abc123",
        )
        
        # Serialize and deserialize
        json_str = checkpoint.to_json()
        data = json.loads(json_str)
        loaded = PolicyCheckpoint.from_dict(data)
        
        assert loaded.policy_state.epoch == checkpoint.policy_state.epoch
        assert loaded.policy_state.hash() == checkpoint.policy_state.hash()
        assert loaded.verify_integrity()
    
    def test_delta_log_creation(self):
        """Test DeltaLog creation and append."""
        log = DeltaLog(
            initial_policy_hash="abc123",
            experiment_id="test_001",
        )
        
        update = PolicyUpdate(deltas={"len": -0.1}, step_size=0.1)
        log.append(
            epoch=1,
            update=update,
            source_event_hash="def456",
            timestamp="2025-01-01T01:00:00Z",
        )
        
        assert len(log.entries) == 1
        assert log.entries[0].epoch == 1
    
    def test_replay_from_deltas(self):
        """Test deterministic replay from delta log."""
        initial_policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Create delta log
        log = DeltaLog(initial_policy_hash=initial_policy.hash())
        
        # Add updates
        for i in range(5):
            update = PolicyUpdate(deltas={"len": -0.01, "depth": 0.005}, step_size=0.1)
            log.append(
                epoch=i + 1,
                update=update,
                source_event_hash=f"event_{i}",
                timestamp=f"2025-01-01T0{i+1}:00:00Z",
            )
        
        # Replay
        final_policy, warnings = replay_from_deltas(initial_policy, log)
        
        assert final_policy.epoch == 5
        assert final_policy.weights["len"] == pytest.approx(-0.005)
        assert final_policy.weights["depth"] == pytest.approx(0.0025)
        assert len(warnings) == 0


# -----------------------------------------------------------------------------
# Test Step-Size Schedules
# -----------------------------------------------------------------------------

class TestStepSizeSchedules:
    """Tests for step-size schedules η_t."""
    
    def test_constant_schedule(self):
        """Test constant schedule: η_t = η_0."""
        schedule = ConstantSchedule(learning_rate=0.1)
        
        for epoch in range(10):
            eta_t = schedule.get_step_size(epoch)
            assert eta_t == 0.1
    
    def test_exponential_decay_schedule(self):
        """Test exponential decay: η_t = η_0 * exp(-λt)."""
        schedule = ExponentialDecaySchedule(
            initial_rate=0.1,
            decay_rate=0.1,
            min_rate=0.01,
        )
        
        eta_0 = schedule.get_step_size(0)
        eta_5 = schedule.get_step_size(5)
        eta_10 = schedule.get_step_size(10)
        
        assert eta_0 == pytest.approx(0.1)
        assert eta_5 < eta_0  # Decayed
        assert eta_10 < eta_5  # Further decayed
        assert eta_10 >= 0.01  # Respects min_rate
    
    def test_adaptive_schedule(self):
        """Test adaptive schedule with metric feedback."""
        schedule = AdaptiveSchedule(
            initial_rate=0.1,
            min_rate=0.01,
            max_rate=0.5,
            patience=2,
        )
        
        # Epoch 0: initial rate (no gradient norm)
        eta_0 = schedule.get_step_size(0, gradient_norm=0.0, context={"abstention_rate": 0.5})
        assert eta_0 >= 0.01  # At least min_rate
        
        # Epoch 1: improvement → maintain/increase rate
        eta_1 = schedule.get_step_size(1, gradient_norm=0.0, context={"abstention_rate": 0.4})
        assert eta_1 >= 0.01  # At least min_rate
        
        # Epochs 2-3: no improvement → reduce rate after patience
        eta_2 = schedule.get_step_size(2, gradient_norm=0.0, context={"abstention_rate": 0.4})
        eta_3 = schedule.get_step_size(3, gradient_norm=0.0, context={"abstention_rate": 0.4})
        
        # Rate should be within bounds
        assert 0.01 <= eta_3 <= 0.5
    
    def test_schedule_serialization(self):
        """Test schedule serialization and deserialization."""
        schedule = ExponentialDecaySchedule(
            initial_rate=0.1,
            decay_rate=0.05,
        )
        
        # Serialize
        data = schedule.to_dict()
        
        # Deserialize
        loaded = load_schedule(data)
        
        assert isinstance(loaded, ExponentialDecaySchedule)
        assert loaded.initial_rate == schedule.initial_rate
        assert loaded.decay_rate == schedule.decay_rate


# -----------------------------------------------------------------------------
# Test Event Verification
# -----------------------------------------------------------------------------

class TestEventVerification:
    """Tests for dual-attested event verification."""
    
    def test_composite_root_computation(self):
        """Test composite root: H_t = SHA256(R_t || U_t)."""
        reasoning_root = "a" * 64
        ui_root = "b" * 64
        
        composite = compute_composite_root(reasoning_root, ui_root)
        
        assert len(composite) == 64
        
        # Verify determinism
        composite2 = compute_composite_root(reasoning_root, ui_root)
        assert composite == composite2
    
    def test_attested_event_creation(self):
        """Test AttestedEvent creation and validation."""
        reasoning_root = "a" * 64
        ui_root = "b" * 64
        composite_root = compute_composite_root(reasoning_root, ui_root)
        
        event = AttestedEvent(
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=composite_root,
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        assert event.verify_composite_root()
    
    def test_attested_event_invalid_composite(self):
        """Test that invalid composite root is detected."""
        reasoning_root = "a" * 64
        ui_root = "b" * 64
        invalid_composite = "c" * 64  # Wrong!
        
        event = AttestedEvent(
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=invalid_composite,
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        assert not event.verify_composite_root()
    
    def test_event_verifier_valid_event(self):
        """Test EventVerifier with valid event."""
        verifier = EventVerifier(strict_mode=True)
        
        reasoning_root = "a" * 64
        ui_root = "b" * 64
        composite_root = compute_composite_root(reasoning_root, ui_root)
        
        event = AttestedEvent(
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=composite_root,
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        result = verifier.verify_event(event)
        
        assert result.is_valid
        assert result.status == VerificationStatus.VALID
    
    def test_event_verifier_invalid_event(self):
        """Test EventVerifier with invalid event."""
        verifier = EventVerifier(strict_mode=True)
        
        reasoning_root = "a" * 64
        ui_root = "b" * 64
        invalid_composite = "c" * 64
        
        event = AttestedEvent(
            reasoning_root=reasoning_root,
            ui_root=ui_root,
            composite_root=invalid_composite,
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        result = verifier.verify_event(event)
        
        assert not result.is_valid
        assert result.status == VerificationStatus.INVALID_COMPOSITE_ROOT
    
    def test_filter_attested_events(self):
        """Test filtering event stream."""
        verifier = EventVerifier(strict_mode=True)
        
        # Create mix of valid and invalid events
        valid_event = AttestedEvent(
            reasoning_root="a" * 64,
            ui_root="b" * 64,
            composite_root=compute_composite_root("a" * 64, "b" * 64),
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        invalid_event = AttestedEvent(
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            composite_root="e" * 64,  # Wrong!
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        events = [valid_event, invalid_event, valid_event]
        
        valid, rejected = filter_attested_events(events, verifier)
        
        assert len(valid) == 2
        assert len(rejected) == 1
    
    def test_rfl_event_gate(self):
        """Test RFL event gate (fail-closed)."""
        gate = RFLEventGate(fail_closed=True)
        
        # Valid event
        valid_event = AttestedEvent(
            reasoning_root="a" * 64,
            ui_root="b" * 64,
            composite_root=compute_composite_root("a" * 64, "b" * 64),
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        admitted, reason = gate.admit_event(valid_event)
        assert admitted
        
        # Invalid event
        invalid_event = AttestedEvent(
            reasoning_root="c" * 64,
            ui_root="d" * 64,
            composite_root="e" * 64,
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        admitted, reason = gate.admit_event(invalid_event)
        assert not admitted
        assert "fail-closed" in reason.lower()


# -----------------------------------------------------------------------------
# Integration Tests: RFL Engine + Epistemic Risk Functional J(π)
# -----------------------------------------------------------------------------

class TestRFLEngineIntegration:
    """Integration tests connecting RFL engine to epistemic risk functional J(π)."""
    
    def test_policy_update_reduces_epistemic_risk(self):
        """
        Test that policy updates aim to reduce epistemic risk J(π).
        
        J(π) = E[α(π)] where α is abstention rate.
        RFL should update policy to minimize J.
        """
        # Initial policy
        policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0, "success": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Simulate high abstention rate (high J)
        abstention_rate = 0.5
        verified_count = 3
        target_verified = 7
        
        # Compute gradient (reward signal)
        reward = verified_count - target_verified  # -4 (negative)
        
        # Gradient should push policy to reduce abstention
        # Negative reward → try different strategy
        eta = 0.1
        deltas = {
            "len": eta * (+0.1) * abs(reward),  # Try longer formulas
            "depth": eta * (-0.05) * abs(reward),  # Try different depth
            "success": eta * 0.1 * reward,  # Small penalty
        }
        
        update = PolicyUpdate(
            deltas=deltas,
            step_size=eta,
            gradient_norm=compute_gradient_norm(deltas),
        )
        
        new_policy = apply_update(policy, update, "2025-01-01T01:00:00Z")
        
        # Policy should have changed
        assert new_policy.weights["len"] != policy.weights["len"]
        
        # Verify update was applied correctly (scaled by step_size)
        expected_len = 0.0 + (eta * (+0.1) * abs(reward)) * eta
        assert new_policy.weights["len"] == pytest.approx(expected_len, rel=0.01)
    
    def test_deterministic_policy_evolution(self):
        """
        Test that policy evolution is deterministic: same inputs → same outputs.
        
        This is critical for RFL Law compliance: π_{t+1} = π_t ⊕ η_t Φ(V(e_t), π_t).
        """
        # Initial policy
        policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Same update applied twice
        update = PolicyUpdate(
            deltas={"len": -0.1, "depth": 0.05},
            step_size=0.1,
        )
        
        new_policy_1 = apply_update(policy, update, "2025-01-01T01:00:00Z")
        new_policy_2 = apply_update(policy, update, "2025-01-01T01:00:00Z")
        
        # Results must be identical (determinism)
        assert new_policy_1.hash() == new_policy_2.hash()
        assert new_policy_1.weights == new_policy_2.weights
    
    def test_rfl_consumes_only_dual_attested_events(self):
        """
        Test that RFL only consumes dual-attested events.
        
        Invariant: RFL must never read unverifiable or unattested events.
        """
        gate = RFLEventGate(fail_closed=True)
        
        # Unattested event (invalid composite root - must be 64 chars)
        unattested = AttestedEvent(
            reasoning_root="a" * 64,
            ui_root="b" * 64,
            composite_root="c" * 64,  # Wrong composite (not SHA256 of a||b)
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        admitted, reason = gate.admit_event(unattested)
        
        # Must be rejected
        assert not admitted
        
        # Valid attested event
        attested = AttestedEvent(
            reasoning_root="a" * 64,
            ui_root="b" * 64,
            composite_root=compute_composite_root("a" * 64, "b" * 64),
            reasoning_event_count=10,
            ui_event_count=5,
        )
        
        admitted, reason = gate.admit_event(attested)
        
        # Must be admitted
        assert admitted
    
    def test_policy_replayable_from_logs(self):
        """
        Test that policies are replayable from logs.
        
        Invariant: Policies must be replayable from logs for auditability.
        """
        # Initial policy
        initial_policy = PolicyState(
            weights={"len": 0.0, "depth": 0.0, "success": 0.0},
            epoch=0,
            timestamp="2025-01-01T00:00:00Z",
        )
        
        # Create evolution chain
        chain = PolicyEvolutionChain(states=[initial_policy])
        
        # Apply 10 updates
        for i in range(10):
            update = PolicyUpdate(
                deltas={"len": -0.01, "depth": 0.005, "success": 0.02},
                step_size=0.1,
            )
            chain.append(update, f"2025-01-01T{i+1:02d}:00:00Z")
        
        # Verify chain
        is_valid, errors = chain.verify_chain()
        assert is_valid
        
        # Replay from delta log
        delta_log = DeltaLog(
            initial_policy_hash=initial_policy.hash(),
            entries=[
                DeltaLogEntry(
                    epoch=i + 1,
                    update=chain.updates[i],
                    source_event_hash=f"event_{i}",
                    timestamp=f"2025-01-01T{i+1:02d}:00:00Z",
                )
                for i in range(10)
            ],
        )
        
        final_policy, warnings = replay_from_deltas(initial_policy, delta_log)
        
        # Final policy should match chain
        assert final_policy.hash() == chain.states[-1].hash()
        assert len(warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
