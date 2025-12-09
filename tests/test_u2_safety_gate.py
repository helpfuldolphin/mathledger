"""
Tests for U2 Safety Gate (Neural Link)

Verifies:
- evaluate_hard_gate_decision() blocks candidates
- Deterministic gate decisions
- Safety context tracking
- Snapshot/restore of safety state
"""

import pytest
from typing import Any, Tuple

from rfl.prng import DeterministicPRNG
from experiments.u2 import (
    U2Runner,
    U2Config,
    U2SafetyContext,
    SafetyEnvelope,
    GateDecision,
    evaluate_hard_gate_decision,
    validate_safety_envelope,
)


class TestSafetyGateBlocking:
    """Test that safety gate blocks execution as expected."""
    
    def test_gate_approves_simple_candidate(self):
        """Safety gate approves simple candidates."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        candidate = {"item": "simple", "depth": 2}
        envelope = evaluate_hard_gate_decision(
            candidate=candidate,
            cycle=0,
            safety_context=context,
            prng=prng,
            max_depth=10,
        )
        
        assert envelope.decision == GateDecision.APPROVED
        assert context.total_candidates_evaluated == 1
        assert context.total_approvals == 1
        assert context.approval_rate == 1.0
    
    def test_gate_rejects_deep_candidate(self):
        """Safety gate rejects candidates exceeding depth limit."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        candidate = {"item": "deep", "depth": 15}
        envelope = evaluate_hard_gate_decision(
            candidate=candidate,
            cycle=0,
            safety_context=context,
            prng=prng,
            max_depth=10,
        )
        
        assert envelope.decision == GateDecision.REJECTED
        assert "depth_exceeded" in envelope.reason
        assert not envelope.slo_compliant
        assert context.total_rejections == 1
        assert context.safety_violations == 1
    
    def test_gate_rejects_complex_candidate(self):
        """Safety gate rejects candidates exceeding complexity limit."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        # Create a very complex candidate
        complex_candidate = {"item": "x" * 2000, "depth": 2}
        envelope = evaluate_hard_gate_decision(
            candidate=complex_candidate,
            cycle=0,
            safety_context=context,
            prng=prng,
            max_depth=10,
            max_complexity=1000.0,
        )
        
        assert envelope.decision == GateDecision.REJECTED
        assert "complexity_exceeded" in envelope.reason
        assert not envelope.slo_compliant
        assert context.total_rejections == 1
    
    def test_gate_slo_protection(self):
        """Safety gate uses abstention for SLO protection."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        # Build up high rejection rate
        for i in range(15):
            deep_candidate = {"item": f"deep_{i}", "depth": 20}
            evaluate_hard_gate_decision(
                candidate=deep_candidate,
                cycle=i,
                safety_context=context,
                prng=prng.for_path(str(i)),
                max_depth=10,
            )
        
        # Now rejection rate should be high
        assert context.rejection_rate > 0.5
        
        # Try a valid candidate - might get abstention for SLO protection
        valid_candidate = {"item": "valid", "depth": 2}
        envelope = evaluate_hard_gate_decision(
            candidate=valid_candidate,
            cycle=100,
            safety_context=context,
            prng=prng.for_path("100"),
            max_depth=10,
        )
        
        # Could be approved or abstained depending on PRNG
        assert envelope.decision in [GateDecision.APPROVED, GateDecision.ABSTAINED]


class TestSafetyGateDeterminism:
    """Test that safety gate decisions are deterministic."""
    
    def test_same_seed_same_decision(self):
        """Same PRNG seed produces same gate decision."""
        candidate = {"item": "test", "depth": 2}
        
        prng1 = DeterministicPRNG("0xsame")
        context1 = U2SafetyContext()
        envelope1 = evaluate_hard_gate_decision(
            candidate=candidate,
            cycle=0,
            safety_context=context1,
            prng=prng1,
        )
        
        prng2 = DeterministicPRNG("0xsame")
        context2 = U2SafetyContext()
        envelope2 = evaluate_hard_gate_decision(
            candidate=candidate,
            cycle=0,
            safety_context=context2,
            prng=prng2,
        )
        
        assert envelope1.decision == envelope2.decision
        assert envelope1.reason == envelope2.reason
        assert envelope1.confidence == envelope2.confidence
    
    def test_different_seed_consistent_rejection(self):
        """Different seeds still reject bad candidates consistently."""
        deep_candidate = {"item": "deep", "depth": 20}
        
        for seed in ["0xa", "0xb", "0xc", "0xd"]:
            prng = DeterministicPRNG(seed)
            context = U2SafetyContext()
            envelope = evaluate_hard_gate_decision(
                candidate=deep_candidate,
                cycle=0,
                safety_context=context,
                prng=prng,
                max_depth=10,
            )
            
            # All should reject regardless of PRNG seed
            assert envelope.decision == GateDecision.REJECTED


class TestSafetyContextTracking:
    """Test safety context state tracking."""
    
    def test_context_records_decisions(self):
        """Safety context correctly tracks decision counts."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        # Approve 5
        for i in range(5):
            candidate = {"item": f"good_{i}", "depth": 2}
            evaluate_hard_gate_decision(
                candidate=candidate,
                cycle=i,
                safety_context=context,
                prng=prng.for_path(str(i)),
            )
        
        # Reject 3
        for i in range(3):
            candidate = {"item": f"bad_{i}", "depth": 20}
            evaluate_hard_gate_decision(
                candidate=candidate,
                cycle=i + 5,
                safety_context=context,
                prng=prng.for_path(str(i + 5)),
                max_depth=10,
            )
        
        assert context.total_candidates_evaluated == 8
        assert context.total_approvals == 5
        assert context.total_rejections == 3
        assert abs(context.approval_rate - 5/8) < 0.001
        assert abs(context.rejection_rate - 3/8) < 0.001
    
    def test_context_serialization(self):
        """Safety context can be serialized and restored."""
        prng = DeterministicPRNG("0xtest")
        context = U2SafetyContext()
        
        # Build up some state
        for i in range(10):
            candidate = {"item": f"item_{i}", "depth": i % 5}
            evaluate_hard_gate_decision(
                candidate=candidate,
                cycle=i,
                safety_context=context,
                prng=prng.for_path(str(i)),
                max_depth=10,
            )
        
        # Serialize
        data = context.to_dict()
        
        # Restore
        restored = U2SafetyContext.from_dict(data)
        
        assert restored.total_candidates_evaluated == context.total_candidates_evaluated
        assert restored.total_approvals == context.total_approvals
        assert restored.total_rejections == context.total_rejections
        assert restored.approval_rate == context.approval_rate


class TestRunnerIntegration:
    """Test safety gate integration with U2Runner."""
    
    def create_mock_execute_fn(self, prng: DeterministicPRNG):
        """Create mock execution function."""
        def execute(item: Any, seed: int) -> Tuple[bool, Any]:
            item_prng = prng.for_path("execute", str(item), str(seed))
            success = item_prng.random() > 0.3
            result = {"outcome": "success" if success else "failure"}
            return success, result
        return execute
    
    def test_runner_blocks_deep_candidates(self):
        """Runner blocks candidates rejected by safety gate."""
        config = U2Config(
            experiment_id="test_safety",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
            max_beam_width=10,
            max_depth=5,  # Low depth limit
        )
        
        runner = U2Runner(config)
        
        # Push candidates with varying depth
        for i in range(10):
            depth = i % 8  # Some will exceed max_depth=5
            runner.frontier.push(
                item={"item": f"candidate_{i}", "depth": depth},
                priority=float(i),
                depth=depth,
            )
        
        prng = DeterministicPRNG(42)
        execute_fn = self.create_mock_execute_fn(prng)
        
        # Run a cycle
        result = runner.run_cycle(0, execute_fn)
        
        # Safety gate should have evaluated candidates
        assert runner.safety_context.total_candidates_evaluated > 0
        
        # Some should have been rejected due to depth
        assert runner.safety_context.total_rejections > 0
    
    def test_runner_state_includes_safety(self):
        """Runner state export includes safety context."""
        config = U2Config(
            experiment_id="test_safety",
            slice_name="test_slice",
            mode="baseline",
            total_cycles=1,
            master_seed=42,
        )
        
        runner = U2Runner(config)
        runner.frontier.push("test_item", priority=1.0, depth=0)
        
        prng = DeterministicPRNG(42)
        runner.run_cycle(0, self.create_mock_execute_fn(prng))
        
        state = runner.get_state()
        
        assert "safety_context" in state
        assert "total_candidates_evaluated" in state["safety_context"]
        assert state["safety_context"]["total_candidates_evaluated"] > 0


class TestEnvelopeValidation:
    """Test safety envelope validation."""
    
    def test_valid_envelope(self):
        """Valid envelope passes validation."""
        envelope = SafetyEnvelope(
            decision=GateDecision.APPROVED,
            candidate_id="test_candidate",
            cycle=0,
            reason="passed_all_checks",
            confidence=1.0,
            slo_compliant=True,
        )
        
        assert validate_safety_envelope(envelope)
    
    def test_invalid_confidence(self):
        """Invalid confidence fails validation."""
        envelope = SafetyEnvelope(
            decision=GateDecision.APPROVED,
            candidate_id="test_candidate",
            cycle=0,
            reason="test",
            confidence=1.5,  # Invalid: > 1.0
            slo_compliant=True,
        )
        
        assert not validate_safety_envelope(envelope)
    
    def test_invalid_cycle(self):
        """Negative cycle fails validation."""
        envelope = SafetyEnvelope(
            decision=GateDecision.APPROVED,
            candidate_id="test_candidate",
            cycle=-1,  # Invalid
            reason="test",
            confidence=1.0,
            slo_compliant=True,
        )
        
        assert not validate_safety_envelope(envelope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
