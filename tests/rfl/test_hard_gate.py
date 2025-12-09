"""
Tests for TDA Hard Gate Decision Evaluation
"""

import pytest

from rfl.hard_gate import (
    HardGateMode,
    HSSTrace,
    PolicyDelta,
    HardGateDecision,
    evaluate_hard_gate_decision,
    aggregate_hss_traces,
)
from rfl.evidence_fusion import TDAOutcome


class TestHardGateDecision:
    """Tests for evaluate_hard_gate_decision function."""
    
    def test_basic_evaluation(self):
        """Test basic hard gate evaluation."""
        policy_state = {
            "weights": {"len": 0.5, "depth": 0.3, "success": 0.2},
            "version": "v1",
        }
        
        event_stats = {
            "total_events": 100,
            "blocked": 10,
            "passed": 90,
        }
        
        decision = evaluate_hard_gate_decision(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
            mode=HardGateMode.SHADOW,
        )
        
        assert decision.outcome in [
            TDAOutcome.PASS,
            TDAOutcome.WARN,
            TDAOutcome.BLOCK,
            TDAOutcome.SHADOW,
        ]
        assert decision.tda_fields.block_rate == pytest.approx(0.1)
        assert len(decision.hss_traces) == 1
    
    def test_disabled_mode(self):
        """Test hard gate in DISABLED mode."""
        decision = evaluate_hard_gate_decision(
            cycle=0,
            policy_state={},
            event_stats={},
            mode=HardGateMode.DISABLED,
        )
        
        assert decision.outcome == TDAOutcome.UNKNOWN
        assert "disabled" in decision.decision_reason.lower()
    
    def test_high_block_rate_detection(self):
        """Test detection of high block rate."""
        policy_state = {"weights": {}}
        event_stats = {
            "total_events": 100,
            "blocked": 96,  # 96% blocking
            "passed": 4,
        }
        
        decision = evaluate_hard_gate_decision(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
            mode=HardGateMode.ENFORCE,
        )
        
        # Should detect critical blocking
        assert decision.outcome == TDAOutcome.BLOCK
        assert "blocking rate" in decision.decision_reason.lower()
    
    def test_shadow_mode_vs_enforce_mode(self):
        """Test difference between SHADOW and ENFORCE modes."""
        policy_state = {"weights": {}}
        event_stats = {
            "total_events": 100,
            "blocked": 96,
            "passed": 4,
        }
        
        # SHADOW mode
        decision_shadow = evaluate_hard_gate_decision(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
            mode=HardGateMode.SHADOW,
        )
        
        # ENFORCE mode
        decision_enforce = evaluate_hard_gate_decision(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
            mode=HardGateMode.ENFORCE,
        )
        
        # SHADOW should return SHADOW outcome
        assert decision_shadow.outcome == TDAOutcome.SHADOW
        
        # ENFORCE should return BLOCK outcome
        assert decision_enforce.outcome == TDAOutcome.BLOCK
    
    def test_hss_computation(self):
        """Test HSS computation across cycles."""
        policy_state_1 = {"weights": {"a": 0.5}}
        policy_state_2 = {"weights": {"a": 0.5}}  # Same policy
        policy_state_3 = {"weights": {"a": 0.7}}  # Different policy
        
        event_stats = {"total_events": 100, "blocked": 5, "passed": 95}
        
        # Cycle 0 - first cycle (HSS should be neutral)
        decision1 = evaluate_hard_gate_decision(
            cycle=0,
            policy_state=policy_state_1,
            event_stats=event_stats,
            mode=HardGateMode.SHADOW,
        )
        
        assert decision1.hss_traces[0].stability_score == pytest.approx(0.5)
        
        # Cycle 1 - same policy (HSS should be 1.0)
        decision2 = evaluate_hard_gate_decision(
            cycle=1,
            policy_state=policy_state_2,
            event_stats=event_stats,
            previous_policy_hash=decision1.hss_traces[0].policy_hash,
            mode=HardGateMode.SHADOW,
        )
        
        assert decision2.hss_traces[0].stability_score == pytest.approx(1.0)
        
        # Cycle 2 - different policy (HSS should be < 1.0)
        decision3 = evaluate_hard_gate_decision(
            cycle=2,
            policy_state=policy_state_3,
            event_stats=event_stats,
            previous_policy_hash=decision2.hss_traces[0].policy_hash,
            mode=HardGateMode.SHADOW,
        )
        
        assert decision3.hss_traces[0].stability_score < 1.0


class TestAggregateHSSTraces:
    """Tests for aggregate_hss_traces function."""
    
    def test_empty_traces(self):
        """Test aggregation with empty traces."""
        result = aggregate_hss_traces([])
        
        assert result["mean_stability"] == 0.0
        assert result["trend"] == "unknown"
    
    def test_single_trace(self):
        """Test aggregation with single trace."""
        traces = [
            HSSTrace(cycle=0, policy_hash="abc123", stability_score=0.8),
        ]
        
        result = aggregate_hss_traces(traces)
        
        assert result["mean_stability"] == pytest.approx(0.8)
        assert result["min_stability"] == pytest.approx(0.8)
        assert result["max_stability"] == pytest.approx(0.8)
        assert result["trend"] == "insufficient_data"
    
    def test_improving_trend(self):
        """Test detection of improving stability trend."""
        traces = [
            HSSTrace(cycle=i, policy_hash=f"hash_{i}", stability_score=0.5 + i * 0.1)
            for i in range(10)
        ]
        
        result = aggregate_hss_traces(traces)
        
        assert result["trend"] == "improving"
        assert result["mean_stability"] > 0.5
    
    def test_degrading_trend(self):
        """Test detection of degrading stability trend."""
        traces = [
            HSSTrace(cycle=i, policy_hash=f"hash_{i}", stability_score=0.9 - i * 0.1)
            for i in range(10)
        ]
        
        result = aggregate_hss_traces(traces)
        
        assert result["trend"] == "degrading"
    
    def test_stable_trend(self):
        """Test detection of stable trend."""
        traces = [
            HSSTrace(cycle=i, policy_hash=f"hash_{i}", stability_score=0.8)
            for i in range(10)
        ]
        
        result = aggregate_hss_traces(traces)
        
        assert result["trend"] == "stable"
        assert result["mean_stability"] == pytest.approx(0.8)
