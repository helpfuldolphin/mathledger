"""
Tests for RFL Runner Integration with TDA Hard Gate
"""

import pytest

from rfl.runner_integration import (
    HardGateIntegration,
    create_mock_event_stats,
    create_policy_state_snapshot,
)
from rfl.hard_gate import HardGateMode
from rfl.evidence_fusion import TDAOutcome


class TestHardGateIntegration:
    """Tests for HardGateIntegration class."""
    
    def test_initialization(self):
        """Test integration initialization."""
        integration = HardGateIntegration(
            mode=HardGateMode.SHADOW,
            enable_traces=True,
        )
        
        assert integration.mode == HardGateMode.SHADOW
        assert integration.enable_traces is True
        assert len(integration.decisions) == 0
    
    def test_evaluate_single_cycle(self):
        """Test evaluation of single cycle."""
        integration = HardGateIntegration(mode=HardGateMode.SHADOW)
        
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5, "depth": 0.3, "success": 0.2},
            success_count={},
            attempt_count={},
        )
        
        event_stats = create_mock_event_stats(
            total_events=100,
            blocked_events=10,
        )
        
        decision = integration.evaluate_cycle(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
        )
        
        assert decision.outcome in [
            TDAOutcome.PASS,
            TDAOutcome.WARN,
            TDAOutcome.BLOCK,
            TDAOutcome.SHADOW,
        ]
        assert len(integration.decisions) == 1
    
    def test_evaluate_multiple_cycles(self):
        """Test evaluation across multiple cycles."""
        integration = HardGateIntegration(mode=HardGateMode.SHADOW)
        
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5},
            success_count={},
            attempt_count={},
        )
        
        # Evaluate 5 cycles
        for cycle in range(5):
            event_stats = create_mock_event_stats(
                total_events=100,
                blocked_events=cycle * 2,  # Increasing block rate
            )
            
            integration.evaluate_cycle(
                cycle=cycle,
                policy_state=policy_state,
                event_stats=event_stats,
            )
        
        assert len(integration.decisions) == 5
    
    def test_get_aggregate_metrics(self):
        """Test aggregate metrics computation."""
        integration = HardGateIntegration(mode=HardGateMode.SHADOW)
        
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5},
            success_count={},
            attempt_count={},
        )
        
        # Evaluate a few cycles
        for cycle in range(3):
            event_stats = create_mock_event_stats(
                total_events=100,
                blocked_events=10,
            )
            
            integration.evaluate_cycle(
                cycle=cycle,
                policy_state=policy_state,
                event_stats=event_stats,
            )
        
        metrics = integration.get_aggregate_metrics()
        
        assert "hss_aggregate" in metrics
        assert "mean_block_rate" in metrics
        assert "outcome_distribution" in metrics
        assert metrics["total_cycles"] == 3
    
    def test_create_run_entry_with_tda(self):
        """Test creation of RunEntry with TDA fields."""
        integration = HardGateIntegration(mode=HardGateMode.SHADOW)
        
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5},
            success_count={},
            attempt_count={},
        )
        
        # Evaluate a cycle
        event_stats = create_mock_event_stats(total_events=100, blocked_events=5)
        integration.evaluate_cycle(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
        )
        
        # Create run entry
        performance_metrics = {
            "coverage_rate": 0.75,
            "novelty_rate": 0.5,
            "throughput": 10.0,
            "success_rate": 0.8,
            "abstention_fraction": 0.2,
        }
        
        run_entry = integration.create_run_entry_with_tda(
            run_id="test_run_001",
            experiment_id="EXP_001",
            slice_name="slice_a",
            mode="rfl",
            performance_metrics=performance_metrics,
            cycle_count=1,
        )
        
        assert run_entry.run_id == "test_run_001"
        assert run_entry.tda.HSS is not None
        assert run_entry.tda.block_rate is not None
        assert run_entry.tda.tda_outcome is not None
    
    def test_reset(self):
        """Test integration reset."""
        integration = HardGateIntegration(mode=HardGateMode.SHADOW)
        
        # Evaluate a cycle
        policy_state = create_policy_state_snapshot(
            policy_weights={"len": 0.5},
            success_count={},
            attempt_count={},
        )
        event_stats = create_mock_event_stats(total_events=100, blocked_events=10)
        
        integration.evaluate_cycle(
            cycle=0,
            policy_state=policy_state,
            event_stats=event_stats,
        )
        
        assert len(integration.decisions) == 1
        assert integration.previous_policy_hash is not None
        
        # Reset
        integration.reset()
        
        assert len(integration.decisions) == 0
        assert integration.previous_policy_hash is None


class TestMockHelpers:
    """Tests for mock helper functions."""
    
    def test_create_mock_event_stats(self):
        """Test mock event stats creation."""
        stats = create_mock_event_stats(
            total_events=100,
            blocked_events=10,
        )
        
        assert stats["total_events"] == 100
        assert stats["blocked"] == 10
        assert stats["passed"] == 90
    
    def test_create_mock_event_stats_explicit_passed(self):
        """Test mock event stats with explicit passed count."""
        stats = create_mock_event_stats(
            total_events=100,
            blocked_events=10,
            passed_events=85,
        )
        
        assert stats["total_events"] == 100
        assert stats["blocked"] == 10
        assert stats["passed"] == 85
    
    def test_create_policy_state_snapshot(self):
        """Test policy state snapshot creation."""
        snapshot = create_policy_state_snapshot(
            policy_weights={"a": 0.5, "b": 0.3},
            success_count={"x": 10, "y": 5},
            attempt_count={"x": 20, "y": 15},
        )
        
        assert "weights" in snapshot
        assert "success_count" in snapshot
        assert "attempt_count" in snapshot
        assert snapshot["weights"]["a"] == 0.5
