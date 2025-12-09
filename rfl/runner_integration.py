"""
RFL Runner Integration Helpers for TDA Hard Gate

Provides integration hooks for wiring evaluate_hard_gate_decision() into
U2Runner and RFLRunner. Produces HSS traces and Δp metrics.

PHASE II — U2 Uplift Experiments Extension
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from .hard_gate import (
    evaluate_hard_gate_decision,
    HardGateMode,
    HardGateDecision,
    aggregate_hss_traces,
)
from .evidence_fusion import RunEntry, TDAFields

logger = logging.getLogger(__name__)


class HardGateIntegration:
    """
    Integration helper for TDA Hard Gate evaluation in runners.
    
    Maintains state across cycles and aggregates HSS traces.
    """
    
    def __init__(
        self,
        mode: HardGateMode = HardGateMode.SHADOW,
        enable_traces: bool = True,
    ):
        """
        Initialize hard gate integration.
        
        Args:
            mode: Hard gate evaluation mode
            enable_traces: Whether to collect HSS traces
        """
        self.mode = mode
        self.enable_traces = enable_traces
        
        # State tracking
        self.decisions: List[HardGateDecision] = []
        self.previous_policy_hash: Optional[str] = None
        
        logger.info(f"Hard gate integration initialized: mode={mode.value}, traces={enable_traces}")
    
    def evaluate_cycle(
        self,
        cycle: int,
        policy_state: Dict[str, Any],
        event_stats: Dict[str, Any],
    ) -> HardGateDecision:
        """
        Evaluate hard gate for a single cycle.
        
        Args:
            cycle: Cycle number
            policy_state: Current policy state
            event_stats: Event verification statistics
        
        Returns:
            HardGateDecision for this cycle
        """
        decision = evaluate_hard_gate_decision(
            cycle=cycle,
            policy_state=policy_state,
            event_stats=event_stats,
            previous_policy_hash=self.previous_policy_hash,
            mode=self.mode,
        )
        
        # Update state
        if self.enable_traces:
            self.decisions.append(decision)
        
        if decision.hss_traces:
            self.previous_policy_hash = decision.hss_traces[0].policy_hash
        
        return decision
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate TDA metrics across all cycles.
        
        Returns:
            Dictionary with aggregate HSS, block rate, and outcome statistics
        """
        if not self.decisions:
            return {
                "hss_aggregate": {},
                "mean_block_rate": 0.0,
                "outcome_distribution": {},
            }
        
        # Aggregate HSS traces
        all_hss_traces = []
        for decision in self.decisions:
            all_hss_traces.extend(decision.hss_traces)
        
        hss_aggregate = aggregate_hss_traces(all_hss_traces)
        
        # Compute mean block rate
        block_rates = [
            d.tda_fields.block_rate
            for d in self.decisions
            if d.tda_fields.block_rate is not None
        ]
        mean_block_rate = sum(block_rates) / len(block_rates) if block_rates else 0.0
        
        # Outcome distribution
        outcome_counts = {}
        for decision in self.decisions:
            outcome = decision.outcome.value
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        return {
            "hss_aggregate": hss_aggregate,
            "mean_block_rate": mean_block_rate,
            "outcome_distribution": outcome_counts,
            "total_cycles": len(self.decisions),
        }
    
    def create_run_entry_with_tda(
        self,
        run_id: str,
        experiment_id: str,
        slice_name: str,
        mode: str,
        performance_metrics: Dict[str, float],
        cycle_count: int,
    ) -> RunEntry:
        """
        Create a RunEntry with TDA fields populated from hard gate decisions.
        
        Args:
            run_id: Run identifier
            experiment_id: Experiment identifier
            slice_name: Curriculum slice name
            mode: Run mode ("baseline" or "rfl")
            performance_metrics: Performance metrics dict
            cycle_count: Number of cycles executed
        
        Returns:
            RunEntry with TDA fields
        """
        # Compute aggregate TDA metrics
        aggregate = self.get_aggregate_metrics()
        
        # Determine overall TDA outcome
        outcome_dist = aggregate.get("outcome_distribution", {})
        if outcome_dist.get("block", 0) > 0:
            tda_outcome_value = "block"
        elif outcome_dist.get("warn", 0) > 0:
            tda_outcome_value = "warn"
        elif outcome_dist.get("pass", 0) > 0:
            tda_outcome_value = "pass"
        else:
            tda_outcome_value = "shadow"
        
        from .evidence_fusion import TDAOutcome
        tda_outcome = TDAOutcome(tda_outcome_value)
        
        # Create TDA fields
        hss_agg = aggregate.get("hss_aggregate", {})
        tda_fields = TDAFields(
            HSS=hss_agg.get("mean_stability", 0.5),
            block_rate=aggregate.get("mean_block_rate", 0.0),
            tda_outcome=tda_outcome,
        )
        
        # Create run entry
        run_entry = RunEntry(
            run_id=run_id,
            experiment_id=experiment_id,
            slice_name=slice_name,
            mode=mode,
            coverage_rate=performance_metrics.get("coverage_rate", 0.0),
            novelty_rate=performance_metrics.get("novelty_rate", 0.0),
            throughput=performance_metrics.get("throughput", 0.0),
            success_rate=performance_metrics.get("success_rate", 0.0),
            abstention_fraction=performance_metrics.get("abstention_fraction", 1.0),
            tda=tda_fields,
            cycle_count=cycle_count,
            metadata={
                "hss_trend": hss_agg.get("trend", "unknown"),
                "tda_hard_gate_mode": self.mode.value,
            },
        )
        
        return run_entry
    
    def reset(self) -> None:
        """Reset integration state for new run."""
        self.decisions.clear()
        self.previous_policy_hash = None


def create_mock_event_stats(
    total_events: int = 100,
    blocked_events: int = 0,
    passed_events: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create mock event statistics for hard gate evaluation.
    
    Useful for testing or when event verification is not yet integrated.
    
    Args:
        total_events: Total number of events
        blocked_events: Number of blocked events
        passed_events: Number of passed events (defaults to total - blocked)
    
    Returns:
        Event statistics dictionary
    """
    if passed_events is None:
        passed_events = total_events - blocked_events
    
    return {
        "total_events": total_events,
        "blocked": blocked_events,
        "passed": passed_events,
    }


def create_policy_state_snapshot(
    policy_weights: Dict[str, float],
    success_count: Dict[str, int],
    attempt_count: Dict[str, int],
) -> Dict[str, Any]:
    """
    Create policy state snapshot for hard gate evaluation.
    
    Args:
        policy_weights: Policy weight parameters
        success_count: Success count per candidate
        attempt_count: Attempt count per candidate
    
    Returns:
        Policy state dictionary
    """
    return {
        "weights": policy_weights,
        "success_count": success_count,
        "attempt_count": attempt_count,
    }


# Example integration patterns for runners

def integrate_with_rfl_runner(runner: Any, mode: HardGateMode = HardGateMode.SHADOW) -> HardGateIntegration:
    """
    Integrate hard gate evaluation with RFLRunner.
    
    Usage:
        runner = RFLRunner(config)
        hard_gate = integrate_with_rfl_runner(runner, mode=HardGateMode.SHADOW)
        
        # In run loop:
        decision = hard_gate.evaluate_cycle(
            cycle=cycle,
            policy_state=create_policy_state_snapshot(
                runner.policy_weights,
                runner.success_count,
                runner.attempt_count,
            ),
            event_stats=create_mock_event_stats(total_events=100, blocked_events=5),
        )
    
    Args:
        runner: RFLRunner instance
        mode: Hard gate mode
    
    Returns:
        HardGateIntegration instance attached to runner
    """
    integration = HardGateIntegration(mode=mode, enable_traces=True)
    
    # Store integration on runner for later access
    if not hasattr(runner, '_hard_gate_integration'):
        runner._hard_gate_integration = integration
        logger.info(f"Hard gate integration attached to RFLRunner (mode={mode.value})")
    
    return integration


def integrate_with_u2_runner(runner: Any, mode: HardGateMode = HardGateMode.SHADOW) -> HardGateIntegration:
    """
    Integrate hard gate evaluation with U2Runner.
    
    Usage:
        runner = U2Runner(config)
        hard_gate = integrate_with_u2_runner(runner, mode=HardGateMode.SHADOW)
        
        # In cycle execution:
        decision = hard_gate.evaluate_cycle(
            cycle=cycle,
            policy_state={"policy": runner.policy},
            event_stats=create_mock_event_stats(total_events=50, blocked_events=2),
        )
    
    Args:
        runner: U2Runner instance
        mode: Hard gate mode
    
    Returns:
        HardGateIntegration instance attached to runner
    """
    integration = HardGateIntegration(mode=mode, enable_traces=True)
    
    # Store integration on runner for later access
    if not hasattr(runner, '_hard_gate_integration'):
        runner._hard_gate_integration = integration
        logger.info(f"Hard gate integration attached to U2Runner (mode={mode.value})")
    
    return integration


__all__ = [
    "HardGateIntegration",
    "create_mock_event_stats",
    "create_policy_state_snapshot",
    "integrate_with_rfl_runner",
    "integrate_with_u2_runner",
]
