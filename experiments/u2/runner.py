"""
U2 Planner Runner

Core execution engine for U2 experiments with:
- Deterministic cycle execution
- Snapshot support
- Trace logging
- Policy-driven search
- Runtime safety enforcement (Neural Link)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rfl.prng import DeterministicPRNG, int_to_hex_seed

from .frontier import FrontierManager, BeamAllocator
from .logging import U2TraceLogger
from .policy import create_policy, SearchPolicy
from .schema import EventType
from .snapshots import SnapshotData, create_snapshot_name, save_snapshot
from .safety import (
    U2SafetyContext,
    SafetyEnvelope,
    GateDecision,
    evaluate_hard_gate_decision,
)


@dataclass
class U2Config:
    """Configuration for U2 experiment."""
    
    experiment_id: str
    slice_name: str
    mode: str  # "baseline" or "rfl"
    total_cycles: int
    master_seed: int
    
    # Optional parameters
    snapshot_interval: int = 0
    snapshot_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    slice_config: Dict[str, Any] = field(default_factory=dict)
    
    # Search parameters
    max_beam_width: int = 100
    max_depth: int = 10
    
    # Budget parameters
    cycle_budget_s: float = 60.0
    max_candidates_per_cycle: int = 100


@dataclass
class CycleResult:
    """Result of a single cycle execution."""
    
    cycle: int
    success: bool
    candidates_processed: int
    candidates_generated: int
    time_elapsed_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TracedExperimentContext:
    """Context for traced experiment execution."""
    
    trace_logger: U2TraceLogger
    current_cycle: int = 0


class U2Runner:
    """
    U2 Planner execution engine.
    
    INVARIANTS:
    - Deterministic execution given same config and seed
    - State is fully serializable for snapshots
    - All randomness flows through hierarchical PRNG
    """
    
    def __init__(self, config: U2Config):
        """
        Initialize U2 runner.
        
        Args:
            config: U2 configuration
        """
        self.config = config
        
        # Initialize PRNG hierarchy
        master_seed_hex = int_to_hex_seed(config.master_seed)
        self.master_prng = DeterministicPRNG(master_seed_hex)
        self.slice_prng = self.master_prng.for_path("slice", config.slice_name)
        
        # Initialize safety context (Neural Link Cortex)
        self.safety_context = U2SafetyContext()
        self.safety_prng = self.master_prng.for_path("safety")
        
        # Initialize frontier manager
        self.frontier = FrontierManager(
            max_beam_width=config.max_beam_width,
            max_depth=config.max_depth,
            prng=self.slice_prng.for_path("frontier"),
        )
        
        # Initialize beam allocator
        self.beam_allocator = BeamAllocator(
            total_budget=config.max_candidates_per_cycle * config.total_cycles,
        )
        
        # Initialize policy
        self.policy: Optional[SearchPolicy] = None
        
        # Execution state
        self.current_cycle = 0
        self.results: List[CycleResult] = []
        
        # Statistics
        self.stats = {
            "total_candidates_processed": 0,
            "total_candidates_generated": 0,
            "total_time_s": 0.0,
        }
    
    def initialize_policy(self, feedback_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize search policy.
        
        Args:
            feedback_data: Feedback data for RFL policy
        """
        policy_prng = self.slice_prng.for_path("policy")
        self.policy = create_policy(
            mode=self.config.mode,
            prng=policy_prng,
            feedback_data=feedback_data,
        )
    
    def run_cycle(
        self,
        cycle: int,
        execute_fn: Callable[[str, int], Tuple[bool, Any]],
        trace_ctx: Optional[TracedExperimentContext] = None,
    ) -> CycleResult:
        """
        Run a single cycle.
        
        Args:
            cycle: Cycle number
            execute_fn: Function to execute candidates
            trace_ctx: Optional trace context
            
        Returns:
            CycleResult
        """
        start_time = time.time()
        
        # Create cycle-specific PRNG
        cycle_prng = self.slice_prng.for_path("cycle", str(cycle))
        cycle_seed = int_to_hex_seed(cycle)
        
        # Log cycle start
        if trace_ctx:
            trace_ctx.trace_logger.start_cycle(cycle, cycle_seed)
            trace_ctx.current_cycle = cycle
        
        # Initialize policy if not done
        if self.policy is None:
            self.initialize_policy()
        
        candidates_processed = 0
        candidates_generated = 0
        
        # Process candidates from frontier
        budget_remaining = self.config.max_candidates_per_cycle
        
        while budget_remaining > 0 and not self.frontier.is_empty():
            # Pop candidate
            candidate = self.frontier.pop()
            if candidate is None:
                break
            
            # Log frontier pop
            if trace_ctx:
                trace_ctx.trace_logger.log_event(
                    EventType.FRONTIER_POP,
                    cycle=cycle,
                    data={
                        "item": str(candidate.item),
                        "priority": candidate.priority,
                        "depth": candidate.depth,
                    }
                )
            
            # BLOCKING CALL: Cortex approval via Hard Gate
            # NO candidate executes without this approval
            safety_envelope = evaluate_hard_gate_decision(
                candidate=candidate.item,
                cycle=cycle,
                safety_context=self.safety_context,
                prng=self.safety_prng.for_path("gate", str(cycle)),
                max_depth=self.config.max_depth,
            )
            
            # Log safety decision
            if trace_ctx:
                trace_ctx.trace_logger.log_event(
                    EventType.FRONTIER_POP,  # Reuse existing event type
                    cycle=cycle,
                    data={
                        "safety_gate": safety_envelope.to_dict(),
                    }
                )
            
            # Block execution if not approved
            if safety_envelope.decision != GateDecision.APPROVED:
                # Log rejection/abstention
                if trace_ctx:
                    trace_ctx.trace_logger.log_event(
                        EventType.DERIVE_FAILURE,
                        cycle=cycle,
                        data={
                            "item": str(candidate.item),
                            "blocked_by_safety_gate": True,
                            "decision": safety_envelope.decision.value,
                            "reason": safety_envelope.reason,
                        }
                    )
                # Skip this candidate - Cortex rejected
                continue
            
            # Execute candidate (only if approved)
            try:
                success, result = execute_fn(candidate.item, cycle)
                
                # Log result
                if trace_ctx:
                    event_type = EventType.DERIVE_SUCCESS if success else EventType.DERIVE_FAILURE
                    trace_ctx.trace_logger.log_event(
                        event_type,
                        cycle=cycle,
                        data={
                            "item": str(candidate.item),
                            "result": str(result),
                        }
                    )
                
                candidates_processed += 1
                budget_remaining -= 1
                
                # Generate new candidates if successful
                if success:
                    new_candidates = self._generate_candidates(
                        candidate.item,
                        result,
                        candidate.depth + 1,
                        cycle_prng,
                    )
                    
                    # Rank new candidates
                    ranked = self.policy.rank(new_candidates)
                    
                    # Push to frontier
                    for new_item, score in ranked:
                        priority = self.policy.get_priority(new_item, score)
                        pushed = self.frontier.push(
                            item=new_item,
                            priority=priority,
                            depth=candidate.depth + 1,
                            score=score,
                        )
                        
                        if pushed:
                            candidates_generated += 1
                            
                            # Log frontier push
                            if trace_ctx:
                                trace_ctx.trace_logger.log_event(
                                    EventType.FRONTIER_PUSH,
                                    cycle=cycle,
                                    data={
                                        "item": str(new_item),
                                        "priority": priority,
                                        "depth": candidate.depth + 1,
                                        "score": score,
                                    }
                                )
            
            except Exception as e:
                # Log error
                if trace_ctx:
                    trace_ctx.trace_logger.log_event(
                        EventType.DERIVE_FAILURE,
                        cycle=cycle,
                        data={
                            "item": str(candidate.item),
                            "error": str(e),
                        }
                    )
        
        # Check budget exhaustion
        if budget_remaining == 0:
            if trace_ctx:
                trace_ctx.trace_logger.log_event(
                    EventType.BUDGET_EXCEEDED,
                    cycle=cycle,
                    data={"budget": self.config.max_candidates_per_cycle}
                )
        
        time_elapsed = time.time() - start_time
        
        # Create result
        result = CycleResult(
            cycle=cycle,
            success=candidates_processed > 0,
            candidates_processed=candidates_processed,
            candidates_generated=candidates_generated,
            time_elapsed_s=time_elapsed,
            metadata={
                "frontier_size": self.frontier.size(),
                "beam_stats": self.beam_allocator.get_stats(),
            }
        )
        
        # Log cycle end
        if trace_ctx:
            trace_ctx.trace_logger.end_cycle(metadata=result.metadata)
        
        # Update stats
        self.stats["total_candidates_processed"] += candidates_processed
        self.stats["total_candidates_generated"] += candidates_generated
        self.stats["total_time_s"] += time_elapsed
        
        self.results.append(result)
        self.current_cycle = cycle
        
        return result
    
    def _generate_candidates(
        self,
        parent: Any,
        result: Any,
        depth: int,
        prng: DeterministicPRNG,
    ) -> List[Any]:
        """
        Generate new candidates from successful execution.
        
        Args:
            parent: Parent candidate
            result: Execution result
            depth: New candidate depth
            prng: PRNG for generation
            
        Returns:
            List of new candidates
        """
        # Simple mock generation
        # In real implementation, this would use result to generate candidates
        num_candidates = prng.randint(1, 3)
        
        candidates = []
        for i in range(num_candidates):
            candidate = {
                "item": f"{parent}_child_{i}",
                "depth": depth,
                "parent": parent,
            }
            candidates.append(candidate)
        
        return candidates
    
    def save_snapshot(self, cycle: int) -> str:
        """
        Save snapshot of current state.
        
        Args:
            cycle: Current cycle number
            
        Returns:
            Snapshot hash
        """
        snapshot = SnapshotData(
            experiment_id=self.config.experiment_id,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            master_seed=int_to_hex_seed(self.config.master_seed),
            current_cycle=cycle,
            total_cycles=self.config.total_cycles,
            frontier_state=self.frontier.get_state(),
            prng_state=self.slice_prng.get_state(),
            stats=self.stats,
            safety_context=self.safety_context.to_dict(),
            snapshot_cycle=cycle,
            snapshot_timestamp=int(time.time()),
        )
        
        snapshot_dir = self.config.snapshot_dir or Path("snapshots")
        snapshot_path = snapshot_dir / create_snapshot_name(self.config.experiment_id, cycle)
        
        return save_snapshot(snapshot, snapshot_path)
    
    def restore_state(self, snapshot: SnapshotData) -> None:
        """
        Restore state from snapshot.
        
        Args:
            snapshot: Snapshot data
        """
        # Restore execution state
        self.current_cycle = snapshot.current_cycle
        self.stats = snapshot.stats
        
        # Restore safety context
        if snapshot.safety_context:
            self.safety_context = U2SafetyContext.from_dict(snapshot.safety_context)
        
        # Restore frontier
        self.frontier.set_state(snapshot.frontier_state)
        
        # Restore PRNG
        self.slice_prng.set_state(snapshot.prng_state)
        
        # Reinitialize policy
        self.initialize_policy()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current runner state."""
        return {
            "current_cycle": self.current_cycle,
            "stats": self.stats,
            "frontier_stats": self.frontier.get_stats(),
            "beam_stats": self.beam_allocator.get_stats(),
            "safety_context": self.safety_context.to_dict(),
        }


def run_with_traces(
    config: U2Config,
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    trace_path: Path,
    trace_events: Optional[set] = None,
) -> List[CycleResult]:
    """
    Run U2 experiment with trace logging.
    
    Args:
        config: U2 configuration
        execute_fn: Candidate execution function
        trace_path: Path for trace output
        trace_events: Set of event types to log
        
    Returns:
        List of cycle results
    """
    runner = U2Runner(config)
    
    with U2TraceLogger(
        output_path=trace_path,
        experiment_id=config.experiment_id,
        slice_name=config.slice_name,
        mode=config.mode,
        master_seed=int_to_hex_seed(config.master_seed),
        event_filter=trace_events,
    ) as logger:
        trace_ctx = TracedExperimentContext(trace_logger=logger)
        
        for cycle in range(config.total_cycles):
            runner.run_cycle(cycle, execute_fn, trace_ctx)
            
            # Save snapshot if configured
            if config.snapshot_interval > 0 and (cycle + 1) % config.snapshot_interval == 0:
                runner.save_snapshot(cycle)
    
    return runner.results
