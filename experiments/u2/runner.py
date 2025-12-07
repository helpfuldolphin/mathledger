"""
U2 Planner Runner

Core execution engine for U2 experiments with:
- Deterministic cycle execution
- Snapshot support
- Trace logging
- Policy-driven search
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from rfl.prng import DeterministicPRNG, int_to_hex_seed

from .frontier import FrontierManager, BeamAllocator
from .logging import U2TraceLogger
from .policy import create_policy, SearchPolicy
from .schema import EventType
from .snapshots import SnapshotData, create_snapshot_name, save_snapshot
from .safety_slo import SafetyEnvelope, SafetyStatus


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


@dataclass(frozen=True)
class U2SafetyContext:
    """
    Type-safe safety context for U2 experiment execution.
    
    Ensures all safety-critical parameters are explicit and typed.
    
    INVARIANTS:
    - config must be a valid U2Config
    - perf_threshold_ms must be positive
    - max_cycles must be positive
    - slice_name and mode must match config
    """
    config: U2Config
    perf_threshold_ms: float
    max_cycles: int
    enable_safe_eval: bool
    slice_name: str
    mode: Literal["baseline", "rfl"]
    
    def __post_init__(self) -> None:
        """Validate safety context invariants."""
        if self.perf_threshold_ms <= 0:
            raise ValueError(f"perf_threshold_ms must be positive, got {self.perf_threshold_ms}")
        if self.max_cycles <= 0:
            raise ValueError(f"max_cycles must be positive, got {self.max_cycles}")
        if self.slice_name != self.config.slice_name:
            raise ValueError(
                f"slice_name mismatch: context={self.slice_name}, config={self.config.slice_name}"
            )
        if self.mode != self.config.mode:
            raise ValueError(
                f"mode mismatch: context={self.mode}, config={self.config.mode}"
            )


@dataclass(frozen=True)
class U2Snapshot:
    """
    Type-safe snapshot representation.
    
    Replaces bare dict snapshots with typed structure.
    """
    config: U2Config
    cycles_completed: int
    state_hash: str
    snapshot_data: SnapshotData


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
            
            # Execute candidate
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


# Type-safe wrapper functions

def safe_eval_expression(expr: str) -> float:
    """
    Type-safe wrapper for evaluating expressions.
    
    SECURITY: Never uses bare eval(). Restricts to safe numeric evaluation.
    
    Args:
        expr: Expression string (must be numeric)
        
    Returns:
        Evaluated float value
        
    Raises:
        ValueError: If expression is not a valid numeric literal
    """
    try:
        # Only allow numeric literals - no eval() of arbitrary code
        value = float(expr.strip())
        return value
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid numeric expression: {expr}") from e


def save_u2_snapshot(path: Path, snapshot: U2Snapshot) -> str:
    """
    Type-safe snapshot save.
    
    Args:
        path: Path to save snapshot
        snapshot: Typed U2Snapshot object
        
    Returns:
        Snapshot hash
    """
    return save_snapshot(snapshot.snapshot_data, path)


def load_u2_snapshot(path: Path) -> U2Snapshot:
    """
    Type-safe snapshot load.
    
    Args:
        path: Path to snapshot file
        
    Returns:
        Typed U2Snapshot object
        
    Raises:
        NoSnapshotFoundError: Snapshot file not found
        SnapshotCorruptionError: Snapshot file is corrupted
    """
    from .snapshots import load_snapshot
    
    snapshot_data = load_snapshot(path)
    
    # Reconstruct config from snapshot
    config = U2Config(
        experiment_id=snapshot_data.experiment_id,
        slice_name=snapshot_data.slice_name,
        mode=snapshot_data.mode,
        total_cycles=snapshot_data.total_cycles,
        master_seed=int(snapshot_data.master_seed, 16),  # Convert hex back to int
    )
    
    # Create typed snapshot
    return U2Snapshot(
        config=config,
        cycles_completed=snapshot_data.current_cycle,
        state_hash=snapshot_data.compute_hash(),
        snapshot_data=snapshot_data,
    )


def run_u2_experiment(
    safety_ctx: U2SafetyContext,
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    output_dir: Optional[Path] = None,
) -> SafetyEnvelope:
    """
    Type-safe entrypoint for running U2 experiments with safety envelope.
    
    This is the primary entrypoint that enforces type safety and produces
    a SafetyEnvelope for SLO tracking.
    
    Args:
        safety_ctx: Type-safe safety context
        execute_fn: Candidate execution function
        output_dir: Optional output directory for traces
        
    Returns:
        SafetyEnvelope with run results and safety status
    """
    start_time = time.time()
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(f"./outputs/{safety_ctx.config.experiment_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up trace path
    trace_path = output_dir / f"{safety_ctx.config.experiment_id}_trace.jsonl"
    
    # Run experiment
    runner = U2Runner(safety_ctx.config)
    
    lint_issues: List[str] = []
    warnings: List[str] = []
    
    try:
        with U2TraceLogger(
            output_path=trace_path,
            experiment_id=safety_ctx.config.experiment_id,
            slice_name=safety_ctx.slice_name,
            mode=safety_ctx.mode,
            master_seed=int_to_hex_seed(safety_ctx.config.master_seed),
        ) as logger:
            trace_ctx = TracedExperimentContext(trace_logger=logger)
            
            for cycle in range(min(safety_ctx.max_cycles, safety_ctx.config.total_cycles)):
                cycle_start = time.time()
                
                result = runner.run_cycle(cycle, execute_fn, trace_ctx)
                
                cycle_elapsed_ms = (time.time() - cycle_start) * 1000
                
                # Check performance threshold
                if cycle_elapsed_ms > safety_ctx.perf_threshold_ms:
                    warnings.append(
                        f"cycle {cycle} exceeded perf threshold: {cycle_elapsed_ms:.1f}ms > {safety_ctx.perf_threshold_ms}ms"
                    )
    
    except Exception as e:
        lint_issues.append(f"Experiment failed with error: {str(e)}")
    
    # Compute total elapsed time
    total_elapsed_ms = (time.time() - start_time) * 1000
    perf_ok = total_elapsed_ms < (safety_ctx.perf_threshold_ms * safety_ctx.max_cycles)
    
    # Determine safety status
    safety_status: SafetyStatus
    if len(lint_issues) > 0:
        safety_status = "BLOCK"
    elif len(warnings) > 3 or not perf_ok:
        safety_status = "WARN"
    else:
        safety_status = "OK"
    
    # Build safety envelope
    envelope: SafetyEnvelope = {
        "schema_version": "1.0",
        "config": {
            "experiment_id": safety_ctx.config.experiment_id,
            "slice_name": safety_ctx.slice_name,
            "mode": safety_ctx.mode,
            "master_seed": safety_ctx.config.master_seed,
            "total_cycles": safety_ctx.config.total_cycles,
        },
        "perf_ok": perf_ok,
        "safety_status": safety_status,
        "lint_issues": lint_issues,
        "warnings": warnings,
        "run_id": safety_ctx.config.experiment_id,
        "slice_name": safety_ctx.slice_name,
        "mode": safety_ctx.mode,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    return envelope
