# PHASE II — NOT USED IN PHASE I
# U2 Runner for uplift experiments

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from rfl.prng import DeterministicPRNG, int_to_hex_seed
from .snapshots import SnapshotData, save_snapshot
from .logging import U2TraceLogger


@dataclass
class U2Config:
    """Configuration for U2 experiment."""
    experiment_id: str
    slice_name: str
    mode: str  # "baseline" or "rfl"
    total_cycles: int
    master_seed: int
    snapshot_interval: int
    snapshot_dir: Path
    output_dir: Path
    slice_config: Dict[str, Any]


@dataclass
class CycleResult:
    """Result from a single experiment cycle."""
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Any
    success: bool


class TracedExperimentContext:
    """
    Context manager for traced experiment execution.
    
    Provides cycle timing and telemetry logging hooks.
    """
    
    def __init__(self, logger: U2TraceLogger, slice_name: str, mode: str):
        """
        Initialize traced context.
        
        Args:
            logger: Trace logger instance
            slice_name: Experiment slice name
            mode: Execution mode (baseline/rfl)
        """
        self.logger = logger
        self.slice_name = slice_name
        self.mode = mode
        self._cycle_start_time = None
    
    def begin_cycle(self, cycle: int):
        """Mark the start of a cycle."""
        self._cycle_start_time = time.time()
    
    def end_cycle(self, cycle: int):
        """Mark the end of a cycle."""
        # Could log timing information here
        self._cycle_start_time = None
    
    def log_cycle_telemetry(self, cycle: int, telemetry: dict):
        """Log cycle telemetry."""
        self.logger.log_cycle_telemetry(cycle, telemetry)


class U2Runner:
    """
    U2 experiment runner with snapshot support.
    
    Supports two modes:
    - baseline: Random selection
    - rfl: Policy-driven selection based on success rates
    """
    
    def __init__(self, config: U2Config):
        """
        Initialize U2 runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.cycle_index = 0
        self.ht_series: List[str] = []
        self.policy_update_count = 0
        self.success_count: Dict[str, int] = defaultdict(int)
        self.attempt_count: Dict[str, int] = defaultdict(int)
        
        # Initialize PRNG
        self.prng = DeterministicPRNG(int_to_hex_seed(config.master_seed))
    
    def run_cycle(
        self,
        items: List[str],
        execute_fn: Callable[[str, int], Tuple[bool, Any]],
    ) -> CycleResult:
        """
        Run a single experiment cycle.
        
        Args:
            items: List of items to choose from
            execute_fn: Function to execute item (returns success, result)
            
        Returns:
            CycleResult with execution details
        """
        # Generate cycle seed
        cycle_prng = self.prng.for_path("cycle", self.cycle_index)
        cycle_seed = int(cycle_prng.random() * (2**32))
        
        # Select item based on mode
        if self.config.mode == "baseline":
            # Random selection
            item = cycle_prng.choice(items)
        elif self.config.mode == "rfl":
            # Policy-driven selection based on success rates
            item = self._select_rfl_item(items, cycle_prng)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # Execute item
        success, result = execute_fn(item, cycle_seed)
        
        # Update policy stats
        if self.config.mode == "rfl":
            self.attempt_count[item] += 1
            if success:
                self.success_count[item] += 1
                self.policy_update_count += 1
        
        # Update HT series
        self.ht_series.append(item)
        
        # Create result
        cycle_result = CycleResult(
            cycle_index=self.cycle_index,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            seed=cycle_seed,
            item=item,
            result=result,
            success=success,
        )
        
        # Increment cycle
        self.cycle_index += 1
        
        return cycle_result
    
    def _select_rfl_item(self, items: List[str], prng: DeterministicPRNG) -> str:
        """
        Select item using RFL policy (weighted by success rate).
        
        Args:
            items: Available items
            prng: PRNG for this cycle
            
        Returns:
            Selected item
        """
        # Calculate weights based on success rates
        weights = []
        for item in items:
            attempts = self.attempt_count.get(item, 0)
            successes = self.success_count.get(item, 0)
            
            if attempts == 0:
                # Unexplored items get high weight
                weight = 1.0
            else:
                # Weight by success rate, with small baseline
                success_rate = successes / attempts
                weight = 0.1 + success_rate
            
            weights.append(weight)
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            weights = [1.0] * len(items)
            total = len(items)
        
        probs = [w / total for w in weights]
        
        # Weighted random selection
        r = prng.random()
        cumulative = 0.0
        for item, prob in zip(items, probs):
            cumulative += prob
            if r <= cumulative:
                return item
        
        # Fallback (shouldn't happen, but handle floating point edge cases)
        return items[-1]
    
    def maybe_save_snapshot(self) -> Optional[Path]:
        """
        Save snapshot if at snapshot interval.
        
        Returns:
            Path to saved snapshot, or None if not saved
        """
        if self.config.snapshot_interval <= 0:
            return None
        
        if self.cycle_index % self.config.snapshot_interval != 0:
            return None
        
        snapshot = self.capture_state()
        path = self.config.snapshot_dir / f"snapshot_{self.config.experiment_id}_cycle_{self.cycle_index}.snap"
        save_snapshot(snapshot, path)
        return path
    
    def capture_state(self) -> SnapshotData:
        """
        Capture current state as snapshot.
        
        Returns:
            SnapshotData with current state
        """
        return SnapshotData(
            cycle_index=self.cycle_index,
            ht_series=list(self.ht_series),
            policy_update_count=self.policy_update_count,
            success_count=dict(self.success_count),
            attempt_count=dict(self.attempt_count),
            experiment_id=self.config.experiment_id,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            master_seed=self.config.master_seed,
        )
    
    def restore_state(self, snapshot: SnapshotData):
        """
        Restore state from snapshot.
        
        Args:
            snapshot: Snapshot data to restore from
        """
        # Validate snapshot matches configuration
        if snapshot.experiment_id != self.config.experiment_id:
            raise ValueError(
                f"Snapshot experiment_id mismatch: {snapshot.experiment_id} != {self.config.experiment_id}"
            )
        if snapshot.slice_name != self.config.slice_name:
            raise ValueError(
                f"Snapshot slice_name mismatch: {snapshot.slice_name} != {self.config.slice_name}"
            )
        if snapshot.mode != self.config.mode:
            raise ValueError(
                f"Snapshot mode mismatch: {snapshot.mode} != {self.config.mode}"
            )
        if snapshot.master_seed != self.config.master_seed:
            raise ValueError(
                f"Snapshot master_seed mismatch: {snapshot.master_seed} != {self.config.master_seed}"
            )
        
        # Restore state
        self.cycle_index = snapshot.cycle_index
        self.ht_series = list(snapshot.ht_series)
        self.policy_update_count = snapshot.policy_update_count
        self.success_count = defaultdict(int, snapshot.success_count)
        self.attempt_count = defaultdict(int, snapshot.attempt_count)
        
        # Re-initialize PRNG to same state
        self.prng = DeterministicPRNG(int_to_hex_seed(self.config.master_seed))


def run_with_traces(
    config: U2Config,
    items: List[str],
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    trace_logger: U2TraceLogger,
) -> List[CycleResult]:
    """
    Run experiment with full trace logging.
    
    Args:
        config: Experiment configuration
        items: List of items
        execute_fn: Execution function
        trace_logger: Trace logger instance
        
    Returns:
        List of cycle results
    """
    runner = U2Runner(config)
    context = TracedExperimentContext(trace_logger, config.slice_name, config.mode)
    
    results = []
    for i in range(config.total_cycles):
        context.begin_cycle(i)
        result = runner.run_cycle(items, execute_fn)
        context.log_cycle_telemetry(i, {
            "cycle": result.cycle_index,
            "slice": result.slice_name,
            "mode": result.mode,
            "seed": result.seed,
            "item": result.item,
            "result": str(result.result),
            "success": result.success,
        })
        context.end_cycle(i)
        results.append(result)
    
    return results
