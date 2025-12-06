"""
U2 Runner Module for Uplift Experiments

This module provides the core runner logic for U2 uplift experiments with:
- Dataclass-based configuration and results
- Type-safe interfaces
- Performance-optimized cycle execution
- Snapshot support

PHASE II — NOT USED IN PHASE I
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.u2.snapshots import SnapshotData, save_snapshot
from experiments.u2.logging import U2TraceLogger


@dataclass
class U2Config:
    """
    Configuration for U2 uplift experiment.
    
    Attributes:
        experiment_id: Unique identifier for this experiment run
        slice_name: Name of the curriculum slice
        mode: "baseline" or "rfl"
        total_cycles: Total number of cycles to run
        master_seed: Master random seed for determinism
        snapshot_interval: Save snapshot every N cycles (0 = disabled)
        snapshot_dir: Directory for snapshot files
        output_dir: Directory for output files
        slice_config: Configuration dict for the slice
    """
    experiment_id: str
    slice_name: str
    mode: str
    total_cycles: int
    master_seed: int
    snapshot_interval: int
    snapshot_dir: Path
    output_dir: Path
    slice_config: Dict[str, Any]


@dataclass
class CycleResult:
    """
    Result from a single cycle execution.
    
    Attributes:
        cycle_index: Index of this cycle (0-based)
        slice_name: Name of the slice
        mode: Execution mode ("baseline" or "rfl")
        seed: Seed used for this cycle
        item: Item selected for this cycle
        result: Result from execution
        success: Whether execution succeeded
        ht: H_t value for this cycle (hash of telemetry)
    """
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Any
    success: bool
    ht: str = ""


@dataclass
class TelemetryRecord:
    """
    Telemetry record for a cycle.
    
    Performance-optimized representation for serialization.
    """
    cycle: int
    slice: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    label: str = "PHASE II — NOT USED IN PHASE I"


class TracedExperimentContext:
    """
    Context manager for traced experiment execution.
    
    Provides timing and telemetry logging for cycles.
    """
    
    def __init__(self, trace_logger: U2TraceLogger, slice_name: str, mode: str):
        self.trace_logger = trace_logger
        self.slice_name = slice_name
        self.mode = mode
        self._cycle_start_times: Dict[int, float] = {}
    
    def begin_cycle(self, cycle_index: int) -> None:
        """Mark the beginning of a cycle for timing."""
        import time
        self._cycle_start_times[cycle_index] = time.time()
    
    def end_cycle(self, cycle_index: int) -> None:
        """Mark the end of a cycle and log timing."""
        import time
        if cycle_index in self._cycle_start_times:
            elapsed = time.time() - self._cycle_start_times[cycle_index]
            # Could emit timing event here if needed
            del self._cycle_start_times[cycle_index]
    
    def log_cycle_telemetry(self, cycle_index: int, telemetry: Dict[str, Any]) -> None:
        """Log cycle telemetry."""
        # Delegate to trace logger
        pass


class U2Runner:
    """
    Main runner for U2 uplift experiments.
    
    Manages state, policy updates, and cycle execution.
    """
    
    def __init__(self, config: U2Config):
        self.config = config
        self.cycle_index: int = 0
        self.ht_series: List[str] = []
        self.policy_update_count: int = 0
        self.success_count: Dict[str, int] = {}
        self.attempt_count: Dict[str, int] = {}
        
        # Initialize RFL state if in RFL mode
        if config.mode == "rfl":
            self._init_rfl_state()
    
    def _init_rfl_state(self) -> None:
        """Initialize RFL policy state."""
        # In RFL mode, maintain success/attempt counts for policy
        pass
    
    def run_cycle(
        self,
        items: List[str],
        execute_fn: Callable[[str, int], Tuple[bool, Any]]
    ) -> CycleResult:
        """
        Run a single cycle of the experiment.
        
        Args:
            items: List of available items
            execute_fn: Function to execute an item (item, seed) -> (success, result)
            
        Returns:
            CycleResult with execution details
        """
        # Derive cycle seed from master seed and cycle index
        import hashlib
        seed_str = f"{self.config.master_seed}_{self.cycle_index}"
        cycle_seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        
        # Select item based on mode
        if self.config.mode == "baseline":
            item = self._select_item_baseline(items, cycle_seed)
        elif self.config.mode == "rfl":
            item = self._select_item_rfl(items, cycle_seed)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # Execute item
        success, result = execute_fn(item, cycle_seed)
        
        # Update policy if RFL mode
        if self.config.mode == "rfl":
            self._update_rfl_policy(item, success)
        
        # Compute H_t (hash of telemetry)
        import json
        telemetry_str = json.dumps({
            "cycle": self.cycle_index,
            "item": item,
            "success": success,
            "seed": cycle_seed,
        }, sort_keys=True)
        ht = hashlib.sha256(telemetry_str.encode()).hexdigest()
        self.ht_series.append(ht)
        
        # Create result
        cycle_result = CycleResult(
            cycle_index=self.cycle_index,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            seed=cycle_seed,
            item=item,
            result=result,
            success=success,
            ht=ht,
        )
        
        # Increment cycle counter
        self.cycle_index += 1
        
        return cycle_result
    
    def _select_item_baseline(self, items: List[str], seed: int) -> str:
        """Select item using baseline (random) policy."""
        import random
        rng = random.Random(seed)
        return rng.choice(items)
    
    def _select_item_rfl(self, items: List[str], seed: int) -> str:
        """Select item using RFL policy (weighted by success rate)."""
        # Simple policy: weight items by historical success rate
        if not self.attempt_count:
            # Cold start: random selection
            return self._select_item_baseline(items, seed)
        
        # Compute weights based on success rate
        weights = []
        for item in items:
            attempts = self.attempt_count.get(item, 0)
            successes = self.success_count.get(item, 0)
            
            # Use UCB-like exploration: favor items with high success and low attempts
            if attempts == 0:
                weight = 1.0  # Exploration bonus
            else:
                success_rate = successes / attempts
                exploration_bonus = (2 * (self.cycle_index + 1) / attempts) ** 0.5
                weight = success_rate + exploration_bonus
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return self._select_item_baseline(items, seed)
        
        probs = [w / total_weight for w in weights]
        
        # Sample from weighted distribution
        import random
        rng = random.Random(seed)
        return rng.choices(items, weights=probs)[0]
    
    def _update_rfl_policy(self, item: str, success: bool) -> None:
        """Update RFL policy with execution result."""
        self.attempt_count[item] = self.attempt_count.get(item, 0) + 1
        if success:
            self.success_count[item] = self.success_count.get(item, 0) + 1
        self.policy_update_count += 1
    
    def capture_state(self) -> SnapshotData:
        """Capture current state for snapshot."""
        return SnapshotData(
            cycle_index=self.cycle_index,
            ht_series=self.ht_series.copy(),
            policy_update_count=self.policy_update_count,
            success_count=self.success_count.copy(),
            attempt_count=self.attempt_count.copy(),
            config=self.config,
        )
    
    def restore_state(self, snapshot: SnapshotData) -> None:
        """Restore state from snapshot."""
        self.cycle_index = snapshot.cycle_index
        self.ht_series = snapshot.ht_series.copy()
        self.policy_update_count = snapshot.policy_update_count
        self.success_count = snapshot.success_count.copy()
        self.attempt_count = snapshot.attempt_count.copy()
        # Config should match, but don't overwrite
    
    def maybe_save_snapshot(self) -> Optional[Path]:
        """Save snapshot if interval is reached."""
        if self.config.snapshot_interval <= 0:
            return None
        
        if self.cycle_index > 0 and self.cycle_index % self.config.snapshot_interval == 0:
            snapshot = self.capture_state()
            snapshot_path = (
                self.config.snapshot_dir / 
                f"snapshot_{self.config.experiment_id}_{self.cycle_index}.snap"
            )
            save_snapshot(snapshot, snapshot_path)
            return snapshot_path
        
        return None


def run_with_traces(
    runner: U2Runner,
    trace_logger: U2TraceLogger,
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    items: List[str],
    cycles: int,
) -> List[CycleResult]:
    """
    Run multiple cycles with trace logging enabled.
    
    Args:
        runner: U2Runner instance
        trace_logger: Logger for telemetry
        execute_fn: Execution function
        items: Available items
        cycles: Number of cycles to run
        
    Returns:
        List of CycleResult objects
    """
    results: List[CycleResult] = []
    ctx = TracedExperimentContext(trace_logger, runner.config.slice_name, runner.config.mode)
    
    for _ in range(cycles):
        ctx.begin_cycle(runner.cycle_index)
        result = runner.run_cycle(items, execute_fn)
        ctx.end_cycle(result.cycle_index)
        results.append(result)
    
    return results
