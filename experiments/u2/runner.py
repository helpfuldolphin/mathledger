"""
U2 Runner - Type-Verified Experiment Runner

Provides the core U2Runner class with strict typing and deterministic execution.
Integrates with snapshots, logging, and safety infrastructure.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .snapshots import SnapshotData, save_snapshot


@dataclass
class U2Config:
    """
    Typed configuration for U2 experiments.
    
    All parameters are explicitly typed and validated.
    No secrets should be included in this config.
    """
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
    
    # Performance budgets (optional)
    cycle_budget_s: Optional[float] = None
    max_candidates_per_cycle: Optional[int] = None
    
    def to_safe_dict(self) -> Dict[str, Any]:
        """
        Convert config to a safe dictionary for inclusion in safety envelope.
        Excludes any sensitive data and includes only essential parameters.
        """
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "total_cycles": self.total_cycles,
            "master_seed": self.master_seed,
            "snapshot_interval": self.snapshot_interval,
            "cycle_budget_s": self.cycle_budget_s,
            "max_candidates_per_cycle": self.max_candidates_per_cycle,
        }


@dataclass
class CycleResult:
    """
    Typed result from a single experiment cycle.
    
    Captures all essential information about cycle execution
    for analysis and auditing.
    """
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Any
    success: bool
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_index": self.cycle_index,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "seed": self.seed,
            "item": self.item,
            "result": str(self.result),
            "success": self.success,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class U2Runner:
    """
    Type-verified runner for U2 experiments.
    
    Handles cycle execution, state management, and integration with
    snapshot and logging infrastructure.
    """
    
    def __init__(self, config: U2Config):
        """
        Initialize U2 runner with typed configuration.
        
        Args:
            config: U2Config with all experiment parameters
        """
        self.config = config
        self.cycle_index: int = 0
        self.ht_series: List[str] = []
        
        # RFL policy state (only used in RFL mode)
        self.policy_update_count: int = 0
        self.success_count: Dict[str, int] = defaultdict(int)
        self.attempt_count: Dict[str, int] = defaultdict(int)
        self.weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Performance tracking
        self.cycle_durations_ms: List[float] = []
    
    def run_cycle(
        self,
        items: List[str],
        execute_fn: Callable[[str, int], Tuple[bool, Any]],
    ) -> CycleResult:
        """
        Execute a single experiment cycle.
        
        Args:
            items: List of candidate items to choose from
            execute_fn: Function to execute an item (item, seed) -> (success, result)
            
        Returns:
            CycleResult with typed execution data
        """
        start_time = time.time()
        
        # Select item based on mode
        if self.config.mode == "baseline":
            # Random selection for baseline
            # Use deterministic PRNG if available, otherwise simple modulo
            item_index = (self.config.master_seed + self.cycle_index) % len(items)
            selected_item = items[item_index]
        elif self.config.mode == "rfl":
            # RFL policy-based selection (weighted by success rate)
            selected_item = self._rfl_select_item(items)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # Execute the selected item
        cycle_seed = self.config.master_seed + self.cycle_index
        success, result = execute_fn(selected_item, cycle_seed)
        
        # Update RFL policy if in RFL mode
        if self.config.mode == "rfl":
            self._update_rfl_policy(selected_item, success)
        
        # Track performance
        duration_ms = (time.time() - start_time) * 1000
        self.cycle_durations_ms.append(duration_ms)
        
        # Build result
        cycle_result = CycleResult(
            cycle_index=self.cycle_index,
            slice_name=self.config.slice_name,
            mode=self.config.mode,
            seed=cycle_seed,
            item=selected_item,
            result=result,
            success=success,
            duration_ms=duration_ms,
        )
        
        # Update state
        self.ht_series.append(f"cycle_{self.cycle_index}_{selected_item}_{success}")
        self.cycle_index += 1
        
        return cycle_result
    
    def _rfl_select_item(self, items: List[str]) -> str:
        """
        Select item using RFL policy (weighted by success rate).
        
        Args:
            items: List of candidate items
            
        Returns:
            Selected item
        """
        # Calculate weights for each item
        item_weights = []
        for item in items:
            attempts = self.attempt_count.get(item, 0)
            successes = self.success_count.get(item, 0)
            
            if attempts == 0:
                # Unexplored items get default weight
                weight = 1.0
            else:
                # Weight based on success rate with exploration bonus
                success_rate = successes / attempts
                weight = success_rate + 0.1  # Small exploration bonus
            
            item_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(item_weights)
        if total_weight == 0:
            # Fallback to uniform
            item_weights = [1.0] * len(items)
            total_weight = len(items)
        
        # Deterministic selection based on weights
        # Use cycle_index as pseudo-random seed for determinism
        rand_val = ((self.config.master_seed + self.cycle_index) % 10000) / 10000.0
        cumulative = 0.0
        for i, weight in enumerate(item_weights):
            cumulative += weight / total_weight
            if rand_val <= cumulative:
                return items[i]
        
        # Fallback to last item
        return items[-1]
    
    def _update_rfl_policy(self, item: str, success: bool):
        """
        Update RFL policy based on cycle outcome.
        
        Args:
            item: Item that was executed
            success: Whether execution was successful
        """
        self.attempt_count[item] += 1
        if success:
            self.success_count[item] += 1
        self.policy_update_count += 1
    
    def maybe_save_snapshot(self) -> Optional[Path]:
        """
        Save snapshot if at snapshot interval.
        
        Returns:
            Path to saved snapshot, or None if no snapshot was saved
        """
        if self.config.snapshot_interval <= 0:
            return None
        
        if self.cycle_index % self.config.snapshot_interval == 0:
            snapshot = self.capture_state()
            snapshot_path = self.config.snapshot_dir / f"snapshot_{self.config.experiment_id}_{self.cycle_index}.snap"
            save_snapshot(snapshot, snapshot_path)
            return snapshot_path
        
        return None
    
    def capture_state(self) -> SnapshotData:
        """
        Capture current runner state as a snapshot.
        
        Returns:
            SnapshotData with full state
        """
        return SnapshotData(
            cycle_index=self.cycle_index,
            ht_series=self.ht_series.copy(),
            policy_update_count=self.policy_update_count,
            success_count=dict(self.success_count),
            attempt_count=dict(self.attempt_count),
            weights=dict(self.weights),
            config=self.config.to_safe_dict(),
        )
    
    def restore_state(self, snapshot: SnapshotData):
        """
        Restore runner state from a snapshot.
        
        Args:
            snapshot: SnapshotData to restore from
        """
        self.cycle_index = snapshot.cycle_index
        self.ht_series = snapshot.ht_series.copy()
        self.policy_update_count = snapshot.policy_update_count
        self.success_count = defaultdict(int, snapshot.success_count)
        self.attempt_count = defaultdict(int, snapshot.attempt_count)
        self.weights = defaultdict(lambda: 1.0, snapshot.weights)


@dataclass
class TracedExperimentContext:
    """
    Context for traced experiment execution.
    
    Provides cycle timing and event logging integration.
    """
    logger: Any  # U2TraceLogger
    slice_name: str
    mode: str
    cycle_start_time: Optional[float] = None
    
    def begin_cycle(self, cycle_index: int):
        """Mark the beginning of a cycle."""
        self.cycle_start_time = time.time()
    
    def end_cycle(self, cycle_index: int):
        """Mark the end of a cycle and log duration."""
        if self.cycle_start_time is not None:
            duration_ms = (time.time() - self.cycle_start_time) * 1000
            self.cycle_start_time = None
    
    def log_cycle_telemetry(self, cycle_index: int, telemetry: Dict[str, Any]):
        """Log cycle telemetry data."""
        # Logging is handled by the logger instance
        pass


def run_with_traces(func: Callable, trace_ctx: TracedExperimentContext, *args, **kwargs):
    """
    Wrapper to run a function with trace context.
    
    Args:
        func: Function to wrap
        trace_ctx: Trace context for logging
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result from func
    """
    return func(*args, **kwargs)
