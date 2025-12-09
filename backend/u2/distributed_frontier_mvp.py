"""
Distributed Frontier Manager MVP

This module provides an MVP implementation of the distributed frontier manager.
For simplicity, this MVP uses an in-memory priority queue instead of Redis.

For production, replace InMemoryFrontier with RedisFrontier.

Author: Manus-F
Date: 2025-12-06
Status: Phase V MVP Implementation
"""

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.u2.fosubstrate_executor_skeleton import CandidateItem


# ============================================================================
# FRONTIER INTERFACE
# ============================================================================

class FrontierInterface:
    """Abstract interface for frontier implementations."""
    
    def push(self, candidate: CandidateItem, priority: float):
        """Push candidate to frontier with priority."""
        raise NotImplementedError
    
    def pop(self) -> Optional[CandidateItem]:
        """Pop highest priority candidate from frontier."""
        raise NotImplementedError
    
    def size(self) -> int:
        """Return number of candidates in frontier."""
        raise NotImplementedError
    
    def clear(self):
        """Clear all candidates from frontier."""
        raise NotImplementedError


# ============================================================================
# IN-MEMORY FRONTIER (MVP)
# ============================================================================

class InMemoryFrontier(FrontierInterface):
    """
    In-memory priority queue implementation of frontier.
    
    Uses Python's heapq for deterministic priority ordering.
    Tie-breaking is done using candidate hash for determinism.
    """
    
    def __init__(self):
        """Initialize empty frontier."""
        self.heap = []  # Min-heap of (-priority, tie_break, candidate_json)
        self.seen_hashes = set()  # Deduplicate candidates
    
    def push(self, candidate: CandidateItem, priority: float):
        """
        Push candidate to frontier with priority.
        
        Args:
            candidate: Candidate to push
            priority: Priority (higher = more important)
        """
        candidate_hash = candidate.statement.hash
        
        # Deduplicate
        if candidate_hash in self.seen_hashes:
            return
        
        self.seen_hashes.add(candidate_hash)
        
        # Compute tie-break value from hash (deterministic)
        tie_break = self._hash_to_tie_break(candidate_hash)
        
        # Serialize candidate
        candidate_json = json.dumps(
            candidate.to_canonical_dict(),
            sort_keys=True,
            separators=(",", ":")
        )
        
        # Push to heap (negate priority for max-heap behavior)
        heapq.heappush(self.heap, (-priority, tie_break, candidate_json))
    
    def pop(self) -> Optional[CandidateItem]:
        """
        Pop highest priority candidate from frontier.
        
        Returns:
            CandidateItem or None if frontier is empty
        """
        if not self.heap:
            return None
        
        _, _, candidate_json = heapq.heappop(self.heap)
        
        # Deserialize candidate
        candidate_dict = json.loads(candidate_json)
        return self._dict_to_candidate(candidate_dict)
    
    def size(self) -> int:
        """Return number of candidates in frontier."""
        return len(self.heap)
    
    def clear(self):
        """Clear all candidates from frontier."""
        self.heap.clear()
        self.seen_hashes.clear()
    
    def _hash_to_tie_break(self, candidate_hash: str) -> float:
        """
        Convert candidate hash to tie-break value.
        
        This ensures deterministic ordering for identical priorities.
        
        Args:
            candidate_hash: SHA-256 hash (hex string)
            
        Returns:
            Tie-break value in [0, 1)
        """
        # Convert first 16 hex chars to integer
        hash_int = int(candidate_hash[:16], 16)
        # Normalize to [0, 1)
        return hash_int / (16 ** 16)
    
    def _dict_to_candidate(self, d: Dict[str, Any]) -> CandidateItem:
        """Reconstruct CandidateItem from dictionary."""
        from backend.u2.fosubstrate_executor_skeleton import StatementRecord
        
        statement_dict = d["statement"]
        statement = StatementRecord(
            normalized=statement_dict["normalized"],
            hash=statement_dict["hash"],
            pretty=statement_dict.get("pretty", statement_dict["normalized"]),
            rule=statement_dict["rule"],
            is_axiom=statement_dict["is_axiom"],
            mp_depth=statement_dict["mp_depth"],
            parents=tuple(statement_dict["parents"]),
            verification_method=statement_dict["verification_method"],
        )
        
        return CandidateItem(
            statement=statement,
            depth=d["depth"],
            priority=0.0,  # Priority not stored in dict
            parent_hashes=tuple(d["parent_hashes"]),
            generation_cycle=d["generation_cycle"],
            generation_seed=d["generation_seed"],
        )


# ============================================================================
# WORKER HARNESS
# ============================================================================

@dataclass
class WorkerConfig:
    """Configuration for a worker."""
    worker_id: int
    experiment_id: str
    slice_name: str
    total_cycles: int
    master_seed: str


class Worker:
    """
    Worker that executes candidates from the frontier.
    
    In distributed mode, each worker would run on a separate machine.
    For MVP, workers run sequentially in the same process.
    """
    
    def __init__(
        self,
        config: WorkerConfig,
        frontier: FrontierInterface,
        executor,  # FOSubstrateExecutor
    ):
        """
        Initialize worker.
        
        Args:
            config: Worker configuration
            frontier: Shared frontier
            executor: FOSubstrateExecutor instance
        """
        self.config = config
        self.frontier = frontier
        self.executor = executor
        
        # Local trace buffer
        self.trace_events = []
    
    def run_cycle(self, cycle: int, cycle_budget_ms: int) -> Dict[str, Any]:
        """
        Run a single cycle.
        
        Args:
            cycle: Cycle number
            cycle_budget_ms: Time budget for cycle
            
        Returns:
            Cycle statistics
        """
        import time
        
        cycle_start_time_ms = time.time_ns() // 1_000_000
        executions = 0
        successes = 0
        failures = 0
        
        # Execute candidates until budget exhausted or frontier empty
        while True:
            # Check budget
            elapsed_ms = (time.time_ns() // 1_000_000) - cycle_start_time_ms
            if elapsed_ms >= cycle_budget_ms:
                break
            
            # Pop candidate
            candidate = self.frontier.pop()
            if candidate is None:
                break  # Frontier empty
            
            # Generate execution seed
            seed = self._generate_execution_seed(cycle, executions)
            
            # Execute
            success, result = self.executor.execute(
                candidate, seed, cycle_start_time_ms
            )
            
            executions += 1
            if success:
                successes += 1
            else:
                failures += 1
            
            # Log to trace
            self._log_execution(cycle, candidate, result)
            
            # Push new candidates from derivation
            for new_stmt in result.new_statements:
                new_candidate = self._create_candidate(
                    new_stmt, candidate, cycle
                )
                self.frontier.push(new_candidate, priority=1.0)
        
        return {
            "cycle": cycle,
            "worker_id": self.config.worker_id,
            "executions": executions,
            "successes": successes,
            "failures": failures,
            "frontier_size": self.frontier.size(),
        }
    
    def _generate_execution_seed(self, cycle: int, execution_idx: int) -> int:
        """Generate deterministic execution seed."""
        seed_str = f"{self.config.master_seed}:cycle{cycle}:exec{execution_idx}"
        seed_hash = hashlib.sha256(seed_str.encode()).hexdigest()
        return int(seed_hash[:16], 16)
    
    def _log_execution(
        self,
        cycle: int,
        candidate: CandidateItem,
        result,  # ExecutionResult
    ):
        """Log execution to trace buffer."""
        import time
        
        event = {
            "event_type": "execution_result",
            "cycle": cycle,
            "worker_id": self.config.worker_id,
            "timestamp_ms": time.time_ns() // 1_000_000,
            "data": {
                "candidate_hash": candidate.statement.hash,
                "candidate": candidate.to_canonical_dict(),
                "result": result.to_canonical_dict(),
            },
        }
        self.trace_events.append(event)
    
    def _create_candidate(
        self,
        statement,  # StatementRecord
        parent_candidate: CandidateItem,
        cycle: int,
    ) -> CandidateItem:
        """Create new candidate from derived statement."""
        return CandidateItem(
            statement=statement,
            depth=parent_candidate.depth + 1,
            priority=1.0,  # Default priority
            parent_hashes=(parent_candidate.statement.hash,),
            generation_cycle=cycle,
            generation_seed=parent_candidate.generation_seed,
        )
    
    def save_trace(self, output_path: Path):
        """Save trace to JSONL file."""
        with open(output_path, 'w') as f:
            for event in self.trace_events:
                f.write(json.dumps(event, sort_keys=True) + '\n')


# ============================================================================
# SINGLE-NODE DETERMINISM VALIDATION HARNESS
# ============================================================================

class DeterminismValidator:
    """
    Validates that execution is deterministic across runs.
    """
    
    def validate(
        self,
        trace1_path: Path,
        trace2_path: Path,
    ) -> Tuple[bool, str]:
        """
        Validate that two traces are identical.
        
        Args:
            trace1_path: Path to first trace
            trace2_path: Path to second trace
            
        Returns:
            Tuple of (is_deterministic, message)
        """
        # Compute trace hashes
        hash1 = self._compute_trace_hash(trace1_path)
        hash2 = self._compute_trace_hash(trace2_path)
        
        if hash1 == hash2:
            return True, f"✓ Traces match: {hash1}"
        else:
            return False, f"✗ Traces differ:\n  Run 1: {hash1}\n  Run 2: {hash2}"
    
    def _compute_trace_hash(self, trace_path: Path) -> str:
        """Compute SHA-256 hash of trace file."""
        hasher = hashlib.sha256()
        
        with open(trace_path, 'r') as f:
            for line in f:
                # Parse event
                event = json.loads(line)
                
                # Remove non-deterministic fields
                if "timestamp_ms" in event:
                    del event["timestamp_ms"]
                if "data" in event and "result" in event["data"]:
                    if "timestamp_ms" in event["data"]["result"]:
                        del event["data"]["result"]["timestamp_ms"]
                
                # Canonical serialization
                canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
                hasher.update(canonical.encode())
        
        return hasher.hexdigest()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_single_node_experiment():
    """
    Run a single-node experiment for determinism validation.
    """
    from backend.u2.fosubstrate_executor import FOSubstrateExecutor
    from backend.u2.fosubstrate_executor_skeleton import StatementRecord
    
    # Initialize frontier
    frontier = InMemoryFrontier()
    
    # Initialize executor
    executor = FOSubstrateExecutor(
        slice_name="test_slice",
        budget_config=None,  # Use defaults
    )
    
    # Seed frontier with initial statements
    seed_statement = StatementRecord(
        normalized="p→p",
        hash=hashlib.sha256("p→p".encode()).hexdigest(),
        pretty="p → p",
        rule="axiom",
        is_axiom=True,
        mp_depth=0,
        parents=(),
        verification_method="axiom",
    )
    
    seed_candidate = CandidateItem(
        statement=seed_statement,
        depth=0,
        priority=1.0,
        parent_hashes=(),
        generation_cycle=0,
        generation_seed="0xmaster",
    )
    
    frontier.push(seed_candidate, priority=1.0)
    
    # Initialize worker
    config = WorkerConfig(
        worker_id=0,
        experiment_id="test_exp",
        slice_name="test_slice",
        total_cycles=5,
        master_seed="0xmaster",
    )
    
    worker = Worker(config, frontier, executor)
    
    # Run cycles
    for cycle in range(config.total_cycles):
        stats = worker.run_cycle(cycle, cycle_budget_ms=10000)
        print(f"Cycle {cycle}: {stats}")
    
    # Save trace
    trace_path = Path("/tmp/test_trace.jsonl")
    worker.save_trace(trace_path)
    print(f"Trace saved to {trace_path}")
    
    return trace_path


if __name__ == "__main__":
    print("Running single-node experiment...")
    trace_path = run_single_node_experiment()
    print(f"\n✓ Experiment complete. Trace: {trace_path}")
