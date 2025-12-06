"""
U2 Planner Frontier Manager

Implements:
- Priority-based frontier queue
- Beam search allocation
- Pruning heuristics
- Deterministic candidate selection
"""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from rfl.prng import DeterministicPRNG


@dataclass(order=True)
class FrontierCandidate:
    """
    Candidate in the search frontier.
    
    INVARIANTS:
    - priority is comparable (lower = higher priority)
    - item is the actual candidate data
    - metadata is for debugging/telemetry
    """
    
    priority: float
    item: Any = field(compare=False)
    depth: int = field(default=0, compare=False)
    score: float = field(default=0.0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __hash__(self):
        """Hash based on item for deduplication."""
        return hash(str(self.item))


class FrontierManager:
    """
    Manages search frontier with beam allocation and pruning.
    
    DESIGN:
    - Priority queue for candidate selection
    - Per-depth tracking for depth-aware pruning
    - Deduplication to avoid redundant work
    - Deterministic tie-breaking using PRNG
    """
    
    def __init__(
        self,
        max_beam_width: int = 100,
        max_depth: int = 10,
        prng: Optional[DeterministicPRNG] = None,
    ):
        """
        Initialize frontier manager.
        
        Args:
            max_beam_width: Maximum candidates in frontier
            max_depth: Maximum search depth
            prng: Deterministic PRNG for tie-breaking
        """
        self.max_beam_width = max_beam_width
        self.max_depth = max_depth
        self.prng = prng or DeterministicPRNG(0)
        
        # Priority queue (min-heap)
        self.heap: List[FrontierCandidate] = []
        
        # Deduplication set
        self.seen: Set[str] = set()
        
        # Per-depth tracking
        self.depth_counts: Dict[int, int] = defaultdict(int)
        
        # Statistics
        self.stats = {
            "total_pushed": 0,
            "total_popped": 0,
            "duplicates_skipped": 0,
            "pruned_by_beam": 0,
            "pruned_by_depth": 0,
        }
    
    def push(
        self,
        item: Any,
        priority: float,
        depth: int = 0,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push candidate to frontier.
        
        Args:
            item: Candidate data
            priority: Priority (lower = higher priority)
            depth: Search depth
            score: Policy score (for telemetry)
            metadata: Additional metadata
            
        Returns:
            True if pushed, False if duplicate or pruned
        """
        # Check depth limit
        if depth > self.max_depth:
            self.stats["pruned_by_depth"] += 1
            return False
        
        # Deduplicate
        item_key = str(item)
        if item_key in self.seen:
            self.stats["duplicates_skipped"] += 1
            return False
        
        # Add deterministic tie-breaker to priority
        # This ensures same-priority items are selected deterministically
        tie_breaker = self.prng.random() * 1e-9
        adjusted_priority = priority + tie_breaker
        
        # Create candidate
        candidate = FrontierCandidate(
            priority=adjusted_priority,
            item=item,
            depth=depth,
            score=score,
            metadata=metadata or {},
        )
        
        # Push to heap
        heapq.heappush(self.heap, candidate)
        self.seen.add(item_key)
        self.depth_counts[depth] += 1
        self.stats["total_pushed"] += 1
        
        # Prune if beam width exceeded
        if len(self.heap) > self.max_beam_width:
            self._prune_lowest_priority()
        
        return True
    
    def pop(self) -> Optional[FrontierCandidate]:
        """
        Pop highest-priority candidate from frontier.
        
        Returns:
            FrontierCandidate or None if frontier empty
        """
        if not self.heap:
            return None
        
        candidate = heapq.heappop(self.heap)
        self.depth_counts[candidate.depth] -= 1
        self.stats["total_popped"] += 1
        
        return candidate
    
    def _prune_lowest_priority(self) -> None:
        """Prune lowest-priority candidate when beam width exceeded."""
        if not self.heap:
            return
        
        # Remove lowest priority (largest value in min-heap)
        # This is expensive (O(n)), but ensures beam width constraint
        worst_idx = max(range(len(self.heap)), key=lambda i: self.heap[i].priority)
        worst = self.heap[worst_idx]
        
        # Remove from heap
        self.heap[worst_idx] = self.heap[-1]
        self.heap.pop()
        if worst_idx < len(self.heap):
            heapq._siftup(self.heap, worst_idx)
            heapq._siftdown(self.heap, 0, worst_idx)
        
        # Update tracking
        self.seen.discard(str(worst.item))
        self.depth_counts[worst.depth] -= 1
        self.stats["pruned_by_beam"] += 1
    
    def size(self) -> int:
        """Get current frontier size."""
        return len(self.heap)
    
    def is_empty(self) -> bool:
        """Check if frontier is empty."""
        return len(self.heap) == 0
    
    def get_depth_distribution(self) -> Dict[int, int]:
        """Get distribution of candidates by depth."""
        return dict(self.depth_counts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frontier statistics."""
        return {
            **self.stats,
            "current_size": len(self.heap),
            "depth_distribution": self.get_depth_distribution(),
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get serializable state for snapshots.
        
        Returns:
            Dictionary with frontier state
        """
        return {
            "heap": [
                {
                    "priority": c.priority,
                    "item": c.item,
                    "depth": c.depth,
                    "score": c.score,
                    "metadata": c.metadata,
                }
                for c in self.heap
            ],
            "seen": list(self.seen),
            "depth_counts": dict(self.depth_counts),
            "stats": self.stats,
            "prng_state": self.prng.get_state(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore state from snapshot.
        
        Args:
            state: State dictionary from get_state()
        """
        # Restore heap
        self.heap = [
            FrontierCandidate(
                priority=c["priority"],
                item=c["item"],
                depth=c["depth"],
                score=c["score"],
                metadata=c["metadata"],
            )
            for c in state["heap"]
        ]
        heapq.heapify(self.heap)
        
        # Restore tracking
        self.seen = set(state["seen"])
        self.depth_counts = defaultdict(int, state["depth_counts"])
        self.stats = state["stats"]
        
        # Restore PRNG
        self.prng.set_state(state["prng_state"])


class BeamAllocator:
    """
    Allocates beam budget across search branches.
    
    DESIGN:
    - Dynamic allocation based on branch promise
    - Depth-aware allocation (favor shallow branches)
    - Budget tracking and exhaustion detection
    """
    
    def __init__(
        self,
        total_budget: int,
        depth_decay: float = 0.5,
    ):
        """
        Initialize beam allocator.
        
        Args:
            total_budget: Total beam budget
            depth_decay: Decay factor for deeper branches
        """
        self.total_budget = total_budget
        self.depth_decay = depth_decay
        
        self.remaining_budget = total_budget
        self.allocations: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "total_allocated": 0,
            "total_exhausted": 0,
            "per_depth_allocated": defaultdict(int),
        }
    
    def allocate(
        self,
        branch_id: str,
        depth: int,
        score: float,
    ) -> int:
        """
        Allocate beam budget to a branch.
        
        Args:
            branch_id: Branch identifier
            depth: Branch depth
            score: Branch promise score
            
        Returns:
            Allocated budget amount
        """
        if self.remaining_budget <= 0:
            return 0
        
        # Compute allocation based on depth and score
        # Higher score and shallower depth get more budget
        depth_factor = (1.0 - self.depth_decay) ** depth
        allocation = max(1, int(score * depth_factor * 10))
        
        # Cap by remaining budget
        allocation = min(allocation, self.remaining_budget)
        
        # Record allocation
        self.allocations[branch_id] = allocation
        self.remaining_budget -= allocation
        
        # Update stats
        self.stats["total_allocated"] += allocation
        self.stats["per_depth_allocated"][depth] += allocation
        
        return allocation
    
    def consume(self, branch_id: str, amount: int = 1) -> bool:
        """
        Consume budget from a branch allocation.
        
        Args:
            branch_id: Branch identifier
            amount: Amount to consume
            
        Returns:
            True if budget available, False if exhausted
        """
        if branch_id not in self.allocations:
            return False
        
        if self.allocations[branch_id] < amount:
            self.stats["total_exhausted"] += 1
            return False
        
        self.allocations[branch_id] -= amount
        return True
    
    def is_exhausted(self) -> bool:
        """Check if total budget is exhausted."""
        return self.remaining_budget <= 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        return {
            **self.stats,
            "remaining_budget": self.remaining_budget,
            "total_budget": self.total_budget,
            "utilization": 1.0 - (self.remaining_budget / self.total_budget),
        }
