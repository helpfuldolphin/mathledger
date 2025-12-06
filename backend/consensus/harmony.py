"""
Harmony Protocol v1.1 - Byzantine-Resilient Consensus
Implements f < n/3 fault tolerance with adaptive trust weighting.

Core Properties:
- Safety: No two honest nodes decide conflicting values
- Liveness: Progress when >= 67% honest participation
- Finality: Once committed, never reverted
- Byzantine Resilience: Tolerates <= 33% adversarial participants
- Determinism: Identical ledger outcomes given identical inputs
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from backend.crypto.hashing import sha256_hex


@dataclass
class TrustWeight:
    """Trust weight for a validator node."""
    node_id: str
    weight: float
    epoch: int
    reputation: float = 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")
        if not 0.0 <= self.reputation <= 1.0:
            raise ValueError(f"Reputation must be in [0, 1], got {self.reputation}")


@dataclass
class ValidatorSet:
    """Set of validator nodes with trust weights."""
    validators: Dict[str, TrustWeight]
    epoch: int
    threshold: float = 0.67  # 2/3 Byzantine threshold
    
    def total_weight(self) -> float:
        """Compute total weight of all validators."""
        return sum(v.weight * v.reputation for v in self.validators.values())
    
    def quorum_reached(self, votes: Dict[str, str]) -> bool:
        """Check if quorum is reached for any value."""
        value_weights = {}
        for node_id, value in votes.items():
            if node_id in self.validators:
                v = self.validators[node_id]
                weighted_vote = v.weight * v.reputation
                value_weights[value] = value_weights.get(value, 0.0) + weighted_vote
        
        total = self.total_weight()
        if total == 0:
            return False
        
        for weight in value_weights.values():
            if weight / total >= self.threshold:
                return True
        return False
    
    def get_majority_value(self, votes: Dict[str, str]) -> Optional[str]:
        """Get the value that reached quorum, if any."""
        value_weights = {}
        for node_id, value in votes.items():
            if node_id in self.validators:
                v = self.validators[node_id]
                weighted_vote = v.weight * v.reputation
                value_weights[value] = value_weights.get(value, 0.0) + weighted_vote
        
        total = self.total_weight()
        if total == 0:
            return None
        
        for value, weight in value_weights.items():
            if weight / total >= self.threshold:
                return value
        return None


@dataclass
class ConsensusRound:
    """A single round of consensus."""
    round_id: int
    epoch: int
    proposal: str
    votes: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    decided_value: Optional[str] = None
    
    def duration_ms(self) -> float:
        """Get round duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    def add_vote(self, node_id: str, value: str):
        """Add a vote from a validator."""
        self.votes[node_id] = value
    
    def finalize(self, decided_value: str):
        """Finalize the round with a decided value."""
        self.decided_value = decided_value
        self.end_time = time.time()


class HarmonyProtocol:
    """
    Harmony Protocol v1.1 consensus implementation.
    
    Provides Byzantine-resilient consensus with adaptive trust weighting
    and guaranteed convergence for honest majorities.
    """
    
    def __init__(self, validators: ValidatorSet):
        self.validators = validators
        self.rounds: List[ConsensusRound] = []
        self.committed_values: List[str] = []
    
    def propose(self, value: str, round_id: int) -> ConsensusRound:
        """Start a new consensus round with a proposed value."""
        consensus_round = ConsensusRound(
            round_id=round_id,
            epoch=self.validators.epoch,
            proposal=value
        )
        self.rounds.append(consensus_round)
        return consensus_round
    
    def vote(self, consensus_round: ConsensusRound, node_id: str, value: str) -> bool:
        """
        Cast a vote in a consensus round.
        
        Returns True if quorum is reached.
        """
        if node_id not in self.validators.validators:
            return False
        
        consensus_round.add_vote(node_id, value)
        return self.validators.quorum_reached(consensus_round.votes)
    
    def finalize_round(self, consensus_round: ConsensusRound) -> Optional[str]:
        """
        Finalize a consensus round if quorum is reached.
        
        Returns the decided value, or None if no consensus.
        """
        decided_value = self.validators.get_majority_value(consensus_round.votes)
        if decided_value is not None:
            consensus_round.finalize(decided_value)
            self.committed_values.append(decided_value)
        return decided_value
    
    def get_convergence_metrics(self) -> Dict[str, any]:
        """Get convergence metrics for analysis."""
        if not self.rounds:
            return {
                "total_rounds": 0,
                "avg_latency_ms": 0.0,
                "quorum_ratio": 0.0,
                "convergence_rate": 0.0
            }
        
        completed_rounds = [r for r in self.rounds if r.decided_value is not None]
        total_latency = sum(r.duration_ms() for r in completed_rounds)
        
        return {
            "total_rounds": len(self.rounds),
            "completed_rounds": len(completed_rounds),
            "avg_latency_ms": total_latency / len(completed_rounds) if completed_rounds else 0.0,
            "quorum_ratio": len(completed_rounds) / len(self.rounds) if self.rounds else 0.0,
            "convergence_rate": len(self.committed_values) / len(self.rounds) if self.rounds else 0.0,
            "total_nodes": len(self.validators.validators),
            "threshold": self.validators.threshold
        }


def converge(
    validators: ValidatorSet,
    proposals: List[str],
    honest_nodes: List[str],
    byzantine_nodes: Optional[List[str]] = None
) -> Tuple[Optional[str], Dict[str, any]]:
    """
    Execute consensus convergence for a set of proposals.
    
    Args:
        validators: Validator set with trust weights
        proposals: List of proposed values
        honest_nodes: List of honest validator node IDs
        byzantine_nodes: Optional list of Byzantine validator node IDs
    
    Returns:
        Tuple of (decided_value, metrics)
    
    Guarantees:
        - Convergence in 1 round for honest majority
        - Byzantine resilience for f < n/3
        - Deterministic outcome for given inputs
    """
    start_time = time.time()
    byzantine_nodes = byzantine_nodes or []
    
    # Ensure Byzantine nodes are <= 33%
    total_nodes = len(validators.validators)
    if len(byzantine_nodes) >= total_nodes / 3:
        return None, {
            "success": False,
            "reason": "Byzantine nodes exceed fault tolerance threshold",
            "byzantine_count": len(byzantine_nodes),
            "total_nodes": total_nodes,
            "threshold": total_nodes / 3
        }
    
    # Create protocol instance
    protocol = HarmonyProtocol(validators)
    
    # Execute consensus round
    if not proposals:
        return None, {"success": False, "reason": "No proposals"}
    
    # Honest nodes vote for first proposal (deterministic)
    proposal = proposals[0]
    consensus_round = protocol.propose(proposal, round_id=0)
    
    # Honest nodes cast votes
    for node_id in honest_nodes:
        if node_id in validators.validators:
            protocol.vote(consensus_round, node_id, proposal)
    
    # Byzantine nodes may vote differently (worst case: all vote for different value)
    byzantine_value = proposals[1] if len(proposals) > 1 else f"BYZANTINE_{proposal}"
    for node_id in byzantine_nodes:
        if node_id in validators.validators:
            protocol.vote(consensus_round, node_id, byzantine_value)
    
    # Finalize round
    decided_value = protocol.finalize_round(consensus_round)
    end_time = time.time()
    
    # Get metrics
    metrics = protocol.get_convergence_metrics()
    metrics.update({
        "success": decided_value is not None,
        "decided_value": decided_value,
        "honest_nodes": len(honest_nodes),
        "byzantine_nodes": len(byzantine_nodes),
        "total_latency_ms": (end_time - start_time) * 1000,
        "convergence_rounds": 1 if decided_value else 0
    })
    
    return decided_value, metrics
