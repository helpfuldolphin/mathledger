"""
RFL Reward Computation

Compute reward signals based on proof success/failure.
Uses ONLY verifiable feedback (no human preferences, no proxy metrics).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RewardSignal:
    """
    Reward signal for a derivation attempt.
    
    Attributes:
        reward: Scalar reward value
        metadata: Optional metadata about the outcome
    """
    reward: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reward": self.reward,
            "metadata": self.metadata or {}
        }


def compute_reward(
    success: bool,
    abstained: bool = False,
    chain_length: int = 0,
    bonus_for_short_proof: bool = False
) -> RewardSignal:
    """
    Compute reward from derivation outcome.
    
    Reward scheme (verifiable feedback only):
    - Success: +1.0
    - Failure: -1.0
    - Abstention: 0.0
    
    Optional bonus for short proofs (if enabled):
    - Add -0.01 * chain_length to encourage shorter derivations
    
    Args:
        success: Whether proof succeeded
        abstained: Whether system abstained from attempting proof
        chain_length: Length of derivation chain (for optional bonus)
        bonus_for_short_proof: Whether to include chain length bonus
        
    Returns:
        RewardSignal with computed reward
    """
    if abstained:
        reward = 0.0
        outcome = "abstention"
    elif success:
        reward = 1.0
        outcome = "success"
        
        # Optional: Small penalty for long proofs
        if bonus_for_short_proof and chain_length > 0:
            reward -= 0.01 * chain_length
    else:
        reward = -1.0
        outcome = "failure"
    
    metadata = {
        "outcome": outcome,
        "chain_length": chain_length,
        "bonus_applied": bonus_for_short_proof
    }
    
    return RewardSignal(reward=reward, metadata=metadata)
