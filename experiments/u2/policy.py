"""
U2 Planner Search Policies

Implements:
- Baseline (random) policy
- RFL (feedback-driven) policy
- Policy-based candidate ranking
- Deterministic policy evaluation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from rfl.prng import DeterministicPRNG


class SearchPolicy(ABC):
    """
    Abstract base class for search policies.
    
    INVARIANTS:
    - rank() must be deterministic given same PRNG seed
    - Higher scores indicate higher priority
    """
    
    def __init__(self, prng: DeterministicPRNG):
        """
        Initialize policy with PRNG.
        
        Args:
            prng: Deterministic PRNG for policy decisions
        """
        self.prng = prng
    
    @abstractmethod
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates by priority.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        pass
    
    @abstractmethod
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert score to priority (lower = higher priority).
        
        Args:
            candidate: Candidate item
            score: Policy score
            
        Returns:
            Priority value for frontier queue
        """
        pass


class BaselinePolicy(SearchPolicy):
    """
    Baseline random policy.
    
    Assigns random scores to candidates for unbiased exploration.
    Used as control in RFL experiments.
    """
    
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates randomly.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples with random scores
        """
        scored = [(c, self.prng.random()) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert score to priority.
        
        For baseline, priority is just inverse of score.
        
        Args:
            candidate: Candidate item
            score: Random score in [0, 1]
            
        Returns:
            Priority (lower = higher priority)
        """
        return 1.0 - score


class RFLPolicy(SearchPolicy):
    """
    RFL (Reflexive Formal Learning) policy.
    
    Uses verifiable feedback to guide search:
    - Success rate of similar candidates
    - Depth-based heuristics
    - Structural features
    
    IMPORTANT: Only uses verifiable feedback (no RLHF, no preferences)
    """
    
    def __init__(
        self,
        prng: DeterministicPRNG,
        feedback_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RFL policy.
        
        Args:
            prng: Deterministic PRNG
            feedback_data: Verifiable feedback from previous runs
        """
        super().__init__(prng)
        self.feedback_data = feedback_data or {}
        
        # Feature weights (learned from verifiable feedback)
        self.weights = {
            "depth": -0.3,  # Prefer shallower
            "complexity": -0.2,  # Prefer simpler
            "success_rate": 0.5,  # Prefer high success rate
        }
    
    def rank(self, candidates: List[Any]) -> List[Tuple[Any, float]]:
        """
        Rank candidates using RFL policy.
        
        Args:
            candidates: List of candidate items
            
        Returns:
            List of (candidate, score) tuples sorted by RFL score
        """
        scored = []
        
        for candidate in candidates:
            # Extract features
            features = self._extract_features(candidate)
            
            # Compute score
            score = self._compute_score(features)
            
            scored.append((candidate, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def get_priority(self, candidate: Any, score: float) -> float:
        """
        Convert RFL score to priority.
        
        Args:
            candidate: Candidate item
            score: RFL score
            
        Returns:
            Priority (lower = higher priority)
        """
        # Invert score to get priority
        # Add small noise for tie-breaking
        noise = self.prng.random() * 1e-6
        return -score + noise
    
    def _extract_features(self, candidate: Any) -> Dict[str, float]:
        """
        Extract features from candidate.
        
        Args:
            candidate: Candidate item
            
        Returns:
            Feature dictionary
        """
        # Default features
        features = {
            "depth": 0.0,
            "complexity": 0.0,
            "success_rate": 0.5,  # Default to neutral
        }
        
        # Extract from candidate if it's a dict
        if isinstance(candidate, dict):
            features["depth"] = float(candidate.get("depth", 0))
            features["complexity"] = float(len(str(candidate)))
            
            # Look up success rate from feedback
            candidate_key = str(candidate.get("item", candidate))
            if candidate_key in self.feedback_data:
                features["success_rate"] = self.feedback_data[candidate_key].get("success_rate", 0.5)
        else:
            # Simple heuristics for non-dict candidates
            features["complexity"] = float(len(str(candidate)))
        
        return features
    
    def _compute_score(self, features: Dict[str, float]) -> float:
        """
        Compute RFL score from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            RFL score (higher = better)
        """
        score = 0.0
        
        for feature_name, feature_value in features.items():
            weight = self.weights.get(feature_name, 0.0)
            score += weight * feature_value
        
        # Normalize to [0, 1]
        score = 1.0 / (1.0 + abs(score))
        
        return score


def create_policy(
    mode: str,
    prng: DeterministicPRNG,
    feedback_data: Optional[Dict[str, Any]] = None,
) -> SearchPolicy:
    """
    Factory function to create search policy.
    
    Args:
        mode: "baseline" or "rfl"
        prng: Deterministic PRNG
        feedback_data: Feedback data for RFL policy
        
    Returns:
        SearchPolicy instance
    """
    if mode == "baseline":
        return BaselinePolicy(prng)
    elif mode == "rfl":
        return RFLPolicy(prng, feedback_data)
    else:
        raise ValueError(f"Unknown policy mode: {mode}")
