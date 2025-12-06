"""
RFL Policy Scoring — PHASE II

Provides deterministic candidate scoring using feature vectors and policy weights.

NOTE: PHASE II — NOT USED IN PHASE I
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from substrate.repro.determinism import SeededRNG

from rfl.policy.features import (
    FeatureVector,
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_features,
    apply_feature_mask,
    get_feature_mask,
)


@dataclass
class ScoredCandidate:
    """A candidate with its computed score and features."""

    candidate: str
    score: float
    features: FeatureVector
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate": self.candidate,
            "score": self.score,
            "features": self.features.to_dict(),
            "rank": self.rank,
        }


class PolicyScorer:
    """
    Deterministic policy scorer for derivation candidates.

    Computes scores as weighted sum of masked features.
    All operations are deterministic given the seed.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
        slice_name: str = "default",
    ):
        """
        Initialize the policy scorer.

        Args:
            weights: Feature weights (default: uniform weights)
            seed: Random seed for deterministic tie-breaking
            slice_name: Slice name for feature masking
        """
        self.seed = seed
        self.slice_name = slice_name
        self.rng = SeededRNG(seed)

        # Initialize weights
        if weights is None:
            # Default: small positive weights for basic features
            self.weights = {
                "text_length": -0.01,    # Prefer shorter
                "ast_depth": 0.02,       # Slight preference for depth
                "atom_count": 0.01,
                "connective_count": 0.01,
                "chain_depth": 0.03,
                "dependency_count": 0.02,
                "goal_overlap": 0.1,
                "required_goal": 0.2,
                "per_goal_hit": 0.1,
                "success_count": 0.15,
                "attempt_count": -0.02,
                "success_rate": 0.1,
                "candidate_rarity": 0.05,
            }
        else:
            self.weights = weights.copy()

        # Get feature mask for this slice
        self.feature_mask = get_feature_mask(slice_name)

    def score_single(
        self,
        candidate: str,
        context: Optional[Dict[str, Any]] = None,
        success_history: Optional[Dict[str, int]] = None,
        attempt_history: Optional[Dict[str, int]] = None,
        total_candidates_seen: int = 1,
    ) -> ScoredCandidate:
        """
        Score a single candidate.

        Args:
            candidate: Candidate formula string
            context: Derivation context
            success_history: Success history dict
            attempt_history: Attempt history dict
            total_candidates_seen: Total unique candidates seen

        Returns:
            ScoredCandidate with score and features
        """
        # Extract features
        features = extract_features(
            candidate=candidate,
            context=context,
            success_history=success_history,
            attempt_history=attempt_history,
            total_candidates_seen=total_candidates_seen,
        )

        # Apply feature mask
        masked_features = apply_feature_mask(features, self.feature_mask)

        # Compute weighted score
        score = self._compute_score(masked_features)

        return ScoredCandidate(
            candidate=candidate,
            score=score,
            features=features,
        )

    def _compute_score(self, features: FeatureVector) -> float:
        """Compute weighted score from features."""
        score = 0.0
        feature_dict = features.to_dict()
        for name, value in feature_dict.items():
            weight = self.weights.get(name, 0.0)
            score += weight * value
        return score

    def score_batch(
        self,
        candidates: List[str],
        context: Optional[Dict[str, Any]] = None,
        success_history: Optional[Dict[str, int]] = None,
        attempt_history: Optional[Dict[str, int]] = None,
    ) -> List[ScoredCandidate]:
        """
        Score a batch of candidates and rank them.

        Args:
            candidates: List of candidate formula strings
            context: Shared derivation context
            success_history: Success history dict
            attempt_history: Attempt history dict

        Returns:
            List of ScoredCandidate sorted by score (descending)
        """
        total_candidates = len(candidates)

        # Score all candidates
        scored = []
        for candidate in candidates:
            sc = self.score_single(
                candidate=candidate,
                context=context,
                success_history=success_history,
                attempt_history=attempt_history,
                total_candidates_seen=total_candidates,
            )
            scored.append(sc)

        # Sort by score descending, with deterministic tie-breaking
        # Tie-breaker: candidate hash for determinism
        import hashlib
        scored.sort(
            key=lambda x: (
                -x.score,
                hashlib.sha256(x.candidate.encode()).hexdigest()
            )
        )

        # Assign ranks
        for i, sc in enumerate(scored):
            sc.rank = i + 1

        return scored

    def get_weights_dict(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights from dictionary."""
        self.weights = weights.copy()


def score_candidates(
    candidates: List[str],
    weights: Optional[Dict[str, float]] = None,
    slice_name: str = "default",
    context: Optional[Dict[str, Any]] = None,
    success_history: Optional[Dict[str, int]] = None,
    attempt_history: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> List[ScoredCandidate]:
    """
    Score and rank candidates using policy weights.

    Convenience function that creates a scorer and scores candidates.

    Args:
        candidates: List of candidate formula strings
        weights: Feature weights (default: uniform)
        slice_name: Slice name for feature masking
        context: Derivation context
        success_history: Success history dict
        attempt_history: Attempt history dict
        seed: Random seed for deterministic tie-breaking

    Returns:
        List of ScoredCandidate sorted by score (descending)
    """
    scorer = PolicyScorer(
        weights=weights,
        seed=seed,
        slice_name=slice_name,
    )
    return scorer.score_batch(
        candidates=candidates,
        context=context,
        success_history=success_history,
        attempt_history=attempt_history,
    )
