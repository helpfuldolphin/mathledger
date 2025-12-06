"""
Policy Scoring for RFL
======================

Deterministic candidate scoring using weighted feature vectors.

Scoring uses:
- Weighted dot-product of feature vector and weight vector
- Deterministic tie-breaking (lexicographic by formula)
- Optional UCB-style exploration term

All randomness uses SeededRNG for reproducibility.

Usage:
    from rfl.policy.scoring import PolicyScorer, score_candidates

    scorer = PolicyScorer(seed=42)
    ranked = scorer.score_and_rank(candidates, feature_vectors)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from substrate.repro.determinism import SeededRNG

from .features import FeatureVector


@dataclass
class PolicyWeights:
    """
    Weight vector for policy scoring.

    Each weight corresponds to a feature in FeatureVector.
    Positive weights increase score for larger feature values.
    """
    len: float = 0.0
    depth: float = 0.0
    connectives: float = 0.0
    negations: float = 0.0
    overlap: float = 0.0
    goal_flag: float = 0.0
    success_count: float = 0.0
    chain_depth: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "len": self.len,
            "depth": self.depth,
            "connectives": self.connectives,
            "negations": self.negations,
            "overlap": self.overlap,
            "goal_flag": self.goal_flag,
            "success_count": self.success_count,
            "chain_depth": self.chain_depth,
        }

    def to_list(self) -> List[float]:
        """Convert to ordered list for dot-product computation."""
        return [
            self.len,
            self.depth,
            self.connectives,
            self.negations,
            self.overlap,
            self.goal_flag,
            self.success_count,
            self.chain_depth,
        ]

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PolicyWeights":
        """Create from dictionary."""
        return cls(
            len=d.get("len", 0.0),
            depth=d.get("depth", 0.0),
            connectives=d.get("connectives", 0.0),
            negations=d.get("negations", 0.0),
            overlap=d.get("overlap", 0.0),
            goal_flag=d.get("goal_flag", 0.0),
            success_count=d.get("success_count", 0.0),
            chain_depth=d.get("chain_depth", 0.0),
        )

    @classmethod
    def default(cls) -> "PolicyWeights":
        """
        Return default weights that prioritize:
        - Shorter formulas (negative len weight)
        - Higher overlap with target
        - Goal formulas
        - Historical success
        """
        return cls(
            len=-0.01,          # Prefer shorter formulas
            depth=0.0,          # Neutral on depth
            connectives=0.0,    # Neutral on connectives
            negations=-0.005,   # Slight preference for fewer negations
            overlap=0.1,        # Prefer formulas overlapping with target
            goal_flag=1.0,      # Strongly prefer goal formulas
            success_count=0.05, # Prefer formulas with success history
            chain_depth=0.0,    # Neutral on chain depth
        )


def dot_product(weights: PolicyWeights, features: FeatureVector) -> float:
    """Compute dot product of weight and feature vectors."""
    w = weights.to_list()
    f = features.to_list()
    return sum(wi * fi for wi, fi in zip(w, f))


def _deterministic_tiebreaker(formula: str) -> str:
    """
    Generate deterministic tiebreaker key for a formula.

    Uses SHA256 hash prefix to ensure consistent ordering.
    """
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScoredCandidate:
    """A candidate formula with its computed score and ranking metadata."""
    formula: str
    features: FeatureVector
    base_score: float
    exploration_bonus: float
    total_score: float
    tiebreaker: str

    def __lt__(self, other: "ScoredCandidate") -> bool:
        """Order by total_score desc, then by tiebreaker asc."""
        if self.total_score != other.total_score:
            return self.total_score > other.total_score  # Higher score first
        return self.tiebreaker < other.tiebreaker


class PolicyScorer:
    """
    Deterministic policy scorer for RFL candidate selection.

    Supports:
    - Weighted dot-product scoring
    - Deterministic tie-breaking
    - Optional UCB-style exploration term
    """

    def __init__(
        self,
        weights: Optional[PolicyWeights] = None,
        seed: int = 42,
        exploration_weight: float = 0.0,
        total_selections: int = 0,
    ):
        """
        Initialize policy scorer.

        Args:
            weights: Policy weight vector. If None, uses default weights.
            seed: Random seed for deterministic behavior.
            exploration_weight: Coefficient for UCB exploration term.
                              If 0, no exploration bonus is added.
            total_selections: Total number of selections made (for UCB).
        """
        self.weights = weights or PolicyWeights.default()
        self.seed = seed
        self.rng = SeededRNG(seed)
        self.exploration_weight = exploration_weight
        self.total_selections = total_selections

        # Track selection counts for UCB
        self.selection_counts: Dict[str, int] = {}

    def score(
        self,
        formula: str,
        features: FeatureVector,
        formula_selection_count: int = 0,
    ) -> ScoredCandidate:
        """
        Score a single candidate formula.

        Args:
            formula: The candidate formula string.
            features: Pre-computed feature vector for the formula.
            formula_selection_count: How many times this formula has been selected.

        Returns:
            ScoredCandidate with base score, exploration bonus, and total score.
        """
        base_score = dot_product(self.weights, features)

        # UCB exploration bonus: c * sqrt(ln(N) / n)
        # where N = total selections, n = selections of this formula
        exploration_bonus = 0.0
        if self.exploration_weight > 0 and self.total_selections > 0:
            n = max(1, formula_selection_count)  # Avoid division by zero
            exploration_bonus = self.exploration_weight * math.sqrt(
                math.log(self.total_selections + 1) / n
            )

        total_score = base_score + exploration_bonus
        tiebreaker = _deterministic_tiebreaker(formula)

        return ScoredCandidate(
            formula=formula,
            features=features,
            base_score=base_score,
            exploration_bonus=exploration_bonus,
            total_score=total_score,
            tiebreaker=tiebreaker,
        )

    def score_and_rank(
        self,
        formulas: Sequence[str],
        features: Sequence[FeatureVector],
        selection_counts: Optional[Dict[str, int]] = None,
    ) -> List[ScoredCandidate]:
        """
        Score and rank a batch of candidate formulas.

        Args:
            formulas: Sequence of candidate formula strings.
            features: Corresponding feature vectors.
            selection_counts: Optional mapping of formula -> selection count for UCB.

        Returns:
            List of ScoredCandidates sorted by total score (descending),
            with deterministic tie-breaking.
        """
        if len(formulas) != len(features):
            raise ValueError("formulas and features must have same length")

        selection_counts = selection_counts or {}

        candidates = []
        for formula, fv in zip(formulas, features):
            count = selection_counts.get(formula, 0)
            scored = self.score(formula, fv, count)
            candidates.append(scored)

        # Sort: higher score first, then deterministic tiebreaker
        candidates.sort()
        return candidates

    def select_top_k(
        self,
        formulas: Sequence[str],
        features: Sequence[FeatureVector],
        k: int,
        selection_counts: Optional[Dict[str, int]] = None,
    ) -> List[ScoredCandidate]:
        """
        Select top-k candidates by score.

        Args:
            formulas: Sequence of candidate formula strings.
            features: Corresponding feature vectors.
            k: Number of candidates to select.
            selection_counts: Optional mapping for UCB.

        Returns:
            List of top-k ScoredCandidates.
        """
        ranked = self.score_and_rank(formulas, features, selection_counts)
        return ranked[:k]

    def update_selection_count(self, formula: str) -> None:
        """Record that a formula was selected."""
        self.selection_counts[formula] = self.selection_counts.get(formula, 0) + 1
        self.total_selections += 1


def score_candidate(
    formula: str,
    features: FeatureVector,
    weights: Optional[PolicyWeights] = None,
) -> float:
    """
    Convenience function to score a single candidate.

    Args:
        formula: The candidate formula string.
        features: Pre-computed feature vector.
        weights: Policy weights. If None, uses default weights.

    Returns:
        The total score for the candidate.
    """
    weights = weights or PolicyWeights.default()
    return dot_product(weights, features)


def score_candidates(
    formulas: Sequence[str],
    features: Sequence[FeatureVector],
    weights: Optional[PolicyWeights] = None,
    seed: int = 42,
) -> List[Tuple[str, float]]:
    """
    Convenience function to score multiple candidates.

    Args:
        formulas: Sequence of candidate formula strings.
        features: Corresponding feature vectors.
        weights: Policy weights. If None, uses default weights.
        seed: Random seed for deterministic behavior.

    Returns:
        List of (formula, score) tuples sorted by score descending.
    """
    scorer = PolicyScorer(weights=weights, seed=seed)
    ranked = scorer.score_and_rank(formulas, features)
    return [(c.formula, c.total_score) for c in ranked]


__all__ = [
    "PolicyWeights",
    "PolicyScorer",
    "ScoredCandidate",
    "dot_product",
    "score_candidate",
    "score_candidates",
]
