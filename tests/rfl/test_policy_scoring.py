"""
Unit Tests for RFL Policy Scoring
==================================

Tests for policy scoring and candidate ranking.
Verifies determinism of scoring and tie-breaking.
"""

import pytest
import math
from typing import List

from rfl.policy.features import FeatureVector, extract_features
from rfl.policy.scoring import (
    PolicyWeights,
    PolicyScorer,
    ScoredCandidate,
    dot_product,
    score_candidate,
    score_candidates,
)


class TestPolicyWeights:
    """Tests for PolicyWeights."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        w = PolicyWeights(len=-0.1, depth=0.05, success_count=0.5)
        d = w.to_dict()
        assert d["len"] == -0.1
        assert d["depth"] == 0.05
        assert d["success_count"] == 0.5
        assert d["connectives"] == 0.0  # Default

    def test_to_list(self) -> None:
        """Test conversion to ordered list."""
        w = PolicyWeights(
            len=1.0, depth=2.0, connectives=3.0, negations=4.0,
            overlap=5.0, goal_flag=6.0, success_count=7.0, chain_depth=8.0,
        )
        lst = w.to_list()
        assert lst == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_from_dict(self) -> None:
        """Test creation from dict."""
        d = {"len": -0.1, "depth": 0.05, "success_count": 0.5}
        w = PolicyWeights.from_dict(d)
        assert w.len == -0.1
        assert w.depth == 0.05
        assert w.success_count == 0.5
        assert w.connectives == 0.0  # Default

    def test_default(self) -> None:
        """Test default weights."""
        w = PolicyWeights.default()
        assert w.len < 0  # Prefer shorter
        assert w.overlap > 0  # Prefer overlap
        assert w.goal_flag > 0  # Prefer goals
        assert w.success_count >= 0  # Prefer success


class TestDotProduct:
    """Tests for dot product computation."""

    def test_zero_weights(self) -> None:
        """Test dot product with zero weights."""
        w = PolicyWeights()  # All zeros
        fv = FeatureVector(
            len=10, depth=5, connectives=3, negations=1,
            overlap=2, goal_flag=1, success_count=5, chain_depth=2,
        )
        assert dot_product(w, fv) == 0.0

    def test_zero_features(self) -> None:
        """Test dot product with zero features."""
        w = PolicyWeights(len=1.0, depth=2.0, overlap=3.0)
        fv = FeatureVector.zero()
        assert dot_product(w, fv) == 0.0

    def test_simple_product(self) -> None:
        """Test simple dot product."""
        w = PolicyWeights(len=1.0, depth=2.0)
        fv = FeatureVector(
            len=3, depth=4, connectives=0, negations=0,
            overlap=0, goal_flag=0, success_count=0, chain_depth=0,
        )
        # 1.0 * 3 + 2.0 * 4 = 11.0
        assert dot_product(w, fv) == 11.0

    def test_negative_weights(self) -> None:
        """Test dot product with negative weights."""
        w = PolicyWeights(len=-0.1)
        fv = FeatureVector(
            len=10, depth=0, connectives=0, negations=0,
            overlap=0, goal_flag=0, success_count=0, chain_depth=0,
        )
        assert dot_product(w, fv) == -1.0


class TestPolicyScorer:
    """Tests for PolicyScorer."""

    def test_score_single(self) -> None:
        """Test scoring a single candidate."""
        scorer = PolicyScorer(seed=42)
        fv = FeatureVector(
            len=4, depth=1, connectives=1, negations=0,
            overlap=2, goal_flag=0, success_count=3, chain_depth=0,
        )

        scored = scorer.score("p->q", fv)
        assert scored.formula == "p->q"
        assert scored.features == fv
        assert isinstance(scored.base_score, float)
        assert scored.exploration_bonus == 0.0  # No exploration by default
        assert len(scored.tiebreaker) == 16  # SHA256 prefix

    def test_score_with_exploration(self) -> None:
        """Test scoring with UCB exploration."""
        scorer = PolicyScorer(
            seed=42,
            exploration_weight=1.0,
            total_selections=100,
        )
        fv = FeatureVector(
            len=4, depth=1, connectives=1, negations=0,
            overlap=2, goal_flag=0, success_count=3, chain_depth=0,
        )

        # First selection (count=0 → uses 1)
        scored = scorer.score("p->q", fv, formula_selection_count=1)
        assert scored.exploration_bonus > 0

        # Many selections → smaller bonus
        scored_many = scorer.score("p->q", fv, formula_selection_count=50)
        assert scored_many.exploration_bonus < scored.exploration_bonus

    def test_score_and_rank(self) -> None:
        """Test scoring and ranking multiple candidates."""
        scorer = PolicyScorer(
            weights=PolicyWeights(goal_flag=10.0),
            seed=42,
        )

        formulas = ["p", "p->q", "goal"]
        features = [
            FeatureVector(
                len=1, depth=0, connectives=0, negations=0,
                overlap=0, goal_flag=0, success_count=0, chain_depth=0,
            ),
            FeatureVector(
                len=4, depth=1, connectives=1, negations=0,
                overlap=0, goal_flag=0, success_count=0, chain_depth=0,
            ),
            FeatureVector(
                len=4, depth=0, connectives=0, negations=0,
                overlap=0, goal_flag=1, success_count=0, chain_depth=0,
            ),
        ]

        ranked = scorer.score_and_rank(formulas, features)

        # Goal formula should rank first (goal_flag=1, weight=10)
        assert ranked[0].formula == "goal"
        assert ranked[0].total_score == 10.0

    def test_deterministic_tiebreaking(self) -> None:
        """Test that tie-breaking is deterministic."""
        scorer = PolicyScorer(
            weights=PolicyWeights(),  # All zeros → all scores equal
            seed=42,
        )

        # All features identical → scores tied
        fv = FeatureVector.zero()
        formulas = ["aaa", "bbb", "ccc"]
        features = [fv, fv, fv]

        ranked = scorer.score_and_rank(formulas, features)

        # Order should be consistent based on hash tiebreaker
        tiebreakers = [c.tiebreaker for c in ranked]
        assert tiebreakers == sorted(tiebreakers)

    def test_select_top_k(self) -> None:
        """Test selecting top-k candidates."""
        scorer = PolicyScorer(
            weights=PolicyWeights(success_count=1.0),
            seed=42,
        )

        formulas = ["a", "b", "c", "d", "e"]
        features = [
            FeatureVector(
                len=1, depth=0, connectives=0, negations=0,
                overlap=0, goal_flag=0, success_count=i, chain_depth=0,
            )
            for i in [5, 3, 1, 4, 2]
        ]

        top3 = scorer.select_top_k(formulas, features, k=3)

        assert len(top3) == 3
        # Should be ordered by success_count desc
        assert top3[0].features.success_count == 5
        assert top3[1].features.success_count == 4
        assert top3[2].features.success_count == 3

    def test_update_selection_count(self) -> None:
        """Test selection count tracking."""
        scorer = PolicyScorer(seed=42)

        assert scorer.total_selections == 0
        assert scorer.selection_counts.get("p->q", 0) == 0

        scorer.update_selection_count("p->q")
        assert scorer.total_selections == 1
        assert scorer.selection_counts["p->q"] == 1

        scorer.update_selection_count("p->q")
        assert scorer.total_selections == 2
        assert scorer.selection_counts["p->q"] == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_score_candidate(self) -> None:
        """Test score_candidate function."""
        fv = FeatureVector(
            len=4, depth=1, connectives=1, negations=0,
            overlap=2, goal_flag=1, success_count=3, chain_depth=0,
        )
        weights = PolicyWeights(goal_flag=5.0, overlap=1.0)

        score = score_candidate("p->q", fv, weights)
        # 5.0 * 1 + 1.0 * 2 = 7.0
        assert score == 7.0

    def test_score_candidates(self) -> None:
        """Test score_candidates function."""
        formulas = ["a", "b"]
        features = [
            FeatureVector(
                len=1, depth=0, connectives=0, negations=0,
                overlap=0, goal_flag=1, success_count=0, chain_depth=0,
            ),
            FeatureVector(
                len=1, depth=0, connectives=0, negations=0,
                overlap=0, goal_flag=0, success_count=0, chain_depth=0,
            ),
        ]
        weights = PolicyWeights(goal_flag=10.0)

        result = score_candidates(formulas, features, weights)

        assert len(result) == 2
        # First should be higher scoring
        assert result[0][0] == "a"  # Goal formula
        assert result[0][1] == 10.0
        assert result[1][0] == "b"
        assert result[1][1] == 0.0


class TestScoredCandidateOrdering:
    """Tests for ScoredCandidate comparison."""

    def test_higher_score_first(self) -> None:
        """Test that higher scores sort first."""
        fv = FeatureVector.zero()
        c1 = ScoredCandidate(
            formula="a", features=fv, base_score=10.0,
            exploration_bonus=0.0, total_score=10.0, tiebreaker="aaa"
        )
        c2 = ScoredCandidate(
            formula="b", features=fv, base_score=5.0,
            exploration_bonus=0.0, total_score=5.0, tiebreaker="bbb"
        )

        sorted_list = sorted([c2, c1])
        assert sorted_list[0].formula == "a"  # Higher score

    def test_tiebreaker_on_equal_score(self) -> None:
        """Test tiebreaker when scores are equal."""
        fv = FeatureVector.zero()
        c1 = ScoredCandidate(
            formula="a", features=fv, base_score=5.0,
            exploration_bonus=0.0, total_score=5.0, tiebreaker="zzz"
        )
        c2 = ScoredCandidate(
            formula="b", features=fv, base_score=5.0,
            exploration_bonus=0.0, total_score=5.0, tiebreaker="aaa"
        )

        sorted_list = sorted([c1, c2])
        assert sorted_list[0].tiebreaker == "aaa"  # Lower tiebreaker


@pytest.mark.determinism
class TestScoringDeterminism:
    """Tests verifying scoring determinism for RFL Law compliance."""

    def test_scorer_determinism_100_iterations(self) -> None:
        """
        Verify scorer produces identical rankings across 100 iterations.

        This is a core RFL Law requirement.
        """
        formulas = ["p->q", "q->r", "r->s", "~p", "p/\\q"]
        features = [extract_features(f) for f in formulas]

        reference_scorer = PolicyScorer(seed=12345)
        reference = reference_scorer.score_and_rank(formulas, features)
        reference_order = [c.formula for c in reference]

        for i in range(100):
            scorer = PolicyScorer(seed=12345)
            result = scorer.score_and_rank(formulas, features)
            result_order = [c.formula for c in result]

            assert result_order == reference_order, f"Determinism violation at iteration {i}"

    def test_tiebreaker_determinism(self) -> None:
        """Verify tiebreaker is deterministic."""
        # Create many tied candidates
        formulas = [f"var_{i}" for i in range(50)]
        features = [FeatureVector.zero() for _ in formulas]

        reference_scorer = PolicyScorer(seed=42)
        reference = reference_scorer.score_and_rank(formulas, features)
        reference_order = [c.formula for c in reference]

        for _ in range(20):
            scorer = PolicyScorer(seed=42)
            result = scorer.score_and_rank(formulas, features)
            result_order = [c.formula for c in result]

            assert result_order == reference_order
