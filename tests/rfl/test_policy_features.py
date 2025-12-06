"""
Unit Tests for RFL Policy Features
===================================

Tests for feature extraction from propositional formulas.
Verifies determinism and correctness of all feature computations.
"""

import pytest
from typing import FrozenSet

from rfl.policy.features import (
    FeatureVector,
    extract_features,
    extract_features_batch,
)


class TestFeatureVector:
    """Tests for FeatureVector dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        fv = FeatureVector(
            len=10,
            depth=2,
            connectives=3,
            negations=1,
            overlap=2,
            goal_flag=1,
            success_count=5,
            chain_depth=3,
        )
        d = fv.to_dict()
        assert d["len"] == 10
        assert d["depth"] == 2
        assert d["connectives"] == 3
        assert d["negations"] == 1
        assert d["overlap"] == 2
        assert d["goal_flag"] == 1
        assert d["success_count"] == 5
        assert d["chain_depth"] == 3

    def test_to_list(self) -> None:
        """Test conversion to ordered list."""
        fv = FeatureVector(
            len=10, depth=2, connectives=3, negations=1,
            overlap=2, goal_flag=1, success_count=5, chain_depth=3,
        )
        lst = fv.to_list()
        assert lst == [10, 2, 3, 1, 2, 1, 5, 3]

    def test_field_names(self) -> None:
        """Test field name ordering matches to_list."""
        names = FeatureVector.field_names()
        assert names == [
            "len", "depth", "connectives", "negations",
            "overlap", "goal_flag", "success_count", "chain_depth"
        ]

    def test_zero(self) -> None:
        """Test zero feature vector."""
        zero = FeatureVector.zero()
        assert all(v == 0 for v in zero.to_list())

    def test_frozen(self) -> None:
        """Test that FeatureVector is immutable."""
        fv = FeatureVector(
            len=10, depth=2, connectives=3, negations=1,
            overlap=2, goal_flag=1, success_count=5, chain_depth=3,
        )
        with pytest.raises(AttributeError):
            fv.len = 20  # type: ignore


class TestExtractFeatures:
    """Tests for feature extraction."""

    def test_simple_atom(self) -> None:
        """Test features of atomic proposition."""
        fv = extract_features("p")
        assert fv.len == 1
        assert fv.depth == 0
        assert fv.connectives == 0
        assert fv.negations == 0

    def test_implication(self) -> None:
        """Test features of simple implication."""
        fv = extract_features("p->q")
        assert fv.len == 4
        assert fv.depth == 1
        assert fv.connectives == 1
        assert fv.negations == 0

    def test_negation(self) -> None:
        """Test features with negation."""
        fv = extract_features("~p")
        assert fv.len == 2
        assert fv.depth == 1
        assert fv.connectives == 0
        assert fv.negations == 1

    def test_double_negation(self) -> None:
        """Test features with double negation."""
        fv = extract_features("~~p")
        assert fv.len == 3
        assert fv.negations == 2

    def test_conjunction(self) -> None:
        """Test features of conjunction."""
        fv = extract_features("p/\\q")
        assert fv.connectives == 1
        assert fv.depth == 1

    def test_disjunction(self) -> None:
        """Test features of disjunction."""
        fv = extract_features("p\\/q")
        assert fv.connectives == 1
        assert fv.depth == 1

    def test_complex_formula(self) -> None:
        """Test features of complex nested formula."""
        # (p /\ q) -> (r \/ ~s)
        fv = extract_features("(p/\\q)->(r\\/~s)")
        assert fv.connectives >= 3  # /\, ->, \/
        assert fv.negations == 1    # ~s
        assert fv.depth >= 2

    def test_overlap_with_target(self) -> None:
        """Test overlap computation with target atoms."""
        target = frozenset({"p", "q", "r"})
        fv = extract_features("p->q", target_atoms=target)
        assert fv.overlap == 2  # p and q overlap

    def test_overlap_partial(self) -> None:
        """Test partial overlap with target."""
        target = frozenset({"p", "x", "y"})
        fv = extract_features("p->q", target_atoms=target)
        assert fv.overlap == 1  # Only p overlaps

    def test_overlap_none(self) -> None:
        """Test no overlap with target."""
        target = frozenset({"x", "y", "z"})
        fv = extract_features("p->q", target_atoms=target)
        assert fv.overlap == 0

    def test_goal_flag(self) -> None:
        """Test goal flag setting."""
        fv_goal = extract_features("p->q", is_goal=True)
        fv_not_goal = extract_features("p->q", is_goal=False)
        assert fv_goal.goal_flag == 1
        assert fv_not_goal.goal_flag == 0

    def test_success_count(self) -> None:
        """Test success count passthrough."""
        fv = extract_features("p->q", success_count=42)
        assert fv.success_count == 42

    def test_chain_depth(self) -> None:
        """Test chain depth passthrough."""
        fv = extract_features("p->q", chain_depth=5)
        assert fv.chain_depth == 5

    def test_determinism(self) -> None:
        """Test that feature extraction is deterministic."""
        target = frozenset({"p", "q"})
        results = [
            extract_features(
                "p->q",
                target_atoms=target,
                is_goal=True,
                success_count=3,
                chain_depth=2,
            )
            for _ in range(100)
        ]

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result == first


class TestExtractFeaturesBatch:
    """Tests for batch feature extraction."""

    def test_batch_basic(self) -> None:
        """Test basic batch extraction."""
        formulas = ["p", "p->q", "~p"]
        features = extract_features_batch(formulas)

        assert len(features) == 3
        assert features[0].len == 1  # p
        assert features[1].len == 4  # p->q
        assert features[2].len == 2  # ~p

    def test_batch_with_target(self) -> None:
        """Test batch with target atoms."""
        formulas = ["p->q", "q->r", "x->y"]
        target = frozenset({"p", "q"})
        features = extract_features_batch(formulas, target_atoms=target)

        assert features[0].overlap == 2  # p, q
        assert features[1].overlap == 1  # q only
        assert features[2].overlap == 0  # no overlap

    def test_batch_with_goal_set(self) -> None:
        """Test batch with goal set."""
        formulas = ["p->q", "q->r", "r->s"]
        goal_set = frozenset({"p->q", "r->s"})
        features = extract_features_batch(formulas, is_goal_set=goal_set)

        assert features[0].goal_flag == 1
        assert features[1].goal_flag == 0
        assert features[2].goal_flag == 1

    def test_batch_with_success_counts(self) -> None:
        """Test batch with success count mapping."""
        formulas = ["p->q", "q->r", "r->s"]
        success_counts = {"p->q": 10, "r->s": 5}
        features = extract_features_batch(formulas, success_counts=success_counts)

        assert features[0].success_count == 10
        assert features[1].success_count == 0
        assert features[2].success_count == 5

    def test_batch_empty(self) -> None:
        """Test empty batch."""
        features = extract_features_batch([])
        assert features == []

    def test_batch_determinism(self) -> None:
        """Test batch extraction is deterministic."""
        formulas = ["p", "p->q", "~p"]
        target = frozenset({"p"})

        results = [
            extract_features_batch(formulas, target_atoms=target)
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert len(result) == len(first)
            for a, b in zip(result, first):
                assert a == b


class TestFeatureCaching:
    """Tests for feature extraction caching."""

    def test_cached_results_identical(self) -> None:
        """Test that cached results are identical."""
        # Call twice with same input
        fv1 = extract_features("(p/\\q)->(r\\/s)")
        fv2 = extract_features("(p/\\q)->(r\\/s)")

        # Should be identical (cached)
        assert fv1.len == fv2.len
        assert fv1.depth == fv2.depth
        assert fv1.connectives == fv2.connectives


@pytest.mark.determinism
class TestDeterminismGuarantees:
    """Tests verifying determinism guarantees for RFL Law compliance."""

    def test_feature_extraction_100_iterations(self) -> None:
        """
        Verify feature extraction is deterministic across 100 iterations.

        This is a core RFL Law requirement: same input → same output.
        """
        formula = "(p/\\q)->r"
        target = frozenset({"p", "q", "r", "s"})

        reference = extract_features(
            formula,
            target_atoms=target,
            is_goal=True,
            success_count=42,
            chain_depth=3,
        )

        for i in range(100):
            result = extract_features(
                formula,
                target_atoms=target,
                is_goal=True,
                success_count=42,
                chain_depth=3,
            )
            assert result == reference, f"Determinism violation at iteration {i}"

    def test_batch_extraction_stability(self) -> None:
        """Verify batch extraction produces stable results."""
        formulas = [
            "p",
            "p->q",
            "p/\\q",
            "p\\/q",
            "~p",
            "(p->q)->(q->r)",
        ]
        target = frozenset({"p", "q"})
        goal_set = frozenset({"p->q"})

        reference = extract_features_batch(
            formulas,
            target_atoms=target,
            is_goal_set=goal_set,
        )

        for _ in range(50):
            result = extract_features_batch(
                formulas,
                target_atoms=target,
                is_goal_set=goal_set,
            )
            assert len(result) == len(reference)
            for a, b in zip(result, reference):
                assert a == b
