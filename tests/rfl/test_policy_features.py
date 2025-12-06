"""
Tests for RFL Policy Feature Extraction Module.

Tests verify:
1. Feature extraction is deterministic
2. Phase I slices return zero features (baseline)
3. Phase II slices return rich features
4. Feature scoring works correctly with weights
5. Context updates work correctly
"""

from __future__ import annotations

import pytest
from typing import Dict

from rfl.policy_features import (
    RFLContext,
    CandidateFeatures,
    extract_features,
    compute_feature_score,
    update_context_from_cycle,
    create_context_for_slice,
    get_feature_names,
    get_default_weights,
    _compute_depth,
    _count_connectives,
    _count_negations,
    _compute_goal_overlap,
    _compute_candidate_hash,
    _is_phase_ii_slice,
    PHASE_I_SLICES,
    PHASE_II_SLICES,
)


class TestStructuralFeatures:
    """Tests for basic structural feature computation."""

    def test_compute_depth_simple(self) -> None:
        """Test depth computation for simple formulas."""
        assert _compute_depth("p") == 0
        assert _compute_depth("(p)") == 1
        assert _compute_depth("((p))") == 2
        assert _compute_depth("(p -> q)") == 1
        assert _compute_depth("((p -> q) -> r)") == 2

    def test_compute_depth_nested(self) -> None:
        """Test depth computation for deeply nested formulas."""
        assert _compute_depth("(((p)))") == 3
        assert _compute_depth("((p -> (q -> r)))") == 3
        assert _compute_depth("(p /\\ (q \\/ (r -> s)))") == 3

    def test_count_connectives_ascii(self) -> None:
        """Test connective counting with ASCII operators."""
        assert _count_connectives("p") == 0
        assert _count_connectives("p -> q") == 1
        assert _count_connectives("p /\\ q") == 1
        assert _count_connectives("p \\/ q") == 1
        assert _count_connectives("p <-> q") == 1
        assert _count_connectives("p -> q -> r") == 2
        assert _count_connectives("(p /\\ q) -> (r \\/ s)") == 3

    def test_count_connectives_unicode(self) -> None:
        """Test connective counting with Unicode operators."""
        assert _count_connectives("p → q") == 1
        assert _count_connectives("p ∧ q") == 1
        assert _count_connectives("p ∨ q") == 1
        assert _count_connectives("p ↔ q") == 1

    def test_count_negations(self) -> None:
        """Test negation counting."""
        assert _count_negations("p") == 0
        assert _count_negations("~p") == 1
        assert _count_negations("¬p") == 1
        assert _count_negations("!p") == 1
        assert _count_negations("~~p") == 2
        assert _count_negations("~p /\\ ~q") == 2


class TestGoalFeatures:
    """Tests for goal-related feature computation."""

    def test_goal_overlap_empty_goals(self) -> None:
        """Test goal overlap with no target goals."""
        assert _compute_goal_overlap("p -> q", []) == 0.0

    def test_goal_overlap_exact_match(self) -> None:
        """Test goal overlap with exact matches."""
        goals = ["p -> q"]
        assert _compute_goal_overlap("p -> q", goals) == 1.0

    def test_goal_overlap_atom_match(self) -> None:
        """Test goal overlap with atom matching."""
        goals = ["p -> r"]
        # p appears in both
        assert _compute_goal_overlap("p -> q", goals) == 1.0

    def test_goal_overlap_no_match(self) -> None:
        """Test goal overlap with no matching atoms."""
        goals = ["x -> y"]
        # No overlap with p, q
        assert _compute_goal_overlap("p -> q", goals) == 0.0

    def test_goal_overlap_partial(self) -> None:
        """Test goal overlap with partial matches."""
        goals = ["p -> r", "x -> y"]
        # Only first goal matches (p)
        assert _compute_goal_overlap("p -> q", goals) == 0.5


class TestCandidateHash:
    """Tests for candidate hash computation."""

    def test_hash_deterministic(self) -> None:
        """Test that hash is deterministic."""
        h1 = _compute_candidate_hash("p -> q")
        h2 = _compute_candidate_hash("p -> q")
        assert h1 == h2

    def test_hash_ignores_whitespace(self) -> None:
        """Test that hash ignores whitespace differences."""
        h1 = _compute_candidate_hash("p -> q")
        h2 = _compute_candidate_hash("p->q")
        h3 = _compute_candidate_hash("  p  ->  q  ")
        assert h1 == h2 == h3

    def test_hash_ignores_case(self) -> None:
        """Test that hash ignores case differences."""
        h1 = _compute_candidate_hash("P -> Q")
        h2 = _compute_candidate_hash("p -> q")
        assert h1 == h2

    def test_hash_different_formulas(self) -> None:
        """Test that different formulas have different hashes."""
        h1 = _compute_candidate_hash("p -> q")
        h2 = _compute_candidate_hash("q -> p")
        assert h1 != h2


class TestSliceClassification:
    """Tests for slice type classification."""

    def test_phase_i_slices(self) -> None:
        """Test that Phase I slices are correctly identified."""
        for slice_name in PHASE_I_SLICES:
            assert not _is_phase_ii_slice(slice_name)

    def test_phase_ii_slices(self) -> None:
        """Test that Phase II slices are correctly identified."""
        for slice_name in PHASE_II_SLICES:
            assert _is_phase_ii_slice(slice_name)

    def test_phase_ii_prefix_match(self) -> None:
        """Test that Phase II slice variants are matched."""
        assert _is_phase_ii_slice("goal_v2")
        assert _is_phase_ii_slice("tree_deep")
        assert _is_phase_ii_slice("phase_ii_goal_extended")

    def test_unknown_slice_is_phase_i(self) -> None:
        """Test that unknown slices default to Phase I behavior."""
        assert not _is_phase_ii_slice("unknown_slice")
        assert not _is_phase_ii_slice("custom_experiment")


class TestFeatureExtraction:
    """Tests for main feature extraction function."""

    @pytest.fixture
    def phase_i_context(self) -> RFLContext:
        """Create context for Phase I slice."""
        return RFLContext(slice_name="warmup", slice_index=0)

    @pytest.fixture
    def phase_ii_context(self) -> RFLContext:
        """Create context for Phase II slice."""
        return RFLContext(
            slice_name="goal",
            slice_index=10,
            target_goals=["p -> p", "q -> q"],
            success_history={"abc123": 5},
            attempt_history={"abc123": 10},
        )

    def test_phase_i_returns_zero_features(self, phase_i_context: RFLContext) -> None:
        """Test that Phase I slices return zero features."""
        for slice_name in ["warmup", "core", "refinement", "baseline", "tail"]:
            features = extract_features("p -> q", slice_name, phase_i_context)
            for value in features.values():
                assert value == 0.0

    def test_phase_ii_returns_nonzero_features(self, phase_ii_context: RFLContext) -> None:
        """Test that Phase II slices return non-zero features."""
        features = extract_features("p -> q", "goal", phase_ii_context)
        
        # Structural features should be non-zero
        assert features["len"] > 0
        assert features["depth"] >= 0  # Could be 0 for simple formula
        assert features["num_connectives"] > 0  # Has ->

    def test_goal_features_populated(self, phase_ii_context: RFLContext) -> None:
        """Test that goal features are populated for goal slices."""
        features = extract_features("p -> p", "goal", phase_ii_context)
        # p -> p matches target goal
        assert features["goal_overlap"] > 0

    def test_tree_features_populated(self) -> None:
        """Test that tree features are populated for tree slices."""
        context = RFLContext(
            slice_name="tree",
            derivation_tree={"parent_hash": ["child_hash"]},
        )
        features = extract_features("formula", "tree", context)
        # Tree features should be present
        assert "chain_depth" in features
        assert "num_children" in features
        assert "num_parents" in features

    def test_feature_extraction_deterministic(self, phase_ii_context: RFLContext) -> None:
        """Test that feature extraction is deterministic."""
        features1 = extract_features("p -> q", "goal", phase_ii_context)
        features2 = extract_features("p -> q", "goal", phase_ii_context)
        
        assert features1 == features2

    def test_different_formulas_different_features(
        self, phase_ii_context: RFLContext
    ) -> None:
        """Test that different formulas have different features."""
        features1 = extract_features("p", "goal", phase_ii_context)
        features2 = extract_features("p -> q -> r", "goal", phase_ii_context)
        
        assert features1["len"] != features2["len"]


class TestFeatureScoring:
    """Tests for feature scoring with weights."""

    def test_zero_weights_zero_score(self) -> None:
        """Test that zero weights produce zero score."""
        features = {"len": 10.0, "depth": 3.0, "success": 5.0}
        weights = {"len": 0.0, "depth": 0.0, "success": 0.0}
        
        assert compute_feature_score(features, weights) == 0.0

    def test_single_weight(self) -> None:
        """Test scoring with single non-zero weight."""
        features = {"len": 10.0, "depth": 3.0}
        weights = {"len": 0.5, "depth": 0.0}
        
        assert compute_feature_score(features, weights) == 5.0  # 10 * 0.5

    def test_multiple_weights(self) -> None:
        """Test scoring with multiple weights."""
        features = {"len": 10.0, "depth": 4.0}
        weights = {"len": 0.5, "depth": 0.25}
        
        expected = 10.0 * 0.5 + 4.0 * 0.25  # 5.0 + 1.0 = 6.0
        assert compute_feature_score(features, weights) == expected

    def test_missing_weight_uses_zero(self) -> None:
        """Test that missing weights default to zero."""
        features = {"len": 10.0, "missing_feature": 100.0}
        weights = {"len": 0.5}  # No weight for missing_feature
        
        assert compute_feature_score(features, weights) == 5.0

    def test_negative_weights(self) -> None:
        """Test scoring with negative weights (penalize feature)."""
        features = {"len": 10.0}
        weights = {"len": -0.1}  # Penalize long formulas
        
        assert compute_feature_score(features, weights) == -1.0


class TestContextUpdate:
    """Tests for context update from cycle outcomes."""

    def test_update_increments_attempt(self) -> None:
        """Test that update increments attempt count."""
        context = RFLContext(slice_name="goal")
        
        update_context_from_cycle(context, "hash1", cycle_success=False)
        
        assert context.attempt_history["hash1"] == 1

    def test_update_increments_success(self) -> None:
        """Test that successful cycle increments success count."""
        context = RFLContext(slice_name="goal")
        
        update_context_from_cycle(context, "hash1", cycle_success=True)
        
        assert context.success_history["hash1"] == 1
        assert context.attempt_history["hash1"] == 1

    def test_update_no_success_on_failure(self) -> None:
        """Test that failed cycle doesn't increment success count."""
        context = RFLContext(slice_name="goal")
        
        update_context_from_cycle(context, "hash1", cycle_success=False)
        
        assert context.success_history.get("hash1", 0) == 0
        assert context.attempt_history["hash1"] == 1

    def test_update_multiple_cycles(self) -> None:
        """Test updates across multiple cycles."""
        context = RFLContext(slice_name="goal")
        
        update_context_from_cycle(context, "hash1", cycle_success=True)
        update_context_from_cycle(context, "hash1", cycle_success=True)
        update_context_from_cycle(context, "hash1", cycle_success=False)
        
        assert context.success_history["hash1"] == 2
        assert context.attempt_history["hash1"] == 3

    def test_update_verified_count(self) -> None:
        """Test that verified count is updated."""
        context = RFLContext(slice_name="goal")
        
        update_context_from_cycle(context, "hash1", cycle_success=True, verified_count=7)
        
        assert context.cycle_verified_count == 7


class TestContextFactory:
    """Tests for context factory function."""

    def test_create_context_basic(self) -> None:
        """Test basic context creation."""
        context = create_context_for_slice("goal", slice_index=5)
        
        assert context.slice_name == "goal"
        assert context.slice_index == 5
        assert context.success_history == {}
        assert context.attempt_history == {}

    def test_create_context_with_history(self) -> None:
        """Test context creation with existing history."""
        success = {"hash1": 5}
        attempt = {"hash1": 10}
        
        context = create_context_for_slice(
            "goal",
            existing_success_history=success,
            existing_attempt_history=attempt,
        )
        
        assert context.success_history["hash1"] == 5
        assert context.attempt_history["hash1"] == 10

    def test_create_context_copies_dicts(self) -> None:
        """Test that context copies input dicts (no aliasing)."""
        success = {"hash1": 5}
        
        context = create_context_for_slice("goal", existing_success_history=success)
        context.success_history["hash2"] = 10
        
        # Original dict should be unchanged
        assert "hash2" not in success


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_feature_names_returns_list(self) -> None:
        """Test that feature names are returned as list."""
        names = get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert "len" in names
        assert "depth" in names
        assert "success_count" in names

    def test_get_default_weights_returns_zeros(self) -> None:
        """Test that default weights are all zero."""
        weights = get_default_weights()
        
        assert isinstance(weights, dict)
        for name, value in weights.items():
            assert value == 0.0

    def test_default_weights_cover_all_features(self) -> None:
        """Test that default weights cover all feature names."""
        names = get_feature_names()
        weights = get_default_weights()
        
        for name in names:
            assert name in weights


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    @pytest.mark.determinism
    def test_extract_features_deterministic_100x(self) -> None:
        """Test feature extraction is deterministic across 100 calls."""
        context = RFLContext(slice_name="goal", target_goals=["p -> q"])
        
        baseline = extract_features("p -> q", "goal", context)
        
        for _ in range(100):
            features = extract_features("p -> q", "goal", context)
            assert features == baseline

    @pytest.mark.determinism
    def test_score_deterministic_100x(self) -> None:
        """Test scoring is deterministic across 100 calls."""
        features = {"len": 10.0, "depth": 3.0, "success_count": 5.0}
        weights = {"len": 0.1, "depth": 0.2, "success_count": 0.5}
        
        baseline = compute_feature_score(features, weights)
        
        for _ in range(100):
            score = compute_feature_score(features, weights)
            assert score == baseline

    @pytest.mark.determinism
    def test_hash_deterministic_100x(self) -> None:
        """Test hash computation is deterministic across 100 calls."""
        baseline = _compute_candidate_hash("p -> q")
        
        for _ in range(100):
            h = _compute_candidate_hash("p -> q")
            assert h == baseline


class TestCandidateFeaturesDataclass:
    """Tests for CandidateFeatures dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        features = CandidateFeatures(
            length=10.0,
            depth=3.0,
            num_connectives=2.0,
        )
        
        d = features.to_dict()
        
        assert d["length"] == 10.0
        assert d["depth"] == 3.0
        assert d["num_connectives"] == 2.0

    def test_default_values(self) -> None:
        """Test that default values are zero."""
        features = CandidateFeatures()
        
        d = features.to_dict()
        for value in d.values():
            assert value == 0.0
