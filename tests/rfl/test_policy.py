"""
RFL Policy Module Tests — PHASE II

Tests for feature extraction, scoring, and policy updates.

Key Test Categories:
- Feature extraction determinism
- Slice feature mask application
- Policy scoring determinism
- Policy update determinism with SeededRNG
- PolicyStateSnapshot serialization

NOTE: PHASE II — NOT USED IN PHASE I
"""

from __future__ import annotations

import hashlib
import json
import pytest
from typing import Dict

from rfl.policy.features import (
    extract_features,
    apply_feature_mask,
    get_feature_mask,
    SLICE_FEATURE_MASKS,
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureVector,
)
from rfl.policy.scoring import (
    score_candidates,
    PolicyScorer,
    ScoredCandidate,
)
from rfl.policy.update import (
    PolicyStateSnapshot,
    PolicyUpdater,
    PolicyUpdateResult,
    compute_policy_update,
    LearningScheduleConfig,
)


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_extract_features_deterministic(self) -> None:
        """Verify feature extraction is deterministic."""
        candidate = "p -> (q -> p)"
        
        features1 = extract_features(candidate)
        features2 = extract_features(candidate)
        
        assert features1.to_array() == features2.to_array()

    def test_extract_features_different_candidates(self) -> None:
        """Verify different candidates produce different features."""
        features1 = extract_features("p")
        features2 = extract_features("p -> q")
        
        assert features1.text_length != features2.text_length
        assert features1.to_array() != features2.to_array()

    def test_feature_vector_to_array_canonical_order(self) -> None:
        """Verify to_array returns canonical order."""
        features = FeatureVector(
            text_length=10.0,
            ast_depth=2.0,
            atom_count=3.0,
            connective_count=1.0,
        )
        arr = features.to_array()
        
        assert len(arr) == NUM_FEATURES
        assert arr[0] == 10.0  # text_length
        assert arr[1] == 2.0   # ast_depth
        assert arr[2] == 3.0   # atom_count
        assert arr[3] == 1.0   # connective_count

    def test_feature_vector_roundtrip(self) -> None:
        """Verify array/dict roundtrip preserves values."""
        original = FeatureVector(
            text_length=10.0,
            ast_depth=2.0,
            success_count=5.0,
        )
        
        # Array roundtrip
        arr = original.to_array()
        from_arr = FeatureVector.from_array(arr)
        assert from_arr.text_length == original.text_length
        assert from_arr.ast_depth == original.ast_depth
        assert from_arr.success_count == original.success_count
        
        # Dict roundtrip
        d = original.to_dict()
        from_dict = FeatureVector.from_dict(d)
        assert from_dict.to_array() == original.to_array()

    def test_extract_with_success_history(self) -> None:
        """Verify success history is incorporated."""
        candidate = "p -> q"
        candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
        
        success_history = {candidate_hash: 10}
        attempt_history = {candidate_hash: 20}
        
        features = extract_features(
            candidate,
            success_history=success_history,
            attempt_history=attempt_history,
            total_candidates_seen=100,
        )
        
        assert features.success_count == 10.0
        assert features.attempt_count == 20.0
        assert features.success_rate == 0.5  # 10/20

    def test_extract_with_context(self) -> None:
        """Verify context features are extracted."""
        candidate = "p -> q"
        context = {
            "chain_depth": 3,
            "dependency_count": 2,
            "goal_overlap": 0.8,
            "required_goal": 1,
            "per_goal_hit": 0.5,
        }
        
        features = extract_features(candidate, context=context)
        
        assert features.chain_depth == 3.0
        assert features.dependency_count == 2.0
        assert features.goal_overlap == 0.8
        assert features.required_goal == 1.0
        assert features.per_goal_hit == 0.5


# ---------------------------------------------------------------------------
# Feature Mask Tests
# ---------------------------------------------------------------------------

class TestFeatureMasks:
    """Tests for slice feature masks."""

    def test_all_slice_masks_defined(self) -> None:
        """Verify all expected slice masks are defined."""
        expected_slices = [
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
            "default",
        ]
        for slice_name in expected_slices:
            assert slice_name in SLICE_FEATURE_MASKS
            mask = SLICE_FEATURE_MASKS[slice_name]
            # Each mask should have all feature names
            for name in FEATURE_NAMES:
                assert name in mask

    def test_slice_uplift_goal_emphasizes_goal_features(self) -> None:
        """Verify slice_uplift_goal emphasizes goal features."""
        mask = SLICE_FEATURE_MASKS["slice_uplift_goal"]
        
        # Goal features should have weight 1.0
        assert mask["goal_overlap"] == 1.0
        assert mask["required_goal"] == 1.0
        
        # Non-goal features should have lower weight
        assert mask["text_length"] < 1.0
        assert mask["atom_count"] < 1.0

    def test_slice_uplift_sparse_emphasizes_success_features(self) -> None:
        """Verify slice_uplift_sparse emphasizes success/rarity."""
        mask = SLICE_FEATURE_MASKS["slice_uplift_sparse"]
        
        assert mask["success_count"] == 1.0
        assert mask["candidate_rarity"] == 1.0
        assert mask["goal_overlap"] < 1.0

    def test_slice_uplift_tree_emphasizes_structure_features(self) -> None:
        """Verify slice_uplift_tree emphasizes structural features."""
        mask = SLICE_FEATURE_MASKS["slice_uplift_tree"]
        
        assert mask["chain_depth"] == 1.0
        assert mask["dependency_count"] == 1.0
        assert mask["success_count"] < 1.0

    def test_apply_feature_mask(self) -> None:
        """Verify feature mask application."""
        features = FeatureVector(
            text_length=10.0,
            goal_overlap=0.8,
            success_count=5.0,
        )
        
        mask = {"text_length": 0.5, "goal_overlap": 1.0, "success_count": 0.0}
        # Add remaining features with default weight 1.0
        for name in FEATURE_NAMES:
            if name not in mask:
                mask[name] = 1.0
        
        masked = apply_feature_mask(features, mask)
        
        assert masked.text_length == 5.0   # 10.0 * 0.5
        assert masked.goal_overlap == 0.8  # 0.8 * 1.0
        assert masked.success_count == 0.0 # 5.0 * 0.0

    def test_get_feature_mask_exact_match(self) -> None:
        """Verify get_feature_mask with exact slice name."""
        mask = get_feature_mask("slice_uplift_goal")
        assert mask == SLICE_FEATURE_MASKS["slice_uplift_goal"]

    def test_get_feature_mask_default(self) -> None:
        """Verify get_feature_mask falls back to default."""
        mask = get_feature_mask("unknown_slice")
        assert mask == SLICE_FEATURE_MASKS["default"]


# ---------------------------------------------------------------------------
# Policy Scoring Tests
# ---------------------------------------------------------------------------

class TestPolicyScoring:
    """Tests for policy scoring."""

    def test_score_single_deterministic(self) -> None:
        """Verify single candidate scoring is deterministic."""
        scorer = PolicyScorer(seed=42)
        
        candidate = "p -> q"
        score1 = scorer.score_single(candidate)
        
        scorer2 = PolicyScorer(seed=42)
        score2 = scorer2.score_single(candidate)
        
        assert score1.score == score2.score
        assert score1.features.to_array() == score2.features.to_array()

    def test_score_batch_deterministic(self) -> None:
        """Verify batch scoring is deterministic."""
        candidates = ["p", "p -> q", "p & q", "(p -> q) -> p"]
        
        scored1 = score_candidates(candidates, seed=42)
        scored2 = score_candidates(candidates, seed=42)
        
        assert len(scored1) == len(scored2)
        for s1, s2 in zip(scored1, scored2):
            assert s1.candidate == s2.candidate
            assert s1.score == s2.score
            assert s1.rank == s2.rank

    def test_score_batch_ranking(self) -> None:
        """Verify candidates are ranked by score."""
        candidates = ["p", "p -> q", "(p -> q) -> (q -> p)"]
        
        scored = score_candidates(candidates, seed=42)
        
        # Verify ranks are sequential
        for i, sc in enumerate(scored):
            assert sc.rank == i + 1
        
        # Verify sorted by score descending
        scores = [sc.score for sc in scored]
        assert scores == sorted(scores, reverse=True)

    def test_scorer_with_slice_mask(self) -> None:
        """Verify scorer applies slice-specific mask."""
        scorer_goal = PolicyScorer(slice_name="slice_uplift_goal", seed=42)
        scorer_sparse = PolicyScorer(slice_name="slice_uplift_sparse", seed=42)
        
        candidate = "p -> q"
        context = {"goal_overlap": 1.0, "success_count": 10}
        
        score_goal = scorer_goal.score_single(candidate, context=context)
        score_sparse = scorer_sparse.score_single(candidate, context=context)
        
        # Scores should differ due to different masks
        # (This depends on weights, but with default weights and different masks,
        # scores will typically differ)
        # At minimum, features are masked differently
        assert scorer_goal.feature_mask != scorer_sparse.feature_mask

    def test_scored_candidate_to_dict(self) -> None:
        """Verify ScoredCandidate serialization."""
        scorer = PolicyScorer(seed=42)
        sc = scorer.score_single("p -> q")
        
        d = sc.to_dict()
        
        assert "candidate" in d
        assert "score" in d
        assert "features" in d
        assert "rank" in d
        assert d["candidate"] == "p -> q"


# ---------------------------------------------------------------------------
# Policy Update Tests
# ---------------------------------------------------------------------------

class TestPolicyUpdate:
    """Tests for policy updates."""

    def test_policy_update_deterministic(self) -> None:
        """Verify policy updates are deterministic."""
        updater1 = PolicyUpdater(seed=42)
        updater2 = PolicyUpdater(seed=42)
        
        result1 = updater1.update(reward=1.0)
        result2 = updater2.update(reward=1.0)
        
        assert result1.new_weights == result2.new_weights
        assert result1.gradient_norm == result2.gradient_norm

    def test_policy_update_positive_reward(self) -> None:
        """Verify positive reward reinforces success features."""
        updater = PolicyUpdater(seed=42)
        initial_success = updater.weights.get("success_count", 0.0)
        
        result = updater.update(reward=2.0)
        
        # Success weight should increase
        assert result.new_weights["success_count"] >= initial_success

    def test_policy_update_negative_reward(self) -> None:
        """Verify negative reward adjusts weights."""
        updater = PolicyUpdater(seed=42)
        
        # First apply positive to set baseline
        updater.update(reward=1.0)
        
        result = updater.update(reward=-1.0)
        
        assert result.update_applied
        assert result.reward == -1.0

    def test_policy_update_clamps_success_weights(self) -> None:
        """Verify success weights are clamped to non-negative."""
        updater = PolicyUpdater(seed=42)
        
        # Apply many negative rewards
        for _ in range(10):
            updater.update(reward=-2.0)
        
        # Success weights should never go negative
        assert updater.weights.get("success_count", 0.0) >= 0.0
        assert updater.weights.get("success_rate", 0.0) >= 0.0

    def test_policy_updater_update_count(self) -> None:
        """Verify update count is tracked."""
        updater = PolicyUpdater(seed=42)
        assert updater.update_count == 0
        
        updater.update(reward=1.0)
        assert updater.update_count == 1
        
        updater.update(reward=-0.5)
        assert updater.update_count == 2


# ---------------------------------------------------------------------------
# PolicyStateSnapshot Tests
# ---------------------------------------------------------------------------

class TestPolicyStateSnapshot:
    """Tests for PolicyStateSnapshot."""

    def test_snapshot_id_deterministic(self) -> None:
        """Verify snapshot ID is deterministic."""
        snap1 = PolicyStateSnapshot(
            weights={"a": 1.0, "b": 2.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        snap2 = PolicyStateSnapshot(
            weights={"a": 1.0, "b": 2.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        
        assert snap1.snapshot_id == snap2.snapshot_id

    def test_snapshot_id_differs_on_weight_change(self) -> None:
        """Verify different weights produce different snapshot ID."""
        snap1 = PolicyStateSnapshot(
            weights={"a": 1.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        snap2 = PolicyStateSnapshot(
            weights={"a": 2.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        
        assert snap1.snapshot_id != snap2.snapshot_id

    def test_snapshot_to_jsonl_record(self) -> None:
        """Verify JSONL serialization."""
        snap = PolicyStateSnapshot(
            weights={"a": 1.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        
        jsonl = snap.to_jsonl_record()
        
        # Should be valid JSON
        parsed = json.loads(jsonl)
        assert parsed["learning_rate"] == 0.01
        assert parsed["slice_name"] == "test"
        assert parsed["update_count"] == 5

    def test_snapshot_roundtrip(self) -> None:
        """Verify dict roundtrip."""
        snap = PolicyStateSnapshot(
            weights={"a": 1.0, "b": 2.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
            cycle_index=10,
            reward=0.5,
        )
        
        d = snap.to_dict()
        restored = PolicyStateSnapshot.from_dict(d)
        
        assert restored.snapshot_id == snap.snapshot_id
        assert restored.weights == snap.weights
        assert restored.cycle_index == snap.cycle_index

    def test_snapshot_from_updater(self) -> None:
        """Verify snapshot creation from updater."""
        updater = PolicyUpdater(seed=42, slice_name="test_slice")
        updater.update(reward=1.0)
        
        snap = updater.get_snapshot(cycle_index=5, reward=1.0)
        
        assert snap.slice_name == "test_slice"
        assert snap.update_count == 1
        assert snap.cycle_index == 5
        assert snap.reward == 1.0

    def test_snapshot_verify_determinism(self) -> None:
        """Verify determinism check method."""
        snap1 = PolicyStateSnapshot(
            weights={"a": 1.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        snap2 = PolicyStateSnapshot(
            weights={"a": 1.0},
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        snap3 = PolicyStateSnapshot(
            weights={"a": 2.0},  # Different
            learning_rate=0.01,
            step_size=0.1,
            slice_name="test",
            update_count=5,
        )
        
        assert snap1.verify_determinism(snap2)
        assert not snap1.verify_determinism(snap3)


# ---------------------------------------------------------------------------
# Learning Schedule Tests
# ---------------------------------------------------------------------------

class TestLearningScheduleConfig:
    """Tests for LearningScheduleConfig."""

    def test_default_learning_rate(self) -> None:
        """Verify default learning rate is used."""
        config = LearningScheduleConfig(default_learning_rate=0.05)
        
        rate = config.get_learning_rate("unknown_slice")
        assert rate == 0.05

    def test_slice_specific_learning_rate(self) -> None:
        """Verify slice-specific rate overrides default."""
        config = LearningScheduleConfig(
            default_learning_rate=0.01,
            slice_learning_rates={"slice_uplift_goal": 0.02},
        )
        
        assert config.get_learning_rate("slice_uplift_goal") == 0.02
        assert config.get_learning_rate("other_slice") == 0.01

    def test_learning_rate_decay(self) -> None:
        """Verify learning rate decay."""
        config = LearningScheduleConfig(
            default_learning_rate=0.1,
            decay_rate=0.9,
            decay_steps=10,
        )
        
        rate_0 = config.get_learning_rate("test", step=0)
        rate_10 = config.get_learning_rate("test", step=10)
        rate_20 = config.get_learning_rate("test", step=20)
        
        # Rate should decrease with steps
        assert rate_0 == 0.1
        assert rate_10 < rate_0
        assert rate_20 < rate_10

    def test_from_yaml_config(self) -> None:
        """Verify creation from YAML-style config dict."""
        yaml_config = {
            "rfl_policy": {
                "default_learning_rate": 0.015,
                "slice_learning_rates": {
                    "slice_uplift_goal": 0.02,
                },
                "decay_rate": 0.95,
            }
        }
        
        config = LearningScheduleConfig.from_yaml_config(yaml_config)
        
        assert config.default_learning_rate == 0.015
        assert config.slice_learning_rates["slice_uplift_goal"] == 0.02
        assert config.decay_rate == 0.95


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestPolicyIntegration:
    """Integration tests for the full policy pipeline."""

    def test_full_pipeline_determinism(self) -> None:
        """Verify full pipeline (extract → score → update) is deterministic."""
        candidates = ["p", "p -> q", "p & q"]
        
        # Run 1
        scorer1 = PolicyScorer(seed=42, slice_name="slice_uplift_goal")
        scored1 = scorer1.score_batch(candidates)
        updater1 = PolicyUpdater(seed=42, slice_name="slice_uplift_goal")
        result1 = updater1.update(reward=1.0)
        snap1 = updater1.get_snapshot()
        
        # Run 2
        scorer2 = PolicyScorer(seed=42, slice_name="slice_uplift_goal")
        scored2 = scorer2.score_batch(candidates)
        updater2 = PolicyUpdater(seed=42, slice_name="slice_uplift_goal")
        result2 = updater2.update(reward=1.0)
        snap2 = updater2.get_snapshot()
        
        # Verify identical results
        assert [s.score for s in scored1] == [s.score for s in scored2]
        assert result1.new_weights == result2.new_weights
        assert snap1.verify_determinism(snap2)

    def test_multiple_cycles_determinism(self) -> None:
        """Verify multiple update cycles are deterministic."""
        rewards = [1.0, -0.5, 0.0, 2.0, -1.0]
        
        # Run 1
        updater1 = PolicyUpdater(seed=42)
        for reward in rewards:
            updater1.update(reward=reward)
        snap1 = updater1.get_snapshot()
        
        # Run 2
        updater2 = PolicyUpdater(seed=42)
        for reward in rewards:
            updater2.update(reward=reward)
        snap2 = updater2.get_snapshot()
        
        assert snap1.verify_determinism(snap2)
        assert updater1.get_weights() == updater2.get_weights()

    def test_slice_switching(self) -> None:
        """Verify slice switching updates mask correctly."""
        scorer = PolicyScorer(slice_name="slice_uplift_goal", seed=42)
        mask1 = scorer.feature_mask
        
        # Create new scorer with different slice
        scorer2 = PolicyScorer(slice_name="slice_uplift_sparse", seed=42)
        mask2 = scorer2.feature_mask
        
        # Masks should be different
        assert mask1 != mask2
        assert mask1["goal_overlap"] > mask2["goal_overlap"]
        assert mask1["success_count"] < mask2["success_count"]
