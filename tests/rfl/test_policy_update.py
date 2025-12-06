"""
Unit Tests for RFL Policy Update
=================================

Tests for policy weight updates based on verifiable feedback.
Verifies determinism and correctness of update rules.
"""

import pytest
import math

from rfl.policy.features import FeatureVector
from rfl.policy.scoring import PolicyWeights
from rfl.policy.update import (
    SliceFeedback,
    PolicyUpdateResult,
    PolicyUpdater,
    SLICE_FEATURE_MASKS,
)


class TestSliceFeedback:
    """Tests for SliceFeedback."""

    def test_success_rate_calculation(self) -> None:
        """Test success rate computation."""
        feedback = SliceFeedback(
            slice_name="test",
            verified_count=7,
            attempted_count=10,
            target_threshold=7,
        )
        assert feedback.success_rate == 0.7

    def test_success_rate_zero_attempts(self) -> None:
        """Test success rate with zero attempts."""
        feedback = SliceFeedback(
            slice_name="test",
            verified_count=0,
            attempted_count=0,
            target_threshold=7,
        )
        assert feedback.success_rate == 0.0

    def test_reward_signal_positive(self) -> None:
        """Test positive reward signal (exceeded threshold)."""
        feedback = SliceFeedback(
            slice_name="test",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )
        assert feedback.reward_signal == 3.0  # 10 - 7

    def test_reward_signal_negative(self) -> None:
        """Test negative reward signal (below threshold)."""
        feedback = SliceFeedback(
            slice_name="test",
            verified_count=5,
            attempted_count=10,
            target_threshold=7,
        )
        assert feedback.reward_signal == -2.0  # 5 - 7

    def test_reward_signal_zero(self) -> None:
        """Test zero reward signal (at threshold)."""
        feedback = SliceFeedback(
            slice_name="test",
            verified_count=7,
            attempted_count=10,
            target_threshold=7,
        )
        assert feedback.reward_signal == 0.0

    def test_is_success(self) -> None:
        """Test success determination."""
        success = SliceFeedback(
            slice_name="test",
            verified_count=8,
            attempted_count=10,
            target_threshold=7,
        )
        failure = SliceFeedback(
            slice_name="test",
            verified_count=5,
            attempted_count=10,
            target_threshold=7,
        )
        assert success.is_success is True
        assert failure.is_success is False


class TestPolicyUpdater:
    """Tests for PolicyUpdater."""

    def test_update_positive_reward(self) -> None:
        """Test weight update with positive reward."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="core",
            verified_count=12,
            attempted_count=15,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Positive reward should update in positive direction
        assert result.reward_signal == 5.0  # 12 - 7
        assert result.update_magnitude > 0
        assert "success_count" in result.features_updated

    def test_update_negative_reward(self) -> None:
        """Test weight update with negative reward."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="core",
            verified_count=3,
            attempted_count=10,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Negative reward should update in negative direction
        assert result.reward_signal == -4.0  # 3 - 7
        assert result.update_magnitude > 0

    def test_update_zero_reward(self) -> None:
        """Test weight update with zero reward."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="core",
            verified_count=7,
            attempted_count=10,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Zero reward should produce zero updates
        assert result.reward_signal == 0.0
        assert result.update_magnitude == 0.0

    def test_feature_mask_warmup(self) -> None:
        """Test that warmup slice only updates structural features."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="warmup",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Only warmup features should be updated
        warmup_features = SLICE_FEATURE_MASKS["warmup"]
        for feature in result.features_updated:
            assert feature in warmup_features

    def test_feature_mask_refinement(self) -> None:
        """Test that refinement slice updates appropriate features."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="refinement",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Only refinement features should be updated
        refinement_features = SLICE_FEATURE_MASKS["refinement"]
        for feature in result.features_updated:
            assert feature in refinement_features

    def test_success_weight_floor(self) -> None:
        """Test that success_count weight doesn't go negative."""
        updater = PolicyUpdater(
            seed=42,
            success_weight_floor=0.0,
            base_learning_rate=1.0,  # Large rate for test
        )
        weights = PolicyWeights(success_count=0.01)

        # Very negative feedback
        feedback = SliceFeedback(
            slice_name="core",
            verified_count=0,
            attempted_count=100,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # Success weight should not go below floor
        assert result.weights_after.success_count >= 0.0

    def test_weight_clamping(self) -> None:
        """Test that weights are clamped to prevent explosion."""
        updater = PolicyUpdater(
            seed=42,
            weight_clamp=5.0,
            base_learning_rate=10.0,  # Very large for test
        )
        weights = PolicyWeights()

        # Very positive feedback
        feedback = SliceFeedback(
            slice_name="core",
            verified_count=100,
            attempted_count=100,
            target_threshold=7,
        )

        result = updater.update(weights, feedback)

        # All weights should be within clamp bounds
        for value in result.weights_after.to_list():
            assert -5.0 <= value <= 5.0

    def test_update_from_verified_count(self) -> None:
        """Test convenience method."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        result = updater.update_from_verified_count(
            weights=weights,
            slice_name="core",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )

        assert result.reward_signal == 3.0  # 10 - 7
        assert result.feedback_slice == "core"

    def test_deterministic_timestamp(self) -> None:
        """Test that timestamps are deterministic."""
        updater = PolicyUpdater(seed=42)
        weights = PolicyWeights()

        feedback = SliceFeedback(
            slice_name="core",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )

        result1 = updater.update(weights, feedback)
        ts1 = result1.timestamp

        # Reset updater
        updater = PolicyUpdater(seed=42)
        result2 = updater.update(weights, feedback)
        ts2 = result2.timestamp

        assert ts1 == ts2


class TestPolicyUpdateResult:
    """Tests for PolicyUpdateResult."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = PolicyUpdateResult(
            weights_before=PolicyWeights(),
            weights_after=PolicyWeights(len=-0.01),
            update_delta={"len": -0.01},
            update_magnitude=0.01,
            learning_rate=0.01,
            feedback_slice="core",
            reward_signal=3.0,
            features_updated=["len"],
            timestamp="2025-01-01T00:00:00Z",
        )

        d = result.to_dict()
        assert "weights_before" in d
        assert "weights_after" in d
        assert d["update_magnitude"] == 0.01
        assert d["feedback_slice"] == "core"


class TestSliceFeatureMasks:
    """Tests for slice feature masks."""

    def test_warmup_mask(self) -> None:
        """Test warmup mask contains basic structural features."""
        mask = SLICE_FEATURE_MASKS["warmup"]
        assert "len" in mask
        assert "depth" in mask
        assert "connectives" in mask
        # Warmup shouldn't include success history
        assert "success_count" not in mask

    def test_core_mask(self) -> None:
        """Test core mask contains all features."""
        mask = SLICE_FEATURE_MASKS["core"]
        for feature in ["len", "depth", "connectives", "negations",
                        "overlap", "goal_flag", "success_count", "chain_depth"]:
            assert feature in mask

    def test_refinement_mask(self) -> None:
        """Test refinement mask focuses on success/overlap."""
        mask = SLICE_FEATURE_MASKS["refinement"]
        assert "overlap" in mask
        assert "success_count" in mask
        # Refinement shouldn't focus on basic structure
        assert "len" not in mask


@pytest.mark.determinism
class TestUpdateDeterminism:
    """Tests verifying update determinism for RFL Law compliance."""

    def test_update_determinism_100_iterations(self) -> None:
        """
        Verify policy update is deterministic across 100 iterations.

        This is a core RFL Law requirement: same input → same output.
        """
        initial_weights = PolicyWeights(
            len=-0.01, depth=0.05, overlap=0.1,
            goal_flag=1.0, success_count=0.05,
        )
        feedback = SliceFeedback(
            slice_name="core",
            verified_count=10,
            attempted_count=15,
            target_threshold=7,
        )

        # Get reference result
        ref_updater = PolicyUpdater(seed=12345)
        ref_result = ref_updater.update(initial_weights, feedback)

        for i in range(100):
            updater = PolicyUpdater(seed=12345)
            result = updater.update(initial_weights, feedback)

            assert result.weights_after.to_dict() == ref_result.weights_after.to_dict(), \
                f"Determinism violation at iteration {i}"
            assert result.update_magnitude == ref_result.update_magnitude
            assert result.timestamp == ref_result.timestamp

    def test_sequence_of_updates_deterministic(self) -> None:
        """Verify sequence of updates is deterministic."""
        feedbacks = [
            SliceFeedback("warmup", 5, 10, 7),
            SliceFeedback("core", 10, 15, 7),
            SliceFeedback("refinement", 8, 12, 7),
        ]

        # Use explicit known weights to avoid dependency on default implementation
        initial_weights = PolicyWeights(
            len=-0.01, depth=0.0, connectives=0.0, negations=-0.005,
            overlap=0.1, goal_flag=1.0, success_count=0.05, chain_depth=0.0,
        )

        def run_sequence(seed: int) -> PolicyWeights:
            updater = PolicyUpdater(seed=seed)
            weights = initial_weights
            for fb in feedbacks:
                result = updater.update(weights, fb)
                weights = result.weights_after
            return weights

        ref_weights = run_sequence(42)

        for _ in range(20):
            weights = run_sequence(42)
            assert weights.to_dict() == ref_weights.to_dict()
