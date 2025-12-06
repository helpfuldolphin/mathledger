"""
Policy Update Rules for RFL
============================

Deterministic policy weight updates based on verifiable feedback.

Policy updates use:
- Slice-specific success metrics
- Feature relevance masking
- Update magnitude logging

All updates are deterministic and use only formal verification outcomes
(proof success/failure), not human preferences or proxy metrics.

Usage:
    from rfl.policy.update import PolicyUpdater

    updater = PolicyUpdater(seed=42)
    result = updater.update(weights, feedback)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from substrate.repro.determinism import SeededRNG, deterministic_timestamp

from .features import FeatureVector
from .scoring import PolicyWeights

logger = logging.getLogger(__name__)


@dataclass
class SliceFeedback:
    """
    Feedback from a derivation slice for policy update.

    This captures verifiable outcomes only (no human preferences).
    """
    slice_name: str
    verified_count: int           # Number of successfully verified proofs
    attempted_count: int          # Number of proof attempts
    target_threshold: int         # Success threshold for the slice

    # Optional: feature-level signals
    successful_features: Optional[List[FeatureVector]] = None
    failed_features: Optional[List[FeatureVector]] = None

    @property
    def success_rate(self) -> float:
        """Compute success rate (0.0 to 1.0)."""
        if self.attempted_count <= 0:
            return 0.0
        return self.verified_count / self.attempted_count

    @property
    def reward_signal(self) -> float:
        """
        Compute reward signal based on threshold comparison.

        Positive when verified_count > target_threshold.
        Negative when verified_count < target_threshold.
        Zero when exactly at threshold.
        """
        return float(self.verified_count - self.target_threshold)

    @property
    def is_success(self) -> bool:
        """Whether this feedback indicates success (met threshold)."""
        return self.verified_count >= self.target_threshold


@dataclass
class PolicyUpdateResult:
    """
    Result of a policy update operation.

    Captures the update for audit and determinism verification.
    """
    weights_before: PolicyWeights
    weights_after: PolicyWeights
    update_delta: Dict[str, float]
    update_magnitude: float          # L2 norm of update vector
    learning_rate: float
    feedback_slice: str
    reward_signal: float
    features_updated: List[str]      # Which features were updated
    timestamp: str                   # Deterministic timestamp

    def to_dict(self) -> Dict:
        """Serialize for logging and audit."""
        return {
            "weights_before": self.weights_before.to_dict(),
            "weights_after": self.weights_after.to_dict(),
            "update_delta": self.update_delta,
            "update_magnitude": self.update_magnitude,
            "learning_rate": self.learning_rate,
            "feedback_slice": self.feedback_slice,
            "reward_signal": self.reward_signal,
            "features_updated": self.features_updated,
            "timestamp": self.timestamp,
        }


# Feature relevance masks by slice type
# Defines which features are updated for each slice category
SLICE_FEATURE_MASKS: Dict[str, Set[str]] = {
    # Warmup slices: focus on basic structural features
    "warmup": {"len", "depth", "connectives"},

    # Core slices: full feature set
    "core": {
        "len", "depth", "connectives", "negations",
        "overlap", "goal_flag", "success_count", "chain_depth"
    },

    # Refinement slices: focus on success history and overlap
    "refinement": {"overlap", "goal_flag", "success_count", "chain_depth"},

    # Tail/default: all features
    "tail": {
        "len", "depth", "connectives", "negations",
        "overlap", "goal_flag", "success_count", "chain_depth"
    },

    # Default for unknown slices
    "default": {
        "len", "depth", "connectives", "negations",
        "overlap", "goal_flag", "success_count", "chain_depth"
    },
}

# Default scaling factors for success_count updates
# Success reinforcement is stronger than failure penalty to encourage exploration
SUCCESS_COUNT_SUCCESS_SCALE = 0.1   # Scale factor when reward > 0
SUCCESS_COUNT_FAILURE_SCALE = 0.01  # Scale factor when reward < 0


class PolicyUpdater:
    """
    Deterministic policy weight updater.

    Updates weights based on verifiable feedback using:
    - Gradient-free update rules (sign-based or proportional)
    - Slice-specific feature masks
    - Deterministic learning rate
    """

    def __init__(
        self,
        seed: int = 42,
        base_learning_rate: float = 0.01,
        min_learning_rate: float = 0.001,
        max_learning_rate: float = 0.1,
        weight_clamp: float = 10.0,
        success_weight_floor: float = 0.0,
        success_count_success_scale: float = SUCCESS_COUNT_SUCCESS_SCALE,
        success_count_failure_scale: float = SUCCESS_COUNT_FAILURE_SCALE,
    ):
        """
        Initialize policy updater.

        Args:
            seed: Random seed for deterministic behavior.
            base_learning_rate: Base learning rate for updates.
            min_learning_rate: Minimum learning rate.
            max_learning_rate: Maximum learning rate.
            weight_clamp: Maximum absolute value for any weight.
            success_weight_floor: Minimum value for success_count weight
                                 (prevents penalizing successful formulas).
            success_count_success_scale: Scale factor for success_count when reward > 0.
            success_count_failure_scale: Scale factor for success_count when reward < 0.
        """
        self.seed = seed
        self.rng = SeededRNG(seed)
        self.base_learning_rate = base_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.weight_clamp = weight_clamp
        self.success_weight_floor = success_weight_floor
        self.success_count_success_scale = success_count_success_scale
        self.success_count_failure_scale = success_count_failure_scale
        self._update_count = 0

    def _get_feature_mask(self, slice_name: str) -> Set[str]:
        """Get the feature mask for a slice."""
        # Try exact match first
        if slice_name in SLICE_FEATURE_MASKS:
            return SLICE_FEATURE_MASKS[slice_name]

        # Try prefix match (e.g., "warmup_1" matches "warmup")
        for prefix, mask in SLICE_FEATURE_MASKS.items():
            if slice_name.startswith(prefix):
                return mask

        return SLICE_FEATURE_MASKS["default"]

    def _compute_update_direction(
        self,
        feedback: SliceFeedback,
    ) -> Dict[str, float]:
        """
        Compute update direction for each feature.

        Uses heuristics based on verifiable feedback:
        - Positive reward: reinforce current direction
        - Negative reward: reverse direction
        """
        reward = feedback.reward_signal
        update_sign = 1.0 if reward > 0 else (-1.0 if reward < 0 else 0.0)

        # Base update directions (determined by reward sign)
        directions = {
            # Shorter formulas are generally better
            "len": -0.1 * update_sign,

            # Moderate depth is preferred
            "depth": 0.05 * update_sign,

            # Fewer connectives may be cleaner
            "connectives": -0.02 * update_sign,

            # Fewer negations often clearer
            "negations": -0.03 * update_sign,

            # More overlap with target is good
            "overlap": 0.1 * update_sign,

            # Goal formulas are important
            "goal_flag": 0.2 * update_sign,

            # Success history should be reinforced
            # Use configurable scale factors for success_count updates
            "success_count": (
                reward * self.success_count_success_scale
                if reward > 0
                else reward * self.success_count_failure_scale
            ),

            # Chain depth depends on strategy
            "chain_depth": 0.01 * update_sign,
        }

        return directions

    def _compute_learning_rate(self, feedback: SliceFeedback) -> float:
        """
        Compute adaptive learning rate based on feedback magnitude.

        Larger deviations from target get larger updates.
        """
        reward_magnitude = abs(feedback.reward_signal)

        # Scale learning rate based on reward magnitude
        # Cap at 2x for stability
        scale = min(2.0, 1.0 + reward_magnitude * 0.1)

        lr = self.base_learning_rate * scale
        return max(self.min_learning_rate, min(self.max_learning_rate, lr))

    def update(
        self,
        weights: PolicyWeights,
        feedback: SliceFeedback,
    ) -> PolicyUpdateResult:
        """
        Update policy weights based on slice feedback.

        Args:
            weights: Current policy weights.
            feedback: Verifiable feedback from the slice.

        Returns:
            PolicyUpdateResult with old/new weights and update metadata.
        """
        # Get feature mask for this slice
        feature_mask = self._get_feature_mask(feedback.slice_name)

        # Compute update directions
        directions = self._compute_update_direction(feedback)

        # Compute learning rate
        lr = self._compute_learning_rate(feedback)

        # Apply masked update
        weights_dict = weights.to_dict()
        update_delta: Dict[str, float] = {}
        features_updated: List[str] = []

        for feature_name, direction in directions.items():
            if feature_name not in feature_mask:
                update_delta[feature_name] = 0.0
                continue

            delta = lr * direction
            update_delta[feature_name] = delta
            features_updated.append(feature_name)

            new_value = weights_dict[feature_name] + delta

            # Clamp to prevent explosion
            new_value = max(-self.weight_clamp, min(self.weight_clamp, new_value))

            # Special handling for success_count: enforce floor
            if feature_name == "success_count":
                new_value = max(self.success_weight_floor, new_value)

            weights_dict[feature_name] = new_value

        # Create new weights
        new_weights = PolicyWeights.from_dict(weights_dict)

        # Compute update magnitude (L2 norm)
        magnitude = math.sqrt(sum(d**2 for d in update_delta.values()))

        # Deterministic timestamp
        timestamp = deterministic_timestamp(self.seed + self._update_count).isoformat() + "Z"
        self._update_count += 1

        result = PolicyUpdateResult(
            weights_before=weights,
            weights_after=new_weights,
            update_delta=update_delta,
            update_magnitude=magnitude,
            learning_rate=lr,
            feedback_slice=feedback.slice_name,
            reward_signal=feedback.reward_signal,
            features_updated=features_updated,
            timestamp=timestamp,
        )

        # Log update for observability
        logger.info(
            f"[RFL] Policy update: slice={feedback.slice_name}, "
            f"reward={feedback.reward_signal:.3f}, lr={lr:.4f}, "
            f"magnitude={magnitude:.6f}, features={features_updated}"
        )

        return result

    def update_from_verified_count(
        self,
        weights: PolicyWeights,
        slice_name: str,
        verified_count: int,
        attempted_count: int,
        target_threshold: int = 7,
    ) -> PolicyUpdateResult:
        """
        Convenience method to update from basic verification metrics.

        Args:
            weights: Current policy weights.
            slice_name: Name of the curriculum slice.
            verified_count: Number of successful verifications.
            attempted_count: Number of attempted verifications.
            target_threshold: Success threshold (default 7).

        Returns:
            PolicyUpdateResult with update details.
        """
        feedback = SliceFeedback(
            slice_name=slice_name,
            verified_count=verified_count,
            attempted_count=attempted_count,
            target_threshold=target_threshold,
        )
        return self.update(weights, feedback)


__all__ = [
    "SliceFeedback",
    "PolicyUpdateResult",
    "PolicyUpdater",
    "SLICE_FEATURE_MASKS",
    "SUCCESS_COUNT_SUCCESS_SCALE",
    "SUCCESS_COUNT_FAILURE_SCALE",
]
