"""
RFL Policy Update — PHASE II

Provides deterministic policy update logic and state snapshot tracking.

Key Components:
- PolicyStateSnapshot: Captures policy state at a point in time
- PolicyUpdater: Computes deterministic weight updates based on verified feedback
- compute_policy_update: Convenience function for single-step updates

Invariants:
- All updates use SeededRNG for reproducibility
- No wall-clock time or external entropy in policy computation
- Policy state is serializable and replayable
- Determinism contract: same seed + same inputs → same output

NOTE: PHASE II — NOT USED IN PHASE I
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from substrate.repro.determinism import SeededRNG, deterministic_timestamp

from rfl.policy.features import FEATURE_NAMES, NUM_FEATURES


@dataclass
class PolicyStateSnapshot:
    """
    Captures policy state at a point in time for logging and replay.

    This dataclass is designed to be:
    - Serializable to JSON
    - Deterministic (no timestamps unless explicitly provided)
    - Suitable for periodic logging in JSONL format
    """

    # Core policy state
    weights: Dict[str, float]
    learning_rate: float
    step_size: float

    # Context
    slice_name: str
    update_count: int

    # Optional metadata
    cycle_index: Optional[int] = None
    reward: Optional[float] = None
    verified_count: Optional[int] = None

    # Deterministic snapshot ID (computed from state)
    snapshot_id: str = field(default="")

    def __post_init__(self) -> None:
        """Compute snapshot_id if not provided."""
        if not self.snapshot_id:
            self.snapshot_id = self._compute_snapshot_id()

    def _compute_snapshot_id(self) -> str:
        """Compute deterministic snapshot ID from state."""
        # Sort weights for determinism
        sorted_weights = json.dumps(
            dict(sorted(self.weights.items())),
            sort_keys=True,
            separators=(',', ':')
        )
        material = f"{sorted_weights}|{self.learning_rate}|{self.step_size}|{self.slice_name}|{self.update_count}"
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_jsonl_record(self) -> str:
        """Convert to JSONL record string."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PolicyStateSnapshot:
        """Create from dictionary."""
        return cls(**d)

    def verify_determinism(self, other: PolicyStateSnapshot) -> bool:
        """Verify that two snapshots are identical."""
        return self.snapshot_id == other.snapshot_id


@dataclass
class PolicyUpdateResult:
    """Result of a policy update step."""

    # Updated weights
    new_weights: Dict[str, float]

    # Update metadata
    update_applied: bool
    reward: float
    gradient_norm: float

    # Delta from previous weights
    weight_deltas: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PolicyUpdater:
    """
    Deterministic policy updater using gradient-free optimization.

    Update rule (simplified):
        w_new = w_old + eta * reward * direction

    Where direction is determined by feature importance for the slice.
    """

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
        step_size: float = 0.1,
        slice_name: str = "default",
        seed: int = 42,
    ):
        """
        Initialize the policy updater.

        Args:
            initial_weights: Initial feature weights
            learning_rate: Base learning rate (eta)
            step_size: Step size for updates
            slice_name: Current slice name
            seed: Random seed for deterministic updates
        """
        self.seed = seed
        self.slice_name = slice_name
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.rng = SeededRNG(seed)

        # Initialize weights
        if initial_weights is None:
            self.weights = {name: 0.0 for name in FEATURE_NAMES}
        else:
            self.weights = initial_weights.copy()

        # Track update count
        self.update_count = 0

    def update(
        self,
        reward: float,
        verified_count: int = 0,
        target_verified: int = 7,
        success: bool = False,
        cycle_index: Optional[int] = None,
    ) -> PolicyUpdateResult:
        """
        Apply a single policy update based on verified feedback.

        Args:
            reward: Graded reward signal (verified_count - target)
            verified_count: Number of verified proofs
            target_verified: Target for success
            success: Whether cycle was successful
            cycle_index: Optional cycle index for logging

        Returns:
            PolicyUpdateResult with new weights and update info
        """
        self.update_count += 1

        # Compute update direction based on reward sign
        # Positive reward: reinforce current strategy
        # Negative reward: try different strategy
        weight_deltas: Dict[str, float] = {}
        old_weights = self.weights.copy()

        if reward > 0:
            # Success: reinforce preferences
            update_magnitude = min(abs(reward) * 0.5, 2.0)
            deltas = self._compute_success_direction(update_magnitude)
        elif reward < 0:
            # Failure: try opposite direction
            update_magnitude = min(abs(reward) * 0.5, 2.0)
            deltas = self._compute_failure_direction(update_magnitude)
        else:
            # At threshold: small exploration
            deltas = self._compute_exploration_direction()

        # Apply updates
        for name, delta in deltas.items():
            effective_delta = self.learning_rate * delta
            self.weights[name] = self.weights.get(name, 0.0) + effective_delta
            weight_deltas[name] = effective_delta

        # Clamp success-related weights to non-negative
        # (we never penalize successful hashes)
        if "success_count" in self.weights:
            self.weights["success_count"] = max(0.0, self.weights["success_count"])
        if "success_rate" in self.weights:
            self.weights["success_rate"] = max(0.0, self.weights["success_rate"])

        # Compute gradient norm for logging
        gradient_norm = sum(d**2 for d in weight_deltas.values()) ** 0.5

        return PolicyUpdateResult(
            new_weights=self.weights.copy(),
            update_applied=True,
            reward=reward,
            gradient_norm=gradient_norm,
            weight_deltas=weight_deltas,
        )

    def _compute_success_direction(self, magnitude: float) -> Dict[str, float]:
        """Compute update direction for success case."""
        return {
            "text_length": -0.1 * magnitude,    # Prefer shorter
            "ast_depth": 0.05 * magnitude,       # Slight preference for depth
            "success_count": magnitude,          # Strongly reinforce success
            "success_rate": 0.5 * magnitude,     # Reinforce success rate
            "goal_overlap": 0.2 * magnitude,     # Reinforce goal alignment
            "required_goal": 0.2 * magnitude,
        }

    def _compute_failure_direction(self, magnitude: float) -> Dict[str, float]:
        """Compute update direction for failure case."""
        return {
            "text_length": 0.1 * magnitude,     # Try longer
            "ast_depth": -0.05 * magnitude,      # Try shallower
            "success_count": -0.1 * magnitude,   # Small penalty
            "success_rate": -0.05 * magnitude,
            "candidate_rarity": 0.1 * magnitude, # Try rarer candidates
        }

    def _compute_exploration_direction(self) -> Dict[str, float]:
        """Compute small exploration update."""
        return {
            "text_length": -0.01,
            "ast_depth": 0.005,
            "success_count": 0.05,
        }

    def get_snapshot(
        self,
        cycle_index: Optional[int] = None,
        reward: Optional[float] = None,
        verified_count: Optional[int] = None,
    ) -> PolicyStateSnapshot:
        """
        Create a snapshot of current policy state.

        Args:
            cycle_index: Optional cycle index
            reward: Optional reward value
            verified_count: Optional verified count

        Returns:
            PolicyStateSnapshot capturing current state
        """
        return PolicyStateSnapshot(
            weights=self.weights.copy(),
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            slice_name=self.slice_name,
            update_count=self.update_count,
            cycle_index=cycle_index,
            reward=reward,
            verified_count=verified_count,
        )

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights from dictionary."""
        self.weights = weights.copy()

    def set_learning_rate(self, learning_rate: float) -> None:
        """Update learning rate."""
        self.learning_rate = learning_rate

    def set_slice(self, slice_name: str) -> None:
        """Update current slice."""
        self.slice_name = slice_name


def compute_policy_update(
    weights: Dict[str, float],
    reward: float,
    learning_rate: float = 0.01,
    slice_name: str = "default",
    seed: int = 42,
) -> PolicyUpdateResult:
    """
    Compute a single policy update.

    Convenience function for stateless update computation.

    Args:
        weights: Current feature weights
        reward: Reward signal
        learning_rate: Learning rate
        slice_name: Slice name
        seed: Random seed

    Returns:
        PolicyUpdateResult with new weights
    """
    updater = PolicyUpdater(
        initial_weights=weights,
        learning_rate=learning_rate,
        slice_name=slice_name,
        seed=seed,
    )
    return updater.update(reward=reward)


# ---------------------------------------------------------------------------
# Learning Rate Schedule Support
# ---------------------------------------------------------------------------

@dataclass
class LearningScheduleConfig:
    """Configuration for per-slice learning rate schedules."""

    # Default learning rate
    default_learning_rate: float = 0.01

    # Per-slice learning rates
    slice_learning_rates: Dict[str, float] = field(default_factory=dict)

    # Optional decay schedule
    decay_rate: float = 1.0  # 1.0 = no decay
    decay_steps: int = 100

    def get_learning_rate(
        self,
        slice_name: str,
        step: int = 0,
    ) -> float:
        """
        Get learning rate for a slice at a given step.

        Args:
            slice_name: Slice name
            step: Current step number

        Returns:
            Learning rate value
        """
        # Get base rate for slice
        base_rate = self.slice_learning_rates.get(
            slice_name,
            self.default_learning_rate
        )

        # Apply decay if configured
        if self.decay_rate < 1.0 and step > 0:
            decay_factor = self.decay_rate ** (step / self.decay_steps)
            return base_rate * decay_factor

        return base_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LearningScheduleConfig:
        """Create from dictionary."""
        return cls(**d)

    @classmethod
    def from_yaml_config(cls, config: Dict[str, Any]) -> LearningScheduleConfig:
        """
        Create from YAML config dict.

        Expected format:
            rfl_policy:
              default_learning_rate: 0.01
              slice_learning_rates:
                slice_uplift_goal: 0.015
                slice_uplift_sparse: 0.02
              decay_rate: 0.99
              decay_steps: 50
        """
        policy_config = config.get("rfl_policy", {})
        return cls(
            default_learning_rate=policy_config.get("default_learning_rate", 0.01),
            slice_learning_rates=policy_config.get("slice_learning_rates", {}),
            decay_rate=policy_config.get("decay_rate", 1.0),
            decay_steps=policy_config.get("decay_steps", 100),
        )
