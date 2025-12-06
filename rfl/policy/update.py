# PHASE II — NOT USED IN PHASE I
"""
RFL Policy Update Logic
=======================

Policy state management, updates, and safety guards.

Determinism Contract:
    - Same seed + same config => same policy trajectory
    - All updates use SeededRNG for reproducibility
    - No wall-clock time or external entropy in policy computation
    - Policy state is serializable and replayable

Safety Guards:
    - L2 norm clamping: rescale weights if norm exceeds max_weight_norm_l2
    - Per-weight clipping: clip each weight to ±max_abs_weight
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math

import numpy as np

from substrate.repro.determinism import SeededRNG


@dataclass
class LearningScheduleConfig:
    """
    Per-slice learning rate configuration.
    
    Loaded from config/rfl_policy_phase2.yaml.
    
    Attributes:
        learning_rate: Base learning rate for updates
        decay_factor: Multiplicative decay per update
        min_learning_rate: Minimum learning rate floor
        max_weight_norm_l2: Maximum L2 norm for weight vector
        max_abs_weight: Maximum absolute value for any single weight
    """
    learning_rate: float = 0.01
    decay_factor: float = 0.999
    min_learning_rate: float = 0.0001
    max_weight_norm_l2: float = 10.0
    max_abs_weight: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "learning_rate": self.learning_rate,
            "decay_factor": self.decay_factor,
            "min_learning_rate": self.min_learning_rate,
            "max_weight_norm_l2": self.max_weight_norm_l2,
            "max_abs_weight": self.max_abs_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningScheduleConfig":
        """Create from dictionary."""
        return cls(
            learning_rate=d.get("learning_rate", 0.01),
            decay_factor=d.get("decay_factor", 0.999),
            min_learning_rate=d.get("min_learning_rate", 0.0001),
            max_weight_norm_l2=d.get("max_weight_norm_l2", 10.0),
            max_abs_weight=d.get("max_abs_weight", 5.0),
        )


@dataclass
class PolicyStateSnapshot:
    """
    Serializable snapshot of policy state.
    
    Used for warm-start, checkpointing, and telemetry.
    
    Attributes:
        slice_name: Name of the curriculum slice
        weights: Dictionary of feature weights
        update_count: Number of updates applied
        learning_rate: Current effective learning rate
        seed: RNG seed for reproducibility
        clamped: Whether weights were clamped in last update
        clamp_count: Total number of clamp events
        phase: Phase identifier (always "II" for Phase II)
    """
    slice_name: str
    weights: Dict[str, float] = field(default_factory=dict)
    update_count: int = 0
    learning_rate: float = 0.01
    seed: int = 42
    clamped: bool = False
    clamp_count: int = 0
    phase: str = "II"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "slice_name": self.slice_name,
            "weights": dict(sorted(self.weights.items())),  # Fixed key ordering
            "update_count": self.update_count,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "clamped": self.clamped,
            "clamp_count": self.clamp_count,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyStateSnapshot":
        """Create from dictionary."""
        return cls(
            slice_name=d["slice_name"],
            weights=d.get("weights", {}),
            update_count=d.get("update_count", 0),
            learning_rate=d.get("learning_rate", 0.01),
            seed=d.get("seed", 42),
            clamped=d.get("clamped", False),
            clamp_count=d.get("clamp_count", 0),
            phase=d.get("phase", "II"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "PolicyStateSnapshot":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))


def summarize_policy_state(state: PolicyStateSnapshot) -> Dict[str, Any]:
    """
    Return a small, JSON-serializable summary of the policy state.
    
    Used for telemetry logging.
    
    Args:
        state: PolicyStateSnapshot to summarize
    
    Returns:
        Dictionary with:
            - slice_name: str
            - update_count: int
            - learning_rate: float
            - weight_norm_l1: float
            - weight_norm_l2: float
            - nonzero_weights: int
    
    Determinism:
        Computes norms from weights with fixed key ordering.
        No randomness or external entropy.
    """
    # Extract weight values in deterministic order
    weight_values = [state.weights[k] for k in sorted(state.weights.keys())]
    
    if not weight_values:
        l1_norm = 0.0
        l2_norm = 0.0
        nonzero = 0
    else:
        weight_array = np.array(weight_values, dtype=np.float64)
        l1_norm = float(np.sum(np.abs(weight_array)))
        l2_norm = float(np.sqrt(np.sum(weight_array ** 2)))
        nonzero = int(np.sum(np.abs(weight_array) > 1e-9))
    
    return {
        "slice_name": state.slice_name,
        "update_count": state.update_count,
        "learning_rate": state.learning_rate,
        "weight_norm_l1": l1_norm,
        "weight_norm_l2": l2_norm,
        "nonzero_weights": nonzero,
    }


def init_cold_start(slice_name: str, schedule: LearningScheduleConfig, seed: int = 42) -> PolicyStateSnapshot:
    """
    Initialize a cold-start policy state.
    
    Starts from zero weights with update_count=0.
    
    Args:
        slice_name: Name of the curriculum slice
        schedule: Learning schedule configuration
        seed: RNG seed for reproducibility
    
    Returns:
        Fresh PolicyStateSnapshot with zero weights
    """
    return PolicyStateSnapshot(
        slice_name=slice_name,
        weights={},
        update_count=0,
        learning_rate=schedule.learning_rate,
        seed=seed,
        clamped=False,
        clamp_count=0,
        phase="II",
    )


def init_from_file(path: Path) -> PolicyStateSnapshot:
    """
    Load a policy state from a file.
    
    Supports:
        - Single JSON file: loads directly
        - JSONL file: loads last line
    
    Args:
        path: Path to snapshot file
    
    Returns:
        PolicyStateSnapshot loaded from file
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid or slice_name doesn't match
    """
    if not path.exists():
        raise FileNotFoundError(f"Policy snapshot file not found: {path}")
    
    content = path.read_text(encoding="utf-8").strip()
    
    if not content:
        raise ValueError(f"Empty policy snapshot file: {path}")
    
    # Try to parse as single JSON object
    try:
        d = json.loads(content)
        return PolicyStateSnapshot.from_dict(d)
    except json.JSONDecodeError:
        pass
    
    # Try to parse as JSONL (take last line)
    lines = content.split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
                return PolicyStateSnapshot.from_dict(d)
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"Could not parse policy snapshot from: {path}")


class PolicyUpdater:
    """
    Policy update logic with safety guards.
    
    Implements:
        - Learning rate decay
        - L2 norm clamping
        - Per-weight clipping
        - Deterministic updates via SeededRNG
    
    Attributes:
        schedule: Learning schedule configuration
        state: Current policy state snapshot
        rng: Seeded RNG for reproducibility
    """

    def __init__(
        self,
        schedule: LearningScheduleConfig,
        state: PolicyStateSnapshot,
    ):
        """
        Initialize updater.
        
        Args:
            schedule: Learning schedule configuration
            state: Initial policy state
        """
        self.schedule = schedule
        self.state = state
        self.rng = SeededRNG(state.seed + state.update_count)

    def update(
        self,
        feature_name: str,
        gradient: float,
    ) -> PolicyStateSnapshot:
        """
        Apply a single gradient update to a feature weight.
        
        Update formula:
            w_new = w_old + lr * gradient
        
        Then applies safety guards:
            1. L2 norm clamping
            2. Per-weight clipping
        
        Args:
            feature_name: Name of feature to update
            gradient: Gradient value (positive = increase weight)
        
        Returns:
            Updated PolicyStateSnapshot
        """
        # Get current weight
        current_weight = self.state.weights.get(feature_name, 0.0)
        
        # Apply update
        new_weight = current_weight + self.state.learning_rate * gradient
        
        # Update weights dictionary
        new_weights = dict(self.state.weights)
        new_weights[feature_name] = new_weight
        
        # Apply safety guards
        new_weights, clamped = self._apply_safety_guards(new_weights)
        
        # Decay learning rate
        new_lr = max(
            self.schedule.min_learning_rate,
            self.state.learning_rate * self.schedule.decay_factor
        )
        
        # Create new state
        new_state = PolicyStateSnapshot(
            slice_name=self.state.slice_name,
            weights=new_weights,
            update_count=self.state.update_count + 1,
            learning_rate=new_lr,
            seed=self.state.seed,
            clamped=clamped,
            clamp_count=self.state.clamp_count + (1 if clamped else 0),
            phase=self.state.phase,
        )
        
        self.state = new_state
        return new_state

    def batch_update(
        self,
        gradients: Dict[str, float],
    ) -> PolicyStateSnapshot:
        """
        Apply gradient updates to multiple feature weights.
        
        Args:
            gradients: Dictionary mapping feature names to gradients
        
        Returns:
            Updated PolicyStateSnapshot
        """
        new_weights = dict(self.state.weights)
        
        # Apply all updates (in sorted order for determinism)
        for feature_name in sorted(gradients.keys()):
            gradient = gradients[feature_name]
            current_weight = new_weights.get(feature_name, 0.0)
            new_weights[feature_name] = current_weight + self.state.learning_rate * gradient
        
        # Apply safety guards
        new_weights, clamped = self._apply_safety_guards(new_weights)
        
        # Decay learning rate
        new_lr = max(
            self.schedule.min_learning_rate,
            self.state.learning_rate * self.schedule.decay_factor
        )
        
        # Create new state
        new_state = PolicyStateSnapshot(
            slice_name=self.state.slice_name,
            weights=new_weights,
            update_count=self.state.update_count + 1,
            learning_rate=new_lr,
            seed=self.state.seed,
            clamped=clamped,
            clamp_count=self.state.clamp_count + (1 if clamped else 0),
            phase=self.state.phase,
        )
        
        self.state = new_state
        return new_state

    def _apply_safety_guards(
        self,
        weights: Dict[str, float],
    ) -> tuple[Dict[str, float], bool]:
        """
        Apply safety guards to weight dictionary.
        
        1. L2 norm clamping: rescale if norm > max_weight_norm_l2
        2. Per-weight clipping: clip each weight to ±max_abs_weight
        
        Args:
            weights: Dictionary of weights
        
        Returns:
            Tuple of (clamped_weights, was_clamped)
        """
        if not weights:
            return weights, False
        
        clamped = False
        
        # Extract weights in deterministic order
        keys = sorted(weights.keys())
        values = np.array([weights[k] for k in keys], dtype=np.float64)
        
        # Step 1: L2 norm clamping
        l2_norm = float(np.sqrt(np.sum(values ** 2)))
        if l2_norm > self.schedule.max_weight_norm_l2:
            scale = self.schedule.max_weight_norm_l2 / l2_norm
            values = values * scale
            clamped = True
        
        # Step 2: Per-weight clipping
        max_w = self.schedule.max_abs_weight
        original_values = values.copy()
        values = np.clip(values, -max_w, max_w)
        if not np.array_equal(values, original_values):
            clamped = True
        
        # Rebuild dictionary
        result = {k: float(v) for k, v in zip(keys, values)}
        return result, clamped

    def get_state(self) -> PolicyStateSnapshot:
        """Get current policy state."""
        return self.state

    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary of current state."""
        return summarize_policy_state(self.state)
