"""
RFL Policy Update Logic

Implements gradient-free policy updates using verifiable feedback (proof success/failure).
All randomness uses SeededRNG for deterministic reproduction.

Phase II Implementation with:
- Telemetry snapshots (summarize_policy_state)
- Safety guards (L2 norm clamping + per-weight clipping)
- Warm/cold initialization
- Policy introspection tooling
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from backend.repro.determinism import SeededRNG


@dataclass
class PolicyState:
    """
    Policy state containing feature weights and metadata.
    
    Attributes:
        weights: Feature weights (key: feature_name, value: weight)
        step: Update step counter
        total_reward: Cumulative reward
        update_count: Number of updates applied
        seed: RNG seed for reproducibility
    """
    weights: Dict[str, float]
    step: int = 0
    total_reward: float = 0.0
    update_count: int = 0
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyState":
        """Load from dictionary."""
        return cls(**data)


@dataclass
class FeatureTelemetry:
    """
    Per-feature telemetry statistics.
    
    Attributes:
        top_k_positive: Top-K features with highest positive weights
        top_k_negative: Top-K features with most negative weights
        sparsity: Fraction of features with non-zero weights
        num_features: Total number of features
    """
    top_k_positive: List[Tuple[str, float]]
    top_k_negative: List[Tuple[str, float]]
    sparsity: float
    num_features: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PolicyStateSnapshot:
    """
    Complete policy state snapshot for telemetry and debugging.
    
    Attributes:
        step: Update step
        l2_norm: L2 norm of weight vector
        l1_norm: L1 norm of weight vector
        max_abs_weight: Maximum absolute weight value
        mean_weight: Mean weight value
        num_features: Number of features
        total_reward: Cumulative reward
        feature_telemetry: Optional per-feature statistics
    """
    step: int
    l2_norm: float
    l1_norm: float
    max_abs_weight: float
    mean_weight: float
    num_features: int
    total_reward: float
    feature_telemetry: Optional[FeatureTelemetry] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "step": self.step,
            "l2_norm": self.l2_norm,
            "l1_norm": self.l1_norm,
            "max_abs_weight": self.max_abs_weight,
            "mean_weight": self.mean_weight,
            "num_features": self.num_features,
            "total_reward": self.total_reward,
        }
        if self.feature_telemetry:
            result["feature_telemetry"] = self.feature_telemetry.to_dict()
        return result


def summarize_policy_state(
    state: PolicyState,
    include_feature_telemetry: bool = False,
    top_k: int = 5
) -> PolicyStateSnapshot:
    """
    Generate telemetry snapshot from policy state.
    
    Args:
        state: Current policy state
        include_feature_telemetry: Whether to include per-feature stats
        top_k: Number of top features to track
        
    Returns:
        PolicyStateSnapshot with computed statistics
    """
    weights_array = np.array(list(state.weights.values()), dtype=np.float64)
    
    # Compute basic statistics
    l2_norm = float(np.linalg.norm(weights_array, ord=2))
    l1_norm = float(np.linalg.norm(weights_array, ord=1))
    max_abs_weight = float(np.max(np.abs(weights_array))) if len(weights_array) > 0 else 0.0
    mean_weight = float(np.mean(weights_array)) if len(weights_array) > 0 else 0.0
    
    # Compute feature telemetry if requested
    feature_telemetry = None
    if include_feature_telemetry:
        # Sort features by weight
        sorted_features = sorted(state.weights.items(), key=lambda x: x[1], reverse=True)
        
        # Top-K positive and negative
        top_k_positive = sorted_features[:top_k]
        top_k_negative = sorted_features[-top_k:][::-1]  # Reverse to show most negative first
        
        # Sparsity: fraction of non-zero weights
        nonzero_count = np.count_nonzero(weights_array)
        sparsity = float(nonzero_count) / len(weights_array) if len(weights_array) > 0 else 0.0
        
        feature_telemetry = FeatureTelemetry(
            top_k_positive=top_k_positive,
            top_k_negative=top_k_negative,
            sparsity=sparsity,
            num_features=len(state.weights)
        )
    
    return PolicyStateSnapshot(
        step=state.step,
        l2_norm=l2_norm,
        l1_norm=l1_norm,
        max_abs_weight=max_abs_weight,
        mean_weight=mean_weight,
        num_features=len(state.weights),
        total_reward=state.total_reward,
        feature_telemetry=feature_telemetry
    )


class PolicyUpdater:
    """
    Policy update engine with safety guards and deterministic updates.
    
    Safety features:
    - L2 norm clamping: Prevents weight vector from growing unbounded
    - Per-weight clipping: Prevents individual weights from exploding
    - Deterministic updates: All randomness via SeededRNG
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_weight_norm_l2: float = 10.0,
        max_abs_weight: float = 5.0,
        seed: int = 42
    ):
        """
        Initialize policy updater.
        
        Args:
            learning_rate: Learning rate for weight updates
            max_weight_norm_l2: Maximum L2 norm for weight vector
            max_abs_weight: Maximum absolute value for individual weights
            seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_weight_norm_l2 = max_weight_norm_l2
        self.max_abs_weight = max_abs_weight
        self.seed = seed
        self.rng = SeededRNG(seed)
    
    def update(
        self,
        state: PolicyState,
        features: Dict[str, float],
        reward: float
    ) -> PolicyState:
        """
        Apply a single policy update using gradient-free learning.
        
        Update formula (simple additive):
            w_new[f] = w_old[f] + lr * reward * features[f]
        
        Then apply safety guards:
            1. Per-weight clipping
            2. L2 norm clamping
        
        Args:
            state: Current policy state
            features: Feature vector (key: feature_name, value: feature_value)
            reward: Reward signal (+1 for success, -1 for failure, 0 for abstention)
            
        Returns:
            Updated policy state
        """
        # Create new weights dictionary
        new_weights = dict(state.weights)
        
        # Ensure all features are present in weights
        for feature_name in features:
            if feature_name not in new_weights:
                new_weights[feature_name] = 0.0
        
        # Apply update: w += lr * reward * feature
        for feature_name, feature_value in features.items():
            update_delta = self.learning_rate * reward * feature_value
            new_weights[feature_name] += update_delta
        
        # Safety guard 1: Per-weight clipping
        for feature_name in new_weights:
            new_weights[feature_name] = np.clip(
                new_weights[feature_name],
                -self.max_abs_weight,
                self.max_abs_weight
            )
        
        # Safety guard 2: L2 norm clamping
        weights_array = np.array(list(new_weights.values()), dtype=np.float64)
        current_l2_norm = np.linalg.norm(weights_array, ord=2)
        
        if current_l2_norm > self.max_weight_norm_l2:
            # Scale down to max norm while preserving direction
            scale_factor = self.max_weight_norm_l2 / current_l2_norm
            for feature_name in new_weights:
                new_weights[feature_name] *= scale_factor
        
        # Create updated state
        return PolicyState(
            weights=new_weights,
            step=state.step + 1,
            total_reward=state.total_reward + reward,
            update_count=state.update_count + 1,
            seed=state.seed
        )
    
    def score_candidate(self, state: PolicyState, features: Dict[str, float]) -> float:
        """
        Score a candidate using current policy weights.
        
        Score = sum(w[f] * features[f] for all features f)
        
        Args:
            state: Current policy state
            features: Feature vector
            
        Returns:
            Scalar score
        """
        score = 0.0
        for feature_name, feature_value in features.items():
            weight = state.weights.get(feature_name, 0.0)
            score += weight * feature_value
        return score


def init_cold_start(feature_names: List[str], seed: int = 42) -> PolicyState:
    """
    Initialize policy with zero weights (cold start).
    
    Args:
        feature_names: List of feature names
        seed: Random seed
        
    Returns:
        Fresh policy state with zero weights
    """
    weights = {name: 0.0 for name in feature_names}
    return PolicyState(weights=weights, seed=seed)


def init_from_file(filepath: Path) -> PolicyState:
    """
    Load policy state from JSON file (warm start).
    
    Args:
        filepath: Path to saved policy state
        
    Returns:
        Loaded policy state
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is corrupted
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Policy file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return PolicyState.from_dict(data)
    except Exception as e:
        raise ValueError(f"Failed to load policy from {filepath}: {e}")


def save_policy_state(state: PolicyState, filepath: Path) -> None:
    """
    Save policy state to JSON file.
    
    Args:
        state: Policy state to save
        filepath: Destination path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
