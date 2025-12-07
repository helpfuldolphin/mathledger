"""
RFL Update Algebra Module
==========================

Implements the formal update algebra ⊕ for Reflexive Formal Learning.

The update algebra defines deterministic policy evolution:
    π_{t+1} = π_t ⊕ η_t Φ(V(e_t), π_t)

Where:
    - π_t: Policy state at epoch t
    - η_t: Step-size schedule at epoch t
    - Φ: Gradient function based on epistemic risk functional J(π)
    - V: Value function over dual-attested events e_t
    - ⊕: Update composition operator (this module)

Invariants:
    1. Determinism: Same inputs → same output (no randomness)
    2. Commutativity: Not required (order matters for learning)
    3. Associativity: Not required (sequential updates)
    4. Identity: π ⊕ 0 = π (zero update preserves policy)
    5. Invertibility: Not required (learning is irreversible)

Usage:
    from rfl.update_algebra import PolicyState, PolicyUpdate, apply_update

    policy = PolicyState(weights={"len": 0.0, "depth": 0.0, "success": 0.0})
    update = PolicyUpdate(deltas={"len": -0.1, "depth": 0.05}, step_size=0.1)
    new_policy = apply_update(policy, update)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass(frozen=True)
class PolicyState:
    """
    Immutable policy state π_t at epoch t.
    
    Represents the current policy weights that control candidate scoring
    during derivation search. Each weight corresponds to a feature used
    to rank candidate formulas.
    
    Attributes:
        weights: Dictionary mapping feature names to weight values
        epoch: Epoch number (0-indexed)
        timestamp: Deterministic timestamp of policy creation
        parent_hash: SHA256 hash of previous policy state (None for π_0)
    """
    weights: Dict[str, float]
    epoch: int = 0
    timestamp: str = ""
    parent_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate policy state invariants."""
        if not self.weights:
            raise ValueError("PolicyState must have at least one weight")
        
        # Ensure all weights are finite
        for key, value in self.weights.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight '{key}' must be numeric, got {type(value)}")
            if not (-1e10 < value < 1e10):
                raise ValueError(f"Weight '{key}' out of bounds: {value}")
    
    def hash(self) -> str:
        """
        Compute deterministic SHA256 hash of policy state.
        
        This hash serves as the policy identifier and enables
        verification of policy evolution chains.
        """
        # Canonical JSON serialization (sorted keys, no whitespace)
        canonical = json.dumps(
            {
                "weights": dict(sorted(self.weights.items())),
                "epoch": self.epoch,
                "timestamp": self.timestamp,
                "parent_hash": self.parent_hash,
            },
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "weights": self.weights,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "parent_hash": self.parent_hash,
            "hash": self.hash(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyState:
        """Deserialize from dictionary."""
        return cls(
            weights=data["weights"],
            epoch=data.get("epoch", 0),
            timestamp=data.get("timestamp", ""),
            parent_hash=data.get("parent_hash"),
        )


@dataclass(frozen=True)
class PolicyUpdate:
    """
    Symbolic policy update Δπ = η_t Φ(V(e_t), π_t).
    
    Represents the gradient-based update to be applied to the current
    policy state. The update is scaled by the step-size schedule η_t.
    
    Attributes:
        deltas: Dictionary mapping feature names to update magnitudes
        step_size: Step-size η_t from schedule
        gradient_norm: L2 norm of unscaled gradient ||Φ||
        source_event_hash: Hash of dual-attested event that triggered update
        metadata: Additional context (abstention_rate, verified_count, etc.)
    """
    deltas: Dict[str, float]
    step_size: float
    gradient_norm: float = 0.0
    source_event_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate update invariants."""
        if not self.deltas:
            raise ValueError("PolicyUpdate must have at least one delta")
        
        if not (0.0 <= self.step_size <= 1.0):
            raise ValueError(f"Step size must be in [0, 1], got {self.step_size}")
        
        # Ensure all deltas are finite
        for key, value in self.deltas.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Delta '{key}' must be numeric, got {type(value)}")
            if not (-1e10 < value < 1e10):
                raise ValueError(f"Delta '{key}' out of bounds: {value}")
    
    def scaled_deltas(self) -> Dict[str, float]:
        """Return deltas scaled by step_size: η_t * Δ."""
        return {key: value * self.step_size for key, value in self.deltas.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "deltas": self.deltas,
            "step_size": self.step_size,
            "gradient_norm": self.gradient_norm,
            "source_event_hash": self.source_event_hash,
            "metadata": self.metadata,
            "scaled_deltas": self.scaled_deltas(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def apply_update(
    policy: PolicyState,
    update: PolicyUpdate,
    deterministic_timestamp: str,
    constraints: Optional[Dict[str, tuple]] = None,
) -> PolicyState:
    """
    Apply update algebra: π_{t+1} = π_t ⊕ Δπ.
    
    This is the core ⊕ operator that implements deterministic policy evolution.
    The update is applied element-wise to matching features, with optional
    constraints to enforce bounds on weight values.
    
    Args:
        policy: Current policy state π_t
        update: Policy update Δπ = η_t Φ(V(e_t), π_t)
        deterministic_timestamp: Timestamp for new policy state
        constraints: Optional bounds per feature, e.g. {"success": (0.0, 10.0)}
    
    Returns:
        New policy state π_{t+1}
    
    Raises:
        ValueError: If update contains features not in policy
    
    Example:
        >>> policy = PolicyState(weights={"len": 0.0, "depth": 0.0})
        >>> update = PolicyUpdate(deltas={"len": -0.1, "depth": 0.05}, step_size=0.1)
        >>> new_policy = apply_update(policy, update, "2025-01-01T00:00:00Z")
        >>> new_policy.weights
        {"len": -0.01, "depth": 0.005}
    """
    # Validate that all update deltas correspond to existing features
    unknown_features = set(update.deltas.keys()) - set(policy.weights.keys())
    if unknown_features:
        raise ValueError(
            f"Update contains unknown features: {unknown_features}. "
            f"Policy features: {set(policy.weights.keys())}"
        )
    
    # Compute new weights: w_{t+1} = w_t + η_t * Δw
    scaled_deltas = update.scaled_deltas()
    new_weights = {}
    
    for feature, old_weight in policy.weights.items():
        delta = scaled_deltas.get(feature, 0.0)
        new_weight = old_weight + delta
        
        # Apply constraints if specified
        if constraints and feature in constraints:
            lower, upper = constraints[feature]
            new_weight = max(lower, min(upper, new_weight))
        
        new_weights[feature] = new_weight
    
    # Construct new policy state with incremented epoch
    return PolicyState(
        weights=new_weights,
        epoch=policy.epoch + 1,
        timestamp=deterministic_timestamp,
        parent_hash=policy.hash(),
    )


def compute_gradient_norm(deltas: Dict[str, float]) -> float:
    """
    Compute L2 norm of gradient: ||Φ|| = sqrt(Σ Δ_i^2).
    
    This provides a scalar measure of update magnitude, useful for
    monitoring convergence and detecting instability.
    
    Args:
        deltas: Dictionary of unscaled gradient components
    
    Returns:
        L2 norm of gradient vector
    """
    return sum(delta ** 2 for delta in deltas.values()) ** 0.5


def zero_update(step_size: float = 0.0) -> PolicyUpdate:
    """
    Create a zero update (identity element).
    
    Applying a zero update leaves the policy unchanged: π ⊕ 0 = π.
    This is useful for epochs where no update is triggered.
    
    Args:
        step_size: Step size (typically 0.0 for identity)
    
    Returns:
        PolicyUpdate with empty deltas
    """
    return PolicyUpdate(
        deltas={},
        step_size=step_size,
        gradient_norm=0.0,
    )


def is_zero_update(update: PolicyUpdate, epsilon: float = 1e-9) -> bool:
    """
    Check if update is effectively zero (within numerical tolerance).
    
    Args:
        update: Policy update to check
        epsilon: Numerical tolerance for zero comparison
    
    Returns:
        True if update magnitude is below epsilon
    """
    return compute_gradient_norm(update.deltas) * update.step_size < epsilon


# -----------------------------------------------------------------------------
# Policy Evolution Chain
# -----------------------------------------------------------------------------

@dataclass
class PolicyEvolutionChain:
    """
    Complete history of policy evolution across epochs.
    
    Maintains a verifiable chain of policy states, enabling:
    - Deterministic replay from initial state
    - Verification of policy evolution integrity
    - Audit trail for governance and debugging
    
    Attributes:
        states: Ordered list of policy states [π_0, π_1, ..., π_t]
        updates: Ordered list of updates [Δπ_0, Δπ_1, ..., Δπ_{t-1}]
        metadata: Additional context (experiment_id, config, etc.)
    """
    states: List[PolicyState] = field(default_factory=list)
    updates: List[PolicyUpdate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate chain invariants."""
        if self.states and self.updates:
            if len(self.states) != len(self.updates) + 1:
                raise ValueError(
                    f"Chain invariant violated: {len(self.states)} states "
                    f"requires {len(self.states) - 1} updates, got {len(self.updates)}"
                )
    
    def append(
        self,
        update: PolicyUpdate,
        deterministic_timestamp: str,
        constraints: Optional[Dict[str, tuple]] = None,
    ) -> PolicyState:
        """
        Apply update and append new state to chain.
        
        Args:
            update: Policy update to apply
            deterministic_timestamp: Timestamp for new state
            constraints: Optional weight bounds
        
        Returns:
            New policy state π_{t+1}
        
        Raises:
            ValueError: If chain is empty (no initial state)
        """
        if not self.states:
            raise ValueError("Cannot append update to empty chain. Initialize with π_0 first.")
        
        current_policy = self.states[-1]
        new_policy = apply_update(current_policy, update, deterministic_timestamp, constraints)
        
        self.states.append(new_policy)
        self.updates.append(update)
        
        return new_policy
    
    def verify_chain(self) -> tuple[bool, List[str]]:
        """
        Verify integrity of policy evolution chain.
        
        Checks:
        1. Each state's parent_hash matches previous state's hash
        2. Epochs increment sequentially
        3. Updates can be replayed to reconstruct states
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.states:
            return True, []  # Empty chain is trivially valid
        
        # Check initial state
        if self.states[0].parent_hash is not None:
            errors.append(f"Initial state π_0 should have parent_hash=None, got {self.states[0].parent_hash}")
        
        if self.states[0].epoch != 0:
            errors.append(f"Initial state π_0 should have epoch=0, got {self.states[0].epoch}")
        
        # Check chain links
        for i in range(1, len(self.states)):
            prev_state = self.states[i - 1]
            curr_state = self.states[i]
            
            # Verify parent hash
            if curr_state.parent_hash != prev_state.hash():
                errors.append(
                    f"State {i} parent_hash mismatch: "
                    f"expected {prev_state.hash()}, got {curr_state.parent_hash}"
                )
            
            # Verify epoch increment
            if curr_state.epoch != prev_state.epoch + 1:
                errors.append(
                    f"State {i} epoch should be {prev_state.epoch + 1}, got {curr_state.epoch}"
                )
        
        # Verify updates can reconstruct states
        if len(self.updates) > 0:
            for i, update in enumerate(self.updates):
                expected_state = self.states[i + 1]
                reconstructed = apply_update(
                    self.states[i],
                    update,
                    expected_state.timestamp,
                )
                
                if reconstructed.hash() != expected_state.hash():
                    errors.append(
                        f"Update {i} replay mismatch: "
                        f"expected hash {expected_state.hash()}, got {reconstructed.hash()}"
                    )
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize chain to dictionary."""
        return {
            "states": [state.to_dict() for state in self.states],
            "updates": [update.to_dict() for update in self.updates],
            "metadata": self.metadata,
            "chain_length": len(self.states),
            "verified": self.verify_chain()[0],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize chain to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    def save(self, filepath: str) -> None:
        """Save chain to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> PolicyEvolutionChain:
        """Load chain from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        states = [PolicyState.from_dict(s) for s in data["states"]]
        updates = [
            PolicyUpdate(
                deltas=u["deltas"],
                step_size=u["step_size"],
                gradient_norm=u.get("gradient_norm", 0.0),
                source_event_hash=u.get("source_event_hash"),
                metadata=u.get("metadata", {}),
            )
            for u in data["updates"]
        ]
        
        return cls(
            states=states,
            updates=updates,
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "PolicyState",
    "PolicyUpdate",
    "apply_update",
    "compute_gradient_norm",
    "zero_update",
    "is_zero_update",
    "PolicyEvolutionChain",
]
