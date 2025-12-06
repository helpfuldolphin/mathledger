"""
RFL Policy Comparison Tool

Compares two policy states and highlights meaningful differences.
Useful for debugging policy drift and understanding update effects.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .policy import PolicyState


@dataclass
class ComparisonResult:
    """
    Result of comparing two policy states.
    
    Attributes:
        slice_name_a: Identifier for first policy
        slice_name_b: Identifier for second policy
        l2_distance: L2 distance between weight vectors
        l1_distance: L1 distance between weight vectors
        num_sign_flips: Number of features that changed sign
        top_k_deltas: Top-K features with largest absolute delta
        num_features_a: Number of features in policy A
        num_features_b: Number of features in policy B
        feature_set_diff: Features that exist in one but not the other
    """
    slice_name_a: str
    slice_name_b: str
    l2_distance: float
    l1_distance: float
    num_sign_flips: int
    top_k_deltas: List[Tuple[str, float, float, float]]  # (feature, weight_a, weight_b, delta)
    num_features_a: int
    num_features_b: int
    feature_set_diff: Dict[str, List[str]]  # {"only_in_a": [...], "only_in_b": [...]}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def format_summary(self) -> str:
        """Format a human-readable summary."""
        lines = []
        lines.append(f"Policy Comparison: {self.slice_name_a} vs {self.slice_name_b}")
        lines.append("=" * 60)
        lines.append(f"L2 distance: {self.l2_distance:.6f}")
        lines.append(f"L1 distance: {self.l1_distance:.6f}")
        lines.append(f"Sign flips: {self.num_sign_flips}")
        lines.append(f"Features in A: {self.num_features_a}")
        lines.append(f"Features in B: {self.num_features_b}")
        
        if self.feature_set_diff["only_in_a"]:
            lines.append(f"Features only in A: {', '.join(self.feature_set_diff['only_in_a'])}")
        if self.feature_set_diff["only_in_b"]:
            lines.append(f"Features only in B: {', '.join(self.feature_set_diff['only_in_b'])}")
        
        lines.append("\nTop features by absolute delta:")
        lines.append("-" * 60)
        lines.append(f"{'Feature':<20} {'Weight A':>12} {'Weight B':>12} {'Delta':>12}")
        lines.append("-" * 60)
        for feature, weight_a, weight_b, delta in self.top_k_deltas:
            lines.append(f"{feature:<20} {weight_a:>12.6f} {weight_b:>12.6f} {delta:>12.6f}")
        
        return "\n".join(lines)


def compare_policy_states(
    state_a: PolicyState,
    state_b: PolicyState,
    slice_name_a: str = "policy_a",
    slice_name_b: str = "policy_b",
    top_k: int = 10,
    handle_missing: str = "error"
) -> ComparisonResult:
    """
    Compare two policy states and compute difference metrics.
    
    Args:
        state_a: First policy state
        state_b: Second policy state
        slice_name_a: Name/identifier for first policy
        slice_name_b: Name/identifier for second policy
        top_k: Number of top features to report by delta
        handle_missing: How to handle mismatched feature sets:
            - "error": Raise ValueError
            - "union": Use union of features (missing = 0.0)
            - "intersection": Use only common features
            
    Returns:
        ComparisonResult with computed metrics
        
    Raises:
        ValueError: If feature sets don't match and handle_missing="error"
    """
    # Get feature sets
    features_a = set(state_a.weights.keys())
    features_b = set(state_b.weights.keys())
    
    only_in_a = sorted(features_a - features_b)
    only_in_b = sorted(features_b - features_a)
    
    # Handle feature set mismatch
    if only_in_a or only_in_b:
        if handle_missing == "error":
            raise ValueError(
                f"Feature sets don't match. "
                f"Only in A: {only_in_a}, Only in B: {only_in_b}"
            )
        elif handle_missing == "union":
            # Use union: missing features have weight 0.0
            all_features = features_a | features_b
        elif handle_missing == "intersection":
            # Use only common features
            all_features = features_a & features_b
        else:
            raise ValueError(f"Invalid handle_missing: {handle_missing}")
    else:
        all_features = features_a
    
    # Build aligned weight vectors
    all_features = sorted(all_features)  # Sort for determinism
    weights_a = np.array([state_a.weights.get(f, 0.0) for f in all_features])
    weights_b = np.array([state_b.weights.get(f, 0.0) for f in all_features])
    
    # Compute distance metrics
    l2_distance = float(np.linalg.norm(weights_a - weights_b, ord=2))
    l1_distance = float(np.linalg.norm(weights_a - weights_b, ord=1))
    
    # Count sign flips
    sign_a = np.sign(weights_a)
    sign_b = np.sign(weights_b)
    num_sign_flips = int(np.sum((sign_a != sign_b) & (sign_a != 0) & (sign_b != 0)))
    
    # Compute deltas and find top-K
    deltas = np.abs(weights_a - weights_b)
    top_k_indices = np.argsort(deltas)[::-1][:top_k]
    
    top_k_deltas = [
        (
            all_features[i],
            float(weights_a[i]),
            float(weights_b[i]),
            float(weights_b[i] - weights_a[i])
        )
        for i in top_k_indices
    ]
    
    return ComparisonResult(
        slice_name_a=slice_name_a,
        slice_name_b=slice_name_b,
        l2_distance=l2_distance,
        l1_distance=l1_distance,
        num_sign_flips=num_sign_flips,
        top_k_deltas=top_k_deltas,
        num_features_a=len(state_a.weights),
        num_features_b=len(state_b.weights),
        feature_set_diff={
            "only_in_a": only_in_a,
            "only_in_b": only_in_b,
        }
    )


def load_policy_from_json(filepath: Path, index: Optional[int] = None) -> PolicyState:
    """
    Load policy state from JSON or JSONL file.
    
    Args:
        filepath: Path to JSON or JSONL file
        index: If JSONL, which line to load (default: last line)
        
    Returns:
        Loaded PolicyState
        
    Raises:
        ValueError: If file format is invalid
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    # Detect format: JSONL if suffix is .jsonl OR if it's multiline and not valid JSON
    is_jsonl = filepath.suffix == '.jsonl'
    
    if not is_jsonl and '\n' in content:
        # Try to parse as JSON first
        try:
            json.loads(content)
            is_jsonl = False
        except json.JSONDecodeError:
            # Not valid JSON, assume JSONL
            is_jsonl = True
    
    if is_jsonl:
        # JSONL format
        lines = [line for line in content.split('\n') if line.strip()]
        if not lines:
            raise ValueError(f"Empty JSONL file: {filepath}")
        
        if index is None:
            index = -1  # Last line by default
        
        try:
            data = json.loads(lines[index])
        except (IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse JSONL line {index}: {e}")
    else:
        # Regular JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    return PolicyState.from_dict(data)
