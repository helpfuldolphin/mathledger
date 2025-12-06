#!/usr/bin/env python3
"""
Policy-guided derivation system for MathLedger.
Loads trained policies and scores derivation actions.
"""

from backend.repro.determinism import SeededRNG

_GLOBAL_SEED = 0

import os
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import hashlib


def load_policy(path: str) -> Any:
    """
    Load a trained policy from disk.

    Args:
        path: Path to policy file (.bin, .pkl, etc.)

    Returns:
        Policy object with scoring capabilities

    Raises:
        FileNotFoundError: If policy file doesn't exist
        ValueError: If policy file is corrupted
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy file not found: {path}")

    try:
        with open(path, 'rb') as f:
            policy = pickle.load(f)

        # Validate policy has required methods
        if not hasattr(policy, 'score') and not hasattr(policy, 'predict'):
            raise ValueError("Policy object must have 'score' or 'predict' method")

        return policy

    except Exception as e:
        raise ValueError(f"Failed to load policy from {path}: {e}")


def score_batch(policy: Any, feats: np.ndarray) -> np.ndarray:
    """
    Score a batch of features using the policy.

    Args:
        policy: Loaded policy object
        feats: Feature matrix of shape (n_samples, n_features)

    Returns:
        Array of scores of shape (n_samples,)
    """
    if feats.size == 0:
        return np.array([])

    # Ensure feats is 2D
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    try:
        # Try different scoring methods
        if hasattr(policy, 'score'):
            scores = policy.score(feats)
        elif hasattr(policy, 'predict'):
            scores = policy.predict(feats)
        elif hasattr(policy, 'predict_proba'):
            # For classifiers, use max probability
            proba = policy.predict_proba(feats)
            scores = np.max(proba, axis=1)
        else:
            # Fallback: random scores
            scores = SeededRNG(_GLOBAL_SEED).random(feats.shape[0])

        # Ensure scores are 1D
        if scores.ndim > 1:
            scores = scores.flatten()

        return scores.astype(np.float32)

    except Exception as e:
        # Fallback to random scores if policy fails
        print(f"Warning: Policy scoring failed: {e}, using random scores")
        return SeededRNG(_GLOBAL_SEED).random(feats.shape[0]).astype(np.float32)


def get_policy_hash(policy: Any) -> str:
    """
    Generate a hash for the policy object.

    Args:
        policy: Policy object

    Returns:
        64-character hex hash
    """
    try:
        # Try to get a string representation of the policy
        policy_str = str(policy)
        if hasattr(policy, '__dict__'):
            policy_str += str(sorted(policy.__dict__.items()))
    except:
        policy_str = f"policy_{id(policy)}"

    return hashlib.sha256(policy_str.encode()).hexdigest()


class MockPolicy:
    """
    Mock policy for testing when no real policy is available.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Generate mock scores based on feature complexity."""
        if feats.size == 0:
            return np.array([])

        # Simple heuristic: prefer features with more non-zero values
        complexity = np.sum(feats != 0, axis=1)
        base_scores = complexity / (feats.shape[1] + 1)

        # Add some randomness
        noise = SeededRNG(_GLOBAL_SEED).random(feats.shape[0]) * 0.1
        return base_scores + noise

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """Alias for score method."""
        return self.score(feats)


def create_mock_policy(path: str) -> MockPolicy:
    """
    Create and save a mock policy for testing.

    Args:
        path: Where to save the mock policy

    Returns:
        MockPolicy instance
    """
    policy = MockPolicy()

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(policy, f)

    return policy
