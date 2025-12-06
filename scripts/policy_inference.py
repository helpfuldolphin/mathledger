#!/usr/bin/env python3
"""
Policy inference system for guided derivation.
"""
import json
import os
import hashlib
from typing import List, Tuple, Any, Dict
import numpy as np

class PolicyInference:
    """Policy-guided action ranking for derivation."""

    def __init__(self, policy_data: Dict[str, Any], model_hash: str):
        self.policy_data = policy_data
        self.hash = model_hash
        self.version = policy_data.get("version", "v1")

    @classmethod
    def load(cls, policy_path: str) -> 'PolicyInference':
        """Load policy from file."""
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        # Load policy metadata
        policy_dir = os.path.dirname(policy_path)
        policy_json_path = os.path.join(policy_dir, "policy.json")

        if os.path.exists(policy_json_path):
            with open(policy_json_path, 'r') as f:
                policy_data = json.load(f)
            model_hash = policy_data.get("hash", "unknown")
        else:
            # Fallback: compute hash from binary file
            with open(policy_path, 'rb') as f:
                content = f.read()
            model_hash = hashlib.sha256(content).hexdigest()
            policy_data = {"hash": model_hash, "version": "v1"}

        return cls(policy_data, model_hash)

    def rank_actions(self, candidates: List[Any], state_context: Dict[str, Any] = None) -> List[Tuple[Any, float]]:
        """
        Rank candidate actions by policy score.

        Args:
            candidates: List of candidate actions
            state_context: Current state context for scoring

        Returns:
            List of (action, score) tuples, sorted by score (highest first)
        """
        if not candidates:
            return []

        # Mock scoring: assign random scores for testing
        # In real implementation, this would use the loaded model
        np.random.seed(hash(self.hash) % 2**32)  # Deterministic from policy hash

        scored = []
        for i, action in enumerate(candidates):
            # Mock score based on action properties and policy hash
            base_score = np.random.random()

            # Add some deterministic bias based on action type
            if hasattr(action, 'text'):
                text = str(action.text)
                if '->' in text:  # Implication
                    base_score += 0.1
                if 'p' in text.lower():  # Simple propositions
                    base_score += 0.05

            scored.append((action, float(base_score)))

        # Sort by score (highest first)
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def get_hash(self) -> str:
        """Get policy hash for metrics."""
        return self.hash
