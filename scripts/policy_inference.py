#!/usr/bin/env python3
"""
Policy inference system for guided derivation.
"""
import json
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
import numpy as np

POLICY_MANIFEST_FILE = "policy.manifest.json"
LEGACY_METADATA_FILE = "policy.json"


class PolicyInference:
    """Policy-guided action ranking for derivation."""

    def __init__(self, policy_data: Dict[str, Any], model_hash: str):
        self.policy_data = policy_data
        self.hash = model_hash
        self.version = policy_data.get("version", "v1")

    @classmethod
    def load(cls, policy_path: str) -> 'PolicyInference':
        """Load policy metadata from weights, manifest, or legacy JSON."""
        resolved_path = Path(policy_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        metadata, model_hash = _load_metadata(resolved_path)
        return cls(metadata, model_hash)

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


def _load_metadata(path: Path) -> Tuple[Dict[str, Any], str]:
    """Load manifest or legacy metadata, computing a hash if needed."""
    manifest_path = _find_manifest(path)
    if manifest_path:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        policy_section = manifest.get("policy", {})
        model_hash = policy_section.get("hash", "unknown")
        version = policy_section.get("version", manifest.get("version", "v1"))
        policy_section.setdefault("version", version)
        return policy_section, model_hash

    base_dir = path.parent if path.is_file() else path
    legacy_path = base_dir / LEGACY_METADATA_FILE
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        model_hash = data.get("hash", "unknown")
        return data, model_hash

    # Fall back to hashing the weights directly.
    model_hash = _sha256(path)
    return {"hash": model_hash, "version": "v1"}, model_hash


def _find_manifest(path: Path) -> Optional[Path]:
    """Locate manifest file relative to supplied policy path."""
    if path.is_file() and path.name == POLICY_MANIFEST_FILE:
        return path
    base_dir = path.parent if path.is_file() else path
    candidate = base_dir / POLICY_MANIFEST_FILE
    if candidate.exists():
        return candidate
    return None


def _sha256(path: Path) -> str:
    """Return SHA-256 hash of the provided file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
