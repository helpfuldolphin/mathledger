# PHASE II — NOT USED IN PHASE I
"""
RFL Policy Scoring
==================

Score candidates using policy weights and features.

Determinism Contract:
    - All scoring is deterministic
    - Fixed key ordering for weight iteration
    - No external entropy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .features import FeatureVector, extract_features, SLICE_FEATURE_MASKS


@dataclass
class PolicyScorer:
    """
    Score candidates using linear policy weights.
    
    Score = sum(weight_i * feature_i) for active features.
    
    Attributes:
        weights: Dictionary mapping feature names to weights
        slice_name: Current slice name (determines active features)
    """
    weights: Dict[str, float] = field(default_factory=dict)
    slice_name: str = "default"

    def score(self, features: FeatureVector) -> float:
        """
        Score a single feature vector.
        
        Args:
            features: FeatureVector to score
        
        Returns:
            Scalar score value
        """
        active_features = SLICE_FEATURE_MASKS.get(
            self.slice_name,
            SLICE_FEATURE_MASKS["default"]
        )
        
        total = 0.0
        for fname in sorted(active_features):  # Fixed ordering for determinism
            fval = getattr(features, fname, 0.0)
            if isinstance(fval, (int, float)):
                weight = self.weights.get(fname, 0.0)
                total += weight * float(fval)
            elif fname in features.raw:
                weight = self.weights.get(fname, 0.0)
                total += weight * float(features.raw[fname])
        
        return total

    def score_batch(self, features_list: List[FeatureVector]) -> np.ndarray:
        """
        Score a batch of feature vectors.
        
        Args:
            features_list: List of FeatureVectors
        
        Returns:
            1D numpy array of scores
        """
        scores = [self.score(f) for f in features_list]
        return np.array(scores, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": dict(sorted(self.weights.items())),
            "slice_name": self.slice_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyScorer":
        """Create from dictionary."""
        return cls(
            weights=d.get("weights", {}),
            slice_name=d.get("slice_name", "default"),
        )


def score_candidates(
    candidates: List[str],
    scorer: PolicyScorer,
    success_history: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """
    Score and rank candidates using the policy.
    
    Args:
        candidates: List of candidate formula texts
        scorer: PolicyScorer with current weights
        success_history: Optional success rate history by formula hash
    
    Returns:
        List of (candidate, score) tuples, sorted descending by score
    
    Determinism:
        Given same inputs, returns same ordering.
        Ties are broken by lexicographic order of candidate text.
    """
    scored = []
    for text in candidates:
        # Compute deterministic hash for history lookup
        import hashlib
        formula_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        
        features = extract_features(
            text,
            success_history=success_history,
            formula_hash=formula_hash,
        )
        score = scorer.score(features)
        scored.append((text, score, features))
    
    # Sort by score (descending), then by text (ascending) for tie-breaking
    scored.sort(key=lambda x: (-x[1], x[0]))
    
    return [(text, score) for text, score, _ in scored]
