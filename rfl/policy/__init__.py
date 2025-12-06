"""
RFL Policy Module
=================

Feature extraction, scoring, and update rules for Reflexive Formal Learning.

All policy operations are deterministic (via SeededRNG) and use only
verifiable feedback (proof success/failure), not human preferences.
"""

from .features import (
    FeatureVector,
    extract_features,
    extract_features_batch,
)
from .scoring import (
    PolicyScorer,
    PolicyWeights,
    ScoredCandidate,
    score_candidate,
    score_candidates,
)
from .update import (
    PolicyUpdater,
    PolicyUpdateResult,
    SliceFeedback,
    SLICE_FEATURE_MASKS,
)

__all__ = [
    # Features
    "FeatureVector",
    "extract_features",
    "extract_features_batch",
    # Scoring
    "PolicyScorer",
    "PolicyWeights",
    "ScoredCandidate",
    "score_candidate",
    "score_candidates",
    # Update
    "PolicyUpdater",
    "PolicyUpdateResult",
    "SliceFeedback",
    "SLICE_FEATURE_MASKS",
]

