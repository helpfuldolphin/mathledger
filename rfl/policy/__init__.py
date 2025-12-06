"""
RFL Policy Module — PHASE II

Provides slice-aware feature extraction, candidate scoring, and policy updates
for the Reflexive Formal Learning loop.

Absolute Safeguards:
- All policy updates use SeededRNG for reproducibility
- No wall-clock time or external entropy in policy computation
- Policy state is serializable and replayable
- Determinism contract: same seed → same policy trajectory

NOTE: PHASE II — NOT USED IN PHASE I
"""

from rfl.policy.features import (
    extract_features,
    SLICE_FEATURE_MASKS,
    apply_feature_mask,
    get_feature_mask,
    FeatureVector,
)
from rfl.policy.scoring import (
    score_candidates,
    PolicyScorer,
)
from rfl.policy.update import (
    PolicyStateSnapshot,
    PolicyUpdater,
    compute_policy_update,
    LearningScheduleConfig,
)

__all__ = [
    # Features
    "extract_features",
    "SLICE_FEATURE_MASKS",
    "apply_feature_mask",
    "get_feature_mask",
    "FeatureVector",
    # Scoring
    "score_candidates",
    "PolicyScorer",
    # Update
    "PolicyStateSnapshot",
    "PolicyUpdater",
    "compute_policy_update",
    "LearningScheduleConfig",
]
