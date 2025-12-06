# PHASE II — NOT USED IN PHASE I
"""
RFL Policy Module
=================

Phase II policy dynamics, telemetry, and diagnostics.

This module provides:
- PolicyStateSnapshot: Serializable policy state
- PolicyUpdater: Policy update logic with safety guards
- LearningScheduleConfig: Per-slice learning rate configuration
- Policy telemetry and introspection utilities

Determinism Contract:
    Same seed + same config + same inputs => same policy trajectory and telemetry.
    All randomness must go through SeededRNG.
"""

from .update import (
    PolicyStateSnapshot,
    PolicyUpdater,
    LearningScheduleConfig,
    summarize_policy_state,
    init_cold_start,
    init_from_file,
)
from .features import FeatureVector, SLICE_FEATURE_MASKS, extract_features
from .scoring import PolicyScorer, score_candidates

__all__ = [
    "PolicyStateSnapshot",
    "PolicyUpdater",
    "LearningScheduleConfig",
    "summarize_policy_state",
    "init_cold_start",
    "init_from_file",
    "FeatureVector",
    "SLICE_FEATURE_MASKS",
    "extract_features",
    "PolicyScorer",
    "score_candidates",
]
