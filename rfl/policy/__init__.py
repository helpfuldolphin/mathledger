"""
RFL Policy Package

Core policy update logic, feature engineering, and reward computation
for Reflexive Formal Learning.
"""

from .policy import (
    PolicyUpdater,
    PolicyState,
    PolicyStateSnapshot,
    init_cold_start,
    init_from_file,
    summarize_policy_state,
    save_policy_state,
)
from .features import extract_features, FeatureVector, get_feature_names
from .rewards import compute_reward, RewardSignal
from .compare import compare_policy_states, ComparisonResult, load_policy_from_json

__all__ = [
    "PolicyUpdater",
    "PolicyState",
    "PolicyStateSnapshot",
    "init_cold_start",
    "init_from_file",
    "summarize_policy_state",
    "save_policy_state",
    "extract_features",
    "FeatureVector",
    "get_feature_names",
    "compute_reward",
    "RewardSignal",
    "compare_policy_states",
    "ComparisonResult",
    "load_policy_from_json",
]
