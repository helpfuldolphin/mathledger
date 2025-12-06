"""
RFL Policy Feature Extraction — PHASE II

Provides feature extraction for derivation candidates and slice-aware feature masks.

Slice Feature Masks:
- slice_uplift_goal: emphasize goal overlap, required-goal indicators
- slice_uplift_sparse: emphasize success_count, candidate rarity
- slice_uplift_tree: emphasize chain_depth, dependency features
- slice_uplift_dependency: emphasize per-goal hit features

NOTE: PHASE II — NOT USED IN PHASE I
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

# Canonical feature names used across the policy module
FEATURE_NAMES: List[str] = [
    # Basic structural features
    "text_length",           # Length of candidate formula text
    "ast_depth",             # Depth of abstract syntax tree
    "atom_count",            # Number of atomic symbols
    "connective_count",      # Number of logical connectives
    # Chain/derivation features
    "chain_depth",           # Depth in derivation chain
    "dependency_count",      # Number of dependencies
    # Goal-related features
    "goal_overlap",          # Overlap with target goals
    "required_goal",         # Indicator for required goal
    "per_goal_hit",          # Per-goal hit indicator
    # Success/history features
    "success_count",         # Historical success count for this candidate hash
    "attempt_count",         # Historical attempt count
    "success_rate",          # success_count / attempt_count (0 if no attempts)
    "candidate_rarity",      # Inverse frequency measure (rare = high)
]

# Total number of features
NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureVector:
    """
    Feature vector for a single derivation candidate.

    All features are deterministic given the candidate and context.
    """

    # Structural
    text_length: float = 0.0
    ast_depth: float = 0.0
    atom_count: float = 0.0
    connective_count: float = 0.0

    # Chain/derivation
    chain_depth: float = 0.0
    dependency_count: float = 0.0

    # Goal-related
    goal_overlap: float = 0.0
    required_goal: float = 0.0
    per_goal_hit: float = 0.0

    # Success/history
    success_count: float = 0.0
    attempt_count: float = 0.0
    success_rate: float = 0.0
    candidate_rarity: float = 0.0

    def to_array(self) -> List[float]:
        """Convert to list of floats in canonical order."""
        return [
            self.text_length,
            self.ast_depth,
            self.atom_count,
            self.connective_count,
            self.chain_depth,
            self.dependency_count,
            self.goal_overlap,
            self.required_goal,
            self.per_goal_hit,
            self.success_count,
            self.attempt_count,
            self.success_rate,
            self.candidate_rarity,
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> FeatureVector:
        """Create from array of floats in canonical order."""
        if len(arr) != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} features, got {len(arr)}")
        return cls(
            text_length=arr[0],
            ast_depth=arr[1],
            atom_count=arr[2],
            connective_count=arr[3],
            chain_depth=arr[4],
            dependency_count=arr[5],
            goal_overlap=arr[6],
            required_goal=arr[7],
            per_goal_hit=arr[8],
            success_count=arr[9],
            attempt_count=arr[10],
            success_rate=arr[11],
            candidate_rarity=arr[12],
        )

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> FeatureVector:
        """Create from dictionary."""
        return cls(**{k: d.get(k, 0.0) for k in FEATURE_NAMES})


# ---------------------------------------------------------------------------
# Slice Feature Masks
#
# Each mask is a dict mapping feature name → weight multiplier.
# A weight of 1.0 means the feature is fully used; 0.0 means ignored.
# This allows each slice to emphasize relevant features without wasting
# policy capacity on irrelevant ones.
# ---------------------------------------------------------------------------

SLICE_FEATURE_MASKS: Dict[str, Dict[str, float]] = {
    # slice_uplift_goal: emphasize goal overlap, required-goal indicators
    "slice_uplift_goal": {
        "text_length": 0.3,
        "ast_depth": 0.3,
        "atom_count": 0.3,
        "connective_count": 0.3,
        "chain_depth": 0.5,
        "dependency_count": 0.5,
        "goal_overlap": 1.0,       # Primary emphasis
        "required_goal": 1.0,      # Primary emphasis
        "per_goal_hit": 0.8,
        "success_count": 0.5,
        "attempt_count": 0.3,
        "success_rate": 0.5,
        "candidate_rarity": 0.3,
    },

    # slice_uplift_sparse: emphasize success_count, candidate rarity
    "slice_uplift_sparse": {
        "text_length": 0.3,
        "ast_depth": 0.3,
        "atom_count": 0.3,
        "connective_count": 0.3,
        "chain_depth": 0.5,
        "dependency_count": 0.5,
        "goal_overlap": 0.5,
        "required_goal": 0.5,
        "per_goal_hit": 0.5,
        "success_count": 1.0,      # Primary emphasis
        "attempt_count": 0.5,
        "success_rate": 0.8,
        "candidate_rarity": 1.0,   # Primary emphasis
    },

    # slice_uplift_tree: emphasize chain_depth, dependency features
    "slice_uplift_tree": {
        "text_length": 0.5,
        "ast_depth": 0.8,
        "atom_count": 0.5,
        "connective_count": 0.5,
        "chain_depth": 1.0,        # Primary emphasis
        "dependency_count": 1.0,   # Primary emphasis
        "goal_overlap": 0.5,
        "required_goal": 0.5,
        "per_goal_hit": 0.5,
        "success_count": 0.5,
        "attempt_count": 0.3,
        "success_rate": 0.5,
        "candidate_rarity": 0.3,
    },

    # slice_uplift_dependency: emphasize per-goal hit features
    "slice_uplift_dependency": {
        "text_length": 0.3,
        "ast_depth": 0.3,
        "atom_count": 0.3,
        "connective_count": 0.3,
        "chain_depth": 0.5,
        "dependency_count": 0.8,
        "goal_overlap": 0.8,
        "required_goal": 0.8,
        "per_goal_hit": 1.0,       # Primary emphasis
        "success_count": 0.5,
        "attempt_count": 0.3,
        "success_rate": 0.5,
        "candidate_rarity": 0.3,
    },

    # Default mask: all features equally weighted
    "default": {name: 1.0 for name in FEATURE_NAMES},
}


def get_feature_mask(slice_name: str) -> Dict[str, float]:
    """
    Get the feature mask for a given slice.

    Args:
        slice_name: Name of the slice (e.g., "slice_uplift_goal")

    Returns:
        Feature mask dictionary
    """
    # Check for exact match first
    if slice_name in SLICE_FEATURE_MASKS:
        return SLICE_FEATURE_MASKS[slice_name]

    # Try prefix matching for uplift slices
    for key in SLICE_FEATURE_MASKS:
        if slice_name.startswith(key) or key.startswith(slice_name):
            return SLICE_FEATURE_MASKS[key]

    # Return default mask
    return SLICE_FEATURE_MASKS["default"]


def apply_feature_mask(
    features: FeatureVector,
    mask: Dict[str, float],
) -> FeatureVector:
    """
    Apply a feature mask to a feature vector.

    Each feature value is multiplied by its corresponding mask weight.
    This allows slices to de-emphasize irrelevant features.

    Args:
        features: Input feature vector
        mask: Feature mask dictionary

    Returns:
        Masked feature vector
    """
    masked_dict = {}
    for name in FEATURE_NAMES:
        weight = mask.get(name, 1.0)
        original_value = getattr(features, name, 0.0)
        masked_dict[name] = original_value * weight

    return FeatureVector.from_dict(masked_dict)


def extract_features(
    candidate: str,
    context: Optional[Dict[str, Any]] = None,
    success_history: Optional[Dict[str, int]] = None,
    attempt_history: Optional[Dict[str, int]] = None,
    total_candidates_seen: int = 1,
) -> FeatureVector:
    """
    Extract features from a derivation candidate.

    All feature extraction is deterministic given the inputs.

    Args:
        candidate: Candidate formula string
        context: Optional context dict with derivation metadata
        success_history: Dict mapping candidate hash → success count
        attempt_history: Dict mapping candidate hash → attempt count
        total_candidates_seen: Total number of unique candidates seen

    Returns:
        FeatureVector with extracted features
    """
    context = context or {}
    success_history = success_history or {}
    attempt_history = attempt_history or {}

    # Compute candidate hash for history lookup
    import hashlib
    candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()

    # Basic structural features
    text_length = float(len(candidate))
    ast_depth = _estimate_ast_depth(candidate)
    atom_count = _count_atoms(candidate)
    connective_count = _count_connectives(candidate)

    # Chain/derivation features from context
    chain_depth = float(context.get("chain_depth", 0))
    dependency_count = float(context.get("dependency_count", 0))

    # Goal-related features from context
    goal_overlap = float(context.get("goal_overlap", 0.0))
    required_goal = float(context.get("required_goal", 0))
    per_goal_hit = float(context.get("per_goal_hit", 0.0))

    # Success/history features
    success_count = float(success_history.get(candidate_hash, 0))
    attempt_count = float(attempt_history.get(candidate_hash, 0))
    success_rate = success_count / attempt_count if attempt_count > 0 else 0.0

    # Candidate rarity: inverse frequency (higher = rarer)
    frequency = attempt_count / max(total_candidates_seen, 1)
    candidate_rarity = 1.0 - min(frequency, 1.0)

    return FeatureVector(
        text_length=text_length,
        ast_depth=ast_depth,
        atom_count=atom_count,
        connective_count=connective_count,
        chain_depth=chain_depth,
        dependency_count=dependency_count,
        goal_overlap=goal_overlap,
        required_goal=required_goal,
        per_goal_hit=per_goal_hit,
        success_count=success_count,
        attempt_count=attempt_count,
        success_rate=success_rate,
        candidate_rarity=candidate_rarity,
    )


def _estimate_ast_depth(formula: str) -> float:
    """
    Estimate AST depth from formula string.

    Simple heuristic based on nesting of parentheses and operators.
    """
    depth = 0
    max_depth = 0
    for char in formula:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth = max(0, depth - 1)
    return float(max_depth)


def _count_atoms(formula: str) -> float:
    """
    Count atomic symbols in formula.

    Atoms are typically lowercase letters (p, q, r, etc.).
    """
    count = 0
    for char in formula:
        if char.islower() and char.isalpha():
            count += 1
    return float(count)


def _count_connectives(formula: str) -> float:
    """
    Count logical connectives in formula.

    Connectives: ->, <->, &, |, ~, ^, v (depending on notation)
    """
    connectives = ['->','<->','&','|','~','^','v','∧','∨','¬','→','↔']
    count = 0
    for conn in connectives:
        count += formula.count(conn)
    return float(count)
