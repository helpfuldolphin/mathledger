"""
RFL Policy Feature Extraction Module

Feature-rich extraction for Phase II slices with clean slice-based pluggability.
All features are derived from verifiable outcomes (derivation.verified, goal hits, etc.)
-- NO RLHF/RLPF or human preference signals.

DETERMINISM: All feature extraction is deterministic given the same inputs.
No wall-clock time, random sampling, or external entropy sources.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Supported slice names for Phase II feature extraction
PHASE_II_SLICES: Set[str] = frozenset({
    "goal",
    "tree",
    "dependency",
    "uplift_proto",
    "slice_uplift_proto",
    "phase_ii_core",
    "phase_ii_goal",
    "phase_ii_tree",
    "phase_ii_dependency",
})

# Phase I slices use baseline (zero-weight) features
PHASE_I_SLICES: Set[str] = frozenset({
    "warmup",
    "core",
    "refinement",
    "baseline",
    "tail",
})


@dataclass
class RFLContext:
    """
    Context for RFL feature extraction.

    Contains the history and state needed to compute features.
    This is the interface that the RFL runner passes to extract_features.
    """

    # Slice information
    slice_name: str
    slice_index: int = 0

    # History tracking for success-based features
    success_history: Dict[str, int] = field(default_factory=dict)  # candidate_hash -> success count
    attempt_history: Dict[str, int] = field(default_factory=dict)  # candidate_hash -> attempt count

    # Goal tracking (for goal slices)
    target_goals: List[str] = field(default_factory=list)
    achieved_goals: Set[str] = field(default_factory=set)

    # Tree/dependency tracking
    derivation_tree: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    required_statements: Set[str] = field(default_factory=set)

    # Cycle-level metrics (from verified outcomes)
    cycle_verified_count: int = 0
    cycle_total_count: int = 0

    # Current weights (for feature computation)
    policy_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateFeatures:
    """
    Extracted features for a single candidate formula.

    All values are floats for uniform weight application.
    """

    # Basic structural features
    length: float = 0.0
    depth: float = 0.0
    num_connectives: float = 0.0
    num_negations: float = 0.0

    # Goal-related features (for goal slices)
    goal_overlap: float = 0.0
    is_required_goal: float = 0.0

    # History-based features (from verified outcomes)
    success_count: float = 0.0
    success_rate: float = 0.0

    # Tree/dependency features
    chain_depth: float = 0.0
    num_children: float = 0.0
    num_parents: float = 0.0

    # Derived complexity features
    connective_density: float = 0.0
    nesting_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for inspection/logging."""
        return {
            "length": self.length,
            "depth": self.depth,
            "num_connectives": self.num_connectives,
            "num_negations": self.num_negations,
            "goal_overlap": self.goal_overlap,
            "is_required_goal": self.is_required_goal,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "chain_depth": self.chain_depth,
            "num_children": self.num_children,
            "num_parents": self.num_parents,
            "connective_density": self.connective_density,
            "nesting_ratio": self.nesting_ratio,
        }


def _compute_depth(formula: str) -> int:
    """
    Compute AST depth (maximum nesting of parentheses).

    DETERMINISM: Pure function of input string.
    """
    max_depth = 0
    current_depth = 0
    for char in formula:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ")":
            current_depth = max(0, current_depth - 1)
    return max_depth


def _count_connectives(formula: str) -> int:
    """
    Count logical connectives in formula.

    Supports: ->, /\\, \\/, <->
    DETERMINISM: Pure function of input string.
    """
    # Count ASCII representations
    implies = formula.count("->") - formula.count("<->")  # Avoid double-counting biconditional
    biconditional = formula.count("<->")
    conjunction = formula.count("/\\")
    disjunction = formula.count("\\/")

    # Count Unicode representations
    implies += formula.count("→")
    biconditional += formula.count("↔")
    conjunction += formula.count("∧")
    disjunction += formula.count("∨")

    return implies + biconditional + conjunction + disjunction


def _count_negations(formula: str) -> int:
    """
    Count negation operators in formula.

    DETERMINISM: Pure function of input string.
    """
    return formula.count("~") + formula.count("¬") + formula.count("!")


def _compute_goal_overlap(formula: str, target_goals: List[str]) -> float:
    """
    Compute overlap between formula and target goals.

    Returns a value in [0, 1] representing how many target goal patterns
    appear in the formula.

    DETERMINISM: Pure function of inputs.
    """
    if not target_goals:
        return 0.0

    # Extract variables/atoms from formula
    formula_lower = formula.lower()
    formula_atoms = set(re.findall(r"\b[a-z]\b", formula_lower))

    matched = 0
    for goal in target_goals:
        goal_lower = goal.lower()
        goal_atoms = set(re.findall(r"\b[a-z]\b", goal_lower))
        # Check for atom overlap
        if formula_atoms & goal_atoms:
            matched += 1
            continue
        # Check for substring match (normalized)
        goal_norm = re.sub(r"\s+", "", goal_lower)
        formula_norm = re.sub(r"\s+", "", formula_lower)
        if goal_norm in formula_norm or formula_norm in goal_norm:
            matched += 1

    return matched / len(target_goals)


def _compute_candidate_hash(formula: str) -> str:
    """
    Compute deterministic hash for candidate formula.

    DETERMINISM: SHA256 is deterministic.
    """
    normalized = re.sub(r"\s+", "", formula.lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_chain_depth(
    formula_hash: str, derivation_tree: Dict[str, List[str]], visited: Optional[Set[str]] = None
) -> int:
    """
    Compute derivation chain depth (distance from axioms).

    DETERMINISM: Pure function of tree structure.
    """
    if visited is None:
        visited = set()

    if formula_hash in visited:
        return 0  # Cycle detected, stop
    visited.add(formula_hash)

    # Find parents (who derived this formula)
    max_parent_depth = 0
    for parent, children in derivation_tree.items():
        if formula_hash in children:
            parent_depth = _get_chain_depth(parent, derivation_tree, visited)
            max_parent_depth = max(max_parent_depth, parent_depth + 1)

    return max_parent_depth


def extract_features(
    candidate: str,
    slice_name: str,
    context: RFLContext,
) -> Dict[str, float]:
    """
    Extract features for a candidate formula based on slice type.

    This is the main entry point for feature extraction. It returns a dictionary
    of feature names to float values, suitable for weighted scoring.

    For Phase I slices (baseline): Returns zero features (random ordering equivalent).
    For Phase II slices: Returns rich features based on slice type.

    Args:
        candidate: The candidate formula string
        slice_name: The curriculum slice name (e.g., "goal", "tree", "warmup")
        context: RFLContext with history and state

    Returns:
        Dictionary mapping feature names to float values.
        All values are deterministically computed from inputs.

    DETERMINISM: All features are derived from verifiable outcomes.
    No random sampling, wall-clock time, or external entropy.
    """
    # Normalize slice name for matching
    slice_lower = slice_name.lower().strip()

    # Phase I slices use baseline (zero) features
    # This ensures baseline runs use random ordering (all weights = 0)
    if slice_lower in PHASE_I_SLICES or not _is_phase_ii_slice(slice_lower):
        return _extract_baseline_features(candidate)

    # Phase II slices use rich features
    return _extract_phase_ii_features(candidate, slice_lower, context)


def _is_phase_ii_slice(slice_name: str) -> bool:
    """Check if slice name indicates Phase II features should be used."""
    slice_lower = slice_name.lower()
    # Direct match
    if slice_lower in PHASE_II_SLICES:
        return True
    # Prefix match for variants
    for phase_ii in PHASE_II_SLICES:
        if slice_lower.startswith(phase_ii) or phase_ii in slice_lower:
            return True
    return False


def _extract_baseline_features(candidate: str) -> Dict[str, float]:
    """
    Extract baseline features (all zeros for random ordering).

    Used for Phase I slices where RFL policy should not influence ordering.
    """
    return {
        "len": 0.0,
        "depth": 0.0,
        "num_connectives": 0.0,
        "num_negations": 0.0,
        "goal_overlap": 0.0,
        "is_required_goal": 0.0,
        "success_count": 0.0,
        "success_rate": 0.0,
        "chain_depth": 0.0,
        "connective_density": 0.0,
        "nesting_ratio": 0.0,
    }


def _extract_phase_ii_features(
    candidate: str,
    slice_name: str,
    context: RFLContext,
) -> Dict[str, float]:
    """
    Extract rich features for Phase II slices.

    Feature set depends on slice type:
    - goal slices: goal_overlap, is_required_goal
    - tree/dependency slices: chain_depth, num_children, num_parents
    - all Phase II: structural features + history features
    """
    features: Dict[str, float] = {}

    # Compute candidate hash for history lookup
    candidate_hash = _compute_candidate_hash(candidate)

    # ===== Structural Features (all Phase II slices) =====
    length = float(len(candidate))
    depth = float(_compute_depth(candidate))
    num_connectives = float(_count_connectives(candidate))
    num_negations = float(_count_negations(candidate))

    features["len"] = length
    features["depth"] = depth
    features["num_connectives"] = num_connectives
    features["num_negations"] = num_negations

    # Derived complexity features
    features["connective_density"] = num_connectives / max(length, 1.0)
    features["nesting_ratio"] = depth / max(num_connectives + 1, 1.0)

    # ===== History Features (from verified outcomes) =====
    success_count = float(context.success_history.get(candidate_hash, 0))
    attempt_count = float(context.attempt_history.get(candidate_hash, 0))
    success_rate = success_count / max(attempt_count, 1.0)

    features["success_count"] = success_count
    features["success_rate"] = success_rate

    # ===== Goal Features (for goal slices) =====
    if "goal" in slice_name:
        features["goal_overlap"] = _compute_goal_overlap(candidate, context.target_goals)
        features["is_required_goal"] = (
            1.0 if candidate_hash in context.required_statements else 0.0
        )
    else:
        features["goal_overlap"] = 0.0
        features["is_required_goal"] = 0.0

    # ===== Tree/Dependency Features =====
    if "tree" in slice_name or "dependency" in slice_name:
        features["chain_depth"] = float(
            _get_chain_depth(candidate_hash, context.derivation_tree)
        )
        # Count children (formulas derived from this one)
        features["num_children"] = float(
            len(context.derivation_tree.get(candidate_hash, []))
        )
        # Count parents (formulas that led to this one)
        num_parents = 0
        for parent, children in context.derivation_tree.items():
            if candidate_hash in children:
                num_parents += 1
        features["num_parents"] = float(num_parents)
    else:
        features["chain_depth"] = 0.0
        features["num_children"] = 0.0
        features["num_parents"] = 0.0

    return features


def compute_feature_score(
    features: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Compute weighted score from features.

    Score = sum(feature_value * weight) for all features.

    For baseline (all weights = 0), score = 0 for all candidates,
    resulting in random (or default) ordering.

    Args:
        features: Feature dictionary from extract_features
        weights: Weight dictionary (feature_name -> weight)

    Returns:
        Weighted sum of features. Higher scores = preferred candidates.

    DETERMINISM: Pure arithmetic on inputs.
    """
    score = 0.0
    for feature_name, feature_value in features.items():
        weight = weights.get(feature_name, 0.0)
        score += feature_value * weight
    return score


def update_context_from_cycle(
    context: RFLContext,
    candidate_hash: str,
    cycle_success: bool,
    verified_count: int = 0,
) -> None:
    """
    Update RFLContext based on cycle outcome.

    This should be called after each derivation cycle with the verified
    outcome. It updates the history used for success_count and success_rate
    features.

    Args:
        context: RFLContext to update (modified in place)
        candidate_hash: Hash of the candidate formula processed
        cycle_success: Whether the cycle met success criteria
        verified_count: Number of verified proofs in cycle

    DETERMINISM: Deterministic state update from inputs.
    """
    # Update attempt count
    context.attempt_history[candidate_hash] = (
        context.attempt_history.get(candidate_hash, 0) + 1
    )

    # Update success count if cycle succeeded
    if cycle_success:
        context.success_history[candidate_hash] = (
            context.success_history.get(candidate_hash, 0) + 1
        )

    # Update cycle metrics
    context.cycle_verified_count = verified_count
    context.cycle_total_count = context.cycle_total_count + 1


def create_context_for_slice(
    slice_name: str,
    slice_index: int = 0,
    existing_weights: Optional[Dict[str, float]] = None,
    existing_success_history: Optional[Dict[str, int]] = None,
    existing_attempt_history: Optional[Dict[str, int]] = None,
) -> RFLContext:
    """
    Create an RFLContext configured for a specific slice.

    Factory function to create properly initialized context.

    Args:
        slice_name: Curriculum slice name
        slice_index: Index of slice in curriculum
        existing_weights: Pre-existing policy weights to copy
        existing_success_history: Pre-existing success history to copy
        existing_attempt_history: Pre-existing attempt history to copy

    Returns:
        Initialized RFLContext for the slice.
    """
    return RFLContext(
        slice_name=slice_name,
        slice_index=slice_index,
        success_history=dict(existing_success_history or {}),
        attempt_history=dict(existing_attempt_history or {}),
        policy_weights=dict(existing_weights or {}),
    )


def get_feature_names() -> List[str]:
    """
    Return list of all feature names supported by this module.

    Useful for initializing weight vectors or documentation.
    """
    return [
        "len",
        "depth",
        "num_connectives",
        "num_negations",
        "goal_overlap",
        "is_required_goal",
        "success_count",
        "success_rate",
        "chain_depth",
        "num_children",
        "num_parents",
        "connective_density",
        "nesting_ratio",
    ]


def get_default_weights() -> Dict[str, float]:
    """
    Return default (zero) weights for baseline runs.

    Zero weights ensure score = 0 for all candidates,
    resulting in random ordering (or default derivation order).
    """
    return {name: 0.0 for name in get_feature_names()}
