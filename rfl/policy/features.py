"""
Feature Extraction for RFL Policy
==================================

Extracts structural features from propositional formulas for policy scoring.

All feature extraction is deterministic and operates on normalized formulas.
Features include:
- Length: character length of the normalized formula
- Depth: AST depth (nesting level of connectives)
- Connectives: count of binary connectives (/\\, \\/, ->)
- Negations: count of negation operators (~)
- Overlap: atom overlap with a target formula
- Goal flag: whether the formula matches a required goal
- Success count: historical success rate for this formula hash
- Chain depth: derivation chain depth (if available)

Usage:
    from rfl.policy.features import extract_features, FeatureVector

    fv = extract_features(
        formula="p->q",
        target_atoms=frozenset({"p", "q"}),
        is_goal=False,
        success_count=3,
        chain_depth=2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence

from normalization.ast_canon import (
    parse_ast,
    Expr,
    Atom,
    Not,
    And,
    Or,
    Implies,
    Iff,
)
from derivation.structure import formula_depth, atom_frozenset


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """
    Feature vector for a propositional formula.

    All features are deterministic and derived from the formula structure
    or verifiable external state (success history, chain depth).

    Attributes:
        len: Character length of the normalized formula string.
        depth: AST depth (maximum nesting level).
        connectives: Count of binary connectives (/\\, \\/, ->, <->).
        negations: Count of negation operators (~).
        overlap: Number of atoms overlapping with the target set.
        goal_flag: 1 if this formula is a required goal, 0 otherwise.
        success_count: Historical count of successful cycles involving this formula.
        chain_depth: Derivation chain depth (0 if not in a chain or unknown).
    """
    len: int
    depth: int
    connectives: int
    negations: int
    overlap: int
    goal_flag: int
    success_count: int
    chain_depth: int

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_list(self) -> List[int]:
        """Convert to ordered list for dot-product computation."""
        return [
            self.len,
            self.depth,
            self.connectives,
            self.negations,
            self.overlap,
            self.goal_flag,
            self.success_count,
            self.chain_depth,
        ]

    @classmethod
    def field_names(cls) -> List[str]:
        """Return ordered field names for weight vector alignment."""
        return [
            "len",
            "depth",
            "connectives",
            "negations",
            "overlap",
            "goal_flag",
            "success_count",
            "chain_depth",
        ]

    @classmethod
    def zero(cls) -> "FeatureVector":
        """Return a zero feature vector."""
        return cls(
            len=0,
            depth=0,
            connectives=0,
            negations=0,
            overlap=0,
            goal_flag=0,
            success_count=0,
            chain_depth=0,
        )


def _count_connectives(expr: Expr) -> int:
    """Count binary connectives in an AST."""
    if isinstance(expr, Atom):
        return 0
    if isinstance(expr, Not):
        return _count_connectives(expr.operand)
    if isinstance(expr, And):
        # And with n operands has n-1 binary connectives
        count = len(expr.operands) - 1
        for op in expr.operands:
            count += _count_connectives(op)
        return count
    if isinstance(expr, Or):
        # Or with n operands has n-1 binary connectives
        count = len(expr.operands) - 1
        for op in expr.operands:
            count += _count_connectives(op)
        return count
    if isinstance(expr, Implies):
        return 1 + _count_connectives(expr.antecedent) + _count_connectives(expr.consequent)
    if isinstance(expr, Iff):
        return 1 + _count_connectives(expr.left) + _count_connectives(expr.right)
    return 0


def _count_negations(expr: Expr) -> int:
    """Count negation operators in an AST."""
    if isinstance(expr, Atom):
        return 0
    if isinstance(expr, Not):
        return 1 + _count_negations(expr.operand)
    if isinstance(expr, And):
        return sum(_count_negations(op) for op in expr.operands)
    if isinstance(expr, Or):
        return sum(_count_negations(op) for op in expr.operands)
    if isinstance(expr, Implies):
        return _count_negations(expr.antecedent) + _count_negations(expr.consequent)
    if isinstance(expr, Iff):
        return _count_negations(expr.left) + _count_negations(expr.right)
    return 0


@lru_cache(maxsize=4096)
def _extract_structural_features(formula: str) -> tuple:
    """
    Extract structural features from a formula (cached).

    Returns:
        Tuple of (length, depth, connectives, negations, atoms_frozenset)
    """
    length = len(formula)
    depth = formula_depth(formula)

    try:
        ast = parse_ast(formula)
        connectives = _count_connectives(ast)
        negations = _count_negations(ast)
        atoms = ast.atoms()
    except (ValueError, TypeError):
        # Fallback for unparseable formulas
        connectives = formula.count("/\\") + formula.count("\\/") + formula.count("->")
        negations = formula.count("~")
        atoms = atom_frozenset(formula)

    return (length, depth, connectives, negations, atoms)


def extract_features(
    formula: str,
    target_atoms: Optional[FrozenSet[str]] = None,
    is_goal: bool = False,
    success_count: int = 0,
    chain_depth: int = 0,
) -> FeatureVector:
    """
    Extract feature vector from a propositional formula.

    Args:
        formula: Normalized propositional formula string.
        target_atoms: Set of atoms in the target/goal formula for overlap computation.
                     If None, overlap is computed as 0.
        is_goal: Whether this formula is a required goal in the derivation.
        success_count: Historical count of successful derivations involving this formula.
        chain_depth: Depth in the current derivation chain (0 if not applicable).

    Returns:
        FeatureVector with all extracted features.

    Example:
        >>> fv = extract_features("p->q", target_atoms=frozenset({"p", "q"}))
        >>> fv.len
        4
        >>> fv.depth
        1
        >>> fv.connectives
        1
        >>> fv.overlap
        2
    """
    length, depth, connectives, negations, atoms = _extract_structural_features(formula)

    # Compute overlap with target
    if target_atoms is not None:
        overlap = len(atoms & target_atoms)
    else:
        overlap = 0

    return FeatureVector(
        len=length,
        depth=depth,
        connectives=connectives,
        negations=negations,
        overlap=overlap,
        goal_flag=1 if is_goal else 0,
        success_count=success_count,
        chain_depth=chain_depth,
    )


def extract_features_batch(
    formulas: Sequence[str],
    target_atoms: Optional[FrozenSet[str]] = None,
    is_goal_set: Optional[FrozenSet[str]] = None,
    success_counts: Optional[Dict[str, int]] = None,
    chain_depths: Optional[Dict[str, int]] = None,
) -> List[FeatureVector]:
    """
    Extract feature vectors for a batch of formulas.

    Args:
        formulas: Sequence of normalized formula strings.
        target_atoms: Set of atoms in the target/goal for overlap computation.
        is_goal_set: Set of formula hashes that are required goals.
        success_counts: Mapping from formula hash to success count.
        chain_depths: Mapping from formula to chain depth.

    Returns:
        List of FeatureVectors in the same order as input formulas.
    """
    is_goal_set = is_goal_set or frozenset()
    success_counts = success_counts or {}
    chain_depths = chain_depths or {}

    results = []
    for formula in formulas:
        # Use formula itself as key if no hash provided
        formula_key = formula
        is_goal = formula_key in is_goal_set
        success_count = success_counts.get(formula_key, 0)
        chain_depth = chain_depths.get(formula_key, 0)

        fv = extract_features(
            formula=formula,
            target_atoms=target_atoms,
            is_goal=is_goal,
            success_count=success_count,
            chain_depth=chain_depth,
        )
        results.append(fv)

    return results


__all__ = [
    "FeatureVector",
    "extract_features",
    "extract_features_batch",
]
