"""
Structural utilities for propositional formulas in canonical ASCII form.

All helpers operate on already normalized formulas (output of
`backend.logic.canon.normalize`).  They avoid allocating large intermediate
structures and stay deterministic so callers can safely memoise results.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from normalization.canon import get_atomic_propositions

OP_IMPLIES = "->"
OP_AND = "/\\"
OP_OR = "\\/"
OP_NOT = "~"


@lru_cache(maxsize=8192)
def _split_top(formula: str, operator: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a formula at the outermost occurrence of `operator`.

    Returns a tuple (lhs, rhs) when a top-level occurrence is found, or
    (None, None) otherwise.  Parentheses are honoured so nested occurrences
    do not cause accidental splits.
    """
    if not formula:
        return (None, None)

    depth = 0
    width = len(operator)
    for idx in range(len(formula) - width + 1):
        ch = formula[idx]
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            continue
        if depth == 0 and formula[idx : idx + width] == operator:
            left = formula[:idx].strip()
            right = formula[idx + width :].strip()
            if left and right:
                return (left, right)
            return (None, None)
    return (None, None)


@lru_cache(maxsize=8192)
def is_implication(formula: str) -> bool:
    """Check whether a normalized formula is an implication."""
    lhs, rhs = _split_top(formula, OP_IMPLIES)
    return lhs is not None and rhs is not None


@lru_cache(maxsize=8192)
def implication_parts(formula: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (antecedent, consequent) for implications, otherwise (None, None).
    """
    lhs, rhs = _split_top(formula, OP_IMPLIES)
    if lhs is None or rhs is None:
        return (None, None)
    return (strip_outer_parens(lhs), strip_outer_parens(rhs))


@lru_cache(maxsize=8192)
def strip_outer_parens(text: str) -> str:
    """Remove an outer parenthesis pair if it wraps the entire formula."""
    if not text or text[0] != "(" or text[-1] != ")":
        return text
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx < len(text) - 1:
                return text
    return text[1:-1]


@lru_cache(maxsize=8192)
def formula_depth(formula: str) -> int:
    """
    Compute the syntactic depth of a normalized formula.

    Depth counts binary connectives (`->`, `/\\`, `\\/`) and unary negation.
    """
    if not formula:
        return 0

    if formula.startswith(OP_NOT):
        return 1 + formula_depth(formula[len(OP_NOT) :])

    for op in (OP_IMPLIES, OP_AND, OP_OR):
        lhs, rhs = _split_top(formula, op)
        if lhs is not None and rhs is not None:
            return 1 + max(formula_depth(strip_outer_parens(lhs)), formula_depth(strip_outer_parens(rhs)))

    return 0


@lru_cache(maxsize=8192)
def atom_frozenset(formula: str) -> frozenset[str]:
    """Return the deduplicated set of atoms appearing in the formula."""
    return frozenset(sorted(get_atomic_propositions(formula)))

