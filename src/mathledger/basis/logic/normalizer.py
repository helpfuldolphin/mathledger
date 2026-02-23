r"""Deterministic propositional formula normalization."""

from __future__ import annotations

import functools
import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from mathledger.basis.core import NormalizedFormula

OP_IMP = "->"
OP_AND = "/\\"
OP_OR = "\\/"

_SYMBOL_MAP = {
    "\u2192": OP_IMP,
    "\u21d2": OP_IMP,
    "\u27f9": OP_IMP,
    "\u2194": "<->",
    "\u21d4": "<->",
    "\u2227": OP_AND,
    "\u22c0": OP_AND,
    "\u2228": OP_OR,
    "\u22c1": OP_OR,
    "\u00ac": "~",
    "\uffe2": "~",
    "\uff08": "(",
    "\uff09": ")",
    "\u27e8": "(",
    "\u27e9": ")",
    "\u00a0": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2009": " ",
    "\u202f": " ",
    "\u3000": " ",
}

_WHITESPACE_RE = re.compile(r"\s+")
_ATOM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _map_unicode(expr: str) -> str:
    mapped = expr
    for src, dest in _SYMBOL_MAP.items():
        if src in mapped:
            mapped = mapped.replace(src, dest)
    return mapped


def _strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    while len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
        depth = 0
        balanced = True
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and idx < len(expr) - 1:
                balanced = False
                break
        if balanced and depth == 0:
            expr = expr[1:-1].strip()
        else:
            break
    return expr


def _split_top(expr: str, operator: str) -> Tuple[Optional[str], Optional[str]]:
    depth = 0
    width = len(operator)
    idx = 0
    while idx <= len(expr) - width:
        ch = expr[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr[idx : idx + width] == operator:
            left = expr[:idx].strip()
            right = expr[idx + width :].strip()
            return (left or None, right or None)
        idx += 1
    return None, None


def _flatten(expr: str, operator: str) -> List[str]:
    left, right = _split_top(expr, operator)
    if left is None or right is None:
        return [expr]
    return _flatten(left, operator) + _flatten(right, operator)


def _wrap_if_needed(expr: str) -> str:
    if any(op in expr for op in (OP_IMP, OP_AND, OP_OR)) and not expr.startswith("("):
        return f"({expr})"
    return expr


def _normalize_and(expr: str) -> NormalizedFormula:
    operands = [_normalize(_strip_outer_parens(part)) for part in _flatten(expr, OP_AND)]
    unique = sorted(set(operands))
    return OP_AND.join(unique)


def _normalize_or(expr: str) -> NormalizedFormula:
    operands = [_normalize(_strip_outer_parens(part)) for part in _flatten(expr, OP_OR)]
    unique = sorted({_wrap_if_needed(op) if OP_AND in op or OP_IMP in op else op for op in operands})
    return OP_OR.join(unique)


def _normalize_imp(expr: str) -> NormalizedFormula:
    left_raw, right_raw = _split_top(expr, OP_IMP)
    assert left_raw is not None and right_raw is not None
    left_norm = _normalize(_strip_outer_parens(left_raw))
    right = _strip_outer_parens(right_raw)

    chain: List[str] = []
    cursor = right
    while True:
        a, b = _split_top(cursor, OP_IMP)
        if a is None or b is None:
            chain.append(_normalize(cursor))
            break
        chain.append(_normalize(_strip_outer_parens(a)))
        cursor = _strip_outer_parens(b)

    left_emit = f"({left_norm})" if any(op in left_norm for op in (OP_AND, OP_OR, OP_IMP)) else left_norm
    return left_emit + OP_IMP + OP_IMP.join(chain)


@functools.lru_cache(maxsize=8192)
def _normalize(expr: str) -> NormalizedFormula:
    expr = _map_unicode(expr)
    expr = _WHITESPACE_RE.sub(" ", expr.strip())
    expr = _strip_outer_parens(expr)
    if not expr:
        return ""

    if expr.startswith("~"):
        inner = _normalize(expr[1:])
        return f"~{inner}"

    left, right = _split_top(expr, OP_AND)
    if left is not None and right is not None:
        return _normalize_and(expr)

    left, right = _split_top(expr, OP_OR)
    if left is not None and right is not None:
        return _normalize_or(expr)

    left, right = _split_top(expr, OP_IMP)
    if left is not None and right is not None:
        return _normalize_imp(expr)

    return expr.replace(" ", "")


def normalize(expr: str) -> NormalizedFormula:
    """Canonical public normalization entrypoint."""
    return _normalize(expr)


def normalize_pretty(expr: str) -> str:
    """Human-readable variant with spaced implication."""
    return normalize(expr).replace(OP_IMP, " -> ")


def are_equivalent(a: str, b: str) -> bool:
    """Equivalence under canonical normalization."""
    return normalize(a) == normalize(b)


def atoms(expr: str) -> Set[str]:
    """Atomic proposition names present in formula text."""
    return set(_ATOM_RE.findall(_map_unicode(expr)))


def normalize_many(exprs: Iterable[str]) -> Tuple[NormalizedFormula, ...]:
    """Deterministic batch normalization."""
    return tuple(normalize(expr) for expr in exprs)

