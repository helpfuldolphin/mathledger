# -*- coding: utf-8 -*-
"""
Minimal truth table evaluator for propositional logic tautology checking.
Supports variables p, q, r and connectives: ->, /\\, \\/, ~
"""

import itertools
import re
from typing import Dict, List

TOK = {"AND": "/\\", "OR": "\\/", "NOT": "~", "IMP": "->"}

def _vars(formula: str) -> List[str]:
    """Extract variables from formula."""
    return sorted(set([c for c in formula if c in "pqr"]))

def _eval(formula: str, env: Dict[str, bool]) -> bool:
    """Evaluate formula with given variable assignments."""
    s = formula.replace(" ", "")

    def val(t: str) -> bool:
        if t in "pqr":
            return env[t]
        raise ValueError(f"Unknown atom: {t}")

    def paren(s: str) -> str:
        """Handle parentheses by recursively evaluating inner expressions."""
        while "(" in s:
            s = re.sub(r"\(([^()]+)\)", lambda m: inner(m.group(1)), s)
        return inner(s)

    def inner(t: str) -> str:
        """Evaluate inner expression without parentheses."""
        t = t.replace(TOK["AND"], "&").replace(TOK["OR"], "|").replace(TOK["IMP"], "=>")

        # Handle unary NOT
        while "~" in t:
            t = re.sub(r"~([pqr])", lambda m: ("1" if not val(m.group(1)) else "0"), t)

        # Replace atoms with 0/1
        for a in "pqr":
            if a in t:
                t = t.replace(a, "1" if val(a) else "0")

        # Handle AND
        while "&" in t:
            t = re.sub(r"([01])&([01])", lambda m: str(int(m.group(1) == "1" and m.group(2) == "1")), t, count=1)

        # Handle OR
        while "|" in t:
            t = re.sub(r"([01])\|([01])", lambda m: str(int(m.group(1) == "1" or m.group(2) == "1")), t, count=1)

        # Handle IMPLIES (process one at a time to avoid creating new patterns)
        max_iterations = len(t)  # Safety limit to prevent infinite loops
        iterations = 0
        while "=>" in t and iterations < max_iterations:
            iterations += 1
            # Find the leftmost implication and process it
            new_t = re.sub(r"([01])=>([01])", lambda m: ("1" if (m.group(1) == "0" or m.group(2) == "1") else "0"), t, count=1)
            if new_t == t:
                # No change made - pattern doesn't match, break to avoid infinite loop
                break
            t = new_t

        return t

    return paren(s) == "1"

def is_tautology(formula: str) -> bool:
    """
    Check if a propositional logic formula is a tautology using truth tables.

    Args:
        formula: String containing propositional logic formula with variables p, q, r
                and connectives ->, /\\, \\/, ~

    Returns:
        True if formula is a tautology (true for all variable assignments), False otherwise
    """
    vs = _vars(formula)
    if not vs:
        # No variables, evaluate as constant
        try:
            return _eval(formula, {})
        except Exception:
            return False

    # Check all possible variable assignments
    for bits in itertools.product([False, True], repeat=len(vs)):
        env = dict(zip(vs, bits))
        if not _eval(formula, env):
            return False

    return True


class TruthTable:
    """
    Lightweight TruthTable wrapper for compatibility with legacy exports.
    """

    def __init__(self, formula: str):
        self.formula = formula

    def atoms(self) -> List[str]:
        """Return sorted atomic propositions used in the formula."""
        return _vars(self.formula)

    def evaluate(self, env: Dict[str, bool]) -> bool:
        """Evaluate the stored formula under the provided environment."""
        return _eval(self.formula, env)

    def is_tautology(self) -> bool:
        """Check whether the stored formula is a tautology."""
        return is_tautology(self.formula)
