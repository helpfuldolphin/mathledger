"""
Truth table tautology checker for propositional logic.

Determines if a formula is a tautology using truth table evaluation.
"""

from typing import List, Set, Dict, Any
from itertools import product
import re


class TruthTableTimeout(Exception):
    """
    Raised when truth table evaluation exceeds the allowed time/complexity limit.

    This can happen for formulas with many atoms (2^n combinations to check).
    """
    pass


# Internal cache for truth table oracle results (for performance)
_truth_table_cache: Dict[str, bool] = {}


def clear_oracle_cache() -> None:
    """
    Clear the truth table oracle cache.

    This is useful for testing to ensure no state leakage between tests.
    """
    global _truth_table_cache
    _truth_table_cache.clear()


def get_oracle_cache_info() -> Dict[str, Any]:
    """
    Get information about the oracle cache state.

    Returns:
        Dictionary with cache statistics (size, etc.)
    """
    return {
        "size": len(_truth_table_cache),
        "entries": list(_truth_table_cache.keys())[:10],  # First 10 entries
    }


def truth_table_is_tautology(formula_ascii: str) -> bool:
    """
    Check if a propositional formula is a tautology using truth table.

    Args:
        formula_ascii: Formula in ASCII notation (->, /\\, \\/, ~)

    Returns:
        True if the formula is a tautology, False otherwise
    """
    # Extract atomic propositions
    atoms = _extract_atoms(formula_ascii)

    if not atoms:
        return False

    # Generate all possible truth value assignments
    for assignment in product([True, False], repeat=len(atoms)):
        truth_values = dict(zip(atoms, assignment))

        # Evaluate the formula with this assignment
        if not _evaluate_formula(formula_ascii, truth_values):
            return False

    return True


def _extract_atoms(formula: str) -> List[str]:
    """Extract atomic propositions from a formula."""
    # Find all single-letter propositions
    atoms = set(re.findall(r'\b([a-z])\b', formula))
    return sorted(list(atoms))


def _evaluate_formula(formula: str, truth_values: Dict[str, bool]) -> bool:
    """Evaluate a formula with given truth values."""
    # Remove whitespace
    formula = formula.replace(" ", "")

    # Handle parentheses by evaluating innermost first
    while '(' in formula:
        # Find innermost parentheses
        start = formula.rfind('(')
        end = formula.find(')', start)

        if start == -1 or end == -1:
            break

        inner = formula[start+1:end]
        inner_result = _evaluate_simple_formula(inner, truth_values)

        # Replace the parenthesized expression
        formula = formula[:start] + str(inner_result).lower() + formula[end+1:]

    return _evaluate_simple_formula(formula, truth_values)


def _evaluate_simple_formula(formula: str, truth_values: Dict[str, bool]) -> bool:
    """Evaluate a formula without parentheses."""
    # Handle negation first
    while '~' in formula:
        # Find negation operator
        neg_pos = formula.find('~')
        if neg_pos == -1:
            break

        # Find the atom after negation
        atom_start = neg_pos + 1
        atom_end = atom_start
        while atom_end < len(formula) and formula[atom_end].isalpha():
            atom_end += 1

        if atom_end > atom_start:
            atom = formula[atom_start:atom_end]
            if atom in truth_values:
                value = not truth_values[atom]
                formula = formula[:neg_pos] + str(value).lower() + formula[atom_end:]
            else:
                # Unknown atom, treat as false
                formula = formula[:neg_pos] + 'false' + formula[atom_end:]
        else:
            break

    # Handle conjunctions
    while '/\\' in formula:
        formula = _evaluate_binary_op(formula, '/\\', lambda a, b: a and b, truth_values)

    # Handle disjunctions
    while '\\/' in formula:
        formula = _evaluate_binary_op(formula, '\\/', lambda a, b: a or b, truth_values)

    # Handle implications
    while '->' in formula:
        formula = _evaluate_binary_op(formula, '->', lambda a, b: not a or b, truth_values)

    # Final evaluation
    if formula == 'true':
        return True
    elif formula == 'false':
        return False
    elif formula in truth_values:
        return truth_values[formula]
    else:
        return False


def _evaluate_binary_op(formula: str, op: str, op_func, truth_values: Dict[str, bool]) -> str:
    """Evaluate a binary operation."""
    op_pos = formula.find(op)
    if op_pos == -1:
        return formula

    # Find left operand
    left_start = op_pos - 1
    while left_start >= 0 and (formula[left_start].isalpha() or formula[left_start] in '()'):
        left_start -= 1
    left_start += 1

    # Find right operand
    right_end = op_pos + len(op)
    while right_end < len(formula) and (formula[right_end].isalpha() or formula[right_end] in '()'):
        right_end += 1

    left = formula[left_start:op_pos]
    right = formula[op_pos + len(op):right_end]

    # Evaluate operands
    left_val = _evaluate_atom(left, truth_values)
    right_val = _evaluate_atom(right, truth_values)

    # Apply operation
    result = op_func(left_val, right_val)

    # Replace in formula
    return formula[:left_start] + str(result).lower() + formula[right_end:]


def _evaluate_atom(atom: str, truth_values: Dict[str, bool]) -> bool:
    """Evaluate an atomic expression."""
    if atom == 'true':
        return True
    elif atom == 'false':
        return False
    elif atom in truth_values:
        return truth_values[atom]
    else:
        return False


def is_tautology(formula: str) -> bool:
    """Convenience function to check if a formula is a tautology."""
    return truth_table_is_tautology(formula)

