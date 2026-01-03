"""
MV Validator: Simple Arithmetic Equality Checker.

This is the ONE real validator in v0. It performs deterministic mechanical
verification of basic arithmetic equalities.

SCOPE:
- Parses claims of the form "a op b = c" where op is +, -, *, /
- Returns VERIFIED if the equation is mathematically correct
- Returns REFUTED if the equation is mathematically incorrect
- Returns ABSTAINED if the claim is not parseable as arithmetic

CONSTRAINTS:
- Deterministic: same input always produces same output
- No learning: hardcoded parsing rules only
- No external calls: pure computation
- Integer arithmetic only (avoids floating-point nondeterminism)

Examples:
    "2 + 2 = 4"       → VERIFIED
    "3 * 7 = 21"      → VERIFIED
    "10 - 3 = 7"      → VERIFIED
    "2 + 2 = 5"       → REFUTED
    "forall x, ..."   → ABSTAINED (not parseable)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ValidatorOutcome(str, Enum):
    """Outcome of mechanical validation."""
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True)
class ValidationResult:
    """Result of running the MV validator."""
    outcome: ValidatorOutcome
    explanation: str
    parsed_lhs: Optional[str] = None
    parsed_rhs: Optional[str] = None
    computed_value: Optional[int] = None


# Pattern: "a op b = c" where a, b, c are integers and op is +, -, *, /
# Also handles negative numbers and whitespace
ARITHMETIC_PATTERN = re.compile(
    r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)\s*$"
)


def _safe_divide(a: int, b: int) -> Optional[int]:
    """Integer division, returns None if not exact or division by zero."""
    if b == 0:
        return None
    if a % b != 0:
        return None  # Not exact integer division
    return a // b


def _evaluate(a: int, op: str, b: int) -> Optional[int]:
    """Evaluate a op b, returning None if undefined."""
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return _safe_divide(a, b)
    return None


def validate_arithmetic(claim_text: str) -> ValidationResult:
    """
    Validate a claim as a simple arithmetic equality.

    Args:
        claim_text: The claim to validate (e.g., "2 + 2 = 4")

    Returns:
        ValidationResult with outcome and explanation

    This function is:
    - Deterministic: same input → same output
    - Pure: no side effects, no external calls
    - Bounded: O(n) where n is claim length
    """
    # Try to parse as arithmetic
    match = ARITHMETIC_PATTERN.match(claim_text.strip())

    if not match:
        return ValidationResult(
            outcome=ValidatorOutcome.ABSTAINED,
            explanation="Claim is not a simple arithmetic equality (pattern: 'a op b = c')"
        )

    try:
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        c = int(match.group(4))
    except ValueError:
        return ValidationResult(
            outcome=ValidatorOutcome.ABSTAINED,
            explanation="Failed to parse integers from claim"
        )

    # Evaluate the left-hand side
    computed = _evaluate(a, op, b)

    if computed is None:
        return ValidationResult(
            outcome=ValidatorOutcome.ABSTAINED,
            explanation=f"Operation {a} {op} {b} is undefined (division by zero or non-integer result)"
        )

    lhs_str = f"{a} {op} {b}"
    rhs_str = str(c)

    if computed == c:
        return ValidationResult(
            outcome=ValidatorOutcome.VERIFIED,
            explanation=f"Arithmetic verified: {lhs_str} = {computed} = {c}",
            parsed_lhs=lhs_str,
            parsed_rhs=rhs_str,
            computed_value=computed
        )
    else:
        return ValidationResult(
            outcome=ValidatorOutcome.REFUTED,
            explanation=f"Arithmetic refuted: {lhs_str} = {computed} ≠ {c}",
            parsed_lhs=lhs_str,
            parsed_rhs=rhs_str,
            computed_value=computed
        )


def validate_mv_claim(claim_text: str) -> ValidationResult:
    """
    Entry point for MV validation.

    Currently supports only arithmetic validation.
    Future validators could be added here with a routing mechanism.

    Args:
        claim_text: The claim to validate

    Returns:
        ValidationResult
    """
    # v0: Only arithmetic validator
    return validate_arithmetic(claim_text)
