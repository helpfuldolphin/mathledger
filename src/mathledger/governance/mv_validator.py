"""Deterministic MV arithmetic validator for Wave A2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MVValidatorOutcome(str, Enum):
    """Mechanical validation outcomes."""

    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True)
class MVValidationResult:
    """Result of evaluating a claim through MV arithmetic route."""

    outcome: MVValidatorOutcome
    explanation: str
    parsed_lhs: Optional[str] = None
    parsed_rhs: Optional[str] = None
    computed_value: Optional[int] = None


ARITHMETIC_PATTERN = re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)\s*$")


def _safe_divide(a: int, b: int) -> Optional[int]:
    if b == 0:
        return None
    if a % b != 0:
        return None
    return a // b


def _evaluate(a: int, op: str, b: int) -> Optional[int]:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return _safe_divide(a, b)
    return None


def validate_mv_claim(claim_text: str) -> MVValidationResult:
    """
    Validate a claim as integer arithmetic equality: a op b = c.
    """
    match = ARITHMETIC_PATTERN.match(claim_text.strip())
    if not match:
        return MVValidationResult(
            outcome=MVValidatorOutcome.ABSTAINED,
            explanation="Claim is outside MV arithmetic coverage (expected: 'a op b = c').",
        )

    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))
    c = int(match.group(4))

    computed = _evaluate(a, op, b)
    if computed is None:
        return MVValidationResult(
            outcome=MVValidatorOutcome.ABSTAINED,
            explanation=f"Operation {a} {op} {b} is undefined for integer MV validation.",
        )

    lhs_str = f"{a} {op} {b}"
    rhs_str = str(c)

    if computed == c:
        return MVValidationResult(
            outcome=MVValidatorOutcome.VERIFIED,
            explanation=f"Arithmetic verified: {lhs_str} = {c}.",
            parsed_lhs=lhs_str,
            parsed_rhs=rhs_str,
            computed_value=computed,
        )

    return MVValidationResult(
        outcome=MVValidatorOutcome.REFUTED,
        explanation=f"Arithmetic refuted: {lhs_str} = {computed}, expected {c}.",
        parsed_lhs=lhs_str,
        parsed_rhs=rhs_str,
        computed_value=computed,
    )

