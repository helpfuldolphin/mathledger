"""
Trust Monotonicity - Stub Implementation

This module provides stubs for trust class monotonicity invariants.
Full implementation is Phase II scope.

For v0: All functions are no-ops or return safe defaults.
"""

from typing import Any, Dict, Optional


class TrustClassMonotonicityViolation(Exception):
    """Raised when trust class monotonicity is violated."""
    pass


def require_trust_class_monotonicity(
    claim_id: str,
    old_trust_class: Optional[str],
    new_trust_class: str
) -> None:
    """
    Validate trust class monotonicity.

    Trust classes can only move in one direction:
    ADV -> PA -> MV -> FV (toward higher verification)

    v0 stub: No-op (monotonicity not enforced yet).
    """
    pass


def finalize_claim_registration(claim_id: str, trust_class: str) -> None:
    """
    Finalize a claim registration with its trust class.

    v0 stub: No-op (registration not tracked yet).
    """
    pass
