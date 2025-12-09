"""
Compatibility shim for relocated monotone guard module.

The canonical implementation lives in backend.ledger.monotone_guard.
"""

from backend.ledger.monotone_guard import (  # re-exported shims
    InvariantViolation,
    check_monotone_invariants,
    check_monotone_ledger,
)

__all__ = [
    "InvariantViolation",
    "check_monotone_invariants",
    "check_monotone_ledger",
]
