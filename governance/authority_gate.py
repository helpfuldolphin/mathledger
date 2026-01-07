"""
Authority Gate - Stub Implementation

This module provides stubs for authority gate invariants.
Full implementation is Phase II scope.

For v0: All functions are no-ops or return safe defaults.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AuthorityUpdateRequest:
    """Request to update authority state."""
    epoch: int
    claim_id: str
    trust_class: str
    payload: Dict[str, Any]


class SilentAuthorityViolation(Exception):
    """Raised when a silent authority violation is detected."""
    pass


def require_epoch_root(epoch: int) -> None:
    """
    Validate that epoch root exists.

    v0 stub: No-op (no epoch root enforcement yet).
    """
    pass


def get_authority_violations() -> List[str]:
    """
    Return list of authority violations.

    v0 stub: Returns empty list (no violations tracked yet).
    """
    return []
