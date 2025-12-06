"""
Deprecated block sealing shim.

All functionality has been moved to ledger.blocking.
This module re-exports for backwards compatibility.
"""

import warnings

from ledger.blocking import (  # noqa: F401
    seal_block,
    seal_block_with_dual_roots,
)

warnings.warn(
    "backend.ledger.blocking is deprecated; use ledger.blocking.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "seal_block",
    "seal_block_with_dual_roots",
]
