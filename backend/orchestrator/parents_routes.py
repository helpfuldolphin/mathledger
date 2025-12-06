"""
DEPRECATED: kept only for legacy callers; will be removed after VCP 2.2 Wave 1.

This module re-exports from the canonical interface.api.routes.parents namespace.
New code should import directly from interface.api.routes.parents instead.

Reference: MathLedger Whitepaper §6.2 (Parent Provenance API).
"""

# Re-export from canonical namespace
from interface.api.routes.parents import (
    parents_router,
)

__all__ = ["parents_router"]
