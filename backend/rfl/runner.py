"""Deprecated RFL runner shim.

This module is deprecated. Import from rfl.runner directly instead.
"""

import warnings

from rfl.runner import *  # noqa: F401,F403

warnings.warn(
    "backend.rfl.runner is deprecated; import rfl.runner instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RFLRunner",
    "RflResult",
    "RunLedgerEntry",
    "AttestationInput",
]
