"""Deprecated coverage shim.

All functionality has been moved to rfl.coverage.
This module re-exports for backwards compatibility.
"""

import warnings

from rfl.coverage import (  # noqa: F401
    CoverageMetrics,
    CoverageTracker,
    load_baseline_from_db,
)

warnings.warn(
    "backend.rfl.coverage is deprecated; use rfl.coverage instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CoverageMetrics",
    "CoverageTracker",
    "load_baseline_from_db",
]
