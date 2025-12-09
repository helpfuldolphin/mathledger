"""Deprecated bootstrap_stats shim.

All functionality has been moved to rfl.bootstrap_stats.
This module re-exports for backwards compatibility.
"""

import warnings

from rfl.bootstrap_stats import (  # noqa: F401
    BootstrapResult,
    bootstrap_bca,
    bootstrap_percentile,
    compute_coverage_ci,
    compute_uplift_ci,
    verify_metabolism,
)

warnings.warn(
    "backend.rfl.bootstrap_stats is deprecated; use rfl.bootstrap_stats instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BootstrapResult",
    "bootstrap_bca",
    "bootstrap_percentile",
    "compute_coverage_ci",
    "compute_uplift_ci",
    "verify_metabolism",
]
