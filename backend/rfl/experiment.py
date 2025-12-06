"""Deprecated experiment shim.

All functionality has been moved to rfl.experiment.
This module re-exports for backwards compatibility.
"""

import warnings

from rfl.experiment import (  # noqa: F401
    ExperimentResult,
    RFLExperiment,
)

warnings.warn(
    "backend.rfl.experiment is deprecated; use rfl.experiment instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ExperimentResult",
    "RFLExperiment",
]
