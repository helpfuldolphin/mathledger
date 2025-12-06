"""Deprecated tautology helper shim.

TODO(remove after 2025-12-01): Deprecation window for imports that still
reference `backend.logic.taut`. Prefer `normalization.taut`.
"""
import warnings

warnings.warn(
    "backend.logic.taut is deprecated; import normalization.taut instead.",
    DeprecationWarning,
    stacklevel=2,
)

from normalization.taut import *  # noqa: F401,F403
