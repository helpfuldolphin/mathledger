"""Deprecated canonicalization module shim.

TODO(remove after 2025-12-01): Deprecation window for imports that still
reference `backend.logic.canon`. Prefer `normalization.canon`.
"""
import warnings

warnings.warn(
    "backend.logic.canon is deprecated; import normalization.canon instead.",
    DeprecationWarning,
    stacklevel=2,
)

from normalization.canon import *  # noqa: F401,F403
