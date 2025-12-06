"""Deprecated truth-table shim.

TODO(remove after 2025-12-01): Deprecation window for imports that still
reference `backend.logic.truthtab`. Prefer `normalization.truthtab`.
"""
import warnings

warnings.warn(
    "backend.logic.truthtab is deprecated; import normalization.truthtab instead.",
    DeprecationWarning,
    stacklevel=2,
)

from normalization.truthtab import *  # noqa: F401,F403
