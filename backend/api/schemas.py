"""Deprecated API schema shim."""

import warnings

from interface.api.schemas import *  # noqa: F401,F403

warnings.warn(
    "backend.api.schemas is deprecated; import interface.api.schemas instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export __all__ from the canonical module
try:
    from interface.api.schemas import __all__ as _all
    __all__ = list(_all)
except ImportError:
    __all__ = []
