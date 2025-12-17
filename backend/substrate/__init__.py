"""Backend substrate shim.

Re-exports from canonical substrate module for backwards compatibility.
"""

import warnings

warnings.warn(
    "backend.substrate is deprecated; import substrate directly instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical substrate package
try:
    from substrate import *  # noqa: F401,F403
    from substrate import __all__ as _substrate_all
except ImportError:
    _substrate_all = []

__all__ = list(_substrate_all)
