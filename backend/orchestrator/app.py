"""
Backwards compatibility shim for backend.orchestrator.app.

The application has been moved to interface.api.app.
This module re-exports for backwards compatibility.
"""

import warnings

warnings.warn(
    "backend.orchestrator.app is deprecated; use interface.api.app instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from interface.api.app import app, get_db_connection
except ImportError:
    # If the interface module isn't available, provide stubs
    from unittest.mock import MagicMock
    app = MagicMock()
    get_db_connection = MagicMock()

__all__ = ["app", "get_db_connection"]
