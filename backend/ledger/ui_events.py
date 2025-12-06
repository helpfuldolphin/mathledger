"""
Deprecated UI event capture shim.

All functionality has been moved to ledger.ui_events.
This module re-exports for backwards compatibility.
"""

import warnings

from ledger.ui_events import (  # noqa: F401
    UIEventRecord,
    UIEventStore,
    capture_ui_event,
    capture_ui_events,
    materialize_ui_artifacts,
    snapshot_ui_events,
    ui_event_store,
)

warnings.warn(
    "backend.ledger.ui_events is deprecated; import ledger.ui_events instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "UIEventRecord",
    "UIEventStore",
    "capture_ui_event",
    "capture_ui_events",
    "materialize_ui_artifacts",
    "snapshot_ui_events",
    "ui_event_store",
]
