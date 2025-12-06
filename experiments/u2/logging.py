"""
U2 Trace Logging - Structured Event Logging

Provides structured logging for U2 experiments with event filtering
and fail-soft behavior.
"""

import json
from typing import Any, Dict, Optional, Set, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
else:
    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self
from pathlib import Path

from . import schema


# Re-export event constants
CORE_EVENTS = schema.CORE_EVENTS
ALL_EVENT_TYPES = schema.ALL_EVENT_TYPES


class U2TraceLogger:
    """
    Structured trace logger for U2 experiments.
    
    Logs events as JSON lines with optional event filtering.
    Supports fail-soft mode for resilience.
    """
    
    def __init__(
        self,
        log_path: Path,
        fail_soft: bool = True,
        enabled_events: Optional[Set[str]] = None,
    ):
        """
        Initialize trace logger.
        
        Args:
            log_path: Path to log file (JSONL format)
            fail_soft: If True, suppress logging errors; if False, raise them
            enabled_events: Set of event types to log (None = all events)
        """
        self.log_path = log_path
        self.fail_soft = fail_soft
        self.enabled_events = enabled_events if enabled_events is not None else ALL_EVENT_TYPES
        self._file: Optional[TextIO] = None
    
    def __enter__(self) -> Self:
        """Open log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def _should_log(self, event_type: str) -> bool:
        """Check if event type should be logged."""
        return event_type in self.enabled_events
    
    def _write_event(self, event_dict: Dict[str, Any]) -> None:
        """Write event to log file."""
        if self._file is None:
            if not self.fail_soft:
                raise RuntimeError("Logger not opened (use context manager)")
            return
        
        try:
            json_line = json.dumps(event_dict)
            self._file.write(json_line + "\n")
            self._file.flush()
        except Exception as e:
            if not self.fail_soft:
                raise
            # Silently ignore errors in fail-soft mode
    
    def log_session_start(self, event: schema.SessionStartEvent) -> None:
        """Log session start event."""
        if not self._should_log(schema.EVENT_SESSION_START):
            return
        self._write_event(event.model_dump())
    
    def log_session_end(self, event: schema.SessionEndEvent) -> None:
        """Log session end event."""
        if not self._should_log(schema.EVENT_SESSION_END):
            return
        self._write_event(event.model_dump())
    
    def log_cycle_begin(self, event: schema.CycleBeginEvent) -> None:
        """Log cycle begin event."""
        if not self._should_log(schema.EVENT_CYCLE_BEGIN):
            return
        self._write_event(event.model_dump())
    
    def log_cycle_end(self, event: schema.CycleEndEvent) -> None:
        """Log cycle end event."""
        if not self._should_log(schema.EVENT_CYCLE_END):
            return
        self._write_event(event.model_dump())
    
    def log_cycle_telemetry(self, event: schema.CycleTelemetryEvent) -> None:
        """Log cycle telemetry event."""
        if not self._should_log(schema.EVENT_CYCLE_TELEMETRY):
            return
        self._write_event(event.model_dump())
    
    def log_policy_weight_update(self, event: schema.PolicyWeightUpdateEvent) -> None:
        """Log policy weight update event."""
        if not self._should_log(schema.EVENT_POLICY_WEIGHT_UPDATE):
            return
        self._write_event(event.model_dump())
    
    def log_snapshot_saved(self, event: schema.SnapshotSavedEvent) -> None:
        """Log snapshot saved event."""
        if not self._should_log(schema.EVENT_SNAPSHOT_SAVED):
            return
        self._write_event(event.model_dump())
    
    def log_eval_lint(self, event: schema.EvalLintEvent) -> None:
        """Log eval lint event."""
        if not self._should_log(schema.EVENT_EVAL_LINT):
            return
        self._write_event(event.model_dump())
