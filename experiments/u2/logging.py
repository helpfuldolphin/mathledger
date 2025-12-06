"""
Trace Logging for U2 Uplift Experiments

Provides structured telemetry logging with event filtering.

PHASE II — NOT USED IN PHASE I
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set, TextIO


# Event type constants
CORE_EVENTS = {
    'session_start',
    'session_end',
    'cycle_telemetry',
}

ALL_EVENT_TYPES = CORE_EVENTS | {
    'policy_weight_update',
    'snapshot_saved',
    'execution_error',
}


class U2TraceLogger:
    """
    Structured trace logger for U2 experiments.
    
    Logs events as JSONL with optional event filtering.
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
            log_path: Path to JSONL log file
            fail_soft: If True, suppress logging errors
            enabled_events: Set of event types to log (None = all)
        """
        self.log_path = log_path
        self.fail_soft = fail_soft
        self.enabled_events = enabled_events if enabled_events is not None else ALL_EVENT_TYPES
        self._file: Optional[TextIO] = None
    
    def __enter__(self) -> 'U2TraceLogger':
        """Open log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, 'a')
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Close log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        return False
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Log an event if enabled.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
        """
        if event_type not in self.enabled_events:
            return
        
        if self._file is None:
            if not self.fail_soft:
                raise RuntimeError("Logger not initialized (use context manager)")
            return
        
        try:
            event = {
                'event_type': event_type,
                **event_data
            }
            self._file.write(json.dumps(event) + '\n')
            self._file.flush()
        except Exception as e:
            if not self.fail_soft:
                raise RuntimeError(f"Failed to log event: {e}")
    
    def log_session_start(self, event: Any) -> None:
        """Log session start event."""
        if hasattr(event, '__dict__'):
            data = event.__dict__
        else:
            data = event
        self._log_event('session_start', data)
    
    def log_session_end(self, event: Any) -> None:
        """Log session end event."""
        if hasattr(event, '__dict__'):
            data = event.__dict__
        else:
            data = event
        self._log_event('session_end', data)
    
    def log_cycle_telemetry(self, cycle: int, telemetry: Dict[str, Any]) -> None:
        """Log cycle telemetry."""
        self._log_event('cycle_telemetry', {'cycle': cycle, **telemetry})
    
    def log_policy_update(self, cycle: int, weights: Dict[str, float]) -> None:
        """Log policy weight update."""
        self._log_event('policy_weight_update', {'cycle': cycle, 'weights': weights})
    
    def log_snapshot_saved(self, cycle: int, path: str) -> None:
        """Log snapshot saved."""
        self._log_event('snapshot_saved', {'cycle': cycle, 'path': path})
    
    def log_execution_error(self, cycle: int, error: str) -> None:
        """Log execution error."""
        self._log_event('execution_error', {'cycle': cycle, 'error': error})
