# PHASE II — NOT USED IN PHASE I
# Trace logging for U2 experiments

import json
from pathlib import Path
from typing import Any, Optional, Set
from . import schema

# Core event types that are always logged
CORE_EVENTS = {
    "session_start",
    "session_end",
    "cycle_telemetry",
}

# All available event types
ALL_EVENT_TYPES = CORE_EVENTS | {
    "policy_weight_update",
    "snapshot_saved",
    "calibration_complete",
}


class U2TraceLogger:
    """
    Structured trace logger for U2 experiments.
    
    Logs events as JSONL for downstream analysis.
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
            fail_soft: If True, suppress logging errors; if False, raise
            enabled_events: Set of event types to log (None = all events)
        """
        self.log_path = log_path
        self.fail_soft = fail_soft
        self.enabled_events = enabled_events if enabled_events is not None else ALL_EVENT_TYPES
        self._file = None
    
    def __enter__(self):
        """Open log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "w")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close log file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def _log_event(self, event_type: str, event_data: dict):
        """Internal method to log an event."""
        if event_type not in self.enabled_events:
            return
        
        try:
            record = {
                "event": event_type,
                "data": event_data,
            }
            if self._file:
                self._file.write(json.dumps(record) + "\n")
                self._file.flush()
        except Exception as e:
            if not self.fail_soft:
                raise
            # Silently ignore logging errors in fail_soft mode
    
    def log_session_start(self, event: schema.SessionStartEvent):
        """Log session start event."""
        self._log_event("session_start", {
            "run_id": event.run_id,
            "slice_name": event.slice_name,
            "mode": event.mode,
            "schema_version": event.schema_version,
            "config_hash": event.config_hash,
            "total_cycles": event.total_cycles,
            "initial_seed": event.initial_seed,
        })
    
    def log_session_end(self, event: schema.SessionEndEvent):
        """Log session end event."""
        self._log_event("session_end", {
            "run_id": event.run_id,
            "slice_name": event.slice_name,
            "mode": event.mode,
            "schema_version": event.schema_version,
            "manifest_hash": event.manifest_hash,
            "ht_series_hash": event.ht_series_hash,
            "total_cycles": event.total_cycles,
            "completed_cycles": event.completed_cycles,
        })
    
    def log_cycle_telemetry(self, cycle: int, telemetry: dict):
        """Log cycle telemetry."""
        self._log_event("cycle_telemetry", {
            "cycle": cycle,
            **telemetry,
        })
