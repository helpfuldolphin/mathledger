"""
U2 Trace Event Schema - Phase III Telemetry

Defines typed schema for trace events in U2 experiments.
All events are versioned and follow strict typing for reproducibility.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Schema version for trace events
TRACE_SCHEMA_VERSION = "1.0.0"

# Event type definitions
EVENT_SESSION_START = "session_start"
EVENT_SESSION_END = "session_end"
EVENT_CYCLE_BEGIN = "cycle_begin"
EVENT_CYCLE_END = "cycle_end"
EVENT_CYCLE_TELEMETRY = "cycle_telemetry"
EVENT_POLICY_WEIGHT_UPDATE = "policy_weight_update"
EVENT_SNAPSHOT_SAVED = "snapshot_saved"
EVENT_EVAL_LINT = "eval_lint"

# Core events logged by default
CORE_EVENTS = {
    EVENT_SESSION_START,
    EVENT_SESSION_END,
    EVENT_CYCLE_TELEMETRY,
}

# All available event types
ALL_EVENT_TYPES = {
    EVENT_SESSION_START,
    EVENT_SESSION_END,
    EVENT_CYCLE_BEGIN,
    EVENT_CYCLE_END,
    EVENT_CYCLE_TELEMETRY,
    EVENT_POLICY_WEIGHT_UPDATE,
    EVENT_SNAPSHOT_SAVED,
    EVENT_EVAL_LINT,
}


class SessionStartEvent(BaseModel):
    """Event emitted at the start of an experiment session."""
    event_type: Literal["session_start"] = Field(default="session_start")
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    config_hash: str
    total_cycles: int
    initial_seed: int


class SessionEndEvent(BaseModel):
    """Event emitted at the end of an experiment session."""
    event_type: Literal["session_end"] = Field(default="session_end")
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    manifest_hash: Optional[str] = None
    ht_series_hash: Optional[str] = None
    total_cycles: int
    completed_cycles: int


class CycleBeginEvent(BaseModel):
    """Event emitted at the start of a cycle."""
    event_type: Literal["cycle_begin"] = Field(default="cycle_begin")
    cycle_index: int
    timestamp_ms: float


class CycleEndEvent(BaseModel):
    """Event emitted at the end of a cycle."""
    event_type: Literal["cycle_end"] = Field(default="cycle_end")
    cycle_index: int
    timestamp_ms: float
    duration_ms: float


class CycleTelemetryEvent(BaseModel):
    """Full telemetry record for a completed cycle."""
    event_type: Literal["cycle_telemetry"] = Field(default="cycle_telemetry")
    cycle_index: int
    telemetry: Dict[str, Any]


class PolicyWeightUpdateEvent(BaseModel):
    """Event emitted when RFL policy weights are updated."""
    event_type: Literal["policy_weight_update"] = Field(default="policy_weight_update")
    cycle_index: int
    item: str
    old_weight: float
    new_weight: float
    success: bool


class SnapshotSavedEvent(BaseModel):
    """Event emitted when a snapshot is saved."""
    event_type: Literal["snapshot_saved"] = Field(default="snapshot_saved")
    cycle_index: int
    snapshot_path: str
    snapshot_hash: str


class EvalLintEvent(BaseModel):
    """Event emitted for safe eval lint results."""
    event_type: Literal["eval_lint"] = Field(default="eval_lint")
    cycle_index: int
    expression: str
    is_safe: bool
    issues: List[str] = Field(default_factory=list)
