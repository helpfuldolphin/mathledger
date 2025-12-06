"""
Trace Schema for U2 Uplift Experiments

Defines dataclasses for structured trace events.

PHASE II — NOT USED IN PHASE I
"""

from dataclasses import dataclass
from typing import Optional


# Schema version for trace events
TRACE_SCHEMA_VERSION = "1.0"


@dataclass
class SessionStartEvent:
    """Event logged at session start."""
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    config_hash: str
    total_cycles: int
    initial_seed: int


@dataclass
class SessionEndEvent:
    """Event logged at session end."""
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    manifest_hash: Optional[str]
    ht_series_hash: Optional[str]
    total_cycles: int
    completed_cycles: int


@dataclass
class CycleTelemetryEvent:
    """Event logged for each cycle."""
    cycle: int
    slice: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    label: str = "PHASE II — NOT USED IN PHASE I"


@dataclass
class PolicyWeightUpdateEvent:
    """Event logged when policy weights are updated."""
    cycle: int
    item: str
    success_rate: float
    attempt_count: int


@dataclass
class SnapshotSavedEvent:
    """Event logged when snapshot is saved."""
    cycle: int
    path: str
    hash: str


@dataclass
class ExecutionErrorEvent:
    """Event logged when execution error occurs."""
    cycle: int
    item: str
    error: str
