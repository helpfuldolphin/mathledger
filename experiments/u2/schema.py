# PHASE II — NOT USED IN PHASE I
# Trace schema definitions for U2 experiments

from typing import Any, Dict, Optional
from dataclasses import dataclass

TRACE_SCHEMA_VERSION = "1.0.0"


@dataclass
class SessionStartEvent:
    """Event emitted at the start of an experiment session."""
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    config_hash: str
    total_cycles: int
    initial_seed: int


@dataclass
class SessionEndEvent:
    """Event emitted at the end of an experiment session."""
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
    """Event containing full telemetry for a cycle."""
    cycle: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Any
    success: bool
    label: str
