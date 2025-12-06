"""
U2 Planner Module

Provides:
- Deterministic search runtime
- Frontier management with beam allocation
- Policy-driven candidate selection
- Snapshot and replay support
- Trace logging for RFL evidence
"""

from .frontier import FrontierManager, BeamAllocator, FrontierCandidate
from .logging import U2TraceLogger, load_experiment_trace, verify_trace_determinism
from .policy import SearchPolicy, BaselinePolicy, RFLPolicy, create_policy
from .runner import U2Runner, U2Config, CycleResult, TracedExperimentContext, run_with_traces
from .schema import EventType, TraceEvent, CycleTrace, ExperimentTrace
from .snapshots import (
    SnapshotData,
    SnapshotError,
    SnapshotValidationError,
    SnapshotCorruptionError,
    NoSnapshotFoundError,
    save_snapshot,
    load_snapshot,
    find_latest_snapshot,
    rotate_snapshots,
)

__all__ = [
    # Frontier
    "FrontierManager",
    "BeamAllocator",
    "FrontierCandidate",
    
    # Logging
    "U2TraceLogger",
    "load_experiment_trace",
    "verify_trace_determinism",
    
    # Policy
    "SearchPolicy",
    "BaselinePolicy",
    "RFLPolicy",
    "create_policy",
    
    # Runner
    "U2Runner",
    "U2Config",
    "CycleResult",
    "TracedExperimentContext",
    "run_with_traces",
    
    # Schema
    "EventType",
    "TraceEvent",
    "CycleTrace",
    "ExperimentTrace",
    
    # Snapshots
    "SnapshotData",
    "SnapshotError",
    "SnapshotValidationError",
    "SnapshotCorruptionError",
    "NoSnapshotFoundError",
    "save_snapshot",
    "load_snapshot",
    "find_latest_snapshot",
    "rotate_snapshots",
]
