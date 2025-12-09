"""
U2 Planner Module

Provides:
- Deterministic search runtime
- Frontier management with beam allocation
- Policy-driven candidate selection
- Snapshot and replay support
- Trace logging for RFL evidence
- Runtime safety enforcement (Neural Link)
"""

from .frontier import FrontierManager, BeamAllocator, FrontierCandidate
from .logging import U2TraceLogger, load_experiment_trace, verify_trace_determinism
from .policy import SearchPolicy, BaselinePolicy, RFLPolicy, create_policy
from .runner import U2Runner, U2Config, CycleResult, TracedExperimentContext, run_with_traces
from .safety import (
    U2SafetyContext,
    SafetyEnvelope,
    GateDecision,
    evaluate_hard_gate_decision,
    validate_safety_envelope,
)
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
    
    # Safety (Neural Link)
    "U2SafetyContext",
    "SafetyEnvelope",
    "GateDecision",
    "evaluate_hard_gate_decision",
    "validate_safety_envelope",
    
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
