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
from .telemetry import (
    TelemetryReport,
    extract_telemetry_from_trace,
    compare_telemetry,
)
from .runner import (
    U2Runner,
    U2Config,
    CycleResult,
    TracedExperimentContext,
    U2SafetyContext,
    U2Snapshot,
    run_with_traces,
    run_u2_experiment,
    safe_eval_expression,
    save_u2_snapshot,
    load_u2_snapshot,
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
from .safety_slo import (
    SafetyStatus,
    SafetyEnvelope,
    SafetySLOPoint,
    SafetySLOTimeline,
    ScenarioSafetyCell,
    ScenarioSafetyMatrix,
    SafetySLOEvaluation,
    build_safety_slo_timeline,
    build_scenario_safety_matrix,
    evaluate_safety_slo,
    MAX_BLOCK_RATE,
    MAX_WARN_RATE,
    MAX_PERF_FAILURE_RATE,
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
    
    # Telemetry
    "TelemetryReport",
    "extract_telemetry_from_trace",
    "compare_telemetry",
    
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
    "U2SafetyContext",
    "U2Snapshot",
    "run_with_traces",
    "run_u2_experiment",
    "safe_eval_expression",
    "save_u2_snapshot",
    "load_u2_snapshot",
    
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
    
    # Safety SLO
    "SafetyStatus",
    "SafetyEnvelope",
    "SafetySLOPoint",
    "SafetySLOTimeline",
    "ScenarioSafetyCell",
    "ScenarioSafetyMatrix",
    "SafetySLOEvaluation",
    "build_safety_slo_timeline",
    "build_scenario_safety_matrix",
    "evaluate_safety_slo",
    "MAX_BLOCK_RATE",
    "MAX_WARN_RATE",
    "MAX_PERF_FAILURE_RATE",
]
