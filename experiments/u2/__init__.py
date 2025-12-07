"""
U2 Planner Module

Provides:
- Deterministic search runtime
- Frontier management with beam allocation
- Policy-driven candidate selection
- Snapshot and replay support
- Trace logging for RFL evidence
- Multi-run evidence fusion for promotion readiness
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
from .evidence_fusion import (
    fuse_evidence_summaries,
    inject_multi_run_fusion_into_evidence,
    FusedEvidenceSummary,
    PassStatus,
    DeterminismViolation,
    MissingArtifact,
    ConflictingSliceName,
    RunOrderingAnomaly,
)
from .telemetry import (
    TelemetryReport,
    extract_telemetry_from_trace,
    compare_telemetry,
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
    
    # Evidence Fusion
    "fuse_evidence_summaries",
    "inject_multi_run_fusion_into_evidence",
    "FusedEvidenceSummary",
    "PassStatus",
    "DeterminismViolation",
    "MissingArtifact",
    "ConflictingSliceName",
    "RunOrderingAnomaly",
    
    # Telemetry
    "TelemetryReport",
    "extract_telemetry_from_trace",
    "compare_telemetry",
]
