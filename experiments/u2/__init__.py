# PHASE II — NOT RUN IN PHASE I
#
# U2 Uplift Experiment Package
#
# This package provides:
#   - Core runner infrastructure (U2Runner, U2Config, CycleResult)
#   - Structured logging and trace telemetry (U2TraceLogger)
#   - Snapshot/restore for long experiments
#   - Per-cycle execution orchestration
#
# It does NOT modify Phase I behavior or artifacts.

"""
U2 Uplift Experiment Package

STATUS: PHASE II — NOT RUN IN PHASE I

Modules:
    - schema: Event dataclass definitions (frozen, versioned)
    - logging: U2TraceLogger for append-only JSONL output
    - runner: Core runner and trace logging wrapper
    - snapshots: Snapshot save/restore for experiment state
    - runtime: Per-cycle orchestration utilities
    - inspector: Trace log analysis and hotspot detection
    - trace_correlator: Correlate traces with manifest/budget data
    - trace_health: Multi-run health snapshots and trend analysis
"""

from .schema import TRACE_SCHEMA_VERSION
from .logging import U2TraceLogger, CORE_EVENTS, ALL_EVENT_TYPES
from .inspector import (
    TraceLogInspector,
    TraceLogSummary,
    EventHistogram,
    ValidationError,
    LastMileReport,
    HotspotEntry,
    HotspotReport,
    iter_events,
    parse_cycle_range,
)
from .trace_correlator import (
    TraceCorrelator,
    CorrelationSummary,
)
from .trace_health import (
    TRACE_HEALTH_SCHEMA_VERSION,
    TraceHealthSnapshot,
    TraceTrend,
    GlobalTraceHealth,
    build_trace_health_snapshot,
    build_trace_trend,
    summarize_trace_for_global_health,
    analyze_trace_health,
)
from .policy import summarize_lean_failures_for_global_health
from .runner import (
    U2Config,
    U2Runner,
    CycleResult,
    TracedExperimentContext,
    run_with_traces,
    compute_config_hash,
)
from .snapshots import (
    SnapshotData,
    SnapshotValidationError,
    SnapshotCorruptionError,
    NoSnapshotFoundError,
    load_snapshot,
    save_snapshot,
    find_latest_snapshot,
    list_snapshots,
    rotate_snapshots,
)

__all__ = [
    # Schema
    "TRACE_SCHEMA_VERSION",
    # Logging
    "U2TraceLogger",
    "CORE_EVENTS",
    "ALL_EVENT_TYPES",
    # Inspector
    "TraceLogInspector",
    "TraceLogSummary",
    "EventHistogram",
    "ValidationError",
    "LastMileReport",
    "HotspotEntry",
    "HotspotReport",
    "iter_events",
    "parse_cycle_range",
    # Correlator
    "TraceCorrelator",
    "CorrelationSummary",
    # Trace Health
    "TRACE_HEALTH_SCHEMA_VERSION",
    "TraceHealthSnapshot",
    "TraceTrend",
    "GlobalTraceHealth",
    "build_trace_health_snapshot",
    "build_trace_trend",
    "summarize_trace_for_global_health",
    "analyze_trace_health",
    "summarize_lean_failures_for_global_health",
    # Runner
    "U2Config",
    "U2Runner", 
    "CycleResult",
    "TracedExperimentContext",
    "run_with_traces",
    "compute_config_hash",
    # Snapshots
    "SnapshotData",
    "SnapshotValidationError",
    "SnapshotCorruptionError",
    "NoSnapshotFoundError",
    "load_snapshot",
    "save_snapshot",
    "find_latest_snapshot",
    "list_snapshots",
    "rotate_snapshots",
]
