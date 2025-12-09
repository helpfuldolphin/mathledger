# PHASE II — NOT RUN IN PHASE I
"""
U2 Trace Event Schema Definitions

STATUS: PHASE II — NOT RUN IN PHASE I
Used by experiments.u2.logging.U2TraceLogger to emit structured telemetry.

All event types are frozen dataclasses for immutability and hashability.
The TRACE_SCHEMA_VERSION should be bumped when breaking changes occur.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal


TRACE_SCHEMA_VERSION = "u2-trace-1.0.0"


@dataclass(frozen=True)
class CandidateOrderingEvent:
    """Records the ordered list of candidates considered in a cycle.
    
    For baseline mode: ordering is random (seeded).
    For RFL mode: ordering is policy-scored.
    """
    cycle: int
    slice_name: str
    mode: Literal["baseline", "rfl"]
    ordering: tuple  # Tuple of dicts: (rank, item_id, hash, score, selected_flag)


@dataclass(frozen=True)
class ScoringFeaturesEvent:
    """Records feature extraction used for RFL policy scoring.
    
    Each feature dict contains: item_id, hash, len, depth, success_feat, final_score.
    """
    cycle: int
    slice_name: str
    mode: Literal["baseline", "rfl"]
    features: tuple  # Tuple of feature dicts


@dataclass(frozen=True)
class PolicyWeightUpdateEvent:
    """Records policy weight changes after cycle feedback.
    
    Only emitted in RFL mode when policy_update_applied is True.
    """
    cycle: int
    slice_name: str
    mode: Literal["rfl"]
    weights_before: Dict[str, float]
    weights_after: Dict[str, float]
    reward: float
    verified_count: int
    target: int


@dataclass(frozen=True)
class BudgetConsumptionEvent:
    """Records resource budget tracking per cycle."""
    cycle: int
    slice_name: str
    mode: str
    candidates_considered: int
    candidates_limit: Optional[int]
    budget_exhausted: bool


@dataclass(frozen=True)
class CycleDurationEvent:
    """Records timing telemetry for a cycle.
    
    duration_ms: Total cycle wall-clock time.
    substrate_duration_ms: Time spent in FO substrate call (if available).
    """
    cycle: int
    slice_name: str
    mode: str
    duration_ms: float
    substrate_duration_ms: Optional[float]


@dataclass(frozen=True)
class SubstrateResultEvent:
    """Records the outcome from the FO substrate simulation."""
    cycle: int
    slice_name: str
    mode: str
    item_id: str
    seed: int
    result: str  # "VERIFIED" or "FAILED_TO_PROVE"
    verified_hashes: tuple  # Tuple of hash strings


@dataclass(frozen=True)
class CycleTelemetryEvent:
    """Full cycle telemetry record (existing format compatibility).
    
    raw_record contains the complete JSONL-compatible payload that would
    be written to the main results file.
    """
    cycle: int
    slice_name: str
    mode: str
    raw_record: Dict[str, Any]


@dataclass(frozen=True)
class SessionStartEvent:
    """Emitted once at experiment initialization."""
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    config_hash: str
    total_cycles: int
    initial_seed: int


@dataclass(frozen=True)
class SessionEndEvent:
    """Emitted once at experiment completion."""
    run_id: str
    slice_name: str
    mode: str
    schema_version: str
    manifest_hash: Optional[str]
    ht_series_hash: Optional[str]
    total_cycles: int
    completed_cycles: int
    lean_failure_summary: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SnapshotPlanEvent:
    """Emitted during orchestration to indicate snapshot-based run planning decision.
    
    This event is emitted for the global health surface to track snapshot
    planning decisions across multiple runs.
    """
    status: Literal["NO_ACTION", "RESUME", "NEW_RUN"]
    preferred_run_id: Optional[str]
    preferred_snapshot_path: Optional[str]
    total_runs_analyzed: int


@dataclass(frozen=True)
class HashChainEntryEvent:
    """Optional hash-chained log entry for tamper-evidence.
    
    current_hash = SHA256(prev_hash || payload_json)
    """
    cycle: int
    slice_name: str
    mode: str
    entry_index: int
    prev_hash: str
    current_hash: str
    payload_hash: str


# Type alias for any trace event
TraceEvent = (
    CandidateOrderingEvent
    | ScoringFeaturesEvent
    | PolicyWeightUpdateEvent
    | BudgetConsumptionEvent
    | CycleDurationEvent
    | SubstrateResultEvent
    | CycleTelemetryEvent
    | SessionStartEvent
    | SessionEndEvent
    | HashChainEntryEvent
    | SnapshotPlanEvent
)

