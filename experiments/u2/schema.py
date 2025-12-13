"""
U2 Planner Schema Definitions

Defines canonical event types and trace formats for:
- RFL Evidence Packs
- Determinism verification
- Replay debugging
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class EventType(str, Enum):
    """Canonical event types for U2 planner traces."""
    
    # Lifecycle events
    EXPERIMENT_START = "experiment.start"
    EXPERIMENT_END = "experiment.end"
    CYCLE_START = "cycle.start"
    CYCLE_END = "cycle.end"
    
    # Search events
    FRONTIER_PUSH = "frontier.push"
    FRONTIER_POP = "frontier.pop"
    FRONTIER_PRUNE = "frontier.prune"
    BEAM_ALLOCATE = "beam.allocate"
    BEAM_EXHAUST = "beam.exhaust"
    
    # Derivation events
    DERIVE_START = "derive.start"
    DERIVE_SUCCESS = "derive.success"
    DERIVE_FAILURE = "derive.failure"
    DERIVE_TIMEOUT = "derive.timeout"
    
    # Policy events
    POLICY_RANK = "policy.rank"
    POLICY_SELECT = "policy.select"
    
    # Snapshot events
    SNAPSHOT_SAVE = "snapshot.save"
    SNAPSHOT_LOAD = "snapshot.load"
    SNAPSHOT_VERIFY = "snapshot.verify"
    
    # Budget events
    BUDGET_CHECK = "budget.check"
    BUDGET_EXCEEDED = "budget.exceeded"


@dataclass
class TraceEvent:
    """
    Canonical trace event for U2 planner.
    
    INVARIANTS:
    - timestamp_ms is monotonic within a cycle
    - event_type is from EventType enum
    - data is JSON-serializable
    """
    
    timestamp_ms: int
    event_type: EventType
    cycle: int
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "event_type": self.event_type.value,
            "cycle": self.cycle,
            "data": self.data,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TraceEvent':
        """Reconstruct from dict."""
        return cls(
            timestamp_ms=d["timestamp_ms"],
            event_type=EventType(d["event_type"]),
            cycle=d["cycle"],
            data=d.get("data", {}),
        )


@dataclass
class CycleTrace:
    """
    Complete trace for a single cycle.
    
    Used for:
    - Determinism verification (hash of canonical trace)
    - Replay debugging
    - RFL evidence generation
    """
    
    cycle: int
    seed: str
    events: List[TraceEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: TraceEvent) -> None:
        """Add event to trace."""
        self.events.append(event)
    
    def canonicalize(self) -> str:
        """
        Generate canonical string representation for hashing.
        
        INVARIANT: Same cycle execution produces same canonical string
        regardless of machine, OS, or Python version.
        
        Returns:
            Canonical string (sorted keys, stable formatting)
        """
        import json
        
        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda e: e.timestamp_ms)
        
        # Create canonical dict
        canonical = {
            "cycle": self.cycle,
            "seed": self.seed,
            "events": [e.to_dict() for e in sorted_events],
        }
        
        # JSON with sorted keys and stable formatting
        return json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    
    def hash(self) -> str:
        """
        Compute deterministic hash of cycle trace.
        
        Returns:
            SHA-256 hex digest
        """
        import hashlib
        canonical = self.canonicalize()
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "cycle": self.cycle,
            "seed": self.seed,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
            "trace_hash": self.hash(),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CycleTrace':
        """Reconstruct from dict."""
        trace = cls(
            cycle=d["cycle"],
            seed=d["seed"],
            metadata=d.get("metadata", {}),
        )
        trace.events = [TraceEvent.from_dict(e) for e in d.get("events", [])]
        return trace


@dataclass
class ExperimentTrace:
    """
    Complete trace for an experiment run.
    
    Contains:
    - Experiment metadata (slice, mode, seed)
    - Per-cycle traces
    - Aggregate statistics
    """
    
    experiment_id: str
    slice_name: str
    mode: str
    master_seed: str
    cycles: List[CycleTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_cycle(self, cycle_trace: CycleTrace) -> None:
        """Add cycle trace to experiment."""
        self.cycles.append(cycle_trace)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "master_seed": self.master_seed,
            "cycles": [c.to_dict() for c in self.cycles],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentTrace':
        """Reconstruct from dict."""
        trace = cls(
            experiment_id=d["experiment_id"],
            slice_name=d["slice_name"],
            mode=d["mode"],
            master_seed=d["master_seed"],
            metadata=d.get("metadata", {}),
        )
        trace.cycles = [CycleTrace.from_dict(c) for c in d.get("cycles", [])]
        return trace


# Event type groups for filtering
CORE_EVENTS = {
    EventType.CYCLE_START,
    EventType.CYCLE_END,
    EventType.DERIVE_SUCCESS,
    EventType.DERIVE_FAILURE,
}

SEARCH_EVENTS = {
    EventType.FRONTIER_PUSH,
    EventType.FRONTIER_POP,
    EventType.FRONTIER_PRUNE,
    EventType.BEAM_ALLOCATE,
    EventType.BEAM_EXHAUST,
}

POLICY_EVENTS = {
    EventType.POLICY_RANK,
    EventType.POLICY_SELECT,
}

SNAPSHOT_EVENTS = {
    EventType.SNAPSHOT_SAVE,
    EventType.SNAPSHOT_LOAD,
    EventType.SNAPSHOT_VERIFY,
}

ALL_EVENT_TYPES = set(EventType)


# PHASE II — Snapshot Orchestration Events
@dataclass(frozen=True)
class SnapshotPlanEvent:
    """
    Emitted at session start to record the auto-resume planning decision.
    
    This event provides an auditable trail of snapshot orchestration decisions
    for global health dashboards and continuity analysis.
    
    Extended fields (mean_coverage_pct, max_gap) allow reconstruction of
    "why this resume choice was made" from a single event record.
    """
    status: str  # "NO_ACTION", "RESUME", "NEW_RUN"
    preferred_run_id: Optional[str]
    preferred_snapshot_path: Optional[str]
    total_runs_analyzed: int
    mean_coverage_pct: float = 0.0  # Average coverage across all runs
    max_gap: int = 0  # Largest gap between snapshots across all runs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        from dataclasses import asdict
        return asdict(self)
