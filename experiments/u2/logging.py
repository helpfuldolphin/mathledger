# PHASE II — NOT RUN IN PHASE I
"""
U2 Trace Logger

STATUS: PHASE II — NOT RUN IN PHASE I
Provides minimal-overhead, structured telemetry emission for uplift experiments.

Features:
- Append-only JSONL output (no truncation, no overwrites)
- Single file handle per session (context manager)
- Compact JSON serialization
- Optional hash-chaining for tamper-evidence
- Fail-soft: logging errors do not crash experiments
- Event filtering: enable/disable specific event types

Performance targets:
- Per-event overhead: <0.5ms
- Total logging overhead: <5% of cycle time

Filtering:
    logger = U2TraceLogger(path, enabled_events={"cycle_telemetry", "policy_weight_update"})
    
    Core events (default when --trace-log but no --trace-events):
    - session_start, session_end
    - cycle_telemetry
    - cycle_duration
"""

from dataclasses import asdict
from pathlib import Path
from typing import FrozenSet, Optional, Set, TextIO, Any
import hashlib
import json
import sys
import time

from . import schema


# Default "core" events enabled when --trace-log is set but --trace-events is not
CORE_EVENTS: FrozenSet[str] = frozenset({
    "session_start",
    "session_end",
    "cycle_telemetry",
    "cycle_duration",
})

# All available event types (for validation)
ALL_EVENT_TYPES: FrozenSet[str] = frozenset({
    "session_start",
    "session_end",
    "cycle_telemetry",
    "cycle_duration",
    "candidate_ordering",
    "scoring_features",
    "policy_weight_update",
    "budget_consumption",
    "substrate_result",
    "hash_chain_entry",
    "snapshot_plan",
})


class U2TraceLogger:
    """Append-only JSONL trace logger for U2 experiments.
    
    Usage:
        with U2TraceLogger(Path("traces/run_001.jsonl")) as logger:
            logger.log_session_start(event)
            for cycle in range(cycles):
                logger.log_cycle_duration(event)
            logger.log_session_end(event)
    
    Filtered logging:
        # Only log specific event types
        logger = U2TraceLogger(
            path,
            enabled_events={"cycle_telemetry", "policy_weight_update"}
        )
    """
    
    def __init__(
        self,
        path: Path,
        enable_hash_chain: bool = False,
        fail_soft: bool = True,
        enabled_events: Optional[Set[str]] = None,
    ):
        """Initialize the trace logger.
        
        Args:
            path: Output file path for JSONL trace log.
            enable_hash_chain: If True, compute and emit HashChainEntryEvent.
            fail_soft: If True, logging errors are caught and printed to stderr
                       instead of raising exceptions.
            enabled_events: Set of event type names to enable. If None, all events
                           are enabled. Use CORE_EVENTS for minimal logging.
                           Valid event types: session_start, session_end,
                           cycle_telemetry, cycle_duration, candidate_ordering,
                           scoring_features, policy_weight_update, budget_consumption,
                           substrate_result, hash_chain_entry.
        """
        self._path = path
        self._fh: Optional[TextIO] = None
        self._enable_hash_chain = enable_hash_chain
        self._fail_soft = fail_soft
        self._entry_index = 0
        self._prev_hash = "0" * 64  # Genesis hash
        self._event_count = 0
        
        # Event filtering
        if enabled_events is not None:
            # Validate event types
            invalid = enabled_events - ALL_EVENT_TYPES
            if invalid:
                if not fail_soft:
                    raise ValueError(f"Invalid event types: {invalid}. Valid: {ALL_EVENT_TYPES}")
                print(f"[U2TraceLogger] WARNING: Invalid event types ignored: {invalid}", file=sys.stderr)
            self._enabled_events: Optional[FrozenSet[str]] = frozenset(enabled_events & ALL_EVENT_TYPES)
        else:
            self._enabled_events = None  # None = all events enabled
    
    def _is_event_enabled(self, event_name: str) -> bool:
        """Check if an event type is enabled for logging."""
        if self._enabled_events is None:
            return True  # All events enabled
        return event_name in self._enabled_events
    
    def __enter__(self) -> "U2TraceLogger":
        """Open the log file in append mode."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append-only text mode; no truncation
            self._fh = self._path.open("a", encoding="utf-8")
        except Exception as e:
            if self._fail_soft:
                print(f"[U2TraceLogger] WARNING: Failed to open log file: {e}", file=sys.stderr)
                self._fh = None
            else:
                raise
        return self
    
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Flush and close the log file."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception as e:
                if not self._fail_soft:
                    raise
                print(f"[U2TraceLogger] WARNING: Failed to close log file: {e}", file=sys.stderr)
            finally:
                self._fh = None
    
    def _write_event(self, event_type: str, payload: dict) -> None:
        """Write a single event to the log file.
        
        Args:
            event_type: Name of the event class (e.g., "CandidateOrderingEvent").
            payload: Dictionary representation of the event.
        """
        if self._fh is None:
            return  # Fail-soft: skip if file not open
        
        try:
            record = {
                "ts": time.time(),
                "event_type": event_type,
                "schema_version": schema.TRACE_SCHEMA_VERSION,
                "payload": payload,
            }
            line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
            self._fh.write(line)
            self._event_count += 1
            
            # Periodic flush every 100 events for durability
            if self._event_count % 100 == 0:
                self._fh.flush()
                
        except Exception as e:
            if self._fail_soft:
                print(f"[U2TraceLogger] WARNING: Failed to write event: {e}", file=sys.stderr)
            else:
                raise
    
    def _compute_hash(self, data: str) -> str:
        """Compute SHA256 hash of a string."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    
    def _emit_hash_chain_entry(
        self,
        cycle: int,
        slice_name: str,
        mode: str,
        payload_json: str,
    ) -> None:
        """Emit a hash chain entry if enabled."""
        if not self._enable_hash_chain:
            return
        
        payload_hash = self._compute_hash(payload_json)
        chain_input = self._prev_hash + payload_json
        current_hash = self._compute_hash(chain_input)
        
        entry = schema.HashChainEntryEvent(
            cycle=cycle,
            slice_name=slice_name,
            mode=mode,
            entry_index=self._entry_index,
            prev_hash=self._prev_hash,
            current_hash=current_hash,
            payload_hash=payload_hash,
        )
        self._write_event("HashChainEntryEvent", asdict(entry))
        
        self._prev_hash = current_hash
        self._entry_index += 1
    
    # --- Convenience Methods for Each Event Type ---
    # Each method checks if the event type is enabled before writing.
    
    def log_session_start(self, event: schema.SessionStartEvent) -> None:
        """Log experiment session start."""
        if not self._is_event_enabled("session_start"):
            return
        payload = asdict(event)
        self._write_event("SessionStartEvent", payload)
        if self._enable_hash_chain:
            self._emit_hash_chain_entry(
                cycle=-1,
                slice_name=event.slice_name,
                mode=event.mode,
                payload_json=json.dumps(payload, separators=(",", ":")),
            )
    
    def log_session_end(self, event: schema.SessionEndEvent) -> None:
        """Log experiment session end."""
        if not self._is_event_enabled("session_end"):
            return
        payload = asdict(event)
        self._write_event("SessionEndEvent", payload)
    
    def log_snapshot_plan(self, event: schema.SnapshotPlanEvent) -> None:
        """Log snapshot planning decision from multi-run analysis."""
        if not self._is_event_enabled("snapshot_plan"):
            return
        payload = asdict(event)
        self._write_event("SnapshotPlanEvent", payload)
        # Final flush
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
    
    def log_candidate_ordering(self, event: schema.CandidateOrderingEvent) -> None:
        """Log candidate ordering for a cycle."""
        if not self._is_event_enabled("candidate_ordering"):
            return
        payload = asdict(event)
        # Convert tuple to list for JSON serialization
        payload["ordering"] = list(payload["ordering"])
        self._write_event("CandidateOrderingEvent", payload)
    
    def log_scoring_features(self, event: schema.ScoringFeaturesEvent) -> None:
        """Log scoring features for RFL policy."""
        if not self._is_event_enabled("scoring_features"):
            return
        payload = asdict(event)
        payload["features"] = list(payload["features"])
        self._write_event("ScoringFeaturesEvent", payload)
    
    def log_policy_weight_update(self, event: schema.PolicyWeightUpdateEvent) -> None:
        """Log policy weight changes after feedback."""
        if not self._is_event_enabled("policy_weight_update"):
            return
        self._write_event("PolicyWeightUpdateEvent", asdict(event))
    
    def log_budget_consumption(self, event: schema.BudgetConsumptionEvent) -> None:
        """Log resource budget tracking."""
        if not self._is_event_enabled("budget_consumption"):
            return
        self._write_event("BudgetConsumptionEvent", asdict(event))
    
    def log_cycle_duration(self, event: schema.CycleDurationEvent) -> None:
        """Log cycle timing telemetry."""
        if not self._is_event_enabled("cycle_duration"):
            return
        self._write_event("CycleDurationEvent", asdict(event))
    
    def log_substrate_result(self, event: schema.SubstrateResultEvent) -> None:
        """Log FO substrate simulation outcome."""
        if not self._is_event_enabled("substrate_result"):
            return
        payload = asdict(event)
        payload["verified_hashes"] = list(payload["verified_hashes"])
        self._write_event("SubstrateResultEvent", payload)
    
    def log_cycle_telemetry(self, event: schema.CycleTelemetryEvent) -> None:
        """Log full cycle telemetry record."""
        if not self._is_event_enabled("cycle_telemetry"):
            return
        payload = asdict(event)
        self._write_event("CycleTelemetryEvent", payload)
        
        if self._enable_hash_chain:
            self._emit_hash_chain_entry(
                cycle=event.cycle,
                slice_name=event.slice_name,
                mode=event.mode,
                payload_json=json.dumps(payload["raw_record"], separators=(",", ":")),
            )
    
    def log_hash_chain_entry(self, event: schema.HashChainEntryEvent) -> None:
        """Log explicit hash chain entry (for manual chaining)."""
        if not self._is_event_enabled("hash_chain_entry"):
            return
        self._write_event("HashChainEntryEvent", asdict(event))
    
    @property
    def event_count(self) -> int:
        """Return the number of events written."""
        return self._event_count
    
    @property
    def path(self) -> Path:
        """Return the log file path."""
        return self._path
    
    @property
    def enabled_events(self) -> Optional[FrozenSet[str]]:
        """Return the set of enabled event types (None = all enabled)."""
        return self._enabled_events


__all__ = [
    "U2TraceLogger",
    "CORE_EVENTS",
    "ALL_EVENT_TYPES",
]

