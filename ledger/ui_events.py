"""
Deterministic UI event capture for dual-root attestation.

Provides a thread-safe recorder that canonicalizes UI events, derives
deterministic timestamps, and exposes canonical leaves for inclusion in
the UI-side Merkle tree (U_t).

Reference: MathLedger Whitepaper §3.1 (UI Event Canonicalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional

from attestation.dual_root import (
    canonicalize_ui_artifact,
    hash_ui_leaf,
)
from substrate.repro.determinism import deterministic_unix_timestamp


@dataclass(frozen=True)
class UIEventRecord:
    """Canonicalized UI event ready for Merkle inclusion."""

    event_id: str
    timestamp: int
    canonical_value: str
    leaf_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_artifact(self) -> str:
        """Return canonical string representation suitable for attestation."""
        return self.canonical_value


class UIEventStore:
    """Thread-safe in-memory buffer of canonical UI events."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: List[UIEventRecord] = []

    def record(self, event: Mapping[str, Any]) -> UIEventRecord:
        canonical = canonicalize_ui_artifact(event)
        leaf_hash = hash_ui_leaf(canonical)

        provided_id = str(event.get("event_id") or "").strip()
        event_id = provided_id or leaf_hash

        provided_ts = event.get("timestamp")
        if isinstance(provided_ts, (int, float)):
            timestamp = int(provided_ts)
        else:
            timestamp = int(deterministic_unix_timestamp(int(leaf_hash[:12], 16)))

        record = UIEventRecord(
            event_id=event_id,
            timestamp=timestamp,
            canonical_value=canonical,
            leaf_hash=leaf_hash,
            metadata=dict(event),
        )

        with self._lock:
            for idx, existing in enumerate(self._events):
                if existing.event_id == event_id:
                    self._events[idx] = record
                    break
            else:
                self._events.append(record)

            self._events.sort(key=lambda rec: (rec.timestamp, rec.event_id))

        return record

    def bulk_record(self, events: Iterable[Mapping[str, Any]]) -> List[UIEventRecord]:
        return [self.record(event) for event in events]

    def snapshot(self) -> List[UIEventRecord]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


ui_event_store = UIEventStore()


def capture_ui_event(event: Mapping[str, Any]) -> UIEventRecord:
    """Public helper for recording a single UI event."""
    return ui_event_store.record(event)


def capture_ui_events(events: Iterable[Mapping[str, Any]]) -> List[UIEventRecord]:
    """Record a batch of UI events."""
    return ui_event_store.bulk_record(events)


def materialize_ui_artifacts() -> List[str]:
    """Return canonical UI artifacts for attestation."""
    return [record.to_artifact() for record in ui_event_store.snapshot()]


def consume_ui_artifacts() -> List[str]:
    """
    Return canonical UI artifacts and clear the store.

    This drain operation ensures the next attestation sees an empty U_t stream.

    Reference: MathLedger Whitepaper §3.3 (Dual UI Merkle Consumption).
    """
    artifacts = materialize_ui_artifacts()
    ui_event_store.clear()
    return artifacts


def snapshot_ui_events() -> List[UIEventRecord]:
    """Expose full snapshot for auditing/testing."""
    return ui_event_store.snapshot()

