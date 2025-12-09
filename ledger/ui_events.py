"""
Deterministic UI event capture for dual-root attestation.

Provides a thread-safe recorder that canonicalizes UI events, derives
per-epoch Merkle summaries, and exposes canonical leaves for inclusion in
the UI-side Merkle tree (U_t).

Reference: MathLedger Whitepaper A3.1 (UI Event Canonicalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

from attestation.dual_root import canonicalize_ui_artifact, hash_ui_leaf
from backend.telemetry.ui_schema import UIEvent, epoch_merkle_artifacts
from substrate.repro.determinism import deterministic_unix_timestamp


def _iso_to_epoch_seconds(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp())
    except ValueError:
        return int(deterministic_unix_timestamp(0))


def _as_ui_event(event: Union[Mapping[str, Any], UIEvent]) -> UIEvent:
    if isinstance(event, UIEvent):
        return event
    if not isinstance(event, Mapping):
        raise TypeError(f"UI event must be mapping or UIEvent, got {type(event)}")
    return UIEvent.from_mapping(event)


@dataclass(frozen=True)
class UIEventRecord:
    """Canonicalized UI event ready for Merkle inclusion."""

    event: UIEvent
    timestamp: int
    canonical_value: str
    leaf_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return self.event.event_id

    @property
    def epoch_id(self) -> str:
        return self.event.epoch_id

    def to_artifact(self) -> str:
        """Return canonical string representation suitable for attestation."""
        return self.canonical_value


class UIEventStore:
    """Thread-safe in-memory buffer of canonical UI events."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: List[UIEventRecord] = []

    def record(self, event: Union[Mapping[str, Any], UIEvent]) -> UIEventRecord:
        ui_event = _as_ui_event(event)
        canonical = canonicalize_ui_artifact(ui_event.to_payload())
        leaf_hash = hash_ui_leaf(canonical)
        timestamp = _iso_to_epoch_seconds(ui_event.ts)

        record = UIEventRecord(
            event=ui_event,
            timestamp=timestamp,
            canonical_value=canonical,
            leaf_hash=leaf_hash,
            metadata=ui_event.to_payload(),
        )

        with self._lock:
            for idx, existing in enumerate(self._events):
                if existing.event.event_id == ui_event.event_id:
                    self._events[idx] = record
                    break
            else:
                self._events.append(record)

            self._events.sort(key=lambda rec: (rec.event.epoch_id, rec.timestamp, rec.event.event_id))

        return record

    def bulk_record(self, events: Iterable[Union[Mapping[str, Any], UIEvent]]) -> List[UIEventRecord]:
        return [self.record(event) for event in events]

    def snapshot(self) -> List[UIEventRecord]:
        with self._lock:
            return list(self._events)

    def snapshot_events(self) -> List[UIEvent]:
        with self._lock:
            return [record.event for record in self._events]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


def _epoch_artifacts(events: Sequence[UIEvent]) -> List[str]:
    return epoch_merkle_artifacts(events)


ui_event_store = UIEventStore()


def capture_ui_event(event: Union[Mapping[str, Any], UIEvent]) -> UIEventRecord:
    """Public helper for recording a single UI event."""
    return ui_event_store.record(event)


def capture_ui_events(events: Iterable[Union[Mapping[str, Any], UIEvent]]) -> List[UIEventRecord]:
    """Record a batch of UI events."""
    return ui_event_store.bulk_record(events)


def materialize_ui_artifacts() -> List[str]:
    """Return canonical UI artifacts grouped per epoch for attestation."""
    events = ui_event_store.snapshot_events()
    return _epoch_artifacts(events)


def consume_ui_artifacts() -> List[str]:
    """
    Return canonical UI artifacts and clear the store.

    This drain operation ensures the next attestation sees an empty U_t stream.

    Reference: MathLedger Whitepaper A3.3 (Dual UI Merkle Consumption).
    """
    artifacts = materialize_ui_artifacts()
    ui_event_store.clear()
    return artifacts


def snapshot_ui_events() -> List[UIEventRecord]:
    """Expose full snapshot for auditing/testing."""
    return ui_event_store.snapshot()
