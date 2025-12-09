"""
UI telemetry schema + Merkle binding helpers.

Defines the canonical UIEvent dataclass, validation helpers, and
per-epoch Merkle summarization routines so UI evidence is always
anchored to an epoch before contributing to U_t.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

from attestation.dual_root import build_ui_attestation, canonicalize_ui_artifact
from substrate.repro.determinism import deterministic_timestamp_from_content

UI_EVENT_REQUIRED_KEYS = {"epoch_id", "event_id", "kind", "payload", "ts"}


def _deep_copy_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(payload))


def _coerce_iso_timestamp(raw: Any, *, seed: str) -> str:
    def _format(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return _format(parsed)
        except ValueError:
            pass
    if isinstance(raw, (int, float)):
        return _format(datetime.fromtimestamp(raw, tz=timezone.utc))

    det = deterministic_timestamp_from_content(seed)
    return _format(det)


def _derive_event_id(epoch_id: str, payload: Dict[str, Any]) -> str:
    canonical = canonicalize_ui_artifact({"epoch_id": epoch_id, "payload": payload})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_ui_event(event: Mapping[str, Any]) -> None:
    missing = UI_EVENT_REQUIRED_KEYS - set(event.keys())
    if missing:
        raise ValueError(f"missing required UI event field(s): {sorted(missing)}")

    for key in ("epoch_id", "event_id", "kind", "ts"):
        if not isinstance(event.get(key), str):
            raise TypeError(f"{key} must be a string")

    if not isinstance(event.get("payload"), Mapping):
        raise TypeError("payload must be a mapping")


@dataclass(frozen=True)
class UIEvent:
    epoch_id: str
    event_id: str
    kind: str
    payload: Dict[str, Any]
    ts: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "event_id": self.event_id,
            "kind": self.kind,
            "payload": self.payload,
            "ts": self.ts,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "UIEvent":
        payload_value = data.get("payload")
        if isinstance(payload_value, Mapping):
            payload = _deep_copy_payload(payload_value)
        else:
            excluded = {"epoch_id", "event_id", "kind", "payload", "ts"}
            payload = _deep_copy_payload({k: v for k, v in data.items() if k not in excluded})

        epoch_id = str(data.get("epoch_id") or data.get("epoch") or "epoch::legacy")
        event_id_raw = data.get("event_id") or data.get("id")
        event_id = str(event_id_raw) if event_id_raw else _derive_event_id(epoch_id, payload)
        kind = str(data.get("kind") or data.get("event_type") or data.get("action") or "legacy")
        ts = _coerce_iso_timestamp(data.get("ts") or data.get("timestamp"), seed=f"{epoch_id}:{event_id}")
        return cls(epoch_id=epoch_id, event_id=event_id, kind=kind, payload=payload, ts=ts)


@dataclass(frozen=True)
class EpochMerkleSummary:
    epoch_id: str
    epoch_root: str
    event_count: int

    def to_artifact(self) -> str:
        payload = {
            "epoch_id": self.epoch_id,
            "epoch_root": self.epoch_root,
            "event_count": self.event_count,
        }
        return canonicalize_ui_artifact(payload)


def summarize_events_by_epoch(events: Sequence[UIEvent]) -> List[EpochMerkleSummary]:
    grouped: Dict[str, List[str]] = {}
    for event in events:
        grouped.setdefault(event.epoch_id, []).append(canonicalize_ui_artifact(event.to_payload()))

    summaries: List[EpochMerkleSummary] = []
    for epoch_id in sorted(grouped.keys()):
        tree = build_ui_attestation(grouped[epoch_id])
        summaries.append(EpochMerkleSummary(epoch_id=epoch_id, epoch_root=tree.root, event_count=len(grouped[epoch_id])))
    return summaries


def epoch_merkle_artifacts(events: Sequence[UIEvent]) -> List[str]:
    return [summary.to_artifact() for summary in summarize_events_by_epoch(events)]


__all__ = [
    "UIEvent",
    "EpochMerkleSummary",
    "validate_ui_event",
    "summarize_events_by_epoch",
    "epoch_merkle_artifacts",
]
