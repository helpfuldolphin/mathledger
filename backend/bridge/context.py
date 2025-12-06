from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VALID_PROOF_STATUSES = {"success", "abstain", "failure"}
_ABSTENTION_COUNT_KEYS = ("verified", "rejected", "considered")


def _normalize_hex(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hex string, got {type(value).__name__}")
    normalized = value.strip().lower()
    if not _HEX64_PATTERN.match(normalized):
        raise ValueError(f"{name} must be a 64-char lowercase hex string, got {value!r}")
    return normalized


def _normalize_proof_status(status: str) -> str:
    if not isinstance(status, str):
        raise TypeError(f"proof_status must be a string, got {type(status).__name__}")
    normalized = status.strip().lower()
    if normalized not in _VALID_PROOF_STATUSES:
        raise ValueError(f"proof_status must be one of {_VALID_PROOF_STATUSES}, got {status}")
    return normalized


def _normalize_block_id(value: int) -> int:
    try:
        block_id = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"block_id must be an integer, got {value!r}") from exc
    if block_id < 0:
        raise ValueError(f"block_id must be non-negative, got {block_id}")
    return block_id


def _canonicalize_abstention_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw or {})
    try:
        rate = float(data.get("rate", 0.0))
    except (TypeError, ValueError):
        rate = 0.0
    try:
        mass = float(data.get("mass", 0.0))
    except (TypeError, ValueError):
        mass = 0.0

    counts_raw = data.get("counts") or {}
    counts: Dict[str, int] = {}
    for key in _ABSTENTION_COUNT_KEYS:
        try:
            counts[key] = int(counts_raw.get(key, 0))
        except (TypeError, ValueError):
            counts[key] = 0

    reasons_raw = data.get("reasons") or data.get("breakdown") or {}
    reasons: Dict[str, int] = {}
    if isinstance(reasons_raw, Mapping):
        for key, value in reasons_raw.items():
            try:
                reasons[str(key)] = int(value)
            except (TypeError, ValueError):
                reasons[str(key)] = 0

    canonical: Dict[str, Any] = {
        "rate": rate,
        "mass": mass,
        "counts": counts,
        "reasons": reasons,
    }
    # Preserve any additional keys without validation
    for key, value in data.items():
        if key in canonical:
            continue
        canonical[key] = value
    return canonical


def _sanitize_breakdown(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    sanitized: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            sanitized[str(key)] = int(value)
        except (TypeError, ValueError):
            sanitized[str(key)] = 0
    return sanitized


def _sanitize_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(raw or {})
    metadata["abstention_breakdown"] = _sanitize_breakdown(metadata.get("abstention_breakdown"))
    return metadata

@dataclass
class AttestedRunContext:
    """
    Shared data structure bridging Derivation/Ledger outputs to RFL inputs.
    
    Contains all necessary context from a derivation run, attested by the ledger,
    ready for consumption by the metabolism (RFL).
    """
    slice_id: str
    statement_hash: str
    proof_status: str
    block_id: int
    composite_root: str
    reasoning_root: str
    ui_root: str
    abstention_metrics: Dict[str, Any]
    
    policy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def abstention_rate(self) -> float:
        return float(self.abstention_metrics.get("rate", 0.0))

    @property
    def abstention_mass(self) -> float:
        return float(self.abstention_metrics.get("mass", 0.0))

    def __post_init__(self) -> None:
        if not isinstance(self.slice_id, str) or not self.slice_id.strip():
            raise ValueError("slice_id must be a non-empty string")
        self.slice_id = self.slice_id.strip()

        self.statement_hash = _normalize_hex("statement_hash", self.statement_hash)
        self.proof_status = _normalize_proof_status(self.proof_status)
        self.block_id = _normalize_block_id(self.block_id)
        self.composite_root = _normalize_hex("composite_root", self.composite_root)
        self.reasoning_root = _normalize_hex("reasoning_root", self.reasoning_root)
        self.ui_root = _normalize_hex("ui_root", self.ui_root)

        self.abstention_metrics = _canonicalize_abstention_metrics(self.abstention_metrics)
        self.metadata = _sanitize_metadata(self.metadata)
