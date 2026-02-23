"""Wave A4 evidence pack build + replay verification."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping, Sequence

from mathledger.basis.attestation.dual import composite_root
from mathledger.basis.crypto.hash import (
    DOMAIN_NODE,
    DOMAIN_REASONING_EMPTY,
    DOMAIN_UI_EMPTY,
    sha256_bytes,
    sha256_hex,
)
from mathledger.governance import (
    AuthorityRoutingViolation,
    build_authority_reasoning_leaves,
    parse_trust_class,
    route_claims_by_trust_class,
)
from mathledger.uvil import (
    CommittedPartitionSnapshot,
    UVILEvent,
    event_to_dict,
    snapshot_to_dict,
)


EVIDENCE_SCHEMA_VERSION = "v1"
COMPOSITE_FORMULA_NOTE = "H_t = SHA256(R_t || U_t)"
REPLAY_INSTRUCTIONS = (
    "Recompute U_t from uvil_events; recompute R_t from authority-bearing reasoning_artifacts "
    "(ADV excluded); recompute H_t from R_t and U_t; compare all three roots."
)

FORBIDDEN_EXPLORATION_KEYS = frozenset({"proposal_id", "draft_id", "exploration_id"})
REQUIRED_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "committed_partition_snapshot",
        "uvil_events",
        "reasoning_artifacts",
        "u_t",
        "r_t",
        "h_t",
    }
)
ALLOWED_UVIL_EVENT_KEYS = frozenset(
    {"event_id", "event_type", "committed_partition_id", "user_fingerprint", "epoch"}
)
REQUIRED_SNAPSHOT_KEYS = frozenset({"committed_partition_id", "commit_epoch", "claims"})
ALLOWED_SNAPSHOT_CLAIM_KEYS = frozenset({"claim_id", "claim_text", "trust_class", "rationale"})

DOMAIN_REASONING_LEAF = b"\xA0reasoning-leaf"
DOMAIN_UI_LEAF = b"\xA1ui-leaf"


class EvidenceReplayViolation(ValueError):
    """Raised when evidence replay invariants or schema contracts are violated."""

    ERROR_CODE = "EVIDENCE_REPLAY_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        artifact_index: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.artifact_index = artifact_index
        self.details = dict(details or {})

    def to_error_response(self) -> dict[str, Any]:
        """Structured error object for API/audit callers."""
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "field": self.field,
            "artifact_index": self.artifact_index,
            "details": self.details,
        }


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _assert_no_exploration_keys(payload: Any) -> None:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key in FORBIDDEN_EXPLORATION_KEYS:
                    raise EvidenceReplayViolation(
                        "Exploration key is forbidden in evidence-pack authority paths.",
                        field=str(key),
                    )
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _validate_hex_digest(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise EvidenceReplayViolation(f"{field} must be a 64-char hex string.", field=field)
    if len(value) != 64:
        raise EvidenceReplayViolation(f"{field} must be 64 hex chars.", field=field)
    if any(ch not in "0123456789abcdef" for ch in value):
        raise EvidenceReplayViolation(f"{field} must be lowercase hex.", field=field)
    return value


def _compute_stream_root(
    canonical_leaves: Sequence[str],
    *,
    leaf_domain: bytes,
    empty_domain: bytes,
) -> str:
    if not canonical_leaves:
        return sha256_hex(b"", domain=empty_domain)

    level = [sha256_bytes(leaf.encode("utf-8"), domain=leaf_domain) for leaf in sorted(canonical_leaves)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: list[bytes] = []
        for left, right in zip(level[0::2], level[1::2]):
            next_level.append(sha256_bytes(left + right, domain=DOMAIN_NODE))
        level = next_level
    return level[0].hex()


def _normalize_uvil_event(event: Mapping[str, Any] | UVILEvent, *, event_index: int) -> dict[str, Any]:
    payload: dict[str, Any]
    if isinstance(event, UVILEvent):
        payload = event_to_dict(event)
    elif isinstance(event, Mapping):
        payload = dict(event)
    else:
        raise EvidenceReplayViolation(
            "UVIL event must be a mapping or UVILEvent.",
            field="uvil_events",
            artifact_index=event_index,
        )

    _assert_no_exploration_keys(payload)

    keys = set(payload.keys())
    missing = set(ALLOWED_UVIL_EVENT_KEYS) - keys
    unknown = keys - set(ALLOWED_UVIL_EVENT_KEYS)
    if missing:
        raise EvidenceReplayViolation(
            f"UVIL event missing required keys: {sorted(missing)}",
            field="uvil_events",
            artifact_index=event_index,
        )
    if unknown:
        raise EvidenceReplayViolation(
            f"UVIL event has unexpected keys: {sorted(unknown)}",
            field="uvil_events",
            artifact_index=event_index,
        )

    event_id = str(payload["event_id"]).strip()
    event_type = str(payload["event_type"]).strip()
    committed_partition_id = str(payload["committed_partition_id"]).strip()
    user_fingerprint = str(payload["user_fingerprint"]).strip()
    if not event_id or not event_type or not committed_partition_id or not user_fingerprint:
        raise EvidenceReplayViolation(
            "UVIL event fields must be non-empty strings.",
            field="uvil_events",
            artifact_index=event_index,
        )

    try:
        epoch = int(payload["epoch"])
    except (TypeError, ValueError) as exc:
        raise EvidenceReplayViolation(
            "UVIL event epoch must be an integer.",
            field="uvil_events",
            artifact_index=event_index,
        ) from exc
    if epoch < 0:
        raise EvidenceReplayViolation(
            "UVIL event epoch must be non-negative.",
            field="uvil_events",
            artifact_index=event_index,
        )

    return {
        "event_id": event_id,
        "event_type": event_type,
        "committed_partition_id": committed_partition_id,
        "user_fingerprint": user_fingerprint,
        "epoch": epoch,
    }


def _normalize_uvil_events(
    uvil_events: Sequence[Mapping[str, Any] | UVILEvent],
) -> tuple[dict[str, Any], ...]:
    if isinstance(uvil_events, (str, bytes)):
        raise EvidenceReplayViolation("uvil_events must be a sequence.", field="uvil_events")
    return tuple(
        _normalize_uvil_event(event, event_index=idx)
        for idx, event in enumerate(uvil_events)
    )


def _normalize_snapshot(
    snapshot: Mapping[str, Any] | CommittedPartitionSnapshot,
) -> dict[str, Any]:
    payload: dict[str, Any]
    if isinstance(snapshot, CommittedPartitionSnapshot):
        payload = dict(snapshot_to_dict(snapshot))
    elif isinstance(snapshot, Mapping):
        payload = dict(snapshot)
    else:
        raise EvidenceReplayViolation(
            "committed_partition_snapshot must be a mapping or CommittedPartitionSnapshot.",
            field="committed_partition_snapshot",
        )

    _assert_no_exploration_keys(payload)

    keys = set(payload.keys())
    missing = set(REQUIRED_SNAPSHOT_KEYS) - keys
    unknown = keys - set(REQUIRED_SNAPSHOT_KEYS)
    if missing:
        raise EvidenceReplayViolation(
            f"Snapshot missing required keys: {sorted(missing)}",
            field="committed_partition_snapshot",
        )
    if unknown:
        raise EvidenceReplayViolation(
            f"Snapshot has unexpected keys: {sorted(unknown)}",
            field="committed_partition_snapshot",
        )

    committed_partition_id = str(payload["committed_partition_id"]).strip()
    if not committed_partition_id:
        raise EvidenceReplayViolation(
            "Snapshot committed_partition_id must be non-empty.",
            field="committed_partition_snapshot",
        )

    try:
        commit_epoch = int(payload["commit_epoch"])
    except (TypeError, ValueError) as exc:
        raise EvidenceReplayViolation(
            "Snapshot commit_epoch must be an integer.",
            field="committed_partition_snapshot",
        ) from exc
    if commit_epoch < 0:
        raise EvidenceReplayViolation(
            "Snapshot commit_epoch must be non-negative.",
            field="committed_partition_snapshot",
        )

    claims_value = payload["claims"]
    if not isinstance(claims_value, Sequence) or isinstance(claims_value, (str, bytes)):
        raise EvidenceReplayViolation(
            "Snapshot claims must be a sequence.",
            field="committed_partition_snapshot",
        )

    normalized_claims: list[dict[str, Any]] = []
    for idx, raw_claim in enumerate(claims_value):
        if not isinstance(raw_claim, Mapping):
            raise EvidenceReplayViolation(
                "Snapshot claim must be a mapping.",
                field="committed_partition_snapshot",
                artifact_index=idx,
            )
        claim = dict(raw_claim)
        _assert_no_exploration_keys(claim)

        keys = set(claim.keys())
        missing = {"claim_id", "claim_text", "trust_class"} - keys
        unknown = keys - set(ALLOWED_SNAPSHOT_CLAIM_KEYS)
        if missing:
            raise EvidenceReplayViolation(
                f"Snapshot claim missing keys: {sorted(missing)}",
                field="committed_partition_snapshot",
                artifact_index=idx,
            )
        if unknown:
            raise EvidenceReplayViolation(
                f"Snapshot claim has unexpected keys: {sorted(unknown)}",
                field="committed_partition_snapshot",
                artifact_index=idx,
            )

        try:
            trust_class = parse_trust_class(claim["trust_class"]).value
        except ValueError as exc:
            raise EvidenceReplayViolation(
                "Snapshot claim has invalid trust_class.",
                field="committed_partition_snapshot",
                artifact_index=idx,
            ) from exc

        claim_id = str(claim["claim_id"]).strip()
        claim_text = str(claim["claim_text"]).strip()
        if not claim_id or not claim_text:
            raise EvidenceReplayViolation(
                "Snapshot claim_id and claim_text must be non-empty.",
                field="committed_partition_snapshot",
                artifact_index=idx,
            )

        normalized_claims.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "trust_class": trust_class,
                "rationale": str(claim.get("rationale", "")),
            }
        )

    return {
        "committed_partition_id": committed_partition_id,
        "commit_epoch": commit_epoch,
        "claims": normalized_claims,
    }


def _normalize_reasoning_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_index: int,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise EvidenceReplayViolation(
            "Reasoning artifact must be a mapping.",
            field="reasoning_artifacts",
            artifact_index=artifact_index,
        )

    payload = dict(artifact)
    _assert_no_exploration_keys(payload)

    if "trust_class" not in payload:
        raise EvidenceReplayViolation(
            "Reasoning artifact missing trust_class.",
            field="reasoning_artifacts",
            artifact_index=artifact_index,
        )

    try:
        payload["trust_class"] = parse_trust_class(payload["trust_class"]).value
    except ValueError as exc:
        raise EvidenceReplayViolation(
            "Reasoning artifact has invalid trust_class.",
            field="reasoning_artifacts",
            artifact_index=artifact_index,
        ) from exc

    return payload


def _prepare_reasoning_artifacts(
    reasoning_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(reasoning_artifacts, (str, bytes)):
        raise EvidenceReplayViolation(
            "reasoning_artifacts must be a sequence.",
            field="reasoning_artifacts",
        )

    normalized = tuple(
        _normalize_reasoning_artifact(artifact, artifact_index=idx)
        for idx, artifact in enumerate(reasoning_artifacts)
    )

    try:
        routed = route_claims_by_trust_class(normalized)
    except AuthorityRoutingViolation as exc:
        raise EvidenceReplayViolation(
            "Unable to route reasoning artifacts by trust class.",
            field="reasoning_artifacts",
            details={"source_error": str(exc)},
        ) from exc

    try:
        authority_leaves = build_authority_reasoning_leaves(routed.authority_claims)
    except Exception as exc:
        raise EvidenceReplayViolation(
            "Authority reasoning payload construction failed (check validation_outcome typing).",
            field="reasoning_artifacts",
            details={"source_error": str(exc)},
        ) from exc

    r_t = _compute_stream_root(
        authority_leaves,
        leaf_domain=DOMAIN_REASONING_LEAF,
        empty_domain=DOMAIN_REASONING_EMPTY,
    )
    trust_class_distribution = dict(sorted(Counter(a["trust_class"] for a in normalized).items()))

    return {
        "normalized_artifacts": normalized,
        "authority_artifacts": routed.authority_claims,
        "advisory_artifacts": routed.advisory_claims,
        "authority_leaves": authority_leaves,
        "trust_class_distribution": trust_class_distribution,
        "r_t": r_t,
    }


def compute_u_t_from_events(uvil_events: Sequence[Mapping[str, Any] | UVILEvent]) -> str:
    """Compute deterministic U_t from normalized UVIL event payloads."""
    normalized_events = _normalize_uvil_events(uvil_events)
    leaves = tuple(_canonicalize_json(event) for event in normalized_events)
    return _compute_stream_root(leaves, leaf_domain=DOMAIN_UI_LEAF, empty_domain=DOMAIN_UI_EMPTY)


def compute_r_t_from_artifacts(reasoning_artifacts: Sequence[Mapping[str, Any]]) -> str:
    """Compute deterministic R_t from authority-bearing reasoning payloads only."""
    reasoning_state = _prepare_reasoning_artifacts(reasoning_artifacts)
    return str(reasoning_state["r_t"])


def compute_h_t(r_t: str, u_t: str) -> str:
    """Compute composite H_t from R_t and U_t."""
    try:
        return composite_root(r_t, u_t)
    except ValueError as exc:
        raise EvidenceReplayViolation(
            "Unable to compute H_t from provided roots.",
            field="h_t",
            details={"source_error": str(exc)},
        ) from exc


def build_evidence_pack(
    *,
    committed_partition_snapshot: Mapping[str, Any] | CommittedPartitionSnapshot,
    uvil_events: Sequence[Mapping[str, Any] | UVILEvent],
    reasoning_artifacts: Sequence[Mapping[str, Any]],
    schema_version: str = EVIDENCE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """
    Build evidence pack with recorded roots and replay instructions.
    """
    if schema_version != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceReplayViolation(
            f"Unsupported schema_version: {schema_version!r}.",
            field="schema_version",
        )

    normalized_snapshot = _normalize_snapshot(committed_partition_snapshot)
    normalized_events = _normalize_uvil_events(uvil_events)
    reasoning_state = _prepare_reasoning_artifacts(reasoning_artifacts)

    u_t = compute_u_t_from_events(normalized_events)
    r_t = str(reasoning_state["r_t"])
    h_t = compute_h_t(r_t, u_t)

    return {
        "schema_version": schema_version,
        "committed_partition_snapshot": normalized_snapshot,
        "uvil_events": [dict(event) for event in normalized_events],
        "reasoning_artifacts": [dict(artifact) for artifact in reasoning_state["normalized_artifacts"]],
        "u_t": u_t,
        "r_t": r_t,
        "h_t": h_t,
        "composite_formula_note": COMPOSITE_FORMULA_NOTE,
        "replay_instructions": REPLAY_INSTRUCTIONS,
        "counts": {
            "uvil_event_count": len(normalized_events),
            "reasoning_artifact_count": len(reasoning_state["normalized_artifacts"]),
            "authority_reasoning_artifact_count": len(reasoning_state["authority_artifacts"]),
            "advisory_reasoning_artifact_count": len(reasoning_state["advisory_artifacts"]),
            "trust_class_distribution": dict(reasoning_state["trust_class_distribution"]),
        },
    }


def verify_evidence_pack(pack: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recompute roots from raw payloads and compare with recorded values.
    """
    if not isinstance(pack, Mapping):
        raise EvidenceReplayViolation("Evidence pack must be a mapping.")

    missing = REQUIRED_EVIDENCE_FIELDS - set(pack.keys())
    if missing:
        raise EvidenceReplayViolation(
            f"Evidence pack missing required fields: {sorted(missing)}",
        )

    schema_version = str(pack["schema_version"])
    if schema_version != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceReplayViolation(
            f"Unsupported schema_version: {schema_version!r}.",
            field="schema_version",
        )

    _normalize_snapshot(pack["committed_partition_snapshot"])
    normalized_events = _normalize_uvil_events(pack["uvil_events"])
    reasoning_state = _prepare_reasoning_artifacts(pack["reasoning_artifacts"])

    recorded_u_t = _validate_hex_digest(str(pack["u_t"]), field="u_t")
    recorded_r_t = _validate_hex_digest(str(pack["r_t"]), field="r_t")
    recorded_h_t = _validate_hex_digest(str(pack["h_t"]), field="h_t")

    recomputed_u_t = compute_u_t_from_events(normalized_events)
    recomputed_r_t = str(reasoning_state["r_t"])
    recomputed_h_t = compute_h_t(recomputed_r_t, recomputed_u_t)

    matches = {
        "u_t": recomputed_u_t == recorded_u_t,
        "r_t": recomputed_r_t == recorded_r_t,
        "h_t": recomputed_h_t == recorded_h_t,
    }

    return {
        "overall_pass": bool(matches["u_t"] and matches["r_t"] and matches["h_t"]),
        "schema_version": schema_version,
        "matches": matches,
        "recorded": {"u_t": recorded_u_t, "r_t": recorded_r_t, "h_t": recorded_h_t},
        "recomputed": {"u_t": recomputed_u_t, "r_t": recomputed_r_t, "h_t": recomputed_h_t},
        "counts": {
            "uvil_event_count": len(normalized_events),
            "reasoning_artifact_count": len(reasoning_state["normalized_artifacts"]),
            "authority_reasoning_artifact_count": len(reasoning_state["authority_artifacts"]),
            "advisory_reasoning_artifact_count": len(reasoning_state["advisory_artifacts"]),
            "trust_class_distribution": dict(reasoning_state["trust_class_distribution"]),
        },
    }


def assert_evidence_pack_passes(pack: Mapping[str, Any]) -> dict[str, Any]:
    """
    Fail-closed assertion wrapper for replay verification.
    """
    verification = verify_evidence_pack(pack)
    if not verification["overall_pass"]:
        raise EvidenceReplayViolation(
            "Evidence pack replay verification failed.",
            details={
                "matches": verification["matches"],
                "recorded": verification["recorded"],
                "recomputed": verification["recomputed"],
            },
        )
    return verification


def evidence_pack_json(pack: Mapping[str, Any]) -> str:
    """Canonical compact JSON serialization for deterministic evidence artifacts."""
    return json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "COMPOSITE_FORMULA_NOTE",
    "EvidenceReplayViolation",
    "compute_u_t_from_events",
    "compute_r_t_from_artifacts",
    "compute_h_t",
    "build_evidence_pack",
    "verify_evidence_pack",
    "assert_evidence_pack_passes",
    "evidence_pack_json",
]

