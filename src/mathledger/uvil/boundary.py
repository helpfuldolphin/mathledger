"""A3 boundary enforcement: exploration identifiers never enter committed paths."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

from mathledger.governance.trust_class import parse_trust_class
from mathledger.uvil.models import (
    CommitResult,
    CommittedClaim,
    CommittedPartitionSnapshot,
    DraftProposal,
    EditedClaim,
    UVILEvent,
)


ALLOWED_EDITED_CLAIM_KEYS = frozenset({"claim_text", "trust_class", "rationale"})
FORBIDDEN_EXPLORATION_KEYS = frozenset({"proposal_id", "draft_id", "exploration_id"})


class UVILBoundaryViolation(ValueError):
    """Raised when exploration/authority boundary constraints are violated."""

    ERROR_CODE = "UVIL_BOUNDARY_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        claim_index: int | None = None,
        forbidden_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.claim_index = claim_index
        self.forbidden_key = forbidden_key

    def to_error_response(self) -> dict[str, Any]:
        """Structured error representation."""
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "claim_index": self.claim_index,
            "forbidden_key": self.forbidden_key,
        }


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def derive_content_id(content: Mapping[str, Any]) -> str:
    """Derive deterministic SHA-256 content ID."""
    canonical = _canonicalize_json(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coerce_edited_claim(value: EditedClaim | Mapping[str, Any], claim_index: int) -> EditedClaim:
    if isinstance(value, EditedClaim):
        claim_text = value.claim_text.strip()
        if not claim_text:
            raise UVILBoundaryViolation(
                "Edited claim must include non-empty claim_text.",
                claim_index=claim_index,
            )
        try:
            trust_class = parse_trust_class(value.trust_class)
        except ValueError as exc:
            raise UVILBoundaryViolation(
                "Edited claim has invalid trust_class.",
                claim_index=claim_index,
            ) from exc
        return EditedClaim(
            claim_text=claim_text,
            trust_class=trust_class,
            rationale=str(value.rationale),
        )

    if not isinstance(value, Mapping):
        raise UVILBoundaryViolation(
            "Edited claim input must be a mapping or EditedClaim.",
            claim_index=claim_index,
        )

    keys = set(value.keys())
    forbidden = keys & set(FORBIDDEN_EXPLORATION_KEYS)
    if forbidden:
        key = sorted(forbidden)[0]
        raise UVILBoundaryViolation(
            "Exploration identifiers are forbidden in committed claim inputs.",
            claim_index=claim_index,
            forbidden_key=key,
        )

    unknown = keys - set(ALLOWED_EDITED_CLAIM_KEYS)
    if unknown:
        raise UVILBoundaryViolation(
            f"Unexpected keys in edited claim input: {sorted(unknown)}",
            claim_index=claim_index,
        )

    claim_text = str(value.get("claim_text", "")).strip()
    if not claim_text:
        raise UVILBoundaryViolation(
            "Edited claim must include non-empty claim_text.",
            claim_index=claim_index,
        )

    try:
        trust_class = parse_trust_class(value.get("trust_class", ""))
    except ValueError as exc:
        raise UVILBoundaryViolation(
            "Edited claim has invalid trust_class.",
            claim_index=claim_index,
        ) from exc
    rationale = str(value.get("rationale", ""))
    return EditedClaim(claim_text=claim_text, trust_class=trust_class, rationale=rationale)


def build_committed_claims(
    edited_claims: Sequence[EditedClaim | Mapping[str, Any]],
) -> Tuple[CommittedClaim, ...]:
    """Build immutable committed claims from edited claim inputs."""
    if not edited_claims:
        raise UVILBoundaryViolation("Cannot commit empty claim set.")

    claims: list[CommittedClaim] = []
    for idx, value in enumerate(edited_claims):
        edited = _coerce_edited_claim(value, claim_index=idx)
        claim_content = {
            "claim_text": edited.claim_text,
            "trust_class": edited.trust_class.value,
            "rationale": edited.rationale,
        }
        claim_id = derive_content_id(claim_content)
        claims.append(
            CommittedClaim(
                claim_id=claim_id,
                claim_text=edited.claim_text,
                trust_class=edited.trust_class,
                rationale=edited.rationale,
            )
        )
    return tuple(claims)


def build_committed_snapshot(
    edited_claims: Sequence[EditedClaim | Mapping[str, Any]],
    *,
    commit_epoch: int,
) -> CommittedPartitionSnapshot:
    """Build committed snapshot from edited claim set and commit epoch."""
    if commit_epoch < 0:
        raise UVILBoundaryViolation("commit_epoch must be non-negative.")

    committed_claims = build_committed_claims(edited_claims)
    snapshot_payload = {
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "trust_class": claim.trust_class.value,
                "rationale": claim.rationale,
            }
            for claim in committed_claims
        ]
    }
    committed_partition_id = derive_content_id(snapshot_payload)
    return CommittedPartitionSnapshot(
        committed_partition_id=committed_partition_id,
        claims=committed_claims,
        commit_epoch=commit_epoch,
    )


def build_uvil_event(
    snapshot: CommittedPartitionSnapshot,
    *,
    user_fingerprint: str = "anonymous",
    event_type: str = "COMMIT",
) -> UVILEvent:
    """Build deterministic UVIL event for a committed snapshot."""
    if not event_type.strip():
        raise UVILBoundaryViolation("event_type must be non-empty.")
    if not user_fingerprint.strip():
        raise UVILBoundaryViolation("user_fingerprint must be non-empty.")

    event_content = {
        "event_type": event_type,
        "committed_partition_id": snapshot.committed_partition_id,
        "user_fingerprint": user_fingerprint,
        "epoch": snapshot.commit_epoch,
    }
    event_id = derive_content_id(event_content)
    return UVILEvent(
        event_id=event_id,
        event_type=event_type,
        committed_partition_id=snapshot.committed_partition_id,
        user_fingerprint=user_fingerprint,
        epoch=snapshot.commit_epoch,
    )


def snapshot_to_dict(snapshot: CommittedPartitionSnapshot) -> dict[str, Any]:
    """Serialize committed snapshot for logs/evidence."""
    return {
        "committed_partition_id": snapshot.committed_partition_id,
        "commit_epoch": snapshot.commit_epoch,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "trust_class": claim.trust_class.value,
                "rationale": claim.rationale,
            }
            for claim in snapshot.claims
        ],
    }


def event_to_dict(event: UVILEvent) -> dict[str, Any]:
    """Serialize UVIL event for logs/evidence."""
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "committed_partition_id": event.committed_partition_id,
        "user_fingerprint": event.user_fingerprint,
        "epoch": event.epoch,
    }


def _assert_no_exploration_keys(payload: Mapping[str, Any]) -> None:
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key in FORBIDDEN_EXPLORATION_KEYS:
                    raise UVILBoundaryViolation(
                        "Exploration key leaked into committed payload.",
                        forbidden_key=key,
                    )
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def commit_from_draft(
    draft: DraftProposal,
    edited_claims: Sequence[EditedClaim | Mapping[str, Any]],
    *,
    commit_epoch: int,
    user_fingerprint: str = "anonymous",
) -> CommitResult:
    """
    Commit explicit edited claims while ignoring exploration-only identifiers.

    Boundary rule:
    - `draft.proposal_id` is not an input to committed IDs.
    - committed IDs derive only from edited claim content.
    """
    snapshot = build_committed_snapshot(edited_claims, commit_epoch=commit_epoch)
    event = build_uvil_event(snapshot, user_fingerprint=user_fingerprint)
    result = CommitResult(snapshot=snapshot, event=event)

    payload = commit_result_to_dict(result)
    _assert_no_exploration_keys(payload)
    return result


def commit_result_to_dict(result: CommitResult) -> dict[str, Any]:
    """Serialize commit result to deterministic primitive structure."""
    return {
        "snapshot": snapshot_to_dict(result.snapshot),
        "event": event_to_dict(result.event),
    }
