"""UVIL data models for exploration and authority boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple

from mathledger.governance.trust_class import TrustClass


@dataclass
class DraftClaim:
    """Mutable exploration claim suggestion."""

    claim_text: str
    suggested_trust_class: TrustClass
    rationale: str = ""


@dataclass
class DraftProposal:
    """Exploration proposal with random identifier."""

    proposal_id: str
    claims: List[DraftClaim] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass(frozen=True)
class EditedClaim:
    """Claim payload submitted for explicit authority commit."""

    claim_text: str
    trust_class: TrustClass
    rationale: str = ""


@dataclass(frozen=True)
class CommittedClaim:
    """Immutable authority-bearing claim artifact."""

    claim_id: str
    claim_text: str
    trust_class: TrustClass
    rationale: str


@dataclass(frozen=True)
class CommittedPartitionSnapshot:
    """Immutable committed partition state."""

    committed_partition_id: str
    claims: Tuple[CommittedClaim, ...]
    commit_epoch: int


@dataclass(frozen=True)
class UVILEvent:
    """Immutable UVIL event associated with a committed partition."""

    event_id: str
    event_type: str
    committed_partition_id: str
    user_fingerprint: str
    epoch: int


@dataclass(frozen=True)
class CommitResult:
    """Commit result bundle for A3 boundary operations."""

    snapshot: CommittedPartitionSnapshot
    event: UVILEvent

