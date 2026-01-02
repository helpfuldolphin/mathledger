"""
UVIL (User-Verified Input Loop) Data Models and Attestation.

This module defines the exploration/authority boundary:
- DraftProposal, DraftClaim: exploration-only, NEVER hash-committed
- CommittedClaim, CommittedPartitionSnapshot, UVIL_Event: frozen, hash-committed

CRITICAL LEAF CONTRACT:
- compute_ui_root() expects raw dict payloads, NOT pre-hashed strings
- compute_reasoning_root() expects raw dict payloads, NOT pre-hashed strings
- attestation/dual_root handles canonicalization internally via _canonicalize_leaf()
- Domain separation: UI uses b"\\xA1ui-leaf", reasoning uses b"\\xA0reasoning-leaf"

Subordinate to fm.tex.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from governance.trust_class import TrustClass, Outcome, is_authority_bearing
from governance.registry_hash import canonicalize_json
from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
)

__all__ = [
    "DraftClaim",
    "DraftProposal",
    "CommittedClaim",
    "CommittedPartitionSnapshot",
    "UVIL_Event",
    "ReasoningArtifact",
    "build_uvil_event_payload",
    "build_reasoning_artifact_payload",
    "compute_full_attestation",
    "derive_committed_id",
]


# ---------------------------------------------------------------------------
# Exploration-only models (NEVER hash-committed)
# ---------------------------------------------------------------------------


@dataclass
class DraftClaim:
    """
    Mutable claim during exploration. NOT hash-committed.

    These are suggestions from the partitioner (template or LLM).
    User edits these before committing.
    """

    claim_text: str
    suggested_trust_class: TrustClass
    rationale: str = ""


@dataclass
class DraftProposal:
    """
    Exploration-phase proposal.

    CRITICAL: proposal_id is random UUID, MUST NEVER enter hash-committed paths.
    This is the exploration phase - nothing here is authoritative.
    """

    proposal_id: str  # Random UUID, exploration-only
    claims: List[DraftClaim] = field(default_factory=list)
    created_at: Optional[datetime] = None  # Wall-clock, NOT hash-committed


# ---------------------------------------------------------------------------
# Authority-bearing models (frozen, hash-committed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommittedClaim:
    """
    Immutable claim after UVIL commit.

    claim_id derived from sha256(canonical_json(content)).
    Once committed, trust_class and claim_text are immutable.
    """

    claim_id: str  # Content-derived, NOT random
    claim_text: str
    trust_class: TrustClass
    rationale: str


@dataclass(frozen=True)
class CommittedPartitionSnapshot:
    """
    Immutable snapshot of committed claims.

    committed_partition_id derived from sha256(canonical_json(claims)).
    This is the ONLY input to U_t from the claim domain.
    """

    committed_partition_id: str  # Content-derived
    claims: Tuple[CommittedClaim, ...]
    commit_epoch: int  # Monotonic counter, not wall-clock


@dataclass(frozen=True)
class UVIL_Event:
    """
    Immutable record of user action in UVIL.

    Inputs to U_t computation.
    """

    event_id: str  # Content-derived
    event_type: str  # "COMMIT", "EDIT", "PROMOTE", etc.
    committed_partition_id: str
    user_fingerprint: str  # Anonymized
    epoch: int


@dataclass(frozen=True)
class ReasoningArtifact:
    """
    Immutable reasoning/proof artifact.

    Inputs to R_t computation.
    CRITICAL: Only authority-bearing claims (FV/MV/PA) enter here.
    ADV claims MUST NEVER appear in ReasoningArtifact.
    """

    artifact_id: str  # Content-derived
    claim_id: str
    trust_class: TrustClass  # Must NOT be ADV
    proof_payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------


def derive_committed_id(content: Dict[str, Any]) -> str:
    """
    Derive deterministic ID from content.

    Uses sha256(canonical_json(content)).

    Args:
        content: Dictionary to hash

    Returns:
        64-character hex string ID
    """
    canonical = canonicalize_json(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Payload builders (raw dicts for dual_root)
# ---------------------------------------------------------------------------


def build_uvil_event_payload(event: UVIL_Event) -> Dict[str, Any]:
    """
    Build raw dict payload for U_t computation.

    CRITICAL: Returns raw dict, NOT pre-hashed string.
    dual_root handles canonicalization internally via _canonicalize_leaf().

    Args:
        event: UVIL event to convert

    Returns:
        Raw dict payload for compute_ui_root()
    """
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "committed_partition_id": event.committed_partition_id,
        "user_fingerprint": event.user_fingerprint,
        "epoch": event.epoch,
    }


def build_reasoning_artifact_payload(artifact: ReasoningArtifact) -> Dict[str, Any]:
    """
    Build raw dict payload for R_t computation.

    CRITICAL:
    - Returns raw dict, NOT pre-hashed string
    - MUST reject if artifact.trust_class == ADV

    Args:
        artifact: Reasoning artifact to convert

    Returns:
        Raw dict payload for compute_reasoning_root()

    Raises:
        ValueError: If artifact.trust_class is ADV
    """
    if artifact.trust_class == TrustClass.ADV:
        raise ValueError(
            f"ADV trust class MUST NOT enter R_t. "
            f"artifact_id={artifact.artifact_id}, claim_id={artifact.claim_id}"
        )

    if not is_authority_bearing(artifact.trust_class):
        raise ValueError(
            f"Only authority-bearing trust classes (FV/MV/PA) may enter R_t. "
            f"Got: {artifact.trust_class}"
        )

    return {
        "artifact_id": artifact.artifact_id,
        "claim_id": artifact.claim_id,
        "trust_class": artifact.trust_class.value,
        "proof_payload": artifact.proof_payload,
    }


# ---------------------------------------------------------------------------
# Attestation computation
# ---------------------------------------------------------------------------


def compute_full_attestation(
    uvil_events: List[UVIL_Event],
    reasoning_artifacts: List[ReasoningArtifact],
) -> Tuple[str, str, str]:
    """
    Compute (U_t, R_t, H_t) attestation tuple.

    CRITICAL:
    - Builds raw dict payloads via build_*_payload()
    - Passes raw dicts to compute_ui_root() / compute_reasoning_root()
    - dual_root handles canonicalization internally

    Args:
        uvil_events: List of UVIL events for U_t
        reasoning_artifacts: List of reasoning artifacts for R_t
            (MUST NOT contain ADV trust class)

    Returns:
        (u_t, r_t, h_t) - all 64-char hex strings

    Raises:
        ValueError: If any reasoning artifact has ADV trust class
    """
    # Build UI payloads (raw dicts)
    ui_payloads = [build_uvil_event_payload(e) for e in uvil_events]

    # Build reasoning payloads (raw dicts) - will raise if ADV present
    reasoning_payloads = [build_reasoning_artifact_payload(a) for a in reasoning_artifacts]

    # Compute roots - dual_root handles canonicalization internally
    u_t = compute_ui_root(ui_payloads)
    r_t = compute_reasoning_root(reasoning_payloads)

    # Compute composite: H_t = SHA256(R_t || U_t)
    h_t = compute_composite_root(r_t, u_t)

    return (u_t, r_t, h_t)
