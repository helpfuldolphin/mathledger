"""
UVIL API Endpoints.

Exploration vs Authority:
- /propose_partition: Returns DraftProposal (exploration, random ID)
- /commit_uvil: Requires proposal_id, creates CommittedPartitionSnapshot
- /run_verification: REJECTS proposal_id, only accepts committed_partition_id

Storage: In-memory dict (restart-loss accepted for v0).

CRITICAL INVARIANTS:
- proposal_id NEVER enters hash-committed payloads
- Only committed_partition_id is accepted by /run_verification
- Double-commit of same proposal_id returns 409 Conflict
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from governance.uvil import (
    DraftProposal,
    DraftClaim,
    CommittedPartitionSnapshot,
    CommittedClaim,
    UVIL_Event,
    ReasoningArtifact,
    derive_committed_id,
    compute_full_attestation,
)
from governance.trust_class import TrustClass, Outcome, DEFAULT_SUGGESTED_TRUST_CLASS

router = APIRouter(prefix="/uvil", tags=["uvil"])

# ---------------------------------------------------------------------------
# In-memory storage (v0: accept restart-loss)
# ---------------------------------------------------------------------------

_draft_proposals: Dict[str, DraftProposal] = {}
_committed_snapshots: Dict[str, CommittedPartitionSnapshot] = {}
_uvil_events: List[UVIL_Event] = []
_reasoning_artifacts: List[ReasoningArtifact] = []
_committed_proposal_ids: set = set()  # Track already-committed proposals
_epoch_counter: int = 0  # Monotonic counter for epochs


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class DraftClaimResponse(BaseModel):
    """Single claim in a proposal response."""

    claim_text: str
    suggested_trust_class: str
    rationale: str = ""


class ProposePartitionRequest(BaseModel):
    """Request to generate a partition proposal."""

    problem_statement: str = Field(..., min_length=1, max_length=10000)


class ProposePartitionResponse(BaseModel):
    """Response containing a draft proposal."""

    proposal_id: str
    claims: List[DraftClaimResponse]


class EditedClaimRequest(BaseModel):
    """User-edited claim for commit."""

    claim_text: str
    trust_class: str
    rationale: str = ""

    @field_validator("trust_class")
    @classmethod
    def validate_trust_class(cls, v: str) -> str:
        """Validate trust class is valid."""
        try:
            TrustClass(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid trust class: {v}. Must be one of: FV, MV, PA, ADV")


class CommitUVILRequest(BaseModel):
    """Request to commit UVIL edits."""

    proposal_id: str = Field(..., description="MUST exist in draft store")
    edited_claims: List[EditedClaimRequest]
    user_fingerprint: str = Field(default="anonymous", max_length=64)


class CommitUVILResponse(BaseModel):
    """Response containing committed partition and attestation."""

    committed_partition_id: str
    u_t: str
    r_t: str
    h_t: str
    claim_count: int


class RunVerificationRequest(BaseModel):
    """
    Request to run verification.

    CRITICAL: No proposal_id field allowed - only accepts committed_partition_id.
    """

    committed_partition_id: str = Field(..., description="REQUIRED - must be committed")

    # CRITICAL: This model intentionally has NO proposal_id field
    # to enforce the exploration/authority boundary


class RunVerificationResponse(BaseModel):
    """Response containing verification result."""

    outcome: str
    committed_partition_id: str
    attestation: Dict[str, Any]
    authority_basis: Dict[str, Any]  # Explains why claims are authority-bearing


# ---------------------------------------------------------------------------
# Template Partitioner (v0: defaults all claims to ADV)
# ---------------------------------------------------------------------------


def _template_partition(problem_statement: str) -> List[DraftClaim]:
    """
    Template-based partitioner for v0.

    CRITICAL: Defaults ALL claims to ADV (user must promote).

    Args:
        problem_statement: User's problem statement

    Returns:
        List of draft claims, all with ADV trust class
    """
    # TODO: Implement template-based partitioning
    # For now, create a single placeholder claim
    return [
        DraftClaim(
            claim_text=f"Claim derived from: {problem_statement[:100]}...",
            suggested_trust_class=DEFAULT_SUGGESTED_TRUST_CLASS,  # ADV
            rationale="Template partitioner - user must edit and promote",
        )
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/propose_partition", response_model=ProposePartitionResponse)
async def propose_partition(req: ProposePartitionRequest) -> ProposePartitionResponse:
    """
    Generate partition proposal (exploration only).

    Returns DraftProposal with random proposal_id.
    CRITICAL: proposal_id NEVER enters hash-committed paths.

    Args:
        req: Request containing problem statement

    Returns:
        Proposal with random ID and draft claims (all ADV by default)
    """
    global _epoch_counter

    # Generate random proposal_id (exploration-only)
    proposal_id = str(uuid.uuid4())

    # Generate claims via template partitioner (all default to ADV)
    draft_claims = _template_partition(req.problem_statement)

    # Create and store draft proposal
    proposal = DraftProposal(
        proposal_id=proposal_id,
        claims=draft_claims,
        created_at=None,  # Wall-clock not stored for determinism
    )
    _draft_proposals[proposal_id] = proposal

    # Convert to response
    return ProposePartitionResponse(
        proposal_id=proposal_id,
        claims=[
            DraftClaimResponse(
                claim_text=c.claim_text,
                suggested_trust_class=c.suggested_trust_class.value,
                rationale=c.rationale,
            )
            for c in draft_claims
        ],
    )


@router.post("/commit_uvil", response_model=CommitUVILResponse)
async def commit_uvil(req: CommitUVILRequest) -> CommitUVILResponse:
    """
    Commit UVIL edits to create immutable snapshot.

    CRITICAL:
    - REQUIRES proposal_id to exist in _draft_proposals
    - Returns 409 if proposal_id already committed
    - Creates CommittedPartitionSnapshot with content-derived ID
    - Records UVIL_Event
    - Computes (U_t, R_t, H_t)

    Args:
        req: Request containing proposal_id and edited claims

    Returns:
        Committed partition ID and attestation

    Raises:
        HTTPException 404: If proposal_id not found
        HTTPException 409: If proposal_id already committed
    """
    global _epoch_counter

    # 1. Validate proposal_id exists
    if req.proposal_id not in _draft_proposals:
        raise HTTPException(
            status_code=404,
            detail=f"Proposal not found: {req.proposal_id}. "
            "Must call /propose_partition first.",
        )

    # 2. Check double-commit
    if req.proposal_id in _committed_proposal_ids:
        raise HTTPException(
            status_code=409,
            detail=f"Proposal already committed: {req.proposal_id}. "
            "Each proposal can only be committed once.",
        )

    # 3. Build CommittedClaim list with content-derived claim_ids
    committed_claims = []
    for edited in req.edited_claims:
        claim_content = {
            "claim_text": edited.claim_text,
            "trust_class": edited.trust_class,
            "rationale": edited.rationale,
        }
        claim_id = derive_committed_id(claim_content)

        committed_claims.append(
            CommittedClaim(
                claim_id=claim_id,
                claim_text=edited.claim_text,
                trust_class=TrustClass(edited.trust_class),
                rationale=edited.rationale,
            )
        )

    # 4. Build CommittedPartitionSnapshot with content-derived ID
    snapshot_content = {
        "claims": [
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "trust_class": c.trust_class.value,
                "rationale": c.rationale,
            }
            for c in committed_claims
        ]
    }
    committed_partition_id = derive_committed_id(snapshot_content)

    _epoch_counter += 1
    snapshot = CommittedPartitionSnapshot(
        committed_partition_id=committed_partition_id,
        claims=tuple(committed_claims),
        commit_epoch=_epoch_counter,
    )
    _committed_snapshots[committed_partition_id] = snapshot

    # 5. Record UVIL_Event
    event_content = {
        "event_type": "COMMIT",
        "committed_partition_id": committed_partition_id,
        "user_fingerprint": req.user_fingerprint,
        "epoch": _epoch_counter,
    }
    event_id = derive_committed_id(event_content)

    uvil_event = UVIL_Event(
        event_id=event_id,
        event_type="COMMIT",
        committed_partition_id=committed_partition_id,
        user_fingerprint=req.user_fingerprint,
        epoch=_epoch_counter,
    )
    _uvil_events.append(uvil_event)

    # 6. Compute attestation (empty reasoning artifacts for commit phase)
    u_t, r_t, h_t = compute_full_attestation([uvil_event], [])

    # 7. Mark proposal_id as committed
    _committed_proposal_ids.add(req.proposal_id)

    # 8. Return response
    return CommitUVILResponse(
        committed_partition_id=committed_partition_id,
        u_t=u_t,
        r_t=r_t,
        h_t=h_t,
        claim_count=len(committed_claims),
    )


@router.post("/run_verification", response_model=RunVerificationResponse)
async def run_verification(req: RunVerificationRequest) -> RunVerificationResponse:
    """
    Run verification on committed partition.

    CRITICAL:
    - REJECTS any request containing proposal_id (enforced by schema)
    - Only accepts committed_partition_id
    - Returns 404 if committed_partition_id not found

    Args:
        req: Request containing committed_partition_id

    Returns:
        Verification outcome and attestation

    Raises:
        HTTPException 404: If committed_partition_id not found
    """
    # 1. Validate committed_partition_id exists
    if req.committed_partition_id not in _committed_snapshots:
        raise HTTPException(
            status_code=404,
            detail=f"Committed partition not found: {req.committed_partition_id}. "
            "Must call /commit_uvil first.",
        )

    # 2. Retrieve snapshot
    snapshot = _committed_snapshots[req.committed_partition_id]

    # 3. Filter authority-bearing claims (FV/MV/PA only, no ADV)
    authority_claims = [
        c for c in snapshot.claims if c.trust_class != TrustClass.ADV
    ]

    # 4. Build ReasoningArtifacts for authority-bearing claims
    reasoning_artifacts = []
    for claim in authority_claims:
        artifact_content = {
            "claim_id": claim.claim_id,
            "trust_class": claim.trust_class.value,
            "proof_payload": {"v0_mock": True, "claim_text": claim.claim_text},
        }
        artifact_id = derive_committed_id(artifact_content)

        reasoning_artifacts.append(
            ReasoningArtifact(
                artifact_id=artifact_id,
                claim_id=claim.claim_id,
                trust_class=claim.trust_class,
                proof_payload={"v0_mock": True, "claim_text": claim.claim_text},
            )
        )

    # 5. Get all UVIL events for this partition
    partition_events = [
        e for e in _uvil_events if e.committed_partition_id == req.committed_partition_id
    ]

    # 6. Compute final attestation
    u_t, r_t, h_t = compute_full_attestation(partition_events, reasoning_artifacts)

    # 7. Determine outcome and authority basis
    # v0 has no real verifier, so we must be honest about what we can claim:
    # - FV/MV: would be mechanically verified if we had a verifier (v0: ABSTAINED)
    # - PA: user-attested, not mechanically verified (always ABSTAINED for verification)
    # - All ADV: ABSTAINED (nothing to verify)

    # Count claims by trust class
    fv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.FV)
    mv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.MV)
    pa_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.PA)
    adv_count = len(snapshot.claims) - len(authority_claims)

    # Determine outcome
    # v0: No real verifier, so all claims are ABSTAINED for mechanical verification
    # But we distinguish between "accepted into authority stream" vs "verified"
    if not authority_claims:
        outcome = Outcome.ABSTAINED
        outcome_explanation = "No authority-bearing claims. Nothing entered the reasoning stream."
    elif fv_count > 0 or mv_count > 0:
        # FV/MV claims exist but v0 has no verifier
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"v0 demo has no mechanical verifier. "
            f"{fv_count} FV and {mv_count} MV claims are authority-bearing "
            f"but verification is not yet implemented."
        )
    else:
        # PA-only: user-attested, cannot be mechanically verified
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"{pa_count} PA claim(s) accepted via user attestation. "
            f"PA claims are authority-bearing but not mechanically verified."
        )

    # Build authority basis explanation
    authority_basis = {
        "fv_count": fv_count,
        "mv_count": mv_count,
        "pa_count": pa_count,
        "adv_count": adv_count,
        "adv_excluded": adv_count > 0,
        "mechanically_verified": False,  # v0 has no verifier
        "authority_bearing_accepted": len(authority_claims) > 0,
        "explanation": outcome_explanation,
        "trust_class_breakdown": {
            "FV": "Formally Verified - would require machine-checkable proof (v0: not implemented)",
            "MV": "Mechanically Validated - would require automated checks (v0: not implemented)",
            "PA": "Procedurally Attested - user attestation, not mechanically verified",
            "ADV": "Advisory - exploration only, excluded from authority stream",
        },
    }

    return RunVerificationResponse(
        outcome=outcome.value,
        committed_partition_id=req.committed_partition_id,
        authority_basis=authority_basis,
        attestation={
            "u_t": u_t,
            "r_t": r_t,
            "h_t": h_t,
            "authority_claim_count": len(authority_claims),
            "total_claim_count": len(snapshot.claims),
        },
    )
