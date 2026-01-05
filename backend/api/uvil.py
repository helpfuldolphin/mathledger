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
    build_uvil_event_payload,
    build_reasoning_artifact_payload,
)
from governance.trust_class import TrustClass, Outcome, DEFAULT_SUGGESTED_TRUST_CLASS
from governance.mv_validator import validate_mv_claim, ValidatorOutcome
from governance.authority_gate import (
    AuthorityUpdateRequest,
    require_epoch_root,
    SilentAuthorityViolation,
    get_authority_violations,
)
from governance.trust_monotonicity import (
    require_trust_class_monotonicity,
    finalize_claim_registration,
    TrustClassMonotonicityViolation,
)
from attestation.dual_root import (
    compute_ui_root,
    compute_reasoning_root,
    compute_composite_root,
    verify_composite_integrity,
)

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


class DraftPayload(BaseModel):
    """
    Self-sufficient draft payload for commit.

    This payload contains all data needed for commit_uvil to work
    WITHOUT requiring proposal_id lookup from in-memory state.

    v0.2.10: Introduced to fix demo reliability issues caused by
    server restart/multi-instance scenarios where proposal_id
    cache misses occurred.
    """

    problem_statement: str
    claims: List[DraftClaimResponse]
    # proposal_id is included for display/logging but NOT required for commit
    proposal_id: str


class ProposePartitionRequest(BaseModel):
    """Request to generate a partition proposal."""

    problem_statement: str = Field(..., min_length=1, max_length=10000)


class ProposePartitionResponse(BaseModel):
    """Response containing a draft proposal."""

    proposal_id: str
    claims: List[DraftClaimResponse]
    # v0.2.10: Include full draft_payload for self-sufficient commit
    draft_payload: DraftPayload


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
    """Request to commit UVIL edits.

    v0.2.10: draft_payload makes commit self-sufficient.
    If draft_payload is provided, proposal_id lookup is skipped.
    This fixes demo reliability issues with server restart/multi-instance.
    """

    # proposal_id is optional if draft_payload is provided
    proposal_id: Optional[str] = Field(
        default=None,
        description="Optional if draft_payload provided. For backward compatibility."
    )
    edited_claims: List[EditedClaimRequest]
    user_fingerprint: str = Field(default="anonymous", max_length=64)
    # v0.2.10: Self-sufficient payload - no server-side state needed
    draft_payload: Optional[DraftPayload] = Field(
        default=None,
        description="Self-sufficient payload from propose_partition. If provided, proposal_id lookup is skipped."
    )


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


class EvidencePackResponse(BaseModel):
    """
    Audit-grade evidence pack for replay verification.

    Contains all data needed to independently recompute U_t, R_t, H_t.
    """

    schema_version: str = "evidence_pack_v1"
    committed_partition_snapshot: Dict[str, Any]
    uvil_events: List[Dict[str, Any]]
    reasoning_artifacts: List[Dict[str, Any]]
    u_t: str
    r_t: str
    h_t: str
    h_t_formula_note: str = "H_t = SHA256(R_t || U_t)"
    replay_instructions: Dict[str, Any]
    outcome: str
    authority_basis: Dict[str, Any]


class ReplayVerifyRequest(BaseModel):
    """
    Request to replay-verify an evidence pack.

    Recomputes U_t, R_t, H_t from raw data and compares to recorded values.
    """

    uvil_events: List[Dict[str, Any]]
    reasoning_artifacts: List[Dict[str, Any]]
    expected_u_t: str
    expected_r_t: str
    expected_h_t: str


class ReplayVerifyResponse(BaseModel):
    """Response from replay verification."""

    result: str  # "PASS" or "FAIL"
    computed_u_t: str
    computed_r_t: str
    computed_h_t: str
    expected_u_t: str
    expected_r_t: str
    expected_h_t: str
    match_u_t: bool
    match_r_t: bool
    match_h_t: bool
    diff: Optional[Dict[str, Any]] = None


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

    v0.2.10: Now includes draft_payload for self-sufficient commit.

    Args:
        req: Request containing problem statement

    Returns:
        Proposal with random ID, draft claims (all ADV by default),
        and draft_payload for self-sufficient commit
    """
    global _epoch_counter

    # Generate random proposal_id (exploration-only)
    proposal_id = str(uuid.uuid4())

    # Generate claims via template partitioner (all default to ADV)
    draft_claims = _template_partition(req.problem_statement)

    # Create and store draft proposal (still kept for backward compatibility)
    proposal = DraftProposal(
        proposal_id=proposal_id,
        claims=draft_claims,
        created_at=None,  # Wall-clock not stored for determinism
    )
    _draft_proposals[proposal_id] = proposal

    # Build claim responses
    claim_responses = [
        DraftClaimResponse(
            claim_text=c.claim_text,
            suggested_trust_class=c.suggested_trust_class.value,
            rationale=c.rationale,
        )
        for c in draft_claims
    ]

    # v0.2.10: Build self-sufficient draft_payload
    draft_payload = DraftPayload(
        problem_statement=req.problem_statement,
        claims=claim_responses,
        proposal_id=proposal_id,
    )

    # Convert to response
    return ProposePartitionResponse(
        proposal_id=proposal_id,
        claims=claim_responses,
        draft_payload=draft_payload,
    )


@router.post("/commit_uvil", response_model=CommitUVILResponse)
async def commit_uvil(req: CommitUVILRequest) -> CommitUVILResponse:
    """
    Commit UVIL edits to create immutable snapshot.

    CRITICAL:
    - Creates CommittedPartitionSnapshot with content-derived ID
    - Records UVIL_Event
    - Computes (U_t, R_t, H_t)

    v0.2.10: Self-sufficient mode:
    - If draft_payload is provided, proposal_id lookup is skipped
    - This fixes reliability issues with server restart/multi-instance

    Args:
        req: Request containing edited claims and either proposal_id or draft_payload

    Returns:
        Committed partition ID and attestation

    Raises:
        HTTPException 400: If neither proposal_id nor draft_payload provided
        HTTPException 404: If proposal_id not found (only when draft_payload not provided)
        HTTPException 409: If proposal_id already committed
    """
    global _epoch_counter

    # v0.2.10: Determine proposal_id - from draft_payload or direct
    proposal_id = None
    if req.draft_payload is not None:
        # Self-sufficient mode: use draft_payload, no lookup needed
        proposal_id = req.draft_payload.proposal_id
    elif req.proposal_id is not None:
        # Legacy mode: lookup from in-memory state
        proposal_id = req.proposal_id
        if proposal_id not in _draft_proposals:
            # System-responsible error message (not user-blaming)
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "PROPOSAL_STATE_LOST",
                    "message": "Server lost proposal state. This is a demo reliability issue, not a user error. "
                    "Please refresh and retry.",
                    "proposal_id": proposal_id,
                    "recovery_hint": "The server may have restarted. Refresh the page and start over.",
                },
            )
    else:
        # Neither provided
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "MISSING_PROPOSAL_DATA",
                "message": "Either proposal_id or draft_payload must be provided.",
            },
        )

    # 2. Check double-commit (using proposal_id as idempotency key)
    if proposal_id in _committed_proposal_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "DOUBLE_COMMIT",
                "message": f"Proposal already committed: {proposal_id}. "
                "Each proposal can only be committed once.",
                "proposal_id": proposal_id,
            },
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

    # 3.5 Build snapshot_content early to get committed_partition_id for gate
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

    # 3.6 TRUST-CLASS MONOTONICITY GATE: Verify no trust class changes (Tier A invariant)
    #
    # Per FM §6: "Trust-class monotonicity prevents retroactive upgrade."
    #
    # This gate checks that if any claim_id already exists in the registry,
    # its trust_class matches. If not, this is a monotonicity violation.
    #
    # FAIL-CLOSED: Any mismatch raises TrustClassMonotonicityViolation.
    try:
        require_trust_class_monotonicity(
            claims=snapshot_content["claims"],
            committed_partition_id=committed_partition_id,
        )
    except TrustClassMonotonicityViolation as e:
        # Fail-closed: do not commit on monotonicity violation
        raise HTTPException(
            status_code=422,
            detail=e.to_error_response(),
        ) from e

    # 4. Build CommittedPartitionSnapshot (snapshot_content and ID already computed above)
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
    _committed_proposal_ids.add(proposal_id)

    # 7.5 Register claims in monotonicity registry (after successful commit)
    finalize_claim_registration(
        claims=snapshot_content["claims"],
        committed_partition_id=committed_partition_id,
    )

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

    # 4. Run MV validator on MV claims, build ReasoningArtifacts
    reasoning_artifacts = []
    mv_results = []  # Track MV validation results

    for claim in authority_claims:
        # For MV claims: run the arithmetic validator
        if claim.trust_class == TrustClass.MV:
            validation = validate_mv_claim(claim.claim_text)
            proof_payload = {
                "validator": "arithmetic_v1",
                "claim_text": claim.claim_text,
                "validation_outcome": validation.outcome.value,
                "validation_explanation": validation.explanation,
                "parsed_lhs": validation.parsed_lhs,
                "parsed_rhs": validation.parsed_rhs,
                "computed_value": validation.computed_value,
            }
            mv_results.append(validation.outcome)
        else:
            # FV/PA: no validator in v0
            proof_payload = {"v0_mock": True, "claim_text": claim.claim_text}

        artifact_content = {
            "claim_id": claim.claim_id,
            "trust_class": claim.trust_class.value,
            "proof_payload": proof_payload,
        }
        artifact_id = derive_committed_id(artifact_content)

        reasoning_artifacts.append(
            ReasoningArtifact(
                artifact_id=artifact_id,
                claim_id=claim.claim_id,
                trust_class=claim.trust_class,
                proof_payload=proof_payload,
            )
        )

    # 5. Get all UVIL events for this partition
    partition_events = [
        e for e in _uvil_events if e.committed_partition_id == req.committed_partition_id
    ]

    # 6. Store reasoning artifacts for evidence pack retrieval
    _store_reasoning_artifacts(req.committed_partition_id, reasoning_artifacts)

    # 7. Compute final attestation
    u_t, r_t, h_t = compute_full_attestation(partition_events, reasoning_artifacts)

    # 8. Determine outcome and authority basis
    # Count claims by trust class
    fv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.FV)
    mv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.MV)
    pa_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.PA)
    adv_count = len(snapshot.claims) - len(authority_claims)

    # MV validation summary
    mv_verified = sum(1 for r in mv_results if r == ValidatorOutcome.VERIFIED)
    mv_refuted = sum(1 for r in mv_results if r == ValidatorOutcome.REFUTED)
    mv_abstained = sum(1 for r in mv_results if r == ValidatorOutcome.ABSTAINED)

    # Determine outcome based on validation results
    # Priority: REFUTED > VERIFIED > ABSTAINED
    # VERIFIED only if: all MV claims verified AND no PA/FV claims
    if not authority_claims:
        outcome = Outcome.ABSTAINED
        outcome_explanation = "No authority-bearing claims. Nothing entered the reasoning stream."
        mechanically_verified = False
    elif mv_refuted > 0:
        # Any refuted MV claim → overall REFUTED
        outcome = Outcome.REFUTED
        outcome_explanation = (
            f"Mechanical verification REFUTED: {mv_refuted} MV claim(s) failed arithmetic check."
        )
        mechanically_verified = True  # Verification ran (and failed)
    elif mv_verified > 0 and mv_abstained == 0 and fv_count == 0 and pa_count == 0:
        # All MV claims verified, no other claim types → VERIFIED
        outcome = Outcome.VERIFIED
        outcome_explanation = (
            f"Mechanical verification PASSED: {mv_verified} MV claim(s) verified via arithmetic validator."
        )
        mechanically_verified = True
    elif mv_verified > 0:
        # Some MV verified but mixed with PA/FV/abstained → ABSTAINED overall
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"{mv_verified} MV claim(s) verified, but partition contains "
            f"{pa_count} PA and {fv_count} FV claim(s) that cannot be mechanically verified."
        )
        mechanically_verified = False  # Partial verification
    elif fv_count > 0:
        # FV claims but no FV verifier
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"{fv_count} FV claim(s) require formal proof checker (not implemented in v0)."
        )
        mechanically_verified = False
    elif pa_count > 0:
        # PA-only: user-attested
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"{pa_count} PA claim(s) accepted via user attestation. "
            f"PA claims are authority-bearing but not mechanically verified."
        )
        mechanically_verified = False
    else:
        # MV claims that all abstained (unparseable)
        outcome = Outcome.ABSTAINED
        outcome_explanation = (
            f"{mv_abstained} MV claim(s) could not be parsed by arithmetic validator."
        )
        mechanically_verified = False

    # Build authority basis explanation
    authority_basis = {
        "fv_count": fv_count,
        "mv_count": mv_count,
        "pa_count": pa_count,
        "adv_count": adv_count,
        "adv_excluded": adv_count > 0,
        "mechanically_verified": mechanically_verified,
        "authority_bearing_accepted": len(authority_claims) > 0,
        "explanation": outcome_explanation,
        "mv_validation": {
            "verified": mv_verified,
            "refuted": mv_refuted,
            "abstained": mv_abstained,
        },
        "trust_class_breakdown": {
            "FV": "Formally Verified - requires machine-checkable proof (v0: not implemented)",
            "MV": "Mechanically Validated - checked by arithmetic validator",
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


# ---------------------------------------------------------------------------
# Evidence Pack + Replay Verification
# ---------------------------------------------------------------------------


# Store reasoning artifacts by committed_partition_id for evidence pack retrieval
_partition_reasoning_artifacts: Dict[str, List[ReasoningArtifact]] = {}


def _store_reasoning_artifacts(
    committed_partition_id: str, artifacts: List[ReasoningArtifact]
) -> None:
    """Store reasoning artifacts for evidence pack retrieval."""
    _partition_reasoning_artifacts[committed_partition_id] = artifacts


@router.get("/evidence_pack/{committed_partition_id}", response_model=EvidencePackResponse)
async def get_evidence_pack(
    committed_partition_id: str,
    claimed_h_t: Optional[str] = None,
) -> EvidencePackResponse:
    """
    Generate audit-grade evidence pack for replay verification.

    Returns all data needed to independently recompute U_t, R_t, H_t.
    No external API calls are required to verify the pack.

    Args:
        committed_partition_id: ID of committed partition
        claimed_h_t: Optional H_t claim for verification. If provided, the authority
            gate will verify that the computed H_t matches this claim. If mismatched,
            returns HTTP 422 with SILENT_AUTHORITY_VIOLATION error.

    Returns:
        Evidence pack with all attestation data

    Raises:
        HTTPException 404: If committed_partition_id not found
        HTTPException 422: If claimed_h_t doesn't match computed H_t (authority gate failure)
    """
    # 1. Validate committed_partition_id exists
    if committed_partition_id not in _committed_snapshots:
        raise HTTPException(
            status_code=404,
            detail=f"Committed partition not found: {committed_partition_id}. "
            "Must call /run_verification first to generate artifacts.",
        )

    # 2. Get snapshot
    snapshot = _committed_snapshots[committed_partition_id]

    # 3. Get UVIL events
    partition_events = [
        e for e in _uvil_events if e.committed_partition_id == committed_partition_id
    ]

    # 4. Get or rebuild reasoning artifacts
    if committed_partition_id in _partition_reasoning_artifacts:
        reasoning_artifacts = _partition_reasoning_artifacts[committed_partition_id]
    else:
        # Rebuild artifacts from snapshot
        authority_claims = [
            c for c in snapshot.claims if c.trust_class != TrustClass.ADV
        ]
        reasoning_artifacts = []
        for claim in authority_claims:
            if claim.trust_class == TrustClass.MV:
                validation = validate_mv_claim(claim.claim_text)
                proof_payload = {
                    "validator": "arithmetic_v1",
                    "claim_text": claim.claim_text,
                    "validation_outcome": validation.outcome.value,
                    "validation_explanation": validation.explanation,
                    "parsed_lhs": validation.parsed_lhs,
                    "parsed_rhs": validation.parsed_rhs,
                    "computed_value": validation.computed_value,
                }
            else:
                proof_payload = {"v0_mock": True, "claim_text": claim.claim_text}

            artifact_content = {
                "claim_id": claim.claim_id,
                "trust_class": claim.trust_class.value,
                "proof_payload": proof_payload,
            }
            artifact_id = derive_committed_id(artifact_content)

            reasoning_artifacts.append(
                ReasoningArtifact(
                    artifact_id=artifact_id,
                    claim_id=claim.claim_id,
                    trust_class=claim.trust_class,
                    proof_payload=proof_payload,
                )
            )
        _store_reasoning_artifacts(committed_partition_id, reasoning_artifacts)

    # 5. Build raw payloads for verification
    uvil_event_payloads = [build_uvil_event_payload(e) for e in partition_events]
    reasoning_artifact_payloads = [
        build_reasoning_artifact_payload(a) for a in reasoning_artifacts
    ]

    # 6. AUTHORITY GATE: Enforce "No Silent Authority" (Tier A invariant)
    #
    # This is the MANDATORY gate for producing canonical evidence pack output.
    # Per FM §4: "Nothing that influences durable learning authority may occur silently."
    #
    # Evidence packs are "durable influence" because they:
    # - Can be consumed as canonical proof of attestation
    # - Persist beyond UI display
    # - Could feed downstream learning/audit systems
    #
    # The gate recomputes U_t, R_t, H_t and fails-closed on any issue.
    authority_request = AuthorityUpdateRequest(
        committed_partition_id=committed_partition_id,
        uvil_events=uvil_event_payloads,
        reasoning_artifacts=reasoning_artifact_payloads,
        claimed_h_t=claimed_h_t,  # If provided, gate verifies against claim
    )

    try:
        h_t = require_epoch_root(authority_request)
    except SilentAuthorityViolation as e:
        # Fail-closed: do not produce evidence pack on gate failure
        # Return structured error response per governance spec
        raise HTTPException(
            status_code=422,
            detail=e.to_error_response(),
        ) from e

    # Gate passed - compute U_t, R_t for response (H_t already verified)
    u_t = compute_ui_root(uvil_event_payloads)
    r_t = compute_reasoning_root(reasoning_artifact_payloads)

    # 7. Determine outcome (same logic as run_verification)
    authority_claims = [
        c for c in snapshot.claims if c.trust_class != TrustClass.ADV
    ]
    mv_results = []
    for claim in authority_claims:
        if claim.trust_class == TrustClass.MV:
            validation = validate_mv_claim(claim.claim_text)
            mv_results.append(validation.outcome)

    fv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.FV)
    mv_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.MV)
    pa_count = sum(1 for c in authority_claims if c.trust_class == TrustClass.PA)
    adv_count = len(snapshot.claims) - len(authority_claims)

    mv_verified = sum(1 for r in mv_results if r == ValidatorOutcome.VERIFIED)
    mv_refuted = sum(1 for r in mv_results if r == ValidatorOutcome.REFUTED)
    mv_abstained = sum(1 for r in mv_results if r == ValidatorOutcome.ABSTAINED)

    if not authority_claims:
        outcome = Outcome.ABSTAINED
        outcome_explanation = "No authority-bearing claims. Nothing entered the reasoning stream."
        mechanically_verified = False
    elif mv_refuted > 0:
        outcome = Outcome.REFUTED
        outcome_explanation = f"Mechanical verification REFUTED: {mv_refuted} MV claim(s) failed."
        mechanically_verified = True
    elif mv_verified > 0 and mv_abstained == 0 and fv_count == 0 and pa_count == 0:
        outcome = Outcome.VERIFIED
        outcome_explanation = f"Mechanical verification PASSED: {mv_verified} MV claim(s) verified."
        mechanically_verified = True
    elif mv_verified > 0:
        outcome = Outcome.ABSTAINED
        outcome_explanation = f"{mv_verified} MV verified, but mixed with PA/FV."
        mechanically_verified = False
    elif fv_count > 0:
        outcome = Outcome.ABSTAINED
        outcome_explanation = f"{fv_count} FV claim(s) require formal proof checker (v0: not implemented)."
        mechanically_verified = False
    elif pa_count > 0:
        outcome = Outcome.ABSTAINED
        outcome_explanation = f"{pa_count} PA claim(s) accepted via user attestation."
        mechanically_verified = False
    else:
        outcome = Outcome.ABSTAINED
        outcome_explanation = f"{mv_abstained} MV claim(s) could not be parsed."
        mechanically_verified = False

    authority_basis = {
        "fv_count": fv_count,
        "mv_count": mv_count,
        "pa_count": pa_count,
        "adv_count": adv_count,
        "adv_excluded": adv_count > 0,
        "mechanically_verified": mechanically_verified,
        "authority_bearing_accepted": len(authority_claims) > 0,
        "explanation": outcome_explanation,
        "mv_validation": {
            "verified": mv_verified,
            "refuted": mv_refuted,
            "abstained": mv_abstained,
        },
    }

    # 8. Build snapshot dict
    snapshot_dict = {
        "committed_partition_id": snapshot.committed_partition_id,
        "commit_epoch": snapshot.commit_epoch,
        "claims": [
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "trust_class": c.trust_class.value,
                "rationale": c.rationale,
            }
            for c in snapshot.claims
        ],
    }

    return EvidencePackResponse(
        schema_version="evidence_pack_v1",
        committed_partition_snapshot=snapshot_dict,
        uvil_events=uvil_event_payloads,
        reasoning_artifacts=reasoning_artifact_payloads,
        u_t=u_t,
        r_t=r_t,
        h_t=h_t,
        h_t_formula_note="H_t = SHA256(R_t || U_t)",
        replay_instructions={
            "step_1": "Recompute U_t = compute_ui_root(uvil_events)",
            "step_2": "Recompute R_t = compute_reasoning_root(reasoning_artifacts)",
            "step_3": "Recompute H_t = SHA256(R_t || U_t)",
            "step_4": "Compare computed values to recorded u_t, r_t, h_t",
            "expected_result": "All three must match exactly",
            "cli_command": "curl -X POST http://localhost:8000/uvil/replay_verify -H 'Content-Type: application/json' -d @evidence_pack.json",
        },
        outcome=outcome.value,
        authority_basis=authority_basis,
    )


@router.post("/replay_verify", response_model=ReplayVerifyResponse)
async def replay_verify(req: ReplayVerifyRequest) -> ReplayVerifyResponse:
    """
    Replay-verify an evidence pack by recomputing attestation hashes.

    Takes raw UVIL events and reasoning artifacts, recomputes U_t, R_t, H_t,
    and compares to expected values. Returns PASS if all match, FAIL otherwise.

    CRITICAL: No external API calls - all computation is local.

    Args:
        req: Evidence pack data with expected hash values

    Returns:
        Verification result with computed vs expected comparison
    """
    # 1. Recompute U_t from UVIL events
    computed_u_t = compute_ui_root(req.uvil_events)

    # 2. Recompute R_t from reasoning artifacts
    computed_r_t = compute_reasoning_root(req.reasoning_artifacts)

    # 3. Recompute H_t = SHA256(R_t || U_t)
    computed_h_t = compute_composite_root(computed_r_t, computed_u_t)

    # 4. Compare
    match_u_t = computed_u_t == req.expected_u_t
    match_r_t = computed_r_t == req.expected_r_t
    match_h_t = computed_h_t == req.expected_h_t

    all_match = match_u_t and match_r_t and match_h_t
    result = "PASS" if all_match else "FAIL"

    # 5. Build diff on failure
    diff = None
    if not all_match:
        diff = {
            "u_t_diff": None if match_u_t else {
                "computed": computed_u_t,
                "expected": req.expected_u_t,
            },
            "r_t_diff": None if match_r_t else {
                "computed": computed_r_t,
                "expected": req.expected_r_t,
            },
            "h_t_diff": None if match_h_t else {
                "computed": computed_h_t,
                "expected": req.expected_h_t,
            },
        }

    return ReplayVerifyResponse(
        result=result,
        computed_u_t=computed_u_t,
        computed_r_t=computed_r_t,
        computed_h_t=computed_h_t,
        expected_u_t=req.expected_u_t,
        expected_r_t=req.expected_r_t,
        expected_h_t=req.expected_h_t,
        match_u_t=match_u_t,
        match_r_t=match_r_t,
        match_h_t=match_h_t,
        diff=diff,
    )


# ---------------------------------------------------------------------------
# Rejection Demo Endpoints (v0.2.1)
# ---------------------------------------------------------------------------


class ChangeTrustClassRequest(BaseModel):
    """Request to change trust class of a committed claim (will fail)."""

    committed_partition_id: str
    claim_index: int
    new_trust_class: str


class VerifyAttestationRequest(BaseModel):
    """Request to verify attestation with claimed H_t (will fail if tampered)."""

    committed_partition_id: str
    claimed_h_t: str


@router.post("/change_trust_class")
async def change_trust_class(req: ChangeTrustClassRequest):
    """
    Attempt to change trust class of a committed claim.

    This endpoint demonstrates TRUST_CLASS_MONOTONICITY_VIOLATION.
    Trust classes are immutable after commit - this will always fail.

    Args:
        req: Request with committed partition ID and new trust class

    Raises:
        HTTPException 422: Always - trust class cannot be changed
    """
    # Validate partition exists
    if req.committed_partition_id not in _committed_snapshots:
        raise HTTPException(
            status_code=404,
            detail=f"Committed partition not found: {req.committed_partition_id}",
        )

    snapshot = _committed_snapshots[req.committed_partition_id]

    if req.claim_index < 0 or req.claim_index >= len(snapshot.claims):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid claim index: {req.claim_index}",
        )

    claim = snapshot.claims[req.claim_index]

    # Always reject - trust class is immutable
    raise HTTPException(
        status_code=422,
        detail={
            "error_code": "TRUST_CLASS_MONOTONICITY_VIOLATION",
            "message": "Cannot change trust class of committed claim. "
            "Trust classes are immutable after commit.",
            "claim_id": claim.claim_id,
            "current_trust_class": claim.trust_class.value,
            "attempted_trust_class": req.new_trust_class,
            "committed_partition_id": req.committed_partition_id,
        },
    )


@router.post("/verify_attestation")
async def verify_attestation(req: VerifyAttestationRequest):
    """
    Verify attestation by comparing claimed H_t to computed H_t.

    This endpoint demonstrates SILENT_AUTHORITY_VIOLATION when
    the claimed_h_t doesn't match the computed H_t.

    Args:
        req: Request with committed partition ID and claimed H_t

    Raises:
        HTTPException 422: If claimed H_t doesn't match computed H_t
    """
    # Validate partition exists
    if req.committed_partition_id not in _committed_snapshots:
        raise HTTPException(
            status_code=404,
            detail=f"Committed partition not found: {req.committed_partition_id}",
        )

    snapshot = _committed_snapshots[req.committed_partition_id]

    # Get UVIL events
    partition_events = [
        e for e in _uvil_events if e.committed_partition_id == req.committed_partition_id
    ]

    # Get reasoning artifacts
    if req.committed_partition_id in _partition_reasoning_artifacts:
        reasoning_artifacts = _partition_reasoning_artifacts[req.committed_partition_id]
    else:
        # Need to run verification first
        raise HTTPException(
            status_code=400,
            detail="Must call /run_verification first to generate reasoning artifacts.",
        )

    # Compute actual attestation
    from governance.uvil import build_uvil_event_payload, build_reasoning_artifact_payload

    uvil_event_payloads = [build_uvil_event_payload(e) for e in partition_events]
    reasoning_artifact_payloads = [
        build_reasoning_artifact_payload(a) for a in reasoning_artifacts
    ]

    u_t = compute_ui_root(uvil_event_payloads)
    r_t = compute_reasoning_root(reasoning_artifact_payloads)
    computed_h_t = compute_composite_root(r_t, u_t)

    # Compare
    if req.claimed_h_t != computed_h_t:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "SILENT_AUTHORITY_VIOLATION",
                "message": "Attestation verification failed. "
                "Claimed H_t does not match computed H_t. "
                "This indicates tampering or corruption.",
                "claimed_h_t": req.claimed_h_t,
                "computed_h_t": computed_h_t,
                "committed_partition_id": req.committed_partition_id,
            },
        )

    return {
        "result": "PASS",
        "committed_partition_id": req.committed_partition_id,
        "h_t": computed_h_t,
    }
