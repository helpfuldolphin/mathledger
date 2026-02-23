from __future__ import annotations

from copy import deepcopy

import pytest

from mathledger.evidence import (
    EvidenceReplayViolation,
    assert_evidence_pack_passes,
    build_evidence_pack,
    compute_r_t_from_artifacts,
    evidence_pack_json,
    verify_evidence_pack,
)
from mathledger.governance import TrustClass, validate_mv_claim
from mathledger.uvil import DraftClaim, DraftProposal, commit_from_draft, derive_content_id


def _build_commit_result():
    draft = DraftProposal(
        proposal_id="draft-random-id",
        claims=[DraftClaim("exploration note", TrustClass.ADV)],
    )
    edited_claims = [
        {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arith route"},
        {
            "claim_text": "This is a suggestion only.",
            "trust_class": "ADV",
            "rationale": "advisory lane",
        },
    ]
    return commit_from_draft(draft, edited_claims, commit_epoch=5, user_fingerprint="auditor-1")


def _build_reasoning_artifacts(snapshot):
    artifacts = []
    for claim in snapshot.claims:
        if claim.trust_class == TrustClass.MV:
            mv_result = validate_mv_claim(claim.claim_text)
            proof_payload = {
                "validator": "mv_arithmetic_v0",
                "claim_text": claim.claim_text,
                "explanation": mv_result.explanation,
            }
            validation_outcome = mv_result.outcome.value
        else:
            proof_payload = {
                "validator": "none",
                "claim_text": claim.claim_text,
                "note": "advisory artifact recorded for transparency",
            }
            validation_outcome = "ABSTAINED"

        artifact_payload = {
            "claim_id": claim.claim_id,
            "trust_class": claim.trust_class.value,
            "validation_outcome": validation_outcome,
            "proof_payload": proof_payload,
        }
        artifact_id = derive_content_id(artifact_payload)
        artifacts.append({"artifact_id": artifact_id, **artifact_payload})
    return artifacts


def _build_base_pack():
    commit_result = _build_commit_result()
    reasoning_artifacts = _build_reasoning_artifacts(commit_result.snapshot)
    pack = build_evidence_pack(
        committed_partition_snapshot=commit_result.snapshot,
        uvil_events=[commit_result.event],
        reasoning_artifacts=reasoning_artifacts,
    )
    return pack, reasoning_artifacts


def test_build_and_verify_evidence_pack_passes_and_quarantines_adv():
    pack, reasoning_artifacts = _build_base_pack()
    verification = verify_evidence_pack(pack)

    assert verification["overall_pass"] is True
    assert verification["counts"]["advisory_reasoning_artifact_count"] == 1
    assert verification["counts"]["authority_reasoning_artifact_count"] == 1

    authority_only = [a for a in reasoning_artifacts if a["trust_class"] != "ADV"]
    assert pack["r_t"] == compute_r_t_from_artifacts(reasoning_artifacts)
    assert pack["r_t"] == compute_r_t_from_artifacts(authority_only)


def test_replay_deterministic_for_same_inputs():
    pack_1, _ = _build_base_pack()
    pack_2, _ = _build_base_pack()

    assert pack_1["u_t"] == pack_2["u_t"]
    assert pack_1["r_t"] == pack_2["r_t"]
    assert pack_1["h_t"] == pack_2["h_t"]
    assert evidence_pack_json(pack_1) == evidence_pack_json(pack_2)


def test_tampered_uvil_event_detected():
    pack, _ = _build_base_pack()
    tampered = deepcopy(pack)
    tampered["uvil_events"][0]["event_type"] = "EDIT"

    verification = verify_evidence_pack(tampered)

    assert verification["overall_pass"] is False
    assert verification["matches"]["u_t"] is False
    assert verification["matches"]["r_t"] is True
    assert verification["matches"]["h_t"] is False


def test_tampered_reasoning_artifact_detected():
    pack, _ = _build_base_pack()
    tampered = deepcopy(pack)
    for artifact in tampered["reasoning_artifacts"]:
        if artifact["trust_class"] == "MV":
            artifact["proof_payload"]["claim_text"] = "2 + 2 = 5"
            artifact["validation_outcome"] = "REFUTED"
            break

    verification = verify_evidence_pack(tampered)

    assert verification["overall_pass"] is False
    assert verification["matches"]["u_t"] is True
    assert verification["matches"]["r_t"] is False
    assert verification["matches"]["h_t"] is False


def test_tampered_composite_root_detected():
    pack, _ = _build_base_pack()
    tampered = deepcopy(pack)
    tampered["h_t"] = "0" * 64

    verification = verify_evidence_pack(tampered)

    assert verification["overall_pass"] is False
    assert verification["matches"]["u_t"] is True
    assert verification["matches"]["r_t"] is True
    assert verification["matches"]["h_t"] is False


def test_verify_rejects_missing_required_field():
    pack, _ = _build_base_pack()
    malformed = deepcopy(pack)
    del malformed["u_t"]

    with pytest.raises(EvidenceReplayViolation):
        verify_evidence_pack(malformed)


def test_verify_rejects_missing_validation_outcome_on_authority_artifact():
    pack, _ = _build_base_pack()
    malformed = deepcopy(pack)
    for artifact in malformed["reasoning_artifacts"]:
        if artifact["trust_class"] == "MV":
            del artifact["validation_outcome"]
            break

    with pytest.raises(EvidenceReplayViolation):
        verify_evidence_pack(malformed)


def test_assert_wrapper_fails_closed_on_mismatch():
    pack, _ = _build_base_pack()
    tampered = deepcopy(pack)
    tampered["h_t"] = "f" * 64

    with pytest.raises(EvidenceReplayViolation):
        assert_evidence_pack_passes(tampered)


def test_adv_only_artifacts_produce_empty_reasoning_root():
    commit_result = commit_from_draft(
        DraftProposal(proposal_id="p-adv", claims=[DraftClaim("x", TrustClass.ADV)]),
        [{"claim_text": "Advisory claim", "trust_class": "ADV", "rationale": "suggestion"}],
        commit_epoch=9,
        user_fingerprint="adv-user",
    )
    claim = commit_result.snapshot.claims[0]
    artifacts = [
        {
            "artifact_id": derive_content_id({"claim_id": claim.claim_id, "trust_class": "ADV"}),
            "claim_id": claim.claim_id,
            "trust_class": "ADV",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {"note": "adv only"},
        }
    ]

    pack = build_evidence_pack(
        committed_partition_snapshot=commit_result.snapshot,
        uvil_events=[commit_result.event],
        reasoning_artifacts=artifacts,
    )
    verification = verify_evidence_pack(pack)

    assert verification["overall_pass"] is True
    assert verification["counts"]["authority_reasoning_artifact_count"] == 0
    assert pack["r_t"] == compute_r_t_from_artifacts([])

