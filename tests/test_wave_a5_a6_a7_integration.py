from __future__ import annotations

from copy import deepcopy

import pytest

from mathledger.evidence import build_evidence_pack, verify_evidence_pack
from mathledger.governance import TrustClass, validate_mv_claim
from mathledger.integration import (
    AAKBridgeViolation,
    CPFAdapterViolation,
    ExperimentRunnerViolation,
    assert_lane_boundaries,
    build_aak_bridge_packet,
    build_lane_b_psych_context,
    cpf_snapshot_json,
    normalize_cpf_assessments,
    run_three_lane_experiment,
    validate_lane_b_reference,
)
from mathledger.uvil import DraftClaim, DraftProposal, commit_from_draft, derive_content_id


def _build_base_evidence_pack():
    commit_result = commit_from_draft(
        DraftProposal(proposal_id="draft-x", claims=[DraftClaim("ignore", TrustClass.ADV)]),
        [
            {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arith"},
            {"claim_text": "Speculative note", "trust_class": "ADV", "rationale": "advisory"},
        ],
        commit_epoch=2,
        user_fingerprint="u1",
    )

    reasoning_artifacts = []
    for claim in commit_result.snapshot.claims:
        if claim.trust_class == TrustClass.MV:
            mv = validate_mv_claim(claim.claim_text)
            outcome = mv.outcome.value
            proof_payload = {"claim_text": claim.claim_text, "explanation": mv.explanation}
        else:
            outcome = "ABSTAINED"
            proof_payload = {"claim_text": claim.claim_text, "note": "advisory"}

        payload = {
            "claim_id": claim.claim_id,
            "trust_class": claim.trust_class.value,
            "validation_outcome": outcome,
            "proof_payload": proof_payload,
        }
        artifact_id = derive_content_id(payload)
        reasoning_artifacts.append({"artifact_id": artifact_id, **payload})

    return build_evidence_pack(
        committed_partition_snapshot=commit_result.snapshot,
        uvil_events=[commit_result.event],
        reasoning_artifacts=reasoning_artifacts,
    )


def _sample_cpf_assessments():
    return [
        {
            "indicator_id": "7.2",
            "category": "stress",
            "bayesian_score": 0.83,
            "confidence": 0.9,
            "maturity_level": "high",
            "assessor": "SIEM-Simulator",
            "raw_data": {"source": "simulator", "event_count": 4},
        },
        {
            "indicator_id": "1.3",
            "category": "authority",
            "bayesian_score": 0.61,
            "confidence": 0.8,
            "maturity_level": "medium",
            "assessor": "SIEM-Simulator",
            "raw_data": {"source": "simulator", "event_count": 3},
        },
    ]


def test_a5_bridge_links_lane_a_hashes_without_authority_escalation():
    evidence_pack = _build_base_evidence_pack()

    packet = build_aak_bridge_packet(
        evidence_pack=evidence_pack,
        snapshot_id="spf_1100",
        convergence_score=0.75,
        elevated_categories=["stress", "authority"],
        elevated_indicators=[{"indicator_id": "7.2", "activation_level": 83.0}],
        psych_root_hash="a" * 64,
    )

    assert packet["mathledger_refs"]["u_t"] == evidence_pack["u_t"]
    assert packet["mathledger_refs"]["r_t"] == evidence_pack["r_t"]
    assert packet["mathledger_refs"]["h_t"] == evidence_pack["h_t"]
    assert "trust_class" not in packet["psych_context"]
    assert validate_lane_b_reference(packet["psych_context"], evidence_pack=evidence_pack)


def test_a5_bridge_rejects_authority_keys_and_mismatched_refs():
    evidence_pack = _build_base_evidence_pack()
    context = build_lane_b_psych_context(evidence_pack=evidence_pack)

    with_authority_key = deepcopy(context)
    with_authority_key["trust_class"] = "MV"
    with pytest.raises(AAKBridgeViolation):
        validate_lane_b_reference(with_authority_key, evidence_pack=evidence_pack)

    with_bad_ref = deepcopy(context)
    with_bad_ref["reasoning_root_ref"] = "0" * 64
    with pytest.raises(AAKBridgeViolation):
        validate_lane_b_reference(with_bad_ref, evidence_pack=evidence_pack)


def test_a6_cpf_adapter_normalizes_deterministically():
    assessments_a = _sample_cpf_assessments()
    assessments_b = list(reversed(_sample_cpf_assessments()))

    snap_a = normalize_cpf_assessments(assessments_a, snapshot_epoch=10)
    snap_b = normalize_cpf_assessments(assessments_b, snapshot_epoch=10)

    assert snap_a["snapshot_id"] == snap_b["snapshot_id"]
    assert snap_a["indicators"] == snap_b["indicators"]
    assert cpf_snapshot_json(snap_a) == cpf_snapshot_json(snap_b)


def test_a6_cpf_adapter_rejects_invalid_scores():
    assessments = _sample_cpf_assessments()
    assessments[0]["bayesian_score"] = 1.5

    with pytest.raises(CPFAdapterViolation):
        normalize_cpf_assessments(assessments, snapshot_epoch=1)


def test_a7_runner_is_reproducible_and_boundary_safe():
    edited_claims = [
        {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arith"},
        {"claim_text": "Speculative branch", "trust_class": "ADV", "rationale": "lane b"},
    ]
    cpf_assessments = _sample_cpf_assessments()

    run_a = run_three_lane_experiment(
        edited_claims=edited_claims,
        cpf_assessments=cpf_assessments,
        commit_epoch=12,
        proposal_id="proposal-runner",
        user_fingerprint="analyst",
    )
    run_b = run_three_lane_experiment(
        edited_claims=edited_claims,
        cpf_assessments=cpf_assessments,
        commit_epoch=12,
        proposal_id="proposal-runner",
        user_fingerprint="analyst",
    )

    assert run_a["run_id"] == run_b["run_id"]
    assert run_a["lane_a"]["evidence_pack"]["u_t"] == run_b["lane_a"]["evidence_pack"]["u_t"]
    assert run_a["lane_a"]["evidence_pack"]["r_t"] == run_b["lane_a"]["evidence_pack"]["r_t"]
    assert run_a["lane_a"]["evidence_pack"]["h_t"] == run_b["lane_a"]["evidence_pack"]["h_t"]
    assert run_a["lane_a"]["evidence_pack"]["counts"]["advisory_reasoning_artifact_count"] == 1

    verification = verify_evidence_pack(run_a["lane_a"]["evidence_pack"])
    assert verification["overall_pass"] is True
    assert run_a["lane_b"]["aak_bridge_packet"]["mathledger_refs"]["h_t"] == run_a["lane_a"]["evidence_pack"]["h_t"]

    assert_lane_boundaries(run_a)


def test_a7_runner_detects_lane_boundary_tampering():
    run_result = run_three_lane_experiment(
        edited_claims=[{"claim_text": "2 + 2 = 4", "trust_class": "MV"}],
        cpf_assessments=_sample_cpf_assessments(),
        commit_epoch=22,
        proposal_id="proposal-tamper",
        user_fingerprint="auditor",
    )

    tampered_roots = deepcopy(run_result)
    tampered_roots["lane_b"]["aak_bridge_packet"]["mathledger_refs"]["r_t"] = "0" * 64
    with pytest.raises(ExperimentRunnerViolation):
        assert_lane_boundaries(tampered_roots)

    tampered_cpf = deepcopy(run_result)
    tampered_cpf["lane_c"]["cpf_snapshot"]["snapshot_id"] = "f" * 64
    with pytest.raises(ExperimentRunnerViolation):
        assert_lane_boundaries(tampered_cpf)

