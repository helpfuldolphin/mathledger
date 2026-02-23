from __future__ import annotations

from dataclasses import asdict

import pytest

from mathledger.governance import TrustClass
from mathledger.uvil import (
    DraftClaim,
    DraftProposal,
    UVILBoundaryViolation,
    build_committed_snapshot,
    commit_from_draft,
    commit_result_to_dict,
)


def _collect_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    stack: list[object] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                keys.add(str(key))
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return keys


def test_commit_id_independent_of_proposal_id():
    edited_claims = [
        {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arithmetic"},
        {"claim_text": "Policy approved", "trust_class": "PA", "rationale": "attested"},
    ]

    draft_a = DraftProposal(
        proposal_id="proposal-a",
        claims=[DraftClaim("ignored", TrustClass.ADV)],
    )
    draft_b = DraftProposal(
        proposal_id="proposal-b",
        claims=[DraftClaim("different exploration", TrustClass.ADV)],
    )

    result_a = commit_from_draft(draft_a, edited_claims, commit_epoch=7, user_fingerprint="u1")
    result_b = commit_from_draft(draft_b, edited_claims, commit_epoch=7, user_fingerprint="u1")

    assert result_a.snapshot.committed_partition_id == result_b.snapshot.committed_partition_id
    assert result_a.event.event_id == result_b.event.event_id
    assert [c.claim_id for c in result_a.snapshot.claims] == [c.claim_id for c in result_b.snapshot.claims]


def test_commit_payload_contains_no_exploration_keys():
    edited_claims = [
        {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arithmetic"},
    ]
    draft = DraftProposal(proposal_id="proposal-z", claims=[DraftClaim("ignored", TrustClass.ADV)])
    result = commit_from_draft(draft, edited_claims, commit_epoch=1, user_fingerprint="u")
    payload = commit_result_to_dict(result)

    keys = _collect_keys(payload)
    assert "proposal_id" not in keys
    assert "draft_id" not in keys
    assert "exploration_id" not in keys


def test_fail_closed_on_forbidden_exploration_key_in_claim_input():
    edited_claims = [
        {
            "claim_text": "2 + 2 = 4",
            "trust_class": "MV",
            "rationale": "arithmetic",
            "proposal_id": "leak",
        }
    ]
    with pytest.raises(UVILBoundaryViolation):
        build_committed_snapshot(edited_claims, commit_epoch=1)


def test_fail_closed_on_unknown_claim_keys():
    edited_claims = [
        {
            "claim_text": "2 + 2 = 4",
            "trust_class": "MV",
            "rationale": "arithmetic",
            "extra_field": "unexpected",
        }
    ]
    with pytest.raises(UVILBoundaryViolation):
        build_committed_snapshot(edited_claims, commit_epoch=1)


def test_fail_closed_on_empty_commit_set():
    with pytest.raises(UVILBoundaryViolation):
        build_committed_snapshot([], commit_epoch=1)


def test_fail_closed_on_empty_claim_text():
    edited_claims = [{"claim_text": "   ", "trust_class": "MV", "rationale": ""}]
    with pytest.raises(UVILBoundaryViolation):
        build_committed_snapshot(edited_claims, commit_epoch=1)


def test_fail_closed_on_invalid_trust_class():
    edited_claims = [{"claim_text": "2 + 2 = 4", "trust_class": "UNKNOWN"}]
    with pytest.raises(UVILBoundaryViolation):
        build_committed_snapshot(edited_claims, commit_epoch=1)


def test_snapshot_and_event_are_deterministic_for_same_inputs():
    edited_claims = [
        {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "arithmetic"},
    ]
    draft = DraftProposal(proposal_id="random-a", claims=[DraftClaim("x", TrustClass.ADV)])

    res1 = commit_from_draft(draft, edited_claims, commit_epoch=4, user_fingerprint="alice")
    res2 = commit_from_draft(draft, edited_claims, commit_epoch=4, user_fingerprint="alice")

    assert asdict(res1.snapshot) == asdict(res2.snapshot)
    assert asdict(res1.event) == asdict(res2.event)

