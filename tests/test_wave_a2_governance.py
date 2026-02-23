import pytest

from mathledger.governance import (
    AbstentionPreservationViolation,
    AuthorityRoutingViolation,
    TrustClass,
    ValidationOutcome,
    build_authority_reasoning_leaves,
    build_authority_reasoning_payloads,
    enforce_authority_stream_constraints,
    is_authority_bearing,
    parse_trust_class,
    require_abstention_preservation,
    route_claims_by_trust_class,
    validate_outcome_aggregation,
    validate_mv_claim,
    verify_not_coerced_to_null,
    verify_outcome_present,
)


def test_parse_trust_class_normalizes_case():
    assert parse_trust_class("mv") == TrustClass.MV
    assert parse_trust_class("  adv ") == TrustClass.ADV


def test_is_authority_bearing():
    assert is_authority_bearing("FV")
    assert is_authority_bearing("MV")
    assert is_authority_bearing("PA")
    assert not is_authority_bearing("ADV")


def test_route_claims_quarantines_adv():
    claims = [
        {"claim_id": "c1", "trust_class": "MV", "claim_text": "2 + 2 = 4"},
        {"claim_id": "c2", "trust_class": "ADV", "claim_text": "speculation"},
        {"claim_id": "c3", "trust_class": "PA", "claim_text": "attestation"},
    ]
    routed = route_claims_by_trust_class(claims)
    assert len(routed.authority_claims) == 2
    assert len(routed.advisory_claims) == 1
    assert routed.advisory_claims[0]["trust_class"] == "ADV"


def test_route_claims_rejects_invalid_trust_class():
    claims = [{"claim_id": "x", "trust_class": "UNKNOWN"}]
    with pytest.raises(AuthorityRoutingViolation):
        route_claims_by_trust_class(claims)


def test_enforce_authority_stream_rejects_adv():
    artifacts = [
        {
            "claim_id": "a1",
            "trust_class": "ADV",
            "validation_outcome": "ABSTAINED",
        }
    ]
    with pytest.raises(AuthorityRoutingViolation):
        enforce_authority_stream_constraints(artifacts)


def test_verify_outcome_present_accepts_valid_values():
    for outcome in ("VERIFIED", "REFUTED", "ABSTAINED"):
        artifact = {
            "claim_id": "c",
            "trust_class": "MV",
            "validation_outcome": outcome,
        }
        assert verify_outcome_present(artifact) == outcome


def test_verify_outcome_present_rejects_missing_field():
    artifact = {"claim_id": "c", "trust_class": "MV"}
    with pytest.raises(AbstentionPreservationViolation):
        verify_outcome_present(artifact)


def test_verify_outcome_present_rejects_null():
    artifact = {"claim_id": "c", "trust_class": "MV", "validation_outcome": None}
    with pytest.raises(AbstentionPreservationViolation):
        verify_outcome_present(artifact)


def test_verify_outcome_present_rejects_invalid_value():
    artifact = {"claim_id": "c", "trust_class": "MV", "validation_outcome": "PENDING"}
    with pytest.raises(AbstentionPreservationViolation):
        verify_outcome_present(artifact)


def test_require_abstention_preservation_batch():
    artifacts = [
        {"claim_id": "c1", "trust_class": "MV", "validation_outcome": "VERIFIED"},
        {"claim_id": "c2", "trust_class": "PA", "validation_outcome": "ABSTAINED"},
    ]
    require_abstention_preservation(artifacts)


def test_validate_outcome_aggregation_rules():
    assert validate_outcome_aggregation([]) == ValidationOutcome.ABSTAINED.value
    assert (
        validate_outcome_aggregation(["VERIFIED", "ABSTAINED"])
        == ValidationOutcome.ABSTAINED.value
    )
    assert (
        validate_outcome_aggregation(["VERIFIED", "ABSTAINED", "REFUTED"])
        == ValidationOutcome.REFUTED.value
    )
    assert (
        validate_outcome_aggregation(["VERIFIED", "VERIFIED"])
        == ValidationOutcome.VERIFIED.value
    )


def test_validate_outcome_aggregation_rejects_null():
    with pytest.raises(AbstentionPreservationViolation):
        validate_outcome_aggregation(["VERIFIED", None])


def test_verify_not_coerced_to_null():
    before = {"claim_id": "c1", "validation_outcome": "ABSTAINED"}
    after = {"claim_id": "c1", "validation_outcome": None}
    with pytest.raises(AbstentionPreservationViolation):
        verify_not_coerced_to_null(before, after)


def test_build_authority_reasoning_payloads_enforces_both_gates():
    artifacts = [
        {
            "claim_id": "m1",
            "trust_class": "MV",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {"note": "out of scope"},
        },
        {
            "claim_id": "p1",
            "trust_class": "PA",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {},
        },
    ]
    payloads = build_authority_reasoning_payloads(artifacts)
    assert len(payloads) == 2
    assert all(p["trust_class"] in {"MV", "PA"} for p in payloads)
    assert all(p["validation_outcome"] == "ABSTAINED" for p in payloads)


def test_build_authority_reasoning_leaves_deterministic():
    artifacts = [
        {
            "claim_id": "m1",
            "trust_class": "MV",
            "validation_outcome": "VERIFIED",
            "proof_payload": {"computed": 4},
        }
    ]
    leaves_1 = build_authority_reasoning_leaves(artifacts)
    leaves_2 = build_authority_reasoning_leaves(artifacts)
    assert leaves_1 == leaves_2


def test_build_authority_reasoning_payloads_rejects_adv():
    artifacts = [
        {
            "claim_id": "a1",
            "trust_class": "ADV",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {},
        }
    ]
    with pytest.raises(AuthorityRoutingViolation):
        build_authority_reasoning_payloads(artifacts)


def test_mv_validator_determinism_and_outcomes():
    assert validate_mv_claim("2 + 2 = 4").outcome.value == "VERIFIED"
    assert validate_mv_claim("2 + 2 = 5").outcome.value == "REFUTED"
    assert validate_mv_claim("sqrt(2) is irrational").outcome.value == "ABSTAINED"

