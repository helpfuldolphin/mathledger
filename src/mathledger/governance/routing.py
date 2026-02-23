"""Trust-class routing and authority stream gates for Wave A2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from mathledger.governance.abstention import verify_outcome_present
from mathledger.governance.trust_class import TrustClass, is_authority_bearing, parse_trust_class


class AuthorityRoutingViolation(ValueError):
    """Raised when claims/artifacts violate authority routing constraints."""

    ERROR_CODE = "AUTHORITY_ROUTING_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        artifact_index: int | None = None,
        claim_id: str | None = None,
        trust_class: str | None = None,
        violation_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.artifact_index = artifact_index
        self.claim_id = claim_id
        self.trust_class = trust_class
        self.violation_type = violation_type

    def to_error_response(self) -> dict[str, Any]:
        """Structured violation payload."""
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "artifact_index": self.artifact_index,
            "claim_id": self.claim_id,
            "trust_class": self.trust_class,
            "violation_type": self.violation_type,
        }


@dataclass(frozen=True)
class RoutedClaims:
    """Routing result split by authority-bearing vs advisory claims."""

    authority_claims: Tuple[dict[str, Any], ...]
    advisory_claims: Tuple[dict[str, Any], ...]


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    """Deterministic compact JSON encoding."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def route_claims_by_trust_class(claims: Sequence[Mapping[str, Any]]) -> RoutedClaims:
    """
    Route claims into authority-bearing and advisory partitions.

    Fail-closed on missing or invalid trust_class values.
    """
    authority: list[dict[str, Any]] = []
    advisory: list[dict[str, Any]] = []

    for idx, claim in enumerate(claims):
        if "trust_class" not in claim:
            raise AuthorityRoutingViolation(
                "Claim missing trust_class.",
                artifact_index=idx,
                claim_id=str(claim.get("claim_id", "unknown")),
                violation_type="MISSING_TRUST_CLASS",
            )
        try:
            tc = parse_trust_class(claim["trust_class"])
        except ValueError as exc:
            raise AuthorityRoutingViolation(
                str(exc),
                artifact_index=idx,
                claim_id=str(claim.get("claim_id", "unknown")),
                trust_class=str(claim.get("trust_class")),
                violation_type="INVALID_TRUST_CLASS",
            ) from exc

        normalized = dict(claim)
        normalized["trust_class"] = tc.value
        if is_authority_bearing(tc):
            authority.append(normalized)
        else:
            advisory.append(normalized)

    return RoutedClaims(authority_claims=tuple(authority), advisory_claims=tuple(advisory))


def enforce_authority_stream_constraints(reasoning_artifacts: Sequence[Mapping[str, Any]]) -> None:
    """
    Enforce that authority stream contains only FV/MV/PA artifacts.
    """
    for idx, artifact in enumerate(reasoning_artifacts):
        claim_id = str(artifact.get("claim_id", "unknown"))
        if "trust_class" not in artifact:
            raise AuthorityRoutingViolation(
                "Reasoning artifact missing trust_class.",
                artifact_index=idx,
                claim_id=claim_id,
                violation_type="MISSING_TRUST_CLASS",
            )

        tc = parse_trust_class(artifact["trust_class"])
        if tc == TrustClass.ADV:
            raise AuthorityRoutingViolation(
                "ADV artifact cannot enter authority-bearing reasoning stream.",
                artifact_index=idx,
                claim_id=claim_id,
                trust_class=tc.value,
                violation_type="ADV_QUARANTINE",
            )
        if not is_authority_bearing(tc):
            raise AuthorityRoutingViolation(
                "Non-authority trust class in authority stream.",
                artifact_index=idx,
                claim_id=claim_id,
                trust_class=tc.value,
                violation_type="NON_AUTHORITY_STREAM",
            )


def build_authority_reasoning_payloads(
    reasoning_artifacts: Sequence[Mapping[str, Any]],
) -> Tuple[dict[str, Any], ...]:
    """
    Build normalized authority payloads after routing and abstention gates.

    Gates:
    - trust class must be authority-bearing (ADV quarantine)
    - validation_outcome must be present and typed
    """
    enforce_authority_stream_constraints(reasoning_artifacts)

    payloads: list[dict[str, Any]] = []
    for idx, artifact in enumerate(reasoning_artifacts):
        outcome = verify_outcome_present(artifact, artifact_index=idx)
        tc = parse_trust_class(artifact["trust_class"])
        payload = dict(artifact)
        payload["trust_class"] = tc.value
        payload["validation_outcome"] = outcome
        payloads.append(payload)

    return tuple(payloads)


def build_authority_reasoning_leaves(
    reasoning_artifacts: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    """
    Prepare deterministic reasoning leaves for attestation roots.
    """
    payloads = build_authority_reasoning_payloads(reasoning_artifacts)
    return tuple(_canonicalize_json(payload) for payload in payloads)

