"""Wave A7 experiment runner: deterministic three-lane orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from mathledger.evidence import assert_evidence_pack_passes, build_evidence_pack
from mathledger.governance import TrustClass, validate_mv_claim
from mathledger.integration.aak_bridge import (
    AAKBridgeViolation,
    build_aak_bridge_packet,
    validate_lane_b_reference,
)
from mathledger.integration.cpf_adapter import (
    CPFAdapterViolation,
    normalize_cpf_assessments,
)
from mathledger.uvil import (
    DraftProposal,
    EditedClaim,
    commit_from_draft,
    derive_content_id,
    snapshot_to_dict,
)


EXPERIMENT_SCHEMA_VERSION = "v1"


class ExperimentRunnerViolation(ValueError):
    """Raised when experiment pipeline violates lane separation contracts."""

    ERROR_CODE = "EXPERIMENT_RUNNER_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.details = dict(details or {})

    def to_error_response(self) -> dict[str, Any]:
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "field": self.field,
            "details": self.details,
        }


def _canonicalize_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonicalize_json(data).encode("utf-8")).hexdigest()


def _build_reasoning_artifacts(claims: Sequence[Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for claim in claims:
        trust_class = claim.trust_class
        if trust_class == TrustClass.MV:
            mv = validate_mv_claim(claim.claim_text)
            outcome = mv.outcome.value
            proof_payload = {
                "route": "MV",
                "claim_text": claim.claim_text,
                "explanation": mv.explanation,
            }
        elif trust_class == TrustClass.FV:
            outcome = "ABSTAINED"
            proof_payload = {
                "route": "FV",
                "claim_text": claim.claim_text,
                "note": "No FV verifier in this wave.",
            }
        elif trust_class == TrustClass.PA:
            outcome = "ABSTAINED"
            proof_payload = {
                "route": "PA",
                "claim_text": claim.claim_text,
                "note": "PA is recorded as authority route without mechanical validator.",
            }
        else:
            outcome = "ABSTAINED"
            proof_payload = {
                "route": "ADV",
                "claim_text": claim.claim_text,
                "note": "Advisory artifacts are retained in bundle and excluded from R_t.",
            }

        artifact_payload = {
            "claim_id": claim.claim_id,
            "trust_class": trust_class.value,
            "validation_outcome": outcome,
            "proof_payload": proof_payload,
        }
        artifact_id = derive_content_hash_for_artifact(artifact_payload)
        artifacts.append({"artifact_id": artifact_id, **artifact_payload})

    artifacts.sort(key=lambda item: item["claim_id"])
    return artifacts


def derive_content_hash_for_artifact(payload: Mapping[str, Any]) -> str:
    """Deterministic helper for artifact identity."""
    return derive_content_id(dict(payload))


def _build_elevated_indicators(cpf_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    elevated: list[dict[str, Any]] = []
    for item in cpf_snapshot.get("top_indicators", []):
        score = float(item.get("bayesian_score", 0.0))
        elevated.append(
            {
                "indicator_id": str(item.get("indicator_id", "")),
                "activation_level": round(score * 100.0, 6),
            }
        )
    return elevated


def run_three_lane_experiment(
    *,
    edited_claims: Sequence[EditedClaim | Mapping[str, Any]],
    cpf_assessments: Sequence[Mapping[str, Any]],
    commit_epoch: int,
    proposal_id: str = "draft-proposal",
    user_fingerprint: str = "anonymous",
    cpf_snapshot_epoch: int | None = None,
    psych_source: str = "captured",
    psych_capture_mode: str = "hash_ref",
) -> dict[str, Any]:
    """
    Execute deterministic three-lane experiment pipeline.

    Lane A: MathLedger authority commit + evidence replay.
    Lane B: AAK bridge packet referencing Lane-A roots only.
    Lane C: CPF normalized snapshot for experimentation context.
    """
    if commit_epoch < 0:
        raise ExperimentRunnerViolation("commit_epoch must be non-negative.", field="commit_epoch")

    draft = DraftProposal(proposal_id=str(proposal_id), claims=[])
    commit_result = commit_from_draft(
        draft,
        edited_claims,
        commit_epoch=commit_epoch,
        user_fingerprint=user_fingerprint,
    )

    reasoning_artifacts = _build_reasoning_artifacts(commit_result.snapshot.claims)
    evidence_pack = build_evidence_pack(
        committed_partition_snapshot=commit_result.snapshot,
        uvil_events=[commit_result.event],
        reasoning_artifacts=reasoning_artifacts,
    )
    lane_a_verification = assert_evidence_pack_passes(evidence_pack)

    cpf_epoch = commit_epoch if cpf_snapshot_epoch is None else cpf_snapshot_epoch
    cpf_snapshot = normalize_cpf_assessments(
        cpf_assessments,
        snapshot_epoch=cpf_epoch,
    )

    categories = sorted(
        {
            str(item.get("category", "")).strip()
            for item in cpf_snapshot.get("indicators", [])
            if str(item.get("category", "")).strip()
        }
    )
    bridge_packet = build_aak_bridge_packet(
        evidence_pack=evidence_pack,
        source=psych_source,
        capture_mode=psych_capture_mode,
        snapshot_id=cpf_snapshot["snapshot_id"],
        convergence_score=cpf_snapshot["aggregate"]["max_bayesian_score"],
        elevated_categories=categories,
        elevated_indicators=_build_elevated_indicators(cpf_snapshot),
        psych_root_hash=cpf_snapshot["snapshot_id"],
    )

    payload_without_id: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "lane_a": {
            "committed_partition_snapshot": snapshot_to_dict(commit_result.snapshot),
            "evidence_pack": evidence_pack,
            "replay_verification": lane_a_verification,
        },
        "lane_b": {"aak_bridge_packet": bridge_packet},
        "lane_c": {"cpf_snapshot": cpf_snapshot},
        "metadata": {
            "commit_epoch": commit_epoch,
            "proposal_id": str(proposal_id),
            "user_fingerprint": str(user_fingerprint),
        },
    }

    run_id = _content_hash(payload_without_id)
    run_result = {"run_id": run_id, **payload_without_id}
    assert_lane_boundaries(run_result)
    return run_result


def assert_lane_boundaries(run_result: Mapping[str, Any]) -> None:
    """
    Enforce Lane-A/Lane-B/Lane-C boundary invariants for runner output.
    """
    try:
        lane_a = run_result["lane_a"]
        lane_b = run_result["lane_b"]
        lane_c = run_result["lane_c"]
    except KeyError as exc:
        raise ExperimentRunnerViolation(
            "run_result missing required lane keys.",
            field=str(exc),
        ) from exc

    if not isinstance(lane_a, Mapping) or not isinstance(lane_b, Mapping) or not isinstance(lane_c, Mapping):
        raise ExperimentRunnerViolation("lane payloads must be mappings.", field="run_result")

    evidence_pack = lane_a.get("evidence_pack")
    if not isinstance(evidence_pack, Mapping):
        raise ExperimentRunnerViolation("lane_a.evidence_pack must be a mapping.", field="lane_a")
    verification = assert_evidence_pack_passes(evidence_pack)
    roots = verification["recorded"]

    bridge_packet = lane_b.get("aak_bridge_packet")
    if not isinstance(bridge_packet, Mapping):
        raise ExperimentRunnerViolation(
            "lane_b.aak_bridge_packet must be a mapping.",
            field="lane_b",
        )
    psych_context = bridge_packet.get("psych_context")
    if not isinstance(psych_context, Mapping):
        raise ExperimentRunnerViolation(
            "lane_b psych_context missing or invalid.",
            field="lane_b.psych_context",
        )
    validate_lane_b_reference(psych_context, evidence_pack=evidence_pack)

    refs = bridge_packet.get("mathledger_refs")
    if not isinstance(refs, Mapping):
        raise ExperimentRunnerViolation(
            "lane_b.mathledger_refs must be present.",
            field="lane_b.mathledger_refs",
        )
    if refs.get("u_t") != roots["u_t"] or refs.get("r_t") != roots["r_t"] or refs.get("h_t") != roots["h_t"]:
        raise ExperimentRunnerViolation(
            "Lane-B MathLedger root references diverge from Lane-A evidence.",
            field="lane_b.mathledger_refs",
        )

    cpf_snapshot = lane_c.get("cpf_snapshot")
    if not isinstance(cpf_snapshot, Mapping):
        raise ExperimentRunnerViolation("lane_c.cpf_snapshot must be a mapping.", field="lane_c")

    try:
        snapshot_epoch = int(cpf_snapshot["snapshot_epoch"])
    except Exception as exc:
        raise ExperimentRunnerViolation(
            "lane_c.cpf_snapshot.snapshot_epoch must be an integer.",
            field="lane_c.cpf_snapshot.snapshot_epoch",
        ) from exc
    recomputed = normalize_cpf_assessments(
        cpf_snapshot.get("indicators", []),
        snapshot_epoch=snapshot_epoch,
    )
    if recomputed["snapshot_id"] != cpf_snapshot.get("snapshot_id"):
        raise ExperimentRunnerViolation(
            "Lane-C snapshot_id mismatch under deterministic recomputation.",
            field="lane_c.cpf_snapshot.snapshot_id",
        )


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentRunnerViolation",
    "run_three_lane_experiment",
    "assert_lane_boundaries",
]

