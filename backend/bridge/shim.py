from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Sequence, Tuple

from derivation.pipeline import DerivationResult, run_slice_for_test
from ledger.ingest import LedgerIngestor
from backend.bridge.context import AttestedRunContext
from curriculum.gates import CurriculumSlice
from backend.repro.determinism import (
    deterministic_isoformat,
    deterministic_seed_from_content,
    deterministic_timestamp,
    deterministic_unix_timestamp,
)


_SENTINEL_KEYS = ("breadth_max", "max_breadth", "breadth_cap")
_TOTAL_KEYS = ("total_max", "max_total", "total_cap")


def _extract_slice_param(params: Dict[str, Any], keys: Sequence[str], fallback: int) -> int:
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _build_proof_payload(stmt: "StatementRecord", status: str) -> Tuple[str, str, str, str, int, bool]:
    rule = stmt.rule or "formula"
    method = stmt.verification_method or "unknown"
    details = f"{status}|{rule}|{method}|parents={stmt.parents}"
    return (
        stmt.normalized,
        details,
        status,
        rule,
        stmt.mp_depth,
        stmt.is_axiom,
    )


def _collect_proofs(result: DerivationResult) -> Sequence[Tuple[str, str, str, str, int, bool]]:
    seen_hashes: set[str] = set()
    proofs: list[Tuple[str, str, str, str, int, bool]] = []

    def _append(stmt, status: str) -> None:
        if stmt.hash in seen_hashes:
            return
        seen_hashes.add(stmt.hash)
        proofs.append(_build_proof_payload(stmt, status))

    for statement in result.statements:
        _append(statement, "success")
    for abstained in result.abstained_candidates:
        _append(abstained, "abstain")

    return proofs


def _representative_statement(result: DerivationResult):
    if result.statements:
        return result.statements[-1]
    if result.abstained_candidates:
        return result.abstained_candidates[-1]
    raise ValueError("Derivation produced no statements or abstentions to attest")


def run_first_organism_cycle(test_config: Dict[str, Any]) -> AttestedRunContext:
    """
    Orchestrate a single First Organism tick from derivation to RFL-ready context.

    Args:
        test_config: {
            "slice_cfg": CurriculumSlice,
            "cursor": psycopg.Cursor,
            "limit": Optional[int],
            "policy_id": Optional[str],
            "prover": Optional[str],
            "ui_events": Optional[Sequence[str]],
            "sealed_by": Optional[str],
            "module_name": Optional[str],
            "metadata": Optional[Dict[str, Any]],
            "existing": Optional[Sequence[StatementRecord]],
        }
    """
    slice_cfg: CurriculumSlice = test_config["slice_cfg"]
    cursor = test_config["cursor"]
    limit = max(1, int(test_config.get("limit", 1)))
    existing = test_config.get("existing")
    policy_id = test_config.get("policy_id")
    prover = test_config.get("prover", "axiom_engine")
    module_name = test_config.get("module_name", "pipeline")
    sealed_by = test_config.get("sealed_by", "first_organism_cycle")
    ui_events = tuple(str(ev) for ev in test_config.get("ui_events", ()))
    metadata_overrides = dict(test_config.get("metadata", {}))

    # DETERMINISM: Derive timestamps from slice content, not wall-clock
    content_seed = deterministic_seed_from_content(slice_cfg.name, limit, prover, module_name)
    start_ts = deterministic_timestamp(content_seed)
    
    derivation_result: DerivationResult = run_slice_for_test(
        slice_cfg,
        limit=limit,
        existing=existing,
    )
    
    # Deterministic duration based on work done (1 second per candidate considered)
    simulated_duration = max(1.0, float(derivation_result.stats.candidates_considered))
    end_ts = deterministic_timestamp(content_seed + int(simulated_duration))

    proof_payloads = _collect_proofs(derivation_result)
    if not proof_payloads:
        raise ValueError("Derivation produced no proofs (success or abstain) to attest")

    ingestor = LedgerIngestor()
    block = ingestor.ingest_batch(
        cur=cursor,
        theory_name=slice_cfg.name,
        statements=proof_payloads,
        prover=prover,
        module_name=module_name,
        sealed_by=sealed_by,
        ui_events=ui_events,
    )

    representative = _representative_statement(derivation_result)
    breakdown = Counter(
        (stmt.verification_method or stmt.rule or "unknown") for stmt in derivation_result.abstained_candidates
    )
    attempt_mass = max(float(derivation_result.stats.candidates_considered), 1.0)

    # DETERMINISM: All timestamps derived from content, not wall-clock
    metadata = {
        **metadata_overrides,
        "block_number": block.number,
        "block_hash": block.block_hash,
        "derivation_status": derivation_result.status,
        "first_organism_duration_seconds": simulated_duration,
        "first_organism_abstentions": len(derivation_result.abstained_candidates),
        "first_organism_started_at": start_ts.isoformat(),
        "first_organism_ended_at": end_ts.isoformat(),
        "timestamp": float(deterministic_unix_timestamp(content_seed)),
        "abstention_breakdown": dict(breakdown),
        "abstention_rate": derivation_result.abstention_metrics.get("rate", 0.0),
        "abstention_mass": derivation_result.abstention_metrics.get("mass", 0.0),
        "attempt_mass": attempt_mass,
        "derive_steps": limit,
        "max_breadth": _extract_slice_param(slice_cfg.params, _SENTINEL_KEYS, 1),
        "max_total": _extract_slice_param(slice_cfg.params, _TOTAL_KEYS, 1),
    }

    return AttestedRunContext(
        slice_id=slice_cfg.name,
        statement_hash=representative.hash,
        proof_status=derivation_result.status,
        block_id=block.id,
        composite_root=block.composite_root,
        reasoning_root=block.reasoning_root,
        ui_root=block.ui_root,
        abstention_metrics={
            **derivation_result.abstention_metrics,
            "reasons": dict(breakdown),
        },
        policy_id=policy_id,
        metadata=metadata,
    )
