#!/usr/bin/env python3
"""
First Organism Determinism Harness
==================================

This module provides deterministic wrappers for the First Organism closed-loop path:

    UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
    Dual-Attest seal H_t → RFL runner metabolism.

Every function in this module is designed to produce byte-for-byte identical output
when called with the same seed. All timestamps, IDs, and serialized artifacts are
derived from content hashes, not wall-clock time or random sources.

Cryptographic Constraints:
    - NO datetime.now, datetime.utcnow, time.time
    - NO uuid.uuid4, random.*, numpy.random without explicit seed
    - NO SQL NOW() or CURRENT_TIMESTAMP
    - All JSON output uses RFC 8785 canonical serialization

Usage:
    from backend.repro.first_organism_harness import (
        deterministic_ui_event,
        deterministic_gate_verdict,
        deterministic_derivation_result,
        deterministic_seal,
        deterministic_rfl_step,
        run_first_organism_deterministic,
    )
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.repro.determinism import (
    DETERMINISTIC_EPOCH,
    deterministic_hash,
    deterministic_isoformat,
    deterministic_run_id,
    deterministic_seed_from_content,
    deterministic_slug,
    deterministic_timestamp,
    deterministic_timestamp_from_content,
    deterministic_uuid,
)
from backend.frontier.curriculum import (
    CurriculumSlice,
    GateEvaluator,
    NormalizedMetrics,
    build_first_organism_metrics,
    make_first_organism_pl2_hard_slice,
    make_first_organism_slice,
)

# ---------------------------------------------------------------------------
# RFC 8785 Canonical JSON Serialization
# ---------------------------------------------------------------------------


def rfc8785_canonicalize(obj: Any) -> str:
    """
    Serialize object to RFC 8785 canonical JSON.

    This ensures:
        - Keys are sorted lexicographically
        - No unnecessary whitespace
        - ASCII-only output (non-ASCII escaped)
        - Consistent float representation

    Args:
        obj: JSON-serializable object

    Returns:
        Canonical JSON string
    """
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def rfc8785_hash(obj: Any) -> str:
    """Compute SHA-256 hash of canonical JSON representation."""
    canonical = rfc8785_canonicalize(obj)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic UI Event Wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicUIEvent:
    """Deterministic UI event with content-derived timestamp and ID."""

    event_id: str
    timestamp: str
    canonical_json: str
    leaf_hash: str
    payload: Dict[str, Any]


def deterministic_ui_event(
    seed: int,
    payload: Mapping[str, Any],
    *,
    event_type: str = "select_statement",
) -> DeterministicUIEvent:
    """
    Create a deterministic UI event from seed and payload.

    The event_id and timestamp are derived from the payload content and seed,
    ensuring identical inputs always produce identical outputs.

    Args:
        seed: Deterministic seed for timestamp/ID derivation
        payload: UI event payload (action, statement_hash, etc.)
        event_type: Event type string (default: "select_statement")

    Returns:
        DeterministicUIEvent with all fields derived deterministically
    """
    # Canonicalize payload for hashing
    payload_dict = dict(payload)
    payload_dict["event_type"] = event_type
    canonical_payload = rfc8785_canonicalize(payload_dict)

    # Derive event_id from content
    event_id = deterministic_run_id(
        "ui-event",
        seed,
        canonical_payload,
        length=16,
    )

    # Derive timestamp from content
    timestamp = deterministic_isoformat(seed, canonical_payload, resolution="seconds")

    # Compute leaf hash for Merkle inclusion
    leaf_hash = hashlib.sha256(canonical_payload.encode("ascii")).hexdigest()

    # Build final canonical representation
    final_doc = {
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "payload": payload_dict,
    }
    canonical_json = rfc8785_canonicalize(final_doc)

    return DeterministicUIEvent(
        event_id=event_id,
        timestamp=timestamp,
        canonical_json=canonical_json,
        leaf_hash=leaf_hash,
        payload=final_doc,
    )


# ---------------------------------------------------------------------------
# Deterministic Curriculum Gate Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicGateVerdict:
    """Deterministic gate verdict with content-derived audit trail."""

    advance: bool
    reason: str
    audit_json: str
    audit_hash: str
    timestamp: str
    gate_statuses: Tuple[Dict[str, Any], ...]


def deterministic_gate_verdict(
    seed: int,
    slice_name: str,
    metrics: Mapping[str, Any],
    gate_statuses: Sequence[Mapping[str, Any]],
    *,
    advance: bool,
    reason: str,
) -> DeterministicGateVerdict:
    """
    Create a deterministic gate verdict.

    Args:
        seed: Deterministic seed
        slice_name: Curriculum slice name
        metrics: Metrics payload
        gate_statuses: List of gate status dicts
        advance: Whether the gate allows advancement
        reason: Human-readable reason

    Returns:
        DeterministicGateVerdict with all fields derived deterministically
    """
    # Canonicalize inputs
    metrics_canonical = rfc8785_canonicalize(dict(metrics))
    statuses_canonical = rfc8785_canonicalize([dict(s) for s in gate_statuses])

    # Derive timestamp from content
    timestamp = deterministic_isoformat(
        seed, slice_name, metrics_canonical, resolution="seconds"
    )

    # Build audit document
    audit_doc = {
        "version": 2,
        "system": "first-organism",
        "active_slice": slice_name,
        "timestamp": timestamp,
        "advance": advance,
        "reason": reason,
        "gates": [dict(s) for s in gate_statuses],
    }
    audit_json = rfc8785_canonicalize(audit_doc)
    audit_hash = hashlib.sha256(audit_json.encode("ascii")).hexdigest()

    return DeterministicGateVerdict(
        advance=advance,
        reason=reason,
        audit_json=audit_json,
        audit_hash=audit_hash,
        timestamp=timestamp,
        gate_statuses=tuple(dict(s) for s in gate_statuses),
    )


# ---------------------------------------------------------------------------
# Deterministic Derivation Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDerivationResult:
    """Deterministic derivation result with content-derived IDs."""

    run_id: str
    status: str
    n_candidates: int
    n_abstained: int
    abstained_hashes: Tuple[str, ...]
    canonical_json: str
    result_hash: str


def deterministic_derivation_result(
    seed: int,
    slice_name: str,
    *,
    status: str,
    n_candidates: int,
    n_abstained: int,
    abstained_hashes: Sequence[str],
) -> DeterministicDerivationResult:
    """
    Create a deterministic derivation result.

    Args:
        seed: Deterministic seed
        slice_name: Curriculum slice name
        status: Derivation status ("success", "abstain", "failure")
        n_candidates: Number of candidates considered
        n_abstained: Number of abstentions
        abstained_hashes: Hashes of abstained statements

    Returns:
        DeterministicDerivationResult with all fields derived deterministically
    """
    # Sort abstained hashes for determinism
    sorted_hashes = tuple(sorted(abstained_hashes))

    # Derive run_id from content
    run_id = deterministic_run_id(
        "derive",
        seed,
        slice_name,
        status,
        n_candidates,
        n_abstained,
        rfc8785_canonicalize(sorted_hashes),
        length=12,
    )

    # Build result document
    result_doc = {
        "run_id": run_id,
        "slice_name": slice_name,
        "status": status,
        "n_candidates": n_candidates,
        "n_abstained": n_abstained,
        "abstained_hashes": sorted_hashes,
    }
    canonical_json = rfc8785_canonicalize(result_doc)
    result_hash = hashlib.sha256(canonical_json.encode("ascii")).hexdigest()

    return DeterministicDerivationResult(
        run_id=run_id,
        status=status,
        n_candidates=n_candidates,
        n_abstained=n_abstained,
        abstained_hashes=sorted_hashes,
        canonical_json=canonical_json,
        result_hash=result_hash,
    )


# ---------------------------------------------------------------------------
# Deterministic Seal (Dual-Root Attestation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicSealResult:
    """Deterministic seal result with dual-root attestation."""

    block_id: str
    reasoning_root: str  # R_t
    ui_root: str  # U_t
    composite_root: str  # H_t = SHA256(R_t || U_t)
    sealed_at: str
    attestation_json: str
    attestation_hash: str


def deterministic_seal(
    seed: int,
    derivation_result: DeterministicDerivationResult,
    ui_events: Sequence[DeterministicUIEvent],
    *,
    sealed_by: str = "first-organism-harness",
) -> DeterministicSealResult:
    """
    Create a deterministic dual-root seal.

    Computes:
        - R_t: Merkle root over reasoning artifacts (proof hashes)
        - U_t: Merkle root over UI event leaf hashes
        - H_t: SHA256(R_t || U_t) composite root

    Args:
        seed: Deterministic seed
        derivation_result: Derivation result with abstained hashes
        ui_events: Sequence of deterministic UI events
        sealed_by: Sealer identifier

    Returns:
        DeterministicSealResult with R_t, U_t, H_t computed deterministically
    """
    # Compute R_t from reasoning artifacts (abstained hashes)
    reasoning_leaves = sorted(derivation_result.abstained_hashes)
    if reasoning_leaves:
        reasoning_concat = "".join(reasoning_leaves)
        reasoning_root = hashlib.sha256(reasoning_concat.encode("ascii")).hexdigest()
    else:
        reasoning_root = hashlib.sha256(b"REASONING:EMPTY").hexdigest()

    # Compute U_t from UI event leaf hashes
    ui_leaves = sorted(ev.leaf_hash for ev in ui_events)
    if ui_leaves:
        ui_concat = "".join(ui_leaves)
        ui_root = hashlib.sha256(ui_concat.encode("ascii")).hexdigest()
    else:
        ui_root = hashlib.sha256(b"UI:EMPTY").hexdigest()

    # Compute H_t = SHA256(R_t || U_t)
    composite_data = f"{reasoning_root}{ui_root}".encode("ascii")
    composite_root = hashlib.sha256(composite_data).hexdigest()

    # Derive block_id from composite root
    block_id = deterministic_run_id("block", seed, composite_root, length=16)

    # Derive sealed_at timestamp from content
    sealed_at = deterministic_isoformat(seed, composite_root, resolution="seconds")

    # Build attestation document
    attestation_doc = {
        "block_id": block_id,
        "reasoning_merkle_root": reasoning_root,
        "ui_merkle_root": ui_root,
        "composite_attestation_root": composite_root,
        "sealed_at": sealed_at,
        "sealed_by": sealed_by,
        "reasoning_event_count": len(reasoning_leaves),
        "ui_event_count": len(ui_leaves),
        "attestation_version": "v2",
        "algorithm": "SHA256",
        "composite_formula": "SHA256(R_t || U_t)",
        "derivation_run_id": derivation_result.run_id,
        "derivation_status": derivation_result.status,
    }
    attestation_json = rfc8785_canonicalize(attestation_doc)
    attestation_hash = hashlib.sha256(attestation_json.encode("ascii")).hexdigest()

    return DeterministicSealResult(
        block_id=block_id,
        reasoning_root=reasoning_root,
        ui_root=ui_root,
        composite_root=composite_root,
        sealed_at=sealed_at,
        attestation_json=attestation_json,
        attestation_hash=attestation_hash,
    )


# ---------------------------------------------------------------------------
# Deterministic RFL Step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicRflStep:
    """Deterministic RFL metabolism step."""

    step_id: str
    policy_update_applied: bool
    abstention_mass_delta: float
    symbolic_descent: float
    ledger_entry_json: str
    ledger_entry_hash: str


def deterministic_rfl_step(
    seed: int,
    seal_result: DeterministicSealResult,
    derivation_result: DeterministicDerivationResult,
    *,
    slice_name: str,
    policy_id: str = "default",
    abstention_tolerance: float = 0.15,
) -> DeterministicRflStep:
    """
    Execute a deterministic RFL metabolism step.

    Consumes the sealed attestation (H_t, R_t, U_t) and produces a policy
    ledger entry with symbolic descent computed from abstention metrics.

    Args:
        seed: Deterministic seed
        seal_result: Sealed attestation with dual roots
        derivation_result: Derivation result with abstention metrics
        slice_name: Curriculum slice name
        policy_id: Policy identifier
        abstention_tolerance: Abstention tolerance threshold

    Returns:
        DeterministicRflStep with policy update and symbolic descent
    """
    # Derive step_id from content
    step_material = f"{seed}|{slice_name}|{policy_id}|{seal_result.composite_root}"
    step_id = hashlib.sha256(step_material.encode("utf-8")).hexdigest()

    # Compute abstention metrics
    n_total = max(derivation_result.n_candidates, 1)
    abstention_rate = derivation_result.n_abstained / n_total
    abstention_mass = float(derivation_result.n_abstained)

    # Compute expected mass and delta
    expected_mass = abstention_tolerance * n_total
    abstention_mass_delta = abstention_mass - expected_mass
    abstention_rate_delta = abstention_rate - abstention_tolerance

    # Determine if policy update is needed
    policy_update_applied = (
        abs(abstention_mass_delta) > 1e-9 or abs(abstention_rate_delta) > 1e-9
    )

    # Compute reward and symbolic descent
    reward = max(0.0, 1.0 - max(abstention_rate, 0.0))
    symbolic_descent = -abstention_rate_delta

    # Build ledger entry
    ledger_entry = {
        "step_id": step_id,
        "slice_name": slice_name,
        "status": derivation_result.status,
        "coverage_rate": 0.0,
        "novelty_rate": 0.0,
        "throughput": 0.0,
        "success_rate": 0.0,
        "abstention_fraction": abstention_rate,
        "policy_reward": reward,
        "symbolic_descent": symbolic_descent,
        "budget_spent": n_total,
        "derive_steps": 1,
        "max_breadth": n_total,
        "max_total": n_total,
        "abstention_breakdown": {"lean_abstain": derivation_result.n_abstained},
        "composite_root": seal_result.composite_root,
        "reasoning_root": seal_result.reasoning_root,
        "ui_root": seal_result.ui_root,
        "policy_update_applied": policy_update_applied,
    }
    ledger_entry_json = rfc8785_canonicalize(ledger_entry)
    ledger_entry_hash = hashlib.sha256(ledger_entry_json.encode("ascii")).hexdigest()

    return DeterministicRflStep(
        step_id=step_id,
        policy_update_applied=policy_update_applied,
        abstention_mass_delta=abstention_mass_delta,
        symbolic_descent=symbolic_descent,
        ledger_entry_json=ledger_entry_json,
        ledger_entry_hash=ledger_entry_hash,
    )


# ---------------------------------------------------------------------------
# Full First Organism Deterministic Run
# ---------------------------------------------------------------------------


@dataclass
class FirstOrganismResult:
    """Complete result of a deterministic First Organism run."""

    seed: int
    ui_event: DeterministicUIEvent
    gate_verdict: DeterministicGateVerdict
    derivation_result: DeterministicDerivationResult
    seal_result: DeterministicSealResult
    rfl_step: DeterministicRflStep

    # Composite hashes for verification
    composite_root: str  # H_t
    run_hash: str  # Hash of entire run for reproducibility check

    def to_canonical_json(self) -> str:
        """Serialize entire result to canonical JSON."""
        doc = {
            "seed": self.seed,
            "ui_event": self.ui_event.payload,
            "gate_verdict": {
                "advance": self.gate_verdict.advance,
                "reason": self.gate_verdict.reason,
                "audit_hash": self.gate_verdict.audit_hash,
            },
            "derivation_result": {
                "run_id": self.derivation_result.run_id,
                "status": self.derivation_result.status,
                "result_hash": self.derivation_result.result_hash,
            },
            "seal_result": {
                "block_id": self.seal_result.block_id,
                "reasoning_root": self.seal_result.reasoning_root,
                "ui_root": self.seal_result.ui_root,
                "composite_root": self.seal_result.composite_root,
                "attestation_hash": self.seal_result.attestation_hash,
            },
            "rfl_step": {
                "step_id": self.rfl_step.step_id,
                "policy_update_applied": self.rfl_step.policy_update_applied,
                "symbolic_descent": self.rfl_step.symbolic_descent,
                "ledger_entry_hash": self.rfl_step.ledger_entry_hash,
            },
            "composite_root": self.composite_root,
            "run_hash": self.run_hash,
        }
        return rfc8785_canonicalize(doc)


def _slice_for_name(slice_name: str) -> CurriculumSlice:
    if slice_name == "first-organism-pl":
        return make_first_organism_slice()
    if slice_name == "first_organism_pl2_hard":
        return make_first_organism_pl2_hard_slice()
    raise ValueError(f"Unknown curriculum slice '{slice_name}'")


def _default_metrics_for_slice(slice_name: str, seed: int) -> Dict[str, Any]:
    default_attestation_hash = deterministic_slug(seed, "attn")
    if slice_name == "first_organism_pl2_hard":
        return build_first_organism_metrics(
            coverage_ci=0.92,
            sample_size=26,
            abstention_rate=25.0,
            attempt_mass=3200,
            proof_velocity_pph=150.0,
            velocity_cv=0.10,
            runtime_minutes=28.0,
            backlog_fraction=0.35,
            attestation_hash=deterministic_slug(seed, "attn-hard"),
        )
    return build_first_organism_metrics(
        coverage_ci=0.90,
        sample_size=22,
        abstention_rate=13.5,
        attempt_mass=3200,
        proof_velocity_pph=190.0,
        velocity_cv=0.06,
        runtime_minutes=28.0,
        backlog_fraction=0.31,
        attestation_hash=default_attestation_hash,
    )


def run_first_organism_deterministic(
    seed: int,
    *,
    slice_name: str = "first-organism-pl",
    ui_payload: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    n_candidates: int = 10,
    n_abstained: int = 3,
    abstained_hashes: Optional[Sequence[str]] = None,
) -> FirstOrganismResult:
    """
    Execute a complete deterministic First Organism closed loop.

    This function exercises the full path:
        UI Event → Curriculum Gate → Derivation → Lean Verify (abstention) →
        Dual-Attest seal H_t → RFL runner metabolism.

    All outputs are derived deterministically from the seed and inputs.

    Args:
        seed: Master deterministic seed
        slice_name: Curriculum slice name
        ui_payload: UI event payload (defaults to synthetic)
        metrics: Curriculum metrics (defaults to passing thresholds)
        n_candidates: Number of derivation candidates
        n_abstained: Number of abstentions
        abstained_hashes: Hashes of abstained statements (generated if not provided)

    Returns:
        FirstOrganismResult with all intermediate and final artifacts
    """
    # Generate default UI payload if not provided
    if ui_payload is None:
        ui_payload = {
            "action": "toggle_abstain",
            "statement_hash": deterministic_slug(seed, "statement", length=64),
        }

    # Create deterministic UI event
    ui_event = deterministic_ui_event(seed, ui_payload)

    if metrics is None:
        metrics = _default_metrics_for_slice(slice_name, seed)
    else:
        metrics = dict(metrics)

    slice_cfg = _slice_for_name(slice_name)
    normalized_metrics = NormalizedMetrics.from_raw(metrics)
    gate_status_objs = GateEvaluator(normalized_metrics, slice_cfg).evaluate()
    failed_status = next((status for status in gate_status_objs if not status.passed), None)
    if failed_status:
        advance = False
        reason = f"{failed_status.gate} gate: {failed_status.message}"
    else:
        advance = True
        reason = "; ".join(status.message for status in gate_status_objs)

    gate_statuses = [status.to_dict() for status in gate_status_objs]
    gate_verdict = deterministic_gate_verdict(
        seed,
        slice_name,
        metrics,
        gate_statuses,
        advance=advance,
        reason=reason,
    )

    # Generate abstained hashes if not provided
    if abstained_hashes is None:
        abstained_hashes = [
            deterministic_slug(seed, "abstain", i, length=64)
            for i in range(n_abstained)
        ]

    # Create derivation result
    derivation_result = deterministic_derivation_result(
        seed,
        slice_name,
        status="abstain" if n_abstained > 0 else "success",
        n_candidates=n_candidates,
        n_abstained=n_abstained,
        abstained_hashes=abstained_hashes,
    )

    # Create dual-root seal
    seal_result = deterministic_seal(
        seed,
        derivation_result,
        [ui_event],
    )

    # Execute RFL step
    rfl_step = deterministic_rfl_step(
        seed,
        seal_result,
        derivation_result,
        slice_name=slice_name,
    )

    # Compute run hash for reproducibility verification
    run_doc = {
        "seed": seed,
        "ui_event_hash": ui_event.leaf_hash,
        "gate_audit_hash": gate_verdict.audit_hash,
        "derivation_hash": derivation_result.result_hash,
        "attestation_hash": seal_result.attestation_hash,
        "ledger_entry_hash": rfl_step.ledger_entry_hash,
        "composite_root": seal_result.composite_root,
    }
    run_hash = hashlib.sha256(
        rfc8785_canonicalize(run_doc).encode("ascii")
    ).hexdigest()

    return FirstOrganismResult(
        seed=seed,
        ui_event=ui_event,
        gate_verdict=gate_verdict,
        derivation_result=derivation_result,
        seal_result=seal_result,
        rfl_step=rfl_step,
        composite_root=seal_result.composite_root,
        run_hash=run_hash,
    )


def verify_determinism(seed: int, runs: int = 3) -> bool:
    """
    Verify that run_first_organism_deterministic produces identical output.

    Args:
        seed: Seed to test
        runs: Number of runs to compare

    Returns:
        True if all runs produce identical run_hash
    """
    results = [run_first_organism_deterministic(seed) for _ in range(runs)]
    first_hash = results[0].run_hash
    return all(r.run_hash == first_hash for r in results)


__all__ = [
    "DeterministicUIEvent",
    "DeterministicGateVerdict",
    "DeterministicDerivationResult",
    "DeterministicSealResult",
    "DeterministicRflStep",
    "FirstOrganismResult",
    "deterministic_ui_event",
    "deterministic_gate_verdict",
    "deterministic_derivation_result",
    "deterministic_seal",
    "deterministic_rfl_step",
    "run_first_organism_deterministic",
    "verify_determinism",
    "rfc8785_canonicalize",
    "rfc8785_hash",
]

