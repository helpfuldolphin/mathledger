# -*- coding: utf-8 -*-
"""
Block sealing functionality for proof persistence.
Uses centralized crypto module with domain-separated Merkle trees.

Extended with dual-root attestation support (Mirror Auditor integration).

Reference: MathLedger Whitepaper §4.2 (Dual Attestation Block Sealing).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from substrate.crypto.hashing import sha256_hex
from substrate.repro.determinism import deterministic_unix_timestamp
from attestation.dual_root import (
    AttestationTree,
    build_reasoning_attestation,
    build_ui_attestation,
    compute_composite_root,
    generate_attestation_metadata,
)
from ledger.ui_events import materialize_ui_artifacts, consume_ui_artifacts


def _derive_block_identity(
    system: str,
    r_t: str,
    u_t: str,
    proof_count: int,
    ui_event_count: int,
) -> Tuple[str, int, int]:
    """
    Deterministically derive block hash, block number, and sealing timestamp.

    The derivation is content-addressed: identical inputs always yield identical
    outputs, and distinct inputs collide only cryptographically.
    """
    material = f"{system}|{proof_count}|{ui_event_count}|{r_t}|{u_t}"
    block_hash = sha256_hex(material)

    # Use distinct non-overlapping segments for number + timestamp seeds
    block_number = int(block_hash[:12], 16) + 1  # ensure block_number >= 1
    timestamp_seed = int(block_hash[12:24], 16)
    sealed_at = int(deterministic_unix_timestamp(timestamp_seed))
    return block_hash, block_number, sealed_at


def _tree_metadata(tree: AttestationTree) -> List[Dict[str, Any]]:
    return [leaf.to_metadata() for leaf in tree.leaves]


def seal_block(system: str, proofs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Seal a block of proofs with domain-separated Merkle root.

    Args:
        system: System identifier (e.g., "pl")
        proofs: Iterable of proof dictionaries to be sealed

    Returns:
        Dictionary containing block metadata:
        - block_number: Deterministic content-addressed block number
        - block_hash: Hash binding system, proofs, and roots
        - merkle_root: Reasoning Merkle root (alias of reasoning_merkle_root)
        - proof_count: Number of proofs in block
        - sealed_at: Unix timestamp of sealing
        - attestation_metadata: Canonical reasoning attestation bundle

    Security:
        Uses centralized crypto module with proper domain separation to prevent
        second preimage attacks (CVE-2012-2459).
    """
    proofs_sequence: Sequence[Dict[str, Any]] = list(proofs)
    reasoning_tree = build_reasoning_attestation(proofs_sequence)
    ui_tree = build_ui_attestation(())  # empty UI stream for legacy seal_block()

    r_t = reasoning_tree.root
    u_t = ui_tree.root
    h_t = compute_composite_root(r_t, u_t)
    block_hash, block_number, sealed_at = _derive_block_identity(
        system,
        r_t,
        u_t,
        len(proofs_sequence),
        0,
    )

    metadata = generate_attestation_metadata(
        r_t=r_t,
        u_t=u_t,
        h_t=h_t,
        reasoning_event_count=len(proofs_sequence),
        ui_event_count=0,
        reasoning_leaves=reasoning_tree.leaves,
        ui_leaves=ui_tree.leaves,
        extra={
            "system": system,
            "block_hash": block_hash,
            "block_number": block_number,
            "sealed_at": sealed_at,
        },
    )

    return {
        "block_number": block_number,
        "block_hash": block_hash,
        "merkle_root": r_t,
        "proof_count": len(proofs_sequence),
        "sealed_at": sealed_at,
        "attestation_metadata": metadata,
        "reasoning_leaves": _tree_metadata(reasoning_tree),
        "ui_event_count": 0,
        "ui_leaves": _tree_metadata(ui_tree),
    }


def seal_block_with_dual_roots(
    system: str,
    proofs: Sequence[Dict[str, Any]],
    ui_events: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Seal a block with dual-root attestation (R_t ↔ U_t → H_t).

    This function extends seal_block() with Mirror Auditor support,
    computing separate Merkle roots for reasoning events and UI events,
    then binding them via composite attestation root.

    Args:
        system: System identifier (e.g., "pl")
        proofs: List of proof dictionaries to be sealed
        ui_events: Optional list of UI event identifiers/hashes

    Returns:
        Dictionary containing block metadata with dual-root attestation:
        - block_number: Sequential block number
        - merkle_root: Legacy single Merkle root (aliased to reasoning_merkle_root)
        - reasoning_merkle_root: R_t - Merkle root of proof events
        - ui_merkle_root: U_t - Merkle root of UI events
        - composite_attestation_root: H_t - SHA256(R_t || U_t)
        - attestation_metadata: Additional attestation details
        - proof_count: Number of proofs in block
        - sealed_at: Unix timestamp of sealing

    Security:
        - Domain-separated Merkle trees for all roots
        - Cryptographic binding via composite attestation
        - Prevents second preimage attacks (CVE-2012-2459)
    """
    proofs_sequence: Sequence[Dict[str, Any]] = list(proofs)
    if ui_events is None:
        ui_sequence = consume_ui_artifacts()
    else:
        ui_sequence = list(ui_events)

    reasoning_tree = build_reasoning_attestation(proofs_sequence)
    ui_tree = build_ui_attestation(ui_sequence)

    r_t = reasoning_tree.root
    u_t = ui_tree.root
    composite_attestation_root = compute_composite_root(r_t, u_t)

    block_hash, block_number, sealed_at = _derive_block_identity(
        system,
        r_t,
        u_t,
        len(proofs_sequence),
        len(ui_sequence),
    )

    attestation_metadata = generate_attestation_metadata(
        r_t=r_t,
        u_t=u_t,
        h_t=composite_attestation_root,
        reasoning_event_count=len(proofs_sequence),
        ui_event_count=len(ui_sequence),
        reasoning_leaves=reasoning_tree.leaves,
        ui_leaves=ui_tree.leaves,
        extra={
            "system": system,
            "block_hash": block_hash,
            "block_number": block_number,
            "sealed_at": sealed_at,
        },
    )

    return {
        "block_number": block_number,
        "block_hash": block_hash,
        "merkle_root": r_t,  # Legacy field (aliased to R_t)
        "reasoning_merkle_root": r_t,
        "ui_merkle_root": u_t,
        "composite_attestation_root": composite_attestation_root,
        "attestation_metadata": attestation_metadata,
        "proof_count": len(proofs_sequence),
        "ui_event_count": len(ui_sequence),
        "sealed_at": sealed_at,
        "reasoning_leaves": _tree_metadata(reasoning_tree),
        "ui_leaves": _tree_metadata(ui_tree),
    }
