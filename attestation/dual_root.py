"""
Dual-Root Attestation Module
============================

Provides watertight primitives for constructing, binding, and auditing dual-root
attestations.  A dual-root attestation consists of:

- R_t: Merkle root over normalized reasoning/proof artifacts
- U_t: Merkle root over normalized human/UI event artifacts
- H_t: Composite root SHA256(R_t || U_t) binding both streams

This module normalizes all inputs with RFC 8785 canonicalization, hashes leaves
with domain separation, constructs deterministic Merkle trees (including
inclusion proofs), and exposes helpers for metadata emission and verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from substrate.crypto.core import rfc8785_canonicalize
from substrate.crypto.hashing import (
    merkle_root as crypto_merkle_root,
    compute_merkle_proof,
    sha256_hex,
)

# --------------------------------------------------------------------------- #
# Domain separation and sentinels
# --------------------------------------------------------------------------- #

DOMAIN_REASONING_LEAF = b"\xA0reasoning-leaf"
DOMAIN_UI_LEAF = b"\xA1ui-leaf"

REASONING_EMPTY_SENTINEL = b"REASONING:EMPTY"
UI_EMPTY_SENTINEL = b"UI:EMPTY"

RawLeaf = Union[str, bytes, Mapping[str, Any], Sequence[Any], int, float, bool, None]


@dataclass(frozen=True)
class AttestationLeaf:
    """Single leaf entry with canonical value and Merkle inclusion proof."""

    original_index: int
    sorted_index: int
    canonical_value: str
    leaf_hash: str
    merkle_proof: Tuple[Tuple[str, bool], ...]

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize leaf for JSON metadata."""
        return {
            "original_index": self.original_index,
            "sorted_index": self.sorted_index,
            "canonical_value": self.canonical_value,
            "leaf_hash": self.leaf_hash,
            "merkle_proof": [[sibling, is_left] for sibling, is_left in self.merkle_proof],
        }


@dataclass(frozen=True)
class AttestationTree:
    """Deterministic Merkle tree plus leaves."""

    kind: str
    root: str
    leaves: Tuple[AttestationLeaf, ...]
    domain: bytes

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "root": self.root,
            "leaf_count": len(self.leaves),
            "leaves": [leaf.to_metadata() for leaf in self.leaves],
        }


# --------------------------------------------------------------------------- #
# Canonicalization helpers
# --------------------------------------------------------------------------- #

def _canonicalize_leaf(value: RawLeaf) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return rfc8785_canonicalize(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return rfc8785_canonicalize(list(value))
    if isinstance(value, (int, float, bool)) or value is None:
        return rfc8785_canonicalize(value)
    return rfc8785_canonicalize(value)


def hash_reasoning_leaf(canonical_value: str) -> str:
    """Hash a canonical reasoning artifact."""
    return sha256_hex(canonical_value, domain=DOMAIN_REASONING_LEAF)


def hash_ui_leaf(canonical_value: str) -> str:
    """Hash a canonical UI artifact."""
    return sha256_hex(canonical_value, domain=DOMAIN_UI_LEAF)


def canonicalize_reasoning_artifact(value: RawLeaf) -> str:
    """Public helper mirroring internal reasoning canonicalization."""
    return _canonicalize_leaf(value)


def canonicalize_ui_artifact(value: RawLeaf) -> str:
    """Public helper mirroring internal UI canonicalization."""
    return _canonicalize_leaf(value)


def _build_attestation_tree(
    items: Sequence[RawLeaf],
    *,
    kind: str,
    hash_fn,
    empty_sentinel: bytes,
    domain: bytes,
) -> AttestationTree:
    canonical_values = [_canonicalize_leaf(item) for item in items]

    if not canonical_values:
        root = hashlib.sha256(empty_sentinel).hexdigest()
        return AttestationTree(kind=kind, root=root, leaves=tuple(), domain=domain)

    leaf_hashes = [hash_fn(canonical) for canonical in canonical_values]
    root = crypto_merkle_root(leaf_hashes)

    # Establish deterministic sorted indices consistent with merkle_root()
    sorted_pairs = sorted(
        [(leaf_hashes[idx], idx) for idx in range(len(leaf_hashes))],
        key=lambda pair: pair[0],
    )
    index_to_sorted = {original_idx: sorted_idx for sorted_idx, (_, original_idx) in enumerate(sorted_pairs)}

    leaves: List[AttestationLeaf] = []
    for original_index, canonical in enumerate(canonical_values):
        proof = tuple(compute_merkle_proof(original_index, leaf_hashes))
        leaves.append(
            AttestationLeaf(
                original_index=original_index,
                sorted_index=index_to_sorted[original_index],
                canonical_value=canonical,
                leaf_hash=leaf_hashes[original_index],
                merkle_proof=proof,
            )
        )

    return AttestationTree(kind=kind, root=root, leaves=tuple(leaves), domain=domain)


def build_reasoning_attestation(proof_events: Sequence[RawLeaf]) -> AttestationTree:
    """Construct reasoning attestation tree."""
    return _build_attestation_tree(
        proof_events,
        kind="reasoning",
        hash_fn=hash_reasoning_leaf,
        empty_sentinel=REASONING_EMPTY_SENTINEL,
        domain=DOMAIN_REASONING_LEAF,
    )


def build_ui_attestation(ui_events: Sequence[RawLeaf]) -> AttestationTree:
    """Construct UI attestation tree."""
    return _build_attestation_tree(
        ui_events,
        kind="ui",
        hash_fn=hash_ui_leaf,
        empty_sentinel=UI_EMPTY_SENTINEL,
        domain=DOMAIN_UI_LEAF,
    )


# --------------------------------------------------------------------------- #
# Public API (roots, composite, metadata, verification)
# --------------------------------------------------------------------------- #

def compute_reasoning_root(proof_events: Sequence[RawLeaf]) -> str:
    """
    Compute reasoning Merkle root (R_t) from proof events.

    Args:
        proof_events: Iterable of proof artifacts (strings, dicts, etc.)

    Returns:
        64-char hex Merkle root representing reasoning lineage
    """
    return build_reasoning_attestation(proof_events).root


def compute_ui_root(ui_events: Sequence[RawLeaf]) -> str:
    """
    Compute UI Merkle root (U_t) from human interaction events.

    Args:
        ui_events: Iterable of UI artifacts (strings, dicts, etc.)

    Returns:
        64-char hex Merkle root representing human event lineage
    """
    return build_ui_attestation(ui_events).root


def compute_composite_root(r_t: str, u_t: str) -> str:
    """
    Compute composite attestation root: H_t = SHA256(R_t || U_t).

    Args:
        r_t: Reasoning Merkle root (64-char hex)
        u_t: UI Merkle root (64-char hex)

    Returns:
        64-char hex composite attestation root

    Raises:
        ValueError: If either root is invalid
    """
    if not r_t or not u_t:
        raise ValueError("Both R_t and U_t must be non-empty")

    if len(r_t) != 64 or len(u_t) != 64:
        raise ValueError(f"Invalid root length: R_t={len(r_t)}, U_t={len(u_t)}")

    try:
        int(r_t, 16)
        int(u_t, 16)
    except ValueError:
        raise ValueError(f"Invalid hex format: R_t={r_t}, U_t={u_t}")

    composite_data = f"{r_t}{u_t}".encode("ascii")
    return hashlib.sha256(composite_data).hexdigest()


def _serialize_leaves(leaves: Sequence[AttestationLeaf]) -> List[Dict[str, Any]]:
    return [leaf.to_metadata() for leaf in leaves]


def generate_attestation_metadata(
    r_t: str,
    u_t: str,
    h_t: str,
    reasoning_event_count: int = 0,
    ui_event_count: int = 0,
    *,
    reasoning_leaves: Optional[Sequence[AttestationLeaf]] = None,
    ui_leaves: Optional[Sequence[AttestationLeaf]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate attestation metadata for a block.

    Args:
        r_t: Reasoning Merkle root
        u_t: UI Merkle root
        h_t: Composite attestation root
        reasoning_event_count: Number of reasoning events
        ui_event_count: Number of UI events
        reasoning_leaves: Optional sequence of reasoning leaves for inclusion proofs
        ui_leaves: Optional sequence of UI leaves for inclusion proofs
        extra: Additional metadata fields

    Returns:
        Dictionary of attestation metadata
    """
    metadata: Dict[str, Any] = {
        "reasoning_merkle_root": r_t,
        "ui_merkle_root": u_t,
        "composite_attestation_root": h_t,
        "reasoning_event_count": reasoning_event_count,
        "ui_event_count": ui_event_count,
        "attestation_version": "v2",
        "algorithm": "SHA256",
        "composite_formula": "SHA256(R_t || U_t)",
        "leaf_hash_algorithm": "sha256",
    }

    if reasoning_leaves:
        metadata["reasoning_leaves"] = _serialize_leaves(reasoning_leaves)
    if ui_leaves:
        metadata["ui_leaves"] = _serialize_leaves(ui_leaves)

    if extra:
        metadata.update(extra)

    return metadata


def verify_composite_integrity(r_t: str, u_t: str, h_t: str) -> bool:
    """
    Verify that composite root H_t matches SHA256(R_t || U_t).
    """
    try:
        return compute_composite_root(r_t, u_t) == h_t
    except Exception:
        return False


__all__ = [
    "AttestationLeaf",
    "AttestationTree",
    "build_reasoning_attestation",
    "build_ui_attestation",
    "compute_reasoning_root",
    "compute_ui_root",
    "compute_composite_root",
    "generate_attestation_metadata",
    "canonicalize_reasoning_artifact",
    "canonicalize_ui_artifact",
    "hash_reasoning_leaf",
    "hash_ui_leaf",
    "verify_composite_integrity",
]
