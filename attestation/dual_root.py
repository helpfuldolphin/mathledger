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
    sha3_256_hex,
)

# --------------------------------------------------------------------------- #
# Domain separation and sentinels
# --------------------------------------------------------------------------- #

DOMAIN_REASONING_LEAF = b"\xA0reasoning-leaf"
DOMAIN_UI_LEAF = b"\xA1ui-leaf"

REASONING_EMPTY_SENTINEL = b"REASONING:EMPTY"
UI_EMPTY_SENTINEL = b"UI:EMPTY"

DOMAIN_REASONING_PQ = b"\xB0reasoning-pq"
DOMAIN_UI_PQ = b"\xB1ui-pq"
DOMAIN_COMPOSITE_PQ = b"\xB2composite-pq"

PRIMARY_ALGORITHM = "SHA256"
PRIMARY_VERSION = "v1"
POST_QUANTUM_ALGORITHM = "SHA3-256"
POST_QUANTUM_VERSION = "pq.v1"

_PQ_DOMAIN_BY_KIND = {
    "reasoning": DOMAIN_REASONING_PQ,
    "ui": DOMAIN_UI_PQ,
    "composite": DOMAIN_COMPOSITE_PQ,
}

_EMPTY_SENTINEL_BY_KIND = {
    "reasoning": REASONING_EMPTY_SENTINEL,
    "ui": UI_EMPTY_SENTINEL,
}

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
    commitments: "HashCommitmentBundle"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "root": self.root,
            "leaf_count": len(self.leaves),
            "leaves": [leaf.to_metadata() for leaf in self.leaves],
            "hash_commitments": self.commitments.to_metadata(),
        }


@dataclass(frozen=True)
class HashCommitment:
    """Single hash commitment with algorithm metadata."""

    algorithm: str
    version: str
    digest: str

    def to_metadata(self) -> Dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class HashCommitmentBundle:
    """Pair of classical + post-quantum hash commitments."""

    classical: HashCommitment
    post_quantum: HashCommitment

    @classmethod
    def from_roots(cls, classical_digest: str, pq_digest: str) -> "HashCommitmentBundle":
        return cls(
            classical=HashCommitment(PRIMARY_ALGORITHM, PRIMARY_VERSION, classical_digest),
            post_quantum=HashCommitment(POST_QUANTUM_ALGORITHM, POST_QUANTUM_VERSION, pq_digest),
        )

    @classmethod
    def fallback_from_digest(cls, kind: str, digest: str) -> "HashCommitmentBundle":
        domain = _PQ_DOMAIN_BY_KIND[kind]
        pq_digest = sha3_256_hex(digest, domain=domain)
        return cls.from_roots(digest, pq_digest)

    def to_metadata(self) -> Dict[str, Dict[str, str]]:
        return {
            "classical": self.classical.to_metadata(),
            "post_quantum": self.post_quantum.to_metadata(),
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
    pq_domain = _PQ_DOMAIN_BY_KIND.get(kind)
    if pq_domain is None:
        raise ValueError(f"Unsupported attestation tree kind: {kind}")
    pq_root = _compute_post_quantum_root(
        canonical_values,
        domain=pq_domain,
        empty_sentinel=empty_sentinel,
    )

    if not canonical_values:
        root = hashlib.sha256(empty_sentinel).hexdigest()
        return AttestationTree(
            kind=kind,
            root=root,
            leaves=tuple(),
            domain=domain,
            commitments=HashCommitmentBundle.from_roots(root, pq_root),
        )

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

    return AttestationTree(
        kind=kind,
        root=root,
        leaves=tuple(leaves),
        domain=domain,
        commitments=HashCommitmentBundle.from_roots(root, pq_root),
    )


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
    Compute composite attestation root: H_t = SHA256("EPOCH:" || R_t || U_t).

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

    composite_data = f"EPOCH:{r_t}{u_t}".encode("ascii")
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
    reasoning_commitments: Optional[HashCommitmentBundle] = None,
    ui_commitments: Optional[HashCommitmentBundle] = None,
    composite_commitments: Optional[HashCommitmentBundle] = None,
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
        "post_quantum_algorithm": POST_QUANTUM_ALGORITHM,
        "composite_formula": "SHA256(R_t || U_t)",
        "leaf_hash_algorithm": "sha256",
    }

    if reasoning_leaves:
        metadata["reasoning_leaves"] = _serialize_leaves(reasoning_leaves)
    if ui_leaves:
        metadata["ui_leaves"] = _serialize_leaves(ui_leaves)

    reasoning_bundle = reasoning_commitments or _commitments_from_context(
        kind="reasoning",
        digest=r_t,
        leaves=reasoning_leaves,
    )
    ui_bundle = ui_commitments or _commitments_from_context(
        kind="ui",
        digest=u_t,
        leaves=ui_leaves,
    )
    composite_bundle = composite_commitments or build_composite_commitment(
        h_t,
        reasoning_bundle,
        ui_bundle,
    )

    metadata["hash_commitments"] = {
        "reasoning": reasoning_bundle.to_metadata(),
        "ui": ui_bundle.to_metadata(),
        "composite": composite_bundle.to_metadata(),
    }

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
    "HashCommitment",
    "HashCommitmentBundle",
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
    "build_composite_commitment",
]


def _compute_post_quantum_root(
    canonical_values: Sequence[str],
    *,
    domain: bytes,
    empty_sentinel: bytes,
) -> str:
    if canonical_values:
        payload = rfc8785_canonicalize(sorted(canonical_values))
    else:
        payload = empty_sentinel
    return sha3_256_hex(payload, domain=domain)


def _commitments_from_context(
    *,
    kind: str,
    digest: str,
    leaves: Optional[Sequence[AttestationLeaf]],
) -> HashCommitmentBundle:
    if leaves:
        canonical_values = [leaf.canonical_value for leaf in leaves]
        sentinel = _EMPTY_SENTINEL_BY_KIND[kind]
        pq_root = _compute_post_quantum_root(
            canonical_values,
            domain=_PQ_DOMAIN_BY_KIND[kind],
            empty_sentinel=sentinel,
        )
        return HashCommitmentBundle.from_roots(digest, pq_root)
    return HashCommitmentBundle.fallback_from_digest(kind, digest)


def build_composite_commitment(
    classical_digest: str,
    reasoning_bundle: HashCommitmentBundle,
    ui_bundle: HashCommitmentBundle,
) -> HashCommitmentBundle:
    pq_payload = (
        reasoning_bundle.post_quantum.digest + ui_bundle.post_quantum.digest
    )
    pq_digest = sha3_256_hex(pq_payload, domain=DOMAIN_COMPOSITE_PQ)
    return HashCommitmentBundle.from_roots(classical_digest, pq_digest)
