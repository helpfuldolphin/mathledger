"""
Canonical hashing primitives with explicit domain separation.

This module centralises every SHA-256 invocation used by the basis so that
cryptographic intent remains auditable. Hashing functions are pure and return
deterministic values independent of global state.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple, Union

from basis.core import HexDigest

# Domain separation prefixes – single-byte tags keep hashes orthogonal.
DOMAIN_LEAF = b"\x00"
DOMAIN_NODE = b"\x01"
DOMAIN_STMT = b"\x02"
DOMAIN_BLOCK = b"\x03"
DOMAIN_REASONING_EMPTY = b"\x10"
DOMAIN_UI_EMPTY = b"\x11"


BytesLike = Union[bytes, str]


def _ensure_bytes(data: BytesLike) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError(f"Unsupported data type for hashing: {type(data)!r}")


def sha256_hex(data: BytesLike, *, domain: bytes = b"") -> HexDigest:
    """Return the SHA-256 digest as a hex string."""
    digest = hashlib.sha256(domain + _ensure_bytes(data)).hexdigest()
    return digest


def sha256_bytes(data: BytesLike, *, domain: bytes = b"") -> bytes:
    """Return the SHA-256 digest as raw bytes."""
    return hashlib.sha256(domain + _ensure_bytes(data)).digest()


def hash_statement(statement: str) -> HexDigest:
    """Hash a normalised statement with the statement domain tag."""
    from basis.logic.normalizer import normalize

    normalized = normalize(statement)
    return sha256_hex(normalized, domain=DOMAIN_STMT)


def hash_block(serialised_block: BytesLike) -> HexDigest:
    """Hash a block payload (caller controls serialisation)."""
    return sha256_hex(serialised_block, domain=DOMAIN_BLOCK)


def _sorted_leaf_bytes(leaves: Iterable[str]) -> List[bytes]:
    from basis.logic.normalizer import normalize

    return sorted(normalize(leaf).encode("utf-8") for leaf in leaves)


def merkle_root(leaves: Sequence[str]) -> HexDigest:
    """
    Compute a deterministic merkle root for the given leaves.

    Empty input yields the domain-separated hash of the empty string.
    """
    if not leaves:
        return sha256_hex(b"", domain=DOMAIN_LEAF)

    level = [sha256_bytes(leaf, domain=DOMAIN_LEAF) for leaf in _sorted_leaf_bytes(leaves)]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        next_level: List[bytes] = []
        for left, right in zip(level[0::2], level[1::2]):
            next_level.append(sha256_bytes(left + right, domain=DOMAIN_NODE))
        level = next_level

    return level[0].hex()


def compute_merkle_proof(index: int, leaves: Sequence[str]) -> List[Tuple[HexDigest, bool]]:
    """
    Compute the merkle proof for a leaf.

    Returns a list of (sibling_hash, sibling_is_left) tuples.
    """
    if index < 0 or index >= len(leaves):
        raise ValueError(f"Leaf index {index} out of bounds for {len(leaves)} leaves.")

    from basis.logic.normalizer import normalize

    paired = [(normalize(leaf).encode("utf-8"), original_idx) for original_idx, leaf in enumerate(leaves)]
    paired.sort(key=lambda item: item[0])

    target_position = next(pos for pos, (_, original_idx) in enumerate(paired) if original_idx == index)

    level = [sha256_bytes(leaf_bytes, domain=DOMAIN_LEAF) for leaf_bytes, _ in paired]
    proof: List[Tuple[HexDigest, bool]] = []
    cursor = target_position

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        is_left_node = cursor % 2 == 1
        sibling_index = cursor - 1 if is_left_node else cursor + 1
        proof.append((level[sibling_index].hex(), is_left_node))

        next_level: List[bytes] = []
        for left, right in zip(level[0::2], level[1::2]):
            next_level.append(sha256_bytes(left + right, domain=DOMAIN_NODE))

        cursor //= 2
        level = next_level

    return proof


def verify_merkle_proof(leaf: str, proof: Sequence[Tuple[HexDigest, bool]], expected_root: HexDigest) -> bool:
    """
    Verify a merkle inclusion proof.

    Arguments:
        leaf: raw leaf value prior to normalisation
        proof: sequence of (sibling_hash, sibling_is_left)
        expected_root: merkle root to compare against
    """
    from basis.logic.normalizer import normalize

    cursor = sha256_bytes(normalize(leaf).encode("utf-8"), domain=DOMAIN_LEAF)

    for sibling_hex, sibling_is_left in proof:
        sibling = bytes.fromhex(sibling_hex)
        combined = sibling + cursor if sibling_is_left else cursor + sibling
        cursor = sha256_bytes(combined, domain=DOMAIN_NODE)

    return cursor.hex() == expected_root


def reasoning_root(events: Sequence[str]) -> HexDigest:
    """Domain-separated Merkle root for reasoning provenance."""
    if not events:
        return sha256_hex(b"", domain=DOMAIN_REASONING_EMPTY)
    return merkle_root(events)


def ui_root(events: Sequence[str]) -> HexDigest:
    """Domain-separated Merkle root for UI provenance."""
    if not events:
        return sha256_hex(b"", domain=DOMAIN_UI_EMPTY)
    return merkle_root(events)