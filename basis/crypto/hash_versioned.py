"""
Versioned Cryptographic Hashing with Domain Separation.

This module extends the canonical hashing primitives to support multiple
hash algorithm versions for post-quantum migration. Key features:

- Versioned domain separation: <algorithm_id><domain_tag>
- Dual-commitment computation for migration periods
- Backward compatibility with legacy SHA-256 hashes
- Algorithm-agnostic Merkle tree construction

Security Invariants:
- Domain separation prevents cross-algorithm collisions
- Canonical encodings remain unchanged
- Historical hashes remain verifiable under original algorithm IDs
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple, Union

from basis.core import HexDigest
from basis.crypto.hash_registry import (
    HASH_ALG_SHA256,
    HashAlgorithm,
    get_algorithm,
)

# Legacy domain tags (single byte, SHA-256 era)
DOMAIN_LEAF = b"\x00"
DOMAIN_NODE = b"\x01"
DOMAIN_STMT = b"\x02"
DOMAIN_BLOCK = b"\x03"
DOMAIN_REASONING_EMPTY = b"\x10"
DOMAIN_UI_EMPTY = b"\x11"

# Domain tag identifiers (for versioned use)
TAG_LEAF = 0x00
TAG_NODE = 0x01
TAG_STMT = 0x02
TAG_BLOCK = 0x03
TAG_REASONING_EMPTY = 0x10
TAG_UI_EMPTY = 0x11


BytesLike = Union[bytes, str]


def _ensure_bytes(data: BytesLike) -> bytes:
    """Convert string to bytes if needed."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError(f"Unsupported data type for hashing: {type(data)!r}")


def make_versioned_domain(algorithm_id: int, domain_tag: int) -> bytes:
    """
    Create versioned domain separation prefix.
    
    Format: <algorithm_id:1 byte><domain_tag:1 byte>
    
    Args:
        algorithm_id: Hash algorithm version (0x00-0xFF)
        domain_tag: Domain tag identifier (0x00-0xFF)
        
    Returns:
        2-byte domain prefix
        
    Examples:
        >>> make_versioned_domain(0x00, 0x00)  # SHA-256 leaf
        b'\\x00\\x00'
        >>> make_versioned_domain(0x01, 0x01)  # PQ1 node
        b'\\x01\\x01'
    """
    return bytes([algorithm_id, domain_tag])


def sha256_hex_versioned(
    data: BytesLike,
    *,
    algorithm_id: int = HASH_ALG_SHA256,
    domain_tag: Optional[int] = None,
) -> HexDigest:
    """
    Compute hash with versioned domain separation.
    
    Args:
        data: Input data (string will be UTF-8 encoded)
        algorithm_id: Hash algorithm version
        domain_tag: Optional domain tag (if None, no domain separation)
        
    Returns:
        Hex digest string
    """
    algorithm = get_algorithm(algorithm_id)
    
    # Build domain prefix
    if domain_tag is not None:
        domain = make_versioned_domain(algorithm_id, domain_tag)
    else:
        domain = b""
    
    # Hash with domain
    payload = domain + _ensure_bytes(data)
    digest = algorithm.implementation(payload)
    return digest.hex()


def sha256_bytes_versioned(
    data: BytesLike,
    *,
    algorithm_id: int = HASH_ALG_SHA256,
    domain_tag: Optional[int] = None,
) -> bytes:
    """
    Compute hash with versioned domain separation, return bytes.
    
    Args:
        data: Input data (string will be UTF-8 encoded)
        algorithm_id: Hash algorithm version
        domain_tag: Optional domain tag (if None, no domain separation)
        
    Returns:
        Raw digest bytes
    """
    algorithm = get_algorithm(algorithm_id)
    
    # Build domain prefix
    if domain_tag is not None:
        domain = make_versioned_domain(algorithm_id, domain_tag)
    else:
        domain = b""
    
    # Hash with domain
    payload = domain + _ensure_bytes(data)
    return algorithm.implementation(payload)


def hash_statement_versioned(
    statement: str,
    *,
    algorithm_id: int = HASH_ALG_SHA256,
) -> HexDigest:
    """
    Hash a normalized statement with versioned domain separation.
    
    Args:
        statement: Normalized statement text
        algorithm_id: Hash algorithm version
        
    Returns:
        Hex hash digest
    """
    from basis.logic.normalizer import normalize
    
    normalized = normalize(statement)
    return sha256_hex_versioned(
        normalized,
        algorithm_id=algorithm_id,
        domain_tag=TAG_STMT,
    )


def hash_block_versioned(
    serialized_block: BytesLike,
    *,
    algorithm_id: int = HASH_ALG_SHA256,
) -> HexDigest:
    """
    Hash a block payload with versioned domain separation.
    
    Args:
        serialized_block: Serialized block data
        algorithm_id: Hash algorithm version
        
    Returns:
        Hex hash digest
    """
    return sha256_hex_versioned(
        serialized_block,
        algorithm_id=algorithm_id,
        domain_tag=TAG_BLOCK,
    )


def merkle_root_versioned(
    leaves: Sequence[str],
    *,
    algorithm_id: int = HASH_ALG_SHA256,
) -> HexDigest:
    """
    Compute deterministic Merkle root with versioned domain separation.
    
    This implementation uses:
    - Versioned LEAF domain for leaf nodes
    - Versioned NODE domain for internal nodes
    - Sorted leaves for determinism
    - Duplicate last node for odd counts
    
    Args:
        leaves: List of statement IDs or content to hash
        algorithm_id: Hash algorithm version
        
    Returns:
        Hex Merkle root
        
    Security:
    - Versioned domain separation prevents cross-algorithm collisions
    - Sorted leaves ensure deterministic output
    - Proper binary tree construction enables Merkle proofs
    """
    if not leaves:
        return sha256_hex_versioned(
            b"",
            algorithm_id=algorithm_id,
            domain_tag=TAG_LEAF,
        )
    
    from basis.logic.normalizer import normalize
    
    # Normalize and sort leaves
    leaf_bytes = sorted(normalize(leaf).encode("utf-8") for leaf in leaves)
    
    # Hash leaves with versioned domain
    nodes = [
        sha256_bytes_versioned(
            leaf,
            algorithm_id=algorithm_id,
            domain_tag=TAG_LEAF,
        )
        for leaf in leaf_bytes
    ]
    
    # Build tree
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        
        next_level: List[bytes] = []
        for left, right in zip(nodes[0::2], nodes[1::2]):
            parent = sha256_bytes_versioned(
                left + right,
                algorithm_id=algorithm_id,
                domain_tag=TAG_NODE,
            )
            next_level.append(parent)
        
        nodes = next_level
    
    return nodes[0].hex()


def compute_merkle_proof_versioned(
    index: int,
    leaves: Sequence[str],
    *,
    algorithm_id: int = HASH_ALG_SHA256,
) -> List[Tuple[HexDigest, bool]]:
    """
    Compute Merkle proof for a leaf with versioned hashing.
    
    Args:
        index: Index of leaf to prove
        leaves: All leaves in the tree
        algorithm_id: Hash algorithm version
        
    Returns:
        List of (sibling_hash, sibling_is_left) tuples
        
    Raises:
        ValueError: If index is out of bounds
    """
    if index < 0 or index >= len(leaves):
        raise ValueError(f"Leaf index {index} out of bounds for {len(leaves)} leaves.")
    
    from basis.logic.normalizer import normalize
    
    # Normalize and pair with original indices
    paired = [
        (normalize(leaf).encode("utf-8"), original_idx)
        for original_idx, leaf in enumerate(leaves)
    ]
    paired.sort(key=lambda item: item[0])
    
    # Find target position in sorted order
    target_position = next(
        pos for pos, (_, original_idx) in enumerate(paired)
        if original_idx == index
    )
    
    # Hash leaves with versioned domain
    level = [
        sha256_bytes_versioned(
            leaf_bytes,
            algorithm_id=algorithm_id,
            domain_tag=TAG_LEAF,
        )
        for leaf_bytes, _ in paired
    ]
    
    proof: List[Tuple[HexDigest, bool]] = []
    cursor = target_position
    
    # Build proof
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        
        is_left_node = cursor % 2 == 1
        sibling_index = cursor - 1 if is_left_node else cursor + 1
        proof.append((level[sibling_index].hex(), is_left_node))
        
        next_level: List[bytes] = []
        for left, right in zip(level[0::2], level[1::2]):
            parent = sha256_bytes_versioned(
                left + right,
                algorithm_id=algorithm_id,
                domain_tag=TAG_NODE,
            )
            next_level.append(parent)
        
        cursor //= 2
        level = next_level
    
    return proof


def verify_merkle_proof_versioned(
    leaf: str,
    proof: Sequence[Tuple[HexDigest, bool]],
    expected_root: HexDigest,
    *,
    algorithm_id: int = HASH_ALG_SHA256,
) -> bool:
    """
    Verify a Merkle inclusion proof with versioned hashing.
    
    Args:
        leaf: Raw leaf value prior to normalization
        proof: Sequence of (sibling_hash, sibling_is_left)
        expected_root: Merkle root to compare against
        algorithm_id: Hash algorithm version
        
    Returns:
        True if proof is valid
    """
    from basis.logic.normalizer import normalize
    
    cursor = sha256_bytes_versioned(
        normalize(leaf).encode("utf-8"),
        algorithm_id=algorithm_id,
        domain_tag=TAG_LEAF,
    )
    
    for sibling_hex, sibling_is_left in proof:
        sibling = bytes.fromhex(sibling_hex)
        combined = sibling + cursor if sibling_is_left else cursor + sibling
        cursor = sha256_bytes_versioned(
            combined,
            algorithm_id=algorithm_id,
            domain_tag=TAG_NODE,
        )
    
    return cursor.hex() == expected_root


def compute_dual_commitment(
    legacy_hash: HexDigest,
    pq_hash: HexDigest,
    pq_algorithm_id: int,
) -> HexDigest:
    """
    Compute dual commitment binding legacy and PQ hashes.
    
    During migration periods, blocks maintain both legacy (SHA-256) and
    post-quantum hash commitments. This function cryptographically binds
    them together to prevent selective forgery.
    
    Format: SHA256(pq_algorithm_id || legacy_hash || pq_hash)
    
    Args:
        legacy_hash: Legacy SHA-256 hash (64-char hex)
        pq_hash: Post-quantum hash (64-char hex)
        pq_algorithm_id: PQ algorithm identifier
        
    Returns:
        Dual commitment hash (64-char hex)
        
    Raises:
        ValueError: If hashes are invalid format
        
    Examples:
        >>> legacy = "a" * 64
        >>> pq = "b" * 64
        >>> commitment = compute_dual_commitment(legacy, pq, 0x01)
        >>> len(commitment)
        64
    """
    # Validate inputs
    if len(legacy_hash) != 64:
        raise ValueError(f"Invalid legacy hash length: {len(legacy_hash)}")
    if len(pq_hash) != 64:
        raise ValueError(f"Invalid PQ hash length: {len(pq_hash)}")
    
    try:
        int(legacy_hash, 16)
        int(pq_hash, 16)
    except ValueError:
        raise ValueError("Hashes must be valid hex strings")
    
    # Build commitment payload
    payload = (
        bytes([pq_algorithm_id]) +
        legacy_hash.encode("ascii") +
        pq_hash.encode("ascii")
    )
    
    # Use SHA-256 for binding (conservative choice)
    return hashlib.sha256(payload).hexdigest()


def verify_dual_commitment(
    legacy_hash: HexDigest,
    pq_hash: HexDigest,
    pq_algorithm_id: int,
    expected_commitment: HexDigest,
) -> bool:
    """
    Verify a dual commitment.
    
    Args:
        legacy_hash: Legacy SHA-256 hash
        pq_hash: Post-quantum hash
        pq_algorithm_id: PQ algorithm identifier
        expected_commitment: Expected dual commitment hash
        
    Returns:
        True if commitment is valid
    """
    try:
        computed = compute_dual_commitment(legacy_hash, pq_hash, pq_algorithm_id)
        return computed == expected_commitment
    except ValueError:
        return False


__all__ = [
    "DOMAIN_LEAF",
    "DOMAIN_NODE",
    "DOMAIN_STMT",
    "DOMAIN_BLOCK",
    "DOMAIN_REASONING_EMPTY",
    "DOMAIN_UI_EMPTY",
    "TAG_LEAF",
    "TAG_NODE",
    "TAG_STMT",
    "TAG_BLOCK",
    "TAG_REASONING_EMPTY",
    "TAG_UI_EMPTY",
    "make_versioned_domain",
    "sha256_hex_versioned",
    "sha256_bytes_versioned",
    "hash_statement_versioned",
    "hash_block_versioned",
    "merkle_root_versioned",
    "compute_merkle_proof_versioned",
    "verify_merkle_proof_versioned",
    "compute_dual_commitment",
    "verify_dual_commitment",
]
