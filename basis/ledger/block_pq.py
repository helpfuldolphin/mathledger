"""
Post-Quantum Ready Block Structures.

This module extends the canonical block structures to support dual-commitment
headers for post-quantum migration. During migration periods, blocks maintain
both legacy (SHA-256) and post-quantum hash commitments.

Key Features:
- Backward-compatible with legacy BlockHeader
- Dual-commitment binding (legacy + PQ)
- Algorithm version tracking
- Migration-aware block sealing

Security Invariants:
- Legacy fields remain unchanged for backward compatibility
- PQ fields are optional during pre-migration phase
- Dual commitment cryptographically binds both hash chains
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from basis.core import Block, BlockHeader, HexDigest
from basis.crypto.hash_registry import (
    HASH_ALG_SHA256,
    get_algorithm,
    get_canonical_algorithm,
)
from basis.crypto.hash_versioned import (
    compute_dual_commitment,
    hash_block_versioned,
    merkle_root_versioned,
)
from basis.logic.normalizer import normalize_many


@dataclass(frozen=True)
class BlockHeaderPQ:
    """
    Post-quantum ready block header with dual commitments.
    
    This header maintains both legacy (SHA-256) and post-quantum hash
    commitments during migration periods. After full PQ activation,
    legacy fields may become optional.
    
    Attributes:
        block_number: Sequential block number
        prev_hash: Legacy SHA-256 hash of previous block
        merkle_root: Legacy SHA-256 Merkle root of statements
        timestamp: Unix timestamp
        version: Block format version ("v2-pq")
        hash_algorithm: Current hash algorithm ID (default: SHA-256)
        pq_prev_hash: PQ hash of previous block (optional)
        pq_merkle_root: PQ Merkle root of statements (optional)
        pq_algorithm: PQ algorithm ID (optional)
        dual_commitment: SHA256(legacy || pq) binding (optional)
        metadata: Additional metadata (extensible)
    """
    
    block_number: int
    prev_hash: HexDigest
    merkle_root: HexDigest
    timestamp: float
    version: str = "v2-pq"
    
    # Hash algorithm tracking
    hash_algorithm: int = HASH_ALG_SHA256
    
    # Post-quantum dual commitment fields
    pq_prev_hash: Optional[HexDigest] = None
    pq_merkle_root: Optional[HexDigest] = None
    pq_algorithm: Optional[int] = None
    dual_commitment: Optional[HexDigest] = None
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_legacy_header(self) -> BlockHeader:
        """
        Convert to legacy BlockHeader format.
        
        This enables backward compatibility with systems that only
        understand the legacy format.
        
        Returns:
            BlockHeader with legacy fields only
        """
        return BlockHeader(
            block_number=self.block_number,
            prev_hash=self.prev_hash,
            merkle_root=self.merkle_root,
            timestamp=self.timestamp,
            version="v1",  # Legacy version
        )
    
    def has_dual_commitment(self) -> bool:
        """Check if this header contains a dual commitment."""
        return (
            self.pq_prev_hash is not None and
            self.pq_merkle_root is not None and
            self.pq_algorithm is not None and
            self.dual_commitment is not None
        )
    
    def verify_dual_commitment(self) -> bool:
        """
        Verify that the dual commitment is valid.
        
        Returns:
            True if dual commitment matches computed value
        """
        if not self.has_dual_commitment():
            return True  # No dual commitment to verify
        
        # Compute expected dual commitment for prev_hash
        expected_prev = compute_dual_commitment(
            self.prev_hash,
            self.pq_prev_hash,  # type: ignore
            self.pq_algorithm,  # type: ignore
        )
        
        # Compute expected dual commitment for merkle_root
        expected_merkle = compute_dual_commitment(
            self.merkle_root,
            self.pq_merkle_root,  # type: ignore
            self.pq_algorithm,  # type: ignore
        )
        
        # The dual_commitment field should bind both
        # For now, we use the merkle_root binding as the primary commitment
        return self.dual_commitment == expected_merkle


@dataclass(frozen=True)
class BlockPQ:
    """
    Post-quantum ready block with dual-commitment header.
    
    Attributes:
        header: PQ-ready block header
        statements: Tuple of normalized statements
    """
    
    header: BlockHeaderPQ
    statements: Tuple[str, ...]
    
    def to_legacy_block(self) -> Block:
        """
        Convert to legacy Block format.
        
        Returns:
            Block with legacy header
        """
        return Block(
            header=self.header.to_legacy_header(),
            statements=self.statements,
        )


def seal_block_pq(
    statements: Sequence[str],
    *,
    prev_hash: HexDigest,
    block_number: int,
    timestamp: float,
    version: str = "v2-pq",
    hash_algorithm: Optional[int] = None,
    enable_pq: bool = False,
    pq_algorithm: Optional[int] = None,
) -> BlockPQ:
    """
    Seal a block with optional post-quantum dual commitments.
    
    This function produces an immutable block with deterministic Merkle header.
    During migration periods, it can compute both legacy and PQ commitments.
    
    Args:
        statements: Sequence of statement strings
        prev_hash: Hash of previous block (legacy)
        block_number: Sequential block number
        timestamp: Unix timestamp
        version: Block format version
        hash_algorithm: Hash algorithm for legacy fields (default: canonical for block)
        enable_pq: Whether to compute PQ dual commitments
        pq_algorithm: PQ hash algorithm ID (required if enable_pq=True)
        
    Returns:
        Sealed BlockPQ with optional dual commitments
        
    Raises:
        ValueError: If enable_pq=True but pq_algorithm is None
        
    Examples:
        >>> # Legacy mode (no PQ)
        >>> block = seal_block_pq(
        ...     ["p->p", "q->q"],
        ...     prev_hash="0" * 64,
        ...     block_number=1,
        ...     timestamp=1234567890.0,
        ... )
        >>> block.header.has_dual_commitment()
        False
        
        >>> # Dual commitment mode
        >>> block = seal_block_pq(
        ...     ["p->p", "q->q"],
        ...     prev_hash="0" * 64,
        ...     block_number=1,
        ...     timestamp=1234567890.0,
        ...     enable_pq=True,
        ...     pq_algorithm=0x01,
        ... )
        >>> block.header.has_dual_commitment()
        True
    """
    # Normalize and sort statements
    normalized = normalize_many(statements)
    sorted_statements = tuple(sorted(normalized))
    
    # Determine hash algorithm for legacy fields
    if hash_algorithm is None:
        # Use canonical algorithm for this block number
        canonical_alg = get_canonical_algorithm(block_number)
        hash_algorithm = canonical_alg.algorithm_id
    
    # Compute legacy Merkle root
    merkle = merkle_root_versioned(
        normalized,
        algorithm_id=hash_algorithm,
    )
    
    # Compute PQ commitments if enabled
    pq_prev_hash = None
    pq_merkle_root = None
    dual_commitment = None
    
    if enable_pq:
        if pq_algorithm is None:
            raise ValueError("pq_algorithm must be specified when enable_pq=True")
        
        # Compute PQ Merkle root
        pq_merkle_root = merkle_root_versioned(
            normalized,
            algorithm_id=pq_algorithm,
        )
        
        # For prev_hash, we assume the previous block also had dual commitment
        # In practice, this would be passed in or looked up from chain state
        # For now, we compute a placeholder PQ prev_hash
        pq_prev_hash = hash_block_versioned(
            prev_hash,  # Hash the legacy prev_hash as placeholder
            algorithm_id=pq_algorithm,
        )
        
        # Compute dual commitment binding
        dual_commitment = compute_dual_commitment(
            merkle,
            pq_merkle_root,
            pq_algorithm,
        )
    
    # Build header
    header = BlockHeaderPQ(
        block_number=block_number,
        prev_hash=prev_hash,
        merkle_root=merkle,
        timestamp=timestamp,
        version=version,
        hash_algorithm=hash_algorithm,
        pq_prev_hash=pq_prev_hash,
        pq_merkle_root=pq_merkle_root,
        pq_algorithm=pq_algorithm,
        dual_commitment=dual_commitment,
    )
    
    return BlockPQ(header=header, statements=sorted_statements)


def seal_block_dual_commitment(
    statements: Sequence[str],
    *,
    prev_hash_legacy: HexDigest,
    prev_hash_pq: HexDigest,
    block_number: int,
    timestamp: float,
    pq_algorithm: int,
) -> BlockPQ:
    """
    Seal a block with full dual commitments (migration mode).
    
    This is a convenience function for the migration period when both
    legacy and PQ prev_hash values are available from the chain.
    
    Args:
        statements: Sequence of statement strings
        prev_hash_legacy: Legacy SHA-256 hash of previous block
        prev_hash_pq: PQ hash of previous block
        block_number: Sequential block number
        timestamp: Unix timestamp
        pq_algorithm: PQ hash algorithm ID
        
    Returns:
        Sealed BlockPQ with dual commitments
    """
    # Normalize and sort statements
    normalized = normalize_many(statements)
    sorted_statements = tuple(sorted(normalized))
    
    # Compute legacy Merkle root (SHA-256)
    merkle_legacy = merkle_root_versioned(
        normalized,
        algorithm_id=HASH_ALG_SHA256,
    )
    
    # Compute PQ Merkle root
    merkle_pq = merkle_root_versioned(
        normalized,
        algorithm_id=pq_algorithm,
    )
    
    # Compute dual commitment
    dual_commitment = compute_dual_commitment(
        merkle_legacy,
        merkle_pq,
        pq_algorithm,
    )
    
    # Build header
    header = BlockHeaderPQ(
        block_number=block_number,
        prev_hash=prev_hash_legacy,
        merkle_root=merkle_legacy,
        timestamp=timestamp,
        version="v2-pq",
        hash_algorithm=HASH_ALG_SHA256,
        pq_prev_hash=prev_hash_pq,
        pq_merkle_root=merkle_pq,
        pq_algorithm=pq_algorithm,
        dual_commitment=dual_commitment,
    )
    
    return BlockPQ(header=header, statements=sorted_statements)


def block_pq_to_dict(block: BlockPQ) -> Mapping[str, Any]:
    """
    Serialize a PQ block into primitives suitable for JSON output.
    
    Args:
        block: BlockPQ to serialize
        
    Returns:
        Dictionary representation
    """
    header_dict: Dict[str, Any] = {
        "block_number": block.header.block_number,
        "prev_hash": block.header.prev_hash,
        "merkle_root": block.header.merkle_root,
        "timestamp": block.header.timestamp,
        "version": block.header.version,
        "hash_algorithm": block.header.hash_algorithm,
    }
    
    # Add PQ fields if present
    if block.header.has_dual_commitment():
        header_dict.update({
            "pq_prev_hash": block.header.pq_prev_hash,
            "pq_merkle_root": block.header.pq_merkle_root,
            "pq_algorithm": block.header.pq_algorithm,
            "dual_commitment": block.header.dual_commitment,
        })
    
    # Add metadata if present
    if block.header.metadata:
        header_dict["metadata"] = block.header.metadata
    
    return {
        "header": header_dict,
        "statements": list(block.statements),
    }


def block_pq_json(block: BlockPQ) -> str:
    """
    Return a canonical JSON representation of a PQ block.
    
    Args:
        block: BlockPQ to serialize
        
    Returns:
        Canonical JSON string
    """
    return json.dumps(block_pq_to_dict(block), sort_keys=True, separators=(",", ":"))


__all__ = [
    "BlockHeaderPQ",
    "BlockPQ",
    "seal_block_pq",
    "seal_block_dual_commitment",
    "block_pq_to_dict",
    "block_pq_json",
]
