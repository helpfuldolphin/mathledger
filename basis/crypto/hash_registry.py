"""
Hash Algorithm Registry for Post-Quantum Migration.

This module provides a centralized registry of hash algorithms used across
different epochs of the MathLedger blockchain. It enables:
- Algorithm versioning and identification
- Epoch-based algorithm resolution
- Algorithm activation and deprecation
- Historical verification with correct algorithms

Security Invariant: Hash algorithms are never baked into data structures;
they are always identified by version tags.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from basis.core import HexDigest


# Hash algorithm version identifiers
HASH_ALG_SHA256 = 0x00      # Current: SHA-256
HASH_ALG_PQ1 = 0x01         # Reserved: First PQ hash (e.g., SHA-3)
HASH_ALG_PQ2 = 0x02         # Reserved: Second PQ hash (e.g., BLAKE3)
HASH_ALG_PQ3 = 0x03         # Reserved: Third PQ hash
# 0x04-0xFF reserved for future algorithms


class HashFunction(Protocol):
    """Protocol for hash function implementations."""
    
    def __call__(self, data: bytes) -> bytes:
        """
        Compute hash digest.
        
        Args:
            data: Input bytes to hash
            
        Returns:
            Hash digest as bytes
        """
        ...


@dataclass(frozen=True)
class HashAlgorithm:
    """
    Metadata for a hash algorithm version.
    
    Attributes:
        algorithm_id: Unique identifier (0x00-0xFF)
        name: Human-readable name
        output_length: Digest size in bytes
        security_bits: Security level in bits (classical)
        pq_security_bits: Post-quantum security level in bits (if applicable)
        implementation: Hash function implementation
        description: Detailed description
    """
    
    algorithm_id: int
    name: str
    output_length: int
    security_bits: int
    pq_security_bits: int
    implementation: HashFunction
    description: str = ""
    
    def __post_init__(self):
        if not (0 <= self.algorithm_id <= 0xFF):
            raise ValueError(f"Algorithm ID must be 0x00-0xFF, got {self.algorithm_id:#04x}")
        if self.output_length <= 0:
            raise ValueError(f"Output length must be positive, got {self.output_length}")


@dataclass(frozen=True)
class HashEpoch:
    """
    Defines which hash algorithm was canonical at a given block range.
    
    Attributes:
        start_block: First block number using this algorithm
        end_block: Last block number using this algorithm (None = ongoing)
        algorithm_id: Hash algorithm identifier
        algorithm_name: Human-readable algorithm name
        description: Migration notes or rationale
    """
    
    start_block: int
    end_block: Optional[int]
    algorithm_id: int
    algorithm_name: str
    description: str = ""
    
    def contains_block(self, block_number: int) -> bool:
        """Check if this epoch contains the given block number."""
        if block_number < self.start_block:
            return False
        if self.end_block is None:
            return True
        return block_number <= self.end_block


# Hash function implementations

def _sha256_impl(data: bytes) -> bytes:
    """SHA-256 implementation."""
    return hashlib.sha256(data).digest()


def _sha3_256_impl(data: bytes) -> bytes:
    """SHA3-256 implementation."""
    return hashlib.sha3_256(data).digest()


def _pq_placeholder_impl(data: bytes) -> bytes:
    """
    Placeholder for future PQ hash algorithm.
    
    Currently returns SHA-256 for testing. This will be replaced
    with actual PQ algorithm implementation.
    """
    # TODO: Replace with actual PQ hash (e.g., BLAKE3)
    return hashlib.sha256(b"PQ-PLACEHOLDER:" + data).digest()


# Global algorithm registry

_ALGORITHMS: Dict[int, HashAlgorithm] = {
    HASH_ALG_SHA256: HashAlgorithm(
        algorithm_id=HASH_ALG_SHA256,
        name="SHA-256",
        output_length=32,
        security_bits=256,
        pq_security_bits=128,  # Grover's algorithm reduces to 128 bits
        implementation=_sha256_impl,
        description="NIST FIPS 180-4 SHA-256. Current canonical algorithm.",
    ),
    HASH_ALG_PQ1: HashAlgorithm(
        algorithm_id=HASH_ALG_PQ1,
        name="SHA3-256",
        output_length=32,
        security_bits=256,
        pq_security_bits=256,  # Quantum-resistant
        implementation=_sha3_256_impl,
        description="NIST FIPS 202 SHA3-256. First post-quantum algorithm.",
    ),
    HASH_ALG_PQ2: HashAlgorithm(
        algorithm_id=HASH_ALG_PQ2,
        name="PQ-Placeholder-2",
        output_length=32,
        security_bits=256,
        pq_security_bits=256,
        implementation=_pq_placeholder_impl,
        description="Placeholder for second PQ algorithm (e.g., BLAKE3).",
    ),
    HASH_ALG_PQ3: HashAlgorithm(
        algorithm_id=HASH_ALG_PQ3,
        name="PQ-Placeholder-3",
        output_length=32,
        security_bits=256,
        pq_security_bits=256,
        implementation=_pq_placeholder_impl,
        description="Placeholder for third PQ algorithm.",
    ),
}


# Global epoch registry
# This defines which algorithm is canonical at each block range

_EPOCHS: List[HashEpoch] = [
    HashEpoch(
        start_block=0,
        end_block=None,  # Ongoing - will be updated when PQ migration occurs
        algorithm_id=HASH_ALG_SHA256,
        algorithm_name="SHA-256",
        description="Genesis epoch. SHA-256 is canonical.",
    ),
    # Future epochs will be added here when PQ migration is activated
    # Example:
    # HashEpoch(
    #     start_block=1000000,
    #     end_block=None,
    #     algorithm_id=HASH_ALG_PQ1,
    #     algorithm_name="SHA3-256",
    #     description="Post-quantum migration. SHA3-256 becomes canonical.",
    # ),
]


# Public API

def get_algorithm(algorithm_id: int) -> HashAlgorithm:
    """
    Get hash algorithm by ID.
    
    Args:
        algorithm_id: Algorithm identifier (0x00-0xFF)
        
    Returns:
        HashAlgorithm metadata
        
    Raises:
        KeyError: If algorithm ID is not registered
    """
    if algorithm_id not in _ALGORITHMS:
        raise KeyError(f"Unknown hash algorithm ID: {algorithm_id:#04x}")
    return _ALGORITHMS[algorithm_id]


def get_algorithm_by_name(name: str) -> HashAlgorithm:
    """
    Get hash algorithm by name.
    
    Args:
        name: Algorithm name (case-sensitive)
        
    Returns:
        HashAlgorithm metadata
        
    Raises:
        KeyError: If algorithm name is not registered
    """
    for alg in _ALGORITHMS.values():
        if alg.name == name:
            return alg
    raise KeyError(f"Unknown hash algorithm name: {name}")


def list_algorithms() -> List[HashAlgorithm]:
    """
    List all registered hash algorithms.
    
    Returns:
        List of HashAlgorithm metadata, sorted by algorithm ID
    """
    return sorted(_ALGORITHMS.values(), key=lambda a: a.algorithm_id)


def get_epoch_for_block(block_number: int) -> HashEpoch:
    """
    Get the hash epoch for a given block number.
    
    Args:
        block_number: Block number to query
        
    Returns:
        HashEpoch defining the canonical algorithm for this block
        
    Raises:
        ValueError: If no epoch contains this block number
    """
    for epoch in _EPOCHS:
        if epoch.contains_block(block_number):
            return epoch
    raise ValueError(f"No hash epoch defined for block {block_number}")


def get_canonical_algorithm(block_number: int) -> HashAlgorithm:
    """
    Get the canonical hash algorithm for a given block number.
    
    Args:
        block_number: Block number to query
        
    Returns:
        HashAlgorithm that was canonical at this block
        
    Raises:
        ValueError: If no epoch contains this block number
    """
    epoch = get_epoch_for_block(block_number)
    return get_algorithm(epoch.algorithm_id)


def list_epochs() -> List[HashEpoch]:
    """
    List all hash epochs.
    
    Returns:
        List of HashEpoch, sorted by start block
    """
    return sorted(_EPOCHS, key=lambda e: e.start_block)


def register_algorithm(algorithm: HashAlgorithm) -> None:
    """
    Register a new hash algorithm.
    
    Args:
        algorithm: HashAlgorithm metadata
        
    Raises:
        ValueError: If algorithm ID is already registered
    """
    if algorithm.algorithm_id in _ALGORITHMS:
        raise ValueError(
            f"Algorithm ID {algorithm.algorithm_id:#04x} already registered "
            f"as {_ALGORITHMS[algorithm.algorithm_id].name}"
        )
    _ALGORITHMS[algorithm.algorithm_id] = algorithm


def register_epoch(epoch: HashEpoch) -> None:
    """
    Register a new hash epoch.
    
    This is used to activate new hash algorithms at specific block numbers.
    
    Args:
        epoch: HashEpoch defining the new epoch
        
    Raises:
        ValueError: If epoch overlaps with existing epochs
    """
    # Verify algorithm exists
    if epoch.algorithm_id not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm ID: {epoch.algorithm_id:#04x}")
    
    # Check for overlaps
    for existing in _EPOCHS:
        # Check if new epoch overlaps with existing
        if existing.end_block is None:
            # Existing epoch is ongoing - close it
            if epoch.start_block <= existing.start_block:
                raise ValueError(
                    f"New epoch (start={epoch.start_block}) cannot start before "
                    f"or at existing ongoing epoch (start={existing.start_block})"
                )
            # Close the existing ongoing epoch
            _EPOCHS.remove(existing)
            closed_epoch = HashEpoch(
                start_block=existing.start_block,
                end_block=epoch.start_block - 1,
                algorithm_id=existing.algorithm_id,
                algorithm_name=existing.algorithm_name,
                description=existing.description,
            )
            _EPOCHS.append(closed_epoch)
        else:
            # Check for overlap with closed epoch
            if (epoch.start_block <= existing.end_block and 
                (epoch.end_block is None or epoch.end_block >= existing.start_block)):
                raise ValueError(
                    f"New epoch ({epoch.start_block}-{epoch.end_block}) overlaps "
                    f"with existing epoch ({existing.start_block}-{existing.end_block})"
                )
    
    _EPOCHS.append(epoch)


__all__ = [
    "HASH_ALG_SHA256",
    "HASH_ALG_PQ1",
    "HASH_ALG_PQ2",
    "HASH_ALG_PQ3",
    "HashFunction",
    "HashAlgorithm",
    "HashEpoch",
    "get_algorithm",
    "get_algorithm_by_name",
    "list_algorithms",
    "get_epoch_for_block",
    "get_canonical_algorithm",
    "list_epochs",
    "register_algorithm",
    "register_epoch",
]
