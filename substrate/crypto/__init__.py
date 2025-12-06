"""Substrate cryptographic primitives."""

from .core import rfc8785_canonicalize
from .hashing import (
    DOMAIN_ROOT,
    DOMAIN_STMT,
    compute_merkle_proof,
    hash_block,
    hash_statement,
    merkle_root,
    sha256_hex,
)

__all__ = [
    "DOMAIN_ROOT",
    "DOMAIN_STMT",
    "compute_merkle_proof",
    "hash_block",
    "hash_statement",
    "merkle_root",
    "rfc8785_canonicalize",
    "sha256_hex",
]

