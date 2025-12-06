"""Hashing primitives for the canonical basis."""

from .hash import (
    compute_merkle_proof,
    hash_block,
    hash_statement,
    merkle_root,
    reasoning_root,
    sha256_hex,
    ui_root,
    verify_merkle_proof,
)

__all__ = [
    "sha256_hex",
    "hash_statement",
    "hash_block",
    "merkle_root",
    "compute_merkle_proof",
    "verify_merkle_proof",
    "reasoning_root",
    "ui_root",
]


