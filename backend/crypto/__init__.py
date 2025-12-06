"""
MathLedger Cryptographic Primitives Module

Centralized cryptographic operations with domain separation and security best practices.
"""

from backend.crypto.hashing import (
    sha256_hex,
    sha256_bytes,
    merkle_root,
    hash_statement,
    hash_block,
)

from backend.crypto.core import (
    rfc8785_canonicalize,
    sha256_hex_concat,
    ed25519_generate_keypair,
    ed25519_sign_b64,
    ed25519_verify_b64,
)

__all__ = [
    "sha256_hex",
    "sha256_bytes",
    "merkle_root",
    "hash_statement",
    "hash_block",
    "rfc8785_canonicalize",
    "sha256_hex_concat",
    "ed25519_generate_keypair",
    "ed25519_sign_b64",
    "ed25519_verify_b64",
]
