"""
Schema Version Constants for MathLedger Canonicalization
========================================================

This module defines version constants for all canonicalization and hashing
operations in MathLedger. These versions enable:

1. Backward compatibility when algorithms change
2. Audit trail for which version was used
3. Migration paths between schema versions
4. Deterministic replay of historical operations

Version Format: "vMAJOR.MINOR.PATCH"
- MAJOR: Breaking changes to canonical output
- MINOR: Backward-compatible additions
- PATCH: Bug fixes that don't change output

Usage:
    from normalization.schema_version import CANON_SCHEMA_VERSION
    
    def hash_statement_versioned(statement: str) -> dict:
        canonical = canonical_bytes(statement)
        return {
            "hash": sha256_hex(canonical, domain=DOMAIN_STMT),
            "schema_version": CANON_SCHEMA_VERSION,
            "algorithm": HASH_ALGORITHM_VERSION
        }
"""

# String-based canonicalization version (normalization/canon.py)
CANON_SCHEMA_VERSION = "v1.0.0"

# AST-based canonicalization version (normalization/ast_canon.py)
AST_CANON_SCHEMA_VERSION = "v1.0.0"

# Hash algorithm version (backend/crypto/hashing.py)
HASH_ALGORITHM_VERSION = "sha256-domain-sep-v1"

# JSON canonicalization version (backend/basis/canon.py)
JSON_CANON_SCHEMA_VERSION = "rfc8785-v1"

# Merkle tree construction version
MERKLE_SCHEMA_VERSION = "v1.0.0"

# Dual-root attestation version
ATTESTATION_SCHEMA_VERSION = "v2.0.0"

# Version metadata for embedding in canonical output
VERSION_METADATA = {
    "canon_schema": CANON_SCHEMA_VERSION,
    "ast_canon_schema": AST_CANON_SCHEMA_VERSION,
    "hash_algorithm": HASH_ALGORITHM_VERSION,
    "json_canon": JSON_CANON_SCHEMA_VERSION,
    "merkle_schema": MERKLE_SCHEMA_VERSION,
    "attestation_schema": ATTESTATION_SCHEMA_VERSION,
}


def get_version_string() -> str:
    """Get a compact version string for logging."""
    return f"canon:{CANON_SCHEMA_VERSION},hash:{HASH_ALGORITHM_VERSION}"


def get_full_version_metadata() -> dict:
    """Get complete version metadata for embedding in artifacts."""
    return VERSION_METADATA.copy()


# Changelog
CHANGELOG = """
v1.0.0 (2025-12-06)
-------------------
- Initial schema version tracking
- String-based canonicalization (normalization/canon.py)
- AST-based canonicalization (normalization/ast_canon.py)
- SHA-256 with domain separation (backend/crypto/hashing.py)
- RFC 8785 JSON canonicalization (backend/basis/canon.py)
- Merkle tree construction with sorted leaves
- Dual-root attestation (H_t = SHA256(R_t || U_t))
"""
