"""
Governance Module
=================

Provides governance primitives for MathLedger:
- Commitment registry management
- Registry hashing for manifest binding
- Artifact classification (artifact_kind enum)
"""

from governance.registry_hash import (
    compute_registry_hash,
    load_registry,
    get_registry_version,
    canonicalize_json,
)

# Artifact kind enum values
ARTIFACT_KIND_VERIFIED = "VERIFIED"
ARTIFACT_KIND_REFUTED = "REFUTED"
ARTIFACT_KIND_ABSTAINED = "ABSTAINED"
ARTIFACT_KIND_INADMISSIBLE_UPDATE = "INADMISSIBLE_UPDATE"

VALID_ARTIFACT_KINDS = frozenset({
    ARTIFACT_KIND_VERIFIED,
    ARTIFACT_KIND_REFUTED,
    ARTIFACT_KIND_ABSTAINED,
    ARTIFACT_KIND_INADMISSIBLE_UPDATE,
})


def validate_artifact_kind(kind: str) -> bool:
    """
    Validate that an artifact_kind value is valid.

    Args:
        kind: Artifact kind string to validate

    Returns:
        True if valid, False otherwise
    """
    return kind in VALID_ARTIFACT_KINDS


__all__ = [
    "compute_registry_hash",
    "load_registry",
    "get_registry_version",
    "canonicalize_json",
    "validate_artifact_kind",
    "ARTIFACT_KIND_VERIFIED",
    "ARTIFACT_KIND_REFUTED",
    "ARTIFACT_KIND_ABSTAINED",
    "ARTIFACT_KIND_INADMISSIBLE_UPDATE",
    "VALID_ARTIFACT_KINDS",
]
