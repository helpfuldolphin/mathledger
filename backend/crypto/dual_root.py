"""
Deprecated dual-root shim.

All functionality has been moved to attestation.dual_root.
This module re-exports for backwards compatibility.
"""

import warnings

from attestation.dual_root import (  # noqa: F401
    AttestationLeaf,
    AttestationTree,
    build_reasoning_attestation,
    build_ui_attestation,
    canonicalize_reasoning_artifact,
    canonicalize_ui_artifact,
    compute_composite_root,
    compute_reasoning_root,
    compute_ui_root,
    generate_attestation_metadata,
    hash_reasoning_leaf,
    hash_ui_leaf,
    verify_composite_integrity,
)

warnings.warn(
    "backend.crypto.dual_root is deprecated; import from attestation.dual_root instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AttestationLeaf",
    "AttestationTree",
    "build_reasoning_attestation",
    "build_ui_attestation",
    "canonicalize_reasoning_artifact",
    "canonicalize_ui_artifact",
    "compute_composite_root",
    "compute_reasoning_root",
    "compute_ui_root",
    "generate_attestation_metadata",
    "hash_reasoning_leaf",
    "hash_ui_leaf",
    "verify_composite_integrity",
]
