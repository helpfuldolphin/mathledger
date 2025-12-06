"""
MathLedger canonical basis package.

This package exposes the distilled, deterministic primitives extracted from the
legacy spanning set. Importers should depend on this package instead of the
older `backend.*` modules when building new functionality.
"""

from basis.attestation.dual import (
    attestation_from_block,
    build_attestation,
    composite_root,
    reasoning_root,
    ui_root,
    verify_attestation,
)
from basis.core import (
    Block,
    BlockHeader,
    CurriculumIndex,
    CurriculumTier,
    DualAttestation,
    HexDigest,
    NormalizedFormula,
)
from basis.crypto.hash import (
    compute_merkle_proof,
    hash_block,
    hash_statement,
    merkle_root,
    reasoning_root as merkle_reasoning_root,
    sha256_hex,
    ui_root as merkle_ui_root,
    verify_merkle_proof,
)
from basis.curriculum.ladder import (
    CurriculumLadder,
    ladder_from_dict,
    ladder_from_json,
    ladder_to_json,
)
from basis.ledger.block import block_json, block_to_dict, seal_block
from basis.logic.normalizer import (
    are_equivalent,
    atoms,
    normalize,
    normalize_many,
    normalize_pretty,
)

__all__ = [
    # Core types
    "Block",
    "BlockHeader",
    "CurriculumIndex",
    "CurriculumLadder",
    "CurriculumTier",
    "DualAttestation",
    "HexDigest",
    "NormalizedFormula",
    # Logic
    "normalize",
    "normalize_pretty",
    "normalize_many",
    "are_equivalent",
    "atoms",
    # Crypto
    "sha256_hex",
    "hash_statement",
    "hash_block",
    "merkle_root",
    "compute_merkle_proof",
    "verify_merkle_proof",
    "merkle_reasoning_root",
    "merkle_ui_root",
    # Ledger
    "seal_block",
    "block_to_dict",
    "block_json",
    # Attestation
    "attestation_from_block",
    "reasoning_root",
    "ui_root",
    "composite_root",
    "build_attestation",
    "verify_attestation",
    # Curriculum
    "ladder_from_dict",
    "ladder_from_json",
    "ladder_to_json",
]


