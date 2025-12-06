"""
DEPRECATED: kept only for legacy callers; will be removed after VCP 2.2 Wave 1.

This module re-exports from the canonical substrate.dag namespace.
New code should import directly from substrate.dag instead.
"""

# Re-export from canonical namespace
from substrate.dag import (
    FULL_SCHEMA_COLUMNS,
    MINIMAL_CHILD_COLUMNS,
    MINIMAL_PARENT_COLUMNS,
    ProofDag,
    ProofDagRepository,
    ProofDagValidationReport,
    ProofEdge,
    SchemaValidationResult,
    require_proof_parents_schema,
    validate_proof_parents_schema,
)

__all__ = [
    "FULL_SCHEMA_COLUMNS",
    "MINIMAL_CHILD_COLUMNS",
    "MINIMAL_PARENT_COLUMNS",
    "ProofDag",
    "ProofDagRepository",
    "ProofDagValidationReport",
    "ProofEdge",
    "SchemaValidationResult",
    "require_proof_parents_schema",
    "validate_proof_parents_schema",
]

