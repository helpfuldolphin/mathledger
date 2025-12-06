"""
Proof DAG subsystem exports.

Provides:
- ProofEdge dataclass describing parent-child relationships per proof
- ProofDag in-memory structure with lineage queries and validation
- ProofDagRepository for database IO and schema-aware inserts
- Schema validation helpers for pre-flight checks
"""

from .proof_dag import (
    ProofDag,
    ProofDagRepository,
    ProofDagValidationReport,
    ProofEdge,
)
from .schema_validator import (
    FULL_SCHEMA_COLUMNS,
    MINIMAL_CHILD_COLUMNS,
    MINIMAL_PARENT_COLUMNS,
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

