"""
Schema validator for proof_parents table.

Provides a pre-flight check that the proof_parents table exposes the expected
columns before DAG tests or repository operations run. This allows tests to
fail fast with a clear message rather than relying on *_unverified sentinel
issues at runtime.

Expected Columns (Full Schema):
    - proof_id UUID (nullable if schema only stores hashes)
    - child_statement_id UUID
    - child_hash TEXT
    - parent_statement_id UUID
    - parent_hash TEXT
    - edge_index INT DEFAULT 0
    - created_at TIMESTAMPTZ

Minimal Required Columns:
    - At least one of: child_statement_id, child_hash, or proof_id
    - At least one of: parent_statement_id or parent_hash
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set


@dataclass(frozen=True)
class SchemaValidationResult:
    """Outcome of schema validation."""

    valid: bool
    missing_columns: FrozenSet[str]
    extra_columns: FrozenSet[str]
    issues: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "missing_columns": sorted(self.missing_columns),
            "extra_columns": sorted(self.extra_columns),
            "issues": self.issues,
        }


# Columns required for full DAG functionality
FULL_SCHEMA_COLUMNS: FrozenSet[str] = frozenset(
    {
        "proof_id",
        "child_statement_id",
        "child_hash",
        "parent_statement_id",
        "parent_hash",
        "edge_index",
        "created_at",
    }
)

# Minimal columns to satisfy ProofDagRepository constraints
MINIMAL_CHILD_COLUMNS: FrozenSet[str] = frozenset(
    {"child_statement_id", "child_hash", "proof_id"}
)
MINIMAL_PARENT_COLUMNS: FrozenSet[str] = frozenset(
    {"parent_statement_id", "parent_hash"}
)


def _get_table_columns(cur, table: str) -> Set[str]:
    """Introspect column names for a table."""
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {row[0].lower() for row in cur.fetchall()}


def validate_proof_parents_schema(
    cur,
    *,
    require_full_schema: bool = False,
) -> SchemaValidationResult:
    """
    Validate that the proof_parents table has the expected columns.

    Args:
        cur: Database cursor
        require_full_schema: If True, require all columns in FULL_SCHEMA_COLUMNS.
                             If False, only require minimal child/parent identifiers.

    Returns:
        SchemaValidationResult with validation outcome
    """
    actual_columns = _get_table_columns(cur, "proof_parents")
    issues: Dict[str, str] = {}

    if not actual_columns:
        return SchemaValidationResult(
            valid=False,
            missing_columns=FULL_SCHEMA_COLUMNS if require_full_schema else frozenset(),
            extra_columns=frozenset(),
            issues={"table_missing": "proof_parents table does not exist"},
        )

    if require_full_schema:
        missing = FULL_SCHEMA_COLUMNS - actual_columns
        extra = actual_columns - FULL_SCHEMA_COLUMNS
        valid = not missing
        if missing:
            issues["missing_columns"] = f"Missing columns: {sorted(missing)}"
        return SchemaValidationResult(
            valid=valid,
            missing_columns=frozenset(missing),
            extra_columns=frozenset(extra),
            issues=issues,
        )

    # Minimal validation
    has_child_identifier = bool(actual_columns & MINIMAL_CHILD_COLUMNS)
    has_parent_identifier = bool(actual_columns & MINIMAL_PARENT_COLUMNS)

    if not has_child_identifier:
        issues["no_child_identifier"] = (
            f"proof_parents must expose at least one of: {sorted(MINIMAL_CHILD_COLUMNS)}"
        )
    if not has_parent_identifier:
        issues["no_parent_identifier"] = (
            f"proof_parents must expose at least one of: {sorted(MINIMAL_PARENT_COLUMNS)}"
        )

    valid = has_child_identifier and has_parent_identifier
    missing = FULL_SCHEMA_COLUMNS - actual_columns
    extra = actual_columns - FULL_SCHEMA_COLUMNS

    return SchemaValidationResult(
        valid=valid,
        missing_columns=frozenset(missing),
        extra_columns=frozenset(extra),
        issues=issues,
    )


def require_proof_parents_schema(
    cur,
    *,
    require_full_schema: bool = False,
) -> None:
    """
    Validate proof_parents schema and raise if invalid.

    Args:
        cur: Database cursor
        require_full_schema: If True, require all columns in FULL_SCHEMA_COLUMNS.

    Raises:
        RuntimeError: If schema validation fails
    """
    result = validate_proof_parents_schema(cur, require_full_schema=require_full_schema)
    if not result.valid:
        raise RuntimeError(
            f"proof_parents schema validation failed: {result.issues}"
        )

