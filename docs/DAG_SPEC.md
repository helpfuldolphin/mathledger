# Proof DAG Specification

This document specifies the invariants, schema requirements, and validation
rules for the MathLedger proof dependency graph (DAG).

## Overview

The Proof DAG tracks parent-child relationships between statements in the
derivation tree. Each edge represents a dependency: a child statement was
derived using one or more parent statements as premises.

## Schema Requirements

### proof_parents Table

The `proof_parents` table stores DAG edges. The following columns are expected:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `proof_id` | UUID | Conditional | References the proof that created this edge |
| `child_statement_id` | UUID | Conditional | ID of the derived statement |
| `child_hash` | TEXT | Conditional | SHA-256 hash of the derived statement |
| `parent_statement_id` | UUID | Conditional | ID of the parent/premise statement |
| `parent_hash` | TEXT | Conditional | SHA-256 hash of the parent statement |
| `edge_index` | INT | Recommended | Ordering of parents within a proof (0-indexed) |
| `created_at` | TIMESTAMPTZ | Recommended | When the edge was recorded |

### Minimal Requirements

At minimum, the schema must provide:

1. **Child Identifier**: At least one of `child_statement_id`, `child_hash`, or `proof_id`
2. **Parent Identifier**: At least one of `parent_statement_id` or `parent_hash`

If `proof_id` is used without `child_statement_id`, the `proofs` table must
expose a `statement_id` column to resolve the child.

### Full Schema (Recommended)

For complete DAG functionality including all validation checks:

```sql
CREATE TABLE IF NOT EXISTS proof_parents (
    proof_id UUID,
    child_statement_id UUID,
    child_hash TEXT,
    parent_statement_id UUID,
    parent_hash TEXT,
    edge_index INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## DAG Invariants

The following invariants must hold for a valid proof DAG:

### 1. Acyclicity (No Cycles)

**Invariant**: The DAG must be acyclic. No statement can depend on itself
through any chain of derivations.

**Rationale**: Cycles would indicate circular reasoning, which is logically
invalid. A statement cannot be used to prove itself.

**Detection**: Kahn's algorithm (topological sort) identifies nodes with
non-zero in-degree after processing.

### 2. No Self-Loops

**Invariant**: No edge may have the same child and parent.

**Rationale**: A statement cannot be its own parent. This is a degenerate
case of a cycle.

**Detection**: Check `child_statement_id == parent_statement_id` or
`child_hash == parent_hash` for each edge.

### 3. No Duplicate Edges

**Invariant**: Each `(proof_id, child, parent)` triple must be unique.

**Rationale**: Recording the same dependency multiple times indicates a
bug in the derivation pipeline or data corruption.

**Detection**: Group edges by `(proof_id, child_key, parent_key)` and
flag groups with count > 1.

### 4. Hash/ID Consistency

**Invariant**: If a statement ID appears in multiple edges, it must always
map to the same hash.

**Rationale**: Inconsistent hash/ID mappings indicate data corruption or
hash collision.

**Detection**: Build a map of `statement_id -> hash` and flag conflicts.

### 5. Complete Edges

**Invariant**: Every edge must have resolvable child and parent identifiers.

**Rationale**: Incomplete edges cannot be validated for integrity and
break lineage queries.

**Detection**: Flag edges where both `child_statement_id` and `child_hash`
are NULL, or both `parent_statement_id` and `parent_hash` are NULL.

### 6. Edge Index Ordering

**Invariant**: For each proof, `edge_index` values should be sequential
integers starting from 0.

**Rationale**: Edge indices provide deterministic ordering of dependencies
for proof reconstruction.

**Detection**: For each `proof_id`, collect `edge_index` values and verify
they form the sequence `[0, 1, 2, ...]`.

### 7. Referential Integrity

**Invariant**: All `parent_statement_id` and `child_statement_id` values
must reference existing statements.

**Rationale**: Dangling references indicate incomplete data or failed
transactions.

**Detection**: LEFT JOIN to `statements` table and count NULL results.

### 8. Proof Referential Integrity

**Invariant**: All `proof_id` values must reference existing proofs.

**Rationale**: Orphaned edges without corresponding proofs indicate
incomplete transactions.

**Detection**: LEFT JOIN to `proofs` table and count NULL results.

## Validation Levels

### Level 1: In-Memory Validation (ProofDag.validate)

Checks invariants 1-5 using only the loaded edge data:
- Cycle detection
- Self-loop detection
- Duplicate edge detection
- Hash/ID consistency
- Edge completeness

### Level 2: Database Validation (ProofDagRepository.validate)

Adds database-level checks for invariants 6-8:
- Edge index ordering (if `edge_index` column exists)
- Missing parent statements
- Missing child statements
- Missing proofs

### Level 3: Organism Lineage Validation

For First Organism tests, additional checks:
- Lineage completeness (expected parents present)
- Ancestor chain verification
- Descendant chain verification

## Sentinel Issues

When the schema lacks columns needed for a validation check, the validator
emits `*_unverified` sentinel issues instead of silently passing:

| Sentinel | Meaning |
|----------|---------|
| `missing_parents_unverified` | Cannot verify parent references |
| `missing_children_unverified` | Cannot verify child references |
| `missing_proofs_unverified` | Cannot verify proof references |
| `duplicate_edges_unverified` | Cannot detect duplicate edges |

## Usage

### Schema Validation (Pre-Flight Check)

```python
from backend.dag import require_proof_parents_schema

with connection.cursor() as cur:
    # Raises RuntimeError if schema is invalid
    require_proof_parents_schema(cur, require_full_schema=True)
```

### DAG Validation

```python
from backend.dag import ProofDagRepository

with connection.cursor() as cur:
    repo = ProofDagRepository(cur)
    report = repo.validate()
    
    if not report.ok:
        print(f"Issues: {report.issues}")
```

### Organism Lineage Validation

```python
from tests.helpers.dag_assertions import validate_organism_lineage_from_db

with connection.cursor() as cur:
    report = validate_organism_lineage_from_db(
        cur,
        organism_hash="abc123...",
        expected_parent_hashes={"parent1_hash", "parent2_hash"},
        proof_id=42,
    )
    
    if not report.ok:
        for name, inv in report.invariants.items():
            if not inv.passed:
                print(f"  {name}: {inv.message}")
```

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| `tests/test_proof_dag.py` | Unit tests for in-memory DAG operations |
| `tests/integration/test_first_organism_dag.py` | Integration tests with real Postgres |
| `tests/helpers/dag_assertions.py` | Reusable assertion helpers |

## Migration Compatibility

When modifying the `proof_parents` schema:

1. **Adding columns**: Safe. New columns are detected automatically.
2. **Removing columns**: May break validation. Check for `*_unverified` sentinels.
3. **Renaming columns**: Breaks detection. Update `FULL_SCHEMA_COLUMNS` constant.
4. **Changing types**: May affect coercion. Test with real data.

## Related Files

- `backend/dag/proof_dag.py` - Core DAG implementation
- `backend/dag/schema_validator.py` - Schema validation helpers
- `backend/axiom_engine/derive_utils.py` - `record_proof_edge` function
- `migrations/baseline_20251019.sql` - Schema definition

