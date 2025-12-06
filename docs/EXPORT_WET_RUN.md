# Export Tool: Wet-Run Implementation

## [POA]+[ASD]: Proof-Abstain Discipline + Authentic Synthetic Data

This document describes the `--export` wet-run implementation for `backend/tools/export_fol_ab.py`.

## Overview

The exporter now supports three modes:
1. **--dry-run**: Validation-only (no DB/network) - idempotent
2. **--export**: Database UPSERT with batching - wet-run
3. **No flags**: Informational mode (shows guidance)

## Architecture

### Linter-First Design
```
User input → lint_metrics_v1() → [FAIL] → error: ...
                                → [PASS] → --dry-run? → DRY-RUN ok: ...
                                        → --export? → export_to_db() → EXPORT ok: ...
```

### SHA-256 Hash Normalization

**Key decision**: Hash `content_norm` field, not V1's `hash` field.

**Rationale**:
- V1 JSONL `hash` field is user-provided, untrusted
- `content_norm` is the actual statement text
- Hashing content ensures idempotent UPSERT based on semantics

```python
# Export logic
content_hash = hashlib.sha256(record['content_norm'].encode('utf-8')).digest()
# INSERT ... VALUES (..., content_hash, ...) ON CONFLICT (hash) DO NOTHING
```

### UPSERT Semantics

```sql
INSERT INTO statements (theory_id, hash, content_norm, is_axiom, status, derivation_rule, derivation_depth)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (hash) DO NOTHING
RETURNING id
```

**Properties**:
- **Idempotent**: Running twice on same data inserts 0 on second run
- **Atomic**: All batches in single transaction (commit or rollback)
- **Accurate counting**: RETURNING clause tracks actual insertions

### Batch Processing

```
File: 5000 records
├─ Batch 1: records 1-1000 → INSERT (997 inserted, 3 duplicates)
├─ Batch 2: records 1001-2000 → INSERT (1000 inserted, 0 duplicates)
├─ Batch 3: records 2001-3000 → INSERT (1000 inserted, 0 duplicates)
├─ Batch 4: records 3001-4000 → INSERT (1000 inserted, 0 duplicates)
└─ Batch 5: records 4001-5000 → INSERT (1000 inserted, 0 duplicates)
                                 ↓
                          COMMIT transaction
                                 ↓
                    EXPORT ok: 4997 records inserted (skipped 3 duplicates)
```

**Performance**: O(1000) network round-trips instead of O(n)

## Usage

### Prerequisites

1. **PostgreSQL running**:
   ```bash
   docker-compose up -d postgres
   ```

2. **Migrations applied**:
   ```bash
   python run_all_migrations.py
   ```

3. **psycopg installed**:
   ```bash
   pip install psycopg[binary]
   ```

### Commands

**Validation (dry-run)**:
```bash
python backend/tools/export_fol_ab.py --input data.jsonl --dry-run
# Output: DRY-RUN ok: data.jsonl (v1=5000)
```

**Export to database**:
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
python backend/tools/export_fol_ab.py --input data.jsonl --export
# Output: EXPORT ok: 5000 records inserted (skipped 0 duplicates)
```

**Idempotency test** (run twice):
```bash
# First run
python backend/tools/export_fol_ab.py --input data.jsonl --export
# EXPORT ok: 5000 records inserted (skipped 0 duplicates)

# Second run (same data)
python backend/tools/export_fol_ab.py --input data.jsonl --export
# EXPORT ok: 0 records inserted (skipped 5000 duplicates)
```

## Schema Mapping

### V1 JSONL → PostgreSQL `statements`

| V1 Field | DB Column | Transformation |
|----------|-----------|----------------|
| `content_norm` | `hash` | SHA-256 digest (bytea) |
| `content_norm` | `content_norm` | Direct copy (text) |
| `is_axiom` | `is_axiom` | Direct copy (boolean) |
| `theory_id` (ignored) | `theory_id` | Mapped to Propositional UUID |
| `is_axiom=true` | `status` | `'proven'` |
| `is_axiom=false` | `status` | `'unknown'` |
| `is_axiom=true` | `derivation_rule` | `'axiom'` |
| `is_axiom=false` | `derivation_rule` | `NULL` |
| `is_axiom=true` | `derivation_depth` | `0` |
| `is_axiom=false` | `derivation_depth` | `NULL` |

### Example

**Input** (V1 JSONL):
```json
{"id": 1, "theory_id": 1, "hash": "abc123", "content_norm": "p -> p", "is_axiom": false}
```

**Output** (PostgreSQL):
```sql
INSERT INTO statements (
    theory_id,  -- UUID of Propositional theory
    hash,       -- SHA-256("p -> p") as bytea
    content_norm, -- "p -> p"
    is_axiom,   -- false
    status,     -- 'unknown'
    derivation_rule, -- NULL
    derivation_depth -- NULL
) VALUES (...)
```

## Error Handling

### Missing DATABASE_URL
```bash
$ python backend/tools/export_fol_ab.py --input data.jsonl --export
error: DATABASE_URL environment variable not set
```

### psycopg Not Installed
```bash
error: psycopg not installed (pip install psycopg[binary])
```

### Database Connection Failed
```bash
error: Database connection failed: connection refused (is PostgreSQL running?)
```

### Theory Not Found
```bash
error: Propositional theory not found in database (run migrations)
```

### Transaction Rollback
If ANY batch fails, the entire transaction rolls back (no partial imports).

## Testing

### Integration Tests

**Location**: `tests/qa/test_exporter_v1_export.py`

**Run**:
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
pytest tests/qa/test_exporter_v1_export.py -v
```

**Tests**:
1. `test_export_smoke`: Basic export and DB verification
2. `test_export_idempotent`: Verifies UPSERT prevents duplicates
3. `test_export_with_dry_run_error`: Validates mutual exclusivity

**Pytest marker**:
```python
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set (integration test)"
)
```

Tests are automatically skipped if DATABASE_URL not set.

### Verification Queries

**Count inserted records**:
```sql
SELECT COUNT(*) FROM statements WHERE content_norm LIKE '%test%';
```

**Verify specific record**:
```sql
SELECT content_norm, is_axiom, status, derivation_rule
FROM statements
WHERE hash = digest('p -> p', 'sha256');
```

**Check for duplicates** (should be 0):
```sql
SELECT hash, COUNT(*)
FROM statements
GROUP BY hash
HAVING COUNT(*) > 1;
```

## Implementation Details

### Function Signatures

```python
def export_to_db(
    file_path: Path,
    schema_counts: Dict[str, int],
    batch_size: int = 1000
) -> Tuple[bool, str, int]:
    """Returns (success, message, inserted_count)"""

def _insert_batch(
    cur,
    batch: List[Dict[str, Any]]
) -> int:
    """Returns number of records actually inserted"""
```

### Transaction Flow

1. **Connect** to PostgreSQL via DATABASE_URL
2. **Fetch** Propositional theory UUID
3. **Stream** JSONL file line-by-line (memory efficient)
4. **Build** batches of 1000 records
5. **Execute** UPSERT for each batch
6. **Count** insertions via RETURNING clause
7. **Commit** transaction (or rollback on error)
8. **Report** results with standardized prefix

### Message Prefixes

All output follows standardized prefixes:

- `EXPORT ok:` - Successful export
- `error:` - Fatal errors
- `info:` - Informational messages (no-flag mode)

## Performance Characteristics

### Benchmarks

**5000 records** (estimated):
- Dry-run: ~0.1s (parse-only)
- Export: ~1-2s (batched UPSERT)
- Export (all duplicates): ~0.5s (UPSERT fast-path)

**Batch size impact**:
- 100: ~10s (50 round-trips)
- 1000: ~1s (5 round-trips) ← **optimal**
- 10000: ~1.5s (1 round-trip, but larger transaction log)

### Scalability

- **100K records**: ~20s
- **1M records**: ~200s (~5K records/sec)
- **Memory**: O(batch_size) = O(1000) = ~1MB

## Limitations

1. **Theory hardcoded**: Only supports Propositional theory
2. **No progress reporting**: Silent during long imports
3. **No resume**: Crash loses all progress (transaction rollback)
4. **No concurrent imports**: No advisory locks

## Future Enhancements

1. **Multiple theories**: Support `--theory-name` CLI arg
2. **Progress bar**: Report every 10K records
3. **Checkpoint/resume**: Save progress every N batches
4. **Concurrent safety**: Use PostgreSQL advisory locks
5. **V2 schema support**: Auto-detect and route to v2 handler

## References

- **Codebase**: `backend/tools/export_fol_ab.py`
- **Tests**: `tests/qa/test_exporter_v1_export.py`
- **Sample data**: `artifacts/sample_v1.jsonl`
- **Migrations**: `migrations/001_init.sql`, `migrations/002_add_axioms.sql`
