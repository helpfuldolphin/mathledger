# Export Tool Testing Guide

## Implementation Summary

The `--export` flag has been successfully implemented in `backend/tools/export_fol_ab.py` with the following features:

### Key Features
1. **Linter-First**: Always validates V1 schema before processing
2. **--export Flag**: New codepath for database UPSERT operations
3. **Batching**: Processes 1000 records per batch for performance
4. **UPSERT**: Uses `ON CONFLICT (hash) DO NOTHING` for idempotency
5. **Rollback**: Transaction rollback on database errors
6. **Separate from --dry-run**: Mutually exclusive flags

### Schema Mapping
V1 JSONL → PostgreSQL `statements` table:
- `hash` (string) → SHA-256 hash (bytea)
- `content_norm` → `content_norm` (text)
- `is_axiom` → `is_axiom` (boolean)
- `theory_id` → Mapped to Propositional theory UUID
- `status` → 'proven' (if axiom) or 'unknown'
- `derivation_rule` → 'axiom' (if is_axiom) or NULL
- `derivation_depth` → 0 (if axiom) or NULL

## Prerequisites

1. **Docker PostgreSQL running** on port 5432:
   ```bash
   docker-compose up -d postgres
   ```

2. **Database migrated**:
   ```bash
   python run_all_migrations.py
   ```

3. **psycopg installed**:
   ```bash
   pip install psycopg[binary]
   ```

## Testing Commands

### 1. Dry-Run (Validation Only)
```bash
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --dry-run
```

**Expected Output**:
```
DRY-RUN ok: artifacts\sample_v1.jsonl (v1=5)
```

### 2. Export to Database
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --export
```

**Expected Output**:
```
EXPORT ok: 5 records inserted (skipped 0 duplicates)
```

### 3. Verify Insertion
```bash
docker exec infra-postgres-1 psql -U ml -d mathledger -c "
  SELECT content_norm, is_axiom
  FROM statements
  WHERE content_norm LIKE '%p -> p%'
  LIMIT 5;
"
```

### 4. Test Idempotency (Run Twice)
```bash
# First run
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --export

# Second run (should skip duplicates)
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --export
```

**Expected Second Output**:
```
EXPORT ok: 0 records inserted (skipped 5 duplicates)
```

### 5. Run Smoke Tests
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5432/mathledger"
pytest tests/qa/test_exporter_v1_export.py -v
```

**Expected**: 3 tests passing
- `test_export_smoke` - Basic export and verification
- `test_export_idempotent` - Duplicate handling
- `test_export_with_dry_run_error` - Mutual exclusivity

## Sample Data File

`artifacts/sample_v1.jsonl` contains 5 test records:
```jsonl
{"id": 1, "theory_id": 1, "hash": "abc123def456", "content_norm": "p -> p", "is_axiom": false}
{"id": 2, "theory_id": 1, "hash": "def456ghi789", "content_norm": "(p /\\ q) -> p", "is_axiom": false}
{"id": 3, "theory_id": 1, "hash": "ghi789jkl012", "content_norm": "p -> (q -> p)", "is_axiom": true}
{"id": 4, "theory_id": 1, "hash": "jkl012mno345", "content_norm": "(p -> q) -> ((q -> r) -> (p -> r))", "is_axiom": false}
{"id": 5, "theory_id": 1, "hash": "mno345pqr678", "content_norm": "~~p -> p", "is_axiom": false}
```

## Error Handling

### Missing DATABASE_URL
```bash
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --export
```
**Output**: `error: DATABASE_URL environment variable not set`

### psycopg Not Installed
**Output**: `error: psycopg not installed (pip install psycopg[binary])`

### Mutual Exclusivity
```bash
python backend/tools/export_fol_ab.py --input artifacts/sample_v1.jsonl --dry-run --export
```
**Output**: `error: Cannot use --dry-run and --export together`

### Database Connection Failed
**Output**: `error: Database error: <details>`

## Implementation Details

### Function Signatures

```python
def export_to_db(file_path: Path, schema_counts: Dict[str, int], batch_size: int = 1000) -> Tuple[bool, str, int]:
    """Export V1 records to database with batching and transaction rollback."""

def _insert_batch(cur, batch: List[Dict[str, Any]]) -> int:
    """Insert a batch using UPSERT (ON CONFLICT DO NOTHING)."""
```

### Transaction Flow
1. Connect to PostgreSQL
2. Get Propositional theory UUID
3. Read JSONL file line by line
4. Build batches of 1000 records
5. Execute `INSERT ... ON CONFLICT (hash) DO NOTHING`
6. Commit transaction (or rollback on error)

### Message Prefixes
- `EXPORT ok:` - Successful export with counts
- `error:` - Fatal errors (DB connection, missing env var, etc.)

## Next Steps

1. **Start PostgreSQL**: `docker-compose up -d postgres`
2. **Run migrations**: `python run_all_migrations.py`
3. **Test dry-run**: Verify linter works
4. **Test export**: Verify records inserted
5. **Test idempotency**: Run twice, verify no duplicates
6. **Run smoke tests**: `pytest tests/qa/test_exporter_v1_export.py -v`

## Files Modified

- `backend/tools/export_fol_ab.py` - Main implementation (+125 lines)
- `artifacts/sample_v1.jsonl` - Test data (new file)
- `tests/qa/test_exporter_v1_export.py` - Integration tests (new file)
- `TEST_EXPORT.md` - This documentation (new file)
