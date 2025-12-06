# Migration Repair Diagnostic Report

**Generated**: 2025-10-19  
**Agent**: Manus G - Systems Mechanic  
**Issue**: 9 failing migrations (PR #21)  
**Status**: CI migrations disabled, blocking green builds

---

## Executive Summary

The MathLedger migration system has **9 documented failing migrations** that have been temporarily disabled in CI (`.github/workflows/ci.yml` line 29-32). Analysis reveals the failures stem from:

1. **Schema evolution conflicts** between migrations
2. **Postgres 15 syntax incompatibilities** (partially fixed)
3. **Type mismatches** in foreign key relationships
4. **Column name inconsistencies** across migrations
5. **Non-idempotent operations** causing re-run failures

**Recommendation**: **Baseline migration approach** - create a single authoritative schema snapshot that supersedes all previous migrations.

---

## Migration Inventory

### Total Migrations: 17 files

```
001_init.sql                        [BASE SCHEMA]
002_add_axioms.sql                  [CONFLICT: duplicate 002]
002_blocks_lemmas.sql               [CONFLICT: duplicate 002]
003_add_system_id.sql               [CONFLICT: duplicate 003]
003_fix_progress_compatibility.sql  [CONFLICT: duplicate 003]
004_finalize_core_schema.sql        [MAJOR: schema consolidation, DROP TABLE]
005_add_search_indexes.sql          [DEPENDS: 004]
006_add_pg_trgm_extension.sql       [CONFLICT: duplicate 006]
006_add_policy_settings.sql         [CONFLICT: duplicate 006]
007_fix_proofs_schema.sql           [PATCH: fixes 004 issues]
008_fix_statements_hash.sql         [PATCH: hash column type]
009_normalize_statements.sql        [PATCH: text normalization]
010_idempotent_normalize.sql        [PATCH: idempotency fix]
011_schema_parity.sql               [PATCH: alignment]
012_blocks_parity.sql               [PATCH: blocks table]
013_runs_logging.sql                [FEATURE: logging]
014_ensure_slug_column.sql          [PATCH: slug column]
```

---

## Critical Issues Identified

### Issue 1: Duplicate Migration Numbers

**Severity**: CRITICAL  
**Impact**: Undefined execution order, race conditions

**Conflicts**:
- `002_add_axioms.sql` vs `002_blocks_lemmas.sql`
- `003_add_system_id.sql` vs `003_fix_progress_compatibility.sql`
- `006_add_pg_trgm_extension.sql` vs `006_add_policy_settings.sql`

**Analysis**:
The migration runner (`scripts/run-migrations.py` line 79) sorts by first numeric prefix. When multiple files share the same number, execution order becomes filesystem-dependent (non-deterministic).

**Evidence**:
```python
migration_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 999)
```

This sorts `002_add_axioms.sql` and `002_blocks_lemmas.sql` arbitrarily.

---

### Issue 2: 004_finalize_core_schema.sql - Destructive DDL

**Severity**: CRITICAL  
**Impact**: Data loss, breaks idempotency

**Problem** (line 121-122):
```sql
-- Drop old blocks table if it exists with different schema
DROP TABLE IF EXISTS blocks CASCADE;
```

**Consequences**:
1. **Data Loss**: Any existing `blocks` data is destroyed
2. **Non-Idempotent**: Re-running migration succeeds but loses data
3. **Cascade Effects**: Foreign keys from other tables break
4. **Migration Order**: If `002_blocks_lemmas.sql` runs first, its `blocks` table gets dropped

**Expected Schema Conflicts**:
- `002_blocks_lemmas.sql` creates `blocks` with certain columns
- `003_fix_progress_compatibility.sql` modifies `blocks` columns
- `004_finalize_core_schema.sql` drops and recreates `blocks` with different schema
- `012_blocks_parity.sql` tries to modify the dropped table

---

### Issue 3: Type Mismatches - statements.hash

**Severity**: HIGH  
**Impact**: Foreign key failures, query errors

**Schema Evolution**:

**001_init.sql** (line 29):
```sql
hash bytea not null unique,  -- SHA-256 of normalized form
```

**004_finalize_core_schema.sql** (line 54):
```sql
hash bytea not null unique,  -- SHA-256 of normalized form
```

**007_fix_proofs_schema.sql** (line 17):
```sql
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';
```

**008_fix_statements_hash.sql**: Attempts to fix hash type inconsistency

**009_normalize_statements.sql** (line 18):
```sql
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;
```

**Problem**: Migration 001 and 004 define `hash` as `bytea`, but 007 and 009 treat it as `TEXT` (hex-encoded). This causes:
- Type conversion errors
- Index mismatches
- Query failures when comparing bytea vs text

---

### Issue 4: Column Name Inconsistencies

**Severity**: MEDIUM  
**Impact**: UPDATE failures, missing data

**content_norm vs text**:

**001_init.sql**: Uses `content_norm`
```sql
content_norm text not null,  -- normalized s-expr / canonical form
```

**009_normalize_statements.sql** (line 8-18): Assumes `text` column exists
```sql
UPDATE statements SET normalized_text =
  REPLACE(
    REPLACE(
      REPLACE(
        REPLACE(text, '→', '->'),  -- FAILS: column "text" does not exist
```

**root_hash vs merkle_root**:

**004_finalize_core_schema.sql**: Uses `root_hash`
```sql
root_hash text not null,  -- Merkle root of statements in this block
```

**003_fix_progress_compatibility.sql**: Tries to migrate `root_hash` to `merkle_root`
```sql
IF NOT EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'blocks' AND column_name = 'merkle_root') THEN
    ALTER TABLE blocks ADD COLUMN merkle_root TEXT;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'blocks' AND column_name = 'root_hash') THEN
        UPDATE blocks SET merkle_root = root_hash WHERE merkle_root IS NULL;
```

**Problem**: If 004 runs after 003, it recreates `blocks` with `root_hash`, making 003's work obsolete.

---

### Issue 5: Postgres 15 Syntax Issues (Partially Fixed)

**Severity**: MEDIUM (mostly resolved per MIGRATIONS.md)  
**Impact**: Syntax errors on Postgres 15+

**Status**: MIGRATIONS.md documents fixes applied in PR #5, but some migrations may still have issues.

**Pattern Fixed**:
```sql
-- OLD (fails on Postgres 15)
ALTER TABLE statements ADD CONSTRAINT IF NOT EXISTS statements_hash_unique UNIQUE (hash);

-- NEW (works)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'statements_hash_unique') THEN
        ALTER TABLE statements ADD CONSTRAINT statements_hash_unique UNIQUE (hash);
    END IF;
END $$;
```

**Remaining Risk**: 004_finalize_core_schema.sql (line 229-237) uses old syntax:
```sql
ALTER TABLE statements ADD CONSTRAINT IF NOT EXISTS statements_derivation_depth_positive
  CHECK (derivation_depth IS NULL OR derivation_depth >= 0);
```

This will fail on Postgres 15.

---

### Issue 6: Non-Idempotent Operations

**Severity**: MEDIUM  
**Impact**: Re-running migrations fails or produces incorrect state

**Examples**:

**004_finalize_core_schema.sql** (line 121):
```sql
DROP TABLE IF EXISTS blocks CASCADE;
CREATE TABLE blocks (...);
```
- First run: Creates table
- Second run: Drops table (data loss), recreates empty

**007_fix_proofs_schema.sql** (line 17-19):
```sql
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';
```
- Assumes `hash` is TEXT, but schema defines it as `bytea`
- Re-running may fail with type errors

---

### Issue 7: Foreign Key Type Mismatches

**Severity**: HIGH  
**Impact**: Migration fails with FK constraint errors

**002_blocks_lemmas.sql** (documented in MIGRATIONS.md):
```sql
CREATE TABLE lemma_cache (
    statement_id BIGINT REFERENCES statements(id)  -- FAIL: bigint ≠ uuid
);
```

**Fix Applied**: Changed to `UUID` per MIGRATIONS.md line 128.

**Verification Needed**: Confirm fix is in current file.

---

### Issue 8: Missing Table Guards

**Severity**: MEDIUM  
**Impact**: Migrations fail if tables don't exist

**003_fix_progress_compatibility.sql**: Wrapped in guards (good)
**003_add_system_id.sql**: May assume tables exist

**Example** (003_add_system_id.sql):
```sql
ALTER TABLE runs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id);
```

If `runs` table doesn't exist yet, this fails.

---

### Issue 9: Migration Execution Order Undefined

**Severity**: CRITICAL  
**Impact**: Different environments may run migrations in different orders

**Root Cause**: Duplicate numbering + filesystem-dependent sorting

**Current Behavior**:
```python
# scripts/run-migrations.py line 79
migration_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) ...)
```

**Problem**: 
- `002_add_axioms.sql` and `002_blocks_lemmas.sql` both extract `002`
- Sort order depends on filename string comparison after number
- Linux vs Windows may differ
- Git checkout order may differ

**Expected Order** (lexicographic after number):
1. `002_add_axioms.sql`
2. `002_blocks_lemmas.sql`

**Actual Order**: Undefined, filesystem-dependent

---

## Dependency Graph Analysis

```
001_init.sql (BASE)
  ├─> 002_add_axioms.sql (adds axioms table)
  ├─> 002_blocks_lemmas.sql (adds blocks, lemma_cache)
  │     └─> CONFLICT: 004 drops blocks
  ├─> 003_add_system_id.sql (adds system_id to runs, blocks, statements)
  │     └─> DEPENDS: 002_blocks_lemmas (blocks table)
  │     └─> CONFLICT: 004 drops blocks
  └─> 003_fix_progress_compatibility.sql (fixes blocks columns)
        └─> DEPENDS: 002_blocks_lemmas (blocks table)
        └─> CONFLICT: 004 drops blocks

004_finalize_core_schema.sql (MAJOR REFACTOR)
  ├─> DROPS: blocks table (CASCADE)
  ├─> RECREATES: all tables with new schema
  ├─> ADDS: system_id to all tables
  └─> CONFLICTS: 002, 003 migrations

005_add_search_indexes.sql
  └─> DEPENDS: 004 (final schema)

006_add_pg_trgm_extension.sql
  └─> ADDS: pg_trgm extension for fuzzy search

006_add_policy_settings.sql
  └─> ADDS: policy_settings table

007_fix_proofs_schema.sql
  └─> PATCHES: 004 (adds status column, fixes hash)

008_fix_statements_hash.sql
  └─> PATCHES: hash type issues

009_normalize_statements.sql
  └─> PATCHES: text normalization
  └─> CONFLICT: assumes 'text' column exists

010_idempotent_normalize.sql
  └─> PATCHES: makes 009 idempotent

011_schema_parity.sql
  └─> PATCHES: alignment issues

012_blocks_parity.sql
  └─> PATCHES: blocks table (may fail if 004 dropped it)

013_runs_logging.sql
  └─> ADDS: logging columns to runs

014_ensure_slug_column.sql
  └─> PATCHES: ensures slug column exists
```

---

## Repair Strategy Options

### Option A: Baseline Migration (RECOMMENDED)

**Approach**: Create a single authoritative migration that represents the current desired schema.

**Steps**:
1. Analyze the **final intended schema** from 004 + all patches (007-014)
2. Create `migrations/baseline_20251019.sql` with complete schema
3. Add migration tracking table to record applied migrations
4. Update migration runner to:
   - Check if baseline applied
   - Skip old migrations if baseline exists
   - Run only new migrations after baseline

**Advantages**:
- ✓ Clean slate, no historical conflicts
- ✓ Idempotent by design
- ✓ Fast execution (single transaction)
- ✓ Easy to test and validate
- ✓ Future migrations build on solid foundation

**Disadvantages**:
- ⚠ Requires schema analysis and consolidation
- ⚠ Existing databases need migration path

**Implementation**:
```sql
-- migrations/baseline_20251019.sql
-- MathLedger Baseline Schema
-- Consolidates migrations 001-014 into single authoritative schema
-- Generated: 2025-10-19 by Manus G

-- Create migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Record baseline
INSERT INTO schema_migrations (version) VALUES ('baseline_20251019')
ON CONFLICT (version) DO NOTHING;

-- [Complete schema definition follows]
```

---

### Option B: Forward Fix (NOT RECOMMENDED)

**Approach**: Fix each migration in sequence to resolve conflicts.

**Steps**:
1. Renumber duplicate migrations (002a, 002b, etc.)
2. Fix type mismatches in each file
3. Add table existence guards everywhere
4. Remove destructive DDL from 004
5. Fix column name inconsistencies

**Advantages**:
- ✓ Preserves migration history
- ✓ Incremental approach

**Disadvantages**:
- ✗ Complex, error-prone
- ✗ Still non-idempotent in many cases
- ✗ Hard to test all permutations
- ✗ 004 remains problematic (DROP TABLE)
- ✗ Doesn't solve execution order issues

---

### Option C: Hybrid Approach

**Approach**: Baseline for fresh installs, migration path for existing DBs.

**Steps**:
1. Create baseline migration
2. Create "upgrade to baseline" migration for existing DBs
3. Migration runner checks DB state and chooses path

**Advantages**:
- ✓ Best of both worlds
- ✓ Supports existing deployments
- ✓ Clean path for new installs

**Disadvantages**:
- ⚠ More complex migration runner logic
- ⚠ Requires state detection

---

## Recommended Solution: Baseline Migration

### Phase 1: Schema Analysis

**Extract final schema from**:
1. `004_finalize_core_schema.sql` (base structure)
2. `007_fix_proofs_schema.sql` (status column)
3. `008_fix_statements_hash.sql` (hash fixes)
4. `009-011` (normalization columns)
5. `012_blocks_parity.sql` (blocks updates)
6. `013_runs_logging.sql` (logging)
7. `014_ensure_slug_column.sql` (slug)

**Resolve conflicts**:
- Use `root_hash` (not `merkle_root`)
- Use `content_norm` (not `text`)
- Use `hash` as `TEXT` (hex-encoded, not `bytea`)
- Include all `system_id` columns
- Include all indexes from 005

### Phase 2: Create Baseline

**File**: `migrations/baseline_20251019.sql`

**Structure**:
```sql
-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (...);

-- 3. Core tables (theories, symbols, statements, proofs, dependencies)
-- 4. Operational tables (runs, blocks, lemma_cache)
-- 5. Indexes (comprehensive)
-- 6. Constraints (Postgres 15 compatible)
-- 7. Seed data
```

### Phase 3: Update Migration Runner

**File**: `scripts/run-migrations.py`

**Changes**:
```python
def check_baseline_applied(conn):
    """Check if baseline migration has been applied."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'schema_migrations'
            )
        """)
        if not cur.fetchone()[0]:
            return False
        
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM schema_migrations 
                WHERE version = 'baseline_20251019'
            )
        """)
        return cur.fetchone()[0]

def main():
    # Check if baseline applied
    if check_baseline_applied(conn):
        print("Baseline migration detected, skipping legacy migrations")
        migration_files = [f for f in migration_files if f.name >= 'baseline_20251019.sql']
    else:
        print("No baseline detected, running all migrations")
```

### Phase 4: Create 2-Pass Idempotency Test

**File**: `.github/workflows/db-migration-check.yml`

```yaml
name: Database Migration Check

on:
  pull_request:
    paths:
      - 'migrations/**'
      - 'scripts/run-migrations.py'

jobs:
  migration-idempotency:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: ml
          POSTGRES_PASSWORD: mlpass
          POSTGRES_DB: mathledger
        ports: ["5432:5432"]
        options: >-
          --health-cmd="pg_isready -U ml"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5
    
    env:
      DATABASE_URL: postgresql://ml:mlpass@localhost:5432/mathledger
    
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      
      - name: "Pass 1: Run migrations on fresh database"
        run: |
          uv run python scripts/run-migrations.py
          echo "Pass 1 completed"
      
      - name: "Capture schema after Pass 1"
        run: |
          pg_dump -U ml -h localhost -d mathledger --schema-only > schema_pass1.sql
      
      - name: "Pass 2: Re-run migrations (idempotency test)"
        run: |
          uv run python scripts/run-migrations.py
          echo "Pass 2 completed"
      
      - name: "Capture schema after Pass 2"
        run: |
          pg_dump -U ml -h localhost -d mathledger --schema-only > schema_pass2.sql
      
      - name: "Compare schemas (must be identical)"
        run: |
          diff schema_pass1.sql schema_pass2.sql || {
            echo "FAIL: Schema changed between pass 1 and pass 2"
            echo "Migrations are NOT idempotent"
            exit 1
          }
          echo "PASS: Schemas identical, migrations are idempotent"
      
      - name: "Upload schema artifacts"
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: migration-schemas
          path: |
            schema_pass1.sql
            schema_pass2.sql
```

---

## Success Criteria

**[PASS] Migrations: 2-pass idempotent; baseline consistent**

1. ✓ Baseline migration creates complete schema
2. ✓ Running migrations twice produces identical schema
3. ✓ No errors on Postgres 15
4. ✓ CI job `db-migration-check.yml` passes
5. ✓ All tables, indexes, constraints present
6. ✓ Seed data populated correctly
7. ✓ Migration tracking table records baseline

---

## Next Steps

1. **Create baseline schema** from analysis above
2. **Update migration runner** with baseline detection
3. **Create CI workflow** for 2-pass testing
4. **Test locally** with fresh Postgres 15
5. **Create PR** with:
   - `migrations/baseline_20251019.sql`
   - `scripts/run-migrations.py` (updated)
   - `.github/workflows/db-migration-check.yml` (new)
   - `scripts/db/repair_migrations.md` (this document)
6. **Enable migrations in CI** (remove line 29-32 comment in ci.yml)

---

## Appendix A: Migration File Contents Summary

### 001_init.sql
- Creates: theories, symbols, statements, proofs, dependencies
- Hash type: `bytea`
- Column: `content_norm`
- Seed: Propositional theory

### 002_add_axioms.sql
- Creates: axioms table

### 002_blocks_lemmas.sql
- Creates: blocks, lemma_cache
- FK issue: statement_id BIGINT (should be UUID)

### 003_add_system_id.sql
- Adds: system_id to runs, blocks, statements

### 003_fix_progress_compatibility.sql
- Fixes: blocks columns (block_number, merkle_root, header)
- Guards: table existence checks

### 004_finalize_core_schema.sql
- **DESTRUCTIVE**: DROP TABLE blocks CASCADE
- Recreates: all tables with system_id
- Hash type: `bytea`
- Column: `content_norm`
- Constraint syntax: old style (fails on Postgres 15)

### 005_add_search_indexes.sql
- Adds: search indexes
- Issue: CONCURRENTLY in transaction (fixed per MIGRATIONS.md)

### 006_add_pg_trgm_extension.sql
- Adds: pg_trgm extension

### 006_add_policy_settings.sql
- Creates: policy_settings table

### 007_fix_proofs_schema.sql
- Adds: status column to proofs
- Updates: hash as TEXT (conflict with bytea)
- Column: `content_norm`

### 008_fix_statements_hash.sql
- Fixes: hash type issues

### 009_normalize_statements.sql
- Adds: normalized_text column
- Updates: assumes `text` column (doesn't exist)
- Hash: treats as TEXT

### 010_idempotent_normalize.sql
- Fixes: 009 idempotency

### 011_schema_parity.sql
- Fixes: schema alignment

### 012_blocks_parity.sql
- Updates: blocks table (may not exist if 004 ran)

### 013_runs_logging.sql
- Adds: logging columns to runs

### 014_ensure_slug_column.sql
- Ensures: slug column exists in theories

---

**End of Diagnostic Report**

