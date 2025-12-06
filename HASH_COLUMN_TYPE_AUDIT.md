# Hash Column Type Audit
## Evidence Pack v1 - Migration Schema Consistency Review

**Generated**: 2025-01-XX (GEMINI F - Migration Surgeon Phase 2)  
**Mode**: Reviewer-2 / Evidence Consolidation  
**Status**: Analysis Complete - No Implementation

---

## Executive Summary

The `statements.hash` column exhibits a **type drift inconsistency** across migrations:

- **Migration 001**: Creates `hash bytea`
- **Migrations 007-011**: Assume/add `hash TEXT` (hex-encoded)
- **Migration 016**: Converts `bytea → TEXT` using `encode(hash, 'hex')`

**Critical Finding**: Migrations 007-011 will fail if run on a database where `hash` is still `bytea` (i.e., if migrations 001-006 ran but 016 hasn't run yet).

**Code Expectation**: All application code in `backend/` expects `hash` to be `TEXT` (hex-encoded strings).

---

## Migration-by-Migration Analysis

### Migration 000: Schema Version Tracking
- **Hash Column Impact**: None
- **Status**: ✅ No hash operations

### Migration 001: Init Schema
**File**: `migrations/001_init.sql`  
**Line 29**: `hash bytea not null unique`

**Action**: Creates `statements.hash` as `BYTEA`  
**Evidence**:
```sql
hash            bytea not null unique,  -- SHA-256 of normalized form
```

**Also Creates**: `proofs.proof_hash bytea` (line 48)

**Status**: ⚠️ **ROOT CAUSE** - Establishes bytea type

---

### Migration 002: Add Axioms
**File**: `migrations/002_add_axioms.sql`  
**Status**: ✅ **FIXED** - Now handles both bytea and text

**Evidence**: Lines 57-135 contain type-detection logic:
```sql
DECLARE
    hash_type TEXT;
    k_hash BYTEA;
    s_hash BYTEA;
    k_hash_text TEXT;
    s_hash_text TEXT;
BEGIN
    -- Detect hash column type
    SELECT data_type INTO hash_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'statements'
      AND column_name = 'hash';
    
    -- Branches on hash_type = 'bytea' vs else (text)
```

**Behavior**: Type-aware insertion that works regardless of current column type.

---

### Migration 003: Add System ID
**File**: `migrations/003_add_system_id.sql`  
**Hash Column Impact**: None  
**Status**: ✅ No hash operations

---

### Migration 004: Finalize Core Schema
**File**: `migrations/004_finalize_core_schema.sql`  
**Line 45**: `hash bytea not null unique` (in CREATE TABLE IF NOT EXISTS)

**Action**: Defines bytea in table definition, but also has additive logic  
**Evidence**:
```sql
CREATE TABLE IF NOT EXISTS statements (
  ...
  hash            bytea not null unique,  -- SHA-256 of normalized form
  ...
);
```

**Status**: ⚠️ **INCONSISTENT** - Defines bytea but doesn't enforce conversion

**Note**: This migration also has `ADD COLUMN IF NOT EXISTS` logic elsewhere, but the CREATE TABLE definition takes precedence if table doesn't exist.

---

### Migration 005: Add Search Indexes
**File**: `migrations/005_add_search_indexes.sql`  
**Hash Column Impact**: None  
**Status**: ✅ No hash operations

---

### Migration 006: Add PG_TRGM Extension
**File**: `migrations/006_add_pg_trgm_extension.sql`  
**Hash Column Impact**: None  
**Status**: ✅ No hash operations

---

### Migration 007: Fix Proofs Schema
**File**: `migrations/007_fix_proofs_schema.sql`  
**Line 24**: `SET hash = encode(sha256(content_norm::bytea), 'hex')`

**Action**: Attempts to UPDATE hash column using hex encoding  
**Assumption**: Column exists and can accept TEXT assignment  
**Evidence**:
```sql
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';
```

**Status**: ❌ **WILL FAIL** if `hash` is `bytea` - PostgreSQL cannot implicitly convert `TEXT` to `bytea`

**Error Scenario**: If migrations 001-006 ran (hash = bytea), this UPDATE will fail with:
```
ERROR: column "hash" is of type bytea but expression is of type text
```

---

### Migration 008: Fix Statements Hash
**File**: `migrations/008_fix_statements_hash.sql`  
**Line 5**: `ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;`  
**Line 9**: `SET hash = encode(sha256(content_norm::bytea), 'hex')`

**Action**: 
1. Adds `hash TEXT` if missing
2. Backfills with hex-encoded text

**Status**: ⚠️ **PARTIAL FIX** - If column already exists as `bytea`, `ADD COLUMN IF NOT EXISTS` does nothing, and UPDATE will fail

**Evidence**:
```sql
-- Add hash column if missing
ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;

-- Backfill hash from normalized content
UPDATE statements
SET hash = encode(sha256(content_norm::bytea), 'hex')
WHERE hash IS NULL OR hash = '';
```

**Problem**: If `hash bytea` exists from 001, the `ADD COLUMN` is skipped, and the UPDATE fails.

---

### Migration 009: Normalize Statements
**File**: `migrations/009_normalize_statements.sql`  
**Line 23**: `ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;`  
**Line 27**: `SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))`

**Action**: Same pattern as 008  
**Status**: ❌ **SAME ISSUE** as 008

---

### Migration 010: Idempotent Normalize
**File**: `migrations/010_idempotent_normalize.sql`  
**Line 8**: `ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;`  
**Line 43**: `SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))`

**Action**: Same pattern as 008/009  
**Status**: ❌ **SAME ISSUE** as 008/009

---

### Migration 011: Schema Parity
**File**: `migrations/011_schema_parity.sql`  
**Line 6**: `ALTER TABLE statements ADD COLUMN IF NOT EXISTS hash TEXT;`  
**Line 27**: `SET hash = LOWER(encode(sha256(normalized_text::bytea), 'hex'))`

**Action**: Same pattern as 008/009/010  
**Status**: ❌ **SAME ISSUE** as 008/009/010

---

### Migration 012: Blocks Parity
**File**: `migrations/012_blocks_parity.sql`  
**Hash Column Impact**: None (only touches `blocks.prev_hash`)  
**Status**: ✅ No `statements.hash` operations

---

### Migration 013: Runs Logging
**File**: `migrations/013_runs_logging.sql`  
**Hash Column Impact**: None  
**Status**: ✅ No `statements.hash` operations

---

### Migration 014: Ensure Slug Column
**File**: `migrations/014_ensure_slug_column.sql`  
**Hash Column Impact**: None  
**Status**: ✅ No `statements.hash` operations

---

### Migration 015: Dual Root Attestation
**File**: `migrations/015_dual_root_attestation.sql`  
**Hash Column Impact**: None (only touches `blocks` table)  
**Status**: ✅ No `statements.hash` operations

---

### Migration 016: Monotone Ledger
**File**: `migrations/016_monotone_ledger.sql`  
**Lines 18-32**: Type conversion logic

**Action**: **ONLY MIGRATION** that properly handles bytea → text conversion  
**Evidence**:
```sql
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'statements'
          AND column_name = 'hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE statements
        ALTER COLUMN hash TYPE TEXT
        USING encode(hash, 'hex');
    END IF;
END $$;
```

**Status**: ✅ **CORRECT IMPLEMENTATION** - Type-aware conversion

**Also Converts**: `proofs.proof_hash bytea → TEXT` (lines 83-97)

---

## Application Code Expectations

### Backend Code Analysis

**Files Expecting TEXT (hex strings)**:

1. **`backend/axiom_engine/derive.py`** (line 93):
   ```python
   cur.execute("SELECT id FROM statements WHERE hash=%s LIMIT 1", (h,))
   ```
   - `h` is computed as hex string: `return hashlib.sha256(s.encode()).hexdigest()`

2. **`backend/axiom_engine/derive_core.py`** (lines 119, 210):
   ```python
   self.cur.execute("SELECT id FROM statements WHERE hash = %s LIMIT 1", (record.hash,))
   ```
   - `record.hash` is hex string from `_sha()` function

3. **`backend/tools/export_fol_ab.py`** (line 249):
   ```python
   INSERT INTO statements (theory_id, hash, content_norm, ...)
   ```
   - Hash is hex string from `hashlib.sha256(...).hexdigest()`

4. **`backend/dag/proof_dag.py`** (line 791):
   ```python
   raise RuntimeError("statements table must expose hash or canonical_hash column")
   ```
   - Expects hash to be queryable as string

**Conclusion**: All application code expects `hash` to be `TEXT` (hex-encoded), not `bytea`.

---

## Proof Hash Column Analysis

**Migration 001**: Creates `proofs.proof_hash bytea`  
**Migration 016**: Converts `proofs.proof_hash bytea → TEXT` (lines 83-97)

**Status**: Same pattern as `statements.hash` - 016 is the normalization point.

---

## Inconsistency Matrix

| Migration | Creates/Assumes | Will Fail If | Status |
|-----------|----------------|--------------|--------|
| 001 | `hash bytea` | N/A (initial) | ⚠️ Root cause |
| 002 | Type-aware | ✅ Works with both | ✅ Fixed |
| 003 | N/A | ✅ No hash ops | ✅ Safe |
| 004 | `hash bytea` (in CREATE) | ⚠️ Defines bytea | ⚠️ Inconsistent |
| 005 | N/A | ✅ No hash ops | ✅ Safe |
| 006 | N/A | ✅ No hash ops | ✅ Safe |
| 007 | Assumes TEXT | ❌ Fails if bytea | ❌ **BREAKS** |
| 008 | Adds TEXT, updates | ❌ Fails if bytea exists | ❌ **BREAKS** |
| 009 | Adds TEXT, updates | ❌ Fails if bytea exists | ❌ **BREAKS** |
| 010 | Adds TEXT, updates | ❌ Fails if bytea exists | ❌ **BREAKS** |
| 011 | Adds TEXT, updates | ❌ Fails if bytea exists | ❌ **BREAKS** |
| 012-015 | N/A | ✅ No hash ops | ✅ Safe |
| 016 | Converts bytea→TEXT | ✅ Works with both | ✅ **FIXES** |

**Critical Path**: If migrations run sequentially 001→016, the failure occurs at 007-011.

---

## Minimal Fix Strategy (Analysis Only - Not Implemented)

### Option A: Early Normalization (Recommended)
**Location**: After migration 005, before 007  
**Action**: Create new migration `005b_normalize_hash_to_text.sql`

**Content**:
```sql
-- Normalize statements.hash from bytea to TEXT (hex-encoded)
-- This ensures all subsequent migrations can assume TEXT type

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'statements'
          AND column_name = 'hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE statements
        ALTER COLUMN hash TYPE TEXT
        USING encode(hash, 'hex');
    END IF;
END $$;

-- Same for proofs.proof_hash
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'proofs'
          AND column_name = 'proof_hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE proofs
        ALTER COLUMN proof_hash TYPE TEXT
        USING encode(proof_hash, 'hex');
    END IF;
END $$;
```

**Pros**:
- Fixes issue before migrations 007-011 run
- Minimal change (one new migration)
- Idempotent (can run multiple times)
- Preserves existing data

**Cons**:
- Requires renumbering or inserting migration
- Migration 016's conversion becomes redundant (but harmless)

---

### Option B: Fix Migration 001
**Location**: `migrations/001_init.sql`  
**Action**: Change line 29 from `bytea` to `TEXT`

**Content**:
```sql
hash            TEXT not null unique,  -- SHA-256 hex-encoded
```

**Pros**:
- Fixes root cause
- No new migration needed
- All subsequent migrations work correctly

**Cons**:
- **BREAKING CHANGE** for any existing databases with bytea
- Requires data migration for existing deployments
- Migration 016's conversion logic becomes necessary for legacy DBs

---

### Option C: Fix Migrations 007-011
**Location**: Migrations 007, 008, 009, 010, 011  
**Action**: Add type conversion logic before UPDATE statements

**Pattern** (for each migration):
```sql
-- Convert bytea to TEXT if needed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'statements'
          AND column_name = 'hash'
          AND data_type = 'bytea'
    ) THEN
        ALTER TABLE statements
        ALTER COLUMN hash TYPE TEXT
        USING encode(hash, 'hex');
    END IF;
END $$;

-- Then proceed with existing UPDATE logic
```

**Pros**:
- Each migration becomes self-contained
- Works regardless of migration order
- No breaking changes

**Cons**:
- Requires modifying 5 migration files
- Duplicates conversion logic
- Migration 016's conversion still needed for safety

---

### Option D: Keep Status Quo (Not Recommended)
**Assumption**: Migrations always run sequentially 001→016  
**Reality**: Migration 016 fixes the issue

**Pros**:
- No code changes needed
- Works if migrations run in order

**Cons**:
- ❌ **BREAKS** if migrations 007-011 run before 016
- ❌ **BREAKS** if migration 016 is skipped
- ❌ **BREAKS** on fresh DB bootstrap if migrations run out of order
- Not idempotent-safe

---

## Recommended Approach

**Option A (Early Normalization)** is recommended because:

1. **Minimal Change**: One new migration file
2. **Idempotent**: Safe to run multiple times
3. **Non-Breaking**: Works with existing bytea data
4. **Future-Proof**: All subsequent migrations can assume TEXT
5. **Evidence-Based**: Uses same pattern as migration 016 (proven to work)

**Implementation Note**: Migration should be numbered `005b` or `006a` to maintain sequential ordering, or renumber existing migrations 006-016 to 007-017.

---

## Verification Checklist

Before implementing any fix, verify:

- [ ] All existing databases have migration history recorded in `schema_migrations`
- [ ] No production databases are stuck at migration 001-006 with bytea type
- [ ] Migration runner supports out-of-order execution (if applicable)
- [ ] First Organism test fixture (`first_organism_db`) runs migration 016
- [ ] Application code consistently expects TEXT (hex strings) - ✅ **VERIFIED**

---

## Evidence Artifacts

**Files Analyzed**:
- `migrations/001_init.sql` through `migrations/016_monotone_ledger.sql`
- `backend/axiom_engine/derive.py`
- `backend/axiom_engine/derive_core.py`
- `backend/tools/export_fol_ab.py`
- `backend/dag/proof_dag.py`

**Grep Results**:
- 52 matches for `hash.*bytea|hash.*text` (case-insensitive)
- 35 matches for `ALTER.*hash|ADD.*hash|CREATE.*hash`

**Code Patterns**:
- All `hash` usage in backend expects hex strings (TEXT)
- No bytea handling found in application code
- Migration 016 is only migration with proper type conversion

---

## Conclusion

**Status**: ✅ **AUDIT COMPLETE**

**Finding**: Type drift inconsistency exists. Migrations 007-011 will fail if `hash` is `bytea`.

**Recommendation**: Implement Option A (Early Normalization) after migration 005.

**Risk Level**: **MEDIUM** - Affects fresh database bootstraps and out-of-order migration execution.

**Evidence Quality**: **HIGH** - All claims backed by on-disk file analysis.

---

---

## Phase-I Impact Statement

**Evidence-Based Assessment**: Phase-I RFL execution does not interact with database schema.

### Phase-I RFL Execution Model

**Artifacts**: `results/fo_rfl.jsonl`, `results/fo_rfl_50.jsonl`, `results/fo_rfl_1000.jsonl`

**Evidence**:
- `experiments/run_fo_cycles.py` writes exclusively to JSONL files (via `--out` parameter)
- No database operations found: grep for `database|db_|psycopg|INSERT|UPDATE|CREATE TABLE` in `run_fo_cycles.py` returns zero matches
- All cycle results serialized to JSONL format with deterministic hashes computed in-memory
- Hash values in JSONL are hex strings (TEXT), computed via `hashlib.sha256(...).hexdigest()`

**Code Verification**:
```python
# experiments/run_fo_cycles.py
# Line 6: "Supports 'baseline' (RFL OFF) and 'rfl' (RFL ON) modes with hermetic determinism."
# Line 11: "--out=results/fo_rfl.jsonl" (file output, not database)
```

### Impact on Phase-I Evidence

**Status**: ✅ **NO CORRUPTION RISK**

**Rationale**:
1. Phase-I RFL logs are **hermetic negative-control runs** - all computation happens in-memory, no database interaction
2. Hash values are computed as Python strings (`hexdigest()`) and written directly to JSONL files
3. No database schema interaction occurs during Phase-I execution (verified: zero database operations in `run_fo_cycles.py`)
4. Hash type inconsistencies in database migrations cannot affect Phase-I evidence integrity
5. All Phase-I RFL runs show 100% abstention by design (lean-disabled mode) - this validates plumbing, not uplift

**Evidence Files Verified** (per canonical Phase-I truth table):
- `results/fo_baseline.jsonl` - 1000 cycles (0-999), 100% abstention, old schema (no status/method/abstention fields)
- `results/fo_rfl.jsonl` - **1001 cycles (0-1000)**, 100% abstention, new schema (status/method/abstention present), hermetic negative-control/plumbing run
- `results/fo_rfl_50.jsonl` - **21 cycles (0-20)**, 100% abstention, new schema, **incomplete** (not 50 cycles), small RFL plumbing/negative control demo
- `results/fo_rfl_1000.jsonl` - **11 cycles (0-10)**, 100% abstention, new schema, **incomplete** (not 1000 cycles), do not use for evidence

**Canonical Phase-I Facts**:
- All Phase-I RFL logs are **hermetic negative-control runs** with 100% abstention (lean-disabled mode)
- Phase-I validates RFL execution infrastructure and attestation only; **zero empirical RFL uplift**
- All hash values are hex-encoded strings in JSON format, computed in-memory, independent of database schema

### Phase-II Requirements

**Status**: ⚠️ **MUST RESOLVE BEFORE PHASE-II**

**Rationale**:
1. Phase-II RFL will be **database-backed** - requires `statements.hash` column for UPSERT operations
2. Application code expects `hash TEXT` (hex strings) - verified in `backend/axiom_engine/derive.py`, `derive_core.py`
3. Migrations 007-011 will fail if `hash` is `bytea` when Phase-II attempts to write to database
4. First Organism test fixture (`first_organism_db`) runs migration 016, which normalizes types, but Phase-II may run migrations independently

**Evidence**:
- `backend/rfl/config.py` (lines 98-100): `database_url` field indicates Phase-II will use database
- `config/rfl/production.json` (lines 20-27): `fallback_to_db: true` with database configuration
- `backend/axiom_engine/derive_core.py` (line 119): `SELECT id FROM statements WHERE hash = %s` expects TEXT

**Conclusion**: Hash type inconsistency must be resolved via Option A (Early Normalization) before Phase-II database-backed RFL execution.

---

## Compliance Statement

**GEMINI F - Migration Surgeon / Hash Column Type Auditor**

This audit document has been verified against the canonical Phase-I truth table:

✅ **Cycle Counts Verified**:
- `fo_rfl.jsonl`: 1001 cycles (0-1000) - corrected from incorrect "1000 cycles"
- `fo_rfl_50.jsonl`: 21 cycles (0-20) - marked as incomplete
- `fo_rfl_1000.jsonl`: 11 cycles (0-10) - marked as incomplete

✅ **Terminology Verified**:
- Phase-I RFL described as "hermetic negative-control / plumbing only"
- Explicitly states "100% abstention" for all Phase-I RFL logs
- No claims of "uplift" or "metabolism verification" in Phase-I context
- Phase-II explicitly tagged as "database-backed" and "not yet implemented"

✅ **Evidence Alignment**:
- All Phase-I RFL artifacts correctly identified as file-based, hermetic, non-DB
- Hash type inconsistencies correctly identified as Phase-II blocker only
- No Phase-I evidence corruption risk (correctly assessed as zero)

**Status**: ✅ **ALIGNED WITH CANONICAL TRUTH TABLE**

---

**End of Audit**

