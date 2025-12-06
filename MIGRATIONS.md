# Migration Doctrine

## Postgres 15 Compliance via Schema Discipline

This document establishes the **MathLedger Migration Doctrine** for PostgreSQL 15+ compatibility and robust schema evolution.

### Core Principles

1. **Idempotency**: All migrations must be safe to run multiple times
2. **Defensive Checks**: Never assume tables or columns exist
3. **Postgres 15 Syntax**: Use DO blocks instead of inline `IF NOT EXISTS` for constraints
4. **Transaction Safety**: Avoid `CONCURRENTLY` in transactional migrations
5. **Schema Tolerance**: Support both legacy and modern column names

---

## Critical Patterns

### 1. IF NOT EXISTS Constraints (Postgres 15+ Compatible)

**❌ BROKEN (Pre-Postgres 15 syntax):**
```sql
ALTER TABLE statements ADD CONSTRAINT IF NOT EXISTS statements_hash_unique UNIQUE (hash);
```

**✅ FIXED (Postgres 15 compatible):**
```sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'statements_hash_unique') THEN
        ALTER TABLE statements ADD CONSTRAINT statements_hash_unique UNIQUE (hash);
    END IF;
END $$;
```

**Why**: PostgreSQL 15 doesn't support `IF NOT EXISTS` in `ADD CONSTRAINT`. Use anonymous DO blocks with `pg_constraint` lookups.

---

### 2. CREATE INDEX CONCURRENTLY (Transaction Incompatibility)

**❌ BROKEN (Can't run in transaction):**
```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_statements_hash ON statements(hash);
```

**✅ FIXED (Migration-safe):**
```sql
-- NOTE: CONCURRENTLY removed for migration compatibility (can't run in transaction)
-- In production, consider using CONCURRENTLY to avoid table locks
CREATE INDEX IF NOT EXISTS idx_statements_hash ON statements(hash);
```

**Why**: `CONCURRENTLY` cannot execute inside transaction blocks. Omit for automated migrations, run manually with CONCURRENTLY in production if needed.

---

### 3. Table Existence Guards

**❌ FRAGILE (Assumes table exists):**
```sql
ALTER TABLE blocks ADD COLUMN block_number BIGINT;
```

**✅ DEFENSIVE:**
```sql
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks') THEN
        ALTER TABLE blocks ADD COLUMN IF NOT EXISTS block_number BIGINT;
    END IF;
END $$;
```

**Why**: Migration order can vary, tables may be created out of sequence. Always guard with existence checks.

---

### 4. Column Existence for UPDATEs

**❌ FRAGILE (Assumes `text` column):**
```sql
UPDATE statements SET normalized_text = REPLACE(text, '→', '->');
```

**✅ SCHEMA-TOLERANT:**
```sql
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'statements' AND column_name = 'text') THEN
        UPDATE statements SET normalized_text = REPLACE(text, '→', '->');
    ELSIF EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_name = 'statements' AND column_name = 'content_norm') THEN
        UPDATE statements SET normalized_text = REPLACE(content_norm, '→', '->');
    END IF;
END $$;
```

**Why**: Schema evolves. Migrations must support both legacy (`content_norm`) and modern (`text`) column names.

---

### 5. Foreign Key Type Matching

**❌ TYPE MISMATCH:**
```sql
-- statements.id is UUID (from migration 004)
CREATE TABLE lemma_cache (
    statement_id BIGINT REFERENCES statements(id)  -- FAIL: bigint ≠ uuid
);
```

**✅ TYPE ALIGNED:**
```sql
CREATE TABLE lemma_cache (
    statement_id UUID REFERENCES statements(id)
);
```

**Why**: Foreign key columns must exactly match referenced column types. Check target table schema before creating FKs.

---

## Migration Fixes Applied (PR #5 Unblocking)

### 002_blocks_lemmas.sql
- **Issue**: FK type mismatch (bigint vs UUID)
- **Fix**: Changed `statement_id` to UUID

### 003_fix_progress_compatibility.sql
- **Issue**: Assumes `blocks` table exists
- **Fix**: Wrapped all operations in table existence check

### 003_add_system_id.sql
- **Issue**: Assumes `runs`, `blocks`, `statements` exist
- **Fix**: Conditional backfill with table guards

### 004_finalize_core_schema.sql
- **Issue**: `ALTER TABLE ADD CONSTRAINT IF NOT EXISTS` (Postgres 15 syntax error)
- **Fix**: Replaced with DO blocks + `pg_constraint` checks

### 005_add_search_indexes.sql
- **Issue**: `CREATE INDEX CONCURRENTLY` in transaction
- **Fix**: Removed `CONCURRENTLY`, added warning comment

### 006_add_pg_trgm_extension.sql
- **Issue**: INSERT with `slug` column (doesn't exist yet)
- **Fix**: Conditional INSERT based on column existence

### 007_fix_proofs_schema.sql, 008_fix_statements_hash.sql
- **Issue**: `ADD CONSTRAINT IF NOT EXISTS` syntax
- **Fix**: DO block pattern

### 009-011 (normalize_statements, idempotent_normalize, schema_parity)
- **Issue**: UPDATE references `text` column (may not exist)
- **Fix**: Fallback to `content_norm` if `text` missing

### 012_blocks_parity.sql
- **Issue**: Alters `blocks` table unconditionally
- **Fix**: Wrapped in table existence guard

---

## Testing Migrations

### Local Testing
```bash
# Start fresh database
docker compose down -v
docker compose up -d postgres

# Run migrations
python run_all_migrations.py

# Verify no errors
psql -U ml -d mathledger -c "SELECT COUNT(*) FROM theories;"
```

### CI Testing
Migrations run automatically in `.github/workflows/ci.yml`:
```yaml
- run: uv run python scripts/run-migrations.py
```

Expected: All 16 migrations succeed, no syntax errors.

---

## Migration Anti-Patterns (Avoid These)

### ❌ Hard-coding Column Names
```sql
UPDATE statements SET hash = encode(sha256(content_norm::bytea), 'hex');
```
**Problem**: Breaks if schema uses different column name.

### ❌ Assuming Migration Order
```sql
-- Migration 006 assumes 004 already ran
INSERT INTO theories (slug, ...) VALUES (...);
```
**Problem**: Migration numbering doesn't guarantee execution order.

### ❌ Destructive DDL
```sql
DROP TABLE blocks CASCADE;
```
**Problem**: Data loss, no rollback in production.

### ❌ Missing Idempotency
```sql
ALTER TABLE statements ADD COLUMN hash TEXT;  -- Missing IF NOT EXISTS
```
**Problem**: Re-running migration fails.

---

## Rollback Strategy

Migrations are **additive-only** (no destructive changes). Rollback = restore from backup.

**Backup before migrations:**
```bash
powershell -File .\scripts\backup-db.ps1
```

**Restore if needed:**
```bash
powershell -File .\scripts\restore-db.ps1 -BackupPath backups/mathledger-YYYYMMDD-HHMMSS.dump
```

---

## Future Recommendations

1. **Adopt Alembic or Flyway**: Version-tracked migrations with automatic rollback
2. **Schema Migrations as Code**: Generate from SQLAlchemy models
3. **Migration Testing CI**: Dedicated workflow for schema changes
4. **Separate CONCURRENTLY Indexes**: Run post-deployment, not in migrations

---

## PR Tagline

**[RC] — Compliance via Schema Discipline**

- **RC**: Release Candidate (migration compliance achieved)
- **Schema Discipline**: Postgres 15 syntax, defensive checks, idempotent operations

---

## Contact

For migration issues:
1. Check this doctrine for patterns
2. Review `.github/workflows/ci.yml` logs
3. Test locally with `docker compose up -d postgres`

**Doctrine Version**: 1.0 (PR #5 Unblocking, 2025-10-02)
# Migration Compliance
