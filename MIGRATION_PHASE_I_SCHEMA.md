# Migration Phase I Schema Requirements

**Status**: Evidence Consolidation (Reviewer-2 Mode)  
**Date**: 2025-01-XX  
**Agent**: GEMINI E (Migration Surgeon)

## Purpose

This document identifies the **minimal database schema** required for Phase I operations:
1. FO closed-loop happy path (`test_first_organism_closed_loop_happy_path`)
2. FO smoke test (`test_first_organism_closed_loop_smoke`)
3. Attestation structure v1 (dual-root attestation: R_t, U_t, H_t)

**Boundary**: Only tables/columns **actually used** by these tests are listed. No hypothetical requirements.

---

## Required Tables and Columns

### 1. `schema_migrations` (Migration 000)

**Purpose**: Track applied migrations (required by migration runner pre-flight check)

**Required Columns**:
- `version TEXT PRIMARY KEY`
- `applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `checksum TEXT`
- `duration_ms INTEGER`
- `status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'skipped'))`

**Evidence**: `scripts/run-migrations.py::ensure_schema_migrations_table()` creates this table if missing.

---

### 2. `theories` (Migrations 001, 004)

**Purpose**: Logical systems (Propositional, FOL, etc.)

**Required Columns**:
- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `name TEXT NOT NULL UNIQUE`
- `slug TEXT UNIQUE` (added in 003, required by 004)
- `version TEXT DEFAULT 'v0'`
- `logic TEXT DEFAULT 'unspecified'`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

**Evidence**: 
- `ledger/ingest.py` queries `theories` by name to get `system_id`
- `scripts/run_fol_smoke.py::_ensure_system()` uses `systems` table (legacy) OR `theories` table (new)

**Note**: Legacy code may use `systems` table (SERIAL id), but canonical schema uses `theories` (UUID id).

---

### 3. `statements` (Migrations 001, 004, 008, 009, 010, 011)

**Purpose**: Mathematical statements with normalized content

**Required Columns**:
- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `theory_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE`
- `system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE` (004 backfills from theory_id)
- `hash BYTEA NOT NULL UNIQUE` (SHA-256 of normalized form)
- `content_norm TEXT NOT NULL` (normalized s-expr / canonical form)
- `content_lean TEXT` (optional)
- `content_latex TEXT` (optional)
- `status TEXT NOT NULL CHECK (status IN ('proven','disproven','open','unknown')) DEFAULT 'unknown'`
- `derivation_rule TEXT` (rule used to derive: 'mp', 'axiom', etc.)
- `derivation_depth INT` (depth in derivation tree)
- `truth_domain TEXT` (optional)
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

**Evidence**:
- `ledger/ingest.py::_upsert_statement()` inserts/updates statements with these columns
- `tests/integration/test_first_organism.py` verifies statement hashes match canonical computation
- `scripts/run_fol_smoke.py::_upsert_statement()` uses flexible column detection (supports legacy `canonical_hash`, `normalized_text`)

**Note**: Legacy code may use `hash VARCHAR(64)` or `canonical_hash VARCHAR(64)`, but canonical schema uses `hash BYTEA`.

---

### 4. `proofs` (Migrations 001, 004, 007)

**Purpose**: Proof attempts with timing and success information

**Required Columns**:
- `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`
- `statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE`
- `system_id UUID NOT NULL REFERENCES theories(id) ON DELETE CASCADE` (004 backfills from statement)
- `prover TEXT NOT NULL` (e.g., 'lean4', 'lean-interface')
- `method TEXT` (proof method: 'cc', 'axiom', 'mp', etc.)
- `proof_term TEXT` (optional serialized proof term)
- `time_ms INT CHECK (time_ms >= 0)` (optional)
- `steps INT CHECK (steps >= 0)` (optional)
- `success BOOLEAN NOT NULL DEFAULT FALSE`
- `proof_hash BYTEA` (optional hash of proof term)
- `kernel_version TEXT` (optional Lean/mathlib commit)
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

**Evidence**:
- `ledger/ingest.py::_upsert_proof()` inserts proofs with these columns
- `scripts/run_fol_smoke.py::_insert_proof()` uses flexible column detection (supports legacy `status TEXT`)

**Note**: Legacy code may use `status TEXT` instead of `success BOOLEAN`, but canonical schema uses `success BOOLEAN`.

---

### 5. `blocks` (Migrations 004, 015, 016)

**Purpose**: Blockchain-style blocks containing statement batches with dual-root attestation

**Required Columns** (from 004 base + 015 attestation + 016 monotone):
- `id BIGSERIAL PRIMARY KEY`
- `run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE` (004, but may be NULL)
- `system_id UUID NOT NULL REFERENCES theories(id)` (004)
- `block_number BIGINT NOT NULL` (004, sequential within system)
- `prev_hash TEXT` (016, hash of previous block)
- `prev_block_id BIGINT REFERENCES blocks(id) ON DELETE SET NULL` (016, for chaining)
- `root_hash TEXT NOT NULL` (004, legacy merkle root)
- `reasoning_merkle_root TEXT` (015, R_t: Merkle root of proof/reasoning events)
- `ui_merkle_root TEXT` (015, U_t: Merkle root of UI/human interaction events)
- `composite_attestation_root TEXT` (015, H_t: SHA256(R_t || U_t))
- `attestation_metadata JSONB DEFAULT '{}'::jsonb` (015, audit trails)
- `block_hash TEXT` (016, hash of block payload)
- `payload_hash TEXT` (016, hash of canonical payload)
- `statement_count INT NOT NULL DEFAULT 0` (016)
- `proof_count INT NOT NULL DEFAULT 0` (016)
- `sealed_at TIMESTAMPTZ DEFAULT NOW()` (016)
- `sealed_by TEXT DEFAULT 'unknown'` (016)
- `header JSONB NOT NULL` (004, block metadata)
- `statements JSONB NOT NULL` (004, array of statement hashes)
- `canonical_statements JSONB NOT NULL DEFAULT '[]'::jsonb` (016, normalized statement payload)
- `canonical_proofs JSONB NOT NULL DEFAULT '[]'::jsonb` (016, normalized proof payload)
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()` (004)

**Evidence**:
- `ledger/ingest.py::_persist_block()` inserts blocks with all these columns (lines 600-660)
- `tests/integration/test_first_organism.py::_assert_composite_root_recomputable()` verifies H_t = SHA256(R_t || U_t)
- `tests/integration/test_first_organism.py::_assert_block_linkages()` queries `block_statements` and `block_proofs`

**Constraints**:
- `blocks_composite_requires_dual_roots` (015): If `composite_attestation_root` is set, both `reasoning_merkle_root` and `ui_merkle_root` must exist

**Indexes** (015):
- `blocks_reasoning_merkle_root_idx` (WHERE reasoning_merkle_root IS NOT NULL)
- `blocks_ui_merkle_root_idx` (WHERE ui_merkle_root IS NOT NULL)
- `blocks_composite_attestation_root_idx` (WHERE composite_attestation_root IS NOT NULL)

---

### 6. `block_statements` (Migration 016)

**Purpose**: Normalized block-to-statement linkage table

**Required Columns**:
- `block_id BIGINT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE`
- `position INT NOT NULL`
- `statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE`
- `statement_hash TEXT NOT NULL`
- `PRIMARY KEY (block_id, position)`

**Unique Constraint**:
- `block_statements_unique` on `(block_id, statement_id)`

**Evidence**:
- `ledger/ingest.py::_persist_block_links()` inserts into this table (lines 713-722)
- `tests/integration/test_first_organism.py::_assert_block_linkages()` queries this table (line 317)

---

### 7. `block_proofs` (Migration 016)

**Purpose**: Normalized block-to-proof linkage table

**Required Columns**:
- `block_id BIGINT NOT NULL REFERENCES blocks(id) ON DELETE CASCADE`
- `position INT NOT NULL`
- `proof_id UUID NOT NULL REFERENCES proofs(id) ON DELETE CASCADE`
- `proof_hash TEXT NOT NULL`
- `statement_id UUID NOT NULL REFERENCES statements(id) ON DELETE CASCADE`
- `PRIMARY KEY (block_id, position)`

**Unique Constraint**:
- `block_proofs_unique` on `(block_id, proof_id)`

**Evidence**:
- `ledger/ingest.py::_persist_block_links()` inserts into this table (lines 724-732)
- `tests/integration/test_first_organism.py::_assert_block_linkages()` queries this table (line 322)

---

### 8. `ledger_sequences` (Migration 016)

**Purpose**: Track block chain state per system

**Required Columns**:
- `system_id UUID PRIMARY KEY REFERENCES theories(id)`
- `height BIGINT NOT NULL DEFAULT 0` (current block number)
- `prev_block_id BIGINT REFERENCES blocks(id) ON DELETE SET NULL`
- `prev_block_hash TEXT`
- `prev_composite_root TEXT` (H_t of previous block)
- `run_id BIGINT REFERENCES runs(id) ON DELETE SET NULL`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

**Evidence**:
- `ledger/ingest.py::_persist_block()` updates this table (lines 673-693)
- Used to track block chain state and compute next `block_number`

---

### 9. `runs` (Migration 004, 013)

**Purpose**: Execution runs for batch processing

**Required Columns**:
- `id BIGSERIAL PRIMARY KEY`
- `name TEXT`
- `system_id UUID NOT NULL REFERENCES theories(id)`
- `status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')) DEFAULT 'running'`
- `started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`
- `completed_at TIMESTAMPTZ`
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`

**Evidence**:
- `ledger/ingest.py::_ensure_run()` creates/retrieves runs
- `blocks.run_id` references this table (may be NULL for legacy blocks)

---

### 10. `proof_parents` (In baseline, NOT in numbered migrations)

**Purpose**: Track proof dependency edges (child_hash → parent_hash)

**Required Columns**:
- `id SERIAL PRIMARY KEY` (baseline version)
- `child_hash TEXT NOT NULL` (baseline version) OR `VARCHAR(64) NOT NULL` (smoke test version)
- `parent_hash TEXT NOT NULL` (baseline version) OR `VARCHAR(64) NOT NULL` (smoke test version)
- `created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()` (baseline version) OR `TIMESTAMPTZ DEFAULT NOW()` (smoke test version)

**Indexes**:
- `proof_parents_child_hash_idx ON proof_parents(child_hash)` (baseline)
- `proof_parents_parent_hash_idx ON proof_parents(parent_hash)` (baseline)
- OR `idx_pp_child ON proof_parents(child_hash)` (smoke test)
- OR `idx_pp_parent ON proof_parents(parent_hash)` (smoke test)

**Evidence**:
- `migrations/baseline_20251019.sql` creates this table (lines 346-354)
- `scripts/run_fol_smoke.py` creates this table on-the-fly if missing (lines 127-135)
- Used by FOL smoke test to track proof dependencies

**Status**: ⚠️ **INCONSISTENCY**: 
- Table exists in `baseline_20251019.sql` but NOT in numbered migrations (000-017)
- `scripts/run-migrations.py` (current runner) does NOT run baseline migration
- `scripts/run-migrations-updated.py` (alternative runner) supports baseline detection
- Smoke test creates table on-the-fly if missing (works but inconsistent)

**Recommendation**: Add `proof_parents` to a numbered migration (e.g., `018_proof_parents.sql`) OR ensure baseline migration is part of standard migration path.

---

## Optional Tables (Used by Some Tests)

### `policy_settings` (Migration 006)

**Purpose**: Policy configuration storage

**Required Columns** (if used):
- `id SERIAL PRIMARY KEY`
- `key TEXT`
- `value TEXT`
- `policy_hash VARCHAR(64)`
- `created_at TIMESTAMPTZ DEFAULT NOW()`

**Evidence**: `scripts/bootstrap_db.py` creates this table, but FO tests don't require it.

---

## Migration Dependency Graph

**Phase I Minimal Path**:
```
000_schema_version.sql (schema_migrations)
  ↓
001_init.sql (theories, statements, proofs base)
  ↓
003_add_system_id.sql (system_id columns)
  ↓
004_finalize_core_schema.sql (consolidates + adds runs, blocks base)
  ↓
015_dual_root_attestation.sql (adds R_t, U_t, H_t to blocks)
  ↓
016_monotone_ledger.sql (adds block_statements, block_proofs, ledger_sequences, block chain columns)
```

**Total**: 6 migrations required for Phase I.

---

## Inconsistencies Identified (NOT Fixed)

### 1. `proof_parents` Table Missing from Migrations

**Issue**: `scripts/run_fol_smoke.py` creates `proof_parents` table on-the-fly (lines 127-135), but no migration defines it.

**Impact**: 
- FOL smoke test works (creates table if missing)
- FO tests don't use this table
- Inconsistent with migration-driven schema approach

**Recommendation**: Add `proof_parents` to a new migration (e.g., `018_proof_parents.sql`) or include in `016_monotone_ledger.sql` if retrofitting.

---

### 2. Legacy `systems` vs Canonical `theories` Table

**Issue**: 
- `scripts/run_fol_smoke.py::_ensure_system()` uses `systems` table (SERIAL id, name TEXT)
- Canonical schema uses `theories` table (UUID id, name TEXT, slug TEXT)

**Impact**:
- Smoke test works with either (flexible column detection)
- FO tests use `theories` table via `LedgerIngestor`
- No migration creates `systems` table (only `bootstrap_db.py`)

**Recommendation**: 
- Deprecate `systems` table
- Migrate `systems` → `theories` if legacy data exists
- Update `scripts/run_fol_smoke.py` to use `theories` only

---

### 3. Hash Column Type Inconsistency

**Issue**:
- Canonical schema: `statements.hash BYTEA`
- Legacy code: `statements.hash VARCHAR(64)` or `statements.canonical_hash VARCHAR(64)`
- `scripts/run_fol_smoke.py` supports both (flexible column detection)

**Impact**:
- Migration 008 attempts to normalize hash columns
- FO tests expect BYTEA (via `LedgerIngestor`)
- Smoke test works with either

**Recommendation**: Ensure all migrations use BYTEA for hash columns.

---

### 4. Status vs Success Column Inconsistency

**Issue**:
- Canonical schema: `proofs.success BOOLEAN`
- Legacy code: `proofs.status TEXT`
- `scripts/run_fol_smoke.py` supports both

**Impact**:
- Migration 007 attempts to fix proofs schema
- FO tests expect `success BOOLEAN` (via `LedgerIngestor`)
- Smoke test works with either

**Recommendation**: Ensure all migrations use `success BOOLEAN` for proofs.

---

### 5. Unused Migrations (000-016)

**Migrations NOT Required for Phase I**:
- `002_add_axioms.sql` - Seed data only
- `002_blocks_lemmas.sql` - Legacy blocks table (superseded by 004)
- `003_fix_progress_compatibility.sql` - Progress tracking (not used by FO)
- `005_add_search_indexes.sql` - Performance indexes (optional)
- `006_add_pg_trgm_extension.sql` - Text search extension (optional)
- `006_add_policy_settings.sql` - Policy settings (optional)
- `007_fix_proofs_schema.sql` - Schema fixes (may be redundant with 004)
- `008_fix_statements_hash.sql` - Hash normalization (may be redundant with 004)
- `009_normalize_statements.sql` - Statement normalization (may be redundant with 004)
- `010_idempotent_normalize.sql` - Normalization fixes (may be redundant with 004)
- `011_schema_parity.sql` - Schema parity fixes (may be redundant with 004)
- `012_blocks_parity.sql` - Blocks parity fixes (may be redundant with 004)
- `013_runs_logging.sql` - Runs logging (may be redundant with 004)
- `014_ensure_slug_column.sql` - Slug column (may be redundant with 004)
- `017_enforce_app_timestamps.sql` - Timestamp enforcement (optional)

**Status**: These migrations may be redundant if `004_finalize_core_schema.sql` already includes their changes, or they may be needed for specific edge cases. **Not verified** - requires schema diff analysis.

---

## Verification Checklist

- [x] `schema_migrations` table exists (pre-flight check)
- [x] `theories` table with UUID id, name, slug
- [x] `statements` table with UUID id, hash BYTEA, content_norm, system_id, status, derivation_rule, derivation_depth
- [x] `proofs` table with UUID id, statement_id, system_id, prover, method, success BOOLEAN
- [x] `blocks` table with all attestation columns (R_t, U_t, H_t)
- [x] `block_statements` table for normalized linkages
- [x] `block_proofs` table for normalized linkages
- [x] `ledger_sequences` table for block chain state
- [x] `runs` table for execution runs
- [ ] `proof_parents` table (missing from migrations)

---

## RFL Evidence: Phase-I vs Phase-II Database Usage

### Phase-I RFL (Hermetic, File-Based Only)

**Operating Mode**: File-based JSONL output, no database writes

**Artifacts** (canonical Phase-I RFL evidence):
- `results/fo_baseline.jsonl` - 1000 cycles (0-999), old schema, 100% abstention
- `results/fo_rfl.jsonl` - **1001 cycles (0-1000)**, new schema, 100% abstention, hermetic negative-control / plumbing validation
- `results/fo_rfl_50.jsonl` - **21 cycles (0-20), INCOMPLETE**, new schema, 100% abstention, small RFL plumbing demo
- `results/fo_rfl_1000.jsonl` - **11 cycles (0-10), INCOMPLETE**, new schema, 100% abstention, do not use for evidence

**Critical Phase-I Facts**:
- **All Phase-I RFL logs are 100% abstention by design** (lean-disabled mode, ML_ENABLE_LEAN_FALLBACK is OFF)
- **Zero empirical RFL uplift in Phase-I** - every RFL log demonstrates plumbing/execution infrastructure only
- **Purpose**: Validates RFL wiring, attestation, and determinism in hermetic negative-control regime
- **Not evidence of**: Uplift, metabolism verification, performance improvement, or reduced abstention

**Database Operations**: **NONE**
- No INSERT operations
- No UPDATE operations
- No table writes of any kind
- Purely file-based output

**Evidence**:
- `experiments/run_fo_cycles.py` outputs JSONL lines to file
- Each JSONL line contains: `abstention`, `cycle`, `derivation`, `gates_passed`, `method`, `mode`, `rfl`, `roots` (h_t, r_t, u_t), `slice_name`, `status`
- All cycles show `method="lean-disabled"` and `abstention=true` (100% abstention rate)
- No database connection code in Phase-I RFL execution path
- `rfl/runner.py::RFLRunner` has `database_url` config but Phase-I uses file-based output only

**Tables Unused in Phase-I**:
- `statements` - NOT written, NOT read
- `proofs` - NOT written, NOT read
- `blocks` - NOT written, NOT read
- `runs` - NOT written, NOT read
- `rfl_runs` - NOT written, NOT read (if exists)
- `experiment_results` - NOT written, NOT read (if exists)

---

### Phase-II RFL (Database-Backed)

**Operating Mode**: Database writes for statements, proofs, blocks; database reads for metrics collection

**Database Tables That Would Be Touched**:

#### 1. `statements` (WRITE + READ)
- **Write**: `experiments/rfl/derive_wrapper.py` inserts derived statements via derivation engine
- **Read**: `rfl/experiment.py::RFLExperiment._collect_metrics()` queries statements created during experiment window
- **Read**: `rfl/experiment.py::RFLExperiment._get_statement_count()` counts baseline statements
- **Read**: `rfl/coverage.py::load_baseline_from_db()` loads baseline statement hashes

**Evidence**: `rfl/experiment.py` lines 313-316, 348-358, 304-321

#### 2. `proofs` (WRITE + READ)
- **Write**: `experiments/rfl/derive_wrapper.py` inserts proof records via derivation engine
- **Read**: `rfl/experiment.py::RFLExperiment._collect_metrics()` queries proofs created during experiment window (lines 361-373)

**Evidence**: `rfl/experiment.py` lines 361-373

#### 3. `blocks` (WRITE, optional)
- **Write**: `experiments/rfl/derive_wrapper.py` may seal blocks if `--seal` flag is used
- **Read**: Not directly queried by RFL metrics, but may be used for block chain analysis

**Evidence**: Derivation wrapper may call block sealing logic

#### 4. `runs` (WRITE, potential)
- **Write**: May insert run records to track RFL experiment execution
- **Read**: May query run status/history

**Evidence**: `rfl/experiment.py::RFLExperiment` uses `run_id` but doesn't explicitly write to `runs` table in current code

#### 5. `theories` (READ)
- **Read**: `rfl/experiment.py` uses `system_id` which maps to `theories.id`
- **Read**: `backend/axiom_engine/derive_utils::get_or_create_system_id()` queries/creates theory records

**Evidence**: `rfl/experiment.py` lines 309-311, 343-345

#### 6. Hypothetical RFL-Specific Tables (NOT in current migrations)
- `rfl_runs` - Would track RFL experiment suite execution
- `rfl_experiment_results` - Would store `ExperimentResult` records
- `rfl_policy_ledger` - Would store `RunLedgerEntry` records

**Status**: These tables do NOT exist in migrations 000-017. Phase-II would require new migrations to create them.

---

### Summary: Phase-I vs Phase-II

| Table | Phase-I (Hermetic) | Phase-II (DB-Backed) |
|-------|-------------------|---------------------|
| `statements` | ❌ Unused | ✅ WRITE (derivation) + READ (metrics) |
| `proofs` | ❌ Unused | ✅ WRITE (derivation) + READ (metrics) |
| `blocks` | ❌ Unused | ✅ WRITE (optional sealing) |
| `runs` | ❌ Unused | ⚠️ WRITE (potential, not in current code) |
| `theories` | ❌ Unused | ✅ READ (system_id lookup) |
| `rfl_runs` | ❌ Unused | ⚠️ WRITE (hypothetical, table doesn't exist) |
| `rfl_experiment_results` | ❌ Unused | ⚠️ WRITE (hypothetical, table doesn't exist) |

**Key Distinction**:
- **Phase-I RFL**: Zero database schema requirements. Purely file-based JSONL output. **100% abstention by design** (lean-disabled, hermetic negative-control). Validates execution infrastructure and attestation only; **not evidence of uplift or performance improvement**.
- **Phase-II RFL**: Requires `statements`, `proofs`, `theories` tables (already in Phase-I schema). May require new migrations for RFL-specific tables. Would enable actual database-backed derivation and metrics collection for uplift analysis.

**Migration Impact**:
- Phase-I RFL schema requirements: **NONE** (file-based only, no database writes)
- Phase-II RFL schema requirements: Uses Phase-I tables (`statements`, `proofs`, `theories`) + potential new RFL-specific tables

---

## Summary

**Minimal Migration Set for Phase I** (using `scripts/run-migrations.py`): 000, 001, 003, 004, 015, 016 (6 migrations)

**Alternative Path** (using `scripts/run-migrations-updated.py` with baseline): 
- `baseline_20251019.sql` (consolidates 001-014) + 015, 016
- Includes `proof_parents` table (resolves inconsistency #1)

**Critical Inconsistencies**:
1. `proof_parents` table not in migrations
2. Legacy `systems` vs canonical `theories` table
3. Hash column type inconsistency (BYTEA vs VARCHAR)
4. Status vs success column inconsistency (BOOLEAN vs TEXT)

**Action Items** (NOT in scope for Phase I):
- Add `proof_parents` to migration
- Deprecate `systems` table
- Standardize hash columns to BYTEA
- Standardize proof status to BOOLEAN
- Audit unused migrations (002-014, 017) for redundancy

---

**End of Document**

