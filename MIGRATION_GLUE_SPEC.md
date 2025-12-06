# Migration Glue Specification
## First Organism Database Fixture Integration

**Status**: Specification Only (Not Implemented)  
**Author**: GEMINI G (Reviewer-2 Mode)  
**Date**: 2025-01-XX  
**Evidence Base**: `scripts/run-migrations.py`, `tests/integration/conftest.py`, `tests/conftest.py`

---

## Executive Summary

This specification defines the interface and flow for integrating database migrations into the First Organism test fixture (`first_organism_db`). The goal is to replace the current ad-hoc migration execution (direct SQL file execution) with a principled, idempotent migration runner that respects the `schema_migrations` tracking table.

**Current State** (Evidence):
- `first_organism_db` fixture (lines 711-767 in `tests/integration/conftest.py`) has migration execution commented out
- Comment states: "Migration execution removed - handled by root tests/conftest.py"
- `tests/conftest.py` has `_run_migrations_once()` that calls `scripts/run-migrations.py` via subprocess
- `scripts/run-migrations.py` has partial infrastructure: `check_all_migrations_applied()`, `run_migration_file()` with optional params
- Missing: Complete `run_all_migrations()` function that can be imported and called programmatically

**Target State**:
- `first_organism_db` fixture calls migration runner programmatically (not subprocess)
- Migration runner checks `schema_migrations` table before executing
- If all migrations already applied, skip execution (but still truncate data)
- If migrations needed, run all migrations in order via library function
- All operations are idempotent and transaction-safe

---

## Function Signatures

### 1. `run_all_migrations()`

**Location**: `scripts/run-migrations.py` (to be extracted/created)

**Signature**:
```python
def run_all_migrations(
    db_url: Optional[str] = None,
    conn: Optional[psycopg.Connection] = None,
    quiet: bool = False,
    migrations_dir: Optional[Path] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Run all database migrations in order with version tracking.
    
    This function is idempotent: migrations already recorded in schema_migrations
    are skipped automatically.
    
    Args:
        db_url: Optional database URL. If provided, creates new connection.
                Mutually exclusive with `conn`. Defaults to DATABASE_URL env var.
        conn: Optional existing database connection. Mutually exclusive with `db_url`.
              If provided, uses this connection (does not close it).
        quiet: If True, suppress print output (useful for test fixtures).
        migrations_dir: Optional path to migrations directory. Defaults to "migrations/".
    
    Returns:
        Tuple of:
        - successful_migrations: List[str] - Names of successfully applied migrations
        - failed_migrations: List[str] - Names of failed migrations (empty if all succeeded)
        - metadata: Dict[str, Any] - Execution metadata:
            {
                "total_files": int,
                "already_applied": int,
                "newly_applied": int,
                "failed": int,
                "duration_ms": int
            }
    
    Raises:
        RuntimeError: If neither db_url nor conn provided, and DATABASE_URL not set
        psycopg.Error: Database connection or execution errors
    
    Behavior:
        - Ensures schema_migrations table exists (creates if missing)
        - Discovers all .sql files in migrations_dir (excludes baseline_*.sql)
        - Sorts migrations by numeric prefix (000, 001, 002, ...)
        - For each migration:
            - Checks schema_migrations table for existing record
            - If already applied (status='success'), skips execution
            - If not applied, executes migration in transaction
            - Records result in schema_migrations table
        - Stops on first failure (fail-fast behavior)
        - Returns summary of execution
    """
```

**Implementation Notes**:
- Must handle both connection modes: new connection (db_url) or existing connection (conn)
- If using existing connection, must respect its autocommit/transaction state
- Should not commit/rollback if using existing connection (caller controls transaction)
- If creating new connection, manages transaction lifecycle internally

**Dependencies**:
- Uses existing `migration_already_applied(cur, version: str) -> bool`
- Uses existing `record_migration(cur, version, checksum, duration_ms, status)`
- Uses existing `run_migration_file()` (with modifications for connection handling)

---

### 2. `check_all_migrations_applied()`

**Location**: `scripts/run-migrations.py` (already exists, may need refinement)

**Current Signature** (lines 83-108):
```python
def check_all_migrations_applied(conn: psycopg.Connection) -> Tuple[bool, Optional[int]]:
    """
    Check if all migrations have been applied.
    
    Returns:
        Tuple of (all_applied: bool, count: Optional[int])
        count is None if schema_migrations table doesn't exist
    """
```

**Specification Refinement**:
```python
def check_all_migrations_applied(
    conn: psycopg.Connection,
    migrations_dir: Optional[Path] = None
) -> Tuple[bool, Optional[int], Optional[List[str]]]:
    """
    Check if all migrations have been applied by comparing schema_migrations
    records against migration files on disk.
    
    Args:
        conn: Database connection
        migrations_dir: Optional path to migrations directory. Defaults to "migrations/".
    
    Returns:
        Tuple of:
        - all_applied: bool - True if all migration files have corresponding success records
        - applied_count: Optional[int] - Number of successful migrations recorded
                                    (None if schema_migrations table doesn't exist)
        - missing_versions: Optional[List[str]] - List of migration file names not yet applied
                          (None if schema_migrations table doesn't exist)
    
    Behavior:
        - Discovers all .sql files in migrations_dir (excludes baseline_*.sql)
        - Queries schema_migrations table for all records with status='success'
        - Compares file list against recorded versions
        - Returns True only if every migration file has a corresponding success record
    """
```

**Rationale**:
- Current implementation uses count comparison (may be fragile if migrations are skipped)
- Refined version compares actual version strings for robustness
- Returns missing versions list for diagnostic purposes

---

### 3. `first_organism_db` Fixture

**Location**: `tests/integration/conftest.py` (lines 711-767, to be updated)

**Current Behavior** (Evidence):
- Migration execution is commented out
- Relies on `tests/conftest.py` to run migrations via subprocess
- Only performs table truncation

**Target Behavior** (Specification):
```python
@pytest.fixture(scope="function")
def first_organism_db(
    test_db_connection: psycopg.Connection,
    environment_mode: EnvironmentMode,
    test_db_url: str,
) -> Generator[psycopg.Connection, None, None]:
    """
    Prepare database for First Organism tests.
    
    Flow:
    1. Gate check: assert_first_organism_ready() (existing)
    2. Migration check: check_all_migrations_applied(conn)
    3. If not all applied: run_all_migrations(conn=test_db_connection, quiet=True)
    4. Truncate data tables (preserve schema)
    5. Yield connection
    6. Cleanup: rollback + truncate (existing)
    
    Skips with explicit [SKIP][FO] reason if:
    - FIRST_ORGANISM_TESTS not enabled
    - Mock mode detected
    - Database unavailable
    - Migration execution fails
    
    This fixture ensures:
    - Schema is up-to-date (all migrations applied)
    - Data is clean (tables truncated)
    - Operations are idempotent (safe to re-run)
    """
    # Step 1: Gate check (existing)
    assert_first_organism_ready(environment_mode, test_db_url)
    
    # Step 2: Import migration functions
    from scripts.run_migrations import run_all_migrations, check_all_migrations_applied
    
    # Step 3: Check migration state
    all_applied, applied_count, missing_versions = check_all_migrations_applied(
        test_db_connection
    )
    
    # Step 4: Run migrations if needed
    if not all_applied:
        if missing_versions:
            # Log which migrations need to be applied (for diagnostics)
            print(f"[FO] Applying {len(missing_versions)} migrations: {missing_versions[:3]}...")
        
        successful, failed, metadata = run_all_migrations(
            conn=test_db_connection,
            quiet=True  # Suppress output in test fixtures
        )
        
        if failed:
            db_url_trimmed = _trim_url_for_display(test_db_url)
            pytest.skip(
                f"[SKIP][FO] Migrations failed: {failed} "
                f"(mode=<migration_failed>, db_url=<{db_url_trimmed}>)"
            )
    else:
        # All migrations already applied - skip execution
        if applied_count is not None:
            print(f"[FO] All {applied_count} migrations already applied, skipping migration run")
    
    # Step 5: Truncate data tables (existing logic)
    original_autocommit = test_db_connection.autocommit
    test_db_connection.autocommit = True
    
    try:
        with test_db_connection.cursor() as cur:
            cur.execute(
                """
                TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
                         proofs, dependencies, statements, runs, theories
                RESTART IDENTITY CASCADE
                """
            )
    except Exception as e:
        test_db_connection.autocommit = original_autocommit
        db_url_trimmed = _trim_url_for_display(test_db_url)
        pytest.skip(
            f"[SKIP][FO] Table truncation failed: {e} "
            f"(mode=<truncation_error>, db_url=<{db_url_trimmed}>)"
        )
    
    test_db_connection.autocommit = original_autocommit
    
    # Step 6: Yield connection
    yield test_db_connection
    
    # Step 7: Cleanup (existing logic)
    test_db_connection.rollback()
    try:
        with test_db_connection.cursor() as cur:
            cur.execute(
                """
                TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
                         proofs, dependencies, statements, runs, theories
                RESTART IDENTITY CASCADE
                """
            )
    except Exception:
        pass  # Best effort cleanup
```

**Key Design Decisions**:
1. **Use existing connection**: Pass `conn=test_db_connection` to avoid connection management complexity
2. **Idempotent**: Check before running - if all applied, skip execution
3. **Quiet mode**: Suppress migration output in test fixtures (reduce noise)
4. **Fail-fast**: If migrations fail, skip test with clear message
5. **Preserve truncation**: Still truncate data tables after migrations (test isolation)

---

## Integration Points

### A. Connection Management

**Challenge**: Migration runner may create its own connection, but fixture already has a connection.

**Solution**: Support both modes:
- `db_url` parameter: Create new connection (for CLI usage)
- `conn` parameter: Use existing connection (for fixture usage)

**Transaction Handling**:
- If using `conn`, do NOT commit/rollback (caller controls transaction)
- If using `db_url`, manage transaction lifecycle internally
- Migration execution within `first_organism_db` should respect fixture's autocommit state

### B. Error Handling

**Migration Failures**:
- If `run_all_migrations()` returns non-empty `failed` list, fixture should skip test
- Error message should include which migrations failed
- Should not attempt to continue with partial migration state

**Connection Errors**:
- If connection fails, existing skip logic in `assert_first_organism_ready()` handles it
- Migration runner should propagate connection errors (not swallow them)

### C. Idempotency

**Requirement**: Running migrations multiple times must be safe.

**Mechanism**:
- `schema_migrations` table tracks applied migrations
- `migration_already_applied()` checks before execution
- Already-applied migrations are skipped automatically

**Verification**:
- Running `run_all_migrations()` twice on same database should:
  - First run: Apply all migrations
  - Second run: Skip all migrations (already applied)
  - Both runs: Return same final state

---

## Migration File Discovery

**Current Behavior** (Evidence from `scripts/run-migrations.py`):
- Discovers all `.sql` files in `migrations/` directory
- Excludes `baseline_*.sql` files
- Sorts by numeric prefix using regex: `re.findall(r'\d+', x.name)[0]`

**Specification**:
```python
def discover_migration_files(migrations_dir: Path) -> List[Path]:
    """
    Discover and sort migration files.
    
    Args:
        migrations_dir: Path to migrations directory
    
    Returns:
        Sorted list of migration file paths
    
    Rules:
        - Includes all .sql files
        - Excludes baseline_*.sql files
        - Sorts by numeric prefix (000, 001, 002, ...)
        - Files without numeric prefix sorted last (999)
    """
```

**Migration Files** (Evidence from `list_dir`):
- 18 migration files: `000_schema_version.sql` through `017_enforce_app_timestamps.sql`
- 1 baseline file: `baseline_20251019.sql` (excluded)

---

## Schema Migrations Table Schema

**Evidence** (from `migrations/000_schema_version.sql`):
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum TEXT,  -- SHA-256 of migration file contents
    duration_ms INTEGER,  -- Migration execution time
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'skipped'))
);
```

**Usage**:
- `version`: Migration file stem (e.g., "001_init", "016_monotone_ledger")
- `status`: 'success' indicates migration applied successfully
- `checksum`: SHA-256 hash of migration file (for integrity checking)
- `applied_at`: Timestamp of application (for audit trail)

**Query Pattern** (for `check_all_migrations_applied()`):
```sql
SELECT version FROM schema_migrations WHERE status = 'success'
```

---

## Docker Compose Integration

**Current State** (Evidence from `ops/first_organism/docker-compose.yml`):
- Docker compose file exists
- No migration instructions in comments

**Specification** (Documentation Addition):
```markdown
## Database Initialization

After starting the compose stack, run migrations once:

```bash
# Set DATABASE_URL from .env.first_organism
export $(cat .env.first_organism | grep DATABASE_URL | xargs)

# Run migrations (idempotent - safe to re-run)
uv run python scripts/run-migrations.py
```

Alternatively, use the convenience script:

```bash
uv run python scripts/init_first_organism_db.py
```

Migrations are tracked in the `schema_migrations` table. Re-running migrations
is safe - already-applied migrations are automatically skipped.
```

**Convenience Script** (`scripts/init_first_organism_db.py` - Not Implemented):
```python
#!/usr/bin/env python3
"""
Initialize First Organism database with migrations.

Reads DATABASE_URL from .env.first_organism and runs all migrations.
"""

import os
from pathlib import Path
from scripts.run_migrations import run_all_migrations

def main():
    # Load .env.first_organism
    env_file = Path(".env.first_organism")
    if not env_file.exists():
        raise RuntimeError(".env.first_organism not found")
    
    # Parse DATABASE_URL
    db_url = None
    with open(env_file) as f:
        for line in f:
            if line.startswith("DATABASE_URL="):
                db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    
    if not db_url:
        raise RuntimeError("DATABASE_URL not found in .env.first_organism")
    
    # Run migrations
    successful, failed, metadata = run_all_migrations(db_url=db_url)
    
    if failed:
        print(f"❌ Failed migrations: {failed}")
        exit(1)
    
    print(f"✅ All migrations applied successfully")
    print(f"   Applied: {metadata['newly_applied']} new, {metadata['already_applied']} skipped")

if __name__ == "__main__":
    main()
```

---

## Testing Requirements

**Test Scenarios** (Not Implemented - Specification Only):

1. **Fresh Database**:
   - Database has no `schema_migrations` table
   - `run_all_migrations()` should create table and apply all migrations
   - `check_all_migrations_applied()` should return `(False, None, [all_versions])`

2. **Fully Migrated Database**:
   - All migrations already applied
   - `check_all_migrations_applied()` should return `(True, 18, [])`
   - `run_all_migrations()` should skip all migrations

3. **Partially Migrated Database**:
   - Some migrations applied (e.g., 000-010)
   - `check_all_migrations_applied()` should return `(False, 11, [011, 012, ...])`
   - `run_all_migrations()` should apply only remaining migrations

4. **Fixture Integration**:
   - `first_organism_db` fixture should work in all three scenarios above
   - Should truncate data tables after migrations
   - Should not re-run migrations if already applied

5. **Error Handling**:
   - Migration failure should cause fixture to skip test
   - Connection error should be handled by existing skip logic
   - Partial migration state should not leave database inconsistent

---

## Implementation Checklist

**Not Implemented** - This is a specification document only.

**Required Changes**:

- [ ] Extract `run_all_migrations()` function from `main()` in `scripts/run-migrations.py`
- [ ] Refine `check_all_migrations_applied()` to return missing versions list
- [ ] Update `run_migration_file()` to support `conn` parameter (existing connection)
- [ ] Update `first_organism_db` fixture to use migration runner programmatically
- [ ] Create `scripts/init_first_organism_db.py` convenience script
- [ ] Add migration instructions to `ops/first_organism/docker-compose.yml`
- [ ] Add tests for migration runner in various database states
- [ ] Verify idempotency (running migrations twice produces same result)

---

## Evidence Base

**Files Referenced**:
- `scripts/run-migrations.py` (lines 83-108, 110-198)
- `tests/integration/conftest.py` (lines 711-767)
- `tests/conftest.py` (lines 28-39)
- `migrations/000_schema_version.sql` (schema definition)
- `ops/first_organism/docker-compose.yml` (infrastructure)

**Current State Summary**:
- Migration script has partial infrastructure (check function, run function with optional params)
- Fixture has migration execution commented out (relies on subprocess call in root conftest)
- No programmatic interface for running migrations from fixtures
- No convenience script for FO database initialization

**Gap Analysis**:
- Missing: Complete `run_all_migrations()` function that can be imported
- Missing: Connection parameter support in migration runner
- Missing: Fixture integration (currently uses subprocess)
- Missing: Convenience script for FO setup
- Missing: Documentation in docker-compose file

---

## Reviewer-2 Validation

**Claims Verified**:
- ✅ `schema_migrations` table exists (verified in `migrations/000_schema_version.sql`)
- ✅ `check_all_migrations_applied()` exists (verified in `scripts/run-migrations.py:83`)
- ✅ `run_migration_file()` accepts optional params (verified in `scripts/run-migrations.py:110`)
- ✅ `first_organism_db` fixture exists (verified in `tests/integration/conftest.py:711`)
- ✅ Migration execution is commented out in fixture (verified in `tests/integration/conftest.py:750`)
- ✅ Root conftest calls migrations via subprocess (verified in `tests/conftest.py:34`)

**Claims NOT Verified** (Hypothetical):
- ❌ `run_all_migrations()` function does NOT exist as importable function (only in `main()`)
- ❌ Fixture does NOT use programmatic migration runner (uses subprocess)
- ❌ No convenience script exists for FO initialization

**Consistency Check**:
- Migration file count: 18 files (000-017) + 1 baseline = 19 total
- Schema matches: `schema_migrations` table structure matches usage in code
- Fixture flow: Current fixture truncates tables but doesn't run migrations (relies on root conftest)

---

## Conclusion

This specification defines the interface and flow for integrating database migrations into the First Organism test fixture. The design is:

- **Idempotent**: Safe to run multiple times
- **Principled**: Uses `schema_migrations` table as source of truth
- **Testable**: Clear function signatures enable unit testing
- **Maintainable**: Separation of concerns (migration runner vs. fixture)

**Next Steps** (When Authorized):
1. Implement `run_all_migrations()` extraction
2. Update fixture to use programmatic runner
3. Create convenience script
4. Add documentation
5. Test in various database states

**Status**: Specification Complete - Awaiting Implementation Authorization

---

## Appendix A: Phase-I Hermetic RFL Interaction

**Status**: Current State (Evidence-Based)  
**Date**: 2025-01-XX  
**Evidence Base**: `rfl/runner.py`, `rfl/config.py`, `backend/rfl/README.md`, `migrations/` directory

---

### Phase-I State: File-Based RFL Evidence

**Current Implementation** (Evidence Verified):

RFL (Reflexive Formal Learning) experiments in Phase-I operate in **hermetic mode** with file-based evidence storage only. Database migrations are **unaffected** by RFL operations.

**Critical Phase-I Truth** (Canonical Facts):
- **100% Abstention**: All Phase-I RFL runs exhibit 100% abstention due to hermetic lean-disabled mode (`ML_ENABLE_LEAN_FALLBACK` is OFF)
- **Negative Control / Plumbing Only**: Phase-I RFL logs (e.g., `fo_rfl.jsonl`, `fo_rfl_50.jsonl`) are hermetic negative-control runs that validate execution infrastructure and attestation wiring only
- **Zero Empirical Uplift**: Phase-I has zero empirical RFL uplift. Every RFL log demonstrates 100% abstention by design
- **Purpose**: Phase-I proves that RFL plumbing + attestation + determinism work in a hermetic, lean-disabled negative-control regime. It does NOT demonstrate performance improvement, reduced abstention, or metabolism verification

**Evidence**:
- ✅ RFL writes to file-based artifacts (verified in `rfl/config.py:116-119`):
  - `artifacts_dir: str = "artifacts/rfl"`
  - `results_file: str = "rfl_results.json"`
  - `coverage_file: str = "rfl_coverage.json"`
  - `curves_file: str = "rfl_curves.png"`
- ✅ RFL uses JSONL logging (verified in `rfl/metrics_logger.py:59`):
  - `results/rfl_wide_slice_runs.jsonl`
- ✅ No RFL-specific database tables exist (verified via `grep -r "rfl_runs\|CREATE TABLE.*rfl" migrations/` - no matches)
- ✅ RFL may read from existing `runs` table (general purpose, not RFL-specific)
- ✅ RFL configuration includes `database_url` field (verified in `backend/rfl/config.py:98-100`) but is used for reading ledger data, not writing RFL results

**Migration Impact**: **NONE**

- RFL Phase-I operations do not require database schema changes
- RFL evidence is stored in `artifacts/rfl/` directory structure
- Migration runner (`run_all_migrations()`) and `first_organism_db` fixture operate independently of RFL
- No RFL-specific tables need to be created, migrated, or truncated

**Truncation Behavior** (in `first_organism_db` fixture):

Current truncation list (line 752-757 in `tests/integration/conftest.py`):
```sql
TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
         proofs, dependencies, statements, runs, theories
RESTART IDENTITY CASCADE
```

**Note**: `runs` table is truncated, but this is the **general-purpose runs table** (used by ledger ingestion), not RFL-specific. RFL Phase-I does not write to this table for its own evidence.

**Phase-I RFL Evidence Files** (Canonical Reference):
- `fo_baseline.jsonl`: 1000 cycles (0-999), old schema, 100% abstention
- `fo_rfl_50.jsonl`: 21 cycles (0-20), **INCOMPLETE**, new schema, 100% abstention, negative control demo
- `fo_rfl_1000.jsonl`: 11 cycles (0-10), **INCOMPLETE**, new schema, 100% abstention, do not use for evidence
- `fo_rfl.jsonl`: 1001 cycles (0-1000), new schema, 100% abstention, hermetic negative control / plumbing run

**Important**: None of these Phase-I RFL logs demonstrate uplift, reduced abstention, or metabolism verification. They are infrastructure validation runs only.

---

### Phase-II Requirements Checklist

**Status**: Specification Only (Not Implemented)

When RFL transitions to Phase-II (database-backed evidence storage), the following must be added to the migration system:

#### A. Database Schema

- [ ] **Create `rfl_runs` table migration** (e.g., `018_rfl_runs_table.sql`)
  - Schema must include:
    - `experiment_id: TEXT` (references RFL experiment identifier)
    - `run_index: INTEGER` (1-indexed run number within experiment)
    - `slice_name: TEXT` (curriculum slice: "warmup", "core", "refinement")
    - `coverage_rate: REAL` (coverage metric for this run)
    - `proofs_success: INTEGER` (number of successful proofs)
    - `proofs_per_sec: REAL` (throughput metric)
    - `abstention_pct: REAL` (abstention rate)
    - `policy_reward: REAL` (RL policy reward)
    - `symbolic_descent: REAL` (symbolic descent metric)
    - `created_at: TIMESTAMPTZ` (timestamp)
    - Foreign key constraints as needed
  - Indexes for efficient querying:
    - `idx_rfl_runs_experiment_id` on `(experiment_id, run_index)`
    - `idx_rfl_runs_slice_name` on `(slice_name)`
    - `idx_rfl_runs_created_at` on `(created_at DESC)`

- [ ] **Create `rfl_experiments` table migration** (optional, for experiment metadata)
  - Schema must include:
    - `experiment_id: TEXT PRIMARY KEY`
    - `experiment_name: TEXT`
    - `num_runs: INTEGER`
    - `random_seed: INTEGER`
    - `coverage_threshold: REAL`
    - `uplift_threshold: REAL`
    - `status: TEXT` (e.g., "running", "completed", "failed")
    - `started_at: TIMESTAMPTZ`
    - `completed_at: TIMESTAMPTZ`
    - `metadata: JSONB` (flexible storage for config)

- [ ] **Create `rfl_coverage_stats` table migration** (optional, for aggregated statistics)
  - Schema must include:
    - `experiment_id: TEXT` (references `rfl_experiments`)
    - `statistic_type: TEXT` (e.g., "coverage", "uplift", "abstention")
    - `point_estimate: REAL`
    - `ci_lower: REAL`
    - `ci_upper: REAL`
    - `bootstrap_method: TEXT` (e.g., "BCa_95%")
    - `computed_at: TIMESTAMPTZ`

#### B. Migration Ordering Constraints

**Idempotent Migration Ordering** (Critical):

- [ ] **RFL migrations must come AFTER core schema migrations**
  - RFL tables may reference `theories` table (via `system_id`)
  - RFL tables may reference `runs` table (if Phase-II integrates with ledger)
  - Migration numbering: RFL migrations should be `018_*` or later (after `017_enforce_app_timestamps.sql`)

- [ ] **RFL migrations must be idempotent**
  - Use `CREATE TABLE IF NOT EXISTS` patterns
  - Use `DO $$ BEGIN ... END $$` blocks for conditional constraints
  - Use `ON CONFLICT` clauses for inserts
  - Record in `schema_migrations` table with status tracking

- [ ] **Migration dependency graph**:
  ```
  000_schema_version.sql (creates schema_migrations table)
  ↓
  001_init.sql (creates theories, statements, proofs)
  ↓
  ... (intermediate migrations)
  ↓
  017_enforce_app_timestamps.sql
  ↓
  018_rfl_runs_table.sql (NEW - Phase-II)
  ↓
  019_rfl_experiments_table.sql (NEW - Phase-II, optional)
  ↓
  020_rfl_coverage_stats_table.sql (NEW - Phase-II, optional)
  ```

#### C. Fixture Integration

- [ ] **Update `first_organism_db` fixture truncation list**
  - Add RFL tables to truncation (if Phase-II writes to DB):
    ```sql
    TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
             proofs, dependencies, statements, runs, theories,
             rfl_runs, rfl_experiments, rfl_coverage_stats  -- NEW
    RESTART IDENTITY CASCADE
    ```
  - **Decision point**: Should RFL evidence be truncated in test fixtures?
    - **Option A**: Truncate RFL tables (clean slate for each test)
    - **Option B**: Preserve RFL tables (allow test-to-test evidence accumulation)
    - **Recommendation**: Option A (truncate) for test isolation

- [ ] **Update `check_all_migrations_applied()` logic**
  - Must include RFL migrations in discovery (if they exist)
  - Must verify RFL tables exist (if Phase-II enabled)
  - Should return missing RFL migrations in `missing_versions` list

#### D. Migration Runner Updates

- [ ] **No changes required to `run_all_migrations()` function signature**
  - Existing function should handle RFL migrations automatically
  - Migration discovery already includes all `.sql` files (excludes `baseline_*.sql`)
  - Sorting by numeric prefix ensures correct ordering

- [ ] **Verification**: RFL migrations must be:
  - Discovered automatically (no special handling needed)
  - Sorted correctly (018, 019, 020 after 017)
  - Applied idempotently (safe to re-run)
  - Recorded in `schema_migrations` table

#### E. Documentation Updates

- [ ] **Update `ops/first_organism/docker-compose.yml` documentation**
  - Note that RFL Phase-II requires additional migrations
  - Migration command remains the same: `uv run python scripts/run-migrations.py`
  - RFL tables are created automatically if migrations exist

- [ ] **Update RFL documentation** (`backend/rfl/README.md` or equivalent)
  - Document Phase-II database schema
  - Document migration requirements
  - Document fixture truncation behavior

#### F. Testing Requirements (Phase-II)

- [ ] **Test RFL migration application**
  - Fresh database: RFL migrations apply successfully
  - Partially migrated: RFL migrations apply after core migrations
  - Already applied: RFL migrations skip (idempotent)

- [ ] **Test fixture integration**
  - `first_organism_db` fixture truncates RFL tables (if included)
  - RFL migrations are checked by `check_all_migrations_applied()`
  - RFL tables are accessible after fixture setup

- [ ] **Test migration ordering**
  - Verify RFL migrations run after `017_enforce_app_timestamps.sql`
  - Verify foreign key constraints work (if RFL tables reference `theories`)

---

### Phase-II Migration Example (Specification Only)

**Not Implemented** - This is a specification example:

```sql
-- migrations/018_rfl_runs_table.sql
-- Migration 018: RFL Runs Table (Phase-II)
-- Creates table for storing RFL experiment run evidence in database.

BEGIN;

-- Create rfl_runs table
CREATE TABLE IF NOT EXISTS rfl_runs (
    id BIGSERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    run_index INTEGER NOT NULL,
    slice_name TEXT NOT NULL,
    coverage_rate REAL,
    proofs_success INTEGER DEFAULT 0,
    proofs_per_sec REAL DEFAULT 0.0,
    abstention_pct REAL DEFAULT 0.0,
    policy_reward REAL,
    symbolic_descent REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(experiment_id, run_index)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_rfl_runs_experiment_id 
    ON rfl_runs(experiment_id, run_index);
CREATE INDEX IF NOT EXISTS idx_rfl_runs_slice_name 
    ON rfl_runs(slice_name);
CREATE INDEX IF NOT EXISTS idx_rfl_runs_created_at 
    ON rfl_runs(created_at DESC);

-- Add constraints
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'rfl_runs_abstention_pct_check') THEN
        ALTER TABLE rfl_runs ADD CONSTRAINT rfl_runs_abstention_pct_check
            CHECK (abstention_pct >= 0.0 AND abstention_pct <= 100.0);
    END IF;
END $$;

-- Record migration
INSERT INTO schema_migrations (version, checksum, status)
VALUES ('018_rfl_runs_table', 'sha256_hash_here', 'success')
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

**Key Design Decisions**:
- Uses `BEGIN; ... COMMIT;` transaction block
- Uses `CREATE TABLE IF NOT EXISTS` for idempotency
- Uses `DO $$ BEGIN ... END $$` for conditional constraints
- Records in `schema_migrations` table
- Follows existing migration patterns (see `013_runs_logging.sql` for reference)

---

### Phase-I to Phase-II Transition Notes

**Current State** (Phase-I):
- RFL evidence: File-based only (`artifacts/rfl/*.json`, `results/*.jsonl`)
- Database impact: None (RFL does not write to database)
- Migration impact: None (no RFL-specific migrations)

**Future State** (Phase-II):
- RFL evidence: Hybrid (file-based + database)
- Database impact: RFL writes to `rfl_runs`, `rfl_experiments`, `rfl_coverage_stats`
- Migration impact: New migrations (018+) must be applied

**Transition Requirements**:
- Backward compatibility: Phase-I file-based evidence must continue to work
- Forward compatibility: Phase-II must be able to read Phase-I file-based evidence
- Migration safety: RFL migrations must be idempotent (safe to apply even if Phase-II not yet enabled)

---

### Evidence Base (RFL Phase-I)

**Files Referenced**:
- `rfl/config.py` (lines 116-119: artifacts_dir, results_file, coverage_file)
- `rfl/metrics_logger.py` (line 59: JSONL output path)
- `rfl/runner.py` (lines 152, 298, 304: file-based artifact paths)
- `backend/rfl/README.md` (lines 157-239: output artifacts documentation)
- `migrations/` directory (verified: no RFL-specific tables)

**Claims Verified**:
- ✅ RFL Phase-I uses file-based storage only
- ✅ No RFL-specific database tables exist in migrations
- ✅ RFL configuration includes `database_url` but uses it for reading, not writing
- ✅ `runs` table exists but is general-purpose (not RFL-specific)
- ✅ Phase-I RFL runs exhibit 100% abstention (hermetic lean-disabled mode)
- ✅ Phase-I RFL logs are negative-control / plumbing validation runs only
- ✅ Phase-I demonstrates zero empirical RFL uplift (by design)

**Claims NOT Verified** (Hypothetical - Phase-II):
- ❌ `rfl_runs` table does NOT exist (will be created in Phase-II)
- ❌ RFL migrations do NOT exist (will be created in Phase-II)
- ❌ Fixture truncation does NOT include RFL tables (will be updated in Phase-II)

---

### Conclusion (Appendix A)

**Phase-I State**: RFL operates in hermetic, file-based mode with 100% abstention (lean-disabled negative control). Database migrations are unaffected. The migration glue specification applies to core schema migrations only. Phase-I RFL evidence validates execution infrastructure and attestation wiring only; it does NOT demonstrate uplift, reduced abstention, or metabolism verification.

**Phase-II Preparation**: When RFL transitions to database-backed evidence storage, the checklist above must be completed. The existing migration infrastructure (`run_all_migrations()`, `check_all_migrations_applied()`, `first_organism_db` fixture) should handle RFL migrations automatically, provided they follow the idempotent migration patterns and correct ordering constraints.

**Status**: Phase-I Complete (Evidence-Based), Phase-II Specification Only (Awaiting Implementation Authorization)

