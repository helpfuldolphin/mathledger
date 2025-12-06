# SSL Failure Detection and Resolution Guide

## Purpose

This document consolidates SSL/connection error detection patterns, root causes, and remediation steps for PostgreSQL connections in the MathLedger First Organism test environment.

**Status:** Evidence-based consolidation of existing implementation. No new logic added.

## Scope

### Applicable Contexts

This guide applies to:
- **First Organism (FO) integration tests** - Require database connections for ledger ingestion and attestation
- **Infrastructure setup and validation** - Database connectivity checks during environment initialization
- **Test fixtures and diagnostic tools** - Connection testing scripts and pytest fixtures

### Phase-I RFL Exception

**Phase-I RFL (Reflexive Metabolism) runs are hermetic and do not connect to the database.**

**Canonical Truth:** See `docs/RFL_PHASE_I_TRUTH_SOURCE.md` for authoritative Phase-I RFL facts.

RFL execution paths operate in offline mode:
- Read historical data from files (e.g., `docs/progress.md`)
- Process metrics without database queries
- Generate statistical analysis from pre-existing artifacts
- **No PostgreSQL connection attempts occur**

**Implication:** SSL errors do not arise during Phase-I RFL runs because no database connections are attempted. This guide does not apply to hermetic RFL execution paths.

**Phase I vs Phase II:**
- **Phase I (current):** Hermetic, file-based RFL execution. No database connections. SSL errors do not apply.
- **Phase II (future):** Database-backed RFL execution. SSL configuration would apply if Phase II uses database connections.

**Evidence:** RFL gate implementations (`rfl_gate.py`, `backend/rfl/runner.py`) include fallback mechanisms that parse historical data when database connections fail, confirming hermetic operation capability. All Phase-I RFL logs (e.g., `fo_rfl.jsonl`, `fo_rfl_50.jsonl`) are generated without database connections.

## Root Causes

### 1. Missing `sslmode` Parameter

**Symptom:** `could not send SSL negotiation packet` or similar SSL-related errors.

**Root Cause:** PostgreSQL client (psycopg) attempts SSL handshake by default, but:
- Local Docker containers typically lack SSL certificates
- Remote databases may require explicit SSL configuration

**Detection Pattern:** Error messages containing any of:
- `"ssl"`
- `"ssl negotiation"`
- `"could not send"`
- `"tls"`

### 2. Incorrect `sslmode` Value

**Local Docker Postgres:**
- **Required:** `?sslmode=disable`
- **Why:** Containers typically don't have SSL certificates configured
- **Error if missing:** SSL negotiation fails because client expects SSL but server doesn't support it

**Remote Postgres with SSL:**
- **Required:** `?sslmode=require`
- **Why:** Enforces encrypted connections for security
- **Error if missing:** Connection may fail or be insecure (depending on server configuration)

## Detection Implementation

### Code Locations

SSL error detection is implemented in three locations:

1. **`tests/integration/conftest.py:probe_postgres()`** (lines 225-231, 245-250)
   - Infrastructure probing layer
   - Returns `InfraStatus.OFFLINE` with detailed SSL error message
   - Used by `EnvironmentMode` detection

2. **`tests/integration/conftest.py:test_db_connection` fixture** (lines 644-651)
   - Test fixture layer
   - Emits `pytest.skip()` with `[SKIP][FO]` prefix
   - Provides masked URL and SPARK run context

3. **`scripts/test_db_connection.py`** (lines 91-100, 127-135)
   - Standalone diagnostic tool
   - Classifies errors (SSL, auth, timeout, generic)
   - Provides exit codes for CI/CD integration

### Detection Pattern

All implementations use the same pattern matching:

```python
error_str = str(e).lower()
if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
    # SSL error detected
```

**Note:** Pattern matching is case-insensitive and checks both `psycopg.OperationalError` and generic `Exception` types (SQLAlchemy may wrap exceptions).

### Error Message Format

Consistent format across all detection points:

```
[SKIP][FO] SSL negotiation failed; check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md).
  Error: <original_exception>
  Attempted URL: <masked_url>
  For local Docker: use ?sslmode=disable
  For remote DB: use ?sslmode=require
```

## Remediation Steps

### Step 1: Identify Connection Type

**Local Docker:**
- Host: `localhost`, `127.0.0.1`, `::1`, or empty
- Port: Typically `5432` or `5433`
- **Action:** Add `?sslmode=disable`

**Remote Database:**
- Host: Any non-localhost hostname or IP
- **Action:** Add `?sslmode=require`

### Step 2: Update DATABASE_URL

**Before (missing sslmode):**
```
postgresql://user:password@localhost:5432/dbname
```

**After (local Docker):**
```
postgresql://user:password@localhost:5432/dbname?sslmode=disable
```

**After (remote):**
```
postgresql://user:password@remote.example.com:5432/dbname?sslmode=require
```

### Step 3: Verify Fix

Run the connection test script:

```bash
uv run python scripts/test_db_connection.py
```

**Expected output (success):**
```
Testing connection to: postgresql://user:****@localhost:5432/dbname
✓ SUCCESS: Database connection successful
  Connection info: Current: sslmode=disable
```

**Expected output (SSL error):**
```
✗ FAILED: SSL negotiation error
  Error: could not send SSL negotiation packet
  Recommendation: add ?sslmode=disable for local Docker

  Troubleshooting:
  - For local Docker Postgres: add ?sslmode=disable to DATABASE_URL
  - For remote Postgres with SSL: add ?sslmode=require to DATABASE_URL
  - See docs/FIRST_ORGANISM_ENV.md for more details
```

## Configuration Files

### Canonical Examples

**Local Docker (First Organism):**
- `config/first_organism.env`: `?sslmode=disable`
- `config/first_organism.env.template`: `?sslmode=disable`
- `scripts/run_first_organism_spark.ps1`: `?sslmode=disable`

**Remote/Production:**
- `config/nightly.env`: `?sslmode=require`

### Template Format

All environment templates should include explicit `sslmode`:

```bash
# Local Docker
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}?sslmode=disable

# Remote
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}?sslmode=require
```

## Test Suite Integration

### Pytest Skip Behavior

When SSL errors are detected:

1. **Integration tests** (`tests/integration/conftest.py`):
   - `probe_postgres()` returns `InfraStatus.OFFLINE` with SSL error message
   - `test_db_connection` fixture emits `pytest.skip()` with `[SKIP][FO]` prefix
   - Tests are skipped, not failed (infrastructure issue, not test failure)

2. **Unit tests** (`tests/conftest.py`):
   - **Note:** SSL detection was removed in favor of raising exceptions directly
   - This allows unit tests to fail fast on connection issues
   - Integration tests handle SSL errors gracefully via skip

### Skip Message Format

```
[SKIP][FO] SSL negotiation failed; check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md).
  Error: <exception>
  Attempted URL: <masked_url>
  For local Docker: use ?sslmode=disable
  For remote DB: use ?sslmode=require
```

## Consistency Verification

### Documentation References

All SSL guidance references point to:

1. **Primary:** `docs/FIRST_ORGANISM_ENV.md` (section 2.1)
2. **Detailed:** `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` (section on SSL)
3. **This guide:** `docs/SSL_FAILURE_GUIDE.md` (technical details)

### Error Message Consistency

All SSL error messages include:
- Reference to `FIRST_ORGANISM_ENV.md`
- Specific `sslmode` recommendations
- Masked URL for debugging
- Clear distinction between local Docker vs remote

### Code Consistency

**Pattern matching:** Identical across all three detection points:
- Same keyword list: `["ssl", "ssl negotiation", "could not send", "tls"]`
- Same case-insensitive matching
- Same error message structure

**Exception handling:** 
- `probe_postgres()`: Checks both `psycopg.OperationalError` and generic `Exception`
- `test_db_connection` fixture: Checks both `psycopg.OperationalError` and generic `Exception`
- `test_db_connection.py` script: Checks both `psycopg.OperationalError` and generic `Exception`

## Known Limitations

### 1. Pattern Matching Coverage

**Current:** String-based pattern matching on error messages.

**Limitation:** May miss platform-specific or PostgreSQL version-specific error formats.

**Mitigation:** Broad keyword list covers common variants. Real-world testing will reveal edge cases.

### 2. SQLAlchemy Exception Wrapping

**Current:** Generic `Exception` catch-all handles wrapped exceptions.

**Limitation:** SQLAlchemy may wrap psycopg exceptions in ways that obscure original error.

**Mitigation:** String-based detection works even when exceptions are wrapped.

### 3. Unit Test Fixture

**Current:** `tests/conftest.py:db_engine` raises exceptions directly (no SSL-specific skip).

**Rationale:** Unit tests should fail fast on infrastructure issues. Integration tests handle SSL gracefully.

**Consistency Note:** This is intentional divergence - unit tests vs integration tests have different error handling strategies.

## Troubleshooting Checklist

- [ ] Check `DATABASE_URL` contains `sslmode` parameter
- [ ] Verify `sslmode` value matches connection type (disable for local, require for remote)
- [ ] Run `scripts/test_db_connection.py` to isolate connection issues
- [ ] Check PostgreSQL server logs for SSL-related errors
- [ ] Verify Docker container is running (for local setup)
- [ ] Check network connectivity (for remote setup)
- [ ] Review `docs/FIRST_ORGANISM_ENV.md` for environment setup steps

## Related Documentation

- `docs/FIRST_ORGANISM_ENV.md` - Environment setup and SSL configuration
- `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` - Canonical connection string formats
- `config/first_organism.env.template` - Environment template with SSL examples
- `scripts/test_db_connection.py` - Standalone connection test tool

## Evidence Base

This guide consolidates SSL error handling from:

1. **Implementation:** `tests/integration/conftest.py` (probe_postgres, test_db_connection fixture)
2. **Implementation:** `scripts/test_db_connection.py` (standalone tool)
3. **Documentation:** `docs/FIRST_ORGANISM_ENV.md` (section 2.1)
4. **Documentation:** `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` (SSL section)
5. **Configuration:** `config/first_organism.env`, `config/first_organism.env.template`

All claims in this guide are backed by on-disk files and verified code paths.

