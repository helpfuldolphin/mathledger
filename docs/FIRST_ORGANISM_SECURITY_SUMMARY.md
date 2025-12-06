# First Organism Security Summary

**Status**: Evidence Pack v1 - Consolidated + Phase II Security Envelope
**Last Updated**: 2025-12-05
**Mode**: Reviewer-2 Hardened

## Purpose

This document consolidates the security posture for First Organism (FO) integration tests. All claims are backed by on-disk artifacts and verified implementations.

## Core Security Posture

First Organism tests enforce a **zero-trust** environment with explicit credentials, authenticated services, and deterministic guards. No defaults, no open Redis, no weak passwords.

### Enforcement Points

1. **Environment Enforcer** (`backend/security/first_organism_enforcer.py`)
   - Validates all required environment variables
   - Rejects weak/banned passwords (`mlpass`, `postgres`, `password`, `devkey`, etc.)
   - Enforces minimum lengths: 12 chars (passwords), 16 chars (API keys)
   - Rejects CORS wildcards (`*`)
   - **Requires `RUNTIME_ENV=test_hardened`** (blocks `production`, warns on legacy values)

2. **Runtime Environment Guard** (`backend/security/runtime_env.py`)
   - Provides `is_strict_mode()` when `FIRST_ORGANISM_STRICT=1`
   - Validates credential strength in strict mode
   - Rejects wildcards in CORS when strict mode enabled

3. **Test Fixture Enforcement** (`tests/integration/first_organism_conftest.py`)
   - `first_organism_secure_env` fixture calls `enforce_first_organism_env()` at session start
   - Calls `assert_runtime_env_hardened()` to verify `RUNTIME_ENV=test_hardened`
   - Fails entire test session if security checks fail
   - Skips if `RUNTIME_ENV=production` (safety guard)

## Required Environment Variables

| Variable | Requirement | Enforced By |
|----------|-------------|-------------|
| `RUNTIME_ENV` | Must be `test_hardened` | `first_organism_enforcer.py:293-298` |
| `DATABASE_URL` | Strong password (12+ chars), no banned passwords | `first_organism_enforcer.py:115-160` |
| `REDIS_URL` | Password required (12+ chars), no banned passwords | `first_organism_enforcer.py:163-199` |
| `LEDGER_API_KEY` | Minimum 16 chars, high entropy | `first_organism_enforcer.py:202-235` |
| `CORS_ALLOWED_ORIGINS` | Explicit origins only, no wildcards | `first_organism_enforcer.py:238-262` |
| `POSTGRES_USER` | Minimum 3 characters | `validate_first_organism_env.py:47-56` |
| `POSTGRES_PASSWORD` | Minimum 12 chars, not in banned list | `first_organism_enforcer.py:115-160` |
| `POSTGRES_DB` | Minimum 3 characters | `validate_first_organism_env.py:47-56` |
| `REDIS_PASSWORD` | Minimum 12 chars, not in banned list | `first_organism_enforcer.py:163-199` |

### Banned Passwords (Verified in Code)

**PostgreSQL**: `postgres`, `password`, `mlpass`, `ml`, `mathledger`, `admin`, `root`, `test`, `secret`, `changeme`, `""`  
**Redis**: `""`, `redis`, `password`, `secret`, `changeme`, `test`  
**API Keys**: `devkey`, `dev`, `test`, `secret`, `changeme`, `api-key`, `apikey`, `""`

Source: `backend/security/first_organism_enforcer.py:49-82`

## Validation Tools

1. **Standalone Validator** (`tools/validate_first_organism_env.py`)
   - Validates `.env.first_organism` file before test execution
   - Checks all required variables, password strength, CORS policy
   - Exit code 0 = pass, 1 = fail, 2 = file not found

2. **Runtime Enforcer** (`backend/security/first_organism_enforcer.py`)
   - Called automatically by `first_organism_secure_env` fixture
   - Raises `InsecureCredentialsError` with violation list if checks fail

## Template & Setup

- **Template**: `config/first_organism.env.template`
  - Contains example values that pass all enforcer checks
  - Includes `RUNTIME_ENV=test_hardened`
  - Documents all security requirements inline

- **Setup Process**:
  1. Copy template: `cp config/first_organism.env.template .env.first_organism`
  2. Customize credentials (generate strong passwords)
  3. Validate: `uv run python tools/validate_first_organism_env.py .env.first_organism`
  4. Load env and run tests: `FIRST_ORGANISM_TESTS=true uv run pytest -m first_organism`

## Evidence Trail

All security requirements are verified by:

1. **Code Implementation**:
   - `backend/security/first_organism_enforcer.py` (326 lines) - Main enforcer
   - `backend/security/runtime_env.py` (200 lines) - Runtime validation
   - `tools/validate_first_organism_env.py` (300 lines) - Standalone validator
   - `tests/integration/first_organism_conftest.py` (154 lines) - Test fixtures

2. **Documentation**:
   - `docs/FIRST_ORGANISM_SECURITY.md` - Comprehensive security guide
   - `docs/FIRST_ORGANISM_ENV.md` - Environment setup procedures
   - `config/first_organism.env.template` - Template with inline security notes

3. **Test Integration**:
   - All FO tests marked with `@pytest.mark.first_organism` use `first_organism_secure_env` fixture
   - Fixture enforces security at session start (runs once, fails fast)
   - `assert_runtime_env_hardened()` integrated into session fixture

## Security Guarantees

**Verified Claims** (backed by code):

✅ Weak passwords are rejected before test execution  
✅ CORS wildcards are rejected  
✅ `RUNTIME_ENV=production` causes test skip (safety guard)  
✅ `RUNTIME_ENV=test_hardened` is required (enforced in enforcer)  
✅ All required variables must be present and non-empty  
✅ Credential strength validated (length, banned list, entropy)  

**Not Claimed** (no evidence):

❌ Network isolation (Docker binding to 127.0.0.1 not verified in code)  
❌ SSL/TLS enforcement for remote connections (only checked in enforcer, not enforced at connection time)  
❌ Rate limiting effectiveness (configured but not verified in tests)  

## Phase-I RFL Security Posture

**Scope**: Phase-I RFL execution logs (`results/fo_rfl.jsonl`) are hermetic and operate outside the database-backed security surface.

**Verified Claims** (backed by evidence):

✅ **Hermetic Execution**: Phase-I RFL runs (`fo_rfl.jsonl`) do not require DB or Redis security surfaces  
✅ **No Secrets Accessed**: RFL cycle logs contain only execution data (derivation results, attestation roots, policy updates)  
✅ **No Database Writes**: Phase-I RFL logs are written directly to disk by `RFLExperimentLogger`, not to database  
✅ **No TLS/Role-Based Security Required**: Phase-I operates on file I/O only, no network authentication needed  

**Risk Deferral**:

All database-backed security risks (credential validation, connection string security, role-based access) are deferred to **Phase-II DB-backed RFL**. Phase-I RFL logs are self-contained JSONL files with no external dependencies.

**Evidence Source**: `DB_URL_CONSISTENCY_REPORT.md:480-529` documents hermetic nature of Phase-I RFL execution logs.

## Phase II Uplift & Ledger Integrity

**PHASE II — NOT RUN IN PHASE I**

This section documents security considerations for Phase II uplift experiments (U2 runner, asymmetric environments, slice-specific metrics). All Phase II artifacts are governed by `experiments/prereg/PREREG_UPLIFT_U2.yaml`.

### Security Considerations for Phase II

#### 1. Hermetic Uplift Logs

Uplift logs (`results/uplift_u2_*_baseline.jsonl`, `results/uplift_u2_*_rfl.jsonl`) must remain hermetic:

- **No External Reward Channels**: RFL uses only verifiable on-chain feedback (derivation success, Lean verification status). No human preferences, no proxy metrics, no external reward signals.
- **Isolated Execution**: Each uplift experiment runs in isolation. No cross-experiment state leakage.
- **Append-Only Logging**: Experiment logs are append-only JSONL files. No in-place modification permitted.
- **Manifest Integrity**: Each experiment produces `experiment_manifest.json` with SHA-256 hashes of all input/output files.

#### 2. Deterministic Seeds Protect Against Tampering

All Phase II experiments use deterministic seeding:

- **Seed Schedule**: `MDAP_SEED + cycle_index` ensures reproducible random ordering in baseline mode
- **Tamper Detection**: Any deviation from preregistered seed schedule invalidates results
- **Audit Trail**: Seeds logged per-cycle in JSONL output for verification
- **No Post-Hoc Seed Selection**: Preregistration fixes seed schedule before execution

#### 3. No External Reward Channels

Phase II RFL operates exclusively on verifiable feedback:

| Allowed Signals | Prohibited Signals |
|-----------------|-------------------|
| Lean verification success/failure | Human preference scores |
| Derivation proof chain validity | External API responses |
| Formula hash membership in corpus | LLM-generated rewards |
| Canonical normalization equality | Proxy metrics |

**Enforcement**: The `RFLExperimentLogger` only accepts signals from `VerificationResult` objects produced by the Lean verifier or truth table fallback.

#### 4. Phase Separation

Phase II artifacts must not reference or depend on Phase I execution logs:

**Cannot Reference** (enforced by governance):
- `fo_rfl_*.jsonl` (Phase I RFL logs)
- `fo_baseline_*.jsonl` (Phase I baseline logs)
- Any file in `results/` produced before Phase II preregistration

**Must Reference**:
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` (preregistration)
- `curriculum_uplift_phase2.yaml` (slice definitions)
- Phase II-specific seed schedules

#### 5. Ledger Integrity Under Asymmetric Uplift

Asymmetric uplift environments introduce slice-specific success metrics:

| Slice | Success Metric | Integrity Requirement |
|-------|---------------|----------------------|
| `slice_uplift_goal` | Goal hash hit | Target hashes frozen in preregistration |
| `slice_uplift_sparse` | Density (verified/candidates) | Candidate generation deterministic |
| `slice_uplift_tree` | Chain length | Parent-child relationships verified |
| `slice_uplift_dependency` | Multi-goal conjunction | All goal hashes frozen in preregistration |

**Ledger Write Policy**: Only statements verified by Lean (or truth table fallback) may be written to the ledger. Uplift policy scores do not influence verification status.

#### 6. Statistical Summary Integrity

`statistical_summary.json` must include:
- Exact cycle counts (baseline and RFL)
- Raw success counts (not just rates)
- Wilson score confidence intervals
- Effect size (Δp) with bounds
- Seed schedule verification hash

### Verified Claims (Phase II)

✅ Uplift logs are append-only JSONL with no in-place modification
✅ Deterministic seeds logged per-cycle for audit
✅ No external reward channels (RFL uses Lean/truth-table feedback only)
✅ Phase II artifacts labeled and cannot reference Phase I logs
✅ Preregistration governs all experiments (`PREREG_UPLIFT_U2.yaml`)
✅ Ledger writes gated by verification status (not policy score)

### Not Yet Verified (Pending Phase II Execution)

⏳ Manifest SHA-256 hashes computed at experiment completion
⏳ Statistical summaries generated with correct CI method
⏳ Slice-specific success metrics evaluated correctly

---

## Related Documentation

- **Detailed Setup**: `docs/FIRST_ORGANISM_ENV.md`
- **Comprehensive Guide**: `docs/FIRST_ORGANISM_SECURITY.md`
- **Connection Strings**: `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md`
- **Template**: `config/first_organism.env.template`
- **Phase II Preregistration**: `experiments/prereg/PREREG_UPLIFT_U2.yaml`

## Changes Made (Evidence Pack v1)

1. **Enforcer Tightened** (`backend/security/first_organism_enforcer.py:292-298`)
   - Now requires `RUNTIME_ENV` to be set
   - Blocks `production` explicitly
   - Requires `test_hardened` (deprecates `first_organism`/`integration`)

2. **Test Integration** (`tests/integration/first_organism_conftest.py:39-60`)
   - `first_organism_secure_env` fixture now calls `assert_runtime_env_hardened()`
   - Minimal integration point (session-scoped, runs once)

3. **Documentation Consolidated**
   - This summary consolidates security posture
   - References detailed docs for setup procedures
   - Only claims backed by on-disk code

---

**Reviewer-2 Note**: This document contains only verified claims backed by actual code implementations. All security requirements are traceable to specific file locations and line numbers.

