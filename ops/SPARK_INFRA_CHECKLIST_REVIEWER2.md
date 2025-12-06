# SPARK Infrastructure Checklist — Reviewer-2 Edition

**Evidence-Based Verification Only**

This document contains only verified facts from on-disk artifacts. No hypotheticals, no forward-looking claims.

---

## Verified Artifact Locations

**Template:** `config/first_organism.env.template`  
**Launcher:** `scripts/run_first_organism_spark.ps1`  
**Docker Compose:** `ops/first_organism/docker-compose.yml`  
**Test:** `tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path`  
**Log Output:** `ops/logs/SPARK_run_log.txt`

---

## Prerequisites (Evidence from Files)

### 1. Docker Desktop

**Verification Command:**
```powershell
docker --version
docker ps
```

**Evidence:** Script checks this at line 114 of `scripts/run_first_organism_spark.ps1`

**If Missing:** Script exits with code 1 at line 122

---

### 2. Container Configuration

**Container Names (from `ops/first_organism/docker-compose.yml`):**
- Line 46: `container_name: first_organism_postgres`
- Line 86: `container_name: first_organism_redis`

**Port Mappings (from `ops/first_organism/docker-compose.yml`):**
- Line 58: PostgreSQL: `127.0.0.1:5432:5432`
- Line 107: Redis: `127.0.0.1:6380:6379` (host port 6380 → container port 6379)

**Evidence:** Script checks containers at lines 127-142 of `scripts/run_first_organism_spark.ps1`

**Verification Command:**
```powershell
docker ps --format "{{.Names}}"
```

**Expected Output:** Must contain `first_organism_postgres` or container with `postgres` in name, and `first_organism_redis` or container with `redis` in name.

---

### 3. Environment File Configuration

**Required File:** `.env.first_organism` in project root

**Required Variables (from script lines 73-88):**
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `REDIS_PASSWORD`
- `LEDGER_API_KEY`
- `CORS_ALLOWED_ORIGINS`

**Optional Variable:**
- `REDIS_PORT` (defaults to 6379 if not set, but `ops/first_organism/docker-compose.yml` uses 6380)

**Evidence:** Script validates at lines 76-88, constructs URLs at lines 103-104

**Template Source:** `config/first_organism.env.template`

**Example Values (from `config/first_organism.env`):**
- Line 10: `DATABASE_URL=postgresql://first_organism_user:secure_test_password_123@localhost:5432/mathledger_first_organism?sslmode=disable`
- Line 14: `REDIS_URL=redis://:secure_redis_password_456@localhost:6379/0`

**SSL Mode:** `sslmode=disable` (hardcoded in script line 103, matches `config/first_organism.env` line 10)

---

## Redis Port Discrepancy (Verified Inconsistency)

**Evidence of Mismatch:**

1. `ops/first_organism/docker-compose.yml` line 107: Maps host port `6380` to container port `6379`
2. `config/first_organism.env` line 14: Uses port `6379`
3. Script line 99: Defaults to `6379` but reads `REDIS_PORT` if set

**Resolution:** Set `REDIS_PORT=6380` in `.env.first_organism` if using `ops/first_organism/docker-compose.yml`

**Evidence:** Script handles this at line 99: `$redisPort = if ($envVars.ContainsKey("REDIS_PORT")) { $envVars["REDIS_PORT"] } else { "6379" }`

---

## Test Execution

**Command (from script line 154):**
```powershell
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s
```

**Environment Variables Set (script lines 91-107):**
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `REDIS_PASSWORD`
- `LEDGER_API_KEY`
- `CORS_ALLOWED_ORIGINS`
- `DATABASE_URL` (constructed)
- `REDIS_URL` (constructed)
- `FIRST_ORGANISM_TESTS=true`

**Log File:** `ops/logs/SPARK_run_log.txt` (created at line 25, written at lines 161-178)

---

## Verified Error Conditions

### 1. Missing Environment File

**Evidence:** Script checks at line 38, exits with code 1 at line 45

**Error Message:** "ERROR: .env.first_organism not found at: [path]"

---

### 2. Missing Required Variables

**Evidence:** Script validates at lines 76-88, exits with code 1 at line 87

**Error Message:** "ERROR: Missing required environment variables: [list]"

---

### 3. Docker Not Available

**Evidence:** Script checks at lines 114-123, exits with code 1 at line 122

**Error Message:** "ERROR: Docker not found or not running"

---

### 4. Containers Not Running

**Evidence:** Script checks at lines 127-142, warns but continues (does not exit)

**Warning Message:** "WARNING: PostgreSQL container not found" or "WARNING: Redis container not found"

**Note:** Script does not exit on container warnings; test may fail later if containers are actually unavailable.

---

### 5. Test Failure

**Evidence:** Script captures exit code at line 181, exits with test exit code at line 197

**Exit Code:** Non-zero if test fails (propagated from pytest)

---

## Verified File Dependencies

**Script Dependencies:**
- `.env.first_organism` (must exist, validated at line 38)
- `ops/logs/` directory (created if missing at lines 28-30)
- `tests/integration/test_first_organism.py` (test file, not validated by script)

**Test Dependencies (from test file):**
- Database connection via `DATABASE_URL`
- Redis connection via `REDIS_URL`
- Test fixtures from `tests/integration/conftest.py`

---

## Container Startup (If Not Running)

**Command from `ops/first_organism/docker-compose.yml` comments (lines 20-21):**
```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
```

**Evidence:** This command is documented in docker-compose comments, not executed by script.

**Note:** Script does not auto-start containers. User must start them manually.

---

## SSL Mode Configuration

**Evidence:**
- Script line 103: Hardcodes `?sslmode=disable`
- `config/first_organism.env` line 10: Contains `?sslmode=disable`
- Template line 42: Documents `sslmode=disable` for local Docker

**Status:** Consistent across all artifacts.

---

## Container Name Resolution

**Evidence from Script (lines 127-142):**
- Checks for `first_organism_postgres` OR any container with `postgres` in name
- Checks for `first_organism_redis` OR any container with `redis` in name
- Uses `docker ps --format "{{.Names}}"` and `Select-String` pattern matching

**Actual Container Names (from docker-compose):**
- `first_organism_postgres` (line 46)
- `first_organism_redis` (line 86)

**Status:** Script logic matches actual container names.

---

## Summary of Verified Facts

1. ✅ Script reads `.env.first_organism` from project root
2. ✅ Script validates 6 required environment variables
3. ✅ Script constructs `DATABASE_URL` with `sslmode=disable`
4. ✅ Script constructs `REDIS_URL` with configurable port (default 6379)
5. ✅ Script checks Docker availability before proceeding
6. ✅ Script checks container names (supports both first_organism_* and generic names)
7. ✅ Script runs exact test: `test_first_organism_closed_loop_happy_path`
8. ✅ Script logs to `ops/logs/SPARK_run_log.txt`
9. ⚠️ Redis port mismatch exists: docker-compose uses 6380, default is 6379 (resolved by REDIS_PORT variable)
10. ✅ Container names match docker-compose definitions
11. ✅ SSL mode is consistent (`disable` for local)

---

## RFL Artifact Evidence (Phase-I)

**Location:** `results/` directory

### Verified RFL Artifacts

| Artifact | Cycles | Completeness | Method | Status | Evidence Pack Eligible |
|----------|--------|--------------|--------|--------|----------------------|
| `fo_rfl_50.jsonl` | 21 (0-20) | **INCOMPLETE** | `lean-disabled` | All abstain | ⚠️ Phase-I (incomplete, plumbing only) |
| `fo_rfl.jsonl` | 1001 (0-1000) | Complete run | `lean-disabled` | All abstain | ✅ Phase-I (hermetic negative control) |
| `fo_rfl_1000.jsonl` | 11 (0-10) | **INCOMPLETE** | `lean-disabled` | All abstain | ❌ Phase-I (incomplete, do not use) |

**Evidence from File Inspection:**

1. **`fo_rfl_50.jsonl`** (incomplete partial):
   - Location: `results/fo_rfl_50.jsonl`
   - Cycles: 0-20 (21 cycles total, **NOT 50** → **INCOMPLETE**)
   - Schema: JSONL format with fields: `abstention`, `cycle`, `derivation`, `gates_passed`, `method`, `mode`, `rfl`, `roots`, `slice_name`, `status`
   - All entries: `"abstention": true`, `"method": "lean-disabled"`, `"status": "abstain"`
   - Purpose: Small RFL plumbing / negative control demo
   - **Status**: Incomplete run, do not use for cycle count claims

2. **`fo_rfl.jsonl`** (hermetic negative control):
   - Location: `results/fo_rfl.jsonl`
   - Cycles: 0-1000 (1001 cycles total) — **this is the big one**
   - Schema: Same as `fo_rfl_50.jsonl`
   - All entries: `"abstention": true`, `"method": "lean-disabled"`, `"status": "abstain"`
   - **Critical Note:** Hermetic, 1001-cycle RFL **negative control / plumbing** run. **No uplift signal by construction.** All cycles abstain (lean-disabled). Validates execution infrastructure and attestation only, not performance.
   - Purpose: Hermetic negative control / plumbing verification

3. **`fo_rfl_1000.jsonl`**:
   - Location: `results/fo_rfl_1000.jsonl`
   - Cycles: 0-10 (11 cycles total, **NOT 1000** → **INCOMPLETE**)
   - Schema: New (status/method/abstention present)
   - All entries: `"abstention": true`, `"method": "lean-disabled"`, `"status": "abstain"`
   - **Status**: Incomplete run, **do not use for any claim other than "this file exists and is incomplete"**

**Schema Structure (from actual file inspection):**
```json
{
  "abstention": true,
  "cycle": <integer>,
  "derivation": {
    "abstained": 1,
    "candidate_hash": "<sha256>",
    "candidates": 2,
    "verified": 2
  },
  "gates_passed": true,
  "method": "lean-disabled",
  "mode": "rfl",
  "rfl": {
    "abstention_histogram": {...},
    "abstention_rate_after": 1.0,
    "abstention_rate_before": 1.0,
    "executed": true,
    "policy_update": true,
    "symbolic_descent": -0.75
  },
  "roots": {
    "h_t": "<sha256>",
    "r_t": "<sha256>",
    "u_t": "<sha256>"
  },
  "slice_name": "first-organism-pl",
  "status": "abstain"
}
```

**Hash Verification:**
- File hashes not computed in this review (can be added if needed)
- Schema consistent across all inspected entries

**Evidence Pack Eligibility:**

- ✅ **`fo_baseline.jsonl`**: Eligible for Phase-I Evidence Pack (1000 cycles, 0-999, baseline reference)
- ⚠️ **`fo_rfl_50.jsonl`**: Eligible for Phase-I Evidence Pack ONLY as incomplete plumbing evidence (21 cycles, 0-20, marked INCOMPLETE). Do not use for cycle count claims.
- ✅ **`fo_rfl.jsonl`**: Eligible for Phase-I Evidence Pack as hermetic negative control evidence (1001 cycles, 0-1000). NOT eligible as:
  - Uplift evidence (all cycles abstain, 100% abstention by design, no uplift signal by construction)
  - Metabolism verification evidence (lean-disabled, no actual verification)
  - Performance improvement evidence (negative control run)
- ❌ **`fo_rfl_1000.jsonl`**: Do not use for evidence (incomplete, 11 cycles, 0-10, do not use for any claim)

**Explicit Disclaimers:**

1. **Phase I has zero empirical RFL uplift:** Every RFL log shows 100% abstention by design. Phase I only proves that RFL plumbing + attestation + determinism work in a hermetic, lean-disabled negative-control regime.

2. **NOT uplift evidence:** All cycles show `"abstention": true` and `"method": "lean-disabled"`. No proof generation, no throughput measurement, no uplift calculation possible.

3. **Plumbing/infrastructure verification only:** These artifacts verify:
   - RFL runner execution (hermetic, file-based)
   - Policy updates (symbolic descent tracking)
   - Dual attestation sealing (H_t, R_t, U_t computation)
   - Determinism (reproducible cycles)
   
   They do **NOT** demonstrate:
   - Reflexive metabolism
   - Reduced abstention
   - Performance improvement
   - Uplift signals

4. **Incomplete files:** `fo_rfl_50.jsonl` (21 cycles, expected 50) and `fo_rfl_1000.jsonl` (11 cycles, expected 1000) are marked INCOMPLETE and should not be used for completeness claims.

---

## Unverified / Not in Evidence Pack

- Test execution success rate
- Performance characteristics
- Production deployment procedures
- Multi-environment support
- Auto-start container features
- Health check wait logic
- Credential strength validation (enforced by backend, not script)
- RFL uplift metrics (Phase I has zero empirical RFL uplift - all runs are 100% abstention by design)
- Metabolism verification (Phase I demonstrates plumbing only, not metabolism)

---

## Sober-Truth Appendix: RFL Artifact Status

**Review Date:** Based on file inspection as of checklist update

### Artifact Inventory

| File | Exists | Cycles | Completeness | Hash | Schema | Phase-I Eligible | Notes |
|------|--------|--------|---------------|------|--------|------------------|-------|
| `results/fo_baseline.jsonl` | ✅ Yes | 1000 (0-999) | Complete | Not computed | Old schema | ✅ Yes | Baseline reference |
| `results/fo_rfl_50.jsonl` | ✅ Yes | 21 (0-20) | **INCOMPLETE** | Not computed | New schema | ✅ Yes | Canonical partial (expected 50, has 21) |
| `results/fo_rfl.jsonl` | ✅ Yes | 1001 (0-1000) | Complete | Not computed | New schema | ⚠️ Plumbing only | Hermetic negative-control, 100% abstain |
| `results/fo_rfl_1000.jsonl` | ✅ Yes | 11 (0-10) | **INCOMPLETE** | Not computed | New schema | ❌ Do not use | Expected 1000, has 11 only |

### Completeness Assessment

- **fo_baseline.jsonl**: Complete (1000 cycles, 0-999). Purpose: Baseline reference for Phase-I comparison.
- **fo_rfl_50.jsonl**: **INCOMPLETE** (21 cycles, 0-20; expected 50). Purpose: Small RFL plumbing / negative control demo.
- **fo_rfl.jsonl**: Complete (1001 cycles, 0-1000). Purpose: Hermetic negative-control / plumbing verification. **NOT** metabolism verification.
- **fo_rfl_1000.jsonl**: **INCOMPLETE** (11 cycles, 0-10; expected 1000). Do not use for evidence.

### Schema Status

- **fo_baseline.jsonl**: ✅ Schema verified (JSONL, old schema, no top-level status/method/abstention)
- **fo_rfl_50.jsonl**: ✅ Schema verified (JSONL, new schema with status/method/abstention)
- **fo_rfl.jsonl**: ✅ Schema verified (JSONL, matches fo_rfl_50.jsonl structure)
- **fo_rfl_1000.jsonl**: ✅ Schema verified (JSONL, new schema with status/method/abstention)

### Hash Status

- File hashes not computed in this review
- Can be added if required for Evidence Pack sealing

### Evidence Pack Phase-I Eligibility

**Eligible:**
- `fo_baseline.jsonl` - Baseline reference (1000 cycles, old schema, 100% abstention)
- `fo_rfl_50.jsonl` - Canonical partial evidence (21 cycles, marked INCOMPLETE, 100% abstention)
- `fo_rfl.jsonl` - Complete plumbing/infrastructure evidence (1001 cycles, hermetic negative-control, 100% abstention)

**Not Eligible:**
- Uplift claims (**Phase I has zero empirical RFL uplift** - all runs are 100% abstention by design)
- Metabolism verification claims (all cycles abstain, lean-disabled, negative-control only)
- `fo_rfl_1000.jsonl` - Do not use (incomplete, 11 cycles only)

**Critical Phase-I Meta-Truth:**
Phase I has zero empirical RFL uplift. Every RFL log shows 100% abstention by design. Phase I only proves that RFL plumbing + attestation + determinism work in a hermetic, lean-disabled negative-control regime.

**Critical Disclaimers for Evidence Pack:**
1. **Phase I has zero empirical RFL uplift.** Every RFL log is 100% abstention by design.
2. All RFL artifacts inspected show 100% abstention rate (`"abstention": true` in all cycles, or `derivation.abstained > 0` for old schema)
3. All RFL artifacts use `"method": "lean-disabled"` (no actual Lean verification, ML_ENABLE_LEAN_FALLBACK is OFF in Phase I)
4. No throughput metrics available (abstention prevents measurement)
5. No uplift calculation possible (requires successful proof generation, which does not occur in Phase I)
6. `fo_rfl.jsonl` contains 1001 cycles (0-1000), complete hermetic negative-control run
7. `fo_rfl_50.jsonl` contains 21 cycles (0-20), **INCOMPLETE** (expected 50)
8. `fo_rfl_1000.jsonl` contains 11 cycles (0-10), **INCOMPLETE** (expected 1000), do not use for evidence

---

**Last Verified:** Based on files as of checklist update  
**Reviewer-2 Status:** All claims above traceable to on-disk artifacts with line numbers and file inspection

