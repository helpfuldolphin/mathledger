# First Organism Skip Reason Matrix

**Evidence Pack v1 - Reviewer-2 Hardened**

This document maps all `[SKIP][FO]` skip messages to their root causes and remediation steps. All skip conditions are deterministic and traceable to actual code in `tests/integration/conftest.py`.

**Last Verified:** 2025-01-XX (based on `tests/integration/conftest.py` lines 413-468, 626-655, 755-758, 1077-1080)

---

## Skip Message Format

All First Organism skip messages follow this canonical format:

```
[SKIP][FO] <precise reason> (mode=<mode>, db_url=<trimmed-url>)
```

**Components:**
- `[SKIP][FO]` - Prefix for grepability in logs
- `<precise reason>` - Specific failure condition
- `mode=<mode>` - Failure category (see Mode Taxonomy below)
- `db_url=<trimmed-url>` - Connection context (password masked, truncated to 60 chars)

---

## Mode Taxonomy

| Mode | Meaning | Location in Code |
|------|---------|-----------------|
| `<env_disabled>` | FIRST_ORGANISM_TESTS not enabled | `conftest.py:437-440` |
| `<mock>` | Mock URL detected (mock://) | `conftest.py:443-447` |
| `<skip>` | Infrastructure unavailable | `conftest.py:449-465` |
| `<connection_error>` | Connection attempt failed | `conftest.py:651-655` |
| `<ssl_error>` | SSL negotiation failed | `conftest.py:643-649` |
| `<migration_error>` | Database migration failed | `conftest.py:755-758` |
| `<derivation_empty>` | Derivation pipeline produced no statements | `conftest.py:1077-1080` |
| `<rfl_incomplete>` | RFL evidence incomplete or degenerate | Documentation only (no code skip) |

---

## Skip Reason Matrix

### 1. Environment Not Enabled

**Skip Message Pattern:**
```
[SKIP][FO] FIRST_ORGANISM_TESTS not set to true/SPARK_RUN; refusing to run by default. (Set FIRST_ORGANISM_TESTS=true, SPARK_RUN=1, or create .spark_run_enable to enable)
```

**Root Cause:**
- `FIRST_ORGANISM_TESTS` environment variable is not set to `"true"`
- `SPARK_RUN` environment variable is not set to `"1"`
- `.spark_run_enable` file does not exist in project root

**Code Location:** `tests/integration/conftest.py:427-440`

**Remediation:**
```powershell
# Option 1: Set environment variable
$env:FIRST_ORGANISM_TESTS = "true"

# Option 2: Set SPARK_RUN
$env:SPARK_RUN = "1"

# Option 3: Create trigger file
New-Item -Path ".spark_run_enable" -ItemType File

# Then run test
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v
```

**Verification:**
```powershell
# Check if enabled
$env:FIRST_ORGANISM_TESTS
$env:SPARK_RUN
Test-Path .spark_run_enable
```

---

### 2. Mock Mode Detected

**Skip Message Pattern:**
```
[SKIP][FO] EnvironmentMode=MOCK (mock:// URL detected; mode=<mock>, db_url=<postgresql://...>)
```

**Root Cause:**
- `DATABASE_URL` or `DATABASE_URL_TEST` starts with `mock://`
- `ML_USE_LOCAL_DB=1` is set with a mock URL

**Code Location:** `tests/integration/conftest.py:443-447`, `detect_environment_mode()` at line 446

**Remediation:**
```powershell
# Set real database URL
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/mathledger?sslmode=disable"

# Or unset ML_USE_LOCAL_DB if using mock
$env:ML_USE_LOCAL_DB = ""

# Verify URL
$env:DATABASE_URL
```

**Verification:**
```powershell
# Check URL format
$env:DATABASE_URL -notlike "mock://*"
```

---

### 3. Postgres Unreachable (Environment Probe)

**Skip Message Pattern:**
```
[SKIP][FO] EnvironmentMode=SKIP (Postgres unreachable at <host>:<port>; error=<error_detail>; mode=<skip>, db_url=<postgresql://...>)
```

**Root Cause:**
- PostgreSQL container/service is not running
- Network connectivity issue (firewall, wrong host/port)
- Connection timeout (default 3 seconds, 2 retries)
- Authentication failure (wrong credentials)

**Code Location:** `tests/integration/conftest.py:449-465`, `probe_postgres()` at line 192

**Remediation:**
```powershell
# Step 1: Check Docker containers
docker ps | Select-String postgres

# Step 2: Start containers if missing
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d

# Or use main docker-compose
docker compose up -d postgres redis

# Step 3: Verify container health
docker compose ps

# Step 4: Test connection manually
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/mathledger?sslmode=disable"
psql $env:DATABASE_URL -c "SELECT 1"
```

**Verification:**
```powershell
# Check container status
docker ps

# Test connection
python -c "import psycopg; conn = psycopg.connect('$env:DATABASE_URL'); print('OK')"
```

**Reference:** `ops/SPARK_INFRA_CHECKLIST.md` (lines 25-76)

---

### 4. Database Connection Failed (Runtime)

**Skip Message Pattern:**
```
[SKIP][FO] Database connection failed (Postgres unreachable at <host>:<port>; error=<exception>; mode=<connection_error>, db_url=<postgresql://...>)
```

**Root Cause:**
- Connection attempt failed after environment probe passed
- Network issue during actual connection
- Database server rejected connection
- Port conflict or binding issue

**Code Location:** `tests/integration/conftest.py:636-655`

**Remediation:**
```powershell
# Step 1: Verify container is actually running
docker ps | Select-String postgres

# Step 2: Check port binding
docker ps | Select-String "5432"

# Step 3: Test from Python
python -c "import psycopg; psycopg.connect('$env:DATABASE_URL')"

# Step 4: Check for port conflicts
netstat -an | Select-String "5432"

# Step 5: Restart container
docker restart <postgres_container_name>
```

**Verification:**
```powershell
# Direct connection test
psql $env:DATABASE_URL -c "SELECT version();"
```

---

### 5. SSL Negotiation Failed

**Skip Message Pattern:**
```
[SKIP][FO] SSL negotiation failed; check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md). (SPARK_RUN detected)
  Error: <exception>
  Attempted URL: <postgresql://...>
  For local Docker: use ?sslmode=disable
  For remote DB: use ?sslmode=require
```

**Root Cause:**
- SSL/TLS negotiation failed during connection
- `sslmode` parameter missing or incorrect in `DATABASE_URL`
- Local Docker PostgreSQL doesn't support SSL by default
- Remote database requires SSL but `sslmode=disable` is set

**Code Location:** `tests/integration/conftest.py:642-649`

**Remediation:**
```powershell
# For local Docker (default setup)
$env:DATABASE_URL = "postgresql://user:pass@localhost:5432/mathledger?sslmode=disable"

# For remote/production database
$env:DATABASE_URL = "postgresql://user:pass@remote-host:5432/mathledger?sslmode=require"

# Verify URL format
$env:DATABASE_URL -match "sslmode="
```

**Verification:**
```powershell
# Test with correct sslmode
psql "$env:DATABASE_URL" -c "SELECT 1"
```

---

### 6. Migration Failed

**Skip Message Pattern:**
```
[SKIP][FO] Migration failed: <exception> (mode=<migration_error>, db_url=<postgresql://...>)
```

**Root Cause:**
- Database migration SQL execution failed
- Schema conflict (constraints already exist)
- Permission denied on database
- SQL syntax error in migration file

**Code Location:** `tests/integration/conftest.py:752-758`

**Remediation:**
```powershell
# Step 1: Run migrations manually
uv run python scripts/run-migrations.py

# Step 2: Check migration status
psql $env:DATABASE_URL -c "\dt"

# Step 3: Verify database permissions
psql $env:DATABASE_URL -c "SELECT current_user;"

# Step 4: Check for schema conflicts
psql $env:DATABASE_URL -c "SELECT * FROM pg_constraint WHERE conname LIKE '%monotone%';"
```

**Verification:**
```powershell
# Verify migration 016 was applied
psql $env:DATABASE_URL -c "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'blocks');"
```

**Note:** Migration 016 is now handled by `tests/conftest.py` (see `conftest.py:727-736` for commented-out migration check)

---

### 7. Derivation Pipeline Empty

**Skip Message Pattern:**
```
[SKIP][FO] Derivation pipeline produced no statements (mode=<derivation_empty>, db_url=<N/A>)
```

**Root Cause:**
- Derivation pipeline ran but produced zero candidate statements
- Bounds too restrictive (max_atoms=2, max_depth=2, max_breadth=8, max_total=8)
- All candidates filtered out by verifier
- Deterministic seed issue (unlikely but possible)

**Code Location:** `tests/integration/conftest.py:1076-1080`

**Remediation:**
```powershell
# This is a code/logic issue, not infrastructure
# Check derivation pipeline bounds in conftest.py:940-947

# Verify derivation works standalone
python -c "
from derivation.bounds import SliceBounds
from derivation.pipeline import DerivationPipeline
from derivation.verification import StatementVerifier

bounds = SliceBounds(max_atoms=2, max_formula_depth=2, max_mp_depth=2, max_breadth=8, max_total=8)
verifier = StatementVerifier(bounds)
pipeline = DerivationPipeline(bounds, verifier)
outcome = pipeline.run_step(existing=[])
print(f'Statements: {len(outcome.statements)}')
"
```

**Verification:**
- Check `tests/integration/conftest.py:940-950` for derivation bounds
- Verify `DerivationPipeline.run_step()` returns non-empty outcome

---

### 8. RFL Evidence Incompleteness

**Skip Message Pattern:**
```
[SKIP][FO] RFL evidence incomplete or degenerate (mode=<rfl_incomplete>, db_url=<N/A>)
```

**Root Cause:**
- Attempting to run Phase-II-style uplift tests with incomplete or degenerate RFL evidence
- `results/fo_rfl_1000.jsonl` exists but contains only 11 cycles (0–10), not 1000 → **INCOMPLETE**
- `results/fo_rfl.jsonl` exists but shows 100% abstention (hermetic negative-control plumbing run, not evidence of improvement)
- `results/fo_rfl_50.jsonl` is canonical Phase-I evidence (21 cycles, 0–20, all abstain) but insufficient for uplift claims

**Evidence Status (Phase I):**
- `fo_rfl_1000.jsonl`: **Incomplete** - Only 11 cycles (0–10), not 1000 → **Do not use for any claim other than "this file exists and is incomplete"**
- `fo_rfl.jsonl`: **Complete but degenerate** - 1001 cycles (0–1000), all abstain (hermetic negative-control / plumbing run, no uplift signal by construction)
- `fo_rfl_50.jsonl`: **Canonical Phase-I** - 21 cycles (0–20), not 50, all abstain (small RFL plumbing / negative control demo)

**Canonical Truth Source:** See `docs/RFL_PHASE_I_TRUTH_SOURCE.md` for authoritative cycle counts and evidence status.

**Code Location:** Documentation only - This is a policy-level skip, not implemented in `conftest.py`

**Remediation:**
```powershell
# For Phase-I tests: Use canonical evidence only
# Canonical Phase-I RFL evidence: results/fo_rfl_50.jsonl (21 cycles, 0–20, all abstain)
# OR results/fo_rfl.jsonl (1001 cycles, 0–1000, all abstain) for larger plumbing run

# Verify canonical evidence exists
Test-Path results/fo_rfl_50.jsonl
Test-Path results/fo_rfl.jsonl

# Check cycle counts (canonical truth: fo_rfl_50.jsonl = 21, fo_rfl.jsonl = 1001)
(Get-Content results/fo_rfl_50.jsonl | Measure-Object -Line).Lines  # Should be 21
(Get-Content results/fo_rfl.jsonl | Measure-Object -Line).Lines      # Should be 1001

# For Phase-II uplift tests: Generate new RFL runs with conditions that allow verification
# Phase-I evidence does NOT demonstrate uplift (all runs show 100% abstention by design)
```

**Verification:**
```powershell
# Check canonical Phase-I evidence
Test-Path results/fo_rfl_50.jsonl

# Verify cycle counts (canonical truth from docs/RFL_PHASE_I_TRUTH_SOURCE.md)
$lines_50 = (Get-Content results/fo_rfl_50.jsonl | Measure-Object -Line).Lines
$lines_big = (Get-Content results/fo_rfl.jsonl | Measure-Object -Line).Lines
Write-Host "fo_rfl_50.jsonl has $lines_50 cycles (canonical: 21, 0–20)"
Write-Host "fo_rfl.jsonl has $lines_big cycles (canonical: 1001, 0–1000)"

# Check abstention rate (Phase-I shows 100% abstention)
$allAbstain = (Get-Content results/fo_rfl_50.jsonl | ConvertFrom-Json | Where-Object { $_.abstention -eq $true }).Count
$total = (Get-Content results/fo_rfl_50.jsonl | Measure-Object -Line).Lines
Write-Host "Abstention rate: $($allAbstain / $total * 100)% (Phase-I: 100% expected)"
```

**Important Notes:**
- **Phase-I does NOT require RFL uplift** - Phase-I evidence validates RFL execution infrastructure only
- **Phase-I shows 100% abstention by design** - This is expected and demonstrates plumbing works in hermetic lean-disabled mode, not improvement
- **Uplift claims require Phase-II** - New RFL runs with conditions that allow verification (future work)
- **Do NOT use incomplete files** - `fo_rfl_1000.jsonl` (11 cycles, 0–10) is incomplete and not suitable for Phase-I evidence
- **Canonical Phase-I RFL logs:**
  - `fo_rfl_50.jsonl`: 21 cycles (0–20), small plumbing demo
  - `fo_rfl.jsonl`: 1001 cycles (0–1000), large hermetic negative-control run
  - Both show 100% abstention (expected, not a failure)

**Reference:** 
- **`docs/RFL_PHASE_I_TRUTH_SOURCE.md`** - Single source of truth for Phase-I RFL cycle counts and evidence status
- `ops/RUNBOOK_FIRST_ORGANISM_AND_DYNO.md` (lines 178-183, 297-298)
- `experiments/CURSOR_L_SOBER_AUDIT.md` (lines 41-43, 244-245)
- `docs/evidence/EVIDENCE_PACK_V1_AUDIT_CURSOR_O.md` (lines 284-286)

---

## Skip Message Search

To find skip reasons in logs:

```powershell
# Search pytest output
pytest tests/integration/test_first_organism.py -v 2>&1 | Select-String "\[SKIP\]\[FO\]"

# Search SPARK log file
Select-String -Path "ops/logs/SPARK_run_log.txt" -Pattern "\[SKIP\]\[FO\]"

# Search all test output
Get-ChildItem -Recurse -Filter "*.log" | Select-String "\[SKIP\]\[FO\]"
```

---

## Determinism Guarantees

All skip conditions are **deterministic**:

1. **Environment checks** - Based solely on environment variables and file existence
2. **Infrastructure probes** - Use fixed timeouts (3s DB, 2s Redis) and retry counts (2 DB, 1 Redis)
3. **URL parsing** - Regex-based extraction with fallback to "unknown"
4. **Error messages** - Include actual exception text, not synthetic descriptions

**No non-deterministic elements:**
- No random timeouts
- No probabilistic checks
- No time-based heuristics
- No synthetic data generation

---

## Evidence Traceability

Every skip message can be traced to:

1. **Source code location** - Listed in "Code Location" for each skip type
2. **Actual file existence** - References to `ops/SPARK_INFRA_CHECKLIST.md`, `scripts/start_first_organism_infra.ps1`
3. **Environment state** - Captured in skip message (mode, db_url)
4. **Exception details** - Actual error text included in message

**Verification:**
```powershell
# Verify all referenced files exist
Test-Path "ops/SPARK_INFRA_CHECKLIST.md"
Test-Path "scripts/start_first_organism_infra.ps1"
Test-Path "tests/integration/conftest.py"

# Verify skip function exists
Select-String -Path "tests/integration/conftest.py" -Pattern "def assert_first_organism_ready"
```

---

## Consistency Audit

**Last Audit:** 2025-01-XX

**All skip points verified:**
- ✅ `assert_first_organism_ready()` - Uses `[SKIP][FO]` format
- ✅ `test_db_connection` fixture - Uses `[SKIP][FO]` format (lines 626-655)
- ✅ `first_organism_db` fixture - Uses `[SKIP][FO]` format (line 755)
- ✅ `first_organism_attestation_context` fixture - Uses `[SKIP][FO]` format (line 1077)
- ✅ `pytest_collection_modifyitems` - Uses `[SKIP][FO]` format (line 1420)
- ✅ RFL evidence incompleteness - Documented as policy-level skip (no code implementation)

**No inconsistencies detected.**

**Phase-I Clarification:**
- Phase-I First Organism tests do NOT require RFL uplift evidence
- Phase-I RFL evidence (`fo_rfl_50.jsonl`) validates execution infrastructure only
- All Phase-I RFL runs show 100% abstention (expected, not a failure)
- Uplift claims are Phase-II work and require new evidence generation

---

## References

**Primary Code:**
- `tests/integration/conftest.py` - All skip logic
- `tests/integration/test_first_organism.py` - Test documentation

**Infrastructure Documentation:**
- `ops/SPARK_INFRA_CHECKLIST.md` - Infrastructure setup guide
- `scripts/start_first_organism_infra.ps1` - Infrastructure startup script (if exists)

**Log Files:**
- `ops/logs/SPARK_run_log.txt` - SPARK test execution log

---

**Status:** Evidence Pack v1 - Reviewer-2 Hardened  
**Classification:** Internal Documentation  
**Last Updated:** 2025-01-XX

