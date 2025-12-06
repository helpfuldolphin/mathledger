# First Organism & Dyno Chart Runbook

**Objective:** Go from "repo clone" → "SPARK green" → "Dyno Chart generated"

**Phase I Status:** This runbook documents the operational procedure for First Organism experiments. **Phase I Evidence** (what actually exists on disk):
- ✅ `artifacts/first_organism/attestation.json` - Sealed dual-attestation with H_t, R_t, U_t
- ✅ `results/fo_baseline.jsonl` - 1000-cycle baseline run
- ✅ `results/fo_rfl.jsonl` - 1000-cycle RFL plumbing run (all-abstain, metabolism validation only)
- ✅ `results/fo_rfl_50.jsonl` - 50-cycle RFL sanity run
- ✅ `artifacts/figures/rfl_dyno_chart.png` - Dyno Chart visualization

This runbook provides step-by-step instructions to:
1. Set up First Organism infrastructure
2. Run SPARK closed-loop integration test (optional but recommended)
3. Execute First Organism experiments (baseline + RFL) using default slice
4. Generate Dyno Chart visualization from existing or new data

---

## 1. Prerequisites

### System Requirements
- **Windows** with PowerShell 7+ (`pwsh.exe`)
- **Docker Desktop** installed and running
- **Python 3.11+** with `uv` package manager installed
- **Git** for cloning the repository

### Environment Setup

1. **Clone the repository** (if not already done):
   ```powershell
   git clone <repository-url>
   cd mathledger
   ```

2. **Install Python dependencies**:
   ```powershell
   uv sync
   ```

3. **Create First Organism environment file**:
   ```powershell
   Copy-Item ops/first_organism/first_organism.env.template .env.first_organism
   ```

4. **Generate secure credentials** and update `.env.first_organism`:
   ```powershell
   # PostgreSQL password (32 chars)
   $pg_pass = -join ((65..90) + (97..122) + (48..57) + (33,35,37,64) | Get-Random -Count 32 | ForEach-Object {[char]$_})
   Write-Host "POSTGRES_PASSWORD=$pg_pass"
   
   # Redis password (24 chars)
   $redis_pass = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 24 | ForEach-Object {[char]$_})
   Write-Host "REDIS_PASSWORD=$redis_pass"
   
   # API key (48 hex chars)
   $api_key = -join ((48..57) + (97..102) | Get-Random -Count 48 | ForEach-Object {[char]$_})
   Write-Host "LEDGER_API_KEY=$api_key"
   ```

   Replace all `<REPLACE_...>` placeholders in `.env.first_organism` with the generated credentials.

   > **Note:** For detailed environment setup instructions, see [`docs/FIRST_ORGANISM_ENV.md`](../docs/FIRST_ORGANISM_ENV.md).

5. **Verify Docker Desktop is running**:
   ```powershell
   docker ps
   ```
   If Docker is not running, start Docker Desktop and wait for it to fully initialize (whale icon in system tray).

---

## 2. Bring Up Infrastructure

Start the First Organism Docker services (PostgreSQL + Redis):

```powershell
.\scripts\start_first_organism_infra.ps1
```

**What this does:**
- Checks Docker is running
- Validates `.env.first_organism` exists
- Starts PostgreSQL and Redis containers using `ops/first_organism/docker-compose.yml`
- Waits for health checks to pass (up to 60 seconds)

**Expected output:**
```
========================================
✅ First Organism infra is up (Postgres/Redis healthy)
========================================

Services:
  PostgreSQL: localhost:5432
  Redis:      localhost:6380
```

> **Note:** See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical connection string formats and port selection guide.

**Troubleshooting:**
- If services don't become healthy, check logs:
  ```powershell
  docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism logs postgres
  docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism logs redis
  ```

---

## 3. Run SPARK (Optional but Recommended)

SPARK validates the complete First Organism pipeline end-to-end. While optional, it's recommended to verify the system is working before running experiments.

```powershell
.\scripts\run_spark_closed_loop.ps1
```

**What this does:**
- Sets `FIRST_ORGANISM_TESTS=true` and `SPARK_RUN=1`
- Runs the closed-loop integration test: `test_first_organism_closed_loop_happy_path`
- Validates: UI Event → Curriculum Gate → Derivation → Lean Verify → Dual-Attest seal H_t → RFL runner metabolism
- Logs output to `ops/logs/SPARK_run_log.txt`

**What PASS looks like:**
```
========================================
✅ SPARK: PASS
   Found: [PASS] FIRST ORGANISM ALIVE H_t=<short-hex>
========================================
```

The test should also create `artifacts/first_organism/attestation.json` with stable R_t, U_t, H_t hashes.

**If SPARK fails:**
- Check `ops/logs/SPARK_run_log.txt` for detailed error messages
- Verify infrastructure is running: `docker compose -f ops/first_organism/docker-compose.yml ps`
- Ensure `.env.first_organism` has correct credentials
- See [`ops/logs/SPARK_STATUS.md`](logs/SPARK_STATUS.md) for troubleshooting

---

## 4. Run First Organism Experiments

Execute baseline and RFL experiments. **Phase I Evidence:** The following files exist and contain validated data:
- `results/fo_baseline.jsonl` (1000 cycles, baseline mode)
- `results/fo_rfl.jsonl` (1000 cycles, RFL plumbing run — all-abstain, metabolism validation only)
- `results/fo_rfl_50.jsonl` (50 cycles, RFL sanity run)

### 4.1 Run Baseline (No RFL)

**Default slice (first-organism-slice):**
```powershell
uv run python experiments/run_fo_cycles.py `
  --mode=baseline `
  --cycles=1000 `
  --system=pl `
  --out=results/fo_baseline.jsonl
```

**Expected output:**
- Progress indicator every 100 cycles
- Final message: `Done. Results written to results/fo_baseline.jsonl`

### 4.2 Run RFL (With RFL Enabled)

**Quick sanity run (50 cycles) — Recommended for validation:**
```powershell
uv run python experiments/run_fo_cycles.py `
  --mode=rfl `
  --cycles=50 `
  --system=pl `
  --out=results/fo_rfl_50.jsonl
```

**Expected output:**
- Progress indicator every 10 cycles
- Final message: `Done. Results written to results/fo_rfl_50.jsonl`

**Note on `results/fo_rfl.jsonl` (1000 cycles):**
- This file exists as a **Phase I RFL plumbing run** (internal metabolism validation).
- **Status:** All cycles abstain (`abstention: true` for all 1000 cycles).
- **Purpose:** Validates that RFL metabolism executes correctly (`rfl.executed: true`, `rfl.policy_update: true`) even when all cycles abstain.
- **Not for Dyno Chart analysis:** This run does not demonstrate abstention reduction; it's a toolkit validation run.
- **For actual RFL uplift experiments:** Use `fo_rfl_50.jsonl` or generate new runs with conditions that allow verification (Phase II work).

**Note:** 
- Each experiment may take several minutes depending on cycle count and system performance.
- The default slice is `first-organism-slice` (hardcoded in `run_fo_cycles.py`).
- **Wide Slice (`slice_medium`) is not available in Phase I** — this requires a curriculum slice named `slice_medium` which does not exist in the current `curriculum.yaml`. If you need to use a different slice, verify it exists first: `grep -r "slice_medium" curriculum/`

**Output format:**
Each line in the JSONL files is a canonical JSON object with:
- `cycle`: Cycle index
- `slice_name`: Curriculum slice name
- `status`: "verified", "abstain", or "error"
- `method`: Verification method used
- `abstention`: Boolean flag
- `mode`: "baseline" or "rfl"
- `roots`: H_t, R_t, U_t merkle roots
- `derivation`: Candidate counts and hashes
- `rfl`: RFL execution stats (if mode=rfl)

---

## 5. Generate Dyno Chart

Generate the Dyno Chart visualization comparing baseline vs RFL abstention curves.

**Phase I Evidence:** A Dyno Chart already exists at `artifacts/figures/rfl_dyno_chart.png` (generated from existing Phase I data).

### 5.1 Generate from Baseline vs RFL Plumbing Run

**Using baseline + fo_rfl.jsonl (1000 cycles, all-abstain):**
```powershell
uv run python experiments/generate_dyno_chart.py `
  --baseline results/fo_baseline.jsonl `
  --rfl results/fo_rfl.jsonl `
  --window 100
```

**⚠️ Important:** This chart will show **flat abstention curves** (100% abstention for RFL) because `fo_rfl.jsonl` is an all-abstain plumbing run. The chart validates the toolkit works, but does not demonstrate RFL abstention reduction.

**As of Phase I, RFL abstention does not decrease; chart is used to validate toolkit only.**

### 5.2 Generate from Baseline vs RFL Sanity Run

**Using baseline + fo_rfl_50.jsonl (50 cycles):**
```powershell
uv run python experiments/generate_dyno_chart.py `
  --baseline results/fo_baseline.jsonl `
  --rfl results/fo_rfl_50.jsonl `
  --window 50
```

**Note:** The script defaults to `fo_baseline_wide.jsonl` and `fo_rfl_wide.jsonl`, but these files do not exist in Phase I. Use `--baseline` and `--rfl` flags to point to actual files.

**Output location:**
- `artifacts/figures/rfl_dyno_chart.png` (or `.svg`/`.pdf` depending on script configuration)

**What the Dyno Chart shows:**
- Rolling window abstention rate A(t) for baseline vs RFL
- Cumulative abstentions C(t) over time
- Method distribution comparison
- **Phase I limitation:** Charts from `fo_rfl.jsonl` show flat abstention (toolkit validation only)
- **Phase II goal:** Charts from future runs should show RFL reducing abstention rate after burn-in period

**Analysis details:**
For detailed methodology on abstention curve analysis, see [`docs/RFL_ABSTENTION_ANALYSIS.md`](../docs/RFL_ABSTENTION_ANALYSIS.md).

### 5.3 Validating fo_rfl.jsonl

To validate the RFL plumbing run:

```powershell
# Check file exists and is non-empty
Test-Path results/fo_rfl.jsonl
(Get-Item results/fo_rfl.jsonl).Length -gt 0

# Count cycles
(Get-Content results/fo_rfl.jsonl | Measure-Object -Line).Lines

# Verify RFL execution (all cycles should have rfl.executed: true)
Get-Content results/fo_rfl.jsonl | Select-Object -First 1 | ConvertFrom-Json | Select-Object -ExpandProperty rfl

# Check abstention status (all cycles abstain in this plumbing run)
$allAbstain = (Get-Content results/fo_rfl.jsonl | ConvertFrom-Json | Where-Object { $_.abstention -eq $true }).Count
$total = (Get-Content results/fo_rfl.jsonl | Measure-Object -Line).Lines
Write-Host "Abstention rate: $allAbstain/$total = $([math]::Round($allAbstain/$total*100, 1))%"
```

**Expected validation results:**
- File exists and contains 1000 cycles (cycles 0-999)
- All cycles have `abstention: true` and `status: "abstain"`
- All cycles have `rfl.executed: true` and `rfl.policy_update: true`
- Abstention rate: 100% (all-abstain run)
- **Purpose:** Validates RFL metabolism plumbing works correctly even when all cycles abstain

---

## 6. Related Documentation

- **Environment Setup:** [`docs/FIRST_ORGANISM_ENV.md`](../docs/FIRST_ORGANISM_ENV.md) - Detailed First Organism environment configuration
- **SPARK Status:** [`ops/logs/SPARK_STATUS.md`](logs/SPARK_STATUS.md) - Current SPARK operation status and troubleshooting
- **RFL Abstention Analysis:** [`docs/RFL_ABSTENTION_ANALYSIS.md`](../docs/RFL_ABSTENTION_ANALYSIS.md) - Methodology for abstention curve analysis and Dyno Chart interpretation

---

## 7. Quick Checklist

Use this checklist to verify all steps completed successfully:

- [ ] **Docker Desktop running** - `docker ps` succeeds
- [ ] **`.env.first_organism` present** - File exists with secure credentials (no `<REPLACE_...>` placeholders)
- [ ] **Infrastructure up** - `.\scripts\start_first_organism_infra.ps1` completed successfully
- [ ] **`FIRST_ORGANISM_TESTS=true` set** - Environment variable set (SPARK script does this automatically)
- [ ] **SPARK PASS (optional)** - `.\scripts\run_spark_closed_loop.ps1` shows `✅ SPARK: PASS` with H_t hash
- [ ] **`fo_baseline.jsonl` exists** - File in `results/` directory (Phase I evidence: 1000 cycles)
- [ ] **`fo_rfl.jsonl` exists** - File in `results/` directory (Phase I evidence: 1000 cycles, RFL plumbing run — all-abstain, metabolism validation)
- [ ] **`fo_rfl_50.jsonl` exists** - File in `results/` directory (50-cycle RFL sanity run)
- [ ] **`rfl_dyno_chart.png` generated** - File exists in `artifacts/figures/` directory (Phase I evidence: already exists)
- [ ] **`attestation.json` exists** - File in `artifacts/first_organism/` with H_t, R_t, U_t (Phase I evidence: sealed attestation)

**Verification commands:**
```powershell
# Check infrastructure
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism ps

# Check Phase I evidence files
Test-Path results/fo_baseline.jsonl
Test-Path results/fo_rfl.jsonl
Test-Path results/fo_rfl_50.jsonl
Test-Path artifacts/first_organism/attestation.json
Test-Path artifacts/figures/rfl_dyno_chart.png

# Verify file sizes (non-empty)
(Get-Item results/fo_baseline.jsonl).Length -gt 0
(Get-Item results/fo_rfl.jsonl).Length -gt 0
```

---

## Troubleshooting

### Infrastructure Issues

**Problem:** Services don't start
- **Solution:** Check Docker Desktop is running and has sufficient resources allocated
- **Solution:** Verify `.env.first_organism` has valid credentials (no special characters that break shell parsing)

**Problem:** Health checks fail
- **Solution:** Check container logs: `docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism logs`
- **Solution:** Ensure ports 5432 (PostgreSQL) and 6380 (Redis) are not in use by other services

### SPARK Test Issues

**Problem:** Test fails with database connection error
- **Solution:** Verify infrastructure is running: `docker compose -f ops/first_organism/docker-compose.yml ps`
- **Solution:** Check `.env.first_organism` has correct `DATABASE_URL` and `REDIS_URL` values
- **Solution:** See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical connection string formats and troubleshooting

**Problem:** Test passes but no PASS line found
- **Solution:** Check `ops/logs/SPARK_run_log.txt` for full output
- **Solution:** Verify test actually completed (may have timed out or been interrupted)

### Experiment Issues

**Problem:** `run_fo_cycles.py` fails with import errors
- **Solution:** Ensure `uv sync` completed successfully
- **Solution:** Verify you're in the repository root directory

**Problem:** Results files are empty or incomplete
- **Solution:** Check for errors in console output during cycle execution
- **Solution:** Verify sufficient disk space in `results/` directory

### Dyno Chart Issues

**Problem:** `generate_dyno_chart.py` not found
- **Solution:** Verify script exists: `Test-Path experiments/generate_dyno_chart.py`
- **Solution:** If missing, check if `experiments/analyze_abstention_curves.py` can be used instead (may have different CLI)

**Problem:** Chart generation fails
- **Solution:** Verify both JSONL files exist and are valid: `Get-Content results/fo_baseline.jsonl | Select-Object -First 1 | ConvertFrom-Json`
- **Solution:** Check `artifacts/figures/` directory exists and is writable
- **Solution:** Use actual file names (not `fo_*_wide.jsonl` which don't exist in Phase I): `--baseline results/fo_baseline.jsonl --rfl results/fo_rfl.jsonl`

---

## Next Steps

After successfully generating the Dyno Chart:

1. **Review the visualization** in `artifacts/figures/rfl_dyno_chart.png`
2. **Compare abstention rates** - **Note:** Phase I charts from `fo_rfl.jsonl` show flat abstention (100%) because it's an all-abstain plumbing run. This validates the toolkit works but does not demonstrate RFL abstention reduction (Phase II goal).
3. **Analyze method distribution** - Check which verification methods are used in baseline vs RFL
4. **Position in Evidence Pack** - `fo_rfl.jsonl` is documented as "internal plumbing / metabolism validation" — it proves RFL executes correctly, not that it reduces abstention

For deeper analysis, see [`docs/RFL_ABSTENTION_ANALYSIS.md`](../docs/RFL_ABSTENTION_ANALYSIS.md) for the complete methodology.

---

---

## 8. Phase I Evidence Summary

**What Actually Exists (Reviewer-2 Verified):**

| Artifact | Path | Status | Notes |
|----------|------|--------|-------|
| First Organism Attestation | `artifacts/first_organism/attestation.json` | ✅ EXISTS | Contains H_t, R_t, U_t from sealed dual-attestation |
| Baseline Logs | `results/fo_baseline.jsonl` | ✅ EXISTS | 1000 cycles, baseline mode |
| RFL Logs (plumbing run) | `results/fo_rfl.jsonl` | ✅ EXISTS | 1000 cycles, RFL mode, all-abstain, internal plumbing/metabolism validation |
| RFL Logs (50 cycles) | `results/fo_rfl_50.jsonl` | ✅ EXISTS | 50 cycles, RFL sanity run |
| Dyno Chart | `artifacts/figures/rfl_dyno_chart.png` | ✅ EXISTS | Generated from Phase I data |
| SPARK Test | `tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path` | ✅ EXISTS | Validates full pipeline |

**Detailed Status for `results/fo_rfl.jsonl`:**
- **Cycles:** 1000 (cycles 0-999)
- **Status:** Partial / all-abstain (100% abstention rate)
- **Usage:** Internal plumbing / metabolism validation
- **RFL Execution:** ✅ All cycles have `rfl.executed: true` and `rfl.policy_update: true`
- **Purpose:** Validates RFL metabolism works correctly even when all cycles abstain
- **Not for:** Demonstrating RFL abstention reduction (Phase II goal)
- **Evidence Pack Position:** Documented as "RFL plumbing run" — proves toolkit works, not that it reduces abstention

**What Does NOT Exist (Out of Scope for Phase I):**
- ❌ `results/fo_baseline_wide.jsonl` - Not generated (Wide Slice not in Phase I)
- ❌ `results/fo_rfl_wide.jsonl` - Not generated (Wide Slice not in Phase I)
- ❌ `slice_medium` curriculum slice - Not defined in `curriculum.yaml`
- ❌ RFL runs showing abstention reduction - Phase II / future work (Phase I `fo_rfl.jsonl` is all-abstain plumbing run)
- ❌ ΔH scaling experiments - Phase II / future work
- ❌ Imperfect Verifier robustness tests - Phase II / future work

**Phase I RFL Status:**
- ✅ RFL metabolism plumbing validated (`fo_rfl.jsonl` shows RFL executes correctly)
- ❌ RFL abstention reduction not demonstrated (all cycles abstain in Phase I run)
- 📋 RFL abstention reduction is Phase II goal, not Phase I claim

**Manifest Inconsistencies (To Fix):**
- ⚠️ `artifacts/phase_ii/fo_series_1/fo_1000_rfl/experiment_log.jsonl` - Manifest claims this file exists with SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty file hash). File is empty or manifest is incorrect.

---

**Last Updated:** 2025-01-18  
**Maintainer:** Cursor I — FO Ops Runbook Writer  
**Mode:** Sober Truth / Reviewer-2 Compliant

