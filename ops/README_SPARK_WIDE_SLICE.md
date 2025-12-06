# SPARK & Wide Slice Operations Guide

This guide explains how to run the First Organism infrastructure, SPARK closed-loop tests, and Wide Slice experiments.

## Overview

**SPARK** (First Organism Integration Test) validates the complete closed-loop pipeline:
- UI Event → Curriculum Gate → Derivation → Lean Verify → Dual-Attest seal H_t → RFL runner metabolism

**Wide Slice** experiments run multiple First Organism cycles to generate comparative data:
- Baseline mode (RFL OFF): `results/fo_baseline_wide.jsonl`
- RFL mode (RFL ON): `results/fo_rfl_wide.jsonl`

## Quick Start

### Step 1: Start Infrastructure

Start the First Organism infrastructure (PostgreSQL + Redis):

```powershell
.\scripts\start_first_organism_infra.ps1
```

**Prerequisites:**
- Docker Desktop must be running
- `.env.first_organism` file must exist (see setup below)

**What it does:**
- Checks Docker Desktop is running
- Loads `.env.first_organism` environment file
- Starts PostgreSQL and Redis containers
- Waits for health checks to pass
- Prints status: ✅ "First Organism infra is up (Postgres/Redis healthy)"

**Troubleshooting:**
- ❌ "Docker not running" → Start Docker Desktop
- ❌ "Health checks failed" → Check logs: `docker compose -f ops/first_organism/docker-compose.yml logs`

### Step 2: Run SPARK Test (Optional but Recommended)

Run the SPARK closed-loop integration test:

```powershell
.\scripts\run_spark_closed_loop.ps1
```

**What it does:**
- Sets `FIRST_ORGANISM_TESTS=true` and `SPARK_RUN=1`
- Runs `test_first_organism_closed_loop_happy_path`
- Logs output to `ops/logs/SPARK_run_log.txt`
- Searches for `[PASS] FIRST ORGANISM ALIVE H_t=` line
- Prints: ✅ "SPARK: PASS" or ❌ "SPARK: NO PASS LINE FOUND"

**Expected output:**
```
✅ SPARK: PASS
   Found: [PASS] FIRST ORGANISM ALIVE H_t=abc123...
```

### Step 3: Run Wide Slice Experiments

Run the Wide Slice experiments for FO cycles:

```powershell
.\scripts\run_wide_slice_experiments.ps1
```

**With custom parameters:**
```powershell
.\scripts\run_wide_slice_experiments.ps1 -Cycles 500 -SliceName "test"
```

**What it does:**
- Runs baseline experiment → `results/fo_baseline_wide.jsonl`
- Runs RFL experiment → `results/fo_rfl_wide.jsonl`
- Logs to `ops/logs/wide_slice_baseline.log` and `ops/logs/wide_slice_rfl.log`
- Prints summary with success/failure status

**Default configuration:**
- Cycles: 1000
- Slice Name: "wide"
- Output files: `results/fo_baseline_wide.jsonl`, `results/fo_rfl_wide.jsonl`

## Setup

### First-Time Setup

1. **Create `.env.first_organism` file:**

   ```powershell
   Copy-Item ops/first_organism/first_organism.env.template .env.first_organism
   ```

2. **Generate secure credentials:**

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

3. **Edit `.env.first_organism` and replace placeholders:**
   - `<REPLACE_WITH_32_CHAR_PASSWORD>` → Your PostgreSQL password
   - `<REPLACE_WITH_24_CHAR_PASSWORD>` → Your Redis password
   - `<REPLACE_WITH_48_CHAR_HEX_KEY>` → Your API key

4. **Verify Docker Desktop is running**

## Environment Variables

The `.env.first_organism` file should contain:

```bash
# Required
POSTGRES_USER=first_organism_admin
POSTGRES_PASSWORD=<your-secure-password>
POSTGRES_DB=mathledger_first_organism
DATABASE_URL=postgresql://first_organism_admin:<password>@localhost:5432/mathledger_first_organism

REDIS_PASSWORD=<your-secure-password>
REDIS_URL=redis://:<password>@localhost:6380/0

LEDGER_API_KEY=<your-secure-api-key>
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

See `ops/first_organism/first_organism.env.template` for the complete template.

## Output Files

### SPARK Test
- **Log:** `ops/logs/SPARK_run_log.txt`
- **Artifacts:** `artifacts/first_organism/attestation.json`

### Wide Slice Experiments
- **Baseline results:** `results/fo_baseline_wide.jsonl`
- **RFL results:** `results/fo_rfl_wide.jsonl`
- **Baseline log:** `ops/logs/wide_slice_baseline.log`
- **RFL log:** `ops/logs/wide_slice_rfl.log`

## Stopping Infrastructure

To stop the First Organism infrastructure:

```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down
```

To remove volumes (clean state):

```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down -v
```

## Troubleshooting

### Docker Issues
- **"Docker not running"**: Start Docker Desktop and wait for it to fully initialize
- **Port conflicts**: Ensure ports 5432 (PostgreSQL) and 6380 (Redis) are not in use

### Health Check Failures
- Check container logs: `docker compose -f ops/first_organism/docker-compose.yml logs`
- Verify `.env.first_organism` has correct credentials
- Try restarting: `docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism restart`

### SPARK Test Failures
- Ensure infrastructure is running: `docker compose -f ops/first_organism/docker-compose.yml ps`
- Check log file: `ops/logs/SPARK_run_log.txt`
- Verify database migrations are run (if required)

### Wide Slice Failures
- Check individual log files in `ops/logs/`
- Verify Python environment: `uv --version`
- Ensure `experiments/run_fo_cycles.py` exists

## Related Documentation

- `ops/first_organism/first_organism.env.template` - Environment file template
- `ops/first_organism/docker-compose.yml` - Docker Compose configuration
- `tests/integration/test_first_organism.py` - SPARK test implementation
- `experiments/run_fo_cycles.py` - Wide Slice cycle runner

