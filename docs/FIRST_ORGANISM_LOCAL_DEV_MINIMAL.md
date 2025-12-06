# First Organism Local Development (Minimal)

Get First Organism running locally in three steps.

## Prerequisites

1. **Docker Desktop** running (`docker ps` succeeds)
2. **uv** installed (`uv --version` succeeds)
3. **`.env.first_organism`** file in project root (copy from `ops/first_organism/first_organism.env.template` or `config/first_organism.env.template`)

Required variables: `POSTGRES_USER`, `POSTGRES_PASSWORD` (12+ chars), `POSTGRES_DB`, `REDIS_PASSWORD` (12+ chars), `LEDGER_API_KEY` (16+ chars), `CORS_ALLOWED_ORIGINS`.

See [`FIRST_ORGANISM_ENV.md`](FIRST_ORGANISM_ENV.md) for details.

## Steps

### 1. Start Infrastructure
```powershell
.\scripts\start_first_organism_infra.ps1
```
Starts PostgreSQL (localhost:5432) and Redis (localhost:6380) via `ops/first_organism/docker-compose.yml`.

### 2. Run Migrations
```powershell
uv run python scripts/run-migrations.py
```
Run once (or after schema changes).

### 3. Run SPARK
```powershell
.\scripts\run_first_organism_spark.ps1
```
Runs `test_first_organism_closed_loop_happy_path`. Logs saved to `ops/logs/SPARK_run_log.txt`.

## Troubleshooting

**SPARK fails:**
- Check `ops/logs/SPARK_run_log.txt`
- Search for `[SKIP][FO]` markers: `Select-String -Path ops/logs/SPARK_run_log.txt -Pattern "\[SKIP\]\[FO\]"`
- See [`ops/SPARK_INFRA_CHECKLIST.md`](../ops/SPARK_INFRA_CHECKLIST.md)

**Common issues:**
- Docker not running → Start Docker Desktop
- Connection refused → Verify containers: `docker compose -f ops/first_organism/docker-compose.yml ps`
- Authentication failed → Ensure `.env.first_organism` credentials match container credentials

## Quick Reference

```powershell
# First time setup
.\scripts\start_first_organism_infra.ps1
uv run python scripts/run-migrations.py
.\scripts\run_first_organism_spark.ps1

# Daily workflow
.\scripts\start_first_organism_infra.ps1
.\scripts\run_first_organism_spark.ps1

# Stop infrastructure
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down
```

## Phase-I Evidence Locations

RFL execution does not depend on Postgres or Redis in Phase-I. All evidence is file-based.

**Canonical Phase-I RFL evidence (see [`RFL_PHASE_I_TRUTH_SOURCE.md`](RFL_PHASE_I_TRUTH_SOURCE.md)):**
- `results/fo_baseline.jsonl` - Baseline execution log (1000 cycles, 0-999, 100% abstention)
- `results/fo_rfl.jsonl` - RFL execution log (1001 cycles, 0-1000, 100% abstention, hermetic negative control)
- `results/fo_rfl_50.jsonl` - RFL execution log (21 cycles, 0-20, incomplete, 100% abstention)
- `results/fo_rfl_1000.jsonl` - RFL execution log (11 cycles, 0-10, incomplete, 100% abstention)

**RFL artifacts (from `backend/rfl/config.py`):**
- `artifacts/rfl/results.json` - RFL results
- `artifacts/rfl/coverage.json` - Coverage data
- `artifacts/rfl/rfl_curves.png` - Evidence curves (if generated)

**Evidence manifests:**
- `docs/evidence/manifests/RFL_RUN_*.json` - Experiment manifests

**Attestation:**
- `artifacts/first_organism/attestation.json` - Contains H_t, R_t, U_t (H_t = SHA256(R_t || U_t))

**Note:** Phase-I RFL is hermetic, file-based, and demonstrates execution infrastructure only. All Phase-I RFL logs show 100% abstention (lean-disabled negative control). Phase-I does not demonstrate uplift or reduced abstention. All evidence is file-based and does not require database connectivity.

## Related Docs

- [`FIRST_ORGANISM_ENV.md`](FIRST_ORGANISM_ENV.md) - Environment configuration
- [`FIRST_ORGANISM_CONNECTION_STRINGS.md`](FIRST_ORGANISM_CONNECTION_STRINGS.md) - Connection string formats
- [`ops/SPARK_INFRA_CHECKLIST.md`](../ops/SPARK_INFRA_CHECKLIST.md) - Detailed troubleshooting

