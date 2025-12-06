# First Organism Golden Environment Configuration

**Status:** ✅ VERIFIED WORKING - Test passes end-to-end with real DB/Redis

**Last Verified:** 2025-01-18

## Environment Variables

```powershell
# Required for First Organism tests
$env:FIRST_ORGANISM_TESTS = "true"
$env:DATABASE_URL = "postgresql://first_organism_admin:f1rst_0rg4n1sm_l0c4l_s3cur3_k3y!@127.0.0.1:5432/mathledger_first_organism"
$env:LEDGER_API_KEY = "sk_first_organism_test_key_v1_2025"

# Optional (test tolerates Redis failure gracefully)
$env:REDIS_URL = "redis://localhost:6380/0"
```

## Docker Services

```powershell
# Start services
docker compose -f "ops/first_organism/docker-compose.yml" --env-file ".env.first_organism" up -d postgres redis

# Verify services are running
docker compose -f "ops/first_organism/docker-compose.yml" --env-file ".env.first_organism" ps
```

## Database Setup

**Database:** `mathledger_first_organism`  
**User:** `first_organism_admin`  
**Host:** `127.0.0.1:5432`

### Initial Setup (if starting fresh)

1. **Drop and recreate database:**
   ```powershell
   docker exec -i first_organism_postgres psql -U first_organism_admin -d postgres -c "DROP DATABASE IF EXISTS mathledger_first_organism;"
   docker exec -i first_organism_postgres psql -U first_organism_admin -d postgres -c "CREATE DATABASE mathledger_first_organism;"
   ```

2. **Run migrations:**
   ```powershell
   $env:DATABASE_URL = "postgresql://first_organism_admin:f1rst_0rg4n1sm_l0c4l_s3cur3_k3y!@127.0.0.1:5432/mathledger_first_organism"
   uv run python scripts/run-migrations.py
   ```

   **Expected:** "Migration summary: X successful, 0 failed"

## Verification

### Quick DB connectivity test:
```powershell
uv run python -c "import os, psycopg; conn=psycopg.connect(os.environ['DATABASE_URL']); print('✅ Connected'); conn.close()"
```

### Run First Organism test:
```powershell
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s
```

**Expected output:**
- Exit code: 0
- `[PASS] FIRST ORGANISM ALIVE H_t=...` in logs
- File created: `artifacts/first_organism/attestation.json`

**Canonical attestation values (verified):**
- `H_t`: `01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2`
- `R_t`: `a8dc5b2c7778ce38f72e63ecc4b7a9b010969c018d3d7cafff12bf6d85400336`
- `U_t`: `8c11ea1e67666dd3f14a12cdf475a2d7f7c801037f3d273ccca069b1fa703359`

## Known Issues Resolved

1. ✅ **SyntaxError in conftest.py** - Fixed floating docstring
2. ✅ **SSL error masking** - Updated to surface real connection errors
3. ✅ **Duplicate migration execution** - Commented out 016 migration in fixture
4. ✅ **Schema mismatch (theory_id)** - Resolved by fresh DB + full migration run
5. ✅ **Schema_migrations table structure** - Patched to include checksum, status, duration_ms, applied_at

## Next Steps

With this environment verified:

1. **Run FO cycles for experiments:**
   ```powershell
   # Baseline
   uv run python experiments/run_fo_cycles.py --mode=baseline --cycles=1000 --out=results/fo_baseline.jsonl
   
   # RFL
   uv run python experiments/run_fo_cycles.py --mode=rfl --cycles=1000 --out=results/fo_rfl.jsonl
   ```

2. **Generate Dyno Chart:**
   ```powershell
   uv run python experiments/analyze_abstention_curves.py `
       --baseline results/fo_baseline.jsonl `
       --rfl results/fo_rfl.jsonl `
       --window-size 100 `
       --burn-in 200
   ```

## Notes

- Test runtime: ~3-5 minutes is normal (does full derivation, attestation, RFL)
- Redis is optional - test degrades gracefully if Redis is unavailable
- Database is session-scoped - migrations run once per test session via `tests/conftest.py`
- Integration fixture (`first_organism_db`) no longer re-runs migrations to avoid duplicates

