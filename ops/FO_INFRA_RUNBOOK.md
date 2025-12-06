# First Organism Infrastructure Runbook

**Purpose:** Operational procedures for starting, stopping, and troubleshooting First Organism Docker infrastructure.

**Scope:** Docker Compose orchestration only. Does not cover test execution, experiments, or data analysis.

**Evidence Base:** This runbook documents scripts and files that exist on disk as of 2025-01-18.

---

## What Exists

### Files Verified

| File | Location | Purpose | Status |
|------|----------|---------|--------|
| Docker Compose configuration | `ops/first_organism/docker-compose.yml` | Defines PostgreSQL and Redis services | ✅ EXISTS |
| Environment template | `ops/first_organism/first_organism.env.template` | Template for `.env.first_organism` | ✅ EXISTS |
| Startup script (full) | `scripts/start_first_organism_infra.ps1` | Starts infra + waits for health checks | ✅ EXISTS |
| Startup script (minimal) | `scripts/start_fo_docker.ps1` | Starts infra + shows status (no wait loop) | ✅ EXISTS |
| Log collection script | `scripts/collect_first_organism_logs.ps1` | Gathers diagnostic logs | ✅ EXISTS |
| SPARK test launcher | `scripts/run_first_organism_spark.ps1` | Runs integration test | ✅ EXISTS |

### Container Names (From docker-compose.yml)

- `first_organism_postgres` - PostgreSQL 16-alpine
- `first_organism_redis` - Redis 7-alpine

### Network & Volumes

- Network: `first_organism_network` (172.28.0.0/16)
- Volumes: `first_organism_pgdata`, `first_organism_redis`

---

## Phase-I RFL Evidence Snapshot

**Canonical Truth Source:** See `docs/RFL_PHASE_I_TRUTH_SOURCE.md` for authoritative cycle counts and evidence claims.

### RFL Evidence Files (Verified on Disk)

| File | Location | Cycles | Schema | Abstention | Purpose | Status |
|------|----------|--------|--------|------------|---------|--------|
| Baseline run | `results/fo_baseline.jsonl` | 1000 (0–999) | Old (no top-level `status`/`method`/`abstention`) | 100% | Baseline negative control (RFL OFF) | ✅ EXISTS |
| RFL plumbing run | `results/fo_rfl.jsonl` | 1001 (0–1000) | New (`status`/`method`/`abstention` present) | 100% | Hermetic negative control / plumbing validation | ✅ EXISTS |
| RFL sanity run | `results/fo_rfl_50.jsonl` | 21 (0–20) | New | 100% | Small RFL plumbing / negative control demo | ⚠️ INCOMPLETE |
| RFL partial run | `results/fo_rfl_1000.jsonl` | 11 (0–10) | New | 100% | Incomplete run | ⚠️ INCOMPLETE |

**Critical Facts:**
- **All Phase-I RFL runs are 100% abstention by design** (hermetic lean-disabled mode)
- **Phase I demonstrates execution infrastructure only, NOT uplift or performance improvement**
- **fo_rfl.jsonl** (1001 cycles) is the complete hermetic negative-control run
- **fo_rfl_50.jsonl** and **fo_rfl_1000.jsonl** are incomplete and should not be used for cycle count claims
- **No empirical RFL uplift in Phase I** — all runs are negative controls validating plumbing/attestation/determinism

**File Contents:**
- **Old schema** (fo_baseline.jsonl): `cycle`, `slice_name`, `mode`, `roots`, `derivation`
- **New schema** (RFL files): Adds `status`, `method`, `abstention`; includes `rfl` section if mode=rfl

### Infrastructure Requirements Clarification

**RFL runs do NOT require Docker infrastructure.**
- RFL cycle experiments (`run_fo_cycles.py`) are standalone, hermetic, and use mocked/in-memory components
- No PostgreSQL or Redis containers needed for RFL runs
- RFL experiments execute deterministically without external dependencies
- **Phase-I RFL is file-based only** — no DB writes, no network access, no security surfaces required

**FO infrastructure (Docker) is required ONLY for:**
- First Organism SPARK integration tests (`test_first_organism_closed_loop_happy_path`)
- First Organism integration tests that require real database/Redis connectivity
- Full pipeline validation (UI Event → Curriculum Gate → Derivation → Lean Verify → Attestation)

**Summary:**
- **FO infrastructure** = Required for FO tests/integration (SPARK, closed-loop tests)
- **RFL evidence files** = Generated independently, hermetic, no Docker needed
- **Phase-I RFL** = Negative control / plumbing validation only (100% abstention, no uplift signal)

---

## Prerequisites

### Required

1. **Docker Desktop** installed and running
   - Verify: `docker ps` succeeds
   - If not running: Start Docker Desktop application, wait for system tray icon

2. **`.env.first_organism`** file in project root
   - Create from template: `Copy-Item ops/first_organism/first_organism.env.template .env.first_organism`
   - Generate secure credentials (see template for PowerShell commands)
   - Replace all `<REPLACE_...>` placeholders
   - Minimum requirements:
     - `POSTGRES_PASSWORD`: 12+ characters, not in banned list
     - `REDIS_PASSWORD`: 12+ characters
     - `LEDGER_API_KEY`: 16+ characters, high entropy

### Optional

- PowerShell 7+ (`pwsh.exe`) for script execution
- Direct `docker compose` commands work without scripts

---

## Starting Infrastructure

### Option 1: Minimal Script (Quick Start)

```powershell
.\scripts\start_fo_docker.ps1
```

**What it does:**
1. Checks Docker is running
2. Validates `.env.first_organism` exists
3. Runs: `docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d`
4. Displays container status (does not wait for health checks)

**When to use:** Quick startup when you know infrastructure is healthy.

**Limitation:** Does not verify services are ready before exit.

### Option 2: Full Script (With Health Checks)

```powershell
.\scripts\start_first_organism_infra.ps1
```

**What it does:**
1. Checks Docker is running
2. Validates `.env.first_organism` exists
3. Starts services via docker compose
4. Waits up to 60 seconds for health checks to pass
5. Exits with error if services don't become healthy

**When to use:** When you need verified readiness before running tests.

### Option 3: Manual Command

```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
```

**Verify status:**
```powershell
docker compose -f ops/first_organism/docker-compose.yml ps
```

---

## Stopping Infrastructure

### Stop Containers (Keep Data)

```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down
```

### Stop Containers and Remove Volumes (Clean State)

```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down -v
```

**Warning:** This deletes all database and Redis data. Use only for clean test runs.

**Manual volume cleanup (if needed):**
```powershell
docker volume rm first_organism_pgdata first_organism_redis
```

---

## Checking Status

### Container Status

```powershell
docker compose -f ops/first_organism/docker-compose.yml ps
```

**Expected output:**
```
NAME                        IMAGE               COMMAND             SERVICE   CREATED        STATUS          PORTS
first_organism_postgres     postgres:16-alpine  docker-entrypoint…  postgres  X minutes ago  Up X minutes   127.0.0.1:5432->5432/tcp
first_organism_redis        redis:7-alpine      redis-server …      redis     X minutes ago  Up X minutes   127.0.0.1:6380->6379/tcp
```

### Container Logs

**PostgreSQL:**
```powershell
docker compose -f ops/first_organism/docker-compose.yml logs postgres
```

**Redis:**
```powershell
docker compose -f ops/first_organism/docker-compose.yml logs redis
```

**Follow logs (real-time):**
```powershell
docker compose -f ops/first_organism/docker-compose.yml logs -f
```

### Health Check Status

Containers have health checks defined. Check health status:
```powershell
docker inspect first_organism_postgres --format '{{.State.Health.Status}}'
docker inspect first_organism_redis --format '{{.State.Health.Status}}'
```

Expected: `healthy`

---

## Ports and Connections

### Service Endpoints

- **PostgreSQL:** `localhost:5432` (bind: `127.0.0.1:5432`)
- **Redis:** `localhost:6380` (bind: `127.0.0.1:6380`, container port: `6379`)

**Note:** Redis uses port 6380 to avoid conflicts with development Redis on 6379.

### Connection Strings

**PostgreSQL:**
```
postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
```

**Redis:**
```
redis://:${REDIS_PASSWORD}@localhost:6380/0
```

See `ops/first_organism/first_organism.env.template` for full connection string format.

---

## Common Failures and Remedies

### 1. Docker Not Running

**Symptoms:**
```
Error: Cannot connect to the Docker daemon
```

**Remedy:**
1. Start Docker Desktop application
2. Wait for system tray icon to show "Docker Desktop is running"
3. Verify: `docker ps`
4. Retry startup script

---

### 2. `.env.first_organism` Missing

**Symptoms:**
```
❌ .env.first_organism not found
```

**Remedy:**
1. Copy template: `Copy-Item ops/first_organism/first_organism.env.template .env.first_organism`
2. Generate credentials (see template for commands)
3. Replace all `<REPLACE_...>` placeholders
4. Verify file exists: `Test-Path .env.first_organism`

---

### 3. Port Already in Use

**Symptoms:**
```
Error: bind: address already in use
```

**Remedy:**
1. Check what's using the port:
   ```powershell
   netstat -an | Select-String "5432|6380"
   ```
2. Stop conflicting services or containers
3. Alternative: Stop other Docker containers using these ports
   ```powershell
   docker ps
   docker stop <container_name>
   ```

---

### 4. Container Fails to Start

**Symptoms:**
```
Container exits immediately or shows "unhealthy" status
```

**Remedy:**
1. Check logs for specific error:
   ```powershell
   docker compose -f ops/first_organism/docker-compose.yml logs postgres
   docker compose -f ops/first_organism/docker-compose.yml logs redis
   ```
2. Verify `.env.first_organism` credentials are valid (no special characters breaking shell parsing)
3. Check Docker resource limits (Memory, CPU in Docker Desktop settings)
4. Try clean restart:
   ```powershell
   docker compose -f ops/first_organism/docker-compose.yml down -v
   docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
   ```

---

### 5. Health Checks Fail

**Symptoms:**
```
Services start but remain "unhealthy" after 60+ seconds
```

**Remedy:**
1. Inspect health check output:
   ```powershell
   docker inspect first_organism_postgres --format '{{json .State.Health}}' | ConvertFrom-Json
   docker inspect first_organism_redis --format '{{json .State.Health}}' | ConvertFrom-Json
   ```
2. Check container logs for authentication or initialization errors
3. Verify environment variables are correctly loaded:
   ```powershell
   docker exec first_organism_postgres env | Select-String POSTGRES
   docker exec first_organism_redis env | Select-String REDIS
   ```
4. If credentials changed, recreate containers:
   ```powershell
   docker compose -f ops/first_organism/docker-compose.yml down -v
   docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
   ```

---

### 6. Authentication Failures (During Test Runs)

**Symptoms:**
```
psycopg.OperationalError: password authentication failed
redis.exceptions.AuthenticationError: AUTH failed
```

**Remedy:**
1. Verify `.env.first_organism` credentials match container credentials
2. Check connection strings in test environment match `.env.first_organism`
3. Ensure `DATABASE_URL` and `REDIS_URL` are constructed correctly (see template)
4. Redis port note: Use `6380` (host port), not `6379` (container port)

---

## Log Collection for Troubleshooting

### Automated Collection

```powershell
.\scripts\collect_first_organism_logs.ps1
```

**Output:** `ops/logs/SPARK_diag_bundle.txt`

**Contents:**
- PostgreSQL container logs
- Redis container logs
- Latest `SPARK_run_log.txt` (if exists)
- Container status summary

**Use case:** Share diagnostic bundle when troubleshooting infrastructure issues.

### Manual Collection

**Individual container logs:**
```powershell
docker logs first_organism_postgres > ops/logs/postgres.log
docker logs first_organism_redis > ops/logs/redis.log
```

**Compose logs:**
```powershell
docker compose -f ops/first_organism/docker-compose.yml logs > ops/logs/compose.log
```

---

## Verification Checklist

Before running tests, verify:

- [ ] Docker Desktop is running (`docker ps` succeeds)
- [ ] `.env.first_organism` exists and has no `<REPLACE_...>` placeholders
- [ ] Containers are running: `docker compose -f ops/first_organism/docker-compose.yml ps`
- [ ] Containers are healthy: Health status shows `healthy` (or wait for startup script to confirm)
- [ ] Ports accessible:
  - PostgreSQL: `Test-NetConnection localhost -Port 5432`
  - Redis: `Test-NetConnection localhost -Port 6380`

---

## Script Behavior Comparison

### `start_fo_docker.ps1` (Minimal)

- ✅ Checks Docker availability
- ✅ Validates `.env.first_organism` exists
- ✅ Starts services
- ✅ Shows container status
- ❌ Does NOT wait for health checks
- ❌ Does NOT verify services are ready

**Use when:** Quick startup, you'll verify readiness separately.

### `start_first_organism_infra.ps1` (Full)

- ✅ Checks Docker availability
- ✅ Validates `.env.first_organism` exists
- ✅ Starts services
- ✅ Waits for health checks (up to 60 seconds)
- ✅ Verifies services are ready
- ✅ Exits with error if health checks fail

**Use when:** You need verified readiness before proceeding.

---

## Related Documentation

- **Comprehensive Runbook:** `ops/RUNBOOK_FIRST_ORGANISM_AND_DYNO.md` - Full First Organism workflow (infra + experiments + Dyno Chart)
- **SPARK Checklist:** `ops/SPARK_INFRA_CHECKLIST.md` - SPARK-specific troubleshooting
- **Environment Template:** `ops/first_organism/first_organism.env.template` - Environment variable documentation

---

**Last Updated:** 2025-01-18  
**Maintainer:** GEMINI K — Docker Helper & Log Collator  
**Mode:** Sober Truth / Reviewer-2 Compliant  
**Evidence Base:** Files verified on disk; commands tested against actual docker-compose.yml; RFL evidence files verified

