# First Organism Local Development Guide

This guide walks you through getting First Organism (FO) running on your local machine. Follow these steps to bring the system to life without needing help on call.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

### 1. Docker Desktop

**Installation:**
- Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows
- Start Docker Desktop and wait for it to fully initialize (check the system tray icon)

**Verification:**
```powershell
docker --version
docker ps
```

If you see "Cannot connect to the Docker daemon", Docker Desktop is not running. Start it from the Start menu and wait for it to fully initialize.

### 2. uv Package Manager

**Installation:**
- Follow the [uv installation guide](https://github.com/astral-sh/uv#installation)
- Or use: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Verification:**
```powershell
uv --version
```

### 3. Environment Configuration File

You need a `.env.first_organism` file in the project root with secure credentials.

**Setup:**
```powershell
# Copy the template
Copy-Item ops/first_organism/first_organism.env.template .env.first_organism

# Or if using config directory:
Copy-Item config/first_organism.env.template .env.first_organism
```

**Required Variables:**
- `POSTGRES_USER` - Database username
- `POSTGRES_PASSWORD` - Database password (minimum 12 characters)
- `POSTGRES_DB` - Database name
- `REDIS_PASSWORD` - Redis password (minimum 12 characters)
- `LEDGER_API_KEY` - API key (minimum 16 characters)
- `CORS_ALLOWED_ORIGINS` - Comma-separated list of allowed origins

**Verification:**
```powershell
Test-Path .env.first_organism
```

> **Note:** For detailed environment setup instructions, see [`FIRST_ORGANISM_ENV.md`](FIRST_ORGANISM_ENV.md).

---

## Step-by-Step Setup

### Step 1: Start Infrastructure

Start the Docker containers (PostgreSQL and Redis) required for First Organism:

```powershell
.\scripts\start_first_organism_infra.ps1
```

**What this does:**
- Checks Docker Desktop is running
- Verifies `.env.first_organism` exists
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

**If it fails:**
- Ensure Docker Desktop is running: `docker ps`
- Check container logs: `docker compose -f ops/first_organism/docker-compose.yml logs`
- Verify `.env.first_organism` has correct credentials

> **Troubleshooting:** See [`ops/SPARK_INFRA_CHECKLIST.md`](../ops/SPARK_INFRA_CHECKLIST.md) for detailed error resolution.

---

### Step 2: Run Migrations

Initialize the database schema (run this once, or after schema changes):

```powershell
uv run python scripts/run-migrations.py
```

**What this does:**
- Connects to the PostgreSQL database using credentials from `.env.first_organism`
- Applies all pending migrations to create the required tables and schema

**Expected output:**
- Migration files are executed in order
- No errors should appear

**If it fails:**
- Verify database is accessible: `docker ps | Select-String postgres`
- Check `.env.first_organism` has correct `DATABASE_URL` or individual credentials
- Ensure PostgreSQL container is healthy: `docker compose -f ops/first_organism/docker-compose.yml ps`

---

### Step 3: Run SPARK

Execute the First Organism closed-loop integration test:

```powershell
.\scripts\run_first_organism_spark.ps1
```

**What this does:**
- Loads environment variables from `.env.first_organism`
- Verifies Docker containers are running
- Runs the SPARK test: `test_first_organism_closed_loop_happy_path`
- Saves full output to `ops/logs/SPARK_run_log.txt`

**Expected output:**
```
========================================
SPARK test PASSED
========================================
Full log saved to: ops/logs/SPARK_run_log.txt
```

**If SPARK fails:**
- Check the console output for specific error messages
- Review the full log: `ops/logs/SPARK_run_log.txt`
- Search for `[SKIP][FO]` messages in the log (these indicate why tests were skipped)

---

### Step 4: Troubleshooting SPARK Failures

If SPARK fails, follow these steps:

#### 4.1. Collect Logs

The SPARK script automatically saves logs to `ops/logs/SPARK_run_log.txt`. You can also manually check:

```powershell
# View the log file
Get-Content ops/logs/SPARK_run_log.txt

# Or open in your editor
code ops/logs/SPARK_run_log.txt
```

#### 4.2. Search for Skip Messages

The test suite uses `[SKIP][FO]` markers to indicate why tests were skipped. Search the log:

```powershell
Select-String -Path ops/logs/SPARK_run_log.txt -Pattern "\[SKIP\]\[FO\]"
```

**Common skip reasons:**
- `[SKIP][FO] FIRST_ORGANISM_TESTS not set to true/SPARK_RUN` - Environment variable not set
- `[SKIP][FO] Migration failed: ...` - Database migration error
- `[SKIP][FO] Derivation pipeline produced no statements` - Test data generation issue
- `[SKIP][FO] Database not available` - Connection failure

#### 4.3. Common Issues

**Issue: "Cannot connect to Docker daemon"**
- **Solution:** Start Docker Desktop and wait for it to fully initialize

**Issue: "Connection refused" on port 5432**
- **Solution:** Verify PostgreSQL container is running: `docker ps | Select-String postgres`
- **Solution:** Check port mapping: `docker ps` should show `0.0.0.0:5432->5432/tcp`

**Issue: "Authentication failed"**
- **Solution:** Ensure `.env.first_organism` credentials match container credentials
- **Solution:** Verify container environment: `docker inspect <container_name> | Select-String POSTGRES_PASSWORD`

**Issue: "Migration failed"**
- **Solution:** Check database connection: `psql $env:DATABASE_URL -c "SELECT 1"`
- **Solution:** Verify migrations directory exists: `Test-Path migrations`

**Issue: Test skipped with `[SKIP][FO]`**
- **Solution:** Read the skip message in the log for the exact reason
- **Solution:** Check prerequisites (infra running, migrations applied, environment variables set)

> **Detailed troubleshooting:** See [`ops/SPARK_INFRA_CHECKLIST.md`](../ops/SPARK_INFRA_CHECKLIST.md) for comprehensive error resolution.

---

## Quick Reference

### Start Everything (First Time)
```powershell
# 1. Start infrastructure
.\scripts\start_first_organism_infra.ps1

# 2. Run migrations (once)
uv run python scripts/run-migrations.py

# 3. Run SPARK
.\scripts\run_first_organism_spark.ps1
```

### Daily Workflow
```powershell
# 1. Start infrastructure (if not already running)
.\scripts\start_first_organism_infra.ps1

# 2. Run SPARK
.\scripts\run_first_organism_spark.ps1
```

### Stop Infrastructure
```powershell
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism down
```

### Health Check
```powershell
# Check Docker
docker ps

# Check containers
docker compose -f ops/first_organism/docker-compose.yml ps

# Test database connection
# See docs/FIRST_ORGANISM_CONNECTION_STRINGS.md for canonical connection string format
$env:DATABASE_URL = "postgresql://first_organism_user:secure_test_password_123@localhost:5432/mathledger_first_organism?sslmode=disable"
psql $env:DATABASE_URL -c "SELECT version();"
```

---

## Related Documentation

- **[`FIRST_ORGANISM_ENV.md`](FIRST_ORGANISM_ENV.md)** - Detailed environment configuration guide
- **[`ops/SPARK_INFRA_CHECKLIST.md`](../ops/SPARK_INFRA_CHECKLIST.md)** - Comprehensive troubleshooting checklist
- **[`FIRST_ORGANISM.md`](FIRST_ORGANISM.md)** - First Organism architecture and design overview

---

## Success Criteria

You'll know First Organism is working when:

1. ✅ Infrastructure starts without errors
2. ✅ Migrations complete successfully
3. ✅ SPARK test passes with `[PASS] FIRST ORGANISM` in the output
4. ✅ No `[SKIP][FO]` messages appear in the log (unless expected)

If all steps complete successfully, you're ready to develop and test First Organism features locally!

