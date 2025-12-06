# SPARK Infrastructure Checklist

This document provides a step-by-step checklist for setting up and running the First Organism SPARK tests.

## Prerequisites

### 1. Docker Desktop

**Check if Docker Desktop is running:**
```powershell
docker --version
docker ps
```

**If Docker is not running:**
- Start Docker Desktop from the Start menu
- Wait for the Docker icon in the system tray to show "Docker Desktop is running"
- Verify with: `docker ps`

**Common Error:** "Cannot connect to the Docker daemon"
- **Solution:** Ensure Docker Desktop is fully started (not just installed)

---

### 2. Database and Redis Containers

**Check if containers are running:**
```powershell
docker ps
```

**Expected containers:**
- A container with `postgres` in the name (e.g., `first_organism_postgres`, `postgres`)
- A container with `redis` in the name (e.g., `first_organism_redis`, `redis`)

**If containers are not running:**

**Option A: Using docker-compose (recommended)**
```powershell
# Navigate to project root
cd C:\dev\mathledger

# Start containers using the First Organism docker-compose
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d

# Or use the main docker-compose (if using standard setup)
docker compose up -d postgres redis
```

**Option B: Check existing containers**
```powershell
# List all containers (including stopped)
docker ps -a

# Start existing containers
docker start <container_name>
```

**Verify containers are healthy:**
```powershell
# Check container status
docker compose -f ops/first_organism/docker-compose.yml ps

# Or for main docker-compose
docker compose ps
```

**Common Errors:**
- **"Named pipe not found"** or **"Socket not connected"**
  - **Solution:** Docker Desktop is not running or not fully initialized
  - **Fix:** Restart Docker Desktop and wait 30 seconds, then retry

- **"Connection refused"** or **"Cannot connect to localhost:5432"**
  - **Solution:** Container is not running or port is not exposed
  - **Fix:** Check `docker ps` and ensure port mapping shows `0.0.0.0:5432->5432/tcp` for postgres

---

### 3. Environment Configuration

**Create `.env.first_organism` file:**

1. Copy the template:
   ```powershell
   cp config/first_organism.env.template .env.first_organism
   ```

2. Edit `.env.first_organism` and set the following values:
   - `POSTGRES_USER` - Database username
   - `POSTGRES_PASSWORD` - Database password (minimum 12 characters)
   - `POSTGRES_DB` - Database name
   - `REDIS_PASSWORD` - Redis password (minimum 12 characters)
   - `LEDGER_API_KEY` - API key (minimum 16 characters)
   - `CORS_ALLOWED_ORIGINS` - Comma-separated list of allowed origins

**Verify the file exists:**
```powershell
Test-Path .env.first_organism
```

**Common Errors:**
- **"Bad password"** or authentication failure
  - **Solution:** Ensure `.env.first_organism` values match the credentials used by your Docker containers
  - **Fix:** Check `docker compose` environment variables or container environment

- **"Environment variable not set"**
  - **Solution:** `.env.first_organism` is missing or incomplete
  - **Fix:** Ensure all required variables are set (see template)

---

## Running SPARK Tests

### Quick Start

1. **Ensure prerequisites are met** (see above)

2. **Run the launcher script:**
   ```powershell
   .\scripts\run_first_organism_spark.ps1
   ```

3. **Check the output:**
   - Test results will be displayed in the console
   - Full log saved to `ops/logs/SPARK_run_log.txt`

### Manual Test Execution

If you prefer to run the test manually:

```powershell
# Load environment (PowerShell)
$env:POSTGRES_USER = "first_organism_user"
$env:POSTGRES_PASSWORD = "secure_test_password_123"
$env:POSTGRES_DB = "mathledger_first_organism"
$env:REDIS_PASSWORD = "secure_redis_password_456"
$env:LEDGER_API_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"

$env:DATABASE_URL = "postgresql://$env:POSTGRES_USER:$env:POSTGRES_PASSWORD@localhost:5432/$env:POSTGRES_DB?sslmode=disable"
$env:REDIS_URL = "redis://:$env:REDIS_PASSWORD@localhost:6380/0"
$env:FIRST_ORGANISM_TESTS = "true"

# Run test
uv run pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_happy_path -v -s
```

---

## Common Errors and Solutions

### 1. Named Pipe / Socket Errors

**Error:**
```
Error: Cannot connect to the Docker daemon. Is the docker daemon running?
```

**Solution:**
- Start Docker Desktop
- Wait for it to fully initialize (check system tray icon)
- Verify with: `docker ps`

---

### 2. Connection Refused

**Error:**
```
psycopg.OperationalError: connection to server at "localhost" (127.0.0.1), port 5432 failed
```

**Solution:**
- Check if PostgreSQL container is running: `docker ps | Select-String postgres`
- Check if port is exposed: `docker ps` should show `0.0.0.0:5432->5432/tcp`
- Verify container is healthy: `docker compose ps`

---

### 3. Authentication Failure

**Error:**
```
psycopg.OperationalError: password authentication failed for user "first_organism_user"
```

**Solution:**
- Ensure `.env.first_organism` credentials match container credentials
- Check container environment: `docker inspect <container_name> | Select-String POSTGRES_PASSWORD`
- If using `ops/first_organism/docker-compose.yml`, ensure `--env-file .env.first_organism` is used

---

### 4. Redis Connection Error

**Error:**
```
redis.exceptions.AuthenticationError: AUTH <password> failed
```

**Solution:**
- Verify `REDIS_PASSWORD` in `.env.first_organism` matches container password
- Check Redis container: `docker exec <redis_container> redis-cli -a <password> ping`
- Ensure Redis is listening on correct port (6379 for main, 6380 for first_organism)

---

### 5. Test Skipped

**Error:**
```
SKIPPED [1] tests/integration/test_first_organism.py: Database not available
```

**Solution:**
- Database connection failed (see errors above)
- Check `FIRST_ORGANISM_TESTS` environment variable is set to `"true"`
- Verify database is accessible: `psql $env:DATABASE_URL -c "SELECT 1"`

---

## Verification Commands

**Quick health check:**
```powershell
# Docker status
docker ps

# Database connection
$env:DATABASE_URL = "postgresql://first_organism_user:secure_test_password_123@localhost:5432/mathledger_first_organism?sslmode=disable"
psql $env:DATABASE_URL -c "SELECT version();"

# Redis connection
$env:REDIS_URL = "redis://:secure_redis_password_456@localhost:6380/0"
# Use redis-cli if available, or test via Python
python -c "import redis; r = redis.from_url('$env:REDIS_URL'); print(r.ping())"
```

---

## Next Steps

If all checks pass and the test still fails:
1. Check `ops/logs/SPARK_run_log.txt` for detailed error messages
2. Review test output for specific assertion failures
3. Verify database schema is up to date: `uv run python scripts/run-migrations.py`
4. Check for port conflicts: `netstat -an | Select-String "5432|6380"`

