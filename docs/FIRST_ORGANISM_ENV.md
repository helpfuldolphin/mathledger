# First Organism Environment Setup

This document outlines the procedure to set up the hardened security environment for Operation First Organism (FO) integration testing.

## Purpose

The First Organism environment provides a **hardened local test environment** that enforces strict security requirements to prevent accidental execution against production or unsecured development databases. All FO tests must run with:

- Strong, non-default credentials (minimum 12 characters for passwords, 16 for API keys)
- Explicit CORS origins (no wildcards)
- Explicit runtime environment marker (`RUNTIME_ENV=test_hardened`)
- All security enforcer checks passing

## 1. Environment Configuration

The environment template is located at `config/first_organism.env.template`. Copy this to `.env.first_organism` in the project root and customize the values.

### Required Environment Variables

All of the following variables are **required** and must satisfy security requirements:

| Variable | Purpose | Requirements | Example |
|----------|---------|--------------|---------|
| `RUNTIME_ENV` | Runtime environment marker | Must be `test_hardened` | `test_hardened` |
| `DATABASE_URL` | PostgreSQL connection string | Strong password (12+ chars), no banned passwords | See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format |
| `REDIS_URL` | Redis connection string | Password required (12+ chars), no banned passwords | See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format |
| `POSTGRES_USER` | PostgreSQL username | Minimum 3 characters | `first_organism_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | Minimum 12 characters, not in banned list | `secure_test_password_123` |
| `POSTGRES_DB` | PostgreSQL database name | Minimum 3 characters | `mathledger_first_organism` |
| `REDIS_PASSWORD` | Redis password | Minimum 12 characters, not in banned list | `secure_redis_password_456` |
| `LEDGER_API_KEY` | API authentication key | Minimum 16 characters, high entropy | `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6` |
| `CORS_ALLOWED_ORIGINS` | Allowed CORS origins | Comma-separated, **no wildcards** | `http://localhost:3000,http://localhost:8000` |

### Optional but Recommended Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MAX_REQUEST_BODY_BYTES` | Maximum request body size | `10485760` (10MB) |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Rate limit for API requests | `1000` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window | `60` |
| `QUEUE_KEY` | Redis queue key prefix | `ml:first_organism:jobs` |

### Security Requirements

The environment enforcer (`backend/security/first_organism_enforcer.py`) validates:

1. **Password Strength**: 
   - Minimum 12 characters for database/Redis passwords
   - Minimum 16 characters for API keys
   - No banned passwords (e.g., `mlpass`, `postgres`, `password`, `devkey`, etc.)

2. **CORS Policy**:
   - No wildcard (`*`) allowed
   - Must specify explicit origins

3. **Runtime Environment**:
   - `RUNTIME_ENV` must be `test_hardened` (or `first_organism`/`integration` for backward compatibility)

### Template File

See `config/first_organism.env.template` for a complete template with example values. **Do not commit actual credentials to version control.**

## 2. Setup Instructions

### Step 1: Create Environment File

Copy the template to your local environment file:

```bash
# Linux/Mac/WSL
cp config/first_organism.env.template .env.first_organism

# Windows PowerShell
Copy-Item config/first_organism.env.template .env.first_organism
```

### Step 2: Customize Credentials

Edit `.env.first_organism` and replace all placeholder values with strong, unique credentials:

- Generate strong passwords (use a password manager or `openssl rand -base64 24`)
- Ensure `RUNTIME_ENV=test_hardened` is set
- Verify no wildcards in `CORS_ALLOWED_ORIGINS`

### Step 3: Validate Environment

Run the validation tool to ensure all requirements are met:

```bash
uv run python tools/validate_first_organism_env.py .env.first_organism
```

The validator checks:
- All required variables are present
- Password strength requirements
- No banned passwords
- No CORS wildcards
- Proper URL formats

### Step 4: Infrastructure Setup (Docker)

Spin up fresh Postgres and Redis containers with the credentials from your `.env.first_organism`:

#### PostgreSQL
```bash
docker run -d --name ml-first-organism-db \
  -e POSTGRES_USER=first_organism_user \
  -e POSTGRES_PASSWORD=secure_test_password_123 \
  -e POSTGRES_DB=mathledger_first_organism \
  -p 5432:5432 \
  postgres:15
```

#### Redis
```bash
docker run -d --name ml-first-organism-redis \
  -p 6380:6379 \
  redis:7 --requirepass "secure_redis_password_456"
```

*Note: See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical connection string formats and port selection guide (5432 vs 5433, 6379 vs 6380).*

## 2.1. SSL Configuration

PostgreSQL SSL mode must be correctly configured in your `DATABASE_URL`:

- **For local Docker Postgres** (default setup): Use `?sslmode=disable` since local containers typically don't have SSL certificates configured.
- **For remote Postgres with SSL**: Use `?sslmode=require` to enforce encrypted connections.

**Canonical Connection Strings:** See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for complete connection string formats, SSL configuration, and troubleshooting.

**Troubleshooting SSL Errors:**
- If you see `could not send SSL negotiation packet` or similar SSL errors, check your `DATABASE_URL`:
  - Local Docker: Ensure `?sslmode=disable` is present
  - Remote database: Ensure `?sslmode=require` is present
- Test connectivity with: `uv run python scripts/test_db_connection.py`
- See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for detailed troubleshooting.

## 3. Loading Environment Variables

Before running tests, load the environment variables from `.env.first_organism`:

### Option A: Shell Export (Linux/Mac/WSL)
```bash
# Export variables (skip comments and empty lines)
export $(grep -v '^#' .env.first_organism | grep -v '^$' | xargs)

# Verify RUNTIME_ENV is set correctly
echo "RUNTIME_ENV=$RUNTIME_ENV"  # Should output: RUNTIME_ENV=test_hardened
```

### Option B: PowerShell (Windows)
```powershell
# Read and set environment variables
Get-Content .env.first_organism | Where-Object { 
    $_ -match '^\w+=' -and $_ -notmatch '^\s*#' 
} | ForEach-Object {
    $name, $value = $_.Split('=', 2)
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

# Verify RUNTIME_ENV is set correctly
Write-Host "RUNTIME_ENV=$env:RUNTIME_ENV"  # Should output: RUNTIME_ENV=test_hardened
```

## 4. Running Integration Tests

Once the environment is loaded and infrastructure is running:

```bash
# Run migrations (required for fresh DB)
uv run python scripts/run-migrations.py

# Run First Organism tests
FIRST_ORGANISM_TESTS=true uv run pytest -m first_organism -v

# Or use SPARK trigger
SPARK_RUN=1 uv run pytest -m first_organism -v
```

The tests will automatically:
- Enforce security requirements via `enforce_first_organism_env()`
- Check that `RUNTIME_ENV=test_hardened` (or log warning if not)
- Fail fast if insecure credentials are detected

## 5. Verification

### Verify Environment Enforcement

Test that the enforcer correctly rejects insecure configurations:

```bash
# Test 1: Missing DATABASE_URL
unset DATABASE_URL
uv run python -c "from backend.security.runtime_env import get_database_url; print(get_database_url())"
# Expected: MissingEnvironmentVariable exception

# Test 2: Weak password
export DATABASE_URL="postgresql://user:mlpass@localhost:5432/db"
uv run python -c "from backend.security.first_organism_enforcer import enforce_first_organism_env; enforce_first_organism_env()"
# Expected: InsecureCredentialsError with violation message

# Test 3: CORS wildcard
export CORS_ALLOWED_ORIGINS="*"
uv run python -c "from backend.security.first_organism_enforcer import enforce_first_organism_env; enforce_first_organism_env()"
# Expected: InsecureCredentialsError about wildcard
```

### Verify Runtime Environment Check

The tests include an assertion helper that checks `RUNTIME_ENV=test_hardened`. If not set correctly, tests will log a warning or skip.

## 6. Troubleshooting

### Common Issues

**"InsecureCredentialsError: First Organism security check FAILED"**
- Ensure all passwords are 12+ characters
- Check that no banned passwords are used
- Verify `CORS_ALLOWED_ORIGINS` has no wildcards
- Run `tools/validate_first_organism_env.py` to see specific violations

**"RUNTIME_ENV is not 'test_hardened'"**
- Set `RUNTIME_ENV=test_hardened` in `.env.first_organism`
- Ensure environment variables are loaded before running tests

**"MissingEnvironmentVariable"**
- Verify all required variables are set in `.env.first_organism`
- Ensure environment variables are exported/loaded correctly

## Related Documentation

- `config/first_organism.env.template` - Environment template with example values
- `backend/security/first_organism_enforcer.py` - Security enforcer implementation
- `tools/validate_first_organism_env.py` - Standalone validation tool
- `tests/integration/first_organism_conftest.py` - Pytest fixtures for FO tests
