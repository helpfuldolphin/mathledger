# First Organism Connection Strings

**Purpose:** Canonical reference for First Organism (FO) database and Redis connection string formats. This document centralizes all connection string patterns to prevent rediscovery and ensure consistency across documentation and configuration.

---

## Quick Reference

### PostgreSQL

**Canonical Format:**
```
postgresql://first_organism_user:<password>@localhost:5432/mathledger_first_organism?sslmode=disable
```

**Example (with placeholder password):**
```
postgresql://first_organism_user:secure_test_password_123@localhost:5432/mathledger_first_organism?sslmode=disable
```

### Redis

**Canonical Format:**
```
redis://:<password>@localhost:6380/0
```

**Example (with placeholder password):**
```
redis://:r3d1s_f1rst_0rg_s3cur3!@localhost:6380/0
```

---

## Port Selection Guide

### PostgreSQL: 5432 vs 5433

| Port | Use Case | When to Use |
|------|----------|-------------|
| **5432** | **First Organism** | Standard port for FO integration tests. The `ops/first_organism/docker-compose.yml` binds to `127.0.0.1:5432:5432`. |
| **5433** | Development/Other | Alternative port when 5432 is occupied by another Postgres instance. **Not used for First Organism.** |

**Decision Rule:** First Organism **always uses port 5432**. If port 5432 is already in use, stop the conflicting service or use a different environment (not First Organism).

### Redis: 6379 vs 6380

| Port | Use Case | When to Use |
|------|----------|-------------|
| **6379** | Development/General | Standard Redis port for development environments. May conflict with other Redis instances. |
| **6380** | **First Organism** | **Dedicated port for FO** to avoid conflicts with development Redis. The `ops/first_organism/docker-compose.yml` maps `127.0.0.1:6380:6379` (host:container). |

**Decision Rule:** First Organism **always uses port 6380** for Redis. This is explicitly configured in the docker-compose file to prevent conflicts with development Redis on 6379.

---

## Connection String Components

### PostgreSQL Connection String Breakdown

```
postgresql://[user]:[password]@[host]:[port]/[database]?[parameters]
```

| Component | First Organism Value | Notes |
|-----------|---------------------|-------|
| `user` | `first_organism_user` or `first_organism_admin` | Username defined in `POSTGRES_USER` env var |
| `password` | (12+ character secure password) | Must meet security requirements (see `docs/FIRST_ORGANISM_ENV.md`) |
| `host` | `localhost` or `127.0.0.1` | When connecting from host machine |
| `host` | `postgres` | When connecting from within Docker network |
| `port` | `5432` | **Always 5432 for First Organism** |
| `database` | `mathledger_first_organism` | Database name from `POSTGRES_DB` env var |
| `parameters` | `sslmode=disable` | Required for local Docker Postgres (no SSL certs) |

**SSL Mode Notes:**
- **Local Docker:** Use `?sslmode=disable` (containers typically don't have SSL certificates)
- **Remote Postgres:** Use `?sslmode=require` to enforce encrypted connections

### Redis Connection String Breakdown

```
redis://:[password]@[host]:[port]/[database]
```

| Component | First Organism Value | Notes |
|-----------|---------------------|-------|
| `password` | (12+ character secure password) | Must meet security requirements (see `docs/FIRST_ORGANISM_ENV.md`) |
| `host` | `localhost` or `127.0.0.1` | When connecting from host machine |
| `host` | `redis` | When connecting from within Docker network |
| `port` | `6380` | **Always 6380 for First Organism** (maps to container's 6379) |
| `database` | `0` | Default Redis database number |

**Note:** The password appears after `redis://:` (empty username field). The colon before the password is required.

---

## Environment Variable Mapping

These connection strings are typically constructed from environment variables:

### PostgreSQL

```bash
# Individual components
POSTGRES_USER=first_organism_user
POSTGRES_PASSWORD=<secure_password>
POSTGRES_DB=mathledger_first_organism
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Full connection string
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}?sslmode=disable
```

### Redis

```bash
# Individual components
REDIS_PASSWORD=<secure_password>
REDIS_HOST=localhost
REDIS_PORT=6380

# Full connection string
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0
```

---

## Docker Compose Port Mapping

The First Organism infrastructure (`ops/first_organism/docker-compose.yml`) maps ports as follows:

| Service | Container Port | Host Port | Connection String Port |
|---------|---------------|-----------|----------------------|
| PostgreSQL | 5432 | 5432 | **5432** |
| Redis | 6379 | 6380 | **6380** |

**Important:** When connecting from the host machine, use the **host port** (5432 for Postgres, 6380 for Redis). When connecting from within the Docker network, use the **container port** (5432 for Postgres, 6379 for Redis) and the service name as hostname.

---

## Common Patterns

### Pattern 1: Host Machine Connection

```bash
# PostgreSQL
DATABASE_URL=postgresql://first_organism_user:password@localhost:5432/mathledger_first_organism?sslmode=disable

# Redis
REDIS_URL=redis://:password@localhost:6380/0
```

### Pattern 2: Docker Network Connection

```bash
# PostgreSQL (from within Docker network)
DATABASE_URL=postgresql://first_organism_user:password@postgres:5432/mathledger_first_organism?sslmode=disable

# Redis (from within Docker network)
REDIS_URL=redis://:password@redis:6379/0
```

**Note:** When connecting from within Docker, use service names (`postgres`, `redis`) as hostnames and the container's internal port (6379 for Redis, not 6380).

---

## Security Requirements

All First Organism connection strings must meet security requirements enforced by `backend/security/first_organism_enforcer.py`:

1. **Password Strength:**
   - PostgreSQL: Minimum 12 characters
   - Redis: Minimum 12 characters
   - No banned passwords (e.g., `postgres`, `mlpass`, `password`, `redis`, etc.)

2. **SSL Configuration:**
   - Local Docker: `sslmode=disable` (required)
   - Remote databases: `sslmode=require` (enforced)

3. **Host Binding:**
   - First Organism containers bind to `127.0.0.1` only (not `0.0.0.0`) to prevent external access

For complete security requirements, see `docs/FIRST_ORGANISM_ENV.md`.

---

## Troubleshooting

### Connection Refused on Port 5432

**Problem:** `connection refused` when connecting to PostgreSQL on port 5432.

**Solutions:**
1. Verify First Organism infrastructure is running:
   ```powershell
   docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism ps
   ```
2. Check if another Postgres instance is using port 5432:
   ```powershell
   netstat -an | findstr :5432
   ```
3. Ensure connection string uses `localhost:5432` (not `127.0.0.1:5433`)

### Connection Refused on Port 6380

**Problem:** `connection refused` when connecting to Redis on port 6380.

**Solutions:**
1. Verify First Organism infrastructure is running (see above)
2. Check if port 6380 is in use:
   ```powershell
   netstat -an | findstr :6380
   ```
3. Ensure connection string uses port **6380** (not 6379) when connecting from host
4. Verify password is correct in `REDIS_URL`

### SSL Negotiation Errors

**Problem:** `could not send SSL negotiation packet` or similar SSL errors.

**Solutions:**
1. For local Docker Postgres, ensure `?sslmode=disable` is present in `DATABASE_URL`
2. For remote Postgres, ensure `?sslmode=require` is present
3. Test connectivity:
   ```bash
   uv run python scripts/test_db_connection.py
   ```

---

## References

- **Environment Setup:** `docs/FIRST_ORGANISM_ENV.md` - Complete First Organism environment configuration
- **Docker Compose:** `ops/first_organism/docker-compose.yml` - Infrastructure definition
- **Environment Template:** `ops/first_organism/first_organism.env.template` - Template with connection string examples
- **Security Enforcer:** `backend/security/first_organism_enforcer.py` - Security validation logic

---

**Last Updated:** 2025-01-XX  
**Maintainer:** GEMINI O — SPARK Observer + Connection String Consultant

