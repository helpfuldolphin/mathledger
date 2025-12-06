# FIRST ORGANISM SECURITY GUIDE

This document codifies the **zero-trust** posture for the First Organism integration run. Every dependency, service, and test must boot with **explicit secrets**, **authenticated services**, and **deterministic guards**. No defaults, no open Redis, no leaked passwords—only the environment variables listed below are trusted.

---

## Quick Reference

```sh
# 1. Copy the template (never commit .env.first_organism)
cp ops/first_organism/first_organism.env.template .env.first_organism

# 2. Generate strong secrets (examples)
openssl rand -base64 24    # POSTGRES_PASSWORD, REDIS_PASSWORD
openssl rand -hex 24       # LEDGER_API_KEY

# 3. Edit .env.first_organism and fill in all <PLACEHOLDER> values

# 4. Validate configuration before starting
make first-organism-validate

# 5. Spin up secure Postgres + Redis
make first-organism-up

# 6. Load env and run the closed-loop test
source .env.first_organism   # or use dotenv-cli / direnv
uv run pytest -m first_organism

# 7. Tear down
make first-organism-down
```

---

## Strict Mode

Set `FIRST_ORGANISM_STRICT=1` to enable strict security enforcement:

```sh
export FIRST_ORGANISM_STRICT=1
```

When strict mode is enabled:

- **No fallback defaults**: Any component that would fall back to `mlpass` or open Redis will raise an error instead.
- **Credential strength validation**: Passwords embedded in `DATABASE_URL` and `REDIS_URL` are validated for minimum length (12 chars) and checked against a blocklist of weak passwords.
- **CORS wildcard rejection**: `CORS_ALLOWED_ORIGINS=*` will be rejected.

Strict mode is automatically enabled by the First Organism test harness.

---

## Overview

The First Organism closed-loop test harness exercises the real UI → Curriculum → Derivation → Lean → Ledger → Dual Root → RFL chain. To keep that loop defensible:

- **No default secrets**: `mlpass`, `redis://localhost:6379/0`, or empty `LEDGER_API_KEY` are forbidden. The runtime enforcer (`backend/security/runtime_env.py`) will crash fast if any required variable is missing.
- **ABSTAIN semantics**: When the integration test cannot connect due to missing configuration it emits `[ABSTAIN] First Organism requires secure configuration: …` and skips rather than failing. This keeps CI green while clearly indicating that the secure run was not attempted.
- **Credential templates**: Copy `ops/first_organism/first_organism.env.template` to `.env.first_organism` and fill it with real values hosted per your environment.
- **Secure local stack**: The `ops/first_organism/docker-compose.yml` recipe brings up Postgres and Redis bound to `127.0.0.1`, with SCRAM-SHA-256 auth, password-protected Redis, resource limits, and security_opt hardening.
- **Rate limiting & request caps**: FastAPI and the worker read `MAX_REQUEST_BODY_BYTES`, `RATE_LIMIT_REQUESTS_PER_MINUTE`, and `RATE_LIMIT_WINDOW_SECONDS` from the same `.env.first_organism`, preserving defense-in-depth.

---

## Environment Variables

| Variable | Purpose | Notes |
| --- | --- | --- |
| `DATABASE_URL` | Primary SQLAlchemy/psycopg DSN | Must include username, password, host, port, and `sslmode=require` for production. Used by the API, worker, ledger ingestors, and test fixtures. |
| `REDIS_URL` | Redis connection string | Should point to password-protected (ideally `rediss://`) Redis instance spun up via the compose service. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Bound to the Dockerized Postgres service | Tied to `DATABASE_URL`. |
| `REDIS_PASSWORD` | Redis AUTH token | Used by the worker and `redis-cli` health checks. Also referenced by `REDIS_URL`. |
| `LEDGER_API_KEY` | API key for the attestation endpoints | Sent with requests to `/attestation/ui-event` and any dashboard guardrails. |
| `CORS_ALLOWED_ORIGINS` | Allowed origins for the FastAPI app | Must be explicit (no wildcards). `get_allowed_origins()` splits and enforces this value. |
| `MAX_REQUEST_BODY_BYTES`, `RATE_LIMIT_REQUESTS_PER_MINUTE`, `RATE_LIMIT_WINDOW_SECONDS` | FastAPI middleware knobs | Protects the organism from oversized payloads or abuse during tests. |
| `FIRST_ORGANISM_STRICT` | Enable strict security mode | Set to `1` to disable all fallback behavior and enforce credential strength. |

Any missing variable will produce the `[ABSTAIN] First Organism requires secure configuration: ...` message and skip the test run.

---

## Validation Tool

Before starting the stack, validate your configuration:

```sh
# Validate .env.first_organism
make first-organism-validate

# Or run directly
python tools/validate_first_organism_env.py
```

The validator checks:
- All required variables are set and non-empty
- Passwords meet minimum length requirements (12+ characters)
- No weak/default passwords (`mlpass`, `postgres`, `password`, etc.)
- CORS origins do not contain wildcards
- URL formats are valid

---

## Getting Started

### 1. Copy the template

```sh
cp ops/first_organism/first_organism.env.template .env.first_organism
```

### 2. Generate and fill credentials

The template includes placeholders like `<GENERATE_32_CHAR_PASSWORD_HERE>`. Replace each with strong random values:

```sh
# PostgreSQL password (32+ chars recommended)
openssl rand -base64 24

# Redis password (24+ chars recommended)
openssl rand -base64 24

# Ledger API key (48-char hex)
openssl rand -hex 24
```

### 3. Bring up the secure stack

```sh
make first-organism-up
```

This target runs:

```sh
docker compose -f ops/first_organism/docker-compose.yml --env-file .env.first_organism up -d
```

Wait for health checks to pass (Postgres and Redis both expose `/health` via Docker's internal checks).

### 4. Load the environment and run the test

```sh
# Option A: source directly
source .env.first_organism

# Option B: use dotenv-cli
dotenv -e .env.first_organism -- uv run pytest -m first_organism
```

The test will:

1. Validate all required env vars (or skip with `[ABSTAIN]`).
2. Connect to Postgres and Redis.
3. Execute the UI → Curriculum → Derivation → Lean → Ledger → Dual Root → RFL chain.
4. Assert deterministic outcomes.

### 5. Tear down

```sh
make first-organism-down
```

This removes containers and volumes created by the compose file.

---

## Security Alignment

- Matches the whitepaper's demand for **dual attestation** and **secret hygiene**: secrets are externalized, attestation data is derived from signed UI events, and the dual roots cannot be computed without fully bootstrapped connections.
- The `.env.first_organism` file lives outside version control (per `.gitignore`), so secrets are never committed.
- Docker services only bind to `127.0.0.1`, so the stack is not exposed to external networks.
- Rate-limit and request-size controls prevent accidental DoS during heavy test loops.
- The First Organism enforcer rejects weak passwords (`mlpass`, `postgres`, `password`, etc.) and short API keys.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[ABSTAIN] … DATABASE_URL is not set` | Missing or empty env var | Ensure `.env.first_organism` is sourced before running pytest. |
| `Redis not reachable` | Wrong port or password | Confirm `REDIS_URL` uses port `6380` (the compose maps `6380→6379` to avoid dev conflicts) and includes the password. See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format. |
| `SCRAM authentication failed` | Password mismatch | Regenerate `POSTGRES_PASSWORD` and update both the compose env and `DATABASE_URL`. |
| Test passes but no blocks sealed | Lean not available | Set `LEAN_MODE=mock` to use mock abstention proofs during local runs. |

---

Follow this guide whenever you're preparing a Wave 1 organism run. Any deviation should first be reviewed by Cursor N before proceeding.

