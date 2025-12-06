# First Organism Connection Strings — Minimal Reference

**Canonical formats for First Organism (FO) database connections.**

## Quick Reference

**PostgreSQL:**
```
postgresql://first_organism_user:<password>@localhost:5432/mathledger_first_organism?sslmode=disable
```

**Redis:**
```
redis://:<password>@localhost:6380/0
```

## Port Rules

| Service | Port | Rule |
|---------|------|------|
| PostgreSQL | **5432** | Always 5432 for FO (from host) |
| Redis | **6380** | Always 6380 for FO (from host). Maps to container's 6379. |

**Decision:** FO uses 5432 (Postgres) and 6380 (Redis). Port 5433 and 6379 are for development/other environments, not FO.

## Docker Network (Internal)

When connecting from within Docker network:
- PostgreSQL: `postgres:5432` (service name, container port)
- Redis: `redis:6379` (service name, container port)

## Environment Variables

```bash
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}?sslmode=disable
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6380/0
```

## Security

- Passwords: Minimum 12 characters, not in banned list
- SSL: `sslmode=disable` for local Docker, `sslmode=require` for remote
- See `docs/FIRST_ORGANISM_ENV.md` for complete requirements

## Full Documentation

See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for:
- Component breakdowns
- Troubleshooting
- Docker compose port mapping details
- Common patterns

---

## Phase-I RFL Note

**RFL logs do not use DB/Redis connection strings.**

Connection string correctness affects only FO infrastructure (integration tests, API, worker), not RFL evidence generation. Phase-I RFL experiments (`experiments/run_fo_cycles.py`, RFL runner) are hermetic and file-based; they operate independently of database connections for evidence collection.

**Scope:**
- **Affected:** First Organism integration tests, API endpoints, worker jobs
- **Not Affected:** RFL evidence logs (hermetic, file-based), RFL experiment results, RFL bootstrap statistics

**Phase-I Context:** All Phase-I RFL runs are hermetic negative-control runs with 100% abstention (lean-disabled mode). They validate execution infrastructure and attestation only; they do not demonstrate uplift. Connection strings are irrelevant to RFL evidence generation in Phase I.

---

**Last Updated:** 2025-01-XX  
**Maintainer:** GEMINI O — SPARK Observer + Connection String Consultant

