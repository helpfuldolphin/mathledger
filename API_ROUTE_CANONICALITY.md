# API Route Canonicality

**Status:** Phase I — Verified
**Last Updated:** 2025-11-30
**Scope:** Evidence Pack v1

---

## Summary

The MathLedger API is served from a single canonical entry point:

```
interface/api/app.py
```

That's it.

---

## Canonical Route Locations

| Route Pattern | Handler Location | Status |
|---------------|------------------|--------|
| `/health` | `interface/api/app.py` | Active |
| `/metrics` | `interface/api/app.py` | Active |
| `/blocks/latest` | `interface/api/app.py` | Active |
| `/statements` | `interface/api/app.py` | Active |
| `/heartbeat` | `interface/api/app.py` | Active |
| `/heartbeat.json` | `interface/api/app.py` | Active |
| `/ui/*` | `interface/api/app.py` | Active |
| `/ui/parents/{hash}.json` | `interface/api/routes/parents.py` | Active |
| `/ui/proofs/{hash}.json` | `interface/api/routes/parents.py` | Active |
| `/attestation/*` | `interface/api/app.py` | Active |

All routes are registered via `app.include_router()` in `interface/api/app.py`.

---

## Shim Modules (Do Not Use)

| Module | Purpose | Status |
|--------|---------|--------|
| `backend/orchestrator/parents_routes.py` | Re-exports `parents_router` from canonical location | **Deprecated shim** — exists for backwards compatibility only |

The shim contains no logic. It performs a single re-export:

```python
from interface.api.routes.parents import parents_router
```

**Do not import from `backend.orchestrator.parents_routes` in new code.**

---

## Evidence Pack v1 Statement

For the Evidence Pack and research paper, the API story is simple:

> The HTTP API is served from `interface.api.app`. Parent and proof lineage endpoints are implemented in `interface.api.routes.parents`. No additional API modules are required.

No complex routing topology. No multi-app federation. No shim dependencies in the critical path.

---

## Impact of RFL Logs: None

The Reviewer Feedback Loop (RFL) system generates logs and telemetry during derivation runs. These logs are:

- Written to disk (`results/fo_rfl/`, `results/fo_baseline/`)
- Stored in Redis counters (`ml:metrics:first_organism:*`)
- Recorded in the `proofs` table via standard ledger ingest

**RFL does not expose any HTTP endpoints.**

There is no `/rfl/*` route. There is no `/feedback/*` route. The RFL runner (`backend/rfl/runner.py`) is a batch process invoked by the First Organism harness, not an API service.

The `/metrics` endpoint reports RFL telemetry (run counts, latency, abstention rate) by reading Redis counters. This is read-only aggregation of data already collected — not an RFL-specific route.

| Component | Touches API? | Explanation |
|-----------|--------------|-------------|
| `backend/rfl/runner.py` | No | Batch execution, no HTTP |
| `backend/rfl/config.py` | No | Configuration dataclass |
| RFL logs on disk | No | Static files, not served |
| Redis RFL counters | No | Read by `/metrics`, not exposed directly |

**Reviewer-2 statement:** RFL is an offline batch subsystem. It has zero impact on API route structure or canonicality.

---

## Phase II: Internal Uplift Telemetry (Design Sketch Only)

> **WARNING: PHASE II — NOT YET IMPLEMENTED**
>
> The endpoints described in this section do not exist. No code has been written. This is a design sketch for potential future internal observability tooling. Do not reference these endpoints as implemented features.

### Motivation

As the derivation system scales, operators may want programmatic access to:

1. **Per-slice abstention curves** — How does abstention rate change as slice complexity increases?
2. **RFL vs baseline comparison** — Side-by-side telemetry for runs with and without the Reviewer Feedback Loop.

Currently, this data exists only in:
- Disk logs (`results/fo_rfl/`, `results/fo_baseline/`)
- Redis counters (aggregated, not per-slice)
- Manual analysis scripts

### Proposed Internal Endpoints

These would be **internal observability endpoints**, not part of the public API surface.

| Endpoint (Hypothetical) | Purpose | Data Source |
|-------------------------|---------|-------------|
| `GET /internal/telemetry/abstention-curve` | Return abstention rate by slice index | Redis or DB aggregation |
| `GET /internal/telemetry/rfl-comparison` | Return baseline vs RFL metrics for a run | Disk logs or DB join |

### Design Constraints

1. **Prefix isolation:** All internal endpoints under `/internal/*` — clearly separated from public routes.
2. **No schema guarantees:** Response shapes may change without notice; these are debugging tools.
3. **Auth required:** Same `X-API-Key` check, but additionally gated by an `INTERNAL_TELEMETRY_ENABLED=1` environment flag (disabled by default).
4. **No UI integration:** These endpoints are for CLI/script consumption, not dashboard binding.

### Example Response Shapes (Hypothetical)

```json
// GET /internal/telemetry/abstention-curve?run_id=fo-rfl-1000
{
  "run_id": "fo-rfl-1000",
  "slices": [
    {"index": 0, "atoms": 2, "depth": 2, "abstention_rate": 0.02},
    {"index": 1, "atoms": 3, "depth": 3, "abstention_rate": 0.05},
    {"index": 2, "atoms": 4, "depth": 4, "abstention_rate": 0.12}
  ]
}
```

```json
// GET /internal/telemetry/rfl-comparison?baseline_run=fo-baseline-1000&rfl_run=fo-rfl-1000
{
  "baseline": {"run_id": "fo-baseline-1000", "total_proofs": 1000, "abstention_rate": 0.15},
  "rfl": {"run_id": "fo-rfl-1000", "total_proofs": 1000, "abstention_rate": 0.08},
  "delta": {"abstention_rate": -0.07}
}
```

### What This Does NOT Change

- The public API surface remains `interface/api/app.py`
- No new routers are added to the canonical route table
- Existing `/metrics` endpoint is unchanged
- Phase I documentation ("RFL has no API impact") remains true

### Implementation Status

| Item | Status |
|------|--------|
| Endpoint implementation | **Not started** |
| Redis schema for per-slice data | **Not designed** |
| Environment flag gating | **Not implemented** |
| Internal router registration | **Not written** |

**This section exists solely as a Phase II design sketch for future consideration.**

---

## What This Document Does NOT Cover

The following are **Phase II — Not Yet Implemented**:

- API versioning schemes
- Rate limiting tuning beyond current defaults
- Authentication beyond `X-API-Key` header
- GraphQL or alternative API surfaces
- Distributed API gateway patterns

These are not part of Evidence Pack v1 and should not be referenced as implemented features.

---

## File Verification

Existence of canonical files can be verified:

```powershell
# Canonical entry point
Test-Path "interface/api/app.py"  # Expected: True

# Canonical parent routes
Test-Path "interface/api/routes/parents.py"  # Expected: True

# Deprecated shim (still present, not removed)
Test-Path "backend/orchestrator/parents_routes.py"  # Expected: True (shim exists)
```

---

## Reviewer-2 Checklist

| Claim | Evidence |
|-------|----------|
| "API served from single entry point" | `interface/api/app.py` exists, contains `app = FastAPI(...)` |
| "Parent routes in canonical location" | `interface/api/routes/parents.py` contains `@parents_router.get(...)` decorators |
| "Shim is pure re-export" | `backend/orchestrator/parents_routes.py` is 16 lines, contains only import and `__all__` |
| "No routing logic in shim" | Shim has zero route handlers, zero business logic |

---

## Cleanup Status

| Action | Status |
|--------|--------|
| Identify canonical implementation | Done |
| Document shim deprecation | Done |
| Produce removal diff | Done (see `MIGRATION_NOTES.md`) |
| Execute removal | **Not done** — deferred to cleanup sprint |
| Update manifests post-removal | **Not done** — blocked on removal |

The shim remains in place. No runtime behavior has changed.
