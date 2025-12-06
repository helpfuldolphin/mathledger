# Migration Notes: Parents/Proofs Route Consolidation

**Date:** 2025-11-29
**Scope:** API Schema Audit — Duplicate Route Handler Removal
**Reference:** MathLedger Whitepaper §6.2 (Parent Provenance API)

---

## Summary

This migration removes the deprecated `backend/orchestrator/parents_routes.py` shim module, consolidating all parent/proof route handling into the canonical `interface/api/routes/parents.py` module.

## Changes

### Files Removed

| File | Reason |
|------|--------|
| `backend/orchestrator/parents_routes.py` | Deprecated re-export shim; canonical implementation lives in `interface/api/routes/parents.py` |

### Files Modified

| File | Change |
|------|--------|
| `.coveragerc` | Removed omit entry for deleted file |
| `CLAUDE.md` | Updated orchestrator documentation to remove reference to `parents_routes.py` |
| `tools/vibe_check.py` | Updated TARGETS list to reference canonical path |

### Files Unchanged (No Action Required)

| File | Status |
|------|--------|
| `interface/api/app.py` | Already imports from canonical location (`interface.api.routes.parents`) at line 1174 |
| `interface/api/routes/parents.py` | Canonical implementation — no changes |
| `interface/api/schemas.py` | Schema definitions unchanged |

---

## Impact on Tests

### Test Files to Verify

No test files directly import `backend.orchestrator.parents_routes`. The following test files reference the orchestrator module generally and should be verified post-migration:

| Test File | Import Pattern | Action |
|-----------|----------------|--------|
| `tests/test_orchestrator_app_extra.py` | `from backend.orchestrator import app as app_module` | **No change needed** — imports `app`, not `parents_routes` |

### Recommended Test Commands

```bash
# Verify no import errors
pytest tests/test_orchestrator_app_extra.py -v

# Full test suite verification
pytest tests/ -v --tb=short

# Specific parent/proof endpoint tests (if they exist)
pytest -k "parent" -v
pytest -k "proof" -v
```

---

## Impact on Clients

### API Endpoints

All endpoints remain unchanged. The following routes are served from `interface/api/routes/parents.py`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ui/parents/{hash}.json` | GET | Returns parent statement summaries for a given hash |
| `/ui/proofs/{hash}.json` | GET | Returns proof summaries for a given statement hash |

### Client Migration

| Client Type | Required Action |
|-------------|-----------------|
| **HTTP Clients** | None — endpoint URLs unchanged |
| **Python imports (legacy)** | Update `from backend.orchestrator.parents_routes import ...` → `from interface.api.routes.parents import ...` |

### Legacy Import Compatibility

If any external code still imports from `backend.orchestrator.parents_routes`, it will fail with `ModuleNotFoundError`. Search for legacy imports:

```bash
# PowerShell
Get-ChildItem -Recurse -Filter "*.py" | Select-String "backend\.orchestrator\.parents_routes"

# Bash
grep -r "backend\.orchestrator\.parents_routes" --include="*.py"
```

---

## Documentation Updates Required

The following documentation files reference `backend/orchestrator/parents_routes.py` and may need updates:

| File | Line Reference | Suggested Action |
|------|---------------|------------------|
| `basis_promotion_report.md` | Line 47-48 | Update import graph documentation |
| `VERSION_LINEAGE_LEDGER.md` | Line 264 | Update file tree |
| `ops/spanning_set_manifest.json` | Line 99183 | Regenerate manifest |
| `spanning_set_manifest.json` | Line 2171 | Regenerate manifest |

---

## Rollback Procedure

If issues arise, restore the shim by creating `backend/orchestrator/parents_routes.py`:

```python
"""
DEPRECATED: kept only for legacy callers; will be removed after VCP 2.2 Wave 1.

This module re-exports from the canonical interface.api.routes.parents namespace.
New code should import directly from interface.api.routes.parents instead.

Reference: MathLedger Whitepaper §6.2 (Parent Provenance API).
"""

# Re-export from canonical namespace
from interface.api.routes.parents import (
    parents_router,
)

__all__ = ["parents_router"]
```

---

## Verification Checklist

- [ ] `backend/orchestrator/parents_routes.py` deleted
- [ ] `.coveragerc` updated
- [ ] `CLAUDE.md` updated
- [ ] `tools/vibe_check.py` updated
- [ ] All tests pass
- [ ] API endpoints `/ui/parents/{hash}.json` and `/ui/proofs/{hash}.json` respond correctly
- [ ] No `ModuleNotFoundError` in application startup logs

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| API Schema Auditor | CLAUDE H | 2025-11-29 |
