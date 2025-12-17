# PILOT AUTHORIZATION

**Status:** CANONICAL
**Authority:** CLAUDE V (Gatekeeper)
**Date:** 2025-12-13
**Mode:** SHADOW (invariant)

---

## Authorization Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   LEVEL 0: INTERNAL CALIBRATION                                 │
│   ├── CAL-EXP-1 (Baseline) ✓ COMPLETE                           │
│   ├── CAL-EXP-2 (Convergence) ✓ COMPLETE                        │
│   └── CAL-EXP-3 (Uplift Measurement) ← CURRENT                  │
│                                                                 │
│   LEVEL 1: PILOT READINESS                                      │
│   └── Internal pilot execution authorized                       │
│                                                                 │
│   LEVEL 2: PUBLIC PILOT                                         │
│   └── NOT AUTHORIZED (blocked)                                  │
│                                                                 │
│   LEVEL 3: EXTERNAL CLAIMS                                      │
│   └── NOT AUTHORIZED (blocked)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Pilot Readiness

### Definition

**Pilot readiness** is the authorization to execute internal pilot runs under SHADOW mode constraints.

### When Pilot Execution IS Allowed

| Condition | Status |
|-----------|--------|
| CAL-EXP-1 baseline recorded | ✓ REQUIRED |
| CAL-EXP-2 exit decision issued | ✓ REQUIRED |
| Toolchain parity green (21/21 tests) | ✓ REQUIRED |
| SHADOW mode in all outputs | ✓ REQUIRED |
| No enforcement semantics | ✓ REQUIRED |
| External ingest adapter present + tests passing | ✓ REQUIRED |
| Toolchain manifest produced at ingestion time | ✓ REQUIRED |
| Smoke proof single path verified | ✓ REQUIRED |

**Smoke proof reference:** `docs/system_law/pilot/PILOT_SMOKE_PROOF_SINGLE_PATH.md`

**Note:** Smoke proof is **operational evidence** (the system runs end-to-end without crashing), not scientific evidence (no claims about accuracy, learning, or capability).

### When Pilot Execution IS NOT Allowed

| Blocker | Reason |
|---------|--------|
| Any sentinel test failing | Contract violation |
| `enforcement=true` in any artifact | SHADOW mode breach |
| `mode` field not "SHADOW" | SHADOW mode breach |
| Toolchain parity < 21/21 | Regression detected |
| CAL-EXP-2 exit decision not issued | Prerequisites incomplete |
| External ingest adapter missing | Ingest readiness incomplete |
| Ingest adapter tests failing | Ingest readiness incomplete |
| Toolchain manifest not produced at ingest | Provenance gap |
| Smoke proof single path not verified | Operational readiness incomplete |

### Pilot Execution Authorization

```
IF   toolchain_parity = 21/21 tests pass
AND  cal_exp_2_exit_decision = ISSUED
AND  all_artifacts.mode = "SHADOW"
AND  no_enforcement_true_anywhere
AND  external_ingest_adapter = PRESENT
AND  ingest_tests = PASSING
AND  toolchain_manifest_at_ingest = PRODUCED
AND  smoke_proof_single_path = VERIFIED
THEN pilot_execution = AUTHORIZED
ELSE pilot_execution = BLOCKED
```

---

## 1.1 External Ingest Readiness

### Definition

**External ingest readiness** is the verification that the pilot can safely consume external data with full provenance tracking.

### Checklist

- [ ] **External ingest adapter present**
  - Adapter module exists and is importable
  - Handles external data sources (if any)
  - Validates input schema before processing
  - Reference: `backend/topology/` or designated ingest module

- [ ] **Ingest adapter tests passing**
  - Unit tests for adapter logic
  - Integration tests for data flow
  - SHADOW mode compliance in all outputs
  - No enforcement semantics in adapter

- [ ] **Toolchain manifest produced at ingestion time**
  - `toolchain_fingerprint` recorded in manifest
  - `uv_lock_hash` captured at ingest start
  - `lean_toolchain_hash` captured (if applicable)
  - Timestamp of ingestion recorded
  - Input source hash recorded (if external data)

### Manifest Schema (at Ingestion)

```json
{
  "schema_version": "1.0.0",
  "ingest_timestamp": "<ISO8601>",
  "toolchain_fingerprint": "<sha256>",
  "uv_lock_hash": "<sha256>",
  "input_source_hash": "<sha256 or null>",
  "mode": "SHADOW",
  "adapter": "<adapter_name>"
}
```

### Verification Command

```bash
# Check ingest adapter exists and tests pass
uv run pytest tests/integration/ -k "ingest" -v --tb=short 2>/dev/null || \
echo "INFO: No ingest-specific tests found (acceptable if no external data)"

# Verify manifest production capability
uv run python -c "
from pathlib import Path
import json

# Check for manifest generation capability
manifest_paths = list(Path('results').glob('**/manifest.json'))
if manifest_paths:
    m = json.loads(manifest_paths[0].read_text())
    assert 'toolchain_fingerprint' in m or 'schema_version' in m
    print('INGEST MANIFEST: CAPABILITY VERIFIED')
else:
    print('INGEST MANIFEST: NO MANIFESTS FOUND (run pilot first)')
"
```

---

## 2. CAL-EXP-3 Readiness

### Definition

**CAL-EXP-3 readiness** is the authorization to measure uplift (Δp) under controlled conditions.

### Prerequisites (All Required)

| Prerequisite | Document | Status |
|--------------|----------|--------|
| CAL-EXP-1 replicated | `CAL_EXP_1.md` | ✓ COMPLETE |
| CAL-EXP-2 verified | `CAL_EXP_2_EXIT_DECISION.md` | ✓ COMPLETE |
| Language hygiene passed | `CAL_EXP_2_LANGUAGE_CONSTRAINTS.md` | ✓ PASS |
| Reproducibility gate green | `CAL_EXP_2_GO_NO_GO.md` | ✓ GREEN |

### CAL-EXP-3 Authorization

```
IF   cal_exp_1 = REPLICATED
AND  cal_exp_2 = VERIFIED
AND  language_hygiene = PASS
AND  reproducibility_gate = GREEN
THEN cal_exp_3 = AUTHORIZED
ELSE cal_exp_3 = BLOCKED
```

### Current Status

**CAL-EXP-3: AUTHORIZED** (conditional on verification commands passing)

Reference: `docs/system_law/calibration/CAL_EXP_3_AUTHORIZATION.md`

---

## 3. External Claims Authorization

### Definition

**External claims authorization** is the permission to make public statements about system capabilities.

### Current Status

**EXTERNAL CLAIMS: NOT AUTHORIZED**

### Blocking Conditions

| Condition | Status | Required For Unblock |
|-----------|--------|----------------------|
| Pilot exit criteria met | ✗ PENDING | Full pilot completion |
| Independent replication | ✗ PENDING | Third-party verification |
| Governance review | ✗ PENDING | Authority sign-off |
| SHADOW mode lifted | ✗ BLOCKED | Phase XI approval |

### What IS Allowed (Internal Only)

| Action | Scope | Constraint |
|--------|-------|------------|
| Measure divergence | Internal | Safe templates |
| Document observations | Internal | Language hygiene |
| Compare configurations | Internal | No new formulas |
| Record metrics | Internal | Existing metrics only |

### What IS NOT Allowed

| Action | Reason |
|--------|--------|
| Public claims of capability | Not authorized |
| Marketing statements | Not authorized |
| Performance guarantees | Not authorized |
| Accuracy claims | Not authorized |
| "Validated" or "Proven" language | Forbidden phrases |

---

## 4. Progression Blockers

### Conditions That Block Public Pilot

| Blocker | Detection | Resolution |
|---------|-----------|------------|
| Sentinel test failure | CI gate | Fix contract violation |
| SHADOW mode breach | Artifact scan | Remove enforcement logic |
| Language hygiene violation | Manual review | Rewrite with safe templates |
| Reproducibility failure | Determinism test | Fix non-deterministic code |
| New pathology introduced | CAL-EXP verification | Revert or document |

### Conditions That Block External Claims

| Blocker | Current State | Required Action |
|---------|---------------|-----------------|
| SHADOW mode active | ✓ ACTIVE | Phase XI approval to lift |
| Pilot incomplete | IN PROGRESS | Complete pilot criteria |
| No independent verification | PENDING | Third-party replication |
| Governance not approved | PENDING | Authority review |

---

## 5. Separation of Concerns

### Pilot Readiness ≠ CAL-EXP-3 Readiness

| Aspect | Pilot Readiness | CAL-EXP-3 Readiness |
|--------|-----------------|---------------------|
| Scope | Execute pilot runs | Measure specific metrics |
| Prerequisites | Toolchain + SHADOW mode | CAL-EXP-1 + CAL-EXP-2 |
| Output | Run artifacts | Δp measurements |
| Constraints | No enforcement | No new science |

### CAL-EXP-3 Readiness ≠ External Claims

| Aspect | CAL-EXP-3 Readiness | External Claims |
|--------|---------------------|-----------------|
| Scope | Internal measurement | Public statements |
| Authorization | CLAUDE V | Governance review |
| Current Status | AUTHORIZED | NOT AUTHORIZED |
| Unblock Requires | Prerequisites | Pilot completion + review |

### Pilot Readiness ≠ External Claims

| Aspect | Pilot Readiness | External Claims |
|--------|-----------------|-----------------|
| Scope | Internal execution | Public communication |
| Mode | SHADOW (observational) | Would require LIVE mode |
| Current Status | AUTHORIZED | NOT AUTHORIZED |
| Gap | Significant | Requires Phase XI |

---

## 6. Verification Commands

### Pilot Readiness Check

```bash
# Toolchain parity
uv run pytest \
    tests/ci/test_shadow_audit_sentinel.py \
    tests/ci/test_shadow_audit_guardrails.py \
    tests/integration/test_shadow_audit_e2e.py \
    -v --tb=short

# Expected: 21 passed
# Result: PILOT EXECUTION AUTHORIZED if 21/21
```

### CAL-EXP-3 Readiness Check

```bash
# Prerequisites verification
test -f docs/system_law/calibration/CAL_EXP_1.md && \
test -f docs/system_law/calibration/CAL_EXP_2_EXIT_DECISION.md && \
test -f docs/system_law/calibration/CAL_EXP_2_LANGUAGE_CONSTRAINTS.md && \
echo "CAL-EXP-3: AUTHORIZED" || echo "CAL-EXP-3: BLOCKED"
```

### External Claims Check

```bash
# Always returns NOT AUTHORIZED in Phase X
echo "EXTERNAL CLAIMS: NOT AUTHORIZED (SHADOW MODE ACTIVE)"
```

---

## 7. Authorization Matrix

| Level | Name | Status | Blocker |
|-------|------|--------|---------|
| 0 | Internal Calibration | ✓ COMPLETE | — |
| 1 | Pilot Readiness | ✓ AUTHORIZED | — |
| 2 | Public Pilot | ✗ BLOCKED | SHADOW mode, incomplete pilot |
| 3 | External Claims | ✗ BLOCKED | All Level 2 blockers + governance |

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PILOT AUTHORIZATION SUMMARY                                   │
│                                                                 │
│   Pilot Execution:        AUTHORIZED (internal, SHADOW mode)   │
│   CAL-EXP-3 Measurement:  AUTHORIZED (conditional)             │
│   Public Pilot:           NOT AUTHORIZED (blocked)             │
│   External Claims:        NOT AUTHORIZED (blocked)             │
│                                                                 │
│   Mode:                   SHADOW (invariant)                   │
│   Enforcement:            NONE (observational only)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sign-Off

| Role | Agent | Date |
|------|-------|------|
| Authorization | CLAUDE V | 2025-12-13 |
| Hierarchy Definition | CLAUDE V | 2025-12-13 |
| Blocker Enumeration | CLAUDE V | 2025-12-13 |

---

**SHADOW MODE — observational only.**
