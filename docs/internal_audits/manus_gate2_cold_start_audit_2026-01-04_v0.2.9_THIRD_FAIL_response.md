# Response: Gate 2 Cold-Start Epistemic Dunk — v0.2.9 (THIRD_FAIL)

**Target Audit:** manus_gate2_cold_start_audit_2026-01-04_v0.2.9_THIRD_FAIL.md

---

## Status

**FIXED**

---

## Root Cause

Stale client cache caused `/versions/` page to display v0.2.8 as CURRENT instead of v0.2.9. The deployment had completed but the auditor's browser served cached content from a previous request.

---

## Evidence of Fix

### 1. Canonical Registry Now Shows v0.2.9

**URL:** `https://mathledger.ai/versions/status.json`

```json
{
  "current_version": "v0.2.9",
  "current_tag": "v0.2.9-abstention-terminal",
  "versions": ["v0", "v0.2.0", "v0.2.1", "v0.2.2", "v0.2.3", "v0.2.4", "v0.2.5", "v0.2.6", "v0.2.7", "v0.2.8", "v0.2.9"],
  "superseded": ["v0", "v0.2.0", "v0.2.1", "v0.2.2", "v0.2.3", "v0.2.4", "v0.2.5", "v0.2.6", "v0.2.7", "v0.2.8"]
}
```

### 2. Canonical u_t Value in examples.json

**URL:** `https://mathledger.ai/v0.2.9/evidence-pack/examples.json`

```
valid_boundary_demo.pack.u_t: 0d1b61da395bb759b4558e1329e9ea561450e66d66421f88b540f7e828c0cd2d
```

### 3. Verifier Contains Domain-Separated Merkle Computation

**URL:** `https://mathledger.ai/v0.2.9/evidence-pack/verify/`

Present in verifier JavaScript:
- `DOMAIN_UI_LEAF` constant
- `DOMAIN_REASONING_LEAF` constant
- `merkleRoot()` function with domain separation
- `computeUiRoot()` function
- `computeReasoningRoot()` function

### 4. Manus Re-Run Result (Cache Bypassed)

**Result:** PASS

```
PHASE 1: Identify CURRENT Version → PASS (v0.2.9 listed as CURRENT)
PHASE 2: Navigate to FOR_AUDITORS → PASS
PHASE 3: Self-Test Verification → PASS
  - SELF-TEST PASSED (3 vectors)
  - valid_boundary_demo: Expected PASS, Actual PASS → PASS
  - tampered_ht_mismatch: Expected FAIL, Actual FAIL → PASS
  - tampered_rt_mismatch: Expected FAIL, Actual FAIL → PASS
PHASE 4: Evidence Pack Flow → PASS
PHASE 5: Demo Coherence → PASS

GATE 2: PASS
```

---

## Resolution Version

**v0.2.9** (tag: v0.2.9-abstention-terminal)

---
