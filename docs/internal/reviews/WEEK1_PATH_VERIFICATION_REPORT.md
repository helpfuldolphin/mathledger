# Week 1 Deliverables Path Verification Report

**Date**: 2025-12-17
**Source**: `ACQUISITION_EXECUTION_PLAN_90DAY.md` Phase 1 (Days 1-30)
**Reviewer**: Governance Path Auditor

---

## Executive Summary

| Deliverable | Path Conflict | Contract Conflict | Recommendation |
|-------------|---------------|-------------------|----------------|
| P0.1 USLA Shadow Gate | **YES** | **YES** | RENAME REQUIRED |
| P0.2 Manifest Signing | NO | NO | PROCEED |
| P0.3 L5 Environment Diversity | NO | **YES** | ALTERNATIVE REQUIRED |

---

## 1. P0.1 — USLA Shadow Gate

### 1.1 Proposed Paths

| Component | Proposed Path |
|-----------|---------------|
| Runtime guard | `backend/health/usla_shadow_gate.py` |
| CI workflow | `.github/workflows/usla-shadow-gate.yml` |
| Tests | `tests/health/test_usla_shadow_gate.py` |

### 1.2 Existing Conflicts

| Existing Path | Purpose | Conflict Type |
|---------------|---------|---------------|
| `backend/topology/usla_shadow.py` | **USLAShadowLogger** — Structured shadow logging for USLA simulator | SEMANTIC COLLISION |
| `.github/workflows/usla-shadow-gate.yml` | **USLA Health Tile Serialization** — Validates USLA observability | NAME COLLISION |
| `tests/topology/test_usla_shadow_integration.py` | Tests for USLAShadowLogger | SEMANTIC COLLISION |

### 1.3 Semantic Confusion Risk

The existing `usla_shadow.py` is a **shadow logger** for the USLA simulator.
The proposed `usla_shadow_gate.py` is a **release gate** preventing SHADOW artifacts from blocking production.

Using similar names would cause:
- Confusion between "shadow logging" and "shadow mode gating"
- Import namespace collisions
- Documentation ambiguity

### 1.4 Contract Conflict

| Contract | Violation |
|----------|-----------|
| `SHADOW_MODE_CONTRACT.md` | Existing workflow says "SHADOW MODE: Enable for CI validation only" (SHADOW-OBSERVE). Proposed gate would be SHADOW-GATED (blocks operations). Name collision implies wrong mode. |

### 1.5 Rename/Relocate Map

| Original Proposed | Recommended Path | Rationale |
|-------------------|------------------|-----------|
| `backend/health/usla_shadow_gate.py` | `backend/health/shadow_release_gate.py` | Avoids "usla_shadow" collision; clarifies purpose |
| `.github/workflows/usla-shadow-gate.yml` | `.github/workflows/shadow-release-gate.yml` | Distinct from existing USLA serialization workflow |
| `tests/health/test_usla_shadow_gate.py` | `tests/health/test_shadow_release_gate.py` | Matches renamed module |

### 1.6 Minimal Safe Diff Plan

```diff
# NEW FILES (no modification to existing)
+ backend/health/shadow_release_gate.py
+ tests/health/test_shadow_release_gate.py
+ .github/workflows/shadow-release-gate.yml

# EXISTING FILES - NO CHANGES
  backend/topology/usla_shadow.py          # UNTOUCHED
  .github/workflows/usla-shadow-gate.yml   # UNTOUCHED
  tests/topology/test_usla_shadow_integration.py  # UNTOUCHED
```

---

## 2. P0.2 — Manifest Signing

### 2.1 Proposed Paths

| Component | Proposed Path |
|-----------|---------------|
| Signing utility | `scripts/sign_manifest.py` |
| Verification utility | `scripts/verify_manifest_signature.py` |
| Documentation | `docs/system_law/MANIFEST_SIGNING.md` |
| Updated generator | `scripts/generate_and_verify_evidence_pack.py` |

### 2.2 Existing Files Check

| Path | Exists | Status |
|------|--------|--------|
| `scripts/sign_manifest.py` | NO | Clear to create |
| `scripts/verify_manifest_signature.py` | NO | Clear to create |
| `docs/system_law/MANIFEST_SIGNING.md` | NO | Clear to create |
| `scripts/generate_and_verify_evidence_pack.py` | **YES** | Modification required |

### 2.3 Contract Conflict

| Contract | Check |
|----------|-------|
| `PILOT_CONTRACT_POSTURE.md` | `scripts/generate_and_verify_evidence_pack.py` **NOT** in frozen surfaces list |
| `SHADOW_MODE_CONTRACT.md` | N/A — manifest signing is infrastructure, not SHADOW semantics |
| `CAL-EXP-*` | N/A — does not modify experiment artifacts or verifiers |

**Result**: NO CONTRACT CONFLICT

### 2.4 Minimal Safe Diff Plan

```diff
# NEW FILES
+ scripts/sign_manifest.py
+ scripts/verify_manifest_signature.py
+ docs/system_law/MANIFEST_SIGNING.md

# MODIFIED FILES
~ scripts/generate_and_verify_evidence_pack.py
  # Add: call sign_manifest() after manifest generation
  # Add: output manifest.json.sig alongside manifest.json
```

---

## 3. P0.3 — L5 Environment Diversity

### 3.1 Proposed Changes

| Component | Proposed Change |
|-----------|-----------------|
| Run metadata | Add `environment_fingerprint` field to `RUN_METADATA.json` |
| Verifier | Update `verify_cal_exp_3_run.py` to check environment diversity |
| Claim logic | Auto-downgrade L5 → L5-lite if fingerprints identical |

### 3.2 Contract Conflict — BLOCKING

| Contract | Violation |
|----------|-----------|
| `CAL_EXP_3_INDEX.md` | Status: **CLOSED — CANONICAL (v1.0)**. Section "Change Control" states: "Modifications to CAL-EXP-3 artifacts require: 1. Update to this index 2. Explicit rationale 3. STRATCOM approval for semantic changes" |

**CAL-EXP-3 is FROZEN.** Modifying `verify_cal_exp_3_run.py` requires STRATCOM authorization.

### 3.3 Proposed Path Assessment

| Path | Frozen | Can Modify |
|------|--------|------------|
| `scripts/verify_cal_exp_3_run.py` | **YES** (CANONICAL v1.0) | NO without STRATCOM |
| `RUN_METADATA.json` schema | **YES** (CAL-EXP-3 contract) | NO without STRATCOM |

### 3.4 Alternative: CAL-EXP-4 Path

Since `verify_cal_exp_4_run.py` is NEW (not frozen), environment diversity could be added there:

| Approach | Frozen Surface | Recommendation |
|----------|----------------|----------------|
| Modify CAL-EXP-3 verifier | YES | **REJECTED** |
| Add to CAL-EXP-4 verifier | NO | **PREFERRED** |
| Create standalone diversity checker | NO | ACCEPTABLE |

### 3.5 Minimal Safe Diff Plan (Alternative)

```diff
# FROZEN — DO NOT MODIFY
  scripts/verify_cal_exp_3_run.py  # CANONICAL v1.0

# NEW FILES (CAL-EXP-4 path)
+ scripts/check_environment_diversity.py  # Standalone checker
~ scripts/verify_cal_exp_4_run.py         # Add environment diversity check

# OR: Evidence pack integration
~ scripts/generate_and_verify_evidence_pack.py
  # Add: environment_fingerprint computation
  # Add: L5 diversity validation at pack level
```

---

## 4. Summary: Rename/Relocate Map

| Plan Path | Actual Path | Reason |
|-----------|-------------|--------|
| `backend/health/usla_shadow_gate.py` | `backend/health/shadow_release_gate.py` | Avoid collision with `usla_shadow.py` |
| `.github/workflows/usla-shadow-gate.yml` | `.github/workflows/shadow-release-gate.yml` | Avoid collision with existing workflow |
| `tests/health/test_usla_shadow_gate.py` | `tests/health/test_shadow_release_gate.py` | Match renamed module |
| `verify_cal_exp_3_run.py` modification | **BLOCKED** | CAL-EXP-3 is CANONICAL — use CAL-EXP-4 or standalone |

---

## 5. Contract Conflicts Summary

### 5.1 SHADOW_MODE_CONTRACT.md

| Deliverable | Conflict | Resolution |
|-------------|----------|------------|
| P0.1 Shadow Gate | Name collision implies wrong mode | Rename to `shadow_release_gate` |
| P0.2 Manifest Signing | None | — |
| P0.3 Environment Diversity | None | — |

### 5.2 PILOT_CONTRACT_POSTURE.md

| Deliverable | Conflict | Resolution |
|-------------|----------|------------|
| P0.1 Shadow Gate | None (not in frozen list) | — |
| P0.2 Manifest Signing | None | — |
| P0.3 Environment Diversity | None (CAL-EXP-3 harnesses frozen, but verifier not explicitly listed) | See CAL-EXP-3 INDEX |

### 5.3 CAL-EXP Locks

| Deliverable | Conflict | Resolution |
|-------------|----------|------------|
| P0.1 Shadow Gate | None | — |
| P0.2 Manifest Signing | None | — |
| P0.3 Environment Diversity | **CAL-EXP-3 CANONICAL** | Use CAL-EXP-4 or standalone checker |

---

## 6. Frozen Surfaces — Do Not Touch

Per `PILOT_CONTRACT_POSTURE.md` and `CAL_EXP_3_INDEX.md`:

| Surface | Path | Status |
|---------|------|--------|
| CAL-EXP-1 Harness | `scripts/first_light_cal_exp1_*.py` | **FROZEN** |
| CAL-EXP-2 Harness | `scripts/first_light_cal_exp2_convergence.py` | **FROZEN** |
| CAL-EXP-3 Verifier | `scripts/verify_cal_exp_3_run.py` | **CANONICAL v1.0** |
| P5 CAL-EXP Harness | `scripts/run_p5_cal_exp1.py` | **FROZEN** |
| RUN_SHADOW_AUDIT Contract | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | **FROZEN** |
| Pilot Ingest Adapter | `external_ingest/adapter_enums.py` | **FROZEN** |

---

## 7. Recommended Implementation Order

1. **P0.2 Manifest Signing** — No conflicts, safe to proceed
2. **P0.1 Shadow Release Gate** — Proceed with renamed paths
3. **P0.3 Environment Diversity** — Implement via CAL-EXP-4 or standalone checker

---

## 8. STRATCOM Decision Required

If P0.3 environment diversity MUST be in CAL-EXP-3:

1. Submit change request to STRATCOM
2. Document rationale: "L5 claim integrity requires environment diversity proof"
3. Obtain explicit authorization
4. Update CAL_EXP_3_INDEX.md with new version

**Alternative**: Add to CAL-EXP-4 (already unfrozen) and apply retroactively to L5 claims.

---

*This report is for internal planning. It does not constitute authorization to modify frozen surfaces.*
