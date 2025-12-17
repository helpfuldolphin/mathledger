# CAL-EXP-4 Readiness Assessment

**Date**: 2025-12-17
**Status**: SCAFFOLDING COMPLETE — BLOCKED ON EXECUTION SPEC
**Author**: Phase-II Planner

---

## 1. What's Ready

### 1.1 Verifier Skeleton

| Component | Status | Path |
|-----------|--------|------|
| Verifier script | **READY** | `scripts/verify_cal_exp_4_run.py` |
| Temporal structure checks | **READY** | 8 predicate checks (F5.1) |
| Variance profile checks | **READY** | 5 predicate checks (F5.2, F5.3, F5.7) |
| Schema validation | **READY** | F5.5 fail-close on malformed |
| Pathology detection | **READY** | F5.6 NaN/Inf detection |
| Claim cap logic | **READY** | F5.4 missing artifact → L3 cap |
| JSON report output | **READY** | Deterministic, sorted |

**Verifier imports and compiles**: YES

### 1.2 JSON Schemas

| Schema | Status | Path | Version |
|--------|--------|------|---------|
| Temporal Structure Audit | **FROZEN** | `schemas/cal_exp_4/temporal_structure_audit.schema.json` | 1.0.0 |
| Variance Profile Audit | **FROZEN** | `schemas/cal_exp_4/variance_profile_audit.schema.json` | 1.0.0 |

**Schemas are valid JSON**: YES

### 1.3 Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/calibration/test_verify_cal_exp_4_run.py` | 52 | **ALL PASS** |
| `tests/calibration/test_cal_exp_4_drift_guard.py` | 13 | **ALL PASS** |

**Total**: 65 tests passing

### 1.4 Documentation

| Document | Status | Purpose |
|----------|--------|---------|
| `CAL_EXP_4_INDEX.md` | LANDED | Index of authoritative documents |
| `CAL_EXP_4_VARIANCE_STRESS_SPEC.md` | BINDING | Charter, definitions, validity conditions |
| `CAL_EXP_4_IMPLEMENTATION_PLAN.md` | PROVISIONAL | Execution machinery, artifact layout |
| `CAL_EXP_4_VERIFIER_PLAN.md` | PROVISIONAL | Verifier check specification |
| `CAL_EXP_4_FREEZE.md` | FROZEN | Semantic freeze declaration |

---

## 2. What's Blocked

### 2.1 Harness Implementation

**Status**: NOT STARTED

The harness that runs the actual CAL-EXP-4 experiment and produces audit artifacts is not implemented. The verifier can validate artifacts, but no harness produces them.

| Component | Blocked On |
|-----------|------------|
| `scripts/run_cal_exp_4_harness.py` | Implementation spec finalization |
| Temporal structure auditor | Harness integration |
| Variance profile auditor | Harness integration |

### 2.2 CI Workflow

**Status**: NOT CREATED

| Component | Blocked On |
|-----------|------------|
| `.github/workflows/cal-exp-4-verification.yml` | Harness completion |

### 2.3 Execution Approval

| Gate | Status |
|------|--------|
| STRATCOM approval to execute | **PENDING** |
| Pilot firewall confirmation | **PENDING** |
| Environment isolation verification | **PENDING** |

---

## 3. What Would Start It

### 3.1 Prerequisites

1. **Harness Implementation**: Create `scripts/run_cal_exp_4_harness.py` that:
   - Extends CAL-EXP-3 harness with variance stress
   - Produces `validity/temporal_structure_audit.json`
   - Produces `validity/variance_profile_audit.json`
   - Outputs F5.x status in `RUN_METADATA.json`

2. **Integration Test**: Create test that runs harness + verifier end-to-end

3. **CI Workflow**: Create `.github/workflows/cal-exp-4-verification.yml`

4. **STRATCOM Sign-off**: Formal approval to execute (non-blocking for scaffolding)

### 3.2 Launch Checklist

```
[ ] Harness script created and tested
[ ] Harness produces all required artifacts (see CAL_EXP_4_IMPLEMENTATION_PLAN.md §2)
[ ] Verifier passes on harness output
[ ] CI workflow validates harness + verifier
[ ] STRATCOM approval obtained
[ ] Pilot firewall confirmed (FORBIDDEN per CAL_EXP_4_INDEX.md)
[ ] First run scheduled
```

---

## 4. Green CI Status

### 4.1 Current State

| Check | Status |
|-------|--------|
| Verifier imports | PASS |
| Schema validation | PASS |
| Unit tests (52) | PASS |
| Drift guard tests (13) | PASS |
| **Total** | **65 PASS, 0 FAIL** |

### 4.2 CI Command

```bash
pytest tests/calibration/test_verify_cal_exp_4_run.py tests/calibration/test_cal_exp_4_drift_guard.py -v
```

---

## 5. Artifact Summary

### 5.1 Committed Artifacts (Ready)

```
scripts/verify_cal_exp_4_run.py                              # Verifier (932 lines)
schemas/cal_exp_4/temporal_structure_audit.schema.json       # Schema v1.0.0
schemas/cal_exp_4/variance_profile_audit.schema.json         # Schema v1.0.0
tests/calibration/test_verify_cal_exp_4_run.py              # 52 tests
tests/calibration/test_cal_exp_4_drift_guard.py             # 13 tests
docs/system_law/calibration/CAL_EXP_4_INDEX.md              # Index
docs/system_law/calibration/CAL_EXP_4_VARIANCE_STRESS_SPEC.md   # Spec
docs/system_law/calibration/CAL_EXP_4_IMPLEMENTATION_PLAN.md    # Plan
docs/system_law/calibration/CAL_EXP_4_VERIFIER_PLAN.md          # Verifier plan
docs/system_law/calibration/CAL_EXP_4_FREEZE.md                 # Freeze declaration
```

### 5.2 Not Yet Created (Blocked)

```
scripts/run_cal_exp_4_harness.py                            # Harness
.github/workflows/cal-exp-4-verification.yml                # CI workflow
tests/integration/test_cal_exp_4_e2e.py                     # E2E test
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Verifier/harness mismatch | Medium | High | Verifier defines correctness per INDEX |
| Threshold creep | Low | Medium | Thresholds FROZEN in CAL_EXP_4_FREEZE.md |
| Pilot contamination | Low | Critical | Drift guard tests enforce FORBIDDEN status |
| Schema evolution | Low | High | Schema version 1.0.0 FROZEN |

---

## 7. Conclusion

**CAL-EXP-4 scaffolding is complete and green.**

The verifier, schemas, tests, and documentation are all in place. The experiment is blocked on harness implementation and STRATCOM approval. No test stubs or skips are required — all tests pass with the current scaffolding.

**Next Action**: Implement harness when STRATCOM approves execution.

---

**SHADOW MODE**: SHADOW-OBSERVE
**Contract Reference**: `CAL_EXP_4_INDEX.md`
