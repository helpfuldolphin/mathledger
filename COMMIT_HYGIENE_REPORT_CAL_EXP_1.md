# COMMIT HYGIENE REPORT: CAL-EXP-1 Reproducibility Audit

**Report Version:** 1.0.0
**Date:** 2025-12-13
**Auditor:** Claude W (Commit Hygiene Sentinel)
**Scope:** CAL-EXP-1 and pilot harnesses

---

## EXECUTIVE SUMMARY

**VERDICT: REPOSITORY IS BROKEN FOR CAL-EXP-1 REPLICATION**

A clean checkout of this repository at `HEAD` **CANNOT** execute CAL-EXP-1 as documented. Multiple critical modules required by the experiment harnesses are **UNTRACKED** in git.

| Category | Tracked | Untracked | Failure Rate |
|----------|---------|-----------|--------------|
| Harness Entry Points | 2 | 6 | 75% |
| Backend Modules | PARTIAL | 2 full dirs | CRITICAL |
| Test Directories | PARTIAL | 6 dirs | SEVERE |
| Configuration | 0 | 1 | 100% |

---

## 1. CAL-EXP-1 HARNESS ENTRY POINTS

### UNTRACKED (Will Fail on Clean Checkout)

| File | Purpose | Status |
|------|---------|--------|
| `scripts/run_p5_cal_exp1.py` | Main P5 CAL-EXP-1 harness | **UNTRACKED** |
| `scripts/first_light_cal_exp1_warm_start.py` | Warm-start harness (200-cycle) | **UNTRACKED** |
| `scripts/first_light_cal_exp1_runtime_stability.py` | Runtime stability metrics | **UNTRACKED** |
| `scripts/first_light_cal_exp2_convergence.py` | CAL-EXP-2 harness | **UNTRACKED** |
| `scripts/first_light_cal_exp3_regime_change.py` | CAL-EXP-3 harness | **UNTRACKED** |
| `scripts/first_light_proof_hash_snapshot.py` | Proof hash snapshot | **UNTRACKED** |

### TRACKED (Available on Clean Checkout)

| File | Purpose | Status |
|------|---------|--------|
| `scripts/first_light_p3_harness.py` | P3 harness | TRACKED |
| `results/cal_exp_1/cal_exp_1_harness.py` | CAL-EXP-1 execution harness | TRACKED |

---

## 2. BACKEND MODULE DEPENDENCIES

### CRITICAL: ENTIRE DIRECTORIES UNTRACKED

#### `experiments/u2/runtime/` - UNTRACKED
**Required by:** `first_light_cal_exp1_runtime_stability.py`

| File | Purpose |
|------|---------|
| `__init__.py` | Runtime profile & feature flag system |
| `profile_guard.py` | Profile drift detection |
| `calibration_correlation.py` | Calibration window correlation |

**Failure Mode:** `ImportError: cannot import name 'build_runtime_health_snapshot' from 'experiments.u2.runtime'`

#### `backend/telemetry/` - UNTRACKED
**Required by:** Multiple CAL-EXP-1 components

| File | Purpose |
|------|---------|
| `__init__.py` | Module init |
| `governance_signal.py` | Governance signal emission |
| `p4_integration.py` | P4 integration layer |
| `rtts_cal_exp_window_join.py` | RTTS CAL-EXP window join |
| `rtts_continuity_tracker.py` | RTTS continuity tracking |
| `rtts_correlation_tracker.py` | RTTS correlation tracking |
| `rtts_mock_detector.py` | RTTS mock detector |
| `rtts_statistical_validator.py` | RTTS statistical validation |
| `rtts_window_validator.py` | RTTS window validation |

**Failure Mode:** `ImportError: No module named 'backend.telemetry'`

### INDIVIDUAL UNTRACKED FILES

| File | Required By | Status |
|------|-------------|--------|
| `derivation/budget_cal_exp_integration.py` | Budget calibration | **UNTRACKED** |
| `derivation/budget_invariants.py` | Budget invariants | **UNTRACKED** |
| `experiments/u2/cal_exp1_reconciliation.py` | CAL-EXP-1 reconciliation | **UNTRACKED** |
| `backend/governance/what_if_engine.py` | What-if scenarios | **UNTRACKED** |
| `backend/logging/` | Logging infrastructure | **UNTRACKED** |
| `backend/nl/` | Natural language module | **UNTRACKED** |
| `backend/metrics/u2_analysis.py` | U2 metrics analysis | **UNTRACKED** |

---

## 3. TEST DIRECTORIES

### ENTIRELY UNTRACKED TEST DIRECTORIES

| Directory | Test Count | Status |
|-----------|------------|--------|
| `tests/telemetry/` | 8 files | **UNTRACKED** |
| `tests/derivation/` | Unknown | **UNTRACKED** |
| `tests/evidence/` | Unknown | **UNTRACKED** |
| `tests/experiments/` | Unknown | **UNTRACKED** |
| `tests/ht/` | Unknown | **UNTRACKED** |
| `tests/synthetic/` | Unknown | **UNTRACKED** |
| `tests/rfl/` (partial) | 4 files | **UNTRACKED** |

**Failure Mode:** Tests cannot validate CAL-EXP-1 components from clean checkout.

---

## 4. CONFIGURATION FILES

| File | Purpose | Status |
|------|---------|--------|
| `config/p5_synthetic.json` | P5 synthetic adapter config | **UNTRACKED** |

**Failure Mode:** `python scripts/run_p5_cal_exp1.py --adapter-config config/p5_synthetic.json` fails with FileNotFoundError.

---

## 5. TRACKED (AVAILABLE) DEPENDENCIES

These components are correctly tracked and will work on clean checkout:

| Directory/File | Status |
|----------------|--------|
| `backend/topology/first_light/*.py` | TRACKED (24 files) |
| `backend/tda/*.py` | TRACKED (9 files) |
| `backend/ledger/*.py` | TRACKED (includes `monotone_guard.py`) |
| `backend/verification/*.py` | TRACKED (includes drift_radar) |
| `backend/ht/*.py` | TRACKED |
| `backend/synthetic/*.py` | TRACKED |
| `backend/rfl/*.py` | TRACKED |
| `experiments/u2/*.py` (core) | TRACKED (8 files, excludes runtime/) |
| `tests/first_light/*.py` | TRACKED (18 files) |
| `tests/integration/test_p5_cal_exp1_harness.py` | TRACKED |

---

## 6. EXPLICIT FAILURE LIST

### Import Failures on Clean Checkout

1. **`scripts/run_p5_cal_exp1.py`**
   - Cannot execute: file does not exist

2. **`scripts/first_light_cal_exp1_warm_start.py`**
   - Cannot execute: file does not exist

3. **`scripts/first_light_cal_exp1_runtime_stability.py`**
   - Cannot execute: file does not exist
   - If it existed, would fail with:
     ```
     ImportError: cannot import name 'build_runtime_health_snapshot' from 'experiments.u2.runtime'
     ```

4. **`results/cal_exp_1/cal_exp_1_harness.py`** (TRACKED but depends on untracked)
   - Will execute but imports may fail depending on which tests reference untracked modules

### Test Failures on Clean Checkout

1. All tests in `tests/telemetry/` - Cannot run (directory missing)
2. All tests in `tests/derivation/` - Cannot run (directory missing)
3. All tests in `tests/evidence/` - Cannot run (directory missing)
4. All tests in `tests/experiments/` - Cannot run (directory missing)

---

## 7. VERIFICATION COMMAND

Run this to verify the hygiene failure:

```bash
# Create clean checkout
git clone <repo-url> clean_checkout
cd clean_checkout

# Attempt CAL-EXP-1 execution
python scripts/run_p5_cal_exp1.py --cycles 200 --seed 42 --output-dir results/test
# EXPECTED: FileNotFoundError or ModuleNotFoundError
```

---

## 8. MANIFEST GAPS

The existing `CRITICAL_FILES_MANIFEST.md` does NOT list:

| Missing Entry | Severity |
|---------------|----------|
| `scripts/run_p5_cal_exp1.py` | CRITICAL |
| `scripts/first_light_cal_exp1_warm_start.py` | CRITICAL |
| `scripts/first_light_cal_exp1_runtime_stability.py` | HIGH |
| `experiments/u2/runtime/*.py` | CRITICAL |
| `backend/telemetry/*.py` | CRITICAL |
| `config/p5_synthetic.json` | HIGH |
| `tests/telemetry/*.py` | HIGH |
| `tests/evidence/*.py` | HIGH |
| `derivation/budget_cal_exp_integration.py` | MEDIUM |

---

## CONCLUSION

**Reproducibility Status: FAILED**

A clean checkout cannot execute CAL-EXP-1. The repository contains "works on my machine" dependencies that are not version controlled.

**Immediate action required.** See ACTION_PLAN section below.

---
