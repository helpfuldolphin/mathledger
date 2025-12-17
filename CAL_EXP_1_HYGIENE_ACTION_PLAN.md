# CAL-EXP-1 HYGIENE ACTION PLAN

**Date:** 2025-12-13
**Priority:** CRITICAL
**Estimated Commands:** 15+ git add operations

---

## PHASE 1: TRACK CRITICAL HARNESS SCRIPTS

**SAVE TO REPO: YES**

```bash
# CAL-EXP-1 Harness Entry Points
git add scripts/run_p5_cal_exp1.py
git add scripts/first_light_cal_exp1_warm_start.py
git add scripts/first_light_cal_exp1_runtime_stability.py
git add scripts/first_light_cal_exp2_convergence.py
git add scripts/first_light_cal_exp3_regime_change.py
git add scripts/first_light_proof_hash_snapshot.py

# Verification
git status scripts/run_p5_cal_exp1.py scripts/first_light_cal_exp1*.py
```

---

## PHASE 2: TRACK CRITICAL BACKEND MODULES

**SAVE TO REPO: YES**

### 2.1 experiments/u2/runtime/ (ENTIRE DIRECTORY)

```bash
# Track the entire runtime module
git add experiments/u2/runtime/__init__.py
git add experiments/u2/runtime/profile_guard.py
git add experiments/u2/runtime/calibration_correlation.py

# Verification
git ls-files experiments/u2/runtime/
```

### 2.2 backend/telemetry/ (ENTIRE DIRECTORY)

```bash
# Track the entire telemetry module
git add backend/telemetry/__init__.py
git add backend/telemetry/governance_signal.py
git add backend/telemetry/p4_integration.py
git add backend/telemetry/rtts_cal_exp_window_join.py
git add backend/telemetry/rtts_continuity_tracker.py
git add backend/telemetry/rtts_correlation_tracker.py
git add backend/telemetry/rtts_mock_detector.py
git add backend/telemetry/rtts_statistical_validator.py
git add backend/telemetry/rtts_window_validator.py

# Verification
git ls-files backend/telemetry/
```

### 2.3 Individual Backend Files

```bash
# Derivation budget modules
git add derivation/budget_cal_exp_integration.py
git add derivation/budget_invariants.py

# Experiments reconciliation
git add experiments/u2/cal_exp1_reconciliation.py

# Governance what-if
git add backend/governance/what_if_engine.py

# Verification
git status derivation/budget*.py experiments/u2/cal_exp1*.py
```

---

## PHASE 3: TRACK CONFIGURATION FILES

**SAVE TO REPO: YES**

```bash
# P5 synthetic config
git add config/p5_synthetic.json

# Verification
git ls-files config/p5_synthetic.json
```

---

## PHASE 4: TRACK TEST DIRECTORIES

**SAVE TO REPO: YES**

### 4.1 tests/telemetry/

```bash
git add tests/telemetry/__init__.py
git add tests/telemetry/test_p4_integration.py
git add tests/telemetry/test_rtts_cal_exp_window_join.py
git add tests/telemetry/test_rtts_components.py
git add tests/telemetry/test_rtts_p52_validation.py
git add tests/telemetry/test_rtts_pipeline_integration.py
git add tests/telemetry/test_tda_feedback.py
git add tests/telemetry/test_telemetry_phaseVI_fusion.py
git add tests/telemetry/test_telemetry_snapshot.py

# Verification
git ls-files tests/telemetry/
```

### 4.2 tests/derivation/

```bash
git add tests/derivation/

# Verification
git ls-files tests/derivation/
```

### 4.3 tests/evidence/

```bash
git add tests/evidence/

# Verification
git ls-files tests/evidence/
```

### 4.4 tests/experiments/

```bash
git add tests/experiments/

# Verification
git ls-files tests/experiments/
```

### 4.5 tests/ht/

```bash
git add tests/ht/

# Verification
git ls-files tests/ht/
```

### 4.6 tests/synthetic/

```bash
git add tests/synthetic/

# Verification
git ls-files tests/synthetic/
```

### 4.7 tests/rfl/ (partial)

```bash
git add tests/rfl/test_abstention_uplift_integration.py
git add tests/rfl/test_epistemic_drift_integration.py
git add tests/rfl/test_taxonomy_phaseV_integrations.py
git add tests/rfl/test_taxonomy_phaseV_p3p4_integration.py

# Verification
git ls-files tests/rfl/
```

---

## PHASE 5: COMMIT HYGIENE FIX

**SAVE TO REPO: YES**

```bash
# Stage all hygiene fixes
git add COMMIT_HYGIENE_REPORT_CAL_EXP_1.md
git add CAL_EXP_1_HYGIENE_ACTION_PLAN.md

# Commit with clear message
git commit -m "fix(hygiene): track all CAL-EXP-1 dependencies for reproducibility

CRITICAL: This commit adds all previously untracked files required
to execute CAL-EXP-1 from a clean checkout.

Files added:
- scripts/run_p5_cal_exp1.py (main harness)
- scripts/first_light_cal_exp1_warm_start.py
- scripts/first_light_cal_exp1_runtime_stability.py
- experiments/u2/runtime/ (entire module)
- backend/telemetry/ (entire module)
- config/p5_synthetic.json
- tests/telemetry/, tests/derivation/, tests/evidence/
- tests/experiments/, tests/ht/, tests/synthetic/

Closes reproducibility gap identified in COMMIT_HYGIENE_REPORT.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## PHASE 6: VERIFICATION

**SAVE TO REPO: NO (verification only)**

```bash
# Full verification script
echo "=== CAL-EXP-1 Hygiene Verification ==="

# Check harness scripts exist in git
echo "Checking harness scripts..."
git ls-files scripts/run_p5_cal_exp1.py scripts/first_light_cal_exp1*.py

# Check backend modules
echo "Checking backend modules..."
git ls-files experiments/u2/runtime/ backend/telemetry/

# Check config
echo "Checking config..."
git ls-files config/p5_synthetic.json

# Check test directories
echo "Checking test directories..."
git ls-files tests/telemetry/ tests/derivation/ tests/evidence/

# Import test
echo "Testing imports..."
python -c "from experiments.u2.runtime import build_runtime_health_snapshot; print('OK')"
python -c "import backend.telemetry; print('OK')"

echo "=== Verification Complete ==="
```

---

## BATCH EXECUTION SCRIPT

For convenience, run all Phase 1-5 in one script:

```bash
#!/bin/bash
# hygiene_fix.sh - Execute all hygiene fixes

set -e

echo "Phase 1: Tracking harness scripts..."
git add scripts/run_p5_cal_exp1.py
git add scripts/first_light_cal_exp1_warm_start.py
git add scripts/first_light_cal_exp1_runtime_stability.py
git add scripts/first_light_cal_exp2_convergence.py
git add scripts/first_light_cal_exp3_regime_change.py
git add scripts/first_light_proof_hash_snapshot.py

echo "Phase 2: Tracking backend modules..."
git add experiments/u2/runtime/
git add backend/telemetry/
git add derivation/budget_cal_exp_integration.py
git add derivation/budget_invariants.py
git add experiments/u2/cal_exp1_reconciliation.py
git add backend/governance/what_if_engine.py

echo "Phase 3: Tracking config..."
git add config/p5_synthetic.json

echo "Phase 4: Tracking test directories..."
git add tests/telemetry/
git add tests/derivation/
git add tests/evidence/
git add tests/experiments/
git add tests/ht/
git add tests/synthetic/
git add tests/rfl/test_abstention_uplift_integration.py
git add tests/rfl/test_epistemic_drift_integration.py
git add tests/rfl/test_taxonomy_phaseV_integrations.py
git add tests/rfl/test_taxonomy_phaseV_p3p4_integration.py

echo "Phase 5: Tracking reports..."
git add COMMIT_HYGIENE_REPORT_CAL_EXP_1.md
git add CAL_EXP_1_HYGIENE_ACTION_PLAN.md

echo "All files staged. Run 'git status' to verify."
```

---

## NOTES

1. **Do NOT ignore these directories** in `.gitignore`
2. **All commands assume execution from repository root**
3. **Verify each phase before proceeding to next**
4. **If any git add fails**, check file existence and path spelling

---

## POST-FIX VALIDATION

After committing, validate reproducibility:

```bash
# Clone to fresh directory
cd /tmp
git clone <repo-url> clean_test
cd clean_test

# Execute CAL-EXP-1
python scripts/run_p5_cal_exp1.py --cycles 200 --seed 42 --output-dir results/hygiene_test

# Expected: Successful execution with output artifacts
```

---
