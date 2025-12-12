# PQ Rehearsal Mode: Smoke-Test Readiness Checklist

**Document Version**: 2.0  
**Author**: Manus-H  
**Date**: December 11, 2025  
**Reality Lock Status**: COMPLIANT

---

## Overview

This checklist validates the rehearsal mode functionality added to the PQ Operator Execution Kit. Rehearsal mode allows operators to train on failure scenarios using JSON fixtures without requiring a live node.

---

## Files Added/Modified

### New Files

- [x] **tests/fixtures/pq_rehearsal_scenarios.json** (REAL-READY)
  - **Status**: ✅ Created, JSON validated
  - **Lines**: 90
  - **Purpose**: Fixture data for 4 rehearsal scenarios

- [x] **tests/unit/test_pq_rehearsal_mode.py** (REAL-READY)
  - **Status**: ✅ Created, executable, all tests passed
  - **Lines**: 180
  - **Purpose**: 8 automated tests (2 per scenario)

### Modified Files

- [x] **scripts/pq_activation_dryrun.py** (REAL-READY)
  - **Status**: ✅ Updated with rehearsal mode support
  - **Changes**: Added `--rehearsal`, `--scenario`, `--fixture` arguments
  - **Backward Compatibility**: ✅ Normal mode unchanged

---

## Rehearsal Scenarios

### 1. Success Scenario

**Purpose**: Train operators on ideal activation readiness

**Command**:
```bash
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario success
```

**Expected Exit Code**: 0

**Expected Report**:
```json
{
  "checks_passed": 6,
  "checks_failed": 0,
  "warnings": 0,
  "ready_for_activation": true
}
```

**Actual Result**: ✅ PASSED
- Exit code: 0
- Report matches expected structure
- All 6 checks passed

---

### 2. Missing Module Scenario

**Purpose**: Train operators to recognize missing PQ consensus modules

**Command**:
```bash
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario missing_module
```

**Expected Exit Code**: 1

**Expected Report**:
```json
{
  "checks_passed": 5,
  "checks_failed": 1,
  "warnings": 0,
  "ready_for_activation": false,
  "failure_reason": "missing_pq_modules"
}
```

**Actual Result**: ✅ PASSED
- Exit code: 1
- Report matches expected structure
- `backend/consensus_pq/rules.py` shown as missing

---

### 3. Drift Radar Disabled Scenario

**Purpose**: Train operators to recognize critical monitoring gaps

**Command**:
```bash
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario drift_radar_disabled
```

**Expected Exit Code**: 1

**Expected Report**:
```json
{
  "checks_passed": 5,
  "checks_failed": 1,
  "warnings": 1,
  "ready_for_activation": false,
  "failure_reason": "drift_radar_disabled"
}
```

**Actual Result**: ✅ PASSED
- Exit code: 1
- Report matches expected structure
- Warning message displayed about drift radar criticality

---

### 4. Low Disk Space Scenario

**Purpose**: Train operators to recognize resource constraints

**Command**:
```bash
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario low_disk_space
```

**Expected Exit Code**: 1

**Expected Report**:
```json
{
  "checks_passed": 5,
  "checks_failed": 1,
  "warnings": 0,
  "ready_for_activation": false,
  "failure_reason": "insufficient_disk_space"
}
```

**Actual Result**: ✅ PASSED
- Exit code: 1
- Report matches expected structure
- Shows 50GB available vs 100GB required

---

## Automated Test Suite

### Test Execution

**Command**:
```bash
cd /home/ubuntu/mathledger  # Or C:\dev\mathledger on Windows
python3 tests/unit/test_pq_rehearsal_mode.py
```

**Test Coverage**: 8 tests (2 per scenario)

| Test # | Scenario | Validates | Status |
|--------|----------|-----------|--------|
| 1 | Success | Exit code 0 | ✅ PASSED |
| 2 | Success | Report structure | ✅ PASSED |
| 3 | Missing Module | Exit code 1 | ✅ PASSED |
| 4 | Missing Module | Report structure | ✅ PASSED |
| 5 | Drift Radar Disabled | Exit code 1 | ✅ PASSED |
| 6 | Drift Radar Disabled | Report structure | ✅ PASSED |
| 7 | Low Disk Space | Exit code 1 | ✅ PASSED |
| 8 | Low Disk Space | Report structure | ✅ PASSED |

**Test Summary**: 8/8 tests passed (100%)

---

## Deterministic Output Validation

### Exit Codes

- [x] Success scenario: Exit code 0 (deterministic)
- [x] Missing module: Exit code 1 (deterministic)
- [x] Drift radar disabled: Exit code 1 (deterministic)
- [x] Low disk space: Exit code 1 (deterministic)

### JSON Report Fields

All scenarios produce consistent JSON schema:

**Required Fields** (all scenarios):
- `timestamp` (float)
- `activation_block` (int)
- `checks_passed` (int)
- `checks_failed` (int)
- `warnings` (int)
- `success_rate` (float)
- `ready_for_activation` (boolean)

**Conditional Fields** (failure scenarios only):
- `failure_reason` (string)

**Validation**: ✅ All reports conform to schema

---

## Operator Training Use Cases

### Use Case 1: Pre-Activation Training

**Scenario**: Operators run all 4 scenarios to familiarize themselves with possible outcomes

**Commands**:
```bash
# Run all scenarios
for scenario in success missing_module drift_radar_disabled low_disk_space; do
  python3 scripts/pq_activation_dryrun.py --rehearsal --scenario $scenario
  echo "---"
done
```

**Learning Outcomes**:
- Recognize what a successful check looks like
- Identify critical vs non-critical failures
- Understand failure_reason taxonomy
- Practice reading colored terminal output

---

### Use Case 2: Automated Pre-Activation Validation

**Scenario**: CI/CD pipeline runs rehearsal tests before deployment

**Command**:
```bash
python3 tests/unit/test_pq_rehearsal_mode.py
```

**Success Criteria**: All 8 tests pass

**Integration**: Can be added to GitHub Actions, GitLab CI, or Jenkins

---

### Use Case 3: Operator Certification

**Scenario**: Operators must demonstrate understanding of failure modes

**Process**:
1. Run each scenario
2. Explain what the failure means
3. Describe remediation steps
4. Demonstrate understanding of exit codes and reports

**Certification Criteria**: Correctly interpret all 4 scenarios

---

## Reality Lock Compliance

### Engineering Truthfulness

✅ **Explicit Tagging**: All files tagged `# REAL-READY`

✅ **No RPC Dependency**: Rehearsal mode uses JSON fixtures, not live node queries

✅ **Deterministic Outputs**: All scenarios produce consistent, testable results

✅ **Backward Compatibility**: Normal mode unchanged, rehearsal mode is additive

✅ **Test Coverage**: 8 automated tests with 100% pass rate

### REAL-READY Components

1. **pq_rehearsal_scenarios.json**: Structured fixture data
2. **test_pq_rehearsal_mode.py**: Automated test suite
3. **pq_activation_dryrun.py** (updated): Rehearsal mode implementation

### No DEMO-SCAFFOLD Components

All rehearsal mode components are production-ready and can be used immediately for operator training.

---

## Integration Readiness

### Ready for Immediate Use

- [x] Operators can run rehearsal scenarios for training
- [x] CI/CD can run automated tests
- [x] Instructors can use scenarios for certification programs
- [x] Documentation is complete and accurate

### Next Steps

1. **Operator Training Sessions**: Schedule walkthrough with all node operators
2. **CI Integration**: Add test suite to pre-deployment checks
3. **Certification Program**: Develop operator certification based on rehearsal scenarios
4. **Expand Scenarios**: Add more failure modes as needed (e.g., version mismatch, network partition)

---

## Smoke-Test Commands

### Quick Validation

```bash
# Test all scenarios
python3 tests/unit/test_pq_rehearsal_mode.py

# Expected output: "✓ ALL TESTS PASSED"
# Expected exit code: 0
```

### Manual Scenario Testing

```bash
# Success scenario
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario success
# Expected: Exit 0, 6/6 checks passed

# Missing module
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario missing_module
# Expected: Exit 1, 5/6 checks passed, failure_reason="missing_pq_modules"

# Drift radar disabled
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario drift_radar_disabled
# Expected: Exit 1, 5/6 checks passed, 1 warning, failure_reason="drift_radar_disabled"

# Low disk space
python3 scripts/pq_activation_dryrun.py --rehearsal --scenario low_disk_space
# Expected: Exit 1, 5/6 checks passed, failure_reason="insufficient_disk_space"
```

---

## Maintenance

### When to Update Scenarios

1. When new checks are added to the dry-run script
2. When failure taxonomies change
3. When operators request additional training scenarios
4. When real-world activation issues reveal gaps in training

### Responsible Party

MathLedger Core Team / Manus-H

---

## Final Verification

**Date**: 2025-12-11  
**Commit**: TBD (to be pushed)  
**Branch**: master  
**Repository**: helpfuldolphin/mathledger

**Status**: ✅ ALL REHEARSAL MODE TESTS PASSED

**MANUS-H: PQ Rehearsal Mode Ready.**
