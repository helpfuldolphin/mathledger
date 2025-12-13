# Auditor Smoke-Test Checklist

**Purpose**: Quick verification guide for auditors reviewing calibration campaign artifacts.
**Metric Definitions Reference**: `METRIC_DEFINITIONS.md@v1.1.0`

---

## Version Alignment

This checklist is validated against **METRIC_DEFINITIONS.md v1.1.0**.

### Changes from v1.0.0 to v1.1.0

| Change | Type | Impact |
|--------|------|--------|
| Added CAL-EXP-2 baseline values | Doc-only | Reference values for H1-H4 thresholds |
| Added UPGRADE-2 hypothesis targets | Doc-only | Target values documented (not gates) |
| Document promoted to canonical authority | Doc-only | Scope Lock compliance |
| Added Phase-Lag Reconciliation section | Doc-only | Advisory function reference |

**Script Compatibility**: v1.1.0 changes are doc-only and do not alter script expectations from v1.0.0.

---

## What to Read First

| Priority | Document | Purpose |
|----------|----------|---------|
| 1 | `METRIC_DEFINITIONS.md@v1.1.0` | Understand metric semantics before reviewing data |
| 2 | `CAL_EXP_2_Canonical_Record.md` | Understand convergence floor and baseline |
| 3 | `UPGRADE_2_DRAFT.md` (Scope Lock section) | Understand what is/isn't claimed |

---

## Formal Verification Table

| CheckID | Artifact | Invariant | Command (PowerShell) | Command (Bash) | Expected |
|---------|----------|-----------|---------------------|----------------|----------|
| CHK-001 | UPGRADE_2_DRAFT.md | No numeric acceptance thresholds | `Select-String -Pattern "acceptance.*threshold" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md` | `grep -i "acceptance.*threshold" docs/system_law/calibration/UPGRADE_2_DRAFT.md` | No matches |
| CHK-002 | UPGRADE_2_DRAFT.md | REALITY LOCK tags present | `(Select-String -Pattern "REALITY LOCK" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md).Count` | `grep -c "REALITY LOCK" docs/system_law/calibration/UPGRADE_2_DRAFT.md` | >= 4 |
| CHK-003 | UPGRADE_2_DRAFT.md | H1-H4 use mean_delta_p only | `Select-String -Pattern "divergence_scalar" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md` | `grep -i "divergence_scalar" docs/system_law/calibration/UPGRADE_2_DRAFT.md` | No matches |
| CHK-004 | METRIC_DEFINITIONS.md | Version header is v1.1.0 | `Select-String -Pattern "Version.*1.1.0" -Path docs\system_law\calibration\METRIC_DEFINITIONS.md` | `grep "Version.*1.1.0" docs/system_law/calibration/METRIC_DEFINITIONS.md` | 1 match |
| CHK-005 | UPGRADE_2_DRAFT.md | Scope Lock section exists | `Select-String -Pattern "^## Scope Lock" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md` | `grep "^## Scope Lock" docs/system_law/calibration/UPGRADE_2_DRAFT.md` | 1 match |
| CHK-006 | UPGRADE_2_DRAFT.md | Status is PROVISIONAL | `Select-String -Pattern "Status.*PROVISIONAL" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md` | `grep "Status.*PROVISIONAL" docs/system_law/calibration/UPGRADE_2_DRAFT.md` | 1 match |
| CHK-007 | METRIC_DEFINITIONS.md | Status is FROZEN | `Select-String -Pattern "Status.*FROZEN" -Path docs\system_law\calibration\METRIC_DEFINITIONS.md` | `grep "Status.*FROZEN" docs/system_law/calibration/METRIC_DEFINITIONS.md` | 1 match |
| CHK-008 | CAL_EXP_2_Canonical_Record.md | Status is CANONICAL | `Select-String -Pattern "Status.*CANONICAL" -Path docs\system_law\calibration\CAL_EXP_2_Canonical_Record.md` | `grep "Status.*CANONICAL" docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md` | 1 match |
| CHK-009 | AUDITOR_CHECKLIST.md | Non-Gating Invariant declared | `Select-String -Pattern "observational only" -Path docs\system_law\calibration\AUDITOR_CHECKLIST.md` | `grep "observational only" docs/system_law/calibration/AUDITOR_CHECKLIST.md` | 1 match |

---

## Non-Gating Invariant

**CHK-009**: AUDITOR_CHECKLIST results are **observational only**.

They must never be used as acceptance gates, deployment blockers, or promotion criteria. Any gating logic requires a separate ratified specification.

---

## Pass/Fail Recording

| CheckID | Pass | Fail | Auditor | Date |
|---------|------|------|---------|------|
| CHK-001 | [ ] | [ ] | ________ | ________ |
| CHK-002 | [ ] | [ ] | ________ | ________ |
| CHK-003 | [ ] | [ ] | ________ | ________ |
| CHK-004 | [ ] | [ ] | ________ | ________ |
| CHK-005 | [ ] | [ ] | ________ | ________ |
| CHK-006 | [ ] | [ ] | ________ | ________ |
| CHK-007 | [ ] | [ ] | ________ | ________ |
| CHK-008 | [ ] | [ ] | ________ | ________ |
| CHK-009 | [ ] | [ ] | ________ | ________ |

**Overall**: All CHK-* must PASS for audit approval.

---

## Artifacts to Confirm Exist

### Documentation Artifacts

| File | Path | Status |
|------|------|--------|
| Metric Definitions | `docs/system_law/calibration/METRIC_DEFINITIONS.md` | FROZEN (v1.1.0) |
| CAL-EXP-2 Record | `docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md` | CANONICAL |
| UPGRADE-2 Draft | `docs/system_law/calibration/UPGRADE_2_DRAFT.md` | PROVISIONAL |
| Auditor Checklist | `docs/system_law/calibration/AUDITOR_CHECKLIST.md` | REFERENCE |
| CAL-EXP-1 Verdict | `docs/calibration/CAL_EXP_1_UPGRADE_1_VERDICT.md` | CANONICAL |

### Experimental Data Artifacts

| Artifact | Path | Required Fields |
|----------|------|-----------------|
| CAL-EXP-2 Run Config | `results/cal_exp_2/p4_*/run_config.json` | `twin_lr_overrides`, `cycles` |
| CAL-EXP-2 Real Cycles | `results/cal_exp_2/p4_*/real_cycles.jsonl` | `usla_state.H`, `usla_state.rho` |
| CAL-EXP-2 Twin Predictions | `results/cal_exp_2/p4_*/twin_predictions.jsonl` | `twin_state.H`, `twin_state.rho` |
| CAL-EXP-2 Summary | `results/cal_exp_2/p4_*/p4_summary.json` | `divergence_rate`, `twin_accuracy` |
| CAL-EXP-2 Metadata | `results/cal_exp_2/p4_*/RUN_METADATA.json` | `verdict`, `convergence_floor` |

---

## Smoke-Test Script

### PowerShell (Windows)

```powershell
# Run all checks
Write-Host "CHK-001: No acceptance thresholds"
$chk001 = Select-String -Pattern "acceptance.*threshold" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md
if ($chk001) { Write-Host "FAIL" -ForegroundColor Red } else { Write-Host "PASS" -ForegroundColor Green }

Write-Host "CHK-002: REALITY LOCK tags >= 4"
$chk002 = (Select-String -Pattern "REALITY LOCK" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md).Count
if ($chk002 -ge 4) { Write-Host "PASS ($chk002)" -ForegroundColor Green } else { Write-Host "FAIL ($chk002)" -ForegroundColor Red }

Write-Host "CHK-003: No divergence_scalar"
$chk003 = Select-String -Pattern "divergence_scalar" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md
if ($chk003) { Write-Host "FAIL" -ForegroundColor Red } else { Write-Host "PASS" -ForegroundColor Green }

Write-Host "CHK-004: Version 1.1.0"
$chk004 = Select-String -Pattern "Version.*1.1.0" -Path docs\system_law\calibration\METRIC_DEFINITIONS.md
if ($chk004) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }

Write-Host "CHK-005: Scope Lock section"
$chk005 = Select-String -Pattern "^## Scope Lock" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md
if ($chk005) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }

Write-Host "CHK-006: UPGRADE_2 PROVISIONAL"
$chk006 = Select-String -Pattern "PROVISIONAL" -Path docs\system_law\calibration\UPGRADE_2_DRAFT.md
if ($chk006) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }

Write-Host "CHK-007: METRIC_DEFINITIONS FROZEN"
$chk007 = Select-String -Pattern "FROZEN" -Path docs\system_law\calibration\METRIC_DEFINITIONS.md
if ($chk007) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }

Write-Host "CHK-008: CAL_EXP_2 CANONICAL"
$chk008 = Select-String -Pattern "CANONICAL" -Path docs\system_law\calibration\CAL_EXP_2_Canonical_Record.md
if ($chk008) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL" -ForegroundColor Red }
```

### Bash (Linux/macOS/Git Bash)

```bash
#!/bin/bash
echo "CHK-001: No acceptance thresholds"
if grep -qi "acceptance.*threshold" docs/system_law/calibration/UPGRADE_2_DRAFT.md; then echo "FAIL"; else echo "PASS"; fi

echo "CHK-002: REALITY LOCK tags >= 4"
count=$(grep -c "REALITY LOCK" docs/system_law/calibration/UPGRADE_2_DRAFT.md)
if [ "$count" -ge 4 ]; then echo "PASS ($count)"; else echo "FAIL ($count)"; fi

echo "CHK-003: No divergence_scalar"
if grep -qi "divergence_scalar" docs/system_law/calibration/UPGRADE_2_DRAFT.md; then echo "FAIL"; else echo "PASS"; fi

echo "CHK-004: Version 1.1.0"
if grep -q "Version.*1.1.0" docs/system_law/calibration/METRIC_DEFINITIONS.md; then echo "PASS"; else echo "FAIL"; fi

echo "CHK-005: Scope Lock section"
if grep -q "^## Scope Lock" docs/system_law/calibration/UPGRADE_2_DRAFT.md; then echo "PASS"; else echo "FAIL"; fi

echo "CHK-006: UPGRADE_2 PROVISIONAL"
if grep -q "PROVISIONAL" docs/system_law/calibration/UPGRADE_2_DRAFT.md; then echo "PASS"; else echo "FAIL"; fi

echo "CHK-007: METRIC_DEFINITIONS FROZEN"
if grep -q "FROZEN" docs/system_law/calibration/METRIC_DEFINITIONS.md; then echo "PASS"; else echo "FAIL"; fi

echo "CHK-008: CAL_EXP_2 CANONICAL"
if grep -q "CANONICAL" docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md; then echo "PASS"; else echo "FAIL"; fi
```

---

## Expected Smoke-Test Output

```
CHK-001: No acceptance thresholds
PASS
CHK-002: REALITY LOCK tags >= 4
PASS (5)
CHK-003: No divergence_scalar
PASS
CHK-004: Version 1.1.0
PASS
CHK-005: Scope Lock section
PASS
CHK-006: UPGRADE_2 PROVISIONAL
PASS
CHK-007: METRIC_DEFINITIONS FROZEN
PASS
CHK-008: CAL_EXP_2 CANONICAL
PASS
```

---

## Red Flags (Immediate Escalation)

| Flag | Condition | Action |
|------|-----------|--------|
| RF-001 | Any CHK-* fails | Block approval, escalate to STRATCOM |
| RF-002 | Acceptance thresholds in UPGRADE_2 | Scope Lock violation |
| RF-003 | Claims "validated" or "confirmed" in UPGRADE_2 | Premature claim |
| RF-004 | Metric definitions modified (not additive) | Change policy violation |
| RF-005 | Missing baseline values in CAL-EXP-3 protocol | Incomplete protocol |

---

## Sign-Off

| Check | Auditor | Date |
|-------|---------|------|
| All CHK-* PASS | __________ | __________ |
| No Red Flags | __________ | __________ |
| Artifacts Verified | __________ | __________ |
| **AUDIT APPROVED** | __________ | __________ |

---

*This checklist is part of the calibration governance framework. Updates require STRATCOM approval.*
*Validated against METRIC_DEFINITIONS.md@v1.1.0*
