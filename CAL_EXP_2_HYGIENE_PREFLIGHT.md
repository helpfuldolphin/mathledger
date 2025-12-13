# CAL-EXP-2 HYGIENE PRE-FLIGHT CHECKLIST

**Date:** 2025-12-13
**Phase:** CAL-EXP-2 (P4 Divergence Minimization)
**Auditor:** Claude W (Commit Hygiene Sentinel)
**Verified on HEAD commit:** `f160c08fe786889bdb9f7c177fa33d4c644c6a99`

---

## PRE-FLIGHT STATUS: UNBLOCKED (with 1 git add)

CAL-EXP-2 can execute after tracking 1 file.

---

## BLOCKER CLASSIFICATION (Re-verified)

| Item | Classification | Actual Status | Remediation |
|------|----------------|---------------|-------------|
| `scripts/first_light_cal_exp2_convergence.py` | **(B) Exists, untracked** | TRUE BLOCKER | `git add` |
| `scripts/plot_budget_vs_divergence.py` | **(D) False dependency** | Post-analysis tool | None (optional) |
| `backend/topology/divergence_monitor.py` | **(D) False dependency** | Not imported by CAL-EXP-2 | None |
| `experiments/u2/runtime/` | **(D) False dependency** | Not imported by CAL-EXP-2 | None |
| `backend/telemetry/` | **(D) False dependency** | Not imported by CAL-EXP-2 | None |
| `config/p5_synthetic.json` | **(D) False dependency** | Not used by CAL-EXP-2 | None |
| `schemas/evidence/p4_divergence_log.schema.json` | **(D) False dependency** | Not validated in harness | None |
| `docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md` | **(D) Documentation** | Not execution blocker | Optional |

### Classification Key:
- **(A)** Missing from repo entirely
- **(B)** Exists but untracked
- **(C)** Exists under different path/name
- **(D)** Not actually required for CAL-EXP-2 (false dependency)

---

## VERIFICATION EVIDENCE

### Import Chain Analysis

CAL-EXP-2 harness (`first_light_cal_exp2_convergence.py`) imports:
```python
from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
```

All three are **TRACKED** in git. No transitive imports to:
- `backend/telemetry` - NOT in chain
- `experiments/u2/runtime` - NOT in chain
- `backend/topology/divergence_monitor` - NOT in chain

### Import Test Result (HEAD):
```
OK: FirstLightConfigP4
OK: FirstLightShadowRunnerP4
OK: MockTelemetryProvider
CAL-EXP-2 imports verified.
```

---

## TRUE BLOCKERS (1)

| File | Status | Remediation |
|------|--------|-------------|
| `scripts/first_light_cal_exp2_convergence.py` | Exists, UNTRACKED | `git add scripts/first_light_cal_exp2_convergence.py` |

---

## FALSE DEPENDENCIES EXPLAINED

### `scripts/plot_budget_vs_divergence.py`
- **Purpose:** Post-analysis cross-plot generator (used AFTER CAL-EXP-2 completes)
- **Required for execution:** NO
- **Recommendation:** Track if you want post-analysis capability, but NOT a blocker

### `backend/topology/divergence_monitor.py`
- **Purpose:** Global divergence monitoring (different from P4 divergence_analyzer)
- **Imported by CAL-EXP-2:** NO (checked via grep)
- **Required for execution:** NO

### `experiments/u2/runtime/`, `backend/telemetry/`, `config/p5_synthetic.json`
- **Purpose:** CAL-EXP-1 infrastructure
- **Imported by CAL-EXP-2:** NO (verified via import chain analysis)
- **Required for execution:** NO

### `schemas/evidence/p4_divergence_log.schema.json`
- **Purpose:** Schema for divergence log validation
- **Used in harness:** NO (harness outputs JSON without schema validation)
- **Required for execution:** NO

---

## TRACKED DEPENDENCIES (All Present)

| File | Status |
|------|--------|
| `backend/topology/first_light/runner_p4.py` | TRACKED |
| `backend/topology/first_light/config_p4.py` | TRACKED |
| `backend/topology/first_light/telemetry_adapter.py` | TRACKED |
| `backend/topology/first_light/divergence_analyzer.py` | TRACKED |
| `backend/topology/first_light/data_structures_p4.py` | TRACKED |
| `backend/topology/first_light/p5_pattern_classifier.py` | TRACKED |
| `tests/first_light/test_cal_exp2_exp3_scaffolds.py` | TRACKED |

---

## MINIMAL FIX PLAN

```bash
# Single command to unblock CAL-EXP-2
git add scripts/first_light_cal_exp2_convergence.py
```

**Optional (for full toolchain):**
```bash
# Post-analysis tools (not execution blockers)
git add scripts/plot_budget_vs_divergence.py

# Documentation (not execution blockers)
git add docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md
```

---

## CI CHECK STATUS

The `.github/workflows/cal_exp_hygiene_gate.yml` CAL-EXP-2 job checks:
- `scripts/first_light_cal_exp2_convergence.py` - TRUE dependency
- `scripts/plot_budget_vs_divergence.py` - Should be OPTIONAL (CI job needs update)
- `backend/topology/divergence_monitor.py` - FALSE dependency (CI job needs update)
- `schemas/evidence/p4_divergence_log.schema.json` - FALSE dependency (CI job needs update)

**CI Job Correction Required:** The `cal-exp2-hygiene` job should only check TRUE dependencies.

---

## PRE-FLIGHT VERDICT

```
╔══════════════════════════════════════════════════════════════════╗
║  CAL-EXP-2 PRE-FLIGHT: ██ UNBLOCKED ██                           ║
║                                                                  ║
║  TRUE BLOCKERS: 1 (scripts/first_light_cal_exp2_convergence.py)  ║
║  FALSE DEPENDENCIES: 7 (reclassified)                            ║
║                                                                  ║
║  Action: git add scripts/first_light_cal_exp2_convergence.py     ║
║  Then CAL-EXP-2 can execute.                                     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**SAVE TO REPO: YES**
