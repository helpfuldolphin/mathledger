# Pilot CI Isolation

**Version:** 1.0.0
**Created:** 2025-12-13
**Purpose:** Document the CI isolation guarantees for pilot-phase development.

---

## Overview

Pilot-phase code is **strictly isolated** from frozen calibration experiments (CAL-EXP-1, CAL-EXP-2, CAL-EXP-3). This document explains how CI enforces this isolation.

---

## Isolation Guarantees

### 1. Path-Based Trigger Scoping

The pilot CI gate (`.github/workflows/pilot-phase-gate.yml`) **only triggers** on:

| Path Pattern | Purpose |
|--------------|---------|
| `pilot/**` | Pilot application code |
| `external_ingest/**` | External data ingestion modules |
| `adapters/**` | Integration adapters |
| `docs/pilot/**` | Pilot documentation |
| `tests/pilot/**` | Pilot test suite |
| `scripts/pilot_*.py` | Pilot scripts (explicit prefix) |

### 2. Frozen Paths Protection

Pilot CI **explicitly blocks** modifications to frozen CAL-EXP harnesses:

```
scripts/run_p5_cal_exp1.py           (FROZEN)
scripts/first_light_cal_exp1_*.py    (FROZEN)
scripts/first_light_cal_exp2_*.py    (FROZEN)
scripts/first_light_cal_exp3_*.py    (FROZEN)
```

### 3. Experiment Results Isolation

Pilot code is **forbidden** from referencing frozen experiment directories:

```
results/cal_exp_1/       (FROZEN)
results/cal_exp_2/       (FROZEN)
results/cal_exp_1_upgrade_1/  (FROZEN)
```

Pilot code must use `results/pilot/` or similar isolated output paths.

---

## Why This Matters

| Concern | Mitigation |
|---------|------------|
| Pilot changes break CAL-EXP CI | Path filters ensure CAL-EXP gate never triggers on pilot paths |
| Pilot code modifies frozen harnesses | Boundary check explicitly blocks harness modifications |
| Pilot artifacts pollute experiment results | Directory reference check prevents writes to frozen paths |
| Docs-only pilot changes cause friction | Docs scoped to `docs/pilot/` only trigger pilot gate |

---

## Developer Checklist

When working on pilot code:

- [ ] All pilot code lives in `pilot/`, `external_ingest/`, or `adapters/`
- [ ] Pilot scripts use `pilot_` prefix (e.g., `scripts/pilot_ingest.py`)
- [ ] Pilot tests live in `tests/pilot/`
- [ ] Pilot docs live in `docs/pilot/`
- [ ] Output directories use `results/pilot/` (never `results/cal_exp_*`)
- [ ] No imports from frozen CAL-EXP harnesses

---

## CI Workflow Reference

**Workflow:** `.github/workflows/pilot-phase-gate.yml`

**Jobs:**
1. `boundary-check` - Verifies no frozen paths are modified
2. `pilot-lint` - Validates Python syntax
3. `pilot-tests` - Runs pilot test suite (if exists)
4. `summary` - Reports overall status

**Gating:** Boundary violations are **blocking**. Lint failures are **blocking**. Test failures are **advisory**.

---

## Related Documents

- `CRITICAL_FILES_MANIFEST.md` - Lists all frozen CAL-EXP files
- `.github/workflows/cal_exp_hygiene_gate.yml` - CAL-EXP CI gate (separate scope)
