# Pilot CI Isolation

**Version:** 1.1.0
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

## Extending Pilot CI Scope Safely

When adding new pilot paths to `pilot-phase-gate.yml`, follow these rules to avoid reintroducing global CI friction or compromising frozen harnesses.

### Rule 1: Never Use Root Wildcards

Path patterns must be **explicitly scoped** to pilot-owned directories. Never match broad patterns that could capture non-pilot files.

### Rule 2: Keep Frozen Harness List Authoritative

The `FROZEN_HARNESSES` array in `pilot-phase-gate.yml` is the **single source of truth** for protected CAL-EXP scripts. If a new harness is frozen:
1. Add it to `FROZEN_HARNESSES` in the workflow
2. Document it in `CRITICAL_FILES_MANIFEST.md`
3. Do NOT remove entries without explicit STRATCOM directive

### Rule 3: Scope Documentation Paths

Documentation triggers must use subdirectory scoping (e.g., `docs/pilot/**`), never `docs/**`.

### Examples

#### Safe Extension

```yaml
# Adding a new pilot module directory
paths:
  - 'pilot/**'
  - 'pilot_integrations/**'    # NEW: explicit, scoped directory
  - 'scripts/pilot_*.py'
  - 'scripts/pilot_integration_*.py'  # NEW: explicit prefix pattern
```

**Why safe:**
- `pilot_integrations/**` is a dedicated directory that cannot match `backend/`, `scripts/first_light_*`, or any frozen path
- `scripts/pilot_integration_*.py` uses explicit prefix, cannot match `scripts/run_p5_cal_exp1.py`

#### Unsafe Extension (DO NOT DO THIS)

```yaml
# DANGEROUS: These patterns cause global CI friction
paths:
  - 'pilot/**'
  - 'scripts/*.py'           # UNSAFE: matches ALL scripts including frozen harnesses
  - 'docs/**'                # UNSAFE: triggers on all documentation changes
  - '**/*.py'                # UNSAFE: matches entire codebase
  - 'backend/**'             # UNSAFE: overlaps with CAL-EXP dependencies
```

**Why unsafe:**
- `scripts/*.py` matches `scripts/first_light_cal_exp2_convergence.py` (frozen)
- `docs/**` triggers pilot CI on `docs/system_law/` changes (not pilot-related)
- `**/*.py` triggers on every Python file in the repository
- `backend/**` overlaps with `backend/topology/first_light/**` (CAL-EXP dependency)

### Checklist for Path Extensions

Before adding a new path to `pilot-phase-gate.yml`:

- [ ] Pattern uses explicit directory name (not `**/` prefix)
- [ ] Pattern cannot match any file in `FROZEN_HARNESSES`
- [ ] Pattern cannot match `backend/topology/first_light/**`
- [ ] Pattern cannot match `results/cal_exp_*`
- [ ] If documentation, uses `docs/pilot/**` subdirectory scoping
- [ ] If scripts, uses `pilot_` prefix pattern

---

## Related Documents

- `CRITICAL_FILES_MANIFEST.md` - Lists all frozen CAL-EXP files
- `.github/workflows/cal_exp_hygiene_gate.yml` - CAL-EXP CI gate (separate scope)
