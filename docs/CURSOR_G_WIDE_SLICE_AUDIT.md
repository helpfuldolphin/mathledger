# Cursor G — Wide Slice Curriculum Audit (Sober Truth / Reviewer 2 Mode)

**Date:** 2025-01-27  
**Agent:** Cursor G (Curriculum Architect / Gates Specialist)  
**Mode:** Sober Truth / Evidence Pack v1  
**Objective:** Audit Wide Slice configuration and documentation against actual evidence

---

## Executive Summary

**Status:** ✅ **CONFIGURATION VALID** | ⚠️ **NO EXPERIMENTAL DATA EXISTS**

The Wide Slice (`slice_medium`) is **configured** in `config/curriculum.yaml` with appropriate gate thresholds for RFL uplift experiments. However, **no Wide Slice experiments have been run yet** — all existing FO cycle logs use `slice_name: "first-organism-pl"`.

**Key Findings:**
- ✅ Curriculum configuration is valid and properly annotated
- ✅ Test suite validates configuration (not data existence)
- ❌ Documentation inconsistencies: some docs reference non-existent Wide Slice log files
- ⚠️ No empirical validation of gate thresholds has been performed

---

## 1. Configuration Audit

### 1.1 Curriculum YAML (`config/curriculum.yaml`)

**Location:** Lines 125-149  
**Status:** ✅ **VALID**

**Current Configuration:**
```yaml
# NOTE: slice_medium is designated as the "Wide Slice" for RFL uplift experiments (Operation Dyno Chart).
# This slice produces non-trivial abstention (5-20%) and does not saturate trivially,
# making it statistically interesting for measuring RFL uplift across 1000-cycle runs.
- name: slice_medium
  params:
    atoms: 5
    depth_max: 7
    breadth_max: 1500
    total_max: 8000
  gates:
    coverage:
      ci_lower_min: 0.85
      sample_min: 20
      require_attestation: true
    abstention:
      max_rate_pct: 15.0
      max_mass: 800
    velocity:
      min_pph: 150
      stability_cv_max: 0.12
      window_minutes: 60
    caps:
      min_attempt_mass: 3000
      min_runtime_minutes: 20
      backlog_max: 0.40
```

**Validation:**
- ✅ Designation comment present and accurate
- ✅ Parameters meet Wide Slice thresholds (atoms≥5, depth_max≥7, total_max≥8000)
- ✅ Gate thresholds configured for non-trivial abstention (5-20% range)
- ✅ All required gates present (coverage, abstention, velocity, caps)

**Documentation Accuracy:** ✅ Correctly states this is for "RFL uplift experiments" (future), not claiming completion.

---

### 1.2 Test Suite (`tests/frontier/test_curriculum_slices.py`)

**Status:** ✅ **VALID — No False Claims**

**Test Coverage:**
- Tests configuration existence and thresholds
- Validates gate compatibility with `NormalizedMetrics.from_raw()`
- Checks slice ordering and monotonicity

**Critical Review:**
- ✅ Tests do NOT assert existence of experimental data
- ✅ Test docstring mentions "planned 1000-cycle runs" (accurate)
- ✅ Only validates configuration structure, not experimental results

**Issue Found & Fixed:**
- Line 55: Updated docstring to clarify "planned 1000-cycle runs" rather than implying they exist
- Added note: "This validates configuration only; no Wide Slice experiments have been run yet."

---

## 2. Evidence Audit

### 2.1 Existing FO Cycle Logs (`results/`)

**Files Found:**
- `fo_baseline.jsonl`
- `fo_rfl.jsonl`
- `fo_rfl_50.jsonl`
- `fo_rfl_1000.jsonl`

**Slice Usage:**
- All logs checked use `slice_name: "first-organism-pl"`
- ❌ **NO logs found with `slice_name: "slice_medium"`**

**Conclusion:** Wide Slice experiments have **NOT been run**. All existing data is from First Organism slice experiments.

---

### 2.2 Expected but Missing Files

**Documented but Missing:**
- `results/fo_baseline_wide.jsonl` — Referenced in multiple docs, does not exist
- `results/fo_rfl_wide.jsonl` — Referenced in multiple docs, does not exist
- `artifacts/figures/rfl_dyno_chart.png` — Expected output, does not exist

**Documentation Status:**
- `experiments/DYNO_CHART_QA_REPORT.md` — ✅ Correctly acknowledges files are missing
- `experiments/DYNO_CHART_QA_SUMMARY.md` — ✅ Correctly acknowledges files are missing
- `experiments/FIGURES_CATALOG.md` — ❌ References files without marking as "planned"

**Fix Applied:**
- Updated `FIGURES_CATALOG.md` line 76 to add note: `[NOTE: Planned / Not yet generated — see experiments/DYNO_CHART_QA_SUMMARY.md]`

---

## 3. Documentation Inconsistencies

### 3.1 False Claims of Completion

**Status:** ⚠️ **MINOR ISSUES FOUND AND FIXED**

1. **`experiments/FIGURES_CATALOG.md`** (Line 76)
   - **Issue:** References `fo_baseline_wide.jsonl` and `fo_rfl_wide.jsonl` as data sources without indicating they don't exist
   - **Fix Applied:** Added note marking as "Planned / Not yet generated"
   - **Severity:** Medium (misleading but not claiming completion)

2. **Test Documentation** (Line 55)
   - **Issue:** Mentioned "1000-cycle runs" without clarifying they're planned
   - **Fix Applied:** Updated to "planned 1000-cycle runs" with clarifying note
   - **Severity:** Low (context makes intent clear)

### 3.2 Accurate Documentation

**These documents correctly handle Wide Slice status:**

- ✅ `experiments/DYNO_CHART_QA_REPORT.md` — Explicitly states files are missing
- ✅ `experiments/DYNO_CHART_QA_SUMMARY.md` — Clear status: "DATA FILES MISSING"
- ✅ `config/curriculum.yaml` — Designation comment uses future tense ("for RFL uplift experiments")
- ✅ `experiments/run_fo_cycles.py` — Usage examples correctly show how to run (not claiming completion)

---

## 4. Gate Configuration Analysis

### 4.1 Threshold Rationale

**Coverage Gate:**
- `ci_lower_min: 0.85` — More lenient than strict slices (0.92-0.93), appropriate for wide exploration
- `sample_min: 20` — Sufficient for CI estimation

**Abstention Gate:**
- `max_rate_pct: 15.0` — Allows 5-20% range for statistical interest (non-trivial but not excessive)
- `max_mass: 800` — With `total_max: 8000`, allows ~10% abstention mass

**Velocity Gate:**
- `min_pph: 150` — More lenient than strict slices (180-220), appropriate for wider search
- `stability_cv_max: 0.12` — Allows reasonable variance

**Caps Gate:**
- `min_attempt_mass: 3000` — Ensures ~37.5% coverage (3000/8000) for statistical significance
- `min_runtime_minutes: 20` — Prevents premature slice completion

**Assessment:** ✅ Thresholds are theoretically sound but **NOT empirically validated**. Actual abstention rates, coverage attainment, and velocity profiles are unknown until experiments are run.

---

## 5. Recommendations

### 5.1 Immediate Actions (Evidence Pack v1)

1. ✅ **DONE:** Mark `FIGURES_CATALOG.md` Wide Slice references as "planned"
2. ✅ **DONE:** Clarify test documentation to indicate configuration-only validation
3. ⚠️ **RECOMMENDED:** Add explicit "Planned / Not Run" status to curriculum slice metadata (future enhancement)

### 5.2 Future Validation (Phase II)

1. **Run Baseline Wide Slice Experiment:**
   ```bash
   uv run python experiments/run_fo_cycles.py \
     --mode=baseline --cycles=1000 \
     --slice-name=slice_medium --system=pl \
     --out=results/fo_baseline_wide.jsonl
   ```

2. **Run RFL Wide Slice Experiment:**
   ```bash
   uv run python experiments/run_fo_cycles.py \
     --mode=rfl --cycles=1000 \
     --slice-name=slice_medium --system=pl \
     --out=results/fo_rfl_wide.jsonl
   ```

3. **Validate Gate Thresholds:**
   - Measure actual abstention rates (target: 5-20%)
   - Verify coverage CI attainment (target: ≥0.85)
   - Assess proof velocity (target: ≥150 pph)
   - Adjust thresholds if empirical data indicates mis-calibration

4. **Generate Dyno Chart:**
   - Run analysis script on generated logs
   - Validate visual integrity
   - Update `FIGURES_CATALOG.md` status to "Available"

---

## 6. Sober Truth Summary

**What EXISTS (Evidence Pack v1):**
- ✅ Curriculum configuration for Wide Slice (`slice_medium`)
- ✅ Test suite validating configuration structure
- ✅ Documentation infrastructure (QA scripts, analysis tools)
- ✅ First Organism experiments using `first-organism-pl` slice

**What DOES NOT EXIST:**
- ❌ Wide Slice experimental data (`fo_*_wide.jsonl` files)
- ❌ Dyno chart visualization
- ❌ Empirical validation of gate thresholds for Wide Slice
- ❌ Evidence of RFL uplift on Wide Slice configuration

**What is CLAIMED vs. REAL:**
- ✅ Configuration: **REAL** — `slice_medium` exists and is properly configured
- ✅ Designation: **REAL** — Correctly marked as "for RFL uplift experiments"
- ⚠️ Experimental Results: **PLANNED** — No data exists yet; documentation now correctly reflects this

---

## 7. Compliance with Master Directive

**Section 1 Compliance:** ✅ **VERIFIED**
- All claims traced to concrete file paths
- No hypothetical results presented as evidence
- Configuration validated against actual YAML

**Section 2 Compliance:** ✅ **VERIFIED**
- No claims about Wide Slice experimental results
- Documentation fixed to mark as "planned" where referenced
- Tests only validate configuration, not data existence

**Section 3 Compliance:** ✅ **VERIFIED**
- Tests focus on configuration validation (hermetic checks)
- No assertions about experimental data
- Documentation inconsistencies identified and fixed

**Section 4 Compliance:** ✅ **VERIFIED**
- Conservative template used: "configured for" not "used in"
- All file references validated
- Missing data clearly documented

---

## 8. Artifacts

**Files Modified:**
1. `config/curriculum.yaml` — Wide Slice designation comment (from previous work)
2. `tests/frontier/test_curriculum_slices.py` — Clarified "planned" status in test docstring
3. `experiments/FIGURES_CATALOG.md` — Added "Planned / Not yet generated" note

**Files Created:**
1. `docs/CURSOR_G_WIDE_SLICE_AUDIT.md` — This audit report

**Files Validated (No Changes Needed):**
1. `experiments/DYNO_CHART_QA_REPORT.md` — Correctly states missing files
2. `experiments/DYNO_CHART_QA_SUMMARY.md` — Correctly states missing files
3. `experiments/run_fo_cycles.py` — Correctly shows usage examples

---

## 9. Conclusion

The Wide Slice curriculum configuration is **valid and ready for experiments**, but **no experimental data exists yet**. Documentation inconsistencies have been corrected to accurately reflect this state. The infrastructure is in place (configuration, tests, analysis tools), but empirical validation must await actual experiment runs.

**Status for Evidence Pack v1:**
- ✅ Configuration: **LOCKED IN**
- ⚠️ Experiments: **PLANNED / NOT RUN**
- ✅ Documentation: **CORRECTED** (now accurately reflects reality)

---

## 10. Post-Audit Verification (2025-01-27 Follow-Up)

**Status:** ✅ **CONFIRMED — NO MISCHARACTERIZATION FOUND**

Verified that `fo_rfl.jsonl` (the new FO RFL run) correctly uses `slice_name: "first-organism-pl"` and is **NOT** a Wide Slice experiment. All documentation correctly separates:
- Phase I RFL runs (`fo_rfl.jsonl`, `fo_rfl_50.jsonl`) = `first-organism-pl` slice
- Wide Slice (`slice_medium`) = Phase II / planned, no data exists

**Fixed Issues:**
- Updated `ops/CURSOR_I_EVIDENCE_PACK_AUDIT.md` table to clarify that `fo_rfl.jsonl` uses `first-organism-pl` slice (NOT Wide Slice)
- Added explicit note that Wide Slice experiments have not been run yet

**Conclusion:** The new FO RFL run (`fo_rfl.jsonl`) remains a `first-organism-pl` slice experiment; Wide Slice (`slice_medium`) still has no data.

---

**Audit Completed By:** Cursor G (Curriculum Architect)  
**Mode:** Sober Truth / Reviewer 2  
**Date:** 2025-01-27  
**Last Updated:** 2025-01-27 (post-audit verification)

