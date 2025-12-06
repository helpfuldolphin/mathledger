# Cursor D: Test Gating & Wide Slice Audit (Sober Truth v1)

**Date**: 2025-01-18  
**Agent**: Cursor D (Test Gating + Wide-Slice Markers)  
**Mode**: Sober Truth / Reviewer 2

## Executive Summary

This audit verifies that:
1. SPARK (First Organism) test gating works correctly
2. Wide Slice tests validate actual log schemas (not hypothetical ones)
3. Tests don't assert existence of data that doesn't exist
4. Hermetic tests can run without DB/Redis
5. SPARK precheck scripts match actual PASS line format

## 1. What Actually Exists (Evidence-Based)

### 1.1 First Organism Attestation Artifact

**File**: `artifacts/first_organism/attestation.json`

**Status**: ✅ EXISTS, VALID

**Contents**:
- `H_t`: `01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2`
- `R_t`: `a8dc5b2c7778ce38f72e63ecc4b7a9b010969c018d3d7cafff12bf6d85400336`
- `U_t`: `8c11ea1e67666dd3f14a12cdf475a2d7f7c801037f3d273ccca069b1fa703359`
- `statement_hash`: `0c90faf28890f9bf1883806f0adbbc433f26f87a75849099ff1dec519aa00679`
- `chain_status`: `"ready"`
- `is_synthetic`: `false`

**Verification**: This file is produced by `test_first_organism_closed_loop_standalone` and is the canonical sealed attestation.

### 1.2 First Organism Log Files

**Files**:
- ✅ `results/fo_baseline.jsonl` - EXISTS (1001 lines, cycles 0-1000)
- ✅ `results/fo_rfl_50.jsonl` - EXISTS (11 lines, cycles 0-10)
- ✅ `results/fo_rfl_1000.jsonl` - EXISTS (size unknown, needs verification)
- ✅ `results/fo_rfl.jsonl` - EXISTS (334 lines, cycles 0-333, **partial RFL run, Phase I prototype**)

**Schema (from actual files)**:
```json
{
  "cycle": 0,
  "slice_name": "first-organism-pl",
  "status": "abstain",
  "method": "lean-disabled",
  "abstention": true,
  "mode": "rfl",
  "roots": {
    "h_t": "...",
    "r_t": "...",
    "u_t": "..."
  },
  "derivation": {
    "candidates": 2,
    "abstained": 1,
    "verified": 2,
    "candidate_hash": "..."
  },
  "rfl": {
    "executed": true,
    "policy_update": true,
    "symbolic_descent": -0.75,
    "abstention_rate_after": 1.0,
    "abstention_rate_before": 1.0
  },
  "gates_passed": true
}
```

**Key Fields**:
- `cycle` (int): Cycle number
- `slice_name` (str): Curriculum slice name
- `status` (str): "abstain", "verified", or "error"
- `method` (str): Verification method (e.g., "lean-disabled")
- `abstention` (bool): Whether cycle abstained (NOT `is_abstention`)
- `mode` (str): "baseline" or "rfl"

### 1.3 Wide Slice Log Files

**Files**:
- ❌ `results/fo_baseline_wide.jsonl` - DOES NOT EXIST
- ❌ `results/fo_rfl_wide.jsonl` - DOES NOT EXIST

**Status**: These files are **planned but not yet generated**. The `run_fo_cycles.py` script supports generating them with `--slice-name=slice_medium`, but they haven't been run yet.

**Action**: Wide Slice tests correctly skip with clear message when files don't exist.

### 1.4 SPARK Precheck Scripts

**File**: `scripts/run_spark_closed_loop.ps1`

**Status**: ✅ EXISTS, VALIDATES CORRECT FORMAT

**PASS Line Pattern**: `[PASS] FIRST ORGANISM ALIVE H_t=`

**Verification**: Matches the format emitted by `log_first_organism_pass()` in `tests/integration/conftest.py`:
```python
sys.stdout.write(f"{color}[PASS] FIRST ORGANISM ALIVE H_t={short_h_t}{reset}\n")
```

**Format**: `[PASS] FIRST ORGANISM ALIVE H_t=<12-char-hex>`

## 2. Test Gating Status

### 2.1 SPARK Gating (First Organism)

**Status**: ✅ WORKING

**Mechanism**:
- Tests marked `@pytest.mark.first_organism` are skipped unless:
  - `FIRST_ORGANISM_TESTS=true`, OR
  - `SPARK_RUN=1`, OR
  - `.spark_run_enable` file exists

**DB-Dependent Tests**:
- Use `first_organism_db` fixture
- Skip with `[SKIP] Database unavailable: ...` when Postgres/Redis are down
- Verified: `test_first_organism_closed_loop_happy_path` correctly skips when DB unavailable

**Hermetic Tests**:
- Marked `@pytest.mark.hermetic`
- Can run without DB/Redis
- Verified: `test_first_organism_closed_loop_standalone` runs without DB

### 2.2 WIDE_SLICE Gating

**Status**: ✅ WORKING

**Mechanism**:
- Tests marked `@pytest.mark.wide_slice` are skipped unless:
  - `WIDE_SLICE_TESTS=true`, OR
  - `-m wide_slice` is used

**Behavior**:
- Tests skip gracefully when `fo_baseline_wide.jsonl` or `fo_rfl_wide.jsonl` don't exist
- Skip message: `[SKIP] Wide Slice logs not found; run run_wide_slice_experiments.ps1 first.`

## 3. Schema Validation Fixes

### 3.1 Wide Slice Test Schema Alignment

**Issue**: Initial Wide Slice tests expected `is_abstention` and `verification_method`, but actual schema uses `abstention` and `method`.

**Fix**: Updated `_validate_log_record_shape()` to:
- Accept `abstention` (bool) OR `is_abstention` (bool) for compatibility
- Require `method` (str) - not `verification_method`
- Match the actual schema from `run_fo_cycles.py`

**Verification**: Tests now validate against the actual schema produced by `experiments/run_fo_cycles.py`.

## 4. Hermetic Test Verification

### 4.1 Test: `test_first_organism_closed_loop_standalone`

**Status**: ✅ VERIFIED HERMETIC

**Marker**: `@pytest.mark.hermetic`

**Behavior**:
- Runs without DB/Redis
- Produces sealed attestation `artifacts/first_organism/attestation.json`
- Emits `[PASS] FIRST ORGANISM ALIVE H_t=...` line
- No external dependencies

**Verification Command**:
```bash
FIRST_ORGANISM_TESTS=true pytest tests/integration/test_first_organism.py::test_first_organism_closed_loop_standalone -v
```

### 4.2 Test: `test_first_organism_closed_loop_smoke`

**Status**: ✅ VERIFIED HERMETIC

**Marker**: `@pytest.mark.hermetic`

**Behavior**:
- Runs without DB/Redis
- Mocks `LedgerIngestor.ingest` to skip DB writes
- Validates logical FO + RFL pipeline wiring

## 5. What's NOT Claimed (Sober Truth)

### 5.1 Wide Slice Experiments

**Status**: ❌ NOT YET RUN

**Files**: `fo_baseline_wide.jsonl`, `fo_rfl_wide.jsonl` do not exist.

**Action**: Tests correctly skip with clear message. No false assertions.

### 5.2 RFL Runs (Partial, Phase I Prototype)

**Status**: ✅ PARTIAL RUNS EXIST, TREATED AS PHASE I PROTOTYPE

**Files**:
- `fo_rfl_50.jsonl`: 11 lines (cycles 0-10) - small sanity run
- `fo_rfl.jsonl`: 334 lines (cycles 0-333) - **partial RFL run, all abstain**
- `fo_rfl_1000.jsonl`: exists but needs verification (line count, cycle range unknown)

**Key Observations**:
- `fo_rfl.jsonl` is a **real but partial Phase I run** (~330 cycles)
- All cycles in `fo_rfl.jsonl` have `abstention: true` and `status: "abstain"` (degenerate all-abstain case)
- This is used for **plumbing validation** (RFL runner executes, policy updates apply, histograms track)
- **NOT used to claim RFL reduces abstention** - this is a degenerate case
- **NOT used to claim 1000-cycle completion** - it's a partial run

**Action**: Tests treat `fo_rfl.jsonl` as:
- A real Phase I artifact (exists, valid schema)
- A partial run (not 1000 cycles)
- A degenerate all-abstain case (not evidence of abstention reduction)
- Useful for validating RFL plumbing, not for empirical claims about RFL effectiveness

### 5.3 Dyno Chart Figures

**Status**: ⚠️ NEEDS VERIFICATION

**Expected**: `artifacts/figures/rfl_abstention_rate.png` or `rfl_dyno_chart.png`

**Action**: Verify file exists and is non-zero size before claiming "Dyno Chart available".

## 6. Recommendations

### 6.1 Immediate Actions

1. ✅ **DONE**: Fix Wide Slice test schema to match actual `run_fo_cycles.py` output
2. ✅ **DONE**: Ensure tests skip gracefully when data doesn't exist
3. ⚠️ **TODO**: Verify `fo_rfl_1000.jsonl` completeness (line count, cycle range)
4. ⚠️ **TODO**: Verify Dyno Chart figure exists and is non-zero size

### 6.2 Documentation Updates

1. ✅ **DONE**: Document SPARK vs WIDE_SLICE marker distinction
2. ⚠️ **TODO**: Document actual log file schemas in `docs/` (not just in code)
3. ⚠️ **TODO**: Create manifest of what experiments have actually been run vs planned

### 6.3 Test Coverage

1. ✅ **DONE**: Hermetic tests verified to run without DB
2. ✅ **DONE**: DB-dependent tests verified to skip gracefully
3. ⚠️ **TODO**: Add test to verify `fo_rfl_1000.jsonl` schema matches expected format

## 7. Evidence Pack Compliance

### 7.1 Phase I Evidence (What IS Real)

✅ **First Organism Attestation**: `artifacts/first_organism/attestation.json` - EXISTS, VALID  
✅ **FO Closed-Loop Test**: `test_first_organism_closed_loop_happy_path` - EXISTS, PASSES  
✅ **FO Baseline Cycles**: `results/fo_baseline.jsonl` - EXISTS, 1001 lines  
✅ **RFL Small Run**: `results/fo_rfl_50.jsonl` - EXISTS, 11 lines  

### 7.2 Phase I Evidence (Partial / Prototype)

✅ **RFL Partial Run**: `results/fo_rfl.jsonl` - EXISTS, 334 lines (cycles 0-333), all abstain, **partial RFL run / Phase I prototype**  
⚠️ **RFL 1000-Cycle Run**: `results/fo_rfl_1000.jsonl` - EXISTS but needs verification (line count, cycle range)  
⚠️ **Dyno Chart**: Figure file - needs verification of existence and non-zero size  

### 7.3 Phase II / Planned (NOT Claimed)

❌ **Wide Slice Experiments**: Not yet run  
❌ **ΔH Scaling Runs**: Not yet run  
❌ **Imperfect Verifier Experiments**: Not yet run  

## 8. Conclusion

**Status**: ✅ TEST GATING IS SOBER AND EVIDENCE-BASED

- SPARK gating works correctly
- Wide Slice tests validate actual schemas (after fix)
- Tests don't assert non-existent data
- Hermetic tests verified to run without DB
- SPARK precheck scripts match actual PASS line format

**Remaining Work**:
- Verify `fo_rfl_1000.jsonl` completeness
- Verify Dyno Chart figure exists
- Document actual log schemas in `docs/`
- Create manifest of run vs planned experiments

## 9. How Tests Treat fo_rfl.jsonl (Schema Alignment Update)

### 9.1 Current Status

**File**: `results/fo_rfl.jsonl`

**Reality**:
- ✅ EXISTS, non-empty (334 lines)
- ✅ Cycles 0-333 (partial run, not 1000 cycles)
- ✅ All cycles have `abstention: true`, `status: "abstain"` (degenerate all-abstain case)
- ✅ Valid schema matching `run_fo_cycles.py` output
- ✅ RFL plumbing works (policy updates, histograms, symbolic descent tracked)

**Test Treatment**:
- Tests **do NOT assume** `fo_rfl.jsonl` is empty
- Tests **do NOT assert** there are 1000 RFL cycles
- Tests **do NOT claim** RFL reduces abstention (this is a degenerate all-abstain case)
- Tests **treat it as**:
  - A real Phase I artifact (exists, valid)
  - A partial RFL run (~330 cycles)
  - A degenerate case used for plumbing validation
  - Evidence that RFL runner executes correctly, not evidence of RFL effectiveness

### 9.2 Docstring Updates

**Updated**:
- `tests/frontier/test_curriculum_slices.py`: Changed "planned 1000-cycle runs" → "partial RFL runs / Phase I prototype"
- `tests/test_first_organism_harness_v2.py`: Changed "1000-cycle Dyno Chart data" → "partial RFL run data, Phase I prototype"

**Already Correct**:
- `tests/rfl/test_runner_first_organism.py`: Already states "No claims about 1000-cycle runs or non-existent data files"

### 9.3 Test Assertions

**No tests assert**:
- ❌ "1000-cycle RFL run complete"
- ❌ "RFL reduces abstention" (based on fo_rfl.jsonl)
- ❌ "fo_rfl.jsonl is empty"

**Tests correctly**:
- ✅ Skip gracefully when files don't exist
- ✅ Validate schema of existing files
- ✅ Treat partial runs as partial (not complete)
- ✅ Recognize degenerate cases (all-abstain) as plumbing validation, not effectiveness evidence

