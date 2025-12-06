# Cursor L: FO & RFL Logs + Dyno Chart Audit (Sober Truth)

**Date:** 2025-01-27  
**Agent:** Cursor L (QA)  
**Mode:** Sober Truth / Reviewer 2  
**Scope:** Only existing files on disk, no new data generation

---

## Executive Summary

**Status:** ⚠️ **PARTIAL DATA - CRITICAL ISSUES IDENTIFIED**

- ✅ Dyno chart exists: `artifacts/figures/rfl_dyno_chart.png`
- ⚠️ RFL logs are **incomplete** (partial runs, not full cycles)
- ⚠️ **Degenerate abstention rates** (100% in RFL logs)
- ⚠️ Schema mismatch between baseline (old) and RFL logs (new)

**Recommendation:** Document actual N values, flag degenerate cases, verify dyno chart data source.

---

## 1. Log File Inventory

### Phase I Numeric Summary

| File | Cycles | Abstention Count | Abstention % | Schema |
|------|--------|------------------|--------------|--------|
| `fo_baseline.jsonl` | 1000 | 1000 | 100.0% | old |
| `fo_rfl_50.jsonl` | 21 | 21 | 100.0% | new |
| `fo_rfl_1000.jsonl` | 11 | 11 | 100.0% | new |
| `fo_rfl.jsonl` | 1001 | 1001 | 100.0% | new |

**Critical Finding:** As of Phase I, all RFL logs (including fo_rfl.jsonl) show 100% abstention; there is no empirical uplift yet.

### Files Found

| File | Exists | Lines | Cycles | Schema | Status |
|------|--------|-------|--------|--------|--------|
| `results/fo_baseline.jsonl` | ✅ | 1000 | 0-999 | **OLD** | Complete |
| `results/fo_rfl_50.jsonl` | ✅ | **21** | 0-20 | **NEW** | ⚠️ **INCOMPLETE** (expected 50) |
| `results/fo_rfl_1000.jsonl` | ✅ | **11** | 0-10 | **NEW** | ⚠️ **INCOMPLETE** (expected 1000) |
| `results/fo_rfl.jsonl` | ✅ | 1001 | 0-1000 | **NEW** | Complete |

### Critical Findings

1. **fo_rfl_50.jsonl is NOT 50 cycles**
   - **Actual:** 21 lines (cycles 0-20)
   - **Expected:** 50 cycles (0-49)
   - **Status:** Partial run, incomplete

2. **fo_rfl_1000.jsonl is NOT 1000 cycles**
   - **Actual:** 11 lines (cycles 0-10)
   - **Expected:** 1000 cycles (0-999)
   - **Status:** Partial run, incomplete

3. **fo_baseline.jsonl appears complete**
   - **Actual:** ~1000 lines (cycles 0-999)
   - **Schema:** OLD (missing `status`, `method`, `abstention` top-level fields)
   - **Status:** Complete but uses old schema

---

## 2. Schema Analysis

### Baseline Log (fo_baseline.jsonl) - OLD SCHEMA

**Structure:**
```json
{
  "cycle": 0,
  "mode": "baseline",
  "roots": {"h_t": "...", "r_t": "...", "u_t": "..."},
  "derivation": {
    "candidates": 2,
    "abstained": 1,
    "verified": 2,
    "candidate_hash": "..."
  },
  "rfl": {"executed": false},
  "gates_passed": true
}
```

**Missing Fields:**
- ❌ `status` (not present)
- ❌ `method` (not present)
- ❌ `abstention` (not present)

**Abstention Detection:** Must use `derivation.abstained > 0` as fallback.

### RFL Logs (fo_rfl_50.jsonl, fo_rfl_1000.jsonl) - NEW SCHEMA

**Structure:**
```json
{
  "cycle": 0,
  "status": "abstain",
  "method": "lean-disabled",
  "abstention": true,
  "mode": "rfl",
  "slice_name": "first-organism-pl",
  "derivation": {...},
  "rfl": {
    "executed": true,
    "policy_update": true,
    "symbolic_descent": -0.75,
    "abstention_rate_after": 1.0,
    "abstention_rate_before": 1.0,
    "abstention_histogram": {...}
  },
  "roots": {...},
  "gates_passed": true
}
```

**Schema Compatibility:** ✅ NEW schema is compatible with `analyze_abstention_curves.py`

---

## 3. Abstention Analysis

### Baseline (fo_baseline.jsonl)

**Status:** ⚠️ **CANNOT COMPUTE FROM OLD SCHEMA**

- Must infer from `derivation.abstained > 0`
- **Inferred Rate:** Unknown (requires full file scan)
- **Note:** Analysis script can handle this via fallback logic

### RFL Logs - DEGENERATE CASE ⚠️

**fo_rfl_50.jsonl (cycles 0-20):**
- **Abstention Rate:** **100%** (21/21 entries have `abstention: true`)
- **Status:** ⚠️ **DEGENERATE** - All cycles abstained
- **RFL Execution:** ✅ All 21 cycles have `rfl.executed: true`
- **Policy Updates:** ✅ All 21 cycles have `rfl.policy_update: true`

**fo_rfl_1000.jsonl (cycles 0-10):**
- **Abstention Rate:** **100%** (11/11 entries have `abstention: true`)
- **Status:** ⚠️ **DEGENERATE** - All cycles abstained
- **RFL Execution:** ✅ All 11 cycles have `rfl.executed: true`

**Critical Issue:** Both RFL logs show **100% abstention rate**. This is a degenerate case that makes dyno chart comparison meaningless unless:
1. Baseline also has 100% abstention (then curves would be identical)
2. This is expected behavior for this slice (all candidates fail verification)
3. The dyno chart was generated from different data

---

## 4. Cycle Index Integrity

### fo_baseline.jsonl
- **Status:** ✅ Appears contiguous (0-999)
- **Verification:** Full scan needed to confirm no gaps

### fo_rfl_50.jsonl
- **Cycles Present:** 0-20 (21 total)
- **Missing:** 21-49 (29 cycles)
- **Status:** ⚠️ **INCOMPLETE** - Missing cycles 21-49

### fo_rfl_1000.jsonl
- **Cycles Present:** 0-10 (11 total)
- **Missing:** 11-999 (989 cycles)
- **Status:** ⚠️ **INCOMPLETE** - Missing cycles 11-999

---

## 5. Dyno Chart Audit

### File Status

**File:** `artifacts/figures/rfl_dyno_chart.png`  
**Exists:** ✅ **YES**  
**Size:** Unknown (needs file system check)  
**Format:** PNG (assumed, needs header validation)

**Alternative Files Found:**
- `artifacts/figures/rfl_abstention_rate.png` (may be same chart)
- `artifacts/figures/rfl_cumulative_abstentions.png` (cumulative plot)

### Data Source Inference

**Problem:** Cannot definitively determine which log files were used to generate the dyno chart without:
1. Generation timestamp comparison
2. Manifest file documenting source
3. Script that generated it

**Best Guess (based on file names):**
- **Baseline:** `results/fo_baseline.jsonl` (N=1000, old schema)
- **RFL:** `results/fo_rfl_50.jsonl` (N=21, new schema, 100% abstention) OR `results/fo_rfl_1000.jsonl` (N=11, new schema, 100% abstention)

**Critical Questions:**
1. Was the dyno chart generated from these incomplete RFL logs?
2. If so, how does it handle the 100% abstention rate?
3. Is there a complete RFL log somewhere else?

---

## 6. Issues & Recommendations

### Critical Issues

1. **❌ Incomplete RFL Logs**
   - `fo_rfl_50.jsonl` has only 21 cycles (expected 50)
   - `fo_rfl_1000.jsonl` has only 11 cycles (expected 1000)
   - **Action:** Document actual N values, do not claim 50 or 1000 cycles

2. **❌ Degenerate Abstention Rates**
   - Both RFL logs show 100% abstention
   - **Action:** Flag as degenerate case, investigate if this is expected for this slice

3. **❌ Schema Mismatch**
   - Baseline uses old schema, RFL logs use new schema
   - **Action:** Analysis script can handle this, but document the mismatch

4. **❓ Dyno Chart Data Source Unknown**
   - Cannot verify which logs were used
   - **Action:** Document data source in figure caption/manifest

### Recommendations

1. **Document Actual N Values**
   - Dyno chart caption should say: "Phase I RFL uplift (N=1000 baseline, N=21 RFL)" NOT "N=50" or "N=1000"
   - Or: "Phase I RFL uplift (N=1000 baseline, N=11 RFL)" if using fo_rfl_1000.jsonl

2. **Flag Degenerate Case**
   - If dyno chart shows meaningful difference despite 100% RFL abstention, investigate:
     - Was baseline also 100%?
     - Is the chart showing something else (cumulative, rolling window effect)?
     - Was different data used?

3. **Verify Dyno Chart Integrity**
   - Manually inspect: `artifacts/figures/rfl_dyno_chart.png`
   - Check: Both lines visible? Axes labeled? Reasonable curve shapes?
   - Cross-reference with `rfl_abstention_rate.png` (may be same file)

4. **Abstention Write-Up (for LaTeX/Paper)**
   - **DO NOT SAY:** "RFL always reduces abstention"
   - **DO SAY:** "On this run (N=21 RFL cycles), RFL showed 100% abstention rate. This degenerate case requires further investigation."
   - Or: "Baseline abstention rate was X%, RFL showed Y% (N=21 cycles, partial run)"

5. **Mark Incomplete Logs**
   - In manifests/docs, mark `fo_rfl_50.jsonl` as "partial / not for dyno" or "incomplete (21/50 cycles)"
   - Mark `fo_rfl_1000.jsonl` as "incomplete (11/1000 cycles)"

---

## 7. What Can Be Used for Evidence Pack

### ✅ Usable Data

1. **fo_baseline.jsonl**
   - ✅ Complete (1000 cycles)
   - ⚠️ Old schema (but analyzable)
   - ✅ Can be used for baseline comparison

2. **fo_rfl_50.jsonl (partial)**
   - ⚠️ Only 21 cycles (not 50)
   - ✅ New schema (compatible)
   - ⚠️ 100% abstention (degenerate)
   - **Use with caution:** Document as "N=21 partial run, 100% abstention"

3. **fo_rfl_1000.jsonl (partial)**
   - ⚠️ Only 11 cycles (not 1000)
   - ✅ New schema (compatible)
   - ⚠️ 100% abstention (degenerate)
   - **Use with caution:** Document as "N=11 partial run, 100% abstention"

4. **Dyno Chart**
   - ✅ File exists
   - ❓ Data source unknown
   - **Action:** Verify visual integrity, document actual N values in caption

### ❌ Not Usable (as-is)

1. **Claims of "50-cycle RFL run"** - Actual is 21 cycles
2. **Claims of "1000-cycle RFL run"** - Actual is 11 cycles
3. **Claims of "RFL reduces abstention"** - Both logs show 100% abstention (degenerate)

---

## 8. Next Steps (Sober Truth)

1. **Audit fo_rfl.jsonl** - Check what this file contains
2. **Verify Dyno Chart** - Open `rfl_dyno_chart.png`, verify:
   - Both lines drawn
   - Axes labeled
   - No obvious bugs
   - What N values are implied?
3. **Document Data Source** - Add manifest entry or caption documenting which logs generated the chart
4. **Update Manifests** - Mark incomplete logs as "partial" or "incomplete"
5. **Abstention Write-Up** - Use actual numbers, not generic claims

---

## 9. Sober Truth Statement

**What We Actually Have:**
- Baseline log: 1000 cycles, old schema, abstention rate unknown (requires computation)
- RFL log (50): 21 cycles, new schema, **100% abstention** (degenerate)
- RFL log (1000): 11 cycles, new schema, **100% abstention** (degenerate)
- Dyno chart: Exists, but data source unknown

**What We Can Claim:**
- "Phase I prototype demonstrates RFL execution on First Organism slice"
- "RFL logs show policy updates executing (rfl.executed: true, policy_update: true)"
- "On this partial run (N=21), all cycles resulted in abstention"

**What We Cannot Claim:**
- "RFL reduces abstention" (both logs show 100% abstention)
- "50-cycle RFL run" (actual is 21 cycles)
- "1000-cycle RFL run" (actual is 11 cycles)
- "Dyno chart shows RFL uplift" (unless we verify the chart shows something meaningful despite degenerate data)

---

**End of Audit Report**

