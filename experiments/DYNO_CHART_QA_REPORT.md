# Dyno Chart QA Cross-Check Report

**Date:** 2025-01-27  
**QA Engineer:** Cursor L  
**Objective:** Verify Dyno Chart data integrity and cross-check with analysis tools

---

## Executive Summary

**Status:** ⚠️ **PARTIAL - Data Files Missing**

The wide slice log files (`fo_baseline_wide.jsonl` and `fo_rfl_wide.jsonl`) are not present in the repository. However, the infrastructure for validation and analysis is in place. This report documents the expected structure and validation criteria.

---

## 1. Log Sanity Checks

### Expected File Locations
- **Baseline Wide Slice:** `results/fo_baseline_wide.jsonl`
- **RFL Wide Slice:** `results/fo_rfl_wide.jsonl`

### Expected Structure (from `run_fo_cycles.py`)

Each JSONL entry should have the following structure:

```json
{
  "cycle": 0,
  "slice_name": "slice_medium",
  "status": "abstain" | "verified" | "error",
  "method": "lean-disabled" | "lean-verified" | "none",
  "abstention": true | false,
  "mode": "baseline" | "rfl",
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
    "executed": true | false,
    "policy_update": true | false,
    "symbolic_descent": {...},
    "abstention_histogram": {...}
  },
  "gates_passed": true | false
}
```

### Validation Criteria

#### ✓ Cycle Index Integrity
- **Requirement:** All cycles from `0` to `N-1` must be present (no gaps)
- **Check:** Verify `cycle` field exists and forms a complete sequence
- **Status:** ⚠️ Cannot verify (files missing)

#### ✓ Abstention Distribution
- **Requirement:** Abstention rate should be reasonable (not 0%, not 100%)
- **Expected Range:** 5% - 95% (typical range: 10% - 50%)
- **Check:** Compute `sum(abstention) / len(entries)`
- **Status:** ⚠️ Cannot verify (files missing)

#### ✓ No Degenerate Cases
- **Requirement:** No obvious artifacts (e.g., 100% abstention always, identical hashes for all cycles)
- **Check:** Verify variation in `status`, `method`, and `candidate_hash`
- **Status:** ⚠️ Cannot verify (files missing)

### Current Baseline Log Analysis (`fo_baseline.jsonl`)

**Note:** The existing `fo_baseline.jsonl` (default slice) has a **schema mismatch**:

- ❌ Missing top-level `status` field
- ❌ Missing top-level `method` field  
- ❌ Missing top-level `abstention` field
- ✓ Has `cycle` field
- ✓ Has `derivation.abstained` field (can be used as fallback)

**Action Required:** The wide slice logs should be generated with the updated `run_fo_cycles.py` that includes these fields (lines 298-300).

---

## 2. Schema Compatibility with Analysis Script

### Analysis Script: `experiments/analyze_abstention_curves.py`

The analysis script expects the following fields for abstention detection:

#### Primary Detection Method
1. **`status == "abstain"`** (case-insensitive)
2. **Fallback:** `method == "lean-disabled"` or `verification_method == "lean-disabled"`

#### Secondary Detection (if primary fails)
- Uses `abstention` boolean field if available
- Falls back to `derivation.abstained > 0` if no other indicators

### Schema Mapping

| Analysis Script Expects | `run_fo_cycles.py` Outputs | Compatibility |
|------------------------|----------------------------|---------------|
| `cycle` | `cycle` | ✅ Direct match |
| `status` | `status` | ✅ Direct match |
| `method` or `verification_method` | `method` | ✅ Direct match |
| `is_abstention` (computed) | `abstention` | ✅ Can use directly |

### Compatibility Status

**✅ SCHEMA COMPATIBLE** (when wide slice logs are generated with current `run_fo_cycles.py`)

The analysis script's `is_abstention()` function (lines 35-55) will work correctly with the expected log structure.

---

## 3. Analysis Script Cross-Check

### Expected Workflow

1. **Load Logs:** `load_logs()` reads JSONL and creates DataFrame
2. **Compute Metrics:** `compute_metrics()` calculates:
   - `abstention_rate_rolling`: Rolling mean with window size (default: 100)
   - `cumulative_abstentions`: Cumulative sum of abstentions
3. **Generate Plots:** 
   - `plot_abstention_rate()`: Creates rolling abstention rate curve
   - `plot_cumulative_abstentions()`: Creates cumulative abstention curve

### Verification Steps

**To verify once logs exist:**

```bash
# Run analysis script
uv run python experiments/analyze_abstention_curves.py \
  --baseline results/fo_baseline_wide.jsonl \
  --rfl results/fo_rfl_wide.jsonl \
  --window-size 100 \
  --burn-in 200
```

**Expected Outputs:**
- `artifacts/figures/rfl_abstention_rate.png` (rolling rate)
- `artifacts/figures/rfl_cumulative_abstentions.png` (cumulative)

**Status:** ⚠️ Cannot verify (input files missing)

---

## 4. Dyno Chart Integrity

### Expected Location
- **File:** `artifacts/figures/rfl_dyno_chart.png`

### Verification Checklist

#### Visual Inspection Required
- [ ] **Both lines drawn:** Baseline and RFL curves visible
- [ ] **Axes labeled:** X-axis (Cycle Index), Y-axis (Abstention Rate)
- [ ] **Legend present:** Labels for "Baseline (No RFL)" and "RFL Enabled"
- [ ] **No obvious bugs:** 
  - No swapped labels
  - No empty/missing lines
  - No overlapping or identical curves
  - Reasonable curve shapes (not flat lines, not noise)

#### File Integrity Checks
- [ ] **File exists:** `artifacts/figures/rfl_dyno_chart.png`
- [ ] **Non-zero size:** File size > 1KB (typical PNG: 50-200KB)
- [ ] **Valid PNG:** Header check passes (`\x89PNG\r\n\x1a\n`)

### Current Status

**❌ FILE NOT FOUND**

The dyno chart is not present at the expected location. It should be generated by:
1. Running the wide slice experiments to create log files
2. Running the analysis script (or a dedicated dyno chart generator)
3. Saving to `artifacts/figures/rfl_dyno_chart.png`

**Note:** The analysis script currently generates `rfl_abstention_rate.png`, which may be the dyno chart under a different name, or a separate dyno chart generator may be needed.

---

## 5. Figures Catalog Integration

### Current Catalog: `experiments/FIGURES_CATALOG.md`

The catalog documents 6 standard figures but does not explicitly mention the "Dyno Chart". 

**Recommendation:** Add entry for Dyno Chart:

```markdown
## 7. RFL Dyno Chart (Abstention Rate Comparison)

*   **Type:** Time-series (Dual Line Plot)
*   **Data Source:** `results/fo_baseline_wide.jsonl` and `results/fo_rfl_wide.jsonl`
*   **X-Axis:** Cycle Index
*   **Y-Axis:** Rolling Abstention Rate (P(abstain))
*   **Narrative:**
    *   **Paper:** "Comparison of abstention dynamics between baseline and RFL-enabled runs. Demonstrates RFL's impact on reducing unnecessary computation attempts."
    *   **Investor Deck:** "The Dyno Chart: Real-time visualization of RFL's efficiency gains."
*   **Code:** `experiments/analyze_abstention_curves.py --baseline ... --rfl ...`
*   **Output:** `artifacts/figures/rfl_dyno_chart.png`
```

---

## 6. Issues and Recommendations

### Critical Issues

1. **❌ Wide Slice Logs Missing**
   - **Impact:** Cannot perform validation
   - **Action:** Generate logs using:
     ```bash
     uv run python experiments/run_fo_cycles.py \
       --mode=baseline \
       --cycles=1000 \
       --slice-name=slice_medium \
       --system=pl \
       --out=results/fo_baseline_wide.jsonl
     
     uv run python experiments/run_fo_cycles.py \
       --mode=rfl \
       --cycles=1000 \
       --slice-name=slice_medium \
       --system=pl \
       --out=results/fo_rfl_wide.jsonl
     ```

2. **❌ Dyno Chart Not Found**
   - **Impact:** Cannot verify visual integrity
   - **Action:** Generate chart after logs are created

### Warnings

1. **⚠️ Schema Mismatch in Default Baseline Log**
   - The existing `fo_baseline.jsonl` lacks `status`, `method`, and `abstention` fields
   - This is expected for older logs; wide slice logs should have correct schema

2. **⚠️ Figures Catalog Missing Dyno Chart Entry**
   - Add documentation for the dyno chart in `FIGURES_CATALOG.md`

### Recommendations

1. **Run QA Script After Log Generation:**
   ```bash
   uv run python experiments/qa_dyno_chart.py
   ```

2. **Verify Analysis Script Compatibility:**
   - Test that `analyze_abstention_curves.py` can load and process the wide slice logs
   - Confirm rolling abstention curves are computed correctly

3. **Manual Visual Inspection:**
   - Once the dyno chart is generated, manually verify:
     - Both lines are visible and distinct
     - Axes are properly labeled
     - Legend is clear
     - Curve shapes are reasonable (not artifacts)

---

## 7. Validation Script

A comprehensive QA script has been created: `experiments/qa_dyno_chart.py`

**Features:**
- Log sanity checks (cycle indices, abstention distribution)
- Schema compatibility validation
- Analysis script compatibility testing
- Dyno chart file integrity checks

**Usage:**
```bash
uv run python experiments/qa_dyno_chart.py
```

**Output:** Detailed report with pass/fail status for each check.

---

## 8. Conclusion

**Current Status:** ⚠️ **DATA FILES MISSING - CANNOT COMPLETE VALIDATION**

The QA infrastructure is in place, but the wide slice log files and dyno chart need to be generated before full validation can be performed.

**Next Steps:**
1. Generate wide slice logs using `run_fo_cycles.py`
2. Run `qa_dyno_chart.py` to validate logs
3. Generate dyno chart (via analysis script or dedicated generator)
4. Perform visual inspection of dyno chart
5. Update `FIGURES_CATALOG.md` with dyno chart entry

**Once data files exist, re-run this QA process to confirm:**
- ✅ Log sanity checks passed
- ✅ Schema compatibility confirmed
- ✅ Analysis script processes logs correctly
- ✅ Dyno chart integrity verified

---

**Report Generated By:** Cursor L (QA Engineer)  
**Validation Script:** `experiments/qa_dyno_chart.py`  
**Analysis Script:** `experiments/analyze_abstention_curves.py`

