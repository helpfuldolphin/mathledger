# Dyno Chart QA Summary

**Status:** ⚠️ **DATA FILES MISSING - VALIDATION PENDING**

## Quick Status

- ❌ **Wide Slice Logs:** `results/fo_baseline_wide.jsonl` and `results/fo_rfl_wide.jsonl` not found
- ❌ **Dyno Chart:** `artifacts/figures/rfl_dyno_chart.png` not found
- ✅ **Analysis Script:** `experiments/analyze_abstention_curves.py` exists and is compatible
- ✅ **QA Script:** `experiments/qa_dyno_chart.py` created for validation
- ✅ **Schema:** Expected log structure is compatible with analysis script

## Action Items

1. **Generate Wide Slice Logs:**
   ```bash
   uv run python experiments/run_fo_cycles.py \
     --mode=baseline --cycles=1000 \
     --slice-name=slice_medium --system=pl \
     --out=results/fo_baseline_wide.jsonl
   
   uv run python experiments/run_fo_cycles.py \
     --mode=rfl --cycles=1000 \
     --slice-name=slice_medium --system=pl \
     --out=results/fo_rfl_wide.jsonl
   ```

2. **Run QA Validation:**
   ```bash
   uv run python experiments/qa_dyno_chart.py
   ```

3. **Generate Dyno Chart:**
   ```bash
   uv run python experiments/analyze_abstention_curves.py \
     --baseline results/fo_baseline_wide.jsonl \
     --rfl results/fo_rfl_wide.jsonl \
     --window-size 100
   ```

4. **Manual Visual Inspection:**
   - Verify both lines (Baseline and RFL) are drawn
   - Check axes are labeled (Cycle Index, Abstention Rate)
   - Confirm no obvious bugs (swapped labels, empty lines)

## Expected Log Schema

Each JSONL entry should have:
- `cycle`: int (0..N-1)
- `status`: "abstain" | "verified" | "error"
- `method`: verification method string
- `abstention`: bool
- `mode`: "baseline" | "rfl"
- `derivation`: {candidates, abstained, verified, candidate_hash}
- `rfl`: {executed, policy_update, ...}

## Validation Criteria

Once logs exist, verify:
- ✅ No missing cycle indices (0..N-1 complete)
- ✅ Reasonable abstention distribution (not 0%, not 100%)
- ✅ Schema matches analysis script expectations
- ✅ Analysis script can compute rolling abstention curves
- ✅ Dyno chart file exists and is valid PNG
- ✅ Visual inspection passes (both lines, labeled axes, no bugs)

---

**Full Report:** See `experiments/DYNO_CHART_QA_REPORT.md` for detailed analysis.

