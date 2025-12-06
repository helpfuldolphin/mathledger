# RFL Instrumentation Harness

**Purpose:** Minimal, manual instrumentation to understand what happens inside a single RFL cycle and small batches, without running full 1000-cycle experiments or depending on DB/Redis.

**Status:** Hermetic, no-DB, no-Redis environment for surgical inspection.

## Prerequisites

**Fresh PowerShell tab setup:**
```powershell
Set-Location C:\dev\mathledger

# CRITICAL: Do NOT set DATABASE_URL or REDIS_URL
# This harness runs standalone without database dependencies

$env:FIRST_ORGANISM_TESTS = "false"
$env:MDAP_SEED = "0x4D444150"
```

## Scripts Overview

1. **instrument_single_cycle.ps1** - Single-cycle RFL probe
2. **instrument_latency.ps1** - Latency measurement for one RFL cycle
3. **instrument_micro_dyno.ps1** - Ten-cycle "micro-Dyno" run
4. **instrument_determinism.ps1** - Determinism check on H_t

---

## 1. Single-Cycle RFL Probe

**Run:**
```powershell
.\experiments\instrument_single_cycle.ps1
```

**What it does:**
- Imports `CycleRunner` from `experiments.run_fo_cycles`
- Instantiates runner in `mode="rfl"`
- Calls `run_cycle(0, runner, mode="rfl")`
- Prints the resulting dict in readable format

**Expected runtime:** < 1 second if healthy

**Output fields:**
- `cycle`: Cycle index (0)
- `mode`: Execution mode ("rfl")
- `status`: "abstain", "verified", or "error"
- `method`: Verification method used
- `abstention`: Boolean indicating if cycle abstained
- `roots.h_t`, `roots.r_t`, `roots.u_t`: Merkle roots
- `derivation`: Candidate counts and hash
- `rfl`: RFL execution stats (executed, policy_update, etc.)
- `gates_passed`: Whether curriculum gates passed

**What to look for:**
- ✅ `rfl.executed` should be `True` (RFL runner is active)
- ✅ `roots.h_t` should be a 64-char hex string
- ✅ `derivation.candidates` > 0 (derivation is producing candidates)
- ⚠️ If `status == "error"`, check derivation pipeline
- ⚠️ If `rfl.executed == False`, RFL runner failed to initialize

**Failure modes:**
- `ImportError`: Check that `experiments.run_fo_cycles` is importable
- DB access attempt: Code already mocks DB via `patch("rfl.runner.load_baseline_from_db")`, but if you see DB errors, check for other DB dependencies
- If imports fail, verify you're in the project root and `uv` environment is active

---

## 2. Latency Measurement

**Run:**
```powershell
.\experiments\instrument_latency.ps1
```

**What it does:**
- Measures wall-clock latency of one RFL cycle
- Reports initialization time vs cycle execution time
- Provides breakdown estimates

**Expected runtime:** < 1 second total

**Normal vs suspicious:**
- **Normal:** < 0.5s per cycle
- **Suspicious:** > 1.0s per cycle
- **Warning zone:** 0.5-1.0s (monitor for consistency)

**Latency breakdown (estimated):**
- Derivation (`run_slice_for_test`): ~60-80% of cycle time
- Attestation (`seal_block_with_dual_roots`): ~10-15%
- RFL metabolism (`run_with_attestation`): ~10-20% (if executed)
- Gate evaluation: <5%

**If latency is high (>1s), most likely culprits:**
1. **Derivation** - Check `run_slice_for_test`:
   - Lean interface calls (if using real Lean)
   - Heavy search space exploration
   - File I/O in derivation pipeline
2. **RFL runner** - Policy updates or histogram computation
3. **Logging** - Excessive file I/O or console output

**Failure modes:**
- If cycle hangs indefinitely, check for:
  - Blocking I/O operations
  - Infinite loops in derivation search
  - Deadlocks in RFL runner

---

## 3. Micro-Dyno (10 Cycles)

**Run:**
```powershell
.\experiments\instrument_micro_dyno.ps1
```

**What it does:**
- Runs 10 RFL cycles in sequence
- Prints one line per cycle: `cycle | abstain | H_t (short)`
- Computes summary statistics

**Output format:**
```
  0 | True  | H_t: 01e5056e567ba57e...
  1 | True  | H_t: 01e5056e567ba57e...
  2 | False | H_t: a8dc5b2c7778ce38f...
  ...
```

**Interpretation:**

**Many early abstentions (`abstain: True`):**
- Expected for hard slices (atoms=7, depth_max=12)
- System is struggling with slice difficulty
- RFL should learn and reduce abstention over time

**Transition from `True` to `False`:**
- Good sign - RFL is learning
- Policy updates are taking effect
- Abstention rate should decrease over 1000 cycles

**H_t changing each cycle:**
- ✅ **Good:** Indicates derivation diversity
- ⚠️ **Bad if all identical:** May indicate limited variation
- ⚠️ **Bad if only 1-2 unique values:** Derivation may be stuck

**Summary statistics:**
- `Abstentions: X (Y%)` - Abstention rate in sample
- `Unique H_t values: N` - Diversity metric
  - Should be > 5 for 10 cycles (good diversity)
  - If == 1, check determinism (may be too deterministic)
  - If < 5, derivation may have limited search space

**Failure modes:**
- If all cycles show `abstain: True` and `status: error`, derivation pipeline may be broken
- If H_t values are all identical, check determinism script

---

## 4. Determinism Check

**Run:**
```powershell
.\experiments\instrument_determinism.ps1
```

**What it does:**
- Calls `run_cycle(0)` twice with same seed/env
- Compares H_t, R_t, U_t values between runs
- Reports determinism status

**Expected behavior:**
- ✅ **All roots match:** MDAP determinism is intact
- ✅ Same inputs → same outputs (hermetic behavior)
- ✅ This is **correct** for hermetic runs

**If roots diverge:**
- ❌ **Non-determinism detected**

**Most likely culprits:**

**R_t mismatch:**
- Derivation has non-deterministic search
- Check: `run_slice_for_test`, candidate ordering, hash computation
- Look for: Random number generation, time-based values, process IDs

**U_t mismatch:**
- UI event capture has timing/ordering issues
- Check: `capture_ui_event`, `ui_event_store` ordering
- Look for: Threading issues, non-deterministic event ordering

**H_t mismatch:**
- Composite root depends on R_t and U_t
- Fix R_t/U_t determinism first
- H_t = SHA256(R_t || U_t), so it will match once R_t and U_t match

**Debug steps:**
1. Check if derivation uses any random number generation
2. Verify all hash functions use deterministic inputs
3. Check for time-based or process-ID based values
4. Ensure UI event store is cleared between cycles (`ui_event_store.clear()`)

---

## Troubleshooting

### ImportError

**Symptom:** `ModuleNotFoundError: No module named 'experiments'`

**Fix:**
- Ensure you're in `C:\dev\mathledger`
- Verify `uv` environment is active: `uv --version`
- Check that `experiments/run_fo_cycles.py` exists

### DB Access Attempts

**Symptom:** `psycopg.errors.*` or connection errors

**Fix:**
- Code already mocks DB via `patch("rfl.runner.load_baseline_from_db")`
- If you see DB errors, check for other DB dependencies:
  - Search for `DATABASE_URL` usage in `experiments/run_fo_cycles.py`
  - Check if `seal_block_with_dual_roots` has DB dependencies (it shouldn't)

### Script Hangs

**Symptom:** Script runs but never completes

**Likely causes:**
1. Derivation search stuck in infinite loop
2. RFL runner waiting on blocked I/O
3. Heavy computation in policy update

**Debug:**
- Add `print()` statements to identify where it hangs
- Use `Ctrl+C` to interrupt and check last printed line
- Consider reducing slice difficulty temporarily

### All Cycles Abstain

**Symptom:** Every cycle shows `abstain: True`

**Interpretation:**
- Expected for very hard slices (atoms=7, depth_max=12)
- System is struggling with difficulty
- RFL should learn over 1000 cycles

**If unexpected:**
- Check slice parameters in `config/curriculum.yaml`
- Verify derivation is producing candidates
- Check if verification methods are all failing

---

## Quick Reference

**Run all instrumentation:**
```powershell
# 1. Single cycle
.\experiments\instrument_single_cycle.ps1

# 2. Latency
.\experiments\instrument_latency.ps1

# 3. Micro-Dyno
.\experiments\instrument_micro_dyno.ps1

# 4. Determinism
.\experiments\instrument_determinism.ps1
```

**Expected healthy output:**
- Single cycle: < 1s, RFL executed, roots present
- Latency: < 0.5s per cycle
- Micro-Dyno: Some abstentions, H_t diversity > 5 unique values
- Determinism: All roots match between runs

**Red flags:**
- Cycle time > 1s consistently
- All cycles abstain AND status == "error"
- H_t values all identical (check determinism)
- Non-determinism detected (roots don't match)

---

## Next Steps

Once instrumentation passes:
1. ✅ Single cycle completes in < 1s
2. ✅ RFL runner executes successfully
3. ✅ H_t values show diversity
4. ✅ Determinism is intact

You can trust larger experiments:
- 1000-cycle baseline runs
- 1000-cycle RFL runs
- Dyno Chart generation

The organism is instrumented and understood at the cycle level.

