# Phase II Governance Stability — Diligence One-Pager

**Classification**: Technical Due Diligence
**Audience**: Research leadership / acquisition diligence
**Date**: 2025-12-21
**Status**: FROZEN (commit 7cd0322)

---

## What Was Tested

Phase II tested **governance stability** — whether the system's fail-closed mechanisms behave correctly and consistently under stress and perturbation. Three experiments were executed:

| Experiment | Question |
|------------|----------|
| CAL-EXP-4 | Does fail-close trigger correctly under variance stress? |
| CAL-EXP-5 | Can variance-aligned conditions avoid fail-close? |
| Phase II.c | Is the governance verdict invariant under auxiliary parameter perturbation? |

**Note**: Phase II tested governance correctness, not capability or learning performance.

---

## Results Summary

| Experiment | Seeds | Runs | Verdict | F5 Codes |
|------------|-------|------|---------|----------|
| CAL-EXP-4 | 42, 43, 44 | 3 | Claims capped to L0 | F5.2, F5.3 |
| CAL-EXP-5 | 42, 43, 44 | 3 | FAIL (fail-close triggered) | F5.2, F5.3 |
| Phase II.c | 42, 43, 44 | 21 | PASS (21/21 invariant) | F5.2, F5.3 (invariant) |

---

## Interpreting CAL-EXP-5 "FAIL"

The CAL-EXP-5 result requires careful interpretation:

**What "FAIL" means**: The hypothesis that variance-aligned conditions would avoid fail-close was **rejected**. The system triggered fail-close (F5.2) even under conditions designed to avoid it.

**What "FAIL" does NOT mean**: This is not a defect or malfunction. The system behaved as designed — it refused to claim validity when variance structures were outside acceptable bounds.

**Why this is positive for diligence**: Governance resists "variance laundering" — attempts to construct conditions that circumvent fail-safe mechanisms. The system is conservative by design.

---

## What Passed

1. **CAL-EXP-4**: Governance correctly triggered fail-close under variance stress. Claims appropriately capped to L0.

2. **Phase II.c**: Governance verdict was invariant under all tested auxiliary perturbations. No hidden dependencies on auxiliary parameters were detected. Frozen predicates fully determined the verdict.

---

## What Did Not Pass (Hypothesis Rejected)

**CAL-EXP-5**: The variance-alignment strategy tested did not avoid fail-close. This is a hypothesis rejection, not a defect. Governance resists variance laundering.

---

## Explicit Non-Claims

Phase II does NOT establish:

| Non-Claim | Reason |
|-----------|--------|
| Capability or performance | Outside scope |
| Convergence or learning | Not tested |
| Threshold optimality | Thresholds frozen, not evaluated |
| Predicate completeness | Only tested frozen predicates |
| Generalization to untested perturbations | Finite perturbation set |
| "Governance is robust" | Overgeneralization beyond tested conditions |

---

## Artifact References

| Artifact | Path |
|----------|------|
| CAL-EXP-4 results | `results/cal_exp_4/` |
| CAL-EXP-5 results | `results/cal_exp_5/` |
| Phase II.c verdict matrices | `results/phase_ii_c/` |
| Phase II.c harness | `scripts/run_phase_ii_verdict_invariance.py` |
| Governance stability memo | `docs/system_law/calibration/PHASE_II_GOVERNANCE_STABILITY_MEMO.md` |
| CAL-EXP-4 freeze declaration | `docs/system_law/calibration/CAL_EXP_4_FREEZE.md` |
| Phase II.c freeze declaration | `docs/system_law/calibration/PHASE_II_VERDICT_INVARIANCE_FREEZE.md` |

---

## Bottom Line

Phase II demonstrated that:

1. **Fail-close works** — The system refuses to claim validity when conditions are out of bounds
2. **Fail-close is resistant** — The system cannot be trivially bypassed
3. **Verdict is deterministic** — Auxiliary parameters do not affect governance decisions

These properties reduce risk of silent degradation, verdict instability, and hidden dependencies.

---

**FROZEN** — No modifications without STRATCOM authorization.

*Derived from: PHASE_II_GOVERNANCE_STABILITY_MEMO.md*
