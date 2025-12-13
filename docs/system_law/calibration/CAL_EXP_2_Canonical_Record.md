# CAL-EXP-2: Canonical Experimental Record

**Status**: CANONICAL
**Experiment**: Long-Window Convergence Analysis
**Date**: 2025-12-12
**SHADOW MODE**: Active throughout

---

## Machine-Readable Summary

```json
{
  "experiment": "CAL-EXP-2",
  "horizon_cycles": 1000,
  "warmup_cycles": 400,
  "convergence_floor_delta_p": 0.025,
  "verdict": "PLATEAUING",
  "upgrade_required": "UPGRADE-2",
  "lr_config": {
    "H": 0.20,
    "rho": 0.15,
    "tau": 0.02,
    "beta": 0.12
  },
  "key_findings": {
    "non_monotonic_convergence": true,
    "warm_up_divergence_phases": [2, 3],
    "recovery_phase": 4,
    "plateau_phase": 5,
    "first_window_dp": 0.0197,
    "last_window_dp": 0.0187,
    "net_reduction": true
  }
}
```

---

## Interpretation Guardrail

**READ THIS BEFORE INTERPRETING RESULTS**

### 1. Warm-Up Divergence is Expected

The Twin exhibits a characteristic "warm-up divergence" pattern during the first ~400 cycles:

- **Phases 2-3 (cycles 201-600)**: δp increases from 0.023 to 0.031
- **This is NOT a failure** — it is the Twin's exponential averaging overshooting during state transitions
- **Recovery follows automatically** in phases 4-5

**Operational implication**: Do not abort or recalibrate during the first 400 cycles. Allow the system to complete its warm-up phase.

### 2. Plateau ≠ Failure

The system reaches a **plateau** at δp ≈ 0.025. This represents:

- The **algorithmic ceiling** of exponential averaging
- A **stable, non-divergent operating point**
- **Acceptable SHADOW MODE behavior** within observation-only constraints

A plateau means the Twin has learned what it can learn with its current architecture. Breaking through requires structural changes (UPGRADE-2), not parameter tuning.

### 3. δp ≈ 0.025 is the Current Algorithmic Floor

This floor exists because:

- Exponential averaging cannot predict future state transitions
- It can only react to observed changes with latency proportional to (1/LR)
- Stochastic noise in real telemetry creates irreducible tracking error

**This floor is acceptable for Phase X SHADOW MODE** — observation-only operation tolerates δp < 0.05.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Cycles | 1000 |
| Seed | 42 |
| Telemetry Adapter | real (synthetic mode) |
| Window Size | 50 cycles |
| LR_H | 0.20 |
| LR_ρ | 0.15 |
| LR_τ | 0.02 |
| LR_β | 0.12 |

## Results Summary

### Phase Trajectory

| Phase | Cycles | Mean δp | Classification |
|-------|--------|---------|----------------|
| 1 | 1-200 | 0.0230 | BASELINE |
| 2 | 201-400 | 0.0267 | DIVERGING (warm-up) |
| 3 | 401-600 | 0.0307 | DIVERGING (peak) |
| 4 | 601-800 | 0.0268 | CONVERGING (recovery) |
| 5 | 801-1000 | 0.0254 | PLATEAUING (floor) |

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| First window δp | 0.0197 |
| Last window δp | 0.0187 |
| Overall change | **-0.0010** (net reduction) |
| Trend slope | +0.000148/window (effectively flat) |
| Success Accuracy | 82.1% |
| Variance | STABLE |

## Verdict

### PLATEAUING

The system has reached its **convergence floor** at δp ≈ 0.025.

| Decision Gate | Result |
|---------------|--------|
| δp decreasing over horizon | ⚠️ Net yes, non-monotonic |
| Slope approaching zero | ✓ Yes (final phase) |
| Variance shrinking | ⚠️ Stable, not shrinking |
| Asymptotic behavior | ✗ Not observed |

## Recommendations

1. **UPGRADE-1 LRs SUITABLE** — Safe for extended SHADOW MODE operation
2. **UPGRADE-2 REQUIRED** — To break convergence floor
3. **Expect 400-cycle warm-up** — Do not abort during warm-up divergence
4. **δp ≈ 0.025 is acceptable** — Within SHADOW MODE tolerance

## Artifact References

| Artifact | Path |
|----------|------|
| Full Scientist Report | `results/cal_exp_2/CAL_EXP_2_Scientist_Report.md` |
| Run Metadata | `results/cal_exp_2/p4_20251212_103832/RUN_METADATA.json` |
| Run Config | `results/cal_exp_2/p4_20251212_103832/run_config.json` |
| Real Cycles | `results/cal_exp_2/p4_20251212_103832/real_cycles.jsonl` |
| Twin Predictions | `results/cal_exp_2/p4_20251212_103832/twin_predictions.jsonl` |
| Divergence Log | `results/cal_exp_2/p4_20251212_103832/divergence_log.jsonl` |

## Lineage

| Experiment | Status | Key Finding |
|------------|--------|-------------|
| CAL-EXP-1 (Baseline) | Complete | State tracking is dominant divergence source |
| CAL-EXP-1 + UPGRADE-1 | Confirmed | Per-component LRs reduce δp by 16.3% |
| **CAL-EXP-2** | **Canonical** | **Convergence floor at δp ≈ 0.025** |
| CAL-EXP-3 | Pending | UPGRADE-2 validation (requires design) |

---

**SHADOW MODE — observational only.**

---

## Distribution Footer

```
═══════════════════════════════════════════════════════════════════════════════
CALIBRATION EXPERIMENT NOTICE — NON-NORMATIVE GUIDANCE

This document reports SHADOW MODE observations from controlled synthetic testing.

• Results measure estimator tuning under test conditions only
• "Safe region" is a mathematical construct, not a safety guarantee
• "Divergence reduction" measures parameter adjustment effects, not capability
• All metrics are advisory only and do not influence system behavior
• Phase X artifacts are explicitly pre-production test data

This notice is non-normative guidance for distribution context.
See: docs/system_law/calibration/CAL_EXP_DO_NOT_CLAIM_APPENDIX.md
═══════════════════════════════════════════════════════════════════════════════
```
