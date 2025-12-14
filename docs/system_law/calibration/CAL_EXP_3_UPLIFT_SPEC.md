# CAL-EXP-3: Learning Uplift Measurement

**Status**: SPECIFICATION (BINDING)
**Authority**: STRATCOM
**Date**: 2025-12-13
**Scope**: CAL-EXP-3 only
**Mutability**: Frozen upon ratification

---

## Charter (AUTHORITATIVE)

**Experiment Name**: CAL-EXP-3 — Learning Uplift Measurement

**Objective**: Measure whether enabling learning (RFL / update loop) produces a statistically detectable improvement in Δp relative to a no-learning baseline, under identical conditions.

### Baseline Arm (Control)

| Property | Binding |
|----------|---------|
| Learning | **Disabled** |
| Parameters | Fixed (no adaptation) |
| Seed discipline | Identical to treatment |
| Toolchain fingerprint | Identical to treatment |

### Treatment Arm (Learning ON)

| Property | Binding |
|----------|---------|
| Learning | **Enabled** |
| Allowed knobs | ONLY those already defined in canon (no new parameters) |
| Initial state | Identical to baseline |

### Primary Metric

**Δp_success** (as already defined in `METRIC_DEFINITIONS.md@v1.1.0`)

No other metric may substitute.

### Required Reporting

| Field | Definition |
|-------|------------|
| `baseline_mean_Δp` | Mean Δp across evaluation window, learning OFF |
| `treatment_mean_Δp` | Mean Δp across evaluation window, learning ON |
| `ΔΔp` | `treatment_mean_Δp − baseline_mean_Δp` |
| `windowed_analysis` | Per-window breakdown excluding warm-up phase |

### Validity Conditions (ALL required)

| Condition | Verification |
|-----------|--------------|
| Toolchain parity | Hash of runtime environment identical |
| Identical input corpus | Same problem set, same order, same encoding |
| Identical evaluation windows | Same cycle ranges, same exclusions |
| No new pathology introduced | No errors, crashes, or anomalies in either arm |

### Explicit Invalidations

| Invalidation | Consequence |
|--------------|-------------|
| Any divergence metric substituted for Δp | Experiment void |
| Any external data ingestion | Experiment void |
| Any post-hoc hypothesis | Finding not reportable |
| Any monotone improvement claim without window evidence | Claim forbidden |

### Claims Allowed

> "Measured uplift of X under CAL-EXP-3 conditions"

### Claims Forbidden

| Forbidden Claim | Reason |
|-----------------|--------|
| "Improved intelligence" | Unoperationalized term |
| "Learning validated" | Overreach beyond measured scope |
| "Generalization proven" | Requires out-of-distribution evidence |

---

## Formal Definition of Uplift

### Mathematical Definition

Let:
- `B = {b_1, b_2, ..., b_n}` be the sequence of Δp values from the baseline arm (learning OFF)
- `T = {t_1, t_2, ..., t_n}` be the sequence of Δp values from the treatment arm (learning ON)
- `W` be the evaluation window (cycle range after warm-up exclusion)

**Uplift** is defined as:

```
ΔΔp = mean(T|W) - mean(B|W)
```

Where:
- `mean(T|W)` = arithmetic mean of Δp values in treatment arm, restricted to window W
- `mean(B|W)` = arithmetic mean of Δp values in baseline arm, restricted to window W

### Prose Definition

**Uplift** is the difference in mean task-success probability between a system with learning enabled and an otherwise-identical system with learning disabled, measured over a shared evaluation window that excludes warm-up transients.

Uplift is:
- A **comparative quantity** (requires both arms)
- **Window-bound** (only valid within the specified evaluation range)
- **Condition-locked** (only valid under identical experimental conditions)

### What Uplift IS

| Property | Description |
|----------|-------------|
| A measured delta | The arithmetic difference between two means |
| Arm-relative | Meaningful only as treatment vs. control |
| Window-specific | Valid only within the stated evaluation window |
| Reproducibility-gated | Valid only if conditions are identical across arms |

### What Uplift IS NOT

| Non-Property | Why |
|--------------|-----|
| An absolute capability | Uplift is relative to a specific baseline, not an external standard |
| A generalization claim | Uplift in-window does not imply uplift out-of-window |
| Intelligence | "Intelligence" is not operationalized; Δp is |
| Proof of learning | Uplift is consistent with learning, but does not prove mechanism |
| Monotonic progress | Uplift can be positive in one window and negative in another |

### The Uplift Null Hypothesis

**H_0**: Enabling learning produces no detectable difference in mean Δp.

```
H_0: ΔΔp = 0  (within measurement noise)
```

CAL-EXP-3 attempts to reject H_0. Failure to reject is not failure of the system—it is an honest measurement.

---

## Failure Taxonomy

### Class 1: Measurement Failures

These invalidate the experiment entirely.

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F1.1: Toolchain drift** | Different runtime versions between arms | Hash comparison fails |
| **F1.2: Corpus contamination** | Different inputs between arms | Input manifest differs |
| **F1.3: Window misalignment** | Different cycle ranges compared | Window bounds differ |
| **F1.4: Warm-up inclusion** | Transient data included in evaluation | Warm-up cycles in window |

### Class 2: Confounding Failures

These produce spurious uplift signals.

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F2.1: Parameter leakage** | Treatment arm has hidden parameter changes | Config diff non-empty |
| **F2.2: Seed divergence** | Different random seeds | Seed logs differ |
| **F2.3: External ingestion** | Treatment arm receives external signal | Network/file access logged |
| **F2.4: Observer effect** | Measurement itself alters behavior | Instrumentation changes state |

### Class 3: Interpretive Failures

These produce invalid claims from valid data.

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F3.1: Overgeneralization** | Claiming uplift beyond measured window | Claim scope exceeds data scope |
| **F3.2: Mechanism attribution** | Claiming "learning works" from uplift | Causal claim without mechanism evidence |
| **F3.3: Monotonicity assumption** | Assuming uplift persists or grows | Single-window data extrapolated |
| **F3.4: Post-hoc hypothesis** | Formulating hypothesis after seeing data | Hypothesis timestamp > data timestamp |

### Class 4: Statistical Failures

These misrepresent the strength of evidence.

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F4.1: Noise floor confusion** | Claiming uplift within measurement noise | ΔΔp < noise threshold |
| **F4.2: Single-run inference** | Claiming significance from one run | N=1 |
| **F4.3: Cherry-picked window** | Selecting window to maximize effect | Window not pre-registered |
| **F4.4: Variance suppression** | Reporting mean without variance | No confidence interval |

---

## Claim Validity Table

### Valid Claims

| Claim Template | Conditions | Example |
|----------------|------------|---------|
| "Measured ΔΔp of X" | X is the computed value | "Measured ΔΔp of +0.012" |
| "Measured ΔΔp of X +/- Y" | Y is standard error or CI | "Measured ΔΔp of +0.012 +/- 0.003" |
| "Measured ΔΔp of X in window W" | W is the evaluation window | "Measured ΔΔp of +0.012 in cycles 801-1000" |
| "Uplift observed under CAL-EXP-3 conditions" | All validity conditions met | (self-documenting) |
| "Failed to reject H_0" | ΔΔp within noise floor | "ΔΔp of +0.002 within noise floor of +/-0.005" |

### Invalid Claims

| Claim | Why Invalid | Correct Alternative |
|-------|-------------|---------------------|
| "Learning works" | Mechanism not measured | "Measured positive ΔΔp with learning enabled" |
| "System improved" | Implies absolute progress | "Treatment arm outperformed baseline in window W" |
| "Intelligence increased" | Term not operationalized | (no alternative—do not claim) |
| "Generalization proven" | OOD not measured | (no alternative—requires different experiment) |
| "Uplift will continue" | Future not measured | "Uplift measured in window W" |
| "Uplift of X" (without conditions) | Missing context | "Uplift of X under CAL-EXP-3 conditions" |
| "Statistically significant" (without test) | Requires formal test | "ΔΔp exceeds noise floor by factor Y" |

### Claim Strength Ladder

| Level | Claim Type | Requirements |
|-------|------------|--------------|
| **L0** | "Experiment completed" | Both arms ran to completion |
| **L1** | "Measurements obtained" | Δp values computed for both arms |
| **L2** | "ΔΔp computed" | Arithmetic difference calculated |
| **L3** | "ΔΔp exceeds noise floor" | ΔΔp > measurement noise |
| **L4** | "Uplift measured" | L3 + all validity conditions |
| **L5** | "Uplift replicated" | L4 across multiple runs |

No claim at level N is valid without satisfying all levels < N.

---

## Binding Constraints

### What This Document Defines

- The conceptual meaning of "uplift"
- The conditions under which uplift claims are valid
- The ways CAL-EXP-3 can fail or lie
- The language permitted in reporting results

### What This Document Does NOT Define

- Implementation details (code, scripts, harnesses)
- New metrics (all metrics reference existing canon)
- Thresholds for "good" uplift (CAL-EXP-3 measures, does not judge)
- Pilot logic or execution machinery

### Document Status

| Property | Value |
|----------|-------|
| Status | SPECIFICATION (BINDING) |
| Mutability | Frozen upon STRATCOM ratification |
| Additive changes | Permitted (failure modes, claim examples) |
| Semantic changes | Require STRATCOM re-ratification |
| Implementation authority | NOT GRANTED by this document |

---

## Appendix: Uplift vs. Divergence Reduction

These are distinct quantities. Conflation invalidates CAL-EXP-3.

| Quantity | Definition | Measured In |
|----------|------------|-------------|
| **Δp (success)** | Task success probability | Treatment arm |
| **δp (divergence)** | Twin tracking error | Calibration experiments |
| **ΔΔp (uplift)** | Treatment - Baseline mean Δp | CAL-EXP-3 |

**Divergence reduction** (lower δp) is about the Twin tracking the real system better.

**Uplift** (positive ΔΔp) is about the real system performing better with learning enabled.

These may correlate, but one does not imply the other. CAL-EXP-3 measures uplift. CAL-EXP-1 and CAL-EXP-2 measured divergence. They are not substitutes.

---

## Appendix: Post-Execution Note (Non-Normative)

*Added: 2025-12-14*

### Example Executions

CAL-EXP-3 has been executed under SHADOW MODE. Example run-pairs exist in `results/cal_exp_3/` using the canonical harness (`scripts/run_cal_exp_3_canonical.py`) and verifier (`scripts/verify_cal_exp_3_run.py`).

### Results Status

All results are **NON-CANONICAL** until:
- L5 is achieved (≥3 independent run-pairs per §5.2.1 of the Implementation Plan)
- All L5 runs share identical `toolchain_fingerprint` and pre-registered window definitions

Current executions demonstrate harness functionality only. No L5 claim is made.

### ΔΔp Sign Convention (Reiteration)

Per the Formal Definition in this document:

```
ΔΔp = mean(T|W) − mean(B|W)
```

- **Positive ΔΔp**: Treatment arm (learning ON) has higher mean Δp than baseline arm (learning OFF)
- **Negative ΔΔp**: Treatment arm has lower mean Δp than baseline arm
- **ΔΔp ≈ 0**: No detectable difference (fail to reject H_0)

The sign convention is **treatment minus baseline**, not the reverse.

### Conformance Verification

The implementation (`CAL_EXP_3_IMPLEMENTATION_PLAN.md`), canonical producer (`run_cal_exp_3_canonical.py`), and verifier (`verify_cal_exp_3_run.py`) have been verified for spec conformance:

| Component | Conformance | Notes |
|-----------|-------------|-------|
| ΔΔp formula | ✓ Correct | `treatment_mean - baseline_mean` |
| Window semantics | ✓ Correct | Inclusive bounds, missing-cycle INVALIDATION |
| Claim ladder | ✓ Correct | L0-L4 single-run, L5 ≥3 runs |
| Artifact layout | ✓ Correct | All required files verified |
| Isolation audit | ✓ Correct | F2.3 negative proof via `isolation_audit.json` |
| Validity checks | ✓ Correct | F1.1-F2.3 automated |

**No critical discrepancies** between spec and implementation.

### Observed Minor Variances (Non-Critical)

| Item | Variance | Status |
|------|----------|--------|
| Original producer (`run_cal_exp_3.py`) | Uses `delta_p_trace.jsonl` instead of `cycles.jsonl` | Non-canonical, superseded by `run_cal_exp_3_canonical.py` |
| `timestamp` field in cycles.jsonl | Included but excluded from determinism comparison | Per §4.3 (correct) |

These do not affect validity of canonical runs.

---

**SHADOW MODE** — observational only.

*Precision > optimism.*
