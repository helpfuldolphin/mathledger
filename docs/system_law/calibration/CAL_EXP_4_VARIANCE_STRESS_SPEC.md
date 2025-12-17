# CAL-EXP-4: Variance Stress — Temporal Comparability Integrity

**Status**: SPECIFICATION (BINDING CHARTER)
**Authority**: STRATCOM
**Date**: 2025-12-14
**Scope**: CAL-EXP-4 only
**Mutability**: Frozen upon ratification
**Mode**: SHADOW (observational only)

---

## Objective

**Experiment Name**: CAL-EXP-4 — Variance Stress Test

**Scientific Question**: Does the CAL-EXP-3-style validity and comparability verifier fail-closed when temporal/variance structure differs between arms or between run-pairs?

**Goal**: Prevent "false comparability" from producing illegitimate ΔΔp claims by stress-testing the verifier's ability to detect temporal structure mismatches.

### Scope Fence

CAL-EXP-4 is a **Phase-II measurement integrity stress test**. It is:

| CAL-EXP-4 IS | CAL-EXP-4 IS NOT |
|--------------|------------------|
| A verifier soundness test | A capability measurement |
| A comparability integrity check | An uplift experiment |
| A stress test of existing machinery | An introduction of new metrics |
| A fail-close validation | A performance benchmark |

**Relationship to CAL-EXP-3**: CAL-EXP-4 re-uses CAL-EXP-3 definitions, artifact layouts, and claim ladder. It extends the verifier to detect a new class of invalidity (temporal structure mismatch) without modifying frozen CAL-EXP-3 semantics.

---

## Baseline and Treatment Arms

CAL-EXP-4 inherits the dual-arm architecture from CAL-EXP-3.

### Baseline Arm (Control)

| Property | Binding |
|----------|---------|
| Learning | **Disabled** |
| Parameters | Fixed (no adaptation) |
| Seed discipline | Identical to treatment |
| Toolchain fingerprint | Identical to treatment |
| **Variance profile** | **Explicitly parameterized** |

### Treatment Arm (Learning ON)

| Property | Binding |
|----------|---------|
| Learning | **Enabled** |
| Allowed knobs | ONLY those already defined in CAL-EXP-3 canon |
| Initial state | Identical to baseline |
| **Variance profile** | **Explicitly parameterized** |

### CAL-EXP-4 Extension: Variance Profile Parameterization

Unlike CAL-EXP-3 (which used a fixed variance profile), CAL-EXP-4 explicitly parameterizes the corpus's inter-cycle variance characteristics:

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `noise_scale` | Magnitude of per-cycle state perturbations | `0.01` (low), `0.05` (medium), `0.10` (high) |
| `drift_rate` | Rate of systematic state evolution | `0.001` (slow), `0.005` (moderate), `0.010` (fast) |
| `spike_probability` | Frequency of sudden state jumps | `0.01` (rare), `0.05` (occasional), `0.10` (frequent) |

**Key constraint**: The verifier must detect when baseline and treatment arms (or different run-pairs) have materially different variance profiles, and either:
1. **INVALIDATE** the comparison, or
2. **CAP** the claim level (cannot exceed L3)

---

## Definition of Temporal Comparability

### What Temporal Comparability IS

**Temporal comparability** is the property that two runs (or two arms within a run) have sufficiently similar variance profiles that comparing their ΔP values is meaningful.

| Property | Description |
|----------|-------------|
| Content identity | Same corpus values (content hash match) — **already checked by F1.2** |
| **Temporal identity** | Same distribution of state changes across cycles — **new for CAL-EXP-4** |

### What Temporal Comparability IS NOT

| Non-Property | Why |
|--------------|-----|
| Identical outputs | Runs may have different outputs due to learning; temporal comparability is about input structure |
| Statistical equivalence | Not a formal hypothesis test; descriptive mismatch detection only |
| A new metric | Temporal comparability is a validity condition, not a measured quantity |

### Temporal Structure Mismatch

A **temporal structure mismatch** occurs when two runs (or arms) have:

1. **Different variance profiles** (noise scale, drift rate, spike probability differ materially)
2. **Different autocorrelation structure** (sequential dependencies differ)
3. **Different window volatility** (variance within evaluation sub-windows differs)

**Materiality threshold**: A mismatch is material if the difference in any variance parameter exceeds 2x between runs/arms.

---

## Explicit Invalidations

| Invalidation | Consequence |
|--------------|-------------|
| Variance profile mismatch between arms | Comparison INVALID; cannot compute meaningful ΔΔp |
| Variance profile mismatch between run-pairs (for L5) | Run-pairs not comparable; cannot aggregate for replication |
| Temporal structure audit missing | Run INVALID; claim capped at L2 |
| Temporal structure audit failed | Run INVALID; claim capped at L3 |
| Any CAL-EXP-3 invalidation (F1.1-F2.3) | Inherited; run INVALID |

### Fail-Close Behavior

CAL-EXP-4 operates under **fail-close** semantics:

- If temporal structure audit is missing → INVALID
- If temporal structure audit shows mismatch → INVALID or capped
- If temporal structure cannot be determined → INVALID (assume mismatch)

This prevents false positives (certifying comparability when it doesn't hold) at the cost of potential false negatives.

---

## Failure Taxonomy

### Class 1: Measurement Failures (Inherited from CAL-EXP-3)

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F1.1: Toolchain drift** | Different runtime versions between arms | Hash comparison fails |
| **F1.2: Corpus contamination** | Different inputs between arms | Input manifest differs |
| **F1.3: Window misalignment** | Different cycle ranges compared | Window bounds differ |
| **F1.4: Warm-up inclusion** | Transient data included in evaluation | Warm-up cycles in window |

### Class 2: Confounding Failures (Inherited from CAL-EXP-3)

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F2.1: Parameter leakage** | Treatment arm has hidden parameter changes | Config diff non-empty |
| **F2.2: Seed divergence** | Different random seeds | Seed logs differ |
| **F2.3: External ingestion** | Treatment arm receives external signal | Network/file access logged |
| **F2.4: Observer effect** | Measurement itself alters behavior | Instrumentation changes state |

### Class 5: Temporal Structure Failures (NEW for CAL-EXP-4)

| Failure Mode | Description | Detection |
|--------------|-------------|-----------|
| **F5.1: Variance profile mismatch** | Arms have different noise_scale, drift_rate, or spike_probability | Profile comparison in temporal_structure_audit.json |
| **F5.2: Autocorrelation mismatch** | Arms have different sequential dependency structure | Lag-1 autocorrelation comparison |
| **F5.3: Window volatility mismatch** | Sub-window variances differ materially between arms | Per-window variance comparison |
| **F5.4: Temporal audit missing** | Required temporal_structure_audit.json not present | File existence check |
| **F5.5: Temporal audit inconclusive** | Audit cannot determine comparability | Audit status field |

### Failure Severity

| Failure Class | Severity | Consequence |
|---------------|----------|-------------|
| F1.x | INVALIDATING | Run void |
| F2.x | INVALIDATING | Run void |
| F5.1-F5.3 | INVALIDATING or CAPPING | Run void OR claim capped at L3 |
| F5.4 | CAPPING | Claim capped at L2 |
| F5.5 | CAPPING | Claim capped at L3 |

---

## Claim Validity Rules

### Claim Level Under Temporal Mismatch

| Condition | Maximum Claim Level |
|-----------|---------------------|
| F5.1, F5.2, or F5.3 detected | **L3** (ΔΔp computed but comparability invalid) |
| F5.4 (audit missing) | **L2** (ΔΔp computed but validity unknown) |
| F5.5 (audit inconclusive) | **L3** |
| All F5.x passed | Per CAL-EXP-3 rules (L4 or L5 possible) |

### Valid Claims Under CAL-EXP-4

| Claim Template | Conditions |
|----------------|------------|
| "Measured ΔΔp of X under CAL-EXP-3 conditions with temporal comparability verified" | All F1.x, F2.x, F5.x passed |
| "ΔΔp computed; temporal comparability not verified" | F5.x failed or inconclusive |
| "Verifier detected temporal structure mismatch; comparison invalid" | F5.1-F5.3 triggered |

### Invalid Claims

| Claim | Why Invalid |
|-------|-------------|
| "ΔΔp of X" (without temporal comparability note) | Missing context if F5.x not verified |
| "Runs are comparable" (when variance profiles differ) | False comparability |
| "L4 achieved" (when F5.x failed) | Claim level capped |

---

## What Becomes Canon vs. What Does Not

### What Becomes Canon (Process)

| Element | Status |
|---------|--------|
| Temporal comparability definition | CANON (validity condition) |
| F5.x failure taxonomy | CANON (detection rules) |
| Claim capping rules under mismatch | CANON (governance) |
| temporal_structure_audit.json contract | CANON (artifact requirement) |
| Fail-close semantics | CANON (policy) |

### What Does NOT Become Canon

| Element | Reason |
|---------|--------|
| Specific variance profile values | Run-dependent |
| Materiality threshold (2x) | May be refined; not frozen |
| Autocorrelation calculation method | Implementation detail |
| Any specific ΔΔp values | Seed-dependent |

---

## Pre-Registration Requirements

### Seed Discipline (Inherited)

Per CAL-EXP-3: Seed must be registered before execution in `run_config.json`.

### Window Discipline (Inherited)

Per CAL-EXP-3: Evaluation windows must be pre-registered.

### Variance Profile Discipline (NEW)

**Requirement**: Variance profile parameters must be declared before execution:

```json
{
  "variance_profile": {
    "noise_scale": 0.03,
    "drift_rate": 0.002,
    "spike_probability": 0.05,
    "registered_at": "<ISO8601>"
  }
}
```

**Post-hoc variance profile selection is forbidden** — changing variance parameters after seeing results invalidates the run.

---

## Binding Constraints

### What This Document Defines

- The meaning of "temporal comparability"
- Conditions under which temporal mismatch invalidates comparisons
- Failure taxonomy extensions for CAL-EXP-4
- Claim capping rules under mismatch
- Pre-registration requirements for variance profiles

### What This Document Does NOT Define

- Implementation details (code, scripts, harnesses)
- New metrics (temporal comparability is a validity condition, not a metric)
- Thresholds for "good" variance (CAL-EXP-4 measures, does not judge)
- Pilot logic or external data sources

### Document Status

| Property | Value |
|----------|-------|
| Status | SPECIFICATION (BINDING) |
| Mutability | Frozen upon STRATCOM ratification |
| Additive changes | Permitted (failure modes, clarifications) |
| Semantic changes | Require STRATCOM re-ratification |
| Implementation authority | NOT GRANTED by this document |

---

## Appendix: Attack Vector Definition

### The Key Attack: Temporal Structure Mismatch

**Attack scenario**: An operator (intentionally or accidentally) runs baseline and treatment arms with different variance profiles, then computes ΔΔp as if the runs were comparable.

**Why this is dangerous**: ΔΔp = treatment_mean - baseline_mean assumes both arms experienced the same input dynamics. If baseline saw low-variance input and treatment saw high-variance input, the difference in ΔP may reflect the variance difference, not the learning effect.

**CAL-EXP-4's defense**: The temporal_structure_audit.json must verify that variance profiles match. If they don't match, the verifier must either INVALIDATE the run or CAP the claim level, preventing false comparability from escalating to L4/L5.

---

**SHADOW MODE** — observational only.

*Precision > optimism.*
