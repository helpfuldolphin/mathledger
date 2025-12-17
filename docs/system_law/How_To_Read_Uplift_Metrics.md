# How to Read Uplift Metrics

**Version:** 1.1.0
**Status:** Advisory Reference
**Author:** CLAUDE D — Metrics Interpretation / Non-Overclaim Guardian
**Date:** 2025-12-14

---

## 1. Core Metrics: δp, Δp, and ΔΔp

### 1.1 δp (Divergence)

| Aspect | Description |
|--------|-------------|
| **What it is** | The instantaneous difference between predicted and observed values at a single cycle |
| **What it measures** | Point-in-time prediction error |
| **What it is NOT** | A measure of system quality, capability, or correctness |
| **Notation** | δp = |predicted - observed| |

δp is raw observation. It tells you how far off a prediction was at one moment. Nothing more.

---

### 1.2 Δp (Success Metric)

| Aspect | Description |
|--------|-------------|
| **What it is** | Aggregated divergence over a measurement window (e.g., mean, median, or percentile of δp values) |
| **What it measures** | Central tendency or distribution of prediction errors across cycles |
| **What it is NOT** | A score, grade, or indicator of system success in any general sense |
| **Notation** | Δp = aggregate(δp₁, δp₂, ..., δpₙ) |

Δp summarizes divergence behavior over time. The word "success" in "success metric" refers to the metric's role in the experimental protocol, not to any claim about the system succeeding at a task.

---

### 1.3 ΔΔp (Uplift)

| Aspect | Description |
|--------|-------------|
| **What it is** | The difference in Δp between two conditions (baseline vs modified) |
| **What it measures** | Whether a structural change altered aggregate divergence |
| **What it is NOT** | A measure of improvement, learning, or capability gain |
| **Notation** | ΔΔp = Δp_baseline - Δp_modified |

ΔΔp > 0 means the modified condition had lower aggregate divergence than baseline in the measured window. This is an observation about error magnitudes under specific conditions, not a claim about system properties.

---

## 2. Common Misreads

| # | Misread | Safe Correction |
|---|---------|-----------------|
| 1 | "ΔΔp > 0 means the uplift worked" | ΔΔp > 0 means aggregate divergence decreased under the modified condition in this experiment. Whether this constitutes "working" depends on operational requirements not defined here. |
| 2 | "Larger ΔΔp is better" | ΔΔp magnitude reflects experimental conditions. Larger values indicate larger differences, not necessarily more useful changes. |
| 3 | "The system learned from the uplift" | The experiment measures prediction error, not internal state changes. No claims about learning are supported. |
| 4 | "Negative ΔΔp means the uplift failed" | Negative ΔΔp means divergence increased under the modified condition. This describes what happened, not whether something "failed." |
| 5 | "δp = 0 means perfect prediction" | δp = 0 means predicted and observed values matched at one cycle. This does not imply the model is correct, only that it was not wrong at that moment. |
| 6 | "Low Δp proves the model is accurate" | Low Δp means aggregate divergence was small in this window. Accuracy is not a permanent property and does not transfer to other contexts. |
| 7 | "ΔΔp generalizes to production" | ΔΔp was measured under experimental conditions. Generalization to production requires separate evidence. |
| 8 | "We achieved an uplift of X%" | Percentage framing implies a denominator that may not be meaningful. Report raw values: "ΔΔp = 0.012 over 500 cycles." |
| 9 | "The modification improved the system" | The modification was associated with lower divergence in this context. "Improvement" implies value judgment not supported by the measurement. |
| 10 | "This proves the approach is correct" | Experiments constrain uncertainty; they do not prove correctness. The approach produced certain measurements under certain conditions. |

---

## 3. Claim Ladder (L0–L5)

The following levels describe what can be said about experimental results. Each level adds interpretive content; higher levels require correspondingly stronger evidence.

| Level | Claim Type | Example Statement | Evidence Required |
|-------|------------|-------------------|-------------------|
| **L0** | Raw Observation | "δp ranged from 0.001 to 0.045 across 500 cycles" | Measurement log |
| **L1** | Aggregation | "Δp (p50) = 0.018 for the baseline condition" | Aggregation method documented |
| **L2** | Comparison | "ΔΔp = 0.007 (modified vs baseline)" | Both conditions measured under comparable protocol |
| **L3** | Pattern | "ΔΔp was positive in 4 of 5 windows" | Multiple measurement windows |
| **L4** | Trend | "ΔΔp has been positive across the last 3 experiments" | Multiple experiments with consistent protocol |
| **L5** | Hypothesis Support | "These observations are consistent with hypothesis H" | Explicit hypothesis stated before measurement |

**Forbidden at all levels:**
- Mechanism claims ("the uplift works because...")
- Capability claims ("the system can now...")
- Generalization claims ("this will work in...")
- Teleological language ("the system achieved...")

---

## 4. Reporting Template

When reporting CAL-EXP-3 results, use the following template. Do not embellish.

---

**CAL-EXP-3 Result Report**

> In experiment [ID], we measured divergence under [baseline condition] and [modified condition] over [N] cycles.
>
> **Observations:**
> - Baseline Δp (p50): [value]
> - Modified Δp (p50): [value]
> - ΔΔp: [value]
>
> **Interpretation (L2):** The modified condition was associated with [higher/lower/comparable] aggregate divergence relative to baseline in this measurement window.
>
> **Scope:** These observations apply to the experimental conditions described. No claims about generalization, capability, or mechanism are made.

---

**Example (filled):**

> In experiment CAL-EXP-3-001, we measured divergence under the unmodified runner (baseline) and the runner with external signal ingestion (modified) over 500 cycles.
>
> **Observations:**
> - Baseline Δp (p50): 0.025
> - Modified Δp (p50): 0.018
> - ΔΔp: 0.007
>
> **Interpretation (L2):** The modified condition was associated with lower aggregate divergence relative to baseline in this measurement window.
>
> **Scope:** These observations apply to the experimental conditions described. No claims about generalization, capability, or mechanism are made.

---

## 5. Non-Claims

The following claims are explicitly forbidden when interpreting CAL-EXP-3 results. This list aligns with the Topologist's spec forbidden claim inventory.

| Forbidden Claim Type | Example | Why Forbidden |
|---------------------|---------|---------------|
| **Mechanism** | "The uplift reduced divergence because it ingests better signals" | Experiments measure outcomes, not causes |
| **Capability** | "The system can now predict more accurately" | Capability is context-dependent; experiments measure specific conditions |
| **Learning** | "The system learned to reduce error" | No internal state change is measured |
| **Generalization** | "This improvement will transfer to production" | Transfer requires separate evidence |
| **Success** | "The experiment succeeded" | Success implies criteria not defined by the measurement |
| **Failure** | "The uplift failed" | Failure implies criteria not defined by the measurement |
| **Intelligence** | "The system is smarter with the modification" | Intelligence is not operationalized in the experiment |
| **Understanding** | "The system understands the external signals" | Understanding is not measured |
| **Value** | "This is a good/bad result" | Value judgments require operational context |
| **Permanence** | "The system is now more accurate" | Properties measured in windows are not permanent |

---

## 6. FAQ-Style Guardrails

### 6.1 Why does ΔΔp exist?

ΔΔp is the difference-of-means across experimental arms (baseline vs treatment). It exists because:

- **Δp alone is arm-local**: It tells you about divergence within one condition
- **ΔΔp enables comparison**: It quantifies whether the treatment arm differed from baseline
- **ΔΔp is not causal**: It shows association, not mechanism

Without ΔΔp, you cannot compare conditions. With ΔΔp, you can compare—but you still cannot claim causation.

### 6.2 When do we report Δp only?

Report Δp only (without ΔΔp) when:

- You are describing a single experimental arm
- No comparable baseline exists for the measurement window
- You are establishing a baseline for future comparison
- The comparison would be invalid (see 6.3)

Δp-only reporting is L1 (Aggregation). It describes one condition, not a comparison.

### 6.3 When is ΔΔp invalid?

ΔΔp is invalid when the comparison is not meaningful. Conditions that invalidate ΔΔp:

| Invalidity Condition | Why It Invalidates |
|---------------------|-------------------|
| **Toolchain drift** | Baseline and treatment used different code versions, configurations, or dependencies |
| **Corpus mismatch** | Baseline and treatment ran on different input data |
| **Window misalignment** | Baseline and treatment windows do not overlap or differ in length |
| **Protocol divergence** | Measurement methods differed between arms |
| **Environmental confound** | External factors (load, timing, resources) differed systematically |

When any of these conditions apply, do not compute or report ΔΔp. Report Δp for each arm separately with a note explaining why comparison is not valid.

---

## 7. Worked Example

### 7.1 Scenario

CAL-EXP-3-042 compared:
- **Baseline**: Learning OFF (static policy)
- **Treatment**: Learning ON (adaptive policy)

Both arms ran on the same corpus, same window (500 cycles), same toolchain version.

### 7.2 Measurements

| Metric | Value | Definition |
|--------|-------|------------|
| baseline_mean_Δp | 0.712 | Mean success probability signal (learning OFF) |
| treatment_mean_Δp | 0.748 | Mean success probability signal (learning ON) |
| ΔΔp | 0.036 | treatment_mean_Δp − baseline_mean_Δp |
| noise_floor (p95 of Δp variance) | 0.018 | Typical measurement variation |

### 7.3 Interpretation

- **ΔΔp (0.036) > noise_floor (0.018)**: The difference exceeds typical measurement noise
- **ΔΔp > 0**: Treatment arm had higher mean Δp than baseline in this window

This is an observation about measured values, not a claim about system properties.

### 7.4 Claim Level Determination

| Claim Level | Applicable? | Reason |
|-------------|-------------|--------|
| L0 (Raw Observation) | ✓ | Per-cycle Δp values recorded |
| L1 (Aggregation) | ✓ | Mean Δp computed for each arm |
| L2 (Comparison) | ✓ | ΔΔp computed; arms are comparable |
| L3 (Pattern) | ✗ | Only one window measured |
| L4 (Trend) | ✗ | Only one experiment |

**Maximum supported claim level: L2**

### 7.5 Approved Statement (L2)

> In CAL-EXP-3-042, the treatment arm (learning ON) was associated with higher mean Δp (0.748) compared to baseline (learning OFF, Δp = 0.712) over 500 cycles. ΔΔp = 0.036, which exceeds the noise floor of 0.018. No claims about mechanism, capability, or generalization are made.

**SHADOW MODE — observational only.**

### 7.6 Forbidden Statements

- ❌ "Learning improved the system" (mechanism claim)
- ❌ "The adaptive policy works" (capability claim)
- ❌ "Learning ON is better" (value claim)
- ❌ "This validates adaptive learning" (generalization claim)

---

## 8. Summary

- **δp**: Point divergence (one cycle)
- **Δp**: Aggregate divergence (one condition, one window)
- **ΔΔp**: Difference in aggregate divergence (two conditions)

All three are measurements, not judgments. They describe what was observed, not what it means.

---

**SHADOW MODE — observational only.**
