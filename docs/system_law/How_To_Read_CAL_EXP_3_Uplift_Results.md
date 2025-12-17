# How to Read CAL-EXP-3 Uplift Results (Without Overclaiming)

**Version:** 1.0.0
**Status:** Advisory Reference
**Author:** Interpretation Hygiene Agent
**Date:** 2025-12-13

---

## Purpose

This document provides guidance on interpreting CAL-EXP-3 uplift results. It is advisory only and does not establish acceptance criteria, thresholds, or enforcement rules. The goal is to reduce hype and prevent overclaiming.

---

## Uplift vs Calibration: What's the Difference?

| Concept | Definition | What It Measures |
|---------|------------|------------------|
| **Calibration** | Adjusting model parameters to minimize prediction-observation divergence | How well the model matches reality within its current structure |
| **Uplift** | Structural changes to the model (new features, external signals, architectural modifications) | Whether a structural change improves prediction-observation alignment |

**Key distinction:**
- Calibration tunes existing knobs
- Uplift adds new knobs or changes the machine

An uplift experiment tests whether a structural modification reduces divergence. A positive result means the modification helped *in this context*—nothing more.

---

## Why Δp > 0 Does NOT Imply General Intelligence

A positive delta-p (Δp > 0) means the modified model produced smaller prediction errors than the baseline in the measured window. This is a narrow, operational observation.

**What Δp > 0 actually says:**
- In this experiment, with this data, using this metric, the modification reduced error

**What Δp > 0 does NOT say:**
- The system is "smarter"
- The system "understands" anything
- The system will perform better on different tasks
- The system has generalized capability
- The modification will help in production
- The improvement will persist over time

**Analogy:** A thermostat that reduces temperature variance by 0.1°C has improved on one metric. This does not make it intelligent, wise, or capable of regulating anything else.

---

## Common Misreads and Corrections

| Misread | Correction |
|---------|------------|
| "Δp > 0 proves the uplift works" | Δp > 0 indicates improvement in the measured context. Whether the uplift "works" depends on operational requirements not defined by the experiment. |
| "Larger Δp means better uplift" | Δp magnitude reflects the specific measurement conditions. A large Δp in a narrow context may be less useful than a small Δp in a broad context. |
| "The system learned something" | The experiment measures prediction error, not learning. No claims about internal representations are supported. |
| "This generalizes to other domains" | Uplift results are specific to the experimental conditions. Generalization requires separate evidence. |
| "Negative Δp means the uplift failed" | Negative Δp means the modification increased error in this context. This may indicate the modification is inappropriate here, not that it is universally wrong. |
| "We can now claim capability X" | Experimental results describe behavior in controlled conditions. Capability claims require broader validation. |
| "The model is now more accurate" | The model produced smaller errors in this window. Accuracy is context-dependent and not a permanent property. |

---

## What Uplift Experiments Can and Cannot Tell You

### Can Tell You
- Whether a structural change reduced prediction error in the experimental window
- The magnitude of error reduction under experimental conditions
- Whether the change introduced instability or variance

### Cannot Tell You
- Whether the change will help in production
- Whether the change generalizes to other contexts
- Whether the system has acquired any "capability"
- Anything about the system's "intelligence"
- Whether the improvement justifies the complexity cost

---

## Interpretation Guidelines

1. **Report observations, not interpretations.** "Δp = 0.015 in 500 cycles" is an observation. "The system improved" is an interpretation requiring additional context.

2. **Scope claims to experimental conditions.** Results apply to the specific data, window, and configuration tested.

3. **Avoid teleological language.** The system did not "achieve" or "succeed at" anything. Metrics changed.

4. **Distinguish statistical from practical significance.** A measurable difference may not be operationally meaningful.

5. **Acknowledge what you don't know.** Experiments constrain uncertainty; they do not eliminate it.

---

## What This Document Does Not Do

- Define acceptable Δp values
- Establish uplift acceptance criteria
- Authorize claims about system capabilities
- Replace domain-specific evaluation
- Provide production readiness assessment

---

## Reference

For experimental design and calibration methodology, see CAL_EXP_* documents in `docs/system_law/calibration/`.

---

*This document is advisory. It reduces hype; it does not generate it.*
