# How to Read Divergence Metrics Without Overclaiming

**Version:** 1.0.0
**Status:** Advisory Reference
**Author:** CLAUDE D — Metrics Conformance Layer
**Date:** 2025-12-13

---

## Purpose

This document provides guidance on interpreting P4 divergence metrics. It is advisory only and does not establish pass/fail criteria, thresholds, or enforcement rules.

---

## Core Principle

**Divergence is observation, not judgment.**

A divergence between twin-predicted and real-observed values indicates a difference—not necessarily an error, failure, or problem. The purpose of divergence tracking is to surface patterns for human review, not to automate decisions.

---

## Common Misreads and Corrections

| Misread | Correction |
|---------|------------|
| "High divergence means the model is broken" | Divergence indicates difference, not defect. The real system may have changed, or the model may need recalibration. |
| "Low divergence proves the model is correct" | Low divergence means predictions match observations in the measured window. It does not validate the model's theoretical correctness. |
| "p95 spike = critical failure" | A p95 spike indicates tail behavior in one window. It may reflect transient load, data variance, or measurement noise. Observe across multiple windows before drawing conclusions. |
| "Divergence trending up = system degrading" | Upward trends suggest increasing prediction-observation gap. This could indicate model drift, environmental change, or improved real-system performance. Context determines meaning. |
| "Zero divergence is the goal" | Perfect agreement is neither expected nor required. Some divergence is normal in any predictive system. |

---

## Reading Percentiles

| Metric | What It Shows | What It Does Not Show |
|--------|---------------|----------------------|
| p50 (median) | Typical divergence magnitude | Tail behavior, outliers |
| p95 | Worst-case divergence in 95% of observations | Whether outliers are problematic |
| p50 shift | Change in central tendency across windows | Root cause of shift |
| p95 spike | Presence of high-divergence outliers | Whether outliers require action |

---

## Interpretation Guidelines

1. **Compare across windows, not single points.** One high value is not a pattern.

2. **Consider external factors.** Workload changes, infrastructure updates, and data distribution shifts can all affect divergence without indicating model error.

3. **Distinguish observation from action.** Divergence metrics inform investigation; they do not prescribe remediation.

4. **Avoid causal claims without evidence.** "Divergence increased" is an observation. "The model caused divergence" is a claim requiring investigation.

5. **Document uncertainty.** When reporting divergence patterns, state what is observed and what remains unknown.

---

## What This Document Does Not Do

- Define acceptable divergence ranges
- Establish pass/fail criteria
- Authorize automated responses to divergence
- Replace human judgment in interpreting results

---

## Reference

For experimental criteria and calibration procedures, see CAL_EXP_2_* documents.

---

*End of Advisory Note*
