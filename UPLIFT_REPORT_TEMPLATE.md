# PHASE II UPLIFT REPORT: [Experiment ID]

- **Date:** `YYYY-MM-DD`
- **Experiment ID:** `(e.g., u2-analysis-2025-01-15-prop4)`
- **Slice ID:** `(e.g., prop_depth4)`
- **Author:** `(Your Name)`

---

## 1. Hypothesis

*(State the primary hypothesis for this experiment slice. Example:)*

> The RFL policy `[policy_version]` will demonstrate a statistically significant throughput uplift of at least `[T_from_spec]%` compared to the baseline policy on the `[slice_id]` problem distribution, while maintaining a success rate of at least `[X_from_spec]%` and an abstention rate no higher than `[Y_from_spec]%`.

---

## 2. Preregistration Reference

This experiment was conducted in accordance with the preregistration document:
- **Preregistration File:** `(e.g., PREREG_UPLIFT_U2.yaml)`

All statistical methods, success criteria, and governance procedures are defined in `UPLIFT_ANALYTICS_GOVERNANCE_SPEC.md`.

---

## 3. Results Summary

The analysis was performed on log files containing `[N_baseline]` baseline cycles and `[N_rfl]` RFL cycles.

*(Fill in the key metrics below by copying the relevant values from the `summary.json` artifact.)*

| Metric | Baseline | RFL | Delta / Uplift | 95% Confidence Interval |
|---|---|---|---|---|
| **Success Rate** | `[metrics.success_rate.baseline]` | `[metrics.success_rate.rfl]` | `[metrics.success_rate.delta]` | `[metrics.success_rate.ci]` |
| **Abstention Rate** | `[metrics.abstention_rate.baseline]` | `[metrics.abstention_rate.rfl]` | `[metrics.abstention_rate.delta]` | `[metrics.abstention_rate.ci]` |
| **Throughput (proofs/s)** | `[metrics.throughput.baseline_stat]` | `[metrics.throughput.treatment_stat]` | `[metrics.throughput.delta_pct]`% | `[metrics.throughput.delta_ci_low, metrics.throughput.delta_ci_high]` |

---

## 4. Governance Evaluation

Based on the **Confidence Readout Ladder** defined in the governance specification, the result of this experiment is:

**Governance Label: `[Strong Positive Effect | Positive Effect | Minor Positive Effect | Inconclusive | Negative Effect]`**

**Justification:**
*(Provide a brief explanation based on the ladder. Example:)*

> The 95% confidence interval for throughput uplift is `[CI_low, CI_high]`, which is entirely above the required threshold of `T%`. This corresponds to a **Strong Positive Effect**.

**Final Governance Checks:**

| Check | Result | Passed? |
|---|---|---|
| Sample Size | `[n_baseline, n_rfl]` >= `[min_samples]` | `[governance.details.sample_size_passed]` |
| Success Rate | `[rfl_success_rate]` >= `[min_success_rate]` | `[governance.details.success_rate_passed]` |
| Abstention Rate| `[rfl_abstention_rate]` <= `[max_abstention_rate]` | `[governance.details.abstention_rate_passed]` |
| Throughput | Lower CI bound >= `[min_throughput_uplift_pct]` | `[governance.details.throughput_uplift_passed]` |
| **Overall** | - | **`[governance.passed]`** |

**Recommendation: `[Proceed / Hold / Rollback]`**

---

## 5. Artifacts & Reproducibility

- **Analysis Code Commit:** `(Git commit hash used for analysis)`
- **`summary.json` Location:** `(Path to the canonical summary.json artifact)`
- **Input Logs:**
    - Baseline: `(Path to baseline.jsonl)`
    - RFL: `(Path to rfl.jsonl)`
- **Reproducibility Parameters:**
    - Bootstrap Seed: `[reproducibility.bootstrap_seed]`
    - Bootstrap Iterations: `[reproducibility.n_bootstrap]`
