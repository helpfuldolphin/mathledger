# U2 Dynamics Semantic Model
---

**PHASE II — GEMINI M — DYNAMICS–THEORY UNIFICATION ENGINE**

**DOCUMENT ID:** U2_DYNAMICS_SEMANTIC_MODEL.md

**AUTHOR:** Gemini M, Dynamics & Conjecture Analyst

**STATUS:** DRAFT

---

## 1. Abstract

This document provides the formal mathematical specification for the empirical estimators and behavioral classifications implemented in the `analysis/u2_dynamics.py` module. It serves as the canonical reference for translating raw experimental JSONL output into the semantic concepts defined in `RFL_UPLIFT_THEORY.md`. It concludes with a formal algorithm for automated conjecture testing, which unifies the empirical results with the theoretical framework.

---

## 2. Formal Definitions of Empirical Estimators

Let $\mathcal{R} = \{r_1, r_2, \ldots, r_T\}$ be a time-ordered series of $T$ JSONL records from a single experimental run. Each record $r_t$ is a dictionary corresponding to cycle $t$.

### 2.1 Abstention Rate Estimator, $\hat{A}(t)$

The empirical abstention rate $\hat{A}(t)$ at cycle $t$ is the maximum likelihood estimator of the true abstention rate $A(t)$, assuming a binomial process.

**Definition 2.1 (Abstention Rate Estimator):**
Given a record $r_t$ containing the fields `abstained_count` and `candidates_count`:
$$
\hat{A}(t) := \frac{r_t[\text{`abstained_count`}]}{r_t[\text{`candidates_count`}]}
$$
This corresponds to the `metrics.abstention_rate` field in the log schema and is extracted by the `estimate_A_t` function.

### 2.2 Uplift Gain Estimator, $\hat{G}_i$

The uplift gain $G_i$ for a slice $i$ is the reduction in abstention achieved by the RFL policy $\pi$ compared to the baseline policy $\pi_0$. The estimator $\hat{G}_i$ is the difference in the sample means of the abstention rates over the full duration of the experiment.

**Definition 2.2 (Uplift Gain Estimator):**
Given baseline records $\mathcal{R}_0$ and RFL records $\mathcal{R}_\pi$:
$$ 
\hat{G}_i := \bar{A}_0 - \bar{A}_\pi = \left(\frac{1}{T}\sum_{t=1}^{T} \hat{A}_0(t)\right) - \left(\frac{1}{T}\sum_{t=1}^{T} \hat{A}_\pi(t)\right) 
$$
This is computed by the `estimate_G_i` function, which also provides a 95% bootstrap confidence interval for this point estimate.

### 2.3 Policy Stability Index, $\Psi_T^{(w)}$

The Policy Stability Index measures the recent rate of change of the policy parameters, normalized by the parameter vector's magnitude. It quantifies policy convergence.

**Definition 2.3 (Policy Stability Index):**
Given a time series of policy parameter vectors $\{\theta_1, \ldots, \theta_T\}$, the stability index at time $T$ over a window $w$ is:
$$ 
\Psi_T^{(w)} := \frac{1}{w} \sum_{t=T-w+1}^{T} \frac{\|\theta_t - \theta_{t-1}\|_2}{\max(\|\theta_t\|_2, \epsilon)} 
$$ 
where $\epsilon$ is a small constant (e.g., $10^{-8}$) to ensure numerical stability. Low values of $\Psi$ indicate that the policy parameters have stabilized.

### 2.4 Oscillation Index, $\Omega_T$

The Oscillation Index measures the frequency of directional reversals in the policy update vector. It quantifies policy instability or "thrashing."

**Definition 2.4 (Oscillation Index):**
Given a series of policy updates $\Delta\theta_t = \theta_t - \theta_{t-1}$:
$$ 
\Omega_T := \frac{1}{T-2} \sum_{t=2}^{T-1} \mathbf{1}[(\Delta\theta_{t+1})^\top (\Delta\theta_t) < 0] 
$$ 
where $\mathbf{1}[\cdot]$ is the indicator function. A high value of $\Omega$ indicates that consecutive updates frequently oppose each other.

---

## 3. Formal Definitions of Behavior Classes

These classes categorize the dynamic behavior of the abstention rate time series, $\{\hat{A}(t)\}$, and are implemented by the `detect_pattern` function.

**Definition 3.1 (Stagnation):**
A time series exhibits **Stagnation** if its sample standard deviation is below a defined threshold, $\sigma_{stagnation}$.
$$ 
\text{Stagnation} \iff \text{Std}(\{\hat{A}(t)\}_{t=1}^T) < \sigma_{stagnation} 
$$

**Definition 3.2 (Negative Drift):**
A time series exhibits **Negative Drift** if there is a statistically significant monotonic downward trend. This is determined by the Kendall's Tau rank correlation coefficient.
$$ 
\text{Negative Drift} \iff \tau(\{\hat{A}(t)\}, \{t\}) < \tau_{drift} \quad \land \quad p_{\tau} < 0.05 
$$

**Definition 3.3 (Logistic-like Decay):**
A special case of Negative Drift that also exhibits an initial period of stability, characteristic of the idealized S-shaped learning curve.
$$ 
\text{Logistic-like Decay} \iff (\text{Negative Drift}) \quad \land \quad (\text{Stagnation is true for } \{\hat{A}(t)\}_{t=1}^{T/5}) 
$$

**Definition 3.4 (Step-Function Decay):**
A time series exhibits **Step-Function Decay** if its evolution is characterized by sudden, large drops rather than a smooth trend.
$$ 
\text{Step-Function Decay} \iff \max_t(|A(t) - A(t-1)|) > \kappa \cdot \text{median}_t(|A(t) - A(t-1)|) 
$$
where $\kappa$ is a multiplier (e.g., 3) that identifies a jump as an outlier relative to typical cycle-to-cycle noise.

**Definition 3.5 (Oscillation):**
A time series exhibits **Oscillation** if it is not stagnant but also does not show a clear negative drift. It is the default classification for a dynamic but non-convergent series. More formally:
$$ 
\text{Oscillation} \iff \text{not Stagnation} \quad \land \quad \text{not Negative Drift} \quad \land \quad \Omega_T > \omega_{osc} 
$$

---

## 4. Threshold Derivation and Justification

The thresholds used in the `detect_pattern` function are critical for robust classification. They are heuristics derived from first principles of signal processing and experimental design.

| Parameter | Symbol | Value | Justification |
|---|---|---|---|
| Stagnation Std Dev | $\sigma_{stagnation}$ | 0.01 | This value is chosen to be just above the expected instrumentation noise floor. A standard deviation of 1% in the abstention rate implies that over 95% of observations (assuming normality) fall within a $\pm 2\%$ band. Any true learning signal should produce variance that significantly exceeds this baseline noise level. |
| Drift Correlation | $\tau_{drift}$ | -0.2 | Kendall's Tau ranges from -1 (perfect negative correlation) to 1. A value of -0.2 represents a weak but noticeable downward trend. It is chosen to be sensitive enough to detect real learning but robust enough to reject spurious correlations arising from random walk behavior in a noisy time series. |
| Oscillation Index | $\omega_{osc}$ | 0.3 | An index of 0.3 means the policy update vector reverses its direction in 30% of cycles. In a smooth, multi-dimensional gradient descent, some reversal is expected due to path curvature. However, a rate this high suggests that a significant fraction of updates are counter-productive, indicating either an overly large learning rate or a pathological, non-convex objective landscape. |
| Step-Function Multiplier | $\kappa$ | 3.0 | This is a standard statistical heuristic for outlier detection. A single-cycle change that is more than three times the median change is unlikely to be part of the normal noise distribution and is therefore treated as a significant event, such as the discovery of a key lemma. |

---

## 5. Conjecture Testing Algorithm Specification

This algorithm provides a systematic procedure for evaluating the conjectures in `RFL_UPLIFT_THEORY.md` against the empirical evidence from a completed U2 experiment.

**Algorithm 1: Automated Conjecture Test**
---
**Inputs:**
-   Baseline records: $\mathcal{R}_0 = \{r_{0,t}\}_{t=1}^T$
-   RFL records: $\mathcal{R}_\pi = \{r_{\pi,t}\}_{t=1}^T$
-   Slice-specific uplift threshold: $\tau_i$
-   List of conjectures to test: $\mathcal{C} = \{C_1, C_2, \ldots\}$

**Output:**
-   A report mapping each conjecture $C_j$ to an evidential status: `{SUPPORTS, CONTRADICTS, CONSISTENT, INCONCLUSIVE}` with supporting data.

**Procedure:**

1.  **Data Extraction:**
    a.  $\hat{A}_0 \leftarrow \text{estimate_A_t}(\mathcal{R}_0)$
    b.  $\hat{A}_\pi \leftarrow \text{estimate_A_t}(\mathcal{R}_\pi)$

2.  **Experiment Validity Check:**
    a.  `baseline_pattern` $\leftarrow$ `detect_pattern`($\hat{A}_0$)
    b.  **If** `baseline_pattern` is 'Stagnation' **and** $\text{mean}(\hat{A}_0) > 0.95$:
        i.  **Return** "Experiment Outcome: **DEGENERATE**. All conjectures **INCONCLUSIVE**."

3.  **Primary Metric Calculation:**
    a.  `uplift_results` $\leftarrow$ `estimate_G_i`($\mathcal{R}_0$, $\mathcal{R}_\pi$)
    b.  $\hat{G}_i \leftarrow$ `uplift_results['delta']`
    c.  `ci` $\leftarrow$ (`uplift_results['ci_95_lower']`, `uplift_results['ci_95_upper']`)

4.  **Behavioral Classification:**
    a.  `rfl_pattern` $\leftarrow$ `detect_pattern`($\hat{A}_\pi$)

5.  **Conjecture Evaluation Loop:**
    a.  Initialize `report` = {}
    b.  **For each** conjecture $C_j$ in $\mathcal{C}$:
        i.   **Switch** ($C_j$):
             *   **Case: Conjecture 3.1 (Supermartingale Property / Negative Drift):**
                 - **If** `rfl_pattern` is 'Negative Drift' or 'Logistic-like Decay': `status` $\leftarrow$ **SUPPORTS**.
                 - **Else if** `rfl_pattern` is 'Stagnation' or 'Oscillation': `status` $\leftarrow$ **CONTRADICTS**.
                 - **Else**: `status` $\leftarrow$ **INCONCLUSIVE**.
             *   **Case: Conjecture 4.1 (Logistic Decay):**
                 - **If** `rfl_pattern` is 'Logistic-like Decay': `status` $\leftarrow$ **SUPPORTS**.
                 - **Else if** `rfl_pattern` is 'Negative Drift': `status` $\leftarrow$ **CONSISTENT** (learning occurred, but not necessarily logistic).
                 - **Else**: `status` $\leftarrow$ **CONTRADICTS**.
             *   **Case: Conjecture 6.1 (Convergence):**
                 - **If** `rfl_pattern` is 'Stagnation' and $\text{mean}(\hat{A}_\pi[-T/5:]) > 0.1$: `status` $\leftarrow$ **CONTRADICTS** (plateaued too high).
                 - **Else if** $\hat{A}_\pi[T] < 0.1$ and `rfl_pattern` shows downward trend: `status` $\leftarrow$ **CONSISTENT**.
                 - **Else**: `status` $\leftarrow$ **INCONCLUSIVE** (experiment may not have run long enough).
             *   **Case: Phase II Uplift (General):**
                 - **If** $\hat{G}_i > \tau_i$ and `ci[lower]` > 0: `status` $\leftarrow$ **SUPPORTS**.
                 - **Else if** $\hat{G}_i > 0$ and `ci[lower]` > 0 but $\hat{G}_i \le \tau_i$: `status` $\leftarrow$ **CONSISTENT** (positive but not conclusive uplift).
                 - **Else**: `status` $\leftarrow$ **CONTRADICTS**.
        ii.  `report[C_j]` = `{ "status": status, "evidence": { "pattern": rfl_pattern, "gain": \hat{G}_i, "ci": ci } }`

6.  **Return** `report`
---
