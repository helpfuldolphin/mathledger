Status: NOT YET RUN

# Experimental Design: RFL Imperfect Verifier Simulation (IVS)

## Phase-I Reality Check

It is critical to note that existing `fo_rfl.jsonl` logs from Phase-I experiments will *not* exhibit the divergence effects predicted by this design. This is due to two primary factors:
1.  **Lean Prover Disabled:** The formal prover (Lean) was not actively engaged in the feedback loop, thus preventing the RFL agent from generating complex, verifiable theorems.
2.  **Perfect Verifier Used:** All Phase-I experiments relied exclusively on the perfect Mirror Auditor ($V_{true}$), meaning no imperfect verifier was introduced to perturb the learning process.

Consequently, previous RFL logs represent a baseline under ideal verification conditions, and are not expected to show sensitivity to noise parameters.

## 1. Statistical Grounding & Sampling Strategy

### Sample Size ($N$) Selection
To distinguish noise-induced drift from stochastic variance with $\alpha=0.05$ and power $1-\beta=0.95$, we model the verification outcome as a Bernoulli process.
For a minimum detectable effect size (MDE) of $\delta = 0.005$ (0.5% drift) at base error rate $\epsilon=0.01$:
$$ N \geq \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot \sigma^2}{\delta^2} $$
Approximation suggests **$N = 10,000$** samples per epoch is sufficient to bound the 95% Confidence Interval (CI) within $\pm 0.2\%$ accuracy.

### $\epsilon$ (Noise) Schedule
We define a logarithmic progression of bias injection to map the phase transition from stable learning to collapse:
*   **Baseline:** $\epsilon = 0.00$ (Control, existing Mirror Auditor)
*   **Perturbation:** $\epsilon \in \{0.001, 0.005, 0.01\}$ (Low noise regime)
*   **Degradation:** $\epsilon \in \{0.02, 0.05, 0.10\}$ (High noise regime)
*   **Collapse:** $\epsilon = 0.25$ (Theoretical maximum entropy threshold for binary classification)

### Confidence Intervals
We adhere to the **Wilson Score Interval** for binomial proportions, as it provides better coverage for small $\epsilon$ than the standard Wald interval.
$$ CI = \frac{1}{1 + \frac{z^2}{n}} \left( \hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \right) $$
*Rationale: Essential for accurately bounding FNR when $\epsilon \to 0$.*

## 2. Expected Confusion Matrix Trajectories

We analyze the divergence of the Imperfect Verifier ($V_{\epsilon}$) from the Ground Truth ($V_{GT}$).
Let $C(\epsilon)$ be the confusion matrix at noise level $\epsilon$.

### Linear Regime (No Learning Feedback)
If $V_{\epsilon}$ is a simple symmetric noise channel:
*   **TP (True Positives):** $\downarrow$ Linearly ($1 - \epsilon$)
*   **FP (False Positives):** $\uparrow$ Linearly ($\epsilon$)
*   **TN (True Negatives):** $\downarrow$ Linearly ($1 - \epsilon$)
*   **FN (False Negatives):** $\uparrow$ Linearly ($\epsilon$)

### Recursive Regime (RFL Feedback Loop)
Inside the RFL loop, we hypothesize a **Non-Linear Amplification**:
*   **False Positive Drift:** $\text{FP}_{RFL}(\epsilon) > \epsilon$. The learner will "exploit" the broken verifier, generating non-theorems that statistically maximize $V_{\epsilon}$'s acceptance.
*   **Trajectory:** Super-linear growth of FP as $\epsilon$ crosses a "Brittleness Threshold".

## 3. Detecting Brittleness & Overfitting

### Metric: The Brittleness Ratio ($B_{\epsilon}$)
$$ B_{\epsilon} = \frac{\text{Observed Error Rate}}{\epsilon} $$
*   **Stable:** $B_{\epsilon} \approx 1$
*   **Brittle:** $B_{\epsilon} > 1.5$ (System amplifies noise)
*   **Robust:** $B_{\epsilon} < 1$ (System corrects noise via ensemble/consistency checks)

### Metric: Noise Overfitting Index (NOI)
To detect if the agent is learning the noise pattern:
1.  Freeze $V_{\epsilon}$ seed (deterministic noise).
2.  Train Agent $A$.
3.  Test $A$ on $V_{\epsilon}'$ (same $\epsilon$, different seed).
$$ \text{NOI} = \text{Accuracy}(A, V_{\epsilon}) - \text{Accuracy}(A, V_{\epsilon}') $$
*   High NOI indicates the agent has memorized the specific stochastic flaws of the verifier rather than learning valid proofs.

## 4. Research Deliverables Specification

To satisfy the "Verifier Robustness" section of the paper, we require:

| Artifact ID | Path/Name | Content Definition |
|:---|:---|:---|
| **D-IVS-01** | `results/ivs_phase_transition.csv` | Table of $\epsilon$ vs. Final Accuracy, F1, and $B_{\epsilon}$ after $K$ epochs. |
| **D-IVS-02** | `figures/brittleness_curve.png` | Plot of Accuracy decay. X-axis: $\epsilon$ (log scale), Y-axis: Accuracy. Overlay: $y=1-\epsilon$ reference. |
| **D-IVS-03** | `audit/noise_exploitation_samples.jsonl` | List of false theorems accepted by $V_{\epsilon}$ but rejected by $V_{GT}$, ranked by "confidence". |
| **D-IVS-04** | `config/ivs_experiment.yaml` | Reproducible config file defining the N, $\epsilon$ schedule, and fixed seeds. |

## 5. Safety & Isolation (MDAP Compliance)
*   **Monotone:** Logs must be append-only.
*   **Deterministic:** Pseudo-random number generators for noise must be seeded explicitly in `config/ivs_experiment.yaml`.
*   **Attestation:** All "Imperfect" runs must be flagged with metadata `{"verifier_mode": "simulation", "epsilon": <val>}` to prevent contamination of the main Ledger.

## Future Work

All noise-schedule analysis remains Phase II.

Future work will involve executing this experimental design to empirically validate the theoretical convergence bounds of RFL under verifier bias. We will also explore advanced metrics for quantifying model brittleness and noise overfitting within recursive learning systems, ultimately aiming to develop noise-resilient attestation protocols.