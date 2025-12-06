# Imperfect Verifier Simulation

**EXPERIMENTAL ONLY -- DO NOT USE IN PRODUCTION**

## Motivation

This module provides a controlled simulation of **verifier bias**, denoted as $\epsilon_v$ in the associated research paper ("Recursive Formal Learning: Convergence Bounds").

The theoretical model posits that a verifier $V$ may not be perfectly sound or complete. Specifically, it may exhibit stochastic errors:
*   **False Positives:** Validating a false statement.
*   **False Negatives:** Rejecting a true statement.

The paper proves that under certain conditions, the recursive learning process converges to a bounded error set $J^* + C\epsilon_v$. This simulation allows us to empirically verify the shape of this error bound by artificially injecting noise into the verification step.

## Implementation Details

### The Model
We model the imperfect verifier $V_{\epsilon}$ as a wrapper around the ground-truth verifier $V_{true}$. For any statement $\phi$:

1.  Compute actual validity $v = V_{true}(\phi)$.
2.  Draw a random variable $X \sim \text{Bernoulli}(\epsilon)$.
3.  If $X=1$, flip the result ($V_{\epsilon}(\phi) = \neg v$).
4.  If $X=0$, return the true result ($V_{\epsilon}(\phi) = v$).

This corresponds to a symmetric noise channel with error probability $\epsilon$.

### Directory Structure
*   `tools/simulation/noisy_verifier.py`: Implementation of `NoisyVerifierWrapper`.
*   `tools/simulation/run_imperfect_verifier_experiment.py`: Harness for running synthetic experiments.
*   `tools/simulation/analyze_imperfect_verifier.py`: Tool to aggregate and tabularize results.
*   `experiments/results/`: Destination for experiment logs.

## Running Experiments

### 1. Synthetic "Tier 1" Experiment
Run the simulation with a specific epsilon and sample size.

```bash
# Baseline (No noise)
python tools/simulation/run_imperfect_verifier_experiment.py --epsilon 0.0 --n-samples 10000 --output experiments/results/iv_eps_0.0.jsonl

# Small Noise (epsilon = 0.01)
python tools/simulation/run_imperfect_verifier_experiment.py --epsilon 0.01 --n-samples 10000 --output experiments/results/iv_eps_0.01.jsonl

# Larger Noise (epsilon = 0.05)
python tools/simulation/run_imperfect_verifier_experiment.py --epsilon 0.05 --n-samples 10000 --output experiments/results/iv_eps_0.05.jsonl
```

### 2. Analysis
Generate a summary table from the result files.

```bash
python tools/simulation/analyze_imperfect_verifier.py experiments/results/iv_eps_*.jsonl
```

## Expected Results
We expect the False Positive Rate (FPR) and False Negative Rate (FNR) to scale linearly with $\epsilon$.
specifically, $FPR \approx \epsilon$ and $FNR \approx \epsilon$ in this symmetric model.

## Safety Constraints
*   **Isolation:** These tools must `never` be imported by the production `ledger` or `derivation` pipelines.
*   **Storage:** Results are written only to ephemeral JSONL files in `experiments/`, never to the immutable ledger.
