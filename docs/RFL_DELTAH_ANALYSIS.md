# RFL Delta H Analysis: Hash State Drift

**Status**: Exploratory / Sanity Check
**Metric**: $\Delta H \propto N_v^{-\beta}$

This document defines the protocol for measuring "Hash State Drift" ($\Delta H$) in the MathLedger First Organism. This is a **cryptographic sanity check**, not an epistemic scaling claim.

## 1. Metric Definitions

*   **$N_v(t)$ (Verified Volume)**: The cumulative count of successful proofs integrated into the ledger at the moment block $t$ was sealed.
*   **$H_t$ (System State)**: The `composite_attestation_root` (SHA-256, 256 bits) of block $t$. It aggregates reasoning ($R_t$) and UI events ($U_t$).
*   **$\\Delta H(t)$ (State Drift)**: The **Hamming distance** (in bits) between $H_t$ and $H_{t-1}$. Range: $[0, 256]$.

## 2. Null Hypothesis & Interpretation

For a healthy system using SHA-256 with proper domain separation:

*   **Null Hypothesis**: The system state performs a random walk in hash space.
    *   $\beta \approx 0$ (No scaling with volume)
    *   Avg $\\Delta H \approx 128$ bits (Gaussian noise centered at N/2)

*   **Alternative Hypothesis (Pathology)**:
    *   $\beta > 0$: The system is "cooling down" or converging to a fixed point (stagnation).
    *   Avg $\\Delta H \ll 128$: The hash function is biased, or we are sealing identical/near-identical blocks.
    *   Avg $\\Delta H \gg 128$: (Statistically unlikely for SHA-256).

**Conclusion**: If $\\beta \neq 0$, we likely have a tooling bug, a logging truncation issue, or a broken merkle implementation.

## 3. How to Run

### Step 1: Extract Data
Query the live database (read-only) to get full-fidelity hashes.

```bash
# Filter by system_id (default=1)
uv run python scripts/extract_ht_data.py --output reports/ht_dynamics.csv --system-id 1
```

### Step 2: Analyze
Compute drift statistics and fit the scaling law.

```bash
uv run python scripts/analyze_ht_scaling.py --input reports/ht_dynamics.csv
```

## 4. Pilot Run Results

*Date: 2025-11-27*
*Note: Synthetic data used for validation; live database was unreachable.*

```text
=== Hash State Drift Analysis (Exploratory) ===
Data points: 4 transitions (from 5 blocks)
Filters: min_nv=10, min_dh=1
Range N_v: 20 -> 100
Range dH:  128 -> 256

--- Statistics ---
Avg Delta H:    191.75 bits (Expected ~128)
Std Delta H:    63.75
Median Delta H: 191.50

--- Scaling Law Fit ---
Equation: Delta H ~ N_v ^ -beta
Beta:      0.3249 (Expected ~0.0)
Intercept: 596.3989

--- Interpretation ---
[WARN] Average drift is far from 128. Possible bias or non-random hashing.
[WARN] Non-zero scaling (beta=0.3249). Potential pathology or artifact.
```
