# Capability Metrics Definition (GEMINI-K)

We track three "low-hanging" capability metrics to assess prover improvement at the Propositional Logic (PL) layer.

## 1. Proofs per Hour (Throughput)
**Definition:** The number of successful proofs generated divided by the duration of the experiment in hours.
- **Formula:** `Successful Proofs / ((End Time - Start Time) in Hours)`
- **Why:** Direct measure of efficiency and metabolic rate.
- **Constraint:** Must be measured over a fixed time budget or fixed statement budget to be comparable.

## 2. Max Depth
**Definition:** The maximum derivation depth of any verified statement produced during the run.
- **Formula:** `MAX(statement.depth)`
- **Why:** Proxies for the "complexity" or "intelligence" of the prover. A prover reaching depth 10 is exploring deeper entailments than one stuck at depth 3.

## 3. Abstention Rate (Precision Proxy)
**Definition:** The percentage of generated statements that the prover *failed* to verify or chose to abstain from (if explicit abstention exists), relative to total generated statements.
- **Formula:** `(Total Statements - Successful Proofs) / Total Statements`
- **Why:** In a "deductive metabolism," waste (unverified statements) is costly. A better prover generates fewer duds.
- **Note:** This also serves as a proxy for "Lemma Reuse/Dedupe" if we consider "re-proving knowns" as a form of abstention or low-value work, though explicit dedupe tracking is a future enhancement.

---

# Experiment Design: Baseline vs. RFL

## Objective
Quantify the uplift provided by the Reflexive Formal Learning (RFL) policy compared to a Baseline (Random/BFS) prover.

## Configuration
- **Budget:** Fixed `max_total` (e.g., 1000 statements) or fixed Time (e.g., 5 minutes).
- **Baseline:** `system_id=1` (PL), Default Derivation Engine (Random/BFS).
- **RFL:** `system_id=1` (PL), Derivation Engine + Learned Policy (if available) or Guided Heuristics.

## Analysis Script
We have provided `experiments/analyze_capability_metrics.py` to automate this comparison.

### Usage
```bash
# Run with a budget of 500 statements per trial, 3 trials per mode
python experiments/analyze_capability_metrics.py --budget 500 --trials 3
```

### Expected Output
The script will output a table comparing Baseline vs. RFL on the three metrics, including Mean, Standard Deviation, and the Throughput Uplift percentage.

## Success Criteria
- **Throughput:** RFL > 1.1x Baseline
- **Abstention:** RFL < 0.9x Baseline
- **Depth:** RFL >= Baseline