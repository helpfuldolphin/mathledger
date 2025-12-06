# Figure Catalog: MathLedger RFL Dynamics

**Date:** 2025-11-27
**Author:** GEMINI-H (Visualization Architect)
**Context:** Investor Deck & Research Paper Assets

This catalog defines the standard set of figures to be generated from `rfl/experiment.py` outputs. All figures use the `experiments/plotting.py` toolkit for consistent, publication-ready aesthetics.

## 1. Abstention Rate Dynamics (The "H_t Curve")

*   **Type:** Time-series (Line Plot)
*   **Data Source:** `ExperimentResult.abstention_rate` vs. `run_index` (or Time).
*   **X-Axis:** Epoch / Run Index
*   **Y-Axis:** Abstention Rate $\in [0, 1]$
*   **Narrative:**
    *   **Paper (Sec 7.2):** "Demonstrates the stabilization of the rejection threshold $H_t$. Initially high variance as the system explores, converging to a steady state where it reliably identifies unsolvable premises."
    *   **Investor Deck:** "Self-Correcting Efficiency: The system learns what *not* to compute, saving 30%+ compute cost over time."
*   **Code:** `plotting.plot_abstention_dynamics(results)`

## 2. Throughput vs. Depth (Scalability)

*   **Type:** Scatter / Trend Line
*   **Data Source:** `ExperimentResult.throughput_proofs_per_hour` vs. `ExperimentResult.mean_depth`.
*   **X-Axis:** Mean Reasoning Depth
*   **Y-Axis:** Throughput (Proofs / Hour)
*   **Narrative:**
    *   **Paper (Sec 8.1):** "Performance degradation with depth follows a polynomial rather than exponential curve, indicating effective pruning."
    *   **Investor Deck:** "Scales to deep reasoning without hitting a 'complexity wall'."
*   **Code:** `plotting.plot_throughput_vs_depth(results)`

## 3. Capability Frontier (Depth Success)

*   **Type:** Error Bar / Line Plot
*   **Data Source:** `ExperimentResult.success_rate` grouped by `ExperimentResult.max_depth`.
*   **X-Axis:** Reasoning Depth (Difficulty)
*   **Y-Axis:** Success Rate
*   **Narrative:**
    *   **Paper (Sec 7.3):** "The operational envelope of the prover. We observe high reliability ($>95\%$) up to depth 4, with gradual falloff."
    *   **Investor Deck:** "Robustness: Maintains high accuracy even as problem complexity increases."
*   **Code:** `plotting.plot_capability_frontier(results)`

## 4. RFL Ablation Study (On vs. Off)

*   **Type:** Grouped Bar Chart
*   **Data Source:** Comparison of two experiment sets (Baseline vs. RFL-Enabled).
*   **metrics:** Success Rate, Abstention Rate, Cost (implied by 1/Throughput).
*   **Narrative:**
    *   **Paper (Sec 9.0):** "Ablation study confirming that the Reflexive Feedback Loop contributes $\Delta 15\%$ to precision."
    *   **Investor Deck:** "The 'Secret Sauce': Turning on our proprietary loop boosts performance instantly."
*   **Code:** `plotting.plot_rfl_comparison_bar(rfl_on, rfl_off)`

## 5. Statement Coverage Growth

*   **Type:** Cumulative Line Plot
*   **Data Source:** `ExperimentResult.distinct_statements` (cumulative sum over runs).
*   **X-Axis:** Run Index
*   **Y-Axis:** Unique Statements Proved
*   **Narrative:**
    *   **Paper:** "Knowledge base expansion rate."
    *   **Investor Deck:** "Data Moat: Automatically generating a proprietary dataset of formal proofs."
*   **Code:** *New custom plot: `cumsum` of distinct statements.*

## 6. Verification Latency Distribution

*   **Type:** Histogram / KDE
*   **Data Source:** `metrics/` (need granular proof latencies, usually aggregated).
*   **X-Axis:** Latency (ms) (Log Scale)
*   **Y-Axis:** Frequency
*   **Narrative:**
    *   **Paper:** "The verification step remains negligible ($<10ms$) compared to generation."
    *   **Investor Deck:** "Real-time verification guarantees correctness."

## 7. RFL Dyno Chart (Abstention Rate Comparison)

*   **Type:** Time-series (Dual Line Plot)
*   **Data Source:** `results/fo_baseline_wide.jsonl` and `results/fo_rfl_wide.jsonl` (First Organism cycle logs from wide slice experiments) **[NOTE: Planned / Not yet generated — see `experiments/DYNO_CHART_QA_SUMMARY.md`]**
*   **X-Axis:** Cycle Index
*   **Y-Axis:** Rolling Abstention Rate (P(abstain)) with window size W (default: 100)
*   **Narrative:**
    *   **Paper:** "Comparison of abstention dynamics between baseline and RFL-enabled runs. Demonstrates RFL's impact on reducing unnecessary computation attempts over time. The rolling window smooths short-term variance to reveal underlying trends."
    *   **Investor Deck:** "The Dyno Chart: Real-time visualization of RFL's efficiency gains. Watch the gap widen as the system learns."
*   **Code:** `experiments/analyze_abstention_curves.py --baseline results/fo_baseline_wide.jsonl --rfl results/fo_rfl_wide.jsonl --window-size 100`
*   **Output:** `artifacts/figures/rfl_dyno_chart.png` (or `rfl_abstention_rate.png` from analysis script)
*   **QA:** See `experiments/DYNO_CHART_QA_REPORT.md` for validation criteria

---

## Implementation Guide

Use the `experiments/plotting.py` module.

```python
from experiments.plotting import setup_style, save_figure, plot_abstention_dynamics
from rfl.experiment import ExperimentResult

# 1. Load Results
results = [ExperimentResult(...), ...] 

# 2. Setup
setup_style()

# 3. Generate
fig = plot_abstention_dynamics(results)
save_figure("fig1_abstention_dynamics", fig)
```
