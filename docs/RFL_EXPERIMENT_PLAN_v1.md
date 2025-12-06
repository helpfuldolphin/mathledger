# RFL Experiment Plan v1.0: Empirical Verification of Reflexive Formal Learning

**Status:** DRAFT
**Author:** GEMINI-A (RFL Experiment Orchestrator)
**Date:** 2025-11-27
**Context:** MathLedger VCP 2.2 Integration

## 1. Executive Summary
This document defines the experimental campaign to verify the core claims of the MathLedger Research Paper regarding **Reflexive Formal Learning (RFL)**. We utilize the existing "First Organism" architecture (UI $\rightarrow$ Derivation $\rightarrow$ Attestation $\rightarrow$ RFL Metabolism) to measure the descent of epistemic risk and the stabilization of the dual-attestation ledger.

The goal is to produce the data required for Section 7 (Experimental Results) of the paper without modifying core logic.

## 2. Mapping Theory to Observables

We map the abstract quantities from the RFL Update Law ($ \Delta \theta \propto -\nabla_{\theta} \mathcal{R} $) to concrete system metrics.

| Research Claim | Abstract Symbol | Concrete Observable (Code/DB) | Metric Source |
| :--- | :--- | :--- | :--- |
| **Epistemic Risk** | $R_t$ (Risk) | `abstention_fraction` | `RFLRunner.policy_ledger` |
| **Descent** | $-\frac{dR}{dt}$ | `symbolic_descent` | `RFLRunner.policy_ledger` |
| **Utility** | $U_t$ | `throughput` (proofs/hr) | `rfl_results.json` |
| **Integrity** | $H_t$ | `composite_attestation_root` | `attestation/dual_root.py` |
| **Policy State** | $\theta_t$ | `{derive_steps, max_breadth}` | `RFLConfig` / `CurriculumSlice` |
| **Metabolism** | $\Psi$ | `verify_metabolism()` pass/fail | `rfl/bootstrap_stats.py` |

## 3. Experimental Campaign

We define three experiments (R0–R2) to characterize the system.

### Experiment R0: Baseline Integrity & Noise Floor (Hermetic)
*   **Goal:** Establish the baseline variance of metrics in the absence of active learning or curriculum shifts. Verify H_t stability.
*   **Hypothesis ($H_0$):** In repeated identical runs, `symbolic_descent` should oscillate around 0, and `abstention_fraction` should remain stable (modulo DB growth).
*   **Configuration:**
    *   `num_runs`: 10
    *   `curriculum`: Single slice "baseline" (constant parameters).
    *   `derive_steps`: 10 (Minimal)
*   **Run Budget:** ~5 minutes, local CPU.
*   **Status:** **Paper-Ready** (Control Group).

### Experiment R1: The Descent of Risk (Accumulation)
*   **Goal:** Observe the "Descent of Risk" as the system accumulates theorems (expands the frontier).
*   **Hypothesis ($H_1$):** Over $N=20$ runs, as the knowledge base grows, the system finds proofs for previously reachable statements more efficiently, or `throughput` increases, leading to `symbolic_descent` > 0 (averaged).
*   **Configuration:**
    *   `num_runs`: 20
    *   `curriculum`: "Ramp" (Gradually increasing `max_total` to allow frontier expansion).
    *   `bootstrap_replicates`: 10,000 (High precision).
*   **Run Budget:** ~30-60 minutes, local CPU + Postgres.
*   **Status:** **Paper-Ready** (Main Result).

### Experiment R2: Adaptive Resilience (Curriculum Shock)
*   **Goal:** Test the system's response to a sudden increase in problem complexity (simulating a "Curriculum Shock").
*   **Hypothesis ($H_2$):** A sharp increase in `max_breadth` (Run 10) will cause a spike in `abstention_fraction` (Risk), followed by a recovery (Descent) in subsequent runs as the system digests the new breadth.
*   **Configuration:**
    *   `num_runs`: 20
    *   `curriculum`:
        *   Runs 1-9: "Warmup" (Breadth 50)
        *   Runs 10-20: "Shock" (Breadth 200)
*   **Run Budget:** ~45 minutes.
*   **Status:** **Exploratory** (Robustness Check).

## 4. Proposed Implementation Structure

We will create a dedicated `experiments/` directory to keep the core codebase clean.

```text
experiments/
├── rfl/
│   ├── configs/
│   │   ├── r0_baseline.json      # Config for R0
│   │   ├── r1_descent.json       # Config for R1
│   │   └── r2_shock.json         # Config for R2
│   ├── runner.py                 # Wrapper around rfl.runner.RFLRunner
│   └── analysis/
│       ├── plot_descent.py       # Generates R(t) vs t plots
│       └── plot_phase_space.py   # Generates Risk vs Utility plots
└── README.md
```

### Sample Run Command
```bash
# Run Experiment R1
python experiments/rfl/runner.py --config experiments/rfl/configs/r1_descent.json
```

## 5. Safe Modification Points (Allowlist)

The following areas are safe for later agents to modify to support these experiments without triggering a full system regression test:

1.  **`rfl/config.py`**: Adding new fields to `RFLConfig` or `CurriculumSlice` (e.g., `shock_factor`) is safe if they have defaults.
2.  **`rfl/runner.py` (Logging)**: Adding `logger.info` or `_increment_metric` calls is safe.
3.  **`rfl/experiment.py` (Exports)**: Adding keys to the `to_dict()` return value is safe.
4.  **`experiments/*`**: Any new files in the experiments directory are safe.

## 6. Wide Slice RFL Integration

**Status:** Phase II / Infrastructure Ready (No experimental data generated yet)

### Overview
The Wide Slice RFL configuration (`configs/rfl_experiment_wide_slice.yaml`) is designed to complement FO cycle logs with RFL-level metrics for research analysis. It targets the `slice_medium` curriculum slice and operates in hermetic mode (no database required).

**Note:** This configuration exists as infrastructure, but no Wide Slice experimental runs have been executed yet. The existing Phase I evidence uses `first-organism-pl` slice (see `results/fo_rfl_50.jsonl`, `results/fo_baseline.jsonl`). Wide Slice experiments targeting `slice_medium` are planned for Phase II.

### Configuration
- **File:** `configs/rfl_experiment_wide_slice.yaml` (YAML) and `configs/rfl_experiment_wide_slice.json` (JSON)
- **Target Slice:** `slice_medium` (runs 11-30 in curriculum.yaml)
- **Runs:** 30 (configurable 20-50 range)
- **Derivation Parameters:** Conservative step sizes matching slice_medium:
  - `derive_steps`: 50
  - `max_breadth`: 100
  - `max_total`: 500
  - `depth_max`: 4
- **Hermetic Mode:** `database_url: ""` and `dual_attestation: false` for research use

### Usage

#### With RFLRunner (AttestedRunContext integration)
```bash
# Load JSON config for RFLRunner
from rfl.config import RFLConfig
from rfl.runner import RFLRunner

config = RFLConfig.from_json("configs/rfl_experiment_wide_slice.json")
runner = RFLRunner(config)
# Use with AttestedRunContext via run_with_attestation()
```

#### With rfl_gate.py (Bootstrap analysis)
```bash
# YAML config for bootstrap gate analysis
python rfl_gate.py configs/rfl_experiment_wide_slice.yaml
```

### Metrics Logging

RFL runs are automatically logged to JSONL format when `experiment_id` contains "wide_slice":

- **Output:** `results/rfl_wide_slice_runs.jsonl`
- **Format:** One JSON object per line with:
  - `run_index`: Sequential run number
  - `slice_name`: Curriculum slice name
  - `abstention_rate_before`: Abstention rate before RFL update
  - `abstention_rate_after`: Abstention rate after RFL update
  - `symbolic_descent`: Computed symbolic descent (∇_sym)
  - `coverage_rate`, `novelty_rate`, `throughput`, `success_rate`: Additional metrics
  - `composite_root`: H_t for cross-referencing with FO cycle logs

### Integration with FO Cycles

The RFL engine integrates loosely with FO cycles (`experiments/run_fo_cycles.py`):

1. **Loose Coupling:** FO cycles write metrics; RFL later consumes them via `AttestedRunContext`
2. **Cross-Reference (Planned):** When Wide Slice runs are executed, both logs can be analyzed together:
   - FO cycle logs: `results/fo_*_wide.jsonl` (Dyno Chart format) — **not yet generated**
   - RFL metrics: `results/rfl_wide_slice_runs.jsonl` (JSONL format) — **not yet generated**
3. **Analysis (Planned):** Metrics can be cross-referenced by `composite_root` (H_t) or `run_index`

### Example Analysis Workflow (Planned for Phase II)

```bash
# 1. Run FO cycles with RFL mode (targeting slice_medium)
#    NOTE: This has not been executed yet. Example command:
uv run python experiments/run_fo_cycles.py \
  --mode=rfl \
  --cycles=1000 \
  --slice-name=slice_medium \
  --system=pl \
  --out=results/fo_rfl_wide.jsonl

# 2. RFL metrics would be automatically logged to:
#    results/rfl_wide_slice_runs.jsonl

# 3. Cross-reference for uplift analysis:
#    - Compare symbolic_descent trends
#    - Analyze abstention_rate deltas
#    - Correlate with FO cycle outcomes
```

**Current Phase I Evidence:**
- Existing FO cycles use `first-organism-pl` slice (see `results/fo_rfl_50.jsonl`, `results/fo_baseline.jsonl`)
- RFL integration verified on `first-organism-pl` slice (50-cycle sanity run)
- Wide Slice (`slice_medium`) experiments are infrastructure-ready but not yet executed

## 7. Next Steps for Agents
1.  **Setup:** Create `experiments/rfl/` structure.
2.  **Config:** Author the 3 JSON config files.
3.  **Execute:** Run R0 to validate the harness.
4.  **Analyze:** Generate the "Descent of Risk" plot from R1 data.
5.  **Wide Slice:** Use `configs/rfl_experiment_wide_slice.json` for slice_medium experiments.
