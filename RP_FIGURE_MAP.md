# RP Figure Provenance Map

This document links Research Paper (RP) figures to their source experiment manifests and raw log data, ensuring full reproducibility of the results presented.

## Figure 1: Coverage Growth & Stability
*Comparison of derivation coverage between Baseline (unsupervised) and RFL (feedback-driven) modes over 1000 cycles.*

*   **Source Experiment (Baseline)**: `fo_1000_baseline`
    *   **Manifest**: [`artifacts/experiments/rfl/fo_1000_baseline/manifest.json`](../../artifacts/experiments/rfl/fo_1000_baseline/manifest.json)
    *   **Raw Logs**: `artifacts/experiments/rfl/fo_1000_baseline/experiment_log.jsonl`
    *   **Config Snapshot**: `config/rfl/fo_1000_baseline.json`
*   **Source Experiment (RFL)**: `fo_1000_rfl`
    *   **Manifest**: [`artifacts/experiments/rfl/fo_1000_rfl/manifest.json`](../../artifacts/experiments/rfl/fo_1000_rfl/manifest.json)
    *   **Raw Logs**: `artifacts/experiments/rfl/fo_1000_rfl/experiment_log.jsonl`
    *   **Config Snapshot**: `config/rfl/fo_1000_rfl.json`

## Figure 2: Reflexive Metabolism (Abstention vs. Descent)
*Scatter plot showing the correlation between abstention mass ($\alpha_{mass}$) and symbolic descent ($\nabla_{sym}$), demonstrating the "Law of Reflexive Descent".*

*   **Source Experiment**: `fo_1000_rfl`
    *   **Manifest**: [`artifacts/experiments/rfl/fo_1000_rfl/manifest.json`](../../artifacts/experiments/rfl/fo_1000_rfl/manifest.json)
    *   **Data Points**: Derived from `rfl_law` section in `experiment_log.jsonl`.
    *   **Audit Log**: `artifacts/rfl/rfl_audit.json` (Linked via Manifest)

## Figure 3: Uplift Ratio Distribution (Bootstrap CI)
*Probability density function of the Uplift Ratio ($U$), showing the shift in derivation efficiency.*

*   **Source Analysis**: `RFLRunner` Phase 3 (Uplift Computation)
    *   **Primary Source**: `fo_1000_rfl` Manifest
    *   **Metric**: `uplift.bootstrap_ci` from `manifest.results`.
    *   **Dual Attestation**: Verified against `dual_attestation.checks.uplift`.

## Reproducibility Command
To regenerate the source data for these figures:

```bash
# 1. Run Baseline (1000 cycles)
export RFL_EXPERIMENT_ID="fo_1000_baseline"
export RFL_NUM_RUNS=1000
python rfl/runner.py --mode baseline

# 2. Run RFL (1000 cycles)
export RFL_EXPERIMENT_ID="fo_1000_rfl"
export RFL_NUM_RUNS=1000
python rfl/runner.py --mode rfl
```
