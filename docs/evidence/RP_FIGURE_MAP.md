# Research Paper Figure Map (RP_FIGURE_MAP)

This document links the figures and tables in the "First Organism" Research Paper (RP) to their source experiment manifests and data artifacts, ensuring full traceability and reproducibility.

## Figure 1: Throughput & Coverage (Baseline vs. RFL)

*   **Description:** Comparison of proof throughput and statement coverage between the 1000-cycle Baseline run and the RFL run. **Note:** The RFL run (`results/fo_rfl.jsonl`) has 1000 cycles but shows 100% abstention (degenerate case). This demonstrates RFL execution infrastructure works but does not provide evidence of abstention reduction or uplift.
*   **Source Experiments:**
    *   `fo_1000_baseline`
    *   `fo_1000_rfl`
*   **Manifests:**
    *   `artifacts/experiments/rfl/fo_1000_baseline/manifest.json`
    *   `artifacts/experiments/rfl/fo_1000_rfl/manifest.json`
*   **Data Sources:**
    *   `artifacts/experiments/rfl/fo_1000_baseline/metrics.json`
    *   `artifacts/experiments/rfl/fo_1000_rfl/metrics.json`
*   **Generation Script:** `backend/causal/generate_report.py --compare fo_1000_baseline fo_1000_rfl`

## Figure 2: Symbolic Descent Trajectory

*   **Description:** Visualization of the symbolic descent metric over the course of the RFL experiment.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest:** `artifacts/experiments/rfl/fo_1000_rfl/manifest.json`
*   **Data Source:** `artifacts/experiments/rfl/fo_1000_rfl/policy_ledger.json`

## Table 1: Metabolism Verification

*   **Description:** Final verification of Coverage and Uplift thresholds.
*   **Source Experiment:** `rfl_production` (Aggregated 40 runs)
*   **Manifests:** `docs/evidence/manifests/RFL_RUN_*.json`
*   **Data Source:** `artifacts/rfl/rfl_results.json`

## Figure 3: Depth Distribution

*   **Description:** Histogram of proof depth for the generated corpus.
*   **Source Experiment:** `fo_1000_rfl`
*   **Manifest:** `artifacts/experiments/rfl/fo_1000_rfl/manifest.json`
*   **Artifact:** `artifacts/experiments/rfl/fo_1000_rfl/depth_dist.png` (if generated)
