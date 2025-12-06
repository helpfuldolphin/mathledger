# RFL Experiment Provenance & Evidence Manifest

**Architect:** GEMINI-I, Data Governance & Provenance Architect
**Date:** November 27, 2025
**Version:** 1.0

## 1. Overview

This document defines the **Experiment Manifest System** for Reflexive Formal Learning (RFL) experiments in MathLedger. The goal is to ensure **perfect reproducibility** and **traceability** for every data point, figure, and table appearing in the "First Organism" Research Paper (RP) and associated technical reports.

Every execution of the RFL loop (`rfl/runner.py`) generates an immutable **Experiment Manifest** (`manifest.json`) that serves as the "birth certificate" for that run's artifacts.

## 2. The Experiment Manifest Schema

The manifest is a JSON file generated at the conclusion of each experiment run. It binds the code, configuration, execution environment, and results into a single cryptographic context.

**Template Location:** `docs/evidence/experiment_manifest_schema.json`

### Key Fields

*   **`experiment_id`**: Unique identifier for the campaign (e.g., `rfl_production`).
*   **`run_index`**: The 1-based index of the run within the curriculum (e.g., `1` of `40`).
*   **`provenance`**: Git commit hash, user, and machine info. **Critical for auditing.**
*   **`configuration.snapshot`**: A full copy of the `RFLConfig` used, ensuring we don't rely on external config files that might change.
*   **`execution.effective_seed`**: The integer seed actually passed to the random number generator.
*   **`artifacts`**: Relative paths to all outputs (logs, CSVs, plots).

## 3. Traceability: From Paper to Code

An external auditor reading the "First Organism" paper should be able to trace any claim back to a specific manifest.

### 3.1. The Evidence Map

We recommend maintaining a top-level evidence map (e.g., `docs/evidence/RP_FIGURE_MAP.md`) linking paper assets to experiment IDs.

**Example Entry:**
> **Figure 3: Depth Distribution of Proofs**
> *   **Source Experiment:** `rfl_production`
> *   **Runs Aggregated:** 1-40
> *   **Manifests:** `artifacts/rfl/production/run_*/manifest.json`
> *   **Generation Script:** `backend/causal/generate_report.py --experiment rfl_production`

### 3.2. Artifact Naming Convention

Artifacts should be stored in a directory structure mirroring the experiment ID:

```
artifacts/
└── rfl/
    └── [experiment_id]/
        ├── run_01/
        │   ├── manifest.json
        │   ├── rfl.log
        │   └── metrics.json
        ├── run_02/
        │   └── ...
        └── summary_report.md
```

## 4. Re-run Instructions for Auditors

To reproduce a specific run (e.g., Run 5 of `rfl_production`), an auditor performs the following:

1.  **Locate the Manifest:** Find `artifacts/rfl/rfl_production/run_05/manifest.json`.
2.  **Checkout Code:**
    ```bash
    git checkout <provenance.git_commit>
    ```
3.  **Inspect Config:** Read `configuration.snapshot` from the manifest.
4.  **Execute Runner:**
    ```bash
    # Using the exact seed and parameters from the manifest
    python rfl/runner.py \
      --system-id 1 \
      --steps 100 \
      --seed <execution.effective_seed> \
      --output-dir artifacts/repro/run_05
    ```
5.  **Verify:** Compare the `results` block in the new output with the original manifest.

## 5. Integration Points

### 5.1. Global `spanning_set_manifest.json`
The global spanning set manifest (which tracks file integrity) should include the **Experiment Manifests** as "golden" artifacts. They should be marked with `category: evidence` to prevent accidental deletion or modification.

### 5.2. First Organism (FO) Attestation
The final "FO Attestation" is a cryptographic signature over the **collection of all 40 Experiment Manifests**.
*   The attestation process reads each `manifest.json`.
*   It verifies the `git_commit` matches the released version.
*   It hashes the `results` block.
*   It produces a final `fo_attestation.json` linking the paper's claims to these verified hashes.

## 6. Next Steps (Implementation)

1.  Update `rfl/runner.py` and `RFLExperiment.run` to accept a `manifest_path` argument.
2.  Implement `ManifestBuilder` class in `rfl/provenance.py`.
3.  Add `git_commit` retrieval using `subprocess.check_output(["git", "rev-parse", "HEAD"])`.
4.  Ensure `RFLConfig` serialization is robust (already exists as `.to_dict()`).
