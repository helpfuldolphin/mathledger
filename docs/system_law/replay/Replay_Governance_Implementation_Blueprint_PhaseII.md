# Replay Governance Implementation Blueprint — Phase II

This document provides the specific engineering execution plan for operationalizing the replay governance specifications. It translates the "what" from the artifact contract into the "how" for implementation. All steps will be implemented in a non-enforcing "SHADOW" mode first.

---

## 1. Orchestrator Specification (`replay_governance_orchestrator.py`)

The orchestrator script is the engine of the governance process. Its sole responsibility is to consume component-level data, apply governance rules, and produce the final `replay_governance_snapshot.json` artifact.

### Data Flow

**Inputs:**
1.  **Component Metrics (Directory):** Reads all `.json` files from a specified input directory (e.g., `./replay_outputs/`). Each file is expected to contain at least:
    ```json
    {
      "name": "component_name",
      "determinism_rate": 99.9,
      "drift_metric": 0.05
    }
    ```
2.  **Governance Rules (File):** Reads the `replay_criticality_rules.yaml` file to get thresholds.
    ```yaml
    thresholds:
      min_determinism_rate: 99.5
      max_drift_metric: 0.10
    ```
3.  **Environment Variables (CI Environment):**
    *   `GITHUB_RUN_ID`: To populate the `run_id` field.

**Core Transforms (In-Script Logic):**
1.  **Initialization:**
    *   Generate `timestamp_utc` (ISO 8601).
    *   Initialize the main snapshot object with `artifact_version: "1.0.0"`, `run_id`, and `timestamp`.
2.  **Aggregation:**
    *   Load all component JSON files from the input directory.
    *   Calculate the overall `determinism_rate` by averaging the rates from all components, weighted by their relative importance (if applicable, otherwise a simple average).
    *   Store all loaded component data in the `components` array of the snapshot.
3.  **Evaluation (`promotion_eval` Logic):**
    *   Initialize `promotion_eval.verdict` to `"promotion_ok"` and `promotion_eval.reasons` to an empty list.
    *   **Check 1 (Determinism):** Compare the aggregated `determinism_rate` against `rules.thresholds.min_determinism_rate`. If it is below the threshold, append a reason string and set the verdict to `"BLOCK"`.
    *   **Check 2 (Component Drift):** Iterate through each component. If a component's `drift_metric` exceeds `rules.thresholds.max_drift_metric`, append a reason, set the component's `is_blocking` flag to `true`, and set the overall verdict to `"BLOCK"`.
4.  **Finalization:**
    *   Based on the final `promotion_eval.verdict`, set the top-level `radar_status` to `"STABLE"` or `"UNSTABLE"`.
    *   If the verdict is `"promotion_ok"`, add a positive confirmation message to the `reasons` list.

**Outputs:**
1.  **Governance Snapshot (File):** Writes the complete, populated JSON object to `./replay_governance_snapshot.json`.

---

## 2. CI Binding Specification

This section details the exact steps to integrate the orchestrator into the existing CI/CD pipeline in a non-enforcing SHADOW mode.

**Target Workflow File:** `.github/workflows/replay-governance.yml`

**New Job: `Orchestration`**
This job will be added to the workflow, running after all component-level replay jobs have completed successfully.

```yaml
jobs:
  # ... existing component replay jobs
  
  orchestration:
    name: Replay Governance Orchestration
    needs: [component_job_1, component_job_2] # Depends on all replay jobs
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Download component artifacts
        uses: actions/download-artifact@v4
        with:
          path: ./replay_outputs/

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # VENV setup and dependency installation would go here

      - name: Run Replay Governance Orchestrator
        id: orchestrate
        run: |
          python replay_governance_orchestrator.py --input-dir ./replay_outputs/

      - name: Upload Governance Snapshot Artifact
        uses: actions/upload-artifact@v4
        with:
          name: replay-governance-snapshot
          path: ./replay_governance_snapshot.json

      - name: (SHADOW MODE) Check Governance Verdict
        id: shadow_check
        run: |
          verdict=$(jq -r '.promotion_eval.verdict' ./replay_governance_snapshot.json)
          echo "Replay Governance Verdict: $verdict"
          # In SHADOW mode, we always exit successfully.
          # The line below ensures the step passes regardless of the verdict.
          exit 0
```

---

## 3. Historical Ledger Pipeline Blueprint

This defines the automated, daily process for aggregating individual run snapshots into a longitudinal historical record.

**Mechanism:** A new, separate GitHub Actions workflow.
**File Location:** `.github/workflows/historical-aggregation.yml`

**Workflow Details:**
*   **Trigger:** Runs on a daily schedule. `schedule: - cron: '0 5 * * *'` (Runs at 05:00 UTC daily).
*   **Permissions:** Requires `actions: read` and `contents: write` to read artifacts and commit the aggregated ledger file.
*   **Aggregation Strategy:**
    1.  The job will use the GitHub API to find all workflow runs of `replay-governance.yml` that completed in the last 24 hours.
    2.  It will download the `replay-governance-snapshot` artifact from each of these runs.
    3.  A dedicated Python script (`scripts/aggregate_replays.py`) will process all downloaded snapshots.
    4.  For each snapshot, the script will extract the required fields (`timestamp_utc`, `run_id`, `commit_hash`, etc.) and transform them into the historical schema.
    5.  The script will append the resulting JSON line to the target ledger file.
*   **Naming Convention:** `governance/replay_history/YYYY-MM-DD.jsonl`. The script will generate the filename based on the UTC date of the run.
*   **Rotation & Retention:**
    *   **Rotation:** A new file is created each day. There is no rotation of existing files.
    *   **Retention:** For Phase II, all historical ledger files will be retained indefinitely. The retention policy will be reviewed in Phase III based on storage costs and query performance.

---

## 4. Developer Acceptance Criteria

The implementation of this blueprint is "done" when the following criteria are met:

1.  **Artifacts Exist:**
    *   The `replay_governance_orchestrator.py` script, when run locally, produces a `replay_governance_snapshot.json` file.
    *   The CI `Orchestration` job successfully produces and uploads a `replay-governance-snapshot` artifact.
    *   The daily aggregation job successfully commits a `governance/replay_history/YYYY-MM-DD.jsonl` file to the repository.

2.  **Scripts Run:**
    *   The `replay_governance_orchestrator.py` script runs without errors in the CI environment.
    *   The `(SHADOW MODE) Check Governance Verdict` step runs and correctly logs the verdict to the CI console.
    *   The historical aggregation script runs without errors and handles cases with zero or many artifacts from a given day.

3.  **Tests Pass:**
    *   A new suite of unit tests for `replay_governance_orchestrator.py` is created.
    *   Tests cover the logic for both `"promotion_ok"` and `"BLOCK"` verdicts based on mock component data and rule thresholds.
    *   Tests validate that the output JSON conforms to the schema defined in the artifact contract.
