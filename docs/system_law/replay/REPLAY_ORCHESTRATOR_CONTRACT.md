# Replay Orchestrator Contract

This document defines the hardened contract for the `replay_governance_orchestrator.py` script. It establishes firm interfaces, schemas, and operational modes to ensure decoupled, reliable implementation across the subsequent pull requests. This contract is binding for PRs 1, 2, and 3.

---

## 1. Orchestrator Contract Definition

### 1.1. Inputs

The orchestrator has two primary inputs: Component Metric Artifacts and Environment State.

*   **Component Metric Artifacts:**
    *   **Path:** A directory path provided via a `--input-dir <path>` argument.
    *   **Content:** The directory is expected to contain one or more `.json` files, each representing a single component's replay metrics.
    *   **Schema:** Each file MUST conform to the `ComponentMetric-v1.0` schema (see Schema Interface Freeze section below).
*   **Environment State:**
    *   **Source:** Standard environment variables, expected to be populated by the CI environment.
    *   **Variables:**
        *   `GITHUB_RUN_ID`: Used for the `run_id` field in the output snapshot. If not present, the orchestrator will populate it with a placeholder (e.g., `local-run-<timestamp>`).

### 1.2. Outputs

*   **Governance Snapshot Artifact ("Governance Tile"):**
    *   **Path:** A file path provided via an `--output-file <path>` argument. Defaults to `./replay_governance_snapshot.json`.
    *   **Content:** A single JSON object conforming to the `GovernanceSnapshot-v1.0` schema.
    *   **Write Condition:** This file is written ONLY if the script is NOT in `--dry-run` mode.
*   **Standard Output (stdout):**
    *   The orchestrator will log diagnostic information about its progress, including the number of components processed and the final verdict. In `--dry-run` mode, it will log the full snapshot it *would have* written to the output file.
*   **Exit Code Semantics:**
    *   `exit 0`: Success. The orchestrator completed its evaluation. In both SHADOW and ENFORCEMENT modes, this code is returned if the script itself runs correctly, regardless of the governance verdict. The verdict is communicated via the artifact, not the exit code.
    *   `exit 1`: Script Failure. A fatal error occurred, such as a missing input directory, malformed component JSON, or a permissions error.

### 1.3. SHADOW Mode Invariants

SHADOW mode is the default and only operational mode for Phase II. It is defined by the CI binding, not the orchestrator script itself. The orchestrator is mode-agnostic, but the following invariants MUST be respected by the CI workflow consuming it:

1.  **No Enforcement:** The CI workflow MUST NOT fail the build based on the content of the `replay_governance_snapshot.json` artifact. The CI step that checks the verdict MUST always pass.
2.  **Full Artifact Generation:** The orchestrator runs its full logic and produces a complete, valid governance snapshot. SHADOW mode is for observing, not short-circuiting.
3.  **Verdict Logging:** The CI workflow MUST log the final `promotion_eval.verdict` to the build console to ensure visibility during the shadow period.

### 1.4. Failure Isolation Rules

The orchestrator must be resilient to partial failures to provide maximum insight.

1.  **Malformed Component Artifact:** If a JSON file in the input directory is malformed or does not match the `ComponentMetric-v1.0` schema, the orchestrator will:
    *   Log a warning to stderr identifying the failing file.
    *   Skip that file and continue processing all other valid component files.
    *   The final aggregated metrics will be based only on the successfully parsed components.
2.  **Empty Input Directory:** If the input directory contains no valid `.json` files, the orchestrator will:
    *   Log an error.
    *   Produce a valid `GovernanceSnapshot-v1.0` artifact with a `radar_status` of `"UNSTABLE"` and a `promotion_eval.verdict` of `"BLOCK"`.
    *   The `promotion_eval.reasons` array will clearly state that no component data was found.
    *   Exit with code `0`.

---

## 2. Schema Interface Freeze

All JSON interfaces are explicitly versioned. Additive changes are permissible with a minor version bump. Breaking changes require a new major version and a new file contract.

*   **Schema: `ComponentMetric-v1.0`** (Input for Orchestrator)
    ```json
    {
      "schema_version": "1.0",
      "name": "string",
      "determinism_rate": "float",
      "drift_metric": "float"
    }
    ```

*   **Schema: `GovernanceSnapshot-v1.0`** (Output of Orchestrator, Input for CI Binding & Historical Pipeline)
    ```json
    {
      "schema_version": "1.0",
      "artifact_version": "1.0.0", // Per original spec, will be harmonized to schema_version in v2
      "run_id": "string",
      "timestamp_utc": "string",
      "radar_status": "string", // "STABLE" or "UNSTABLE"
      "determinism_rate": "float",
      "promotion_eval": {
        "verdict": "string", // "promotion_ok" or "BLOCK"
        "reasons": ["string"]
      },
      "components": [
        {
          "name": "string",
          "determinism_rate": "float",
          "drift_metric": "float",
          "is_blocking": "boolean"
        }
      ]
    }
    ```

*   **Schema: `HistoricalLedgerRecord-v1.0`** (Output of Historical Pipeline)
    ```json
    {
      "schema_version": "1.0",
      "timestamp_utc": "string",
      "run_id": "string",
      "commit_hash": "string",
      "pr_number": "integer | null",
      "radar_status": "string",
      "determinism_rate": "float",
      "blocking_component": "string | null"
    }
    ```

---

## 3. Dry-Run Mode Specification

To facilitate safe integration and debugging, the orchestrator MUST support a `--dry-run` command-line flag.

*   **Invocation:** `python replay_governance_orchestrator.py --input-dir ./replay_outputs/ --dry-run`
*   **Behavior:**
    1.  The script performs all input loading and parsing steps as normal.
    2.  It runs the complete rule evaluation and aggregation logic.
    3.  It constructs the complete `GovernanceSnapshot-v1.0` JSON object in memory.
    4.  It prints the full, formatted JSON object to standard output, prefixed with a clear "DRY RUN MODE" banner.
    5.  It **MUST NOT** write the object to the path specified by `--output-file`.
    6.  It exits with code `0` if successful, or `1` if an error occurred during parsing/evaluation.

This ensures that developers and CI workflows can verify the orchestrator's behavior and logic without creating potentially confusing or invalid artifacts on the file system.
