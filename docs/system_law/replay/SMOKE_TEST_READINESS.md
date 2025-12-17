# Smoke-Test Readiness Checklist

This document provides the exact commands and expected outputs to verify that the replay orchestrator (PR-1) is ready for integration into the CI pipeline (PR-2).

---

### 1. Required Files

Ensure the following files exist and match the PR-1 implementation:
-   `scripts/replay_governance_orchestrator.py`
-   `tests/test_replay_orchestrator.py`
-   `replay_criticality_rules.yaml`

---

### 2. Unit Test Execution

This is the primary gate for accepting the implementation.

*   **Command:**
    ```bash
    # Install dependencies first, e.g., pip install pytest pyyaml
    pytest tests/test_replay_orchestrator.py
    ```
*   **Expected Output:**
    *   The `pytest` command must exit with code `0`.
    *   All 5 test cases (`test_stable_case`, `test_empty_directory_case`, `test_malformed_json_ignored`, `test_dry_run_no_file_written`, `test_deterministic_ordering`) must be marked as `PASSED`.

---

### 3. Manual Smoke Test (Happy Path)

This test verifies the successful generation of a `STABLE` artifact.

*   **Setup:**
    1.  `mkdir -p smoke_test/input`
    2.  Create `smoke_test/input/comp_a.json`:
        ```json
        {"schema_version": "1.0", "name": "core-ledger", "determinism_rate": 100.0, "drift_metric": 0.01}
        ```
    3.  Create `smoke_test/input/comp_b.json`:
        ```json
        {"schema_version": "1.0", "name": "basis-engine", "determinism_rate": 99.8, "drift_metric": 0.02}
        ```

*   **Command:**
    ```bash
    python scripts/replay_governance_orchestrator.py --input-dir smoke_test/input --output-file smoke_test/snapshot.json
    ```

*   **Verification:**
    1.  The command must exit with code `0`.
    2.  The file `smoke_test/snapshot.json` must be created.
    3.  Check key values in the output JSON:
        *   `jq .radar_status smoke_test/snapshot.json` -> must return `"STABLE"`
        *   `jq .promotion_eval.verdict smoke_test/snapshot.json` -> must return `"promotion_ok"`
        *   `jq .determinism_rate smoke_test/snapshot.json` -> must return `99.9`

---

### 4. Manual Smoke Test (BLOCK Path)

This test verifies the correct generation of an `UNSTABLE` artifact due to component drift.

*   **Setup:**
    1.  Use the same `smoke_test/input` directory from the happy path test.
    2.  Modify `smoke_test/input/comp_b.json` to have a high drift:
        ```json
        {"schema_version": "1.0", "name": "basis-engine", "determinism_rate": 99.8, "drift_metric": 0.55}
        ```

*   **Command:**
    ```bash
    python scripts/replay_governance_orchestrator.py --input-dir smoke_test/input --output-file smoke_test/snapshot_block.json
    ```

*   **Verification:**
    1.  The command must exit with code `0`.
    2.  The file `smoke_test/snapshot_block.json` must be created.
    3.  Check key values in the output JSON:
        *   `jq .radar_status smoke_test/snapshot_block.json` -> must return `"UNSTABLE"`
        *   `jq .promotion_eval.verdict smoke_test/snapshot_block.json` -> must return `"BLOCK"`
        *   `jq '.components[] | select(.name=="basis-engine").is_blocking' smoke_test/snapshot_block.json` -> must return `true`

---

### 5. Manual Smoke Test (Dry-Run)

This test verifies the `--dry-run` flag works as specified in the contract.

*   **Setup:**
    *   Use the `smoke_test/input` directory from the BLOCK path test.
    *   Ensure no `smoke_test/snapshot_dryrun.json` file exists.

*   **Command:**
    ```bash
    python scripts/replay_governance_orchestrator.py --input-dir smoke_test/input --output-file smoke_test/snapshot_dryrun.json --dry-run
    ```

*   **Verification:**
    1.  The command must exit with code `0`.
    2.  The command's standard output must contain the text "--- DRY RUN MODE ---" and the full JSON of the UNSTABLE snapshot.
    3.  The file `smoke_test/snapshot_dryrun.json` **MUST NOT** exist.

---

Once all these checks pass, PR-1 is considered complete and ready for review.
