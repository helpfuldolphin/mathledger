# Chronicle Developer Runbook

## Running and Debugging Curriculum Drift Chronicle Locally

This runbook provides instructions for developers to manually execute the Curriculum Drift Chronicle archiver, inspect its output, and respond to its findings.

### Manual Execution

To run the archiver on a small, local curriculum data sample, use the main script from the project root. This allows you to simulate the CI process and debug issues before committing changes.

**Exact Command:**

```bash
# Assumes you are at the root of the 'mathledger' repository.
# The input can be a directory or a specific JSON file for a curriculum subset.
python scripts/curriculum_drift_chronicle.py \
  --input ./curriculum/p3_readiness_sample.json \
  --output /tmp/chronicle_run.json
```

**Inspect the Output:**

After the script completes, inspect the JSON output to see the drift analysis.

```bash
# Use 'cat' or a tool like 'jq' for formatted output
cat /tmp/chronicle_run.json
```

### Interpreting the Results

The `status` field in `chronicle_run.json` is the primary indicator of curriculum health and P3/P4 readiness. Its value determines the required developer action.

*   **`OK`**: The curriculum drift is within acceptable tolerance. No immediate action is required. This is the expected state for a healthy build.
*   **`WARN`**: The drift is approaching the configured threshold. While this will not block the CI gate, it serves as an early warning. The on-call developer should investigate the source of the drift to prevent it from becoming a `BLOCK` condition.
*   **`BLOCK`**: The drift has exceeded the critical threshold. **This is a CI gate failure.** The build or deployment is considered unsafe to proceed. The developer who triggered the run is responsible for immediate remediation.

### Example Scenarios

#### Example 1: A "Good" Run

A developer runs the archiver to validate a minor change.

1.  **Command:**
    ```bash
    python scripts/curriculum_drift_chronicle.py --input ./curriculum/p3_readiness_sample.json --output /tmp/chronicle_run.json
    ```

2.  **Output (`/tmp/chronicle_run.json`):**
    ```json
    {
      "version": "1.2.3",
      "timestamp": "2025-12-10T14:00:00Z",
      "status": "OK",
      "drift_metric": 0.04,
      "thresholds": {
        "warn": 0.08,
        "block": 0.1
      },
      "reason": "Drift metric 0.04 is within OK threshold (< 0.08)."
    }
    ```

3.  **Developer Response:** The status is `OK`. The change is safe. The developer can proceed with their commit/merge.

#### Example 2: A "BLOCK" Run

A developer merges a branch that inadvertently corrupts a data source.

1.  **Command:**
    ```bash
    python scripts/curriculum_drift_chronicle.py --input ./curriculum/p3_readiness_sample.json --output /tmp/chronicle_run.json
    ```

2.  **Output (`/tmp/chronicle_run.json`):**
    ```json
    {
      "version": "1.2.4",
      "timestamp": "2025-12-10T15:30:00Z",
      "status": "BLOCK",
      "drift_metric": 0.12,
      "thresholds": {
        "warn": 0.08,
        "block": 0.1
      },
      "reason": "Drift metric 0.12 exceeds BLOCK threshold (>= 0.1).",
      "details": "High deviation detected in module 'calculus_fundamentals'."
    }
    ```

3.  **Developer Response:** This is a **gate failure**. The release is blocked. The developer must:
    *   Immediately notify the team on the relevant channel.
    *   Analyze the `details` and `reason` fields to pinpoint the cause.
    *   Examine the recent commits/merges to identify the source of the data corruption in the 'calculus_fundamentals' module.
    *   Revert the offending change or develop and apply a hotfix.
    *   Re-run the chronicle to ensure the status returns to `OK` before re-attempting the merge.
