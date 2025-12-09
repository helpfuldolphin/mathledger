# CI Conformance Gating Policy

**Version**: 1.0  
**Status**: ACTIVE  

## 1. Overview

To ensure the integrity and stability of metrics used in governance and analysis, all metric changes proposed via pull requests must pass an automated conformance gate. This document describes the standard CI pattern for enforcing the metric promotion policies defined in `config/metric_promotion_policy.json`.

The primary tool for this enforcement is the `conformance_cli.py` script.

## 2. Gating Mechanism: `gate-family`

The CI pipeline MUST invoke the `gate-family` subcommand for each metric family affected by the code changes in the pull request.

### 2.1. Command Synopsis

```bash
python conformance_cli.py gate-family <family_name> <path/to/baseline/snapshots> <path/to/candidate/snapshots>
```

- **`family_name`**: The prefix for the metric family being evaluated (e.g., `uplift_u2`, `default_metric`).
- **`path/to/baseline/snapshots`**: The path to the directory containing the approved, production-level metric snapshots from the main branch.
- **`path/to/candidate/snapshots`**: The path to the directory containing the new metric snapshots generated from the pull request's code.

### 2.2. Gate Logic

The `gate-family` command performs the following steps:

1.  **Loads Promotion Policy**: It reads `config/metric_promotion_policy.json` to determine the promotion rules for the specified family. If no specific family policy is found, it uses the `default` policy.
2.  **Discovers Candidates**: It finds all candidate snapshots in the `<candidate_dir>` that match the `<family_name>`.
3.  **Compares Snapshots**: For each candidate, it locates the corresponding baseline snapshot in the `<baseline_dir>`.
4.  **Applies Policy**: It invokes the `can_promote_metric` function, which evaluates the candidate against the baseline according to the rules of the loaded policy (e.g., `required_level`, `allow_l3_regression`).
5.  **Exits with Status Code**:
    - If **all** metrics in the family pass their promotion gate, the script prints a success message and exits with code `0`.
    - If **any** metric fails its promotion gate, the script prints the reasons for failure and exits with code `1`.

## 3. Example CI Job Configuration (YAML)

This example shows how to implement the gate in a generic CI system (e.g., GitHub Actions).

```yaml
jobs:
  metric-conformance-gate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      # Assume previous steps generate baseline and candidate artifacts
      - name: 'Download baseline snapshots'
        uses: actions/download-artifact@v3
        with:
          name: baseline-snapshots
          path: artifacts/snapshots/baseline

      - name: 'Generate and store candidate snapshots'
        run: |
          # This command would run the process that creates the snapshots
          python -m your_snapshot_generator --output-dir artifacts/snapshots/candidate
      
      - name: 'Gate: U2 Uplift Metrics'
        run: |
          python conformance_cli.py gate-family uplift_u2 artifacts/snapshots/baseline artifacts/snapshots/candidate

      - name: 'Gate: Default Metrics'
        run: |
          python conformance_cli.py gate-family default_metric artifacts/snapshots/baseline artifacts/snapshots/candidate
          
      # Add a gate for every other relevant metric family...
```

This multi-step approach ensures that a regression in one metric family does not prevent developers from getting feedback on other, unrelated families. A failure in any `gate-family` step will cause the entire CI job to fail, blocking the pull request.