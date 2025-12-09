# Full CI Pipeline Implementation Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **DRAFT**

## 1. Overview

This document provides the complete implementation plan for the five new CI gates that form the core of the MathLedger governance system. It includes the exact YAML for each workflow, a detailed explanation of how snapshots are managed, and how drift results are propagated to pull requests.

This plan builds upon the designs specified in the **CI Governance Stabilization Plan** and the **Snapshot Generator Suite Plan**. The successful implementation of these workflows will fully activate the automated drift detection and governance validation system.

## 2. Core CI Concepts

### 2.1. Snapshot Management

All drift detection workflows follow a consistent pattern for managing snapshots:

1.  **Generate Current Snapshot**: The workflow first runs the appropriate snapshot generator script to create a snapshot of the current state of the pull request branch. This is saved to a temporary file (e.g., `current_snapshot.json`).

2.  **Fetch Baseline Snapshot**: The workflow then uses `git show` to check out the corresponding snapshot file from the pull request's base branch (typically `main`). This is saved to `baseline_snapshot.json`.

3.  **Handle Missing Baseline**: If the baseline file does not exist (e.g., this is the first time the gate has run), the drift radar script is designed to handle this gracefully and exit with a `SKIP` status (exit code 4).

### 2.2. Result Propagation

-   **Exit Codes**: The drift radar and validator scripts use standardized exit codes to signal the outcome of the check.
-   **Workflow Failure**: The CI workflow step that runs the script is configured to fail if the script returns a non-zero exit code (e.g., `set -e`). This causes the entire job and workflow to fail, blocking the pull request.
-   **Artifacts**: All reports (JSON and Markdown) are uploaded as CI artifacts. This allows developers to download and inspect the detailed drift reports.
-   **PR Comments** (Future Work): A future enhancement could be to automatically post the Markdown summary as a comment on the pull request.

---

## 3. Workflow Implementations

This section provides the complete, production-ready YAML for each of the five new CI gates.

### 3.1. `gate-governance-veracity.yml`

**Purpose**: Validates all changed governance artifacts against their JSON schemas and canonicalization rules.

```yaml
# .github/workflows/gate-governance-veracity.yml
name: 'Gate: Governance Veracity'

on:
  pull_request:
    paths:
      - 'artifacts/governance/**.json'
      - 'schemas/**.json'

jobs:
  validate-artifacts:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip3 install jsonschema

      - name: Find and Validate Changed Artifacts
        id: validate
        run: |
          set -e
          baseline_ref="origin/${{ github.base_ref || 'main' }}"
          files=$(git diff --name-only --diff-filter=AM $baseline_ref -- 'artifacts/governance/*.json')

          if [[ -z "$files" ]]; then
            echo "No governance artifacts changed."
            exit 0
          fi

          for file in $files; do
            schema_name=$(basename $file .json)
            echo "Validating $file against schema $schema_name..."
            python3 scripts/validation/governance_validator.py validate \
              --artifact-path $file \
              --schema-name $schema_name
          done
```

### 3.2. `gate-curriculum-drift.yml`

**Purpose**: Detects drift in the MathLedger curriculum.

```yaml
# .github/workflows/gate-curriculum-drift.yml
name: 'Gate: Curriculum Drift'

on:
  pull_request:
    paths:
      - 'curriculum/**'

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip3 install pyyaml

      - name: Generate Current Snapshot
        run: |
          mkdir -p artifacts/drift
          python3 scripts/generators/generate_curriculum_snapshot.py > artifacts/drift/current_snapshot.json

      - name: Get Baseline Snapshot
        run: |
          git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/curriculum_snapshot.json > artifacts/drift/baseline_snapshot.json || \
          echo '{}' > artifacts/drift/baseline_snapshot.json

      - name: Run Curriculum Drift Radar
        run: |
          set -e
          python3 scripts/radars/curriculum_drift_radar.py \
            --baseline artifacts/drift/baseline_snapshot.json \
            --current artifacts/drift/current_snapshot.json \
            --output-dir artifacts/drift

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: curriculum-drift-report
          path: artifacts/drift/
          retention-days: 7
```

### 3.3. `gate-telemetry-drift.yml`

**Purpose**: Detects drift in telemetry event schemas.

```yaml
# .github/workflows/gate-telemetry-drift.yml
name: 'Gate: Telemetry Drift'

on:
  pull_request:
    paths:
      - 'backend/telemetry/events.py'

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Generate Current Snapshot
        run: |
          mkdir -p artifacts/drift
          python3 scripts/generators/generate_telemetry_snapshot.py > artifacts/drift/current_snapshot.json

      - name: Get Baseline Snapshot
        run: |
          git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/telemetry_schema_snapshot.json > artifacts/drift/baseline_snapshot.json || \
          echo '{}' > artifacts/drift/baseline_snapshot.json

      - name: Run Telemetry Drift Radar
        run: |
          set -e
          python3 scripts/radars/telemetry_drift_radar.py \
            --baseline artifacts/drift/baseline_snapshot.json \
            --current artifacts/drift/current_snapshot.json \
            --output-dir artifacts/drift

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: telemetry-drift-report
          path: artifacts/drift/
          retention-days: 7
```

### 3.4. `gate-ledger-drift.yml`

**Purpose**: Detects drift in the ledger state.

```yaml
# .github/workflows/gate-ledger-drift.yml
name: 'Gate: Ledger Drift'

on:
  pull_request:
    paths:
      - 'artifacts/ledger/mathledger.db'

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Generate Current Snapshot
        run: |
          mkdir -p artifacts/drift
          python3 scripts/generators/generate_ledger_snapshot.py > artifacts/drift/current_snapshot.json

      - name: Get Baseline Snapshot
        run: |
          git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/ledger_snapshot.json > artifacts/drift/baseline_snapshot.json || \
          echo '{}' > artifacts/drift/baseline_snapshot.json

      - name: Run Ledger Drift Radar
        run: |
          set -e
          python3 scripts/radars/ledger_drift_radar.py \
            --baseline artifacts/drift/baseline_snapshot.json \
            --current artifacts/drift/current_snapshot.json \
            --output-dir artifacts/drift

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: ledger-drift-report
          path: artifacts/drift/
          retention-days: 7
```

### 3.5. `gate-ht-triangle-drift.yml`

**Purpose**: Verifies the H_t = SHA256(R_t || U_t) cryptographic invariant.

```yaml
# .github/workflows/gate-ht-triangle-drift.yml
name: 'Gate: HT Triangle Drift'

on:
  pull_request:
    paths:
      - 'artifacts/governance/attestation_history.jsonl'

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Generate Current Snapshot
        run: |
          mkdir -p artifacts/drift
          python3 scripts/generators/generate_attestation_snapshot.py > artifacts/drift/current_snapshot.json

      - name: Get Baseline Snapshot
        run: |
          git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/attestation_snapshot.json > artifacts/drift/baseline_snapshot.json || \
          echo '{}' > artifacts/drift/baseline_snapshot.json

      - name: Run HT Triangle Drift Radar
        run: |
          set -e
          python3 scripts/radars/ht_triangle_drift_radar.py \
            --baseline artifacts/drift/baseline_snapshot.json \
            --current artifacts/drift/current_snapshot.json \
            --output-dir artifacts/drift

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: ht-triangle-drift-report
          path: artifacts/drift/
          retention-days: 7
```
