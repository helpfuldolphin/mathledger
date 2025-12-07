# MathLedger CI Governance Stabilization Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **IMPLEMENTATION READY**

## 1. Executive Summary

This document outlines a comprehensive, actionable plan to stabilize and harden the MathLedger project's Continuous Integration (CI) and Governance infrastructure. The successful execution of this plan will result in a fully deterministic, verifiable, and maintainable system that automatically detects and prevents undesirable drift across all critical project domains.

This plan introduces four major initiatives:

1.  **The Drift Radar Suite**: A new set of four automated CI gates to monitor curriculum, telemetry, ledger, and cryptographic invariants.
2.  **Governance Output Standardization**: A schema-first, versioned, and canonicalized approach to all governance artifacts, enforced by a new validation tool.
3.  **Modular Evidence Pack Toolchain**: A rebuilt, command-driven tool for creating, sealing, and auditing evidence packs, replacing legacy scripts.
4.  **CI Normalization Blueprint**: A set of enforced standards for workflow naming, action versioning, artifact retention, and dependency management.

Upon implementation, the MathLedger project will possess a production-grade CI/Governance system that guarantees the integrity and reproducibility of its core assets. This document provides all necessary designs, file paths, and workflow diffs for immediate implementation.

---

## 2. Initiative 1: Drift Radar Suite Deployment

**Objective**: To deploy a suite of automated radars that continuously monitor critical project domains for drift.

**Design Document**: [`docs/governance/drift_radar_architecture.md`](./drift_radar_architecture.md)

### 2.1. New Components

-   **Radar Scripts**: Four new Python scripts have been created in `scripts/radars/`:
    -   `curriculum_drift_radar.py`
    -   `telemetry_drift_radar.py`
    -   `ledger_drift_radar.py`
    -   `ht_triangle_drift_radar.py`
-   **CI Workflows**: Four new GitHub Actions workflows will be created in `.github/workflows/` to integrate these radars.

### 2.2. New CI Workflows

#### `gate-curriculum-drift.yml`

This workflow runs the Curriculum Drift Radar on every pull request affecting curriculum files.

```yaml
# .github/workflows/gate-curriculum-drift.yml
name: 'Gate: Curriculum Drift'

on:
  pull_request:
    paths:
      - 'curriculum/**'
      - 'artifacts/governance/curriculum_snapshot.json'

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
        with:
          fetch-depth: 0 # Need full history for git diff

      - name: Set up Python
        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v5.1.0
        with:
          python-version: '3.11'

      - name: Generate Current Snapshot
        # This step should be replaced with the actual command to generate the snapshot
        run: echo '{"version": "1.0.1", ...}' > artifacts/governance/curriculum_snapshot_current.json

      - name: Get Baseline Snapshot
        run: |
          git show origin/${{ github.base_ref || 'main' }}:artifacts/governance/curriculum_snapshot.json > artifacts/governance/curriculum_snapshot_baseline.json || \
          echo 'Baseline not found, skipping.' > artifacts/governance/curriculum_snapshot_baseline.json

      - name: Run Curriculum Drift Radar
        run: |
          python3 scripts/radars/curriculum_drift_radar.py \
            --baseline artifacts/governance/curriculum_snapshot_baseline.json \
            --current artifacts/governance/curriculum_snapshot_current.json \
            --output artifacts/drift

      - name: Upload Drift Report
        if: always()
        uses: actions/upload-artifact@5d5d22a31266ced268874388b861e4b58bb5c2f3 # v4.3.1
        with:
          name: curriculum-drift-report
          path: artifacts/drift/
          retention-days: 7
```

*(Similar workflows will be created for `gate-telemetry-drift.yml`, `gate-ledger-drift.yml`, and `gate-ht-triangle-drift.yml`)*

---

## 3. Initiative 2: Governance Output Standardization

**Objective**: To enforce a schema-first, versioned, and canonicalized standard for all governance artifacts.

**Design Document**: [`docs/governance/output_standardization.md`](./output_standardization.md)

### 3.1. New Components

-   **JSON Schemas**: A new `/schemas` directory now contains the formal JSON Schema definitions for all governance snapshots:
    -   `schemas/curriculum_snapshot.schema.json`
    -   `schemas/telemetry_schema_snapshot.schema.json`
    -   `schemas/ledger_snapshot.schema.json`
    -   `schemas/attestation_snapshot.schema.json`
-   **Validator Tool**: A new CLI tool, `scripts/validation/governance_validator.py`, has been created to enforce these standards.

### 3.2. New CI Workflow: `gate-governance-veracity.yml`

This workflow will replace the existing `veracity-gate.yml`. It runs on every pull request and uses the `governance_validator.py` tool to check all modified governance artifacts for schema compliance, canonicalization, and version consistency.

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

---

## 4. Initiative 3: Modular Evidence Pack Toolchain

**Objective**: To replace legacy, monolithic evidence pack scripts with a single, modular, and maintainable toolchain.

### 4.1. New Component

-   **Evidence Pack Tool**: The script `scripts/evidence_pack.py` has been created. It consolidates all functionality related to evidence packs under a single command-driven interface.

### 4.2. Toolchain Commands

-   `create`: Discovers artifacts and creates a new evidence pack manifest.
-   `seal`: Applies a cryptographic seal (SHA-256 hash) to a manifest.
-   `audit`: Verifies the integrity of a sealed pack, checking both the manifest hash and all artifact hashes.
-   `diff`: Compares two evidence packs and reports differences.

### 4.3. CI Workflow Update: `gate-evidence.yml`

The existing `evidence-gate.yml` will be updated to use the new toolchain. Instead of custom linting logic, it will now run `evidence_pack.py audit` on any changed evidence packs.

**Diff for `.github/workflows/evidence-gate.yml`**:

```diff
-        run: |
-          python3 tools/ci/docs_lint.py --base origin/${{ github.base_ref || 'integrate/ledger-v0.1' }}
+        run: |
+          # Find changed evidence packs
+          baseline_ref="origin/${{ github.base_ref || 'main' }}"
+          files=$(git diff --name-only --diff-filter=AM $baseline_ref -- 'artifacts/evidence_packs/**/*.json')
+
+          if [[ -z "$files" ]]; then
+            echo "No evidence packs changed."
+            exit 0
+          fi
+
+          for file in $files; do
+            echo "Auditing $file..."
+            python3 scripts/evidence_pack.py audit --pack $file
+          done
```

---

## 5. Initiative 4: CI Normalization Blueprint

**Objective**: To enforce a strict set of standards across all GitHub Actions workflows for consistency, security, and maintainability.

**Design Document**: [`docs/governance/ci_normalization_blueprint.md`](./ci_normalization_blueprint.md)

### 5.1. Summary of Changes

1.  **Workflow Naming**: All workflow files in `.github/workflows/` will be renamed to follow the `[category]-[subject].yml` convention.
    -   **Action**: A one-time rename of all 15 existing workflows.

2.  **Action Versioning**: All third-party actions will be pinned to a specific commit SHA. A central version file, `/.github/actions-versions.yml`, will be created and all workflows will be updated to reference it.
    -   **Action**: Create `actions-versions.yml` and update all `uses:` clauses in all workflows.

3.  **Artifact Retention**: All `actions/upload-artifact` steps will be updated to include a `retention-days` parameter (defaulting to 7 days for PRs, 365 for releases).
    -   **Action**: Audit all workflows and add the `retention-days` parameter where missing.

4.  **Dependency Normalization**: Job dependencies (`needs`) will be reviewed and optimized for parallel execution and fail-fast behavior.
    -   **Action**: Review and refactor the job graphs in complex workflows like `build-main.yml` and `gate-determinism.yml`.

### 5.2. Example Refactoring: `gate-determinism.yml`

**Old Structure**:
- `determinism-core`
- `determinism-extended`
- `determinism-audit`
- `determinism-regression`

*(All jobs run in parallel, which is inefficient as `determinism-audit` is much faster and should run first)*

**New Structure**:

```yaml
jobs:
  audit:
    # Runs first, fails fast if banned patterns are found
    ...
  core:
    needs: audit
    ...
  regression:
    needs: audit
    ...
  extended:
    needs: audit
    if: github.event_name == 'workflow_dispatch'
    ...
```

---

## 6. Implementation and Rollout

This plan will be implemented in a single, comprehensive pull request. The changes are designed to be non-disruptive to ongoing development, as they primarily involve adding new validation gates and standardizing existing infrastructure. The rollout will be considered complete when all checks pass on the implementation PR and it is merged into the `main` branch.
