# Schema Registry Build-Out Plan

**Author**: MANUS-G (CI/Governance Systems Architect)
**Version**: 1.0.0
**Status**: **DRAFT**

## 1. Overview

This document specifies the design and build-out of a formal **Schema Registry** for the MathLedger project. The current schema implementation is a flat directory (`/schemas`) with unversioned filenames. This is not scalable and does not support schema evolution.

The new Schema Registry will introduce a versioned directory structure, a formal compatibility policy, and a CI gate to enforce schema integrity. This will provide a robust foundation for managing the lifecycle of all governance artifacts.

## 2. Schema Registry Architecture

### 2.1. Directory Structure

The existing `/schemas` directory will be restructured to support versioning. The new structure will be:

```
artifacts/schemas/
├── curriculum/
│   ├── v1.0.0.schema.json
│   └── v1.1.0.schema.json
├── telemetry/
│   └── v1.0.0.schema.json
├── ledger/
│   └── v1.0.0.schema.json
└── attestation/
    └── v1.0.0.schema.json
```

- **Top-level directory**: `artifacts/schemas/` will be the root of the registry.
- **Artifact-specific directories**: Each governance artifact (curriculum, telemetry, etc.) will have its own directory.
- **Versioned filenames**: Schema files will be named according to their **exact Semantic Version** (e.g., `v1.0.0.schema.json`).

### 2.2. Version Bump Mechanics

Changing a schema requires a formal, deliberate process:

1.  **Copy and Modify**: To introduce a change, a developer MUST copy the latest schema version to a new file with an incremented version number (e.g., copy `v1.0.0.schema.json` to `v1.1.0.schema.json`).
2.  **Apply Changes**: The developer then applies the desired changes to the new schema file.
3.  **Update Generators**: The corresponding snapshot generator script MUST be updated to produce data compliant with the new schema and to embed the new version number in the snapshot.
4.  **Pull Request**: All changes (new schema, updated generator) are submitted in a single pull request.

This process ensures that all schema changes are explicit, versioned, and auditable through Git history.

## 3. Schema Compatibility Policy

To ensure that downstream consumers of governance artifacts do not break unexpectedly, we will adopt a **Backward Compatibility** policy, inspired by Confluent's Schema Registry [1].

| Change Type | Description | Compatibility | Version Bump |
|---|---|---|---|
| **Add Optional Field** | Adding a new field that is not in the `required` array. | **Backward Compatible** | **MINOR** (e.g., 1.0.0 → 1.1.0) |
| **Remove Optional Field** | Removing a field that was not in the `required` array. | **Backward Compatible** | **MINOR** (e.g., 1.1.0 → 1.2.0) |
| **Add Required Field** | Adding a new field to the `required` array. | **NOT Backward Compatible** | **MAJOR** (e.g., 1.2.0 → 2.0.0) |
| **Remove Required Field** | Removing a field that was in the `required` array. | **NOT Backward Compatible** | **MAJOR** (e.g., 1.2.0 → 2.0.0) |
| **Change Field Type** | Changing the `type` of a field (e.g., `string` to `integer`). | **NOT Backward Compatible** | **MAJOR** (e.g., 1.2.0 → 2.0.0) |
| **Annotation Changes** | Changing `description`, `title`, etc. | **Backward Compatible** | **PATCH** (e.g., 1.0.0 → 1.0.1) |

**Policy Enforcement**: This policy will be enforced through peer review. The PR description for any schema change MUST justify the version bump according to this compatibility matrix.

## 4. CI Schema Verification Plan

To automate the enforcement of schema integrity, the `gate-governance-veracity.yml` workflow will be enhanced.

### 4.1. Schema Self-Validation

Every schema in the registry MUST be a valid JSON Schema (Draft 7). The CI gate will be updated to validate all changed schemas against the official JSON Schema meta-schema.

**Workflow Update (`gate-governance-veracity.yml`)**:

```yaml
- name: Validate Schema Integrity
  run: |
    set -e
    # Find changed schemas
    baseline_ref="origin/${{ github.base_ref || 'main' }}"
    files=$(git diff --name-only --diff-filter=AM $baseline_ref -- 'artifacts/schemas/**/*.json')

    if [[ -z "$files" ]]; then
      echo "No schemas changed."
      exit 0
    fi

    # Download meta-schema
    wget https://json-schema.org/draft-07/schema -O /tmp/draft07.schema.json

    for file in $files; do
      echo "Validating schema $file..."
      jsonschema -i $file --schema /tmp/draft07.schema.json
    done
```

*(This requires `pip install jsonschema[cli]`)*

### 4.2. Version Consistency Check

The `governance-validator` tool will be updated to locate the correct schema from the registry based on the `version` field in the artifact it is validating.

**`governance_validator.py` Update**:

-   The `--schema-name` argument will be removed.
-   The validator will now read the `version` from the artifact (e.g., `1.1.0`).
-   It will then construct the expected schema path (e.g., `artifacts/schemas/curriculum/v1.1.0.schema.json`).
-   If this schema file does not exist, the validation fails.

This change ensures that every governance artifact is validated against the **exact schema version** it was generated for, creating a closed-loop verification system.

## 5. References

[1] **Confluent Schema Registry**: Confluent Schema Registry Documentation, [https://docs.confluent.io/platform/current/schema-registry/index.html](https://docs.confluent.io/platform/current/schema-registry/index.html)
