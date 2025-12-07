# Governance Output Standardization: Architecture & Design

**Author**: MANUS-G (CI/Governance Systems Architect)
**Status**: DRAFT

## 1. Overview

This document specifies the architecture for standardizing all governance-related outputs in the MathLedger project. The primary goal is to ensure that all generated artifacts are **deterministic, versioned, and verifiable**. This system eliminates ambiguity, prevents silent schema drift, and guarantees bitwise reproducibility of governance records.

The core invariants are:
- **Determinism**: The same input data will always produce the exact same byte-for-byte output.
- **Stability**: Artifacts will not change unless their version is explicitly incremented.
- **Verifiability**: All artifacts can be validated against a formal schema and a set of canonicalization rules.

This system will be enforced by a new `governance-validator` tool integrated into the CI/CD pipeline.

## 2. Universal Design Principles

### 2.1. Versioning Policy

All governance artifacts and their corresponding schemas MUST adhere to **Semantic Versioning 2.0.0 (SemVer)** [1]. The version string (e.g., `1.2.3`) will be embedded in every artifact.

- **MAJOR** version (`1.x.x`): Incremented for breaking changes to the schema (e.g., removing a required field, changing a field's type).
- **MINOR** version (`x.1.x`): Incremented for backward-compatible additions (e.g., adding a new optional field).
- **PATCH** version (`x.x.1`): Incremented for backward-compatible bug fixes in the artifact generation logic that do not alter the schema.

CI gates will enforce that any schema change is accompanied by the correct version bump.

### 2.2. Canonicalization Rules

All JSON artifacts MUST be canonicalized according to **RFC 8785: JSON Canonicalization Scheme (JCS)** [2]. This ensures byte-for-byte determinism. The key rules are:

1.  **UTF-8 Encoding**: JSON text MUST be encoded in UTF-8.
2.  **No Whitespace**: No insignificant whitespace is allowed between tokens.
3.  **Sorted Keys**: Object keys MUST be sorted lexicographically by their Unicode code points.
4.  **Standard Number Format**: Numbers MUST conform to the I-JSON number format (no leading zeros, no trailing fractional zeros).
5.  **Escaped Characters**: Non-ASCII characters and certain control characters MUST be escaped (e.g., `\uXXXX`).

All tools generating governance artifacts will use a shared library that enforces JCS serialization.

### 2.3. Schema-First Approach

All governance artifacts are defined by a formal **JSON Schema (Draft 7)** [3]. These schemas serve as the single source of truth for the structure, types, and constraints of the data.

- **Location**: All schemas will reside in the `/schemas` directory.
- **Naming**: Schemas will be named `[artifact_name].schema.json` (e.g., `curriculum_snapshot.schema.json`).
- **Validation**: The `governance-validator` tool will use these schemas to validate all generated artifacts.

---

## 3. Governance Validator Tool

To enforce these standards, a new CLI tool, `governance-validator`, will be created. This tool will be the gatekeeper for all governance artifacts.

- **Location**: `scripts/validation/governance_validator.py`
- **CI Workflow**: `.github/workflows/governance-veracity-gate.yml`

### 3.1. Functionality

The validator will perform a series of checks on a given artifact:

1.  **Schema Validation**: Validate the artifact against its corresponding JSON schema from the `/schemas` directory.
2.  **Canonicalization Check**: Verify that the artifact is in RFC 8785 canonical form. This is done by re-serializing the parsed JSON and comparing the result byte-for-byte with the original file.
3.  **Version Consistency**: Check that the artifact's version matches the version declared in its schema file (via a custom `x-version` field).
4.  **Drift Detection**: For versioned snapshots, compare the artifact against its baseline version from the `main` branch to detect drift.

### 3.2. CLI Interface

```bash
# Validate a single artifact
python3 scripts/validation/governance_validator.py validate \
    --artifact-path artifacts/governance/curriculum_snapshot.json \
    --schema-path schemas/curriculum_snapshot.schema.json

# Validate all artifacts in a directory
python3 scripts/validation/governance_validator.py validate-all \
    --artifacts-dir artifacts/governance/

# Check for drift between two versions of an artifact
python3 scripts/validation/governance_validator.py diff \
    --baseline-path path/to/baseline_snapshot.json \
    --current-path path/to/current_snapshot.json
```

### 3.3. Exit Code Semantics

The validator will use the standardized exit codes:

| Exit Code | Meaning                     | CI Action     |
|-----------|-----------------------------|---------------|
| `0`       | **PASS**                    | Allow Merge   |
| `1`       | **FAIL (Validation Error)** | Block Merge   |
| `3`       | **ERROR (Infra)**           | Retry Job     |

---

## 4. CI Enforcement Logic

A new GitHub Actions workflow, `governance-veracity-gate.yml`, will run on every pull request to enforce these standards.

**Workflow Steps**:

1.  **Checkout Repo**: Check out the pull request branch.
2.  **Find Artifacts**: Find all modified or added governance artifacts in the `artifacts/` directory.
3.  **Run Validator**: For each artifact, invoke the `governance-validator` tool.
4.  **Report Failure**: If any artifact fails validation, the workflow fails, blocking the PR merge. A detailed report is posted as a PR comment.

### Example Workflow Snippet (`.github/workflows/governance-veracity-gate.yml`)

```yaml
- name: Validate Governance Artifacts
  id: validate
  run: |
    # Find changed artifacts
    baseline_ref="origin/${{ github.base_ref || 'main' }}"
    files=$(git diff --name-only --diff-filter=AM $baseline_ref -- artifacts/governance/**/*.json)

    if [[ -z "$files" ]]; then
      echo "No governance artifacts changed."
      exit 0
    fi

    for file in $files; do
      echo "Validating $file..."
      python3 scripts/validation/governance_validator.py validate --artifact-path $file
    done
```

## 5. References

[1] **SemVer**: Semantic Versioning 2.0.0, [https://semver.org/](https://semver.org/)
[2] **RFC 8785**: JSON Canonicalization Scheme (JCS), [https://tools.ietf.org/html/rfc8785](https://tools.ietf.org/html/rfc8785)
[3] **JSON Schema**: JSON Schema Draft 7, [https://json-schema.org/specification.html](https://json-schema.org/specification.html)
