<!-- PHASE II — NOT RUN IN PHASE I -->
# U2 Governance Pipeline Specification

**Version:** `GOVERNANCE-1.0.0`

This document specifies the automated pipeline for verifying the integrity and compliance of Phase II U2 uplift experiment results. The pipeline is designed to be executed by a CI/CD system to produce a definitive, verifiable "Governance Receipt" for each run.

## 1. Scheduling Order and Execution Flow

The governance scripts must be executed in a strict, sequential order. The failure of any step immediately terminates the pipeline with a "fail" status for that gate.

1.  **G1: Preregistration Verification:**
    *   **Script:** `scripts/verify_prereg.py`
    *   **Purpose:** Confirms that the experiment was preregistered *before* the run and that the registration is complete and well-formed.

2.  **G2 & EV1: Manifest Integrity Verification:**
    *   **Script:** `scripts/verify_manifest_integrity.py`
    *   **Purpose:** Performs the core cryptographic verification. It ensures the manifest correctly binds the preregistration, the slice configuration, and the results via SHA256 hashes. This covers Gate G2 (Slice Config Hashing) and Evidence Validation step EV1 (Manifest Integrity).

3.  **G3 & G4 (Implied): Determinism & Metrics Verification:**
    *   **Script:** `scripts/verify_uplift_gates.py` (Master Script)
    *   **Purpose:** Acts as the final orchestrator. It re-verifies G1 and G2/EV1, and includes placeholder checks for determinism replay (G3) and success metric consistency (G4). It is the final checkpoint.

## 2. DAG (Directed Acyclic Graph) Diagram

The pipeline follows a simple linear graph. No parallel execution is permitted as each step depends on the success of the previous one.

```
[ START ]
    |
    v
+-----------------------------+
| G1: verify_prereg.py        |
+-----------------------------+
    | (on exit 0)
    v
+-----------------------------------+
| G2/EV1: verify_manifest_integrity.py |
+-----------------------------------+
    | (on exit 0)
    v
+-----------------------------+
| G3/G4: verify_uplift_gates.py |
+-----------------------------+
    | (on exit 0)
    v
[ SUCCESS: Generate Governance Receipt ]


FORBIDDEN PATH:
[ G1 ] -> [ G3/G4 ]  (Must pass G2/EV1 first)
```

## 3. Failure-Mode Table

Any non-zero exit code from a script signifies a governance failure. The CI runner must map these codes to the corresponding GOV ID.

| GOV ID | Script Exit Code | Failing Script | Example Failure | CI Mapping / Action |
| :--- | :--- | :--- | :--- | :--- |
| **GOV-1** | 1 | `verify_prereg.py` | `PREREG_UPLIFT_U2.yaml` is not found or contains invalid YAML. | `FAIL` |
| **GOV-2** | 2 | `verify_prereg.py` | `experiment_id` from manifest is not found in the prereg file. | `FAIL` |
| **GOV-3** | 3 | `verify_prereg.py` | Preregistration entry is missing a required field like `seed`. | `FAIL` |
| **GOV-10**| 10 | `verify_uplift_gates.py`| The master script's preregistration check failed (see GOV 1-3). | `FAIL` |
| **GOV-20**| 20 | `verify_uplift_gates.py`| Master script's integrity check failed. | `FAIL` |
| **GOV-21**| 1 | `verify_manifest_integrity.py`| Manifest is missing required fields. | `FAIL` |
| **GOV-22**| 2 | `verify_manifest_integrity.py`| `preregistration_hash` in manifest doesn't match computed hash. | `FAIL` |
| **GOV-23**| 3 | `verify_manifest_integrity.py`| `slice_config_hash` in manifest doesn't match file on disk. | `FAIL` |
| **GOV-24**| 4 | `verify_manifest_integrity.py`| `results_hash` in manifest doesn't match file on disk. | `FAIL` |
| **GOV-30**| 30 | `verify_uplift_gates.py`| Determinism replay check failed (Future). | `FAIL` |

## 4. Canonical "Governance Receipt" JSON Schema

Upon successful completion of the entire pipeline, the CI system must generate a `governance_receipt.json` file. This file is the final, immutable attestation of governance compliance for the run.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "U2 Governance Receipt",
  "description": "An immutable record certifying that a U2 experiment run has passed all Phase II governance checks.",
  "type": "object",
  "properties": {
    "receipt_version": {
      "type": "string",
      "pattern": "^GOVERNANCE-\d+\.\d+\.\d+$",
      "description": "The version of the governance framework that produced this receipt."
    },
    "experiment_id": {
      "type": "string",
      "description": "The unique ID of the verified experiment."
    },
    "manifest_path": {
      "type": "string",
      "description": "Relative path to the verified manifest file."
    },
    "verification_timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "The ISO 8601 timestamp of when the verification was completed."
    },
    "governance_record": {
      "type": "object",
      "description": "The detailed evidence record of the governance checks.",
      "$ref": "#/definitions/governanceEvidenceRecord"
    }
  },
  "required": [
    "receipt_version",
    "experiment_id",
    "manifest_path",
    "verification_timestamp",
    "governance_record"
  ],
  "definitions": {
    "governanceEvidenceRecord": {
      "type": "object",
      "properties": {
        "prereg_state": {
          "type": "object",
          "properties": {
            "status": {"type": "string", "enum": ["VERIFIED"]},
            "preregistration_hash": {"type": "string"}
          },
          "required": ["status", "preregistration_hash"]
        },
        "manifest_state": {
          "type": "object",
          "properties": {
            "status": {"type": "string", "enum": ["VERIFIED"]},
            "code_version_hash": {"type": "string"},
            "deterministic_seed": {"type": "integer"}
          },
          "required": ["status", "code_version_hash", "deterministic_seed"]
        },
        "hash_state": {
          "type": "object",
          "properties": {
            "status": {"type": "string", "enum": ["VERIFIED"]},
            "verified_hashes": {
              "type": "object",
              "properties": {
                "slice_config": {"type": "string"},
                "results": {"type": "string"}
              },
              "required": ["slice_config", "results"]
            }
          },
          "required": ["status", "verified_hashes"]
        },
        "integrity_state": {
          "type": "object",
          "properties": {
            "status": {"type": "string", "enum": ["VERIFIED"]},
            "verified_bindings": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["status", "verified_bindings"]
        },
        "final_decision": {
          "type": "object",
          "properties": {
            "decision": {"type": "string", "enum": ["admissible"]},
            "message": {"type": "string"}
          },
          "required": ["decision"]
        }
      },
      "required": [
        "prereg_state",
        "manifest_state",
        "hash_state",
        "integrity_state",
        "final_decision"
      ]
    }
  }
}
```

---

## Governance Evidence Record Format

This defines the structure of the `governance_record` object within the receipt. It's a structured log of the successful verification steps.

-   **`prereg_state{}`**: Captures the result of the G1 check.
    -   `status`: "VERIFIED"
    -   `preregistration_hash`: The computed SHA256 hash of the preregistration entry, confirming its content.

-   **`manifest_state{}`**: Captures key validated data points from the manifest.
    -   `status`: "VERIFIED"
    -   `code_version_hash`: The git commit hash from the manifest.
    -   `deterministic_seed`: The seed from the manifest.

-   **`hash_state{}`**: Records the verification of on-disk artifacts against the manifest.
    -   `status`: "VERIFIED"
    -   `verified_hashes`:
        -   `slice_config`: The manifest's `slice_config_hash`.
        -   `results`: The manifest's `results_hash`.

-   **`integrity_state{}`**: Confirms the successful validation of cryptographic bindings.
    -   `status`: "VERIFIED"
    -   `verified_bindings`: A list of bindings that were successfully checked (e.g., `["preregistration", "slice_config", "results"]`).

-   **`final_decision{}`**: The conclusive outcome of the pipeline.
    -   `decision`: For a valid receipt, this will always be **"admissible"**. An "inadmissible" decision results in a CI failure and no receipt is generated.
    -   `message`: A human-readable summary, e.g., "All Phase II governance gates passed."

---

## Governance Versioning Scheme

To ensure that governance receipts are always interpreted correctly, the governance framework itself will be versioned. This allows for future, backward-incompatible changes to the verification logic or schemas.

**Scheme:** Semantic Versioning (`MAJOR.MINOR.PATCH`) prefixed with `GOVERNANCE-`.

-   **Format:** `GOVERNANCE-MAJOR.MINOR.PATCH` (e.g., `GOVERNANCE-1.0.0`)

-   **`MAJOR`** version increments when a backward-incompatible change is made to the governance process.
    -   *Example*: Adding a mandatory new field to the U2 Manifest, requiring a new verification script. A `GOVERNANCE-1.x.x` pipeline would reject a run designed for `GOVERNANCE-2.x.x`.

-   **`MINOR`** version increments when new, backward-compatible functionality is added.
    -   *Example*: Adding an optional field to the Governance Receipt or improving the JSON output of a script without breaking the existing structure.

-   **`PATCH`** version increments for backward-compatible bug fixes.
    -   *Example*: Fixing a bug in the hash calculation logic of `verify_manifest_integrity.py` that doesn't alter the final schema.

```