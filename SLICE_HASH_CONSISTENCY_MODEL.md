# PHASE II — GEMINI E — NEW MANDATE
# Specification: The Slice Hash Consistency Model

This document specifies the formal model for ensuring end-to-end cryptographic integrity of curriculum slices, from definition to execution. This "Hash-Consistency Observatory" provides a deterministic, auditable, and secure foundation for the MathLedger project's experimental framework.

## 1. Canonical Slice Hash Serialization

To ensure that a given slice configuration has a deterministic and reproducible hash, a strict canonical serialization process is defined.

**Input:** A single slice configuration object (e.g., a dictionary loaded from the `slices` list in a YAML file).

**Process:**

1.  **Data Extraction:** The raw slice object is used as input.
2.  **Field Exclusion:** Non-canonical or volatile fields are excluded from the serialization. The only field currently defined as non-canonical is `name`, as it is metadata that does not affect the logical behavior of the slice. All other fields (`params`, `success_metric`, `formula_pool_entries`, etc.) are considered canonical.
3.  **Serialization to JSON:** The remaining data is serialized into a JSON string according to the following strict rules, conforming to RFC 8785 (JCS):
    *   **Encoding:** The output string MUST be UTF-8 encoded.
    *   **Key Sorting:** All keys in all objects/dictionaries MUST be sorted lexicographically (alphabetically).
    *   **Compactness:** The output string MUST be compact. There shall be no insignificant whitespace between tokens. This is achieved by specifying JSON separators as `("," , ":")`.
4.  **Hashing:** The final canonical hash is produced by applying the **SHA-256** algorithm to the canonical UTF-8 byte string generated in the previous step.

**Example (Conceptual):**

```python
import json
import hashlib

def get_canonical_slice_hash(slice_data: dict) -> str:
    # 1. & 2. Exclude non-canonical 'name' field
    canonical_data = slice_data.copy()
    canonical_data.pop("name", None)

    # 3. Serialize to a canonical JSON string
    canonical_str = json.dumps(
        canonical_data,
        sort_keys=True,
        separators=(",", ":")
    )

    # 4. Hash the resulting UTF-8 byte string
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
```

## 2. Full Slice Hash Lineage

The slice hash serves as a unique fingerprint that follows a configuration through its entire lifecycle. This lineage ensures that the experiment that runs is precisely the experiment that was designed and approved.

1.  **SLICE DEFINITION (`config/curriculum_uplift_phase2.yaml`)**
    *   **Artifact:** The human-readable YAML file containing the full definition of all curriculum slices.
    *   **Action:** A developer defines or modifies a slice's parameters, formulas, and success metrics.
    *   **State:** The slice exists in its raw, editable form.

2.  **PREREGISTRATION (`PREREG_UPLIFT_U2.yaml`)**
    *   **Artifact:** A preregistration file containing a list of planned experiments.
    *   **Action:** A researcher or engineer "preregisters" an experiment by declaring its intent. This involves generating the **Canonical Slice Hash** for a specific version of a slice from the definition file and recording it in the prereg file alongside a unique `experiment_id`.
    *   **State:** The experiment is now "locked" to a specific, immutable slice configuration via its hash. Any subsequent changes to the slice definition in the YAML file will cause a hash mismatch, invalidating the preregistration.

3.  **EXECUTION MANIFEST (Conceptual)**
    *   **Artifact:** A machine-generated manifest file that lists a batch of experiments approved for execution in a given run cycle.
    *   **Action:** An automated process selects one or more `experiment_id`s from the preregistration file and includes them in a manifest. To ensure integrity, the manifest MUST copy the `slice_config_hash` from the preregistration record.
    *   **State:** The manifest serves as a precise, non-ambiguous set of instructions for the U2 Uplift Runner.

4.  **AUDIT (e.g., `verify_slice_hashes.py`)**
    *   **Artifact:** An audit report or a "Hash Reconciliation Ledger".
    *   **Action:** An integrity auditing tool is run. It consumes all the artifacts above (Slice Definition, Preregistration, Manifest) and performs end-to-end verification.
    *   **State:** The audit confirms that the lineage is unbroken. It verifies that the hash of the current slice definition matches the hash in the preregistration file, which in turn matches the hash in the execution manifest. Any discrepancy at any stage is flagged as a `HASH-DRIFT` error.

## 3. Error Classes: Hash Drift Taxonomy

Any deviation from the canonical lineage results in a "Hash Drift" error. The following error classes provide a formal taxonomy for all integrity failures.

*   `HASH-DRIFT-1 (FORMULA_HASH_MISMATCH)`: The `hash` field of an entry in `formula_pool_entries` does not match the canonically computed hash of its `formula` field.
*   `HASH-DRIFT-2 (DANGLING_METRIC_REFERENCE)`: A hash in a `success_metric` field (`target_hashes`, `chain_target_hash`, etc.) does not correspond to any hash in the slice's `formula_pool_entries`.
*   `HASH-DRIFT-3 (PREREG_SLICE_HASH_MISMATCH)`: The `slice_config_hash` in a preregistration file does not match the freshly computed canonical hash of the corresponding slice in the main curriculum configuration file. This indicates that the slice definition has changed since the experiment was preregistered.
*   `HASH-DRIFT-4 (PREREG_SLICE_NOT_FOUND)`: A slice name referenced in a preregistration file does not exist in the main curriculum configuration file.
*   `HASH-DRIFT-5 (MANIFEST_PREREG_ID_NOT_FOUND)`: An execution manifest references an `experiment_id` that does not exist in any preregistration file.
*   `HASH-DRIFT-6 (MANIFEST_HASH_TAMPERING)`: The `slice_config_hash` recorded in an execution manifest does not match the hash in the corresponding preregistration entry for the same `experiment_id`.
*   `HASH-DRIFT-7 (NON_CANONICAL_SERIALIZATION)`: A hash was generated using a non-standard serialization process (e.g., unsorted keys, incorrect whitespace), resulting in a valid but non-canonical hash. This is a process or tooling error.
*   `HASH-DRIFT-8 (DOMAIN_SEPARATION_ERROR)`: A hash was computed without the correct domain separation prefix (e.g., using `DOMAIN_LEAF` instead of `DOMAIN_STMT` for a formula).
*   `HASH-DRIFT-9 (UNTRACKED_CONFIG_MODIFICATION)`: A specific, high-level instance of `HASH-DRIFT-3` where audit logs show the slice definition file was modified but the preregistration file was not updated in the same commit.
*   `HASH-DRIFT-10 (FORMULA_POOL_COLLISION)`: Two distinct formulas within the same `formula_pool_entries` normalize to the same canonical form, or the same formula appears twice, potentially with different roles.
*   `HASH-DRIFT-11 (SLICE_NAME_COLLISION)`: Two or more slices within the same curriculum configuration file share an identical `name`, making deterministic lookups impossible.
*   `HASH-DRIFT-12 (PREREG_ID_COLLISION)`: Two or more experiments in a preregistration file share the same `experiment_id`, creating ambiguity.

## 4. Logic for Dangling Metric Reference Detection

This section formally specifies the algorithm for detecting `HASH-DRIFT-2` errors.

**Objective:** To guarantee that every hash used as a target or requirement in a slice's `success_metric` refers to a formula that is explicitly defined and hashed within that same slice's `formula_pool_entries`.

**Algorithm:**

1.  **Initialize Pool Hash Set:** For the slice being audited, create an empty set data structure, `S_pool_hashes`.

2.  **Populate Hash Set:** Iterate through each `entry` object in the slice's `formula_pool_entries` list. For each `entry`, extract the value of the `hash` key and add it to the `S_pool_hashes` set.

3.  **Access Success Metric:** Retrieve the `success_metric` dictionary for the slice. If it does not exist, the check is complete for this slice.

4.  **Verify List-Based Hashes:**
    *   Identify all keys within `success_metric` that contain a list of hashes. The currently known keys are `target_hashes` and `required_goal_hashes`.
    *   For each such key, iterate through every hash `h` in its list value.
    *   For each `h`, check for its presence in the `S_pool_hashes` set.
    *   If `h` is **not in** `S_pool_hashes`, a `HASH-DRIFT-2` error is flagged. The error report must include the slice name, the key (`target_hashes`), and the specific hash `h` that is dangling.

5.  **Verify Single Hashes:**
    *   Identify all keys within `success_metric` that contain a single hash string. The currently known key is `chain_target_hash`.
    *   For each such key, retrieve its hash value `h`.
    *   Check for the presence of `h` in the `S_pool_hashes` set.
    *   If `h` is **not in** `S_pool_hashes`, a `HASH-DRIFT-2` error is flagged, including the slice name, the key (`chain_target_hash`), and the dangling hash `h`.

**Termination:** The check concludes after all relevant keys within the `success_metric` have been verified. The absence of any flagged errors certifies the binding integrity for that slice.

---

## 5. The Hash Reconciliation Ledger Format

The Hash Reconciliation Ledger is the canonical, machine-readable output of a full hash-consistency audit. It provides a deterministic and auditable proof of the system's integrity at a specific point in time.

**Format:** JSON

**Structure:**

```json
{
  "audit_metadata": {
    "audit_id": "sha256_of_inputs_and_timestamp",
    "timestamp_utc": "2025-12-06T10:00:00Z",
    "tool_version": "1.1.0",
    "inputs": [
      {
        "file_path": "config/curriculum_uplift_phase2.yaml",
        "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "file_path": "PREREG_UPLIFT_U2.yaml",
        "sha256_hash": "d4a7c8b094e9f9a2e6f8a3d2c1b0a9a1d8e6f0c7e9b3a1a0c8e6f0c7e9b3a1a0"
      }
    ]
  },
  "audit_results": [
    {
      "check_id": "FORMULA_HASH_slice-uplift-goal_p->q->p",
      "check_type": "FORMULA_INTEGRITY",
      "status": "PASSED",
      "details": {
        "slice_name": "slice_uplift_goal",
        "subject": "p->q->p",
        "expected": "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1",
        "actual": "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1"
      }
    },
    {
      "check_id": "BINDING_REFERENCE_slice-uplift-dependency_required_goal_hashes",
      "check_type": "BINDING_INTEGRITY",
      "status": "FAILED",
      "details": {
        "slice_name": "slice_uplift_dependency",
        "subject": "required_goal_hashes",
        "expected": "Hash to be present in formula pool",
        "actual": "Hash 'dangling_hash_for_demo' not found",
        "error_code": "HASH-DRIFT-2"
      }
    },
    {
      "check_id": "PREREG_CONSISTENCY_U2_EXP_001",
      "check_type": "PREREG_INTEGRITY",
      "status": "FAILED",
      "details": {
        "slice_name": "slice-uplift-goal",
        "subject": "U2_EXP_001",
        "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "actual": "d4a7c8b094e9f9a2e6f8a3d2c1b0a9a1d8e6f0c7e9b3a1a0c8e6f0c7e9b3a1a0",
        "error_code": "HASH-DRIFT-3"
      }
    }
  ],
  "summary": {
    "total_checks": 52,
    "passed_checks": 50,
    "failed_checks": 2,
    "overall_status": "INCONSISTENT"
  }
}
```

This ledger provides an immutable, comprehensive record that can be stored as a build artifact, used to gate deployments, and serve as a foundation for security and process audits.
