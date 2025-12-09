# U2 EVIDENCE ADMISSIBILITY SPECIFICATION

- **Version**: 1.0
- **Status**: DRAFT
- **Author**: Gemini O, Promotion Governance Engineer

## 1. Overview

This document defines the "Admissibility Law Engine" for U2 Evidence Dossiers. A dossier MUST be admissible before it can be considered as non-blocking, supplementary evidence in the Basis Promotion process. Admissibility is a prerequisite for the review detailed in the DP1 Pre-Promotion Review Template.

Failure to meet these criteria renders a dossier **INADMISSIBLE**, meaning it cannot be used in any governance capacity.

## 2. Minimum Admissible Artifact Set (MAAS)

For a dossier to be considered for admissibility checks, it must contain a complete set of core artifacts. The absence of any single artifact in this set results in an immediate `DOSSIER-1: Incomplete MAAS` error.

| ID | Artifact | Admissibility Role |
|----|----------|--------------------|
| A1 | Preregistration (`...UPLIFT_U2.yaml`) | **CORE** | Defines the experiment's sanctioned parameters. |
| A2 | Preregistration Seal (`...UPLIFT_U2.yaml.sig`) | **CORE** | Guarantees preregistration preceded execution. |
| A3 | Environment Manifests (`manifest.json`) | **CORE** | Provides an unbroken chain of custody for all run outputs. |
| A5 | Statistical Summary (`statistical_summary.json`) | **CORE** | Contains the primary numerical results of the experiment. |
| A7 | Gate Compliance Log (`u2_compliance.jsonl`) | **CORE** | Records the pass/fail status of the G1-G4 compliance gates. |
| A4 | Raw Telemetry (`telemetry_..._.jsonl`) | **SUPPORTING** | Not required for admissibility, but required for deep audit. |
| A6 | Uplift Curve Plots (`uplift_curve_..._.png`) | **SUPPORTING** | Not required for admissibility, but required for human review. |

**Verdict**: Admissibility requires 100% of **CORE** artifacts to be present.

## 3. File-Level Invariants

Each CORE artifact MUST satisfy the following content-level invariants. Failure results in the specified DOSSIER error.

### A1: `PREREG_UPLIFT_U2.yaml` Invariants

| Invariant | Check | Error if Fail |
|-----------|-------|---------------|
| `slice_config_hash` present | Key `slice_config_hash` exists for each experiment. | `DOSSIER-4` |
| `seed` is integer | `seed` value is of type `int`. | `DOSSIER-5` |
| `success_metrics` is list | `success_metrics` is a non-empty list of strings. | `DOSSIER-6` |

### A2: `PREREG_UPLIFT_U2.yaml.sig` Invariants

| Invariant | Check | Error if Fail |
|-----------|-------|---------------|
| `sealed_at` is valid ISO8601 | `sealed_at` timestamp can be parsed as ISO8601. | `DOSSIER-7` |
| Seal predates runs | `sealed_at` timestamp is earlier than the `started_at` timestamp of all runs in all A3 manifests. | `DOSSIER-2` |
| Hash matches A1 | SHA-256 hash of A1 matches the hash in this file. | `DOSSIER-8` |

### A3: `manifest.json` Invariants

| Invariant | Check | Error if Fail |
|-----------|-------|---------------|
| `run_id` is consistent | All `run_id` fields match the dossier's top-level run ID. | `DOSSIER-9` |
| `artifacts` list is valid | All file paths listed under `artifacts` for each run exist on disk. | `DOSSIER-10` |
| `manifest_hash` is valid | The manifest's own hash is internally consistent. | `DOSSIER-11` |

### A5: `statistical_summary.json` Invariants

| Invariant | Check | Error if Fail |
|-----------|-------|---------------|
| `bootstrap_ci_95` exists | Each environment's summary contains this key. | `DOSSIER-12` |
| CI bounds are numbers | `lower` and `upper` values of the CI are floats or ints. | `DOSSIER-3` |
| `p_value` is float | The `p_value` field, if present, is a float between 0 and 1. | `DOSSIER-13` |

### A7: `u2_compliance.jsonl` Invariants

| Invariant | Check | Error if Fail |
|-----------|-------|---------------|
| Contains `gate_evaluation` | Each line is a JSON object with this top-level key. | `DOSSIER-14` |
| All gates present | Each `gate_evaluation` contains entries for G1, G2, G3, and G4. | `DOSSIER-15` |
| Gate status is `PASS` | The `status` for all four gates is exactly `PASS`. | `DOSSIER-16` |

## 4. Error Classes

Any dossier submitted for review is first processed by the Admissibility Law Engine. If any of the following errors are detected, the dossier is ruled INADMISSIBLE and rejected.

| Error ID | Title | Description |
|----------|-------|-------------|
| DOSSIER-1 | Incomplete MAAS | One or more CORE artifacts are missing from the dossier. |
| DOSSIER-2 | Preregistration Seal Timestamp Violation | The `sealed_at` timestamp in A2 is NOT strictly earlier than all run start times. |
| DOSSIER-3 | Invalid CI Format | A `bootstrap_ci_95` lower or upper bound in A5 is not a valid number. |
| DOSSIER-4 | Missing Slice Hash | An experiment in A1 is missing the `slice_config_hash` field. |
| DOSSIER-5 | Invalid Seed Type | An experiment's `seed` in A1 is not an integer. |
| DOSSIER-6 | Invalid Success Metrics | An experiment's `success_metrics` in A1 is empty or not a list. |
| DOSSIER-7 | Invalid Seal Timestamp Format | The `sealed_at` timestamp in A2 is not a valid ISO8601 string. |
| DOSSIER-8 | Preregistration Hash Mismatch | The hash of A1 does not match the hash stored in A2. |
| DOSSIER-9 | Inconsistent Run ID | A `run_id` within an A3 manifest does not match the dossier's run ID. |
| DOSSIER-10| Broken Artifact Link | A file path in an A3 manifest's `artifacts` list does not exist. |
| DOSSIER-11| Manifest Self-Hash Invalid | The `manifest_hash` in an A3 manifest does not match its own content's hash. |
| DOSSIER-12| Missing Confidence Interval | An environment summary in A5 is missing the `bootstrap_ci_95` key. |
| DOSSIER-13| Invalid P-Value Format | A `p_value` in A5 is not a valid float between 0.0 and 1.0. |
| DOSSIER-14| Malformed Compliance Log | A line in A7 is not a valid JSON object with a `gate_evaluation` key. |
| DOSSIER-15| Incomplete Gate Evaluation | A record in A7 is missing one or more of the G1, G2, G3, G4 gates. |
| DOSSIER-16| Gate Failure | At least one of the G1-G4 gates has a status other than `PASS` in A7. |
| DOSSIER-17| Manual Dossier Modification | A file's hash does not match the hash recorded in a sealed manifest. |
| DOSSIER-18| Telemetry Gap Detected | The `cycle_id` sequence in an A4 telemetry log is not continuous. |
| DOSSIER-19| Cross-Environment Contamination | Artifacts from one environment (e.g., ENV_A) are found in another's directory. |
| DOSSIER-20| Forbidden Operation Logged | An entry in a run log indicates a forbidden operation was performed. |

## 5. Explicit Forbidden Operations

The following operations are strictly forbidden. Performing them invalidates the entire U2 evidence chain and results in an INADMISSIBLE dossier, typically flagged by `DOSSIER-17` or `DOSSIER-20`.

1.  **MUST NOT** manually edit any file within a sealed dossier. All modifications must be made by regenerating the artifacts through the sanctioned pipeline.
2.  **MUST NOT** regenerate a single environment's data within a multi-environment run. All four environments must be run and processed as a single atomic unit.
3.  **MUST NOT** change the `slice_config` or `PREREG_UPLIFT_U2.yaml` file after the first U2 run has started, even if the seal (A2) has not yet been formally generated.
4.  **MUST NOT** use non-verifiable feedback sources (e.g., human preference scores, RLHF) in the RFL loop. This is a G4 integrity violation.
5.  **MUST NOT** build a dossier from a mix of different Run IDs. All artifacts must belong to a single, consistent execution.

## 6. Dossier Version Compatibility Matrix

This matrix defines the compatibility of different dossier versions with the promotion governance frameworks of different project phases.

| Dossier Version | Description | Phase I Promotion (Legacy) | Phase II Promotion (Basis) | Phase III Promotion (Metabolism) |
|---|---|---|---|---|
| **v1 (Phase I)** | Ad-hoc evidence packs, no formal structure. | **DEPRECATED** | **INCOMPATIBLE** | **INCOMPATIBLE** |
| **v2 (U2 Spec)** | Formal dossier per this specification. | **INADMISSIBLE** | **ADMISSIBLE (Non-blocking)** | **INADMISSIBLE** |
| **v3 (Future)** | *Hypothetical: Includes metabolism feedback.* | **INADMISSIBLE** | **ADMISSIBLE (as v2)** * | **ADMISSIBLE (Potentially Blocking)** |

### Compatibility Notes:

-   **v1 Dossiers**: Phase I evidence is considered unstructured and cannot be processed by the Admissibility Law Engine. It is archived for historical analysis only.
-   **v2 Dossiers (Current)**: Are the standard for Phase II. They are explicitly **non-blocking** and serve only as supplementary confidence signals. They are inadmissible for Phase I (lack of structure) and Phase III (lack of metabolism metrics).
-   * **Forward Compatibility**: A future v3 dossier can be "down-converted" and evaluated as a v2 dossier for Phase II promotion, provided it meets all v2 admissibility criteria. The v3-specific fields would be ignored during this evaluation.

## 7. Governance Verifier Integration

The **Governance Verifier** is a HARD PRECONDITION for dossier admissibility. No dossier
may be considered admissible unless it includes a valid Governance Verifier Report.

### 7.1 Governance Verifier as Hard Precondition

| Principle | Statement |
|-----------|-----------|
| **GOVV-P1** | Governance Verifier execution is MANDATORY before admissibility evaluation |
| **GOVV-P2** | Missing Governance Verifier Report renders dossier INADMISSIBLE |
| **GOVV-P3** | Governance Verifier Report MUST be included as artifact A8 in the dossier |
| **GOVV-P4** | Governance Verifier failures are BLOCKING — no bypass permitted |

### 7.2 Required Governance Verifier Report Fields

The following fields from `governance_verifier_report.json` are REQUIRED in the dossier:

| Field Path | Type | Required | Purpose |
|------------|------|----------|---------|
| `report_id` | string (UUID) | Yes | Unique identifier for this verification |
| `verified_at` | string (ISO8601) | Yes | Timestamp of verification execution |
| `verifier_version` | string | Yes | Version of governance verifier used |
| `dossier_hash` | string (hex64) | Yes | SHA-256 of dossier at verification time |
| `gates` | object | Yes | Gate verification results |
| `gates.G1` | object | Yes | Preregistration compliance result |
| `gates.G1.status` | string | Yes | "PASS" or "FAIL" |
| `gates.G1.checks` | array | Yes | Individual check results |
| `gates.G2` | object | Yes | Determinism compliance result |
| `gates.G2.status` | string | Yes | "PASS" or "FAIL" |
| `gates.G2.checks` | array | Yes | Individual check results |
| `gates.G3` | object | Yes | Manifest compliance result |
| `gates.G3.status` | string | Yes | "PASS" or "FAIL" |
| `gates.G3.checks` | array | Yes | Individual check results |
| `gates.G4` | object | Yes | RFL integrity compliance result |
| `gates.G4.status` | string | Yes | "PASS" or "FAIL" |
| `gates.G4.checks` | array | Yes | Individual check results |
| `overall_status` | string | Yes | "PASS" only if all gates PASS |
| `artifact_inventory` | object | Yes | List of artifacts verified |
| `artifact_inventory.verified_count` | int | Yes | Number of artifacts verified |
| `artifact_inventory.missing` | array | Yes | List of missing artifact IDs (must be empty) |
| `signature` | object | Yes | Verification signature |
| `signature.algorithm` | string | Yes | Signing algorithm (e.g., "SHA256") |
| `signature.hash` | string (hex64) | Yes | Hash of report content |

### 7.3 Governance Verifier Report Schema

```json
{
  "report_id": "<uuid>",
  "verified_at": "<ISO8601>",
  "verifier_version": "1.0.0",
  "dossier_hash": "sha256-<64 hex>",
  "prereg_hash": "sha256-<64 hex>",
  "gates": {
    "G1": {
      "name": "Preregistration Compliance",
      "status": "PASS",
      "checks": [
        { "id": "G1.1", "name": "PREREG exists", "status": "PASS" },
        { "id": "G1.2", "name": "Seal timestamp valid", "status": "PASS" },
        { "id": "G1.3", "name": "No post-seal amendments", "status": "PASS" },
        { "id": "G1.4", "name": "Hypothesis stated", "status": "PASS" }
      ]
    },
    "G2": {
      "name": "Determinism Compliance",
      "status": "PASS",
      "checks": [
        { "id": "G2.1", "name": "Fixed seeds specified", "status": "PASS" },
        { "id": "G2.2", "name": "Baseline random documented", "status": "PASS" },
        { "id": "G2.3", "name": "Runs reproduce", "status": "PASS" },
        { "id": "G2.4", "name": "H_t stable", "status": "PASS" }
      ]
    },
    "G3": {
      "name": "Manifest Compliance",
      "status": "PASS",
      "checks": [
        { "id": "G3.1", "name": "All artifacts listed", "status": "PASS" },
        { "id": "G3.2", "name": "Manifests sealed", "status": "PASS" },
        { "id": "G3.3", "name": "No sequence gaps", "status": "PASS" },
        { "id": "G3.4", "name": "Artifact integrity", "status": "PASS" }
      ]
    },
    "G4": {
      "name": "RFL Integrity Compliance",
      "status": "PASS",
      "checks": [
        { "id": "G4.1", "name": "Verifiable feedback only", "status": "PASS" },
        { "id": "G4.2", "name": "No proxy metrics", "status": "PASS" },
        { "id": "G4.3", "name": "Curriculum bounds respected", "status": "PASS" },
        { "id": "G4.4", "name": "Coverage computed correctly", "status": "PASS" }
      ]
    }
  },
  "overall_status": "PASS",
  "artifact_inventory": {
    "verified_count": 10,
    "missing": [],
    "artifacts": [
      { "id": "A1", "path": "experiments/prereg/PREREG_UPLIFT_U2.yaml", "hash": "sha256-..." },
      { "id": "A2", "path": "experiments/prereg/PREREG_UPLIFT_U2.yaml.sig", "hash": "sha256-..." }
    ]
  },
  "signature": {
    "algorithm": "SHA256",
    "hash": "sha256-<64 hex>",
    "signed_at": "<ISO8601>"
  }
}
```

### 7.4 Governance Verifier Error Classes

| Error ID | Title | Description |
|----------|-------|-------------|
| **GOVV-1** | Missing Governance Verifier Report | No `governance_verifier_report.json` found in dossier |
| **GOVV-2** | Invalid Report Schema | Report does not match required schema |
| **GOVV-3** | Missing Required Field | A required field is absent from the report |
| **GOVV-4** | Gate Failure Recorded | One or more gates show status other than "PASS" |
| **GOVV-5** | Dossier Hash Mismatch | Report's `dossier_hash` does not match current dossier |
| **GOVV-6** | Stale Report | Report `verified_at` timestamp predates dossier modifications |
| **GOVV-7** | Signature Invalid | Report signature hash does not match content |
| **GOVV-8** | Artifact Inventory Mismatch | Report's artifact list does not match dossier contents |
| **GOVV-9** | Missing Artifacts Listed | Report's `missing` array is non-empty |
| **GOVV-10** | Verifier Version Unsupported | Report's `verifier_version` is deprecated or unknown |

---

## 8. Updated Minimum Admissible Artifact Set (MAAS v2)

With Governance Verifier integration, the MAAS is updated to include A8:

| ID | Artifact | Category | Admissibility Role |
|----|----------|----------|-------------------|
| A1 | `PREREG_UPLIFT_U2.yaml` | **CORE** | Preregistration of hypotheses |
| A2 | `PREREG_UPLIFT_U2.yaml.sig` | **CORE** | Seal signature |
| A3 | Environment Manifests (×4) | **CORE** | Per-environment artifact registry |
| A4 | `ht_series.json` | **CORE** | H_t time series |
| A5 | `statistical_summary.json` | **CORE** | Uplift statistics |
| A6 | DAG/Environment/H_t Audit Reports | **CORE** | Audit verification |
| A7 | `u2_compliance.jsonl` | **CORE** | Gate pass records |
| **A8** | `governance_verifier_report.json` | **CORE** | **Governance Verifier output** |
| A9 | Uplift Curve Plots (×4) | SUPPORTING | Visual evidence |
| A10 | Raw Telemetry | SUPPORTING | Deep audit trail |

**MAAS v2 Verdict**: Admissibility requires 100% of CORE artifacts (A1–A8) to be present.

---

## 9. Dossier Law: Hard Exclusions

The following conditions result in HARD EXCLUSION — the dossier is NOT ADMISSIBLE
under any circumstances, with no waiver or appeal process.

### 9.1 Governance Verification Hard Exclusions

| Exclusion ID | Condition | Consequence |
|--------------|-----------|-------------|
| **HE-GV1** | Missing `governance_verifier_report.json` | Dossier NOT ADMISSIBLE |
| **HE-GV2** | Governance Verifier `overall_status` ≠ "PASS" | Dossier NOT ADMISSIBLE |
| **HE-GV3** | Any gate G1–G4 status ≠ "PASS" | Dossier NOT ADMISSIBLE |
| **HE-GV4** | Governance Verifier Report hash mismatch | Dossier NOT ADMISSIBLE |
| **HE-GV5** | Report `verified_at` > dossier seal time | Dossier NOT ADMISSIBLE (stale) |

### 9.2 Structural Hard Exclusions

| Exclusion ID | Condition | Consequence |
|--------------|-----------|-------------|
| **HE-S1** | Any CORE artifact (A1–A8) missing | Dossier NOT ADMISSIBLE |
| **HE-S2** | PREREG seal timestamp ≥ first run start | Dossier NOT ADMISSIBLE |
| **HE-S3** | Fewer than 4 environment manifests | Dossier NOT ADMISSIBLE |
| **HE-S4** | H_t series has cycle gaps | Dossier NOT ADMISSIBLE |
| **HE-S5** | Statistical summary missing CI bounds | Dossier NOT ADMISSIBLE |

### 9.3 Integrity Hard Exclusions

| Exclusion ID | Condition | Consequence |
|--------------|-----------|-------------|
| **HE-I1** | Any artifact hash mismatch vs manifest | Dossier NOT ADMISSIBLE |
| **HE-I2** | PREREG file modified after seal | Dossier NOT ADMISSIBLE |
| **HE-I3** | Cross-environment contamination detected | Dossier NOT ADMISSIBLE |
| **HE-I4** | Forbidden operation logged | Dossier NOT ADMISSIBLE |

### 9.4 Exclusion Precedence

Hard Exclusions are evaluated in order. The FIRST exclusion triggered determines
the rejection reason. Evaluation stops at first failure.

```
Order: HE-GV* → HE-S* → HE-I* → DOSSIER-*
```

---

## 10. Admissibility Algorithm

This algorithm determines whether a U2 Evidence Dossier is ADMISSIBLE or NOT_ADMISSIBLE.
The algorithm performs STRUCTURAL CHECKS ONLY — no uplift analysis, no performance evaluation.

### 10.1 Algorithm Specification

```
FUNCTION check_admissibility(dossier, governance_report, checks) -> VERDICT

INPUT:
  - dossier: U2 Evidence Dossier (directory structure with artifacts)
  - governance_report: Governance Verifier Report (JSON object or null)
  - checks: Admissibility check configuration

OUTPUT:
  - VERDICT: { status: "ADMISSIBLE" | "NOT_ADMISSIBLE", reason: string | null, code: string | null }

ALGORITHM:

  // ============================================================
  // PHASE 1: GOVERNANCE VERIFIER PRECONDITION (BLOCKING)
  // ============================================================

  IF governance_report IS NULL:
    RETURN { status: "NOT_ADMISSIBLE", reason: "Missing Governance Verifier Report", code: "HE-GV1" }

  IF NOT valid_schema(governance_report, GOVV_SCHEMA):
    RETURN { status: "NOT_ADMISSIBLE", reason: "Invalid report schema", code: "GOVV-2" }

  required_fields := ["report_id", "verified_at", "verifier_version", "dossier_hash",
                      "gates", "overall_status", "artifact_inventory", "signature"]

  FOR EACH field IN required_fields:
    IF field NOT IN governance_report:
      RETURN { status: "NOT_ADMISSIBLE", reason: "Missing field: " + field, code: "GOVV-3" }

  IF governance_report.overall_status ≠ "PASS":
    RETURN { status: "NOT_ADMISSIBLE", reason: "Governance verification failed", code: "HE-GV2" }

  FOR EACH gate IN ["G1", "G2", "G3", "G4"]:
    IF governance_report.gates[gate].status ≠ "PASS":
      RETURN { status: "NOT_ADMISSIBLE", reason: "Gate " + gate + " failed", code: "HE-GV3" }

  current_dossier_hash := compute_hash(dossier)
  IF governance_report.dossier_hash ≠ current_dossier_hash:
    RETURN { status: "NOT_ADMISSIBLE", reason: "Dossier hash mismatch", code: "HE-GV4" }

  IF governance_report.artifact_inventory.missing.length > 0:
    RETURN { status: "NOT_ADMISSIBLE", reason: "Missing artifacts in report", code: "GOVV-9" }

  // ============================================================
  // PHASE 2: CORE ARTIFACT PRESENCE CHECK
  // ============================================================

  core_artifacts := ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

  FOR EACH artifact_id IN core_artifacts:
    IF NOT exists_in_dossier(dossier, artifact_id):
      RETURN { status: "NOT_ADMISSIBLE", reason: "Missing core artifact: " + artifact_id, code: "HE-S1" }

  // A3 special check: must have exactly 4 environment manifests
  env_manifests := get_environment_manifests(dossier)
  IF count(env_manifests) < 4:
    RETURN { status: "NOT_ADMISSIBLE", reason: "Fewer than 4 environment manifests", code: "HE-S3" }

  // ============================================================
  // PHASE 3: STRUCTURAL INTEGRITY CHECKS
  // ============================================================

  // A2 vs A3: Seal must predate all runs
  seal_timestamp := parse_timestamp(dossier.A2.sealed_at)
  FOR EACH manifest IN env_manifests:
    FOR EACH run IN manifest.runs:
      IF seal_timestamp ≥ parse_timestamp(run.started_at):
        RETURN { status: "NOT_ADMISSIBLE", reason: "Seal timestamp violation", code: "HE-S2" }

  // A1 hash must match A2 seal
  a1_hash := compute_hash(dossier.A1)
  IF a1_hash ≠ dossier.A2.hash:
    RETURN { status: "NOT_ADMISSIBLE", reason: "PREREG hash mismatch", code: "DOSSIER-8" }

  // A4: H_t series cycle continuity
  FOR EACH environment IN dossier.A4.environments:
    FOR EACH run IN environment.runs:
      IF NOT is_continuous(run.ht_values, "cycle"):
        RETURN { status: "NOT_ADMISSIBLE", reason: "H_t series gap in " + environment.id, code: "HE-S4" }

  // A5: Statistical summary must have CI bounds
  FOR EACH env_id, summary IN dossier.A5.environments:
    IF "bootstrap_ci_95" NOT IN summary:
      RETURN { status: "NOT_ADMISSIBLE", reason: "Missing CI bounds for " + env_id, code: "HE-S5" }
    IF NOT is_number(summary.bootstrap_ci_95.lower) OR NOT is_number(summary.bootstrap_ci_95.upper):
      RETURN { status: "NOT_ADMISSIBLE", reason: "Invalid CI format for " + env_id, code: "DOSSIER-3" }

  // ============================================================
  // PHASE 4: ARTIFACT INTEGRITY CHECKS
  // ============================================================

  FOR EACH manifest IN env_manifests:
    FOR EACH artifact_ref IN manifest.artifacts:
      IF NOT file_exists(artifact_ref.path):
        RETURN { status: "NOT_ADMISSIBLE", reason: "Broken artifact link: " + artifact_ref.path, code: "DOSSIER-10" }
      IF compute_hash(artifact_ref.path) ≠ artifact_ref.hash:
        RETURN { status: "NOT_ADMISSIBLE", reason: "Artifact hash mismatch: " + artifact_ref.path, code: "HE-I1" }

  // Check for cross-environment contamination
  FOR EACH env_a IN ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]:
    FOR EACH env_b IN ["ENV_A", "ENV_B", "ENV_C", "ENV_D"]:
      IF env_a ≠ env_b:
        IF has_contamination(dossier, env_a, env_b):
          RETURN { status: "NOT_ADMISSIBLE", reason: "Cross-env contamination: " + env_a + " in " + env_b, code: "HE-I3" }

  // ============================================================
  // PHASE 5: RELATIONSHIP INTEGRITY CHECKS
  // ============================================================

  // All experiment_ids in A3 must exist in A1
  prereg_exp_ids := extract_experiment_ids(dossier.A1)
  FOR EACH manifest IN env_manifests:
    IF manifest.experiment_id NOT IN prereg_exp_ids:
      RETURN { status: "NOT_ADMISSIBLE", reason: "Unknown experiment_id in manifest", code: "DOSSIER-9" }

  // All environment_ids in A4 must exist in A3
  manifest_env_ids := extract_environment_ids(env_manifests)
  FOR EACH env_id IN dossier.A4.environments.keys():
    IF env_id NOT IN manifest_env_ids:
      RETURN { status: "NOT_ADMISSIBLE", reason: "Unknown environment in H_t series", code: "REL-3" }

  // ============================================================
  // ALL CHECKS PASSED
  // ============================================================

  RETURN { status: "ADMISSIBLE", reason: null, code: null }

END FUNCTION
```

### 10.2 Algorithm Properties

| Property | Guarantee |
|----------|-----------|
| **Deterministic** | Same inputs always produce same output |
| **Order-Independent** | Check order does not affect final verdict (first failure reported) |
| **No Side Effects** | Algorithm only reads; never modifies dossier |
| **No Performance Logic** | No uplift ratios, CIs, or metrics evaluated |
| **No Promotion Logic** | Does not determine promotion eligibility |
| **Binary Output** | Exactly one of ADMISSIBLE or NOT_ADMISSIBLE |

### 10.3 Algorithm Execution Requirements

| Requirement | Description |
|-------------|-------------|
| **REQ-ALG1** | Must execute atomically — no partial results |
| **REQ-ALG2** | Must log execution to `ops/logs/admissibility_decisions.jsonl` |
| **REQ-ALG3** | Must return within 30 seconds for typical dossier |
| **REQ-ALG4** | Must be implemented identically in all enforcement tools |
| **REQ-ALG5** | Must not access network resources during execution |

### 10.4 Algorithm Output Schema

```json
{
  "decision_id": "<uuid>",
  "executed_at": "<ISO8601>",
  "executor": "<agent_id>",
  "dossier_path": "<path>",
  "dossier_hash": "sha256-<64 hex>",
  "governance_report_hash": "sha256-<64 hex>",
  "phases_executed": {
    "phase_1_governance": { "executed": true, "passed": true },
    "phase_2_artifacts": { "executed": true, "passed": true },
    "phase_3_structure": { "executed": true, "passed": true },
    "phase_4_integrity": { "executed": true, "passed": true },
    "phase_5_relationships": { "executed": true, "passed": true }
  },
  "verdict": {
    "status": "ADMISSIBLE",
    "reason": null,
    "code": null
  },
  "execution_time_ms": 1234
}
```

---

## 11. Normative References

| Document | Purpose |
|----------|---------|
| `WAVE1_PROMOTION_BLUEPRINT.md` | Promotion governance framework |
| `PREREG_UPLIFT_U2.yaml` | Preregistration template |
| `validate-governance.ps1` | Governance gate validation script |
| `validate-admissibility.ps1` | Admissibility check script |

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-06 | Gemini O | Initial draft |
| 1.1 | 2025-12-06 | Claude O | Added Governance Verifier Integration (§7), Updated MAAS v2 (§8), Dossier Law Hard Exclusions (§9), Admissibility Algorithm (§10) |

