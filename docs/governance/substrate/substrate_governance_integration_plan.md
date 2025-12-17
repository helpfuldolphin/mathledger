# Substrate Governance Integration Plan

This document provides the technical blueprint for integrating substrate-level governance into the CI/CD pipeline and the `global_health.json` monitoring system.

---

### 1. Telemetry & Attestation Sources

The `identity_stability_index` is a composite score derived from multiple layers of telemetry. The initial implementation will rely on the following sources, which provide cryptographic measurements of the runtime environment.

-   **Source 1: Trusted Platform Module (TPM) / Platform Configuration Registers (PCRs)**
    -   **Telemetry:** PCR quotes containing SHA-256 hashes of the boot sequence components (UEFI, bootloader, kernel).
    -   **Contribution to Index:** This is the foundational measurement. A mismatch in critical PCRs (e.g., those covering the kernel or secure boot state) will immediately drive the stability index below the `BLOCK` threshold.

-   **Source 2: File System Hashing (Integrity Management Architecture - IMA)**
    -   **Telemetry:** A cryptographically signed manifest of SHA-256 hashes for critical system files (`/etc`, `/usr/lib`, `/bin`, `/sbin`). This manifest is generated at build-time and stored in a secure location. The runtime IMA subsystem provides a measurement list for comparison.
    -   **Contribution to Index:** Deviations in this manifest contribute to index degradation. A small number of unexpected changes (e.g., a patched library) might result in a `YELLOW` state. Widespread or critical mismatches (e.g., `sshd`, `sudo`) will trigger a `RED` or `BLOCK` state.

-   **Source 3: Kernel Module Signature Verification**
    -   **Telemetry:** A runtime audit of all loaded kernel modules, verifying that each is signed by a trusted key from the project's key infrastructure.
    -   **Contribution to Index:** The presence of any unsigned or untrusted kernel module is a critical finding and will immediately set a `BLOCK` flag.

---

### 2. Exact JSON Schema (Draft-07) for Substrate Tile

This schema formalizes the structure of the `substrate` object for use in `global_health.json`.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Substrate Governance Tile",
  "description": "Schema for the substrate identity and stability monitoring tile.",
  "type": "object",
  "properties": {
    "status_light": {
      "description": "Single-word status indicator for the substrate.",
      "type": "string",
      "enum": ["GREEN", "YELLOW", "RED", "BLOCK"]
    },
    "headline": {
      "description": "Concise, human-readable summary of the substrate state.",
      "type": "string"
    },
    "identity_stability_index": {
      "description": "A score from 0.0 to 1.0 indicating confidence in substrate integrity.",
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "drift_flags": {
      "description": "A list of specific, machine-readable flags indicating the nature of any detected drift.",
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "PCR_MISMATCH",
          "FILESYSTEM_HASH_MISMATCH",
          "UNSIGNED_KERNEL_MODULE_LOADED",
          "INSECURE_BOOT_DETECTED",
          "ATTESTATION_STALE"
        ]
      }
    },
    "last_attestation_timestamp": {
      "description": "The ISO 8601 timestamp of the last successful attestation measurement.",
      "type": "string",
      "format": "date-time"
    },
    "attestation_provider": {
      "description": "The service or technology that performed the attestation.",
      "type": "string"
    },
    "details_url": {
      "description": "A link to a detailed report or dashboard for further analysis.",
      "type": "string",
      "format": "uri-reference"
    }
  },
  "required": [
    "status_light",
    "headline",
    "identity_stability_index",
    "drift_flags",
    "last_attestation_timestamp"
  ]
}
```

---

### 3. CI Integration Requirements

The `scripts/substrate_governance_check.py` script will be the primary enforcement point.

-   **Gating Logic:**
    -   The script MUST fetch the latest `global_health.json` from a trusted source.
    -   It will parse the `substrate.status_light` field.
    -   If `status_light` is **`GREEN`** or **`YELLOW`**, the script will proceed silently and exit with code `0`.
    -   If `status_light` is **`RED`** or **`BLOCK`**, the script MUST fail the CI job.

-   **Exit Codes:**
    -   **`0`**: Success (Gate Pass).
    -   **`1`**: General Script Error (e.g., cannot fetch or parse `global_health.json`).
    -   **`64`**: Governance Failure (`RED` status).
    -   **`65`**: Critical Governance Failure (`BLOCK` status).

-   **Failure Messages:**
    -   Upon a `RED` failure, the script MUST print the following to `stderr`:
        ```
        SUBSTRATE GOVERNANCE CHECK: FAILED (RED)
        REASON: Significant substrate drift detected.
        HEADLINE: [contents of substrate.headline]
        DRIFT FLAGS: [contents of substrate.drift_flags]
        Deployment is blocked pending security review. See details at: [substrate.details_url]
        ```
    -   Upon a `BLOCK` failure, the message MUST be more severe:
        ```
        SUBSTRATE GOVERNANCE CHECK: FAILED (BLOCK) - CRITICAL SECURITY ALERT
        REASON: Critical substrate integrity failure.
        HEADLINE: [contents of substrate.headline]
        DRIFT FLAGS: [contents of substrate.drift_flags]
        CI GATE IS HARD-LOCKED. IMMEDIATE MANUAL INTERVENTION REQUIRED.
        See alert details at: [substrate.details_url]
        ```

---

### 4. Runtime Flow Diagram (Text-Only)

This diagram outlines the process from data collection to CI enforcement.

```
+---------------------------+
| 1. Collection (Runtime)   |
| - Agent on host reads:    |
|   - TPM PCRs              |
|   - IMA measurement list  |
|   - Loaded kernel modules |
+-------------+-------------+
              |
              v
+-------------+-------------+
| 2. Hashing & Comparison   |
| - Attestation Service:    |
|   - Compares PCRs to      |
|     "golden" baseline.    |
|   - Compares IMA list to  |
|     signed manifest.      |
|   - Verifies module sigs. |
+-------------+-------------+
              |
              v
+-------------+-------------+
| 3. Drift Detection        |
| - Attestation Service:    |
|   - Calculates `identity_`|
|     `stability_index`.    |
|   - Generates `drift_flags`|
|     for any mismatch.     |
+-------------+-------------+
              |
              v
+-------------+-------------+
| 4. Governance Tile Update |
| - Monitoring Service:     |
|   - Consumes drift report.|
|   - Maps index/flags to a |
|     `status_light`.       |
|   - Writes to `global_`   |
|     `health.json`.        |
+-------------+-------------+
              |
              v
+-------------+-------------+
| 5. CI Enforcement         |
| - `substrate_governance_` |
|   `check.py` in CI job:   |
|   - Reads `status_light`. |
|   - Exits `65` on `BLOCK`.|
|   - Fails build.          |
+---------------------------+
```
