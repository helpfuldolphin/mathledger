# Drift Radar Suite: Architecture & Design

**Author**: MANUS-G (CI/Governance Systems Architect)
**Status**: DRAFT

## 1. Overview

This document specifies the architecture for the MathLedger Drift Radar Suite, a set of automated CI checks designed to detect and prevent unintended changes to critical project components. Each radar monitors a specific domain (Curriculum, Telemetry, Ledger, HT Triangles) by comparing versioned snapshots of artifacts against a baseline. The suite enforces the core invariant: **No drift shall be silently accepted.**

All radars adhere to a standardized design:
- **Snapshot-based**: They operate on structured, versioned JSON snapshots of the system state.
- **Deterministic**: Radars produce the same output given the same input snapshots.
- **CI-Integrated**: Each radar is a standalone GitHub Actions workflow, blocking pull requests on drift detection.
- **Standardized Semantics**: They share common exit codes and severity classifications.

## 2. Universal Design Principles

### 2.1. Exit Code Semantics

All drift radar jobs will use the following exit codes to ensure consistent interpretation by the CI orchestrator:

| Exit Code | Meaning                | CI Action          | Severity      |
|-----------|------------------------|--------------------|---------------|
| `0`       | **PASS**               | Allow Merge        | `NONE`        |
| `1`       | **FAIL (Drift Detected)**| Block Merge        | `CRITICAL`    |
| `2`       | **WARN (Benign Drift)**  | Allow Merge (w/ review) | `WARNING`     |
| `3`       | **ERROR (Infra)**      | Retry Job          | `N/A`         |
| `4`       | **SKIP (No-Op)**       | Allow Merge        | `INFO`        |

### 2.2. Severity Classification

Drift is classified into two main categories:

- **Breaking Drift (`CRITICAL`)**: A change that violates a schema, breaks a dependency, or alters a cryptographic invariant. Always results in a CI failure (exit code `1`).
- **Non-Breaking Drift (`WARNING`)**: A change that is valid but may be unintended, such as adding a new optional field or changing a description. Results in a CI warning (exit code `2`).

### 2.3. Artifact Generation

Each radar job generates a standardized set of artifacts:

1.  **Drift Report (`drift_report.json`)**: A machine-readable JSON file detailing every detected change, its severity, and location.
2.  **Drift Summary (`drift_summary.md`)**: A human-readable Markdown summary posted as a PR comment.
3.  **Diff (`drift.diff`)**: A unified diff view of the changes between the baseline and current snapshots.

---

## 3. Radar Implementations

### 3.1. Curriculum Drift Radar

- **Purpose**: Monitors the academic content and structure of MathLedger's curriculum to prevent unintended changes to problem definitions, difficulty, or topic organization.
- **CI Workflow**: `.github/workflows/curriculum-drift-gate.yml`
- **Snapshot Artifact**: `artifacts/governance/curriculum_snapshot.json`

#### Snapshot Schema (`schemas/curriculum_snapshot.schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Curriculum Snapshot",
  "type": "object",
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.(\\d+)\\.(\\d+)$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "topic_taxonomy": {
      "type": "object",
      "additionalProperties": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "problems": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "title": { "type": "string" },
          "topic": { "type": "string" },
          "difficulty_score": { "type": "number", "minimum": 0, "maximum": 1 },
          "content_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" }
        },
        "required": ["id", "title", "topic", "difficulty_score", "content_hash"]
      }
    }
  },
  "required": ["version", "timestamp", "topic_taxonomy", "problems"]
}
```

#### Drift Detection Examples

| Drift Type                 | Example                                       | Severity   | Exit Code |
|----------------------------|-----------------------------------------------|------------|-----------|
| **Schema Violation**       | `difficulty_score` becomes a string           | `CRITICAL` | `1`       |
| **Taxonomy Change**        | Topic `algebra` is renamed to `algebra-basics`| `CRITICAL` | `1`       |
| **Problem Content Change** | `content_hash` for problem `p-001` changes    | `CRITICAL` | `1`       |
| **Difficulty Score Shift** | Difficulty of `p-001` changes by >10%         | `WARNING`  | `2`       |
| **New Problem Added**      | A new problem is added to the `problems` array| `INFO`     | `0`       |

---

### 3.2. Telemetry Drift Radar

- **Purpose**: Monitors the structure and format of telemetry events (logs) to ensure data integrity and prevent breaking changes for downstream analytics.
- **CI Workflow**: `.github/workflows/telemetry-drift-gate.yml`
- **Snapshot Artifact**: `artifacts/governance/telemetry_schema_snapshot.json`

#### Snapshot Schema (`schemas/telemetry_schema_snapshot.schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Telemetry Schema Snapshot",
  "type": "object",
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.(\\d+)\\.(\\d+)$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "events": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "description": { "type": "string" },
          "schema": { "type": "object" } 
        },
        "required": ["description", "schema"]
      }
    }
  },
  "required": ["version", "timestamp", "events"]
}
```

#### Drift Detection Examples

| Drift Type               | Example                                         | Severity   | Exit Code |
|--------------------------|-------------------------------------------------|------------|-----------|
| **Field Type Change**    | `user_id` changes from `integer` to `string`    | `CRITICAL` | `1`       |
| **Required Field Removed**| `session_id` is removed from an event schema    | `CRITICAL` | `1`       |
| **New Event Added**      | A new event `user_login_failed` is added        | `INFO`     | `0`       |
| **Optional Field Added** | An optional `device_type` field is added        | `INFO`     | `0`       |
| **Description Change**   | The `description` of an event is updated        | `WARNING`  | `2`       |

---

### 3.3. Ledger Drift Radar

- **Purpose**: Ensures the cryptographic integrity and deterministic state of the ledger by comparing block hashes and Merkle roots.
- **CI Workflow**: `.github/workflows/ledger-drift-gate.yml`
- **Snapshot Artifact**: `artifacts/governance/ledger_snapshot.json`

#### Snapshot Schema (`schemas/ledger_snapshot.schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Ledger Snapshot",
  "type": "object",
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.(\\d+)\\.(\\d+)$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "chain_id": { "type": "string" },
    "height": { "type": "integer", "minimum": 0 },
    "last_block_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
    "blocks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "height": { "type": "integer" },
          "hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "prev_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "merkle_root": { "type": "string", "pattern": "^[a-f0-9]{64}$" }
        },
        "required": ["height", "hash", "prev_hash", "merkle_root"]
      }
    }
  },
  "required": ["version", "timestamp", "chain_id", "height", "last_block_hash", "blocks"]
}
```

#### Drift Detection Examples

| Drift Type                 | Example                                       | Severity   | Exit Code |
|----------------------------|-----------------------------------------------|------------|-----------|
| **Broken Chain**           | `block[i].prev_hash != block[i-1].hash`       | `CRITICAL` | `1`       |
| **State Hash Mismatch**    | `last_block_hash` changes for the same height | `CRITICAL` | `1`       |
| **Merkle Root Changed**    | `merkle_root` for a block changes             | `CRITICAL` | `1`       |
| **Non-Linear Height**      | `block[i].height != block[i-1].height + 1`    | `CRITICAL` | `1`       |

---

### 3.4. HT Triangle Drift Radar

- **Purpose**: Verifies the **H_t = SHA256(R_t || U_t)** invariant, which is the core cryptographic seal of the dual-attestation system.
- **CI Workflow**: `.github/workflows/ht-triangle-drift-gate.yml`
- **Snapshot Artifact**: `artifacts/governance/attestation_snapshot.json`

#### Snapshot Schema (`schemas/attestation_snapshot.schema.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Attestation Snapshot",
  "type": "object",
  "properties": {
    "version": { "type": "string", "pattern": "^\\d+\\.(\\d+)\\.(\\d+)$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "attestations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "H_t": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "R_t": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
          "U_t": { "type": "string", "pattern": "^[a-f0-9]{64}$" }
        },
        "required": ["id", "H_t", "R_t", "U_t"]
      }
    }
  },
  "required": ["version", "timestamp", "attestations"]
}
```

#### Drift Detection Examples

| Drift Type                 | Example                                       | Severity   | Exit Code |
|----------------------------|-----------------------------------------------|------------|-----------|
| **H_t Mismatch**           | `H_t != SHA256(R_t || U_t)`                   | `CRITICAL` | `1`       |
| **Invalid Hash Format**    | `H_t` is not a 64-character hex string        | `CRITICAL` | `1`       |
| **Missing Root**           | `R_t` or `U_t` is null or empty               | `CRITICAL` | `1`       |

