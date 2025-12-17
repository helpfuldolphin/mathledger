# Governance Signal: Technical Specification

This document provides the formal technical specification for the Governance Signal framework, complementing the conceptual overview in `GOVERNANCE_SIGNALS.md`.

## 1. Governance Signal JSON Schema

The canonical structure of a Governance Signal is defined by the JSON Schema located at `GOVERNANCE_SIGNAL_SCHEMA.json`. This schema serves as the single source of truth for signal validation.

## 2. End-to-End Flow

The lifecycle of a Governance Signal follows a precise, cryptographically secure flow from creation to decision.

1.  **Signal Emission:** A component (e.g., a monitoring agent, a data integrity checker) detects a state change or completes a scheduled check. It constructs a signal object conforming to the JSON schema, populating all required fields (`originatorId`, `semanticType`, `payload`, etc.).
2.  **Canonicalization & Signature:**
    *   The signal object is canonicalized into a stable string representation (e.g., by sorting keys and removing insignificant whitespace). This ensures that the signature is deterministic.
    *   The originator component uses its private key to generate a digital signature of the canonicalized string.
    *   The signature, along with the originator's public key and the algorithm used, is added to the `cryptographicMetadata` field.
3.  **Ingestion:** The signed signal is transmitted to a central ingestion service (e.g., a message queue like Kafka or a dedicated API endpoint). The ingestion service performs initial validation against the JSON schema. Invalid signals are rejected immediately.
4.  **Verification & Fusion:**
    *   A **Fusion Engine** consumes valid signals from the ingestion service.
    *   For each signal, it first performs cryptographic verification: it re-canonicalizes the received signal (minus the signature) and uses the provided public key to verify the signature. **An invalid signature causes the signal to be discarded and a security alert to be raised.**
    *   Verified signals are then processed by the fusion logic. The engine evaluates the signal based on its `semanticType`, `severity`, `originatorId`, and `ttl`. It considers this signal in the context of other recent, valid signals from related components.
5.  **Decision:** Based on the fusion logic, a "license to operate" decision is rendered. This is not a single binary state but can be a nuanced output, such as:
    *   **NO_ACTION:** The system is nominal.
    *   **LOG_ADVISORY:** A low-severity event occurred; log for human review.
    *   **TRIGGER_ALERT:** A high-severity event or a concerning pattern of events occurred; notify on-call personnel.
    *   **INITIATE_ABORT_GATE:** A critical integrity or security failure was detected; immediately halt the affected process or isolate the component.

## 3. Formal Definition of Signal Semantics

### Enumerations (`semanticType`)

*   **`HEARTBEAT_OK`**: A routine signal indicating a component is alive and has passed its internal self-checks. Low severity (1), short TTL.
*   **`RESOURCE_CONSTRAINT_WARN`**: A component is approaching a resource limit (e.g., CPU, memory, budget). Medium severity (2-3). Payload should contain metric details.
*   **`BEHAVIORAL_ANOMALY_WARN`**: A component's behavior deviates from its expected pattern (e.g., latency spike, unusual error rate). Medium severity (3). Payload should contain anomaly details.
*   **`INTEGRITY_CHECK_FAIL`**: A data integrity check has failed (e.g., a hash mismatch in a critical data structure). High severity (4). Payload must contain `observedHash` and `expectedHash`. **This signal type must be monotonic (`isMonotonic: true`).**
*   **`SECURITY_THRESHOLD_EXCEEDED`**: A security-related metric has been breached (e.g., excessive failed login attempts). High severity (4-5).
*   **`ABORT_GATE_TRIGGERED`**: A component has autonomously halted its own execution due to a critical internal failure. Highest severity (5).

### Constraints

*   **TTL (Time-to-Live):** Every signal has a `ttl` in seconds. The Fusion Engine **must** discard any signal whose `timestamp` is older than the current time minus its `ttl`. This prevents stale signals from influencing decisions.
*   **Monotonicity (`isMonotonic`):** For signals where `isMonotonic` is `true` (like `INTEGRITY_CHECK_FAIL`), the Fusion Engine should maintain a record of the state. If a subsequent signal from the same originator implies a "healing" of this condition (i.e., the failure is no longer present), it should be treated as a highly suspicious event and potentially trigger a higher-level security alert. The state should only escalate.

## 4. Examples

### Example 1: `HEARTBEAT_OK` Signal

This is a routine "I'm alive" signal from a core service.

```json
{
  "signalId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "originatorId": "substrate:prod:us-east-1:node-12345",
  "timestamp": "2025-12-10T22:00:00Z",
  "semanticType": "HEARTBEAT_OK",
  "severity": 1,
  "ttl": 60,
  "isMonotonic": false,
  "payload": {
    "cpuUsage": 0.15,
    "memoryUsage": 0.4
  },
  "cryptographicMetadata": {
    "signature": "30450221008...",
    "publicKey": "0479be667ef...",
    "signingAlgorithm": "ECDSA-P256"
  }
}
```

### Example 2: `RESOURCE_CONSTRAINT_WARN` Signal

A component warns that its computational budget is nearly exhausted.

```json
{
  "signalId": "a1b2c3d4-e5f6-4a7b-8c9d-0a1b2c3d4e5f",
  "originatorId": "agent:manus-d:phase-ix:data-analyzer-7",
  "timestamp": "2025-12-10T22:05:10Z",
  "semanticType": "RESOURCE_CONSTRAINT_WARN",
  "severity": 3,
  "ttl": 300,
  "isMonotonic": false,
  "payload": {
    "metricName": "computation_budget_seconds",
    "value": 2890,
    "threshold": 3000
  },
  "cryptographicMetadata": {
    "signature": "304502207...",
    "publicKey": "04a5c6e7f...",
    "signingAlgorithm": "ECDSA-P256"
  }
}
```

### Example 3: `INTEGRITY_CHECK_FAIL` (BLOCK) Signal

A critical failure. An auditor has detected that the hash of a core data structure does not match the expected value. This signal has the highest severity and `isMonotonic` is true, meaning this failure state cannot be "un-failed" without manual intervention.

```json
{
  "signalId": "e3b0c442-98fc-11eb-a8b3-0242ac130003",
  "originatorId": "auditor:mirror:dag-integrity-sentinel",
  "timestamp": "2025-12-10T22:10:00Z",
  "semanticType": "INTEGRITY_CHECK_FAIL",
  "severity": 5,
  "ttl": 86400,
  "isMonotonic": true,
  "payload": {
    "dataStructure": "main_ht_dag",
    "observedHash": "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
    "expectedHash": "sha256:a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
  },
  "cryptographicMetadata": {
    "signature": "304602210...",
    "publicKey": "04c1d2e3f...",
    "signingAlgorithm": "ECDSA-P256"
  }
}
```
