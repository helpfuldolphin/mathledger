# Natural Language Failure Modes and Governance Responses

This document outlines the governance responses to specific failure modes detected during the validation of Natural Language Intermediate Representations (NL IR). The goal is to create a robust, future-proof system for handling potential NLU hallucinations and ambiguities before they can be formalized into the MathLedger substrate.

NL governance is **strictly advisory and evidence-bound**. The validation results are attached to an evidence pack as a non-mutating, informational block. This ensures that the original NL-derived data is preserved, while providing a clear, auditable record of any potential issues that require human review before formalization.

## Evidence Pack Structure

The results of the NL IR validation are written to the following path within an evidence pack:

`evidence["governance"]["nl_ir"]`

### Example Governance Block

Below is an example of the governance block for an NL IR payload that was structurally valid but failed semantic validation due to an ambiguous referent and an ontology mismatch.

```json
{
  "id": "EV-PACK-12345",
  "data": {
    "nl_ir": {
      "payload": {
        "statement_id": "NL_007",
        "source_nl": "The agent used the transporter.",
        "logical_form": { "...": "..." },
        "grounding_context": {
          "agent-01": { "nl_reference": ["The agent"], "schema_ref": "Person" },
          "transporter-01": { "nl_reference": ["the transporter"], "schema_ref": "device:Teleporter" }
        },
        "confidence_score": 0.85,
        "disambiguation_notes": [
          { "text": "The agent", "interpretation": "Could refer to multiple system agents.", "resolution": "Requires clarification." }
        ]
      }
    }
  },
  "governance": {
    "nl_ir": {
      "schema_version": "1.0.0",
      "payload_hash": "a1b2c3d4e5f6...",
      "findings": [
        {
          "code": "AMBIGUITY_DETECTED",
          "message": "The NLU processor flagged ambiguities that require resolution.",
          "path": "disambiguation_notes"
        },
        {
          "code": "ONTOLOGY_MISMATCH",
          "message": "Schema reference 'device:Teleporter' not found in known ontology.",
          "path": "grounding_context.transporter-01.schema_ref"
        }
      ]
    }
  }
}
```

---

## Smoke-Test Readiness Checklist

This checklist must be verified before deploying the NL IR validator and governance mechanism to a production environment.

-   [x] **Schema Definition**: `schemas/nl_ir.schema.json` is defined, versioned, and accessible.
-   [x] **Validator Implementation**: `backend/nl/ir_validator.py` is implemented with both schema and semantic checks.
-   [x] **Failure Codes**: `backend/nl/nl_ir_failure_codes.py` contains a comprehensive list of standardized error codes.
-   [x] **Non-Mutation Guarantee**: Unit tests confirm that `attach_nl_evidence` does not mutate the original evidence pack.
-   [x] **Deterministic Hashing**: Unit tests confirm that the `payload_hash` is deterministic for identical inputs.
-   [x] **Canonicalization**: A specific unit test confirms the payload hash is deterministic across dictionary key-order changes.
-   [x] **JSON Serialization**: Unit tests confirm the entire output evidence pack is JSON-serializable.
-   [x] **Governance Documentation**: This document (`GOVERNANCE.md`) is up-to-date with the correct evidence pack structure and examples.
-   [ ] **Ontology Service Integration**: The mock `KNOWN_ONTOLOGY_TYPES` is replaced with a connection to the live, managed ontology service.
-   [ ] **Human-in-the-Loop (HITL) Endpoint**: A mechanism (e.g., API endpoint, queue) is in place to route payloads with non-empty `findings` to a human review interface.

## Deferred Integrations

While the current NL IR validation and governance mechanism is robust and auditable, certain integrations have been intentionally deferred. This deferral is a strategic decision to preserve audit integrity and maintain a clear separation of concerns during the foundational development phases.

### Ontology Service Integration

*   **Current State**: The `ir_validator.py` currently uses a mock `KNOWN_ONTOLOGY_TYPES` set for basic type checking.
*   **Deferred Integration**: Replacing this mock with a connection to a live, managed ontology service (e.g., a GraphQL endpoint, a dedicated microservice).
*   **Rationale for Deferral**: Deferring this integration prevents the introduction of external system dependencies and potential points of failure during the core validator development. It ensures that the current validation logic is fully decoupled from the complexities of a dynamic, versioned ontology service, allowing for a "frozen" contract on what constitutes an acceptable type reference. The audit integrity is preserved by validating against a known, static set, preventing unexpected external changes from altering validation outcomes.

### Human-in-the-Loop (HITL) Routing

*   **Current State**: Payloads with validation `findings` are flagged as `invalid` and attached to evidence packs for human review. The routing mechanism for this review is not yet implemented.
*   **Deferred Integration**: Implementing a dedicated mechanism (e.g., an API endpoint, a message queue, a UI integration) to route these flagged payloads to a human review interface.
*   **Rationale for Deferral**: Deferring HITL routing prevents the coupling of core validation logic with application-specific workflow concerns. It ensures that the detection and flagging of issues is robustly implemented and independently verifiable before introducing the complexities of a human interaction layer. This separation maintains audit integrity by clearly delineating automated validation from human-guided disambiguation and decision-making, allowing for independent auditing of both processes.

### Activation Criteria

Before either the Ontology Service Integration or the Human-in-the-Loop (HITL) Routing is initiated, the following conditions must be met to ensure the continued integrity and auditable nature of the MathLedger system:

*   [ ] **Formal Ontology Specification**: A complete, version-controlled, and independently auditable formal ontology is available and approved.
*   [ ] **Ontology Service API Contract**: The API contract for the ontology service is formally defined and stable.
*   [ ] **Ontology Change Management Policy**: A clear policy for ontology backward compatibility is established and agreed upon.
*   [ ] **Human Intervention Protocol & Audit Trail**: A detailed human intervention protocol is formally documented, including immutable audit trails for all HITL decisions.
*   [ ] **HITL Operational Readiness**: HITL system uptime, SLAs, security, and access controls are established and validated.
*   [ ] **HITL Personnel Certification**: All personnel operating the HITL system have undergone formal training and certification.
*   [ ] **Validation Gateway Proof (if automated)**: Formal proof or high-assurance validation for automated integration gateways (ontology or HITL) is provided.

Activation requires explicit version bump + reviewer sign-off.
Activation requires a versioned PR referencing this checklist and explicit reviewer sign-off.
