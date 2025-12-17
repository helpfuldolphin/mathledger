# SHADOW MODE CONTRACT

**Document Type**: Binding Governance Specification
**Status**: CANONICAL
**Version**: 1.0.0
**Effective Date**: 2025-12-17
**Owner**: SHADOW MODE GOVERNANCE AUTHORITY (SMGA)

---

## 1. Definitions

### 1.1 SHADOW MODE

**SHADOW MODE** is a system operating state in which verification, validation, or enforcement logic executes without affecting the primary control flow of the system.

SHADOW MODE exists in exactly TWO sub-modes:

1. **SHADOW-OBSERVE**
2. **SHADOW-GATED**

No other SHADOW MODE variants are recognized. Any reference to "SHADOW MODE" without explicit sub-mode qualification SHALL be interpreted as SHADOW-OBSERVE.

### 1.2 Primary Control Flow

The **primary control flow** is the sequence of operations that determines:
- Whether a process completes successfully or fails
- Whether a CI pipeline passes or fails
- Whether a claim is accepted or rejected
- Whether a transaction is committed or rolled back

### 1.3 Signal

A **signal** is any output emitted by SHADOW MODE logic, including but not limited to:
- Log entries
- Metric emissions
- Structured JSON artifacts
- Warning messages
- Diagnostic reports

---

## 2. SHADOW-OBSERVE

### 2.1 Definition

**SHADOW-OBSERVE** is the SHADOW MODE sub-mode in which logic executes for observational purposes only. SHADOW-OBSERVE logic MUST NOT influence the primary control flow under any circumstance.

### 2.2 Allowed Actions

| Action | Permitted |
|--------|-----------|
| Execute verification logic | YES |
| Emit log entries | YES |
| Emit metrics | YES |
| Write diagnostic artifacts | YES |
| Record audit trails | YES |
| Compute derived values | YES |
| Compare against thresholds | YES |

### 2.3 Forbidden Actions

| Action | Permitted |
|--------|-----------|
| Return non-zero exit code based on verification result | NO |
| Block CI pipeline progression | NO |
| Reject claims based on verification result | NO |
| Modify system state based on verification result | NO |
| Gate deployments | NO |
| Trigger alerts that require human intervention | NO |
| Raise exceptions that propagate to callers | NO |

### 2.4 Signal Emissions

SHADOW-OBSERVE MAY emit the following signals:

```
SHADOW-OBSERVE signals:
  - Level: INFO, DEBUG, WARN (never ERROR that halts)
  - Content: Verification results, threshold comparisons, diagnostics
  - Destination: Logs, metrics systems, artifact files
  - Effect: None on primary control flow
```

### 2.5 System Behavior Impact

**None.** SHADOW-OBSERVE logic MUST be removable from the system with zero change to observable system behavior (excluding the signals themselves).

### 2.6 Failure Handling

If SHADOW-OBSERVE logic encounters an error (exception, timeout, resource unavailability):

1. The error MUST be caught and logged
2. The primary control flow MUST proceed unaffected
3. The SHADOW-OBSERVE operation MUST fail silently from the perspective of the caller

---

## 3. SHADOW-GATED

### 3.1 Definition

**SHADOW-GATED** is the SHADOW MODE sub-mode in which logic executes with the authority to block explicitly named operations. SHADOW-GATED represents a transitional state between SHADOW-OBSERVE and full enforcement.

### 3.2 Allowed Actions

| Action | Permitted |
|--------|-----------|
| All SHADOW-OBSERVE actions | YES |
| Block explicitly named operations (per gate registry) | YES |
| Return non-zero exit code for gated operations | YES |
| Emit ERROR-level signals for gated violations | YES |

### 3.3 Forbidden Actions

| Action | Permitted |
|--------|-----------|
| Block operations not in gate registry | NO |
| Modify production data | NO |
| Reject claims without gate registry entry | NO |
| Enforce thresholds not explicitly registered | NO |

### 3.4 Gate Registry Requirement

SHADOW-GATED enforcement REQUIRES a gate registry entry specifying:

```yaml
gate_registry_entry:
  gate_id: "<unique identifier>"
  operation: "<operation being gated>"
  condition: "<enforcement condition>"
  enforcement_level: "BLOCK | WARN"
  effective_date: "<ISO-8601 date>"
  authority: "<authorizing role>"
```

Operations without a gate registry entry MUST NOT be blocked.

### 3.5 Signal Emissions

SHADOW-GATED MAY emit the following signals:

```
SHADOW-GATED signals:
  - Level: INFO, DEBUG, WARN, ERROR
  - Content: Verification results, gate violations, enforcement actions
  - Destination: Logs, metrics systems, artifact files, CI output
  - Effect: MAY affect primary control flow for registered gates ONLY
```

### 3.6 System Behavior Impact

**Scoped.** SHADOW-GATED logic affects system behavior ONLY for operations explicitly registered in the gate registry. All other operations proceed as under SHADOW-OBSERVE.

### 3.7 Failure Handling

If SHADOW-GATED logic encounters an error:

1. For gated operations: The operation MUST fail-safe (block the operation)
2. For non-gated operations: Proceed as SHADOW-OBSERVE (fail silently)

---

## 4. Prohibited Language

### 4.1 Ambiguous Descriptors

The following phrases are PROHIBITED when describing SHADOW-GATED behavior:

| Prohibited Phrase | Reason |
|-------------------|--------|
| "observational only" | Implies no blocking capability |
| "advisory" | Implies non-binding |
| "informational" | Implies no enforcement |
| "passive" | Implies no system impact |
| "non-blocking" | Contradicts SHADOW-GATED semantics |

### 4.2 Required Precision

SHADOW-GATED documentation MUST explicitly state:
1. Which operations are gated
2. Under what conditions gating occurs
3. What happens when a gate triggers

---

## 5. Ownership

### 5.1 SHADOW MODE GOVERNANCE AUTHORITY (SMGA)

The **SHADOW MODE GOVERNANCE AUTHORITY (SMGA)** is the role responsible for:

| Responsibility | Description |
|----------------|-------------|
| Sub-mode assignment | Determining whether a component operates in SHADOW-OBSERVE or SHADOW-GATED |
| Gate registry management | Approving, modifying, and revoking gate registry entries |
| Escalation authority | Escalating SHADOW-GATED to full enforcement |
| Contract interpretation | Resolving ambiguities in this contract |
| Compliance certification | Certifying that implementations conform to this contract |

### 5.2 Authority Hierarchy

```
STRATCOM GOVERNANCE
       │
       ▼
SHADOW MODE GOVERNANCE AUTHORITY (SMGA)
       │
       ├──▶ SHADOW-OBSERVE components
       │
       └──▶ SHADOW-GATED components
                    │
                    ▼
              Gate Registry
```

SMGA reports to STRATCOM GOVERNANCE. SMGA decisions may be overridden by STRATCOM GOVERNANCE.

---

## 6. Non-Goals

SHADOW MODE is NOT:

| Non-Goal | Clarification |
|----------|---------------|
| A testing mode | SHADOW MODE operates on production data and real operations |
| A dry-run mode | SHADOW MODE logic executes fully, not partially |
| A simulation | SHADOW MODE interacts with real system state |
| A fallback mode | SHADOW MODE is not activated upon failure of other modes |
| A degraded mode | SHADOW MODE is a deliberate operating state, not a failure state |
| An opt-in feature | SHADOW MODE assignment is determined by SMGA, not by operators |
| A debugging tool | SHADOW MODE serves governance purposes, not debugging |

---

## 7. Regulatory Compliance Note

### 7.1 Interpretation Guidance

For regulatory and audit purposes:

**SHADOW-OBSERVE:**
- Represents pre-enforcement validation
- Produces evidence of system behavior without affecting outcomes
- Audit artifacts from SHADOW-OBSERVE demonstrate what enforcement WOULD have done
- No claims about system guarantees should be derived from SHADOW-OBSERVE results
- SHADOW-OBSERVE failures do not constitute system failures

**SHADOW-GATED:**
- Represents partial enforcement with explicit scope
- Gate registry entries constitute binding commitments
- Operations blocked by SHADOW-GATED are formally rejected
- Audit artifacts from SHADOW-GATED demonstrate actual enforcement
- SHADOW-GATED failures for registered gates constitute system enforcement

### 7.2 Evidence Interpretation

| Evidence Type | SHADOW-OBSERVE | SHADOW-GATED |
|---------------|----------------|--------------|
| Verification passed | Informational only | Binding for gated operations |
| Verification failed | Informational only | Binding for gated operations |
| Operation completed | Unrelated to verification | Verification passed (if gated) |
| Operation blocked | N/A (cannot block) | Verification failed (if gated) |

### 7.3 Transition Semantics

The transition from SHADOW-OBSERVE to SHADOW-GATED to full enforcement follows this protocol:

```
SHADOW-OBSERVE
     │
     │ (SMGA approval + gate registry entry)
     ▼
SHADOW-GATED
     │
     │ (STRATCOM approval + enforcement decree)
     ▼
FULL ENFORCEMENT
```

Each transition requires explicit authorization. Implicit escalation is prohibited.

---

## 8. Compliance Requirements

### 8.1 Implementation Requirements

All SHADOW MODE implementations MUST:

1. Declare sub-mode explicitly (SHADOW-OBSERVE or SHADOW-GATED)
2. For SHADOW-GATED: Reference gate registry entries by ID
3. Catch and handle all exceptions internally
4. Never propagate failures to callers (SHADOW-OBSERVE) except for registered gates (SHADOW-GATED)
5. Emit structured signals conforming to the signal schemas
6. Include sub-mode in all emitted artifacts

### 8.2 Documentation Requirements

All documents referencing SHADOW MODE MUST:

1. Specify sub-mode (SHADOW-OBSERVE or SHADOW-GATED)
2. For SHADOW-GATED: List gate registry entries
3. Avoid prohibited language (Section 4)
4. Reference this contract as the authoritative definition

### 8.3 Audit Requirements

SHADOW MODE operations MUST produce audit artifacts containing:

```json
{
  "shadow_mode": "SHADOW-OBSERVE | SHADOW-GATED",
  "gate_ids": ["<gate_id>", ...],
  "verification_result": "<PASS | FAIL | ERROR>",
  "system_impact": "<NONE | BLOCKED | ALLOWED>",
  "timestamp": "<ISO-8601>",
  "contract_version": "1.0.0"
}
```

---

## 9. Amendment Process

This contract may be amended only by:

1. STRATCOM GOVERNANCE resolution
2. Published amendment with effective date
3. Version increment
4. Notification to all dependent documents

Amendments are not retroactive. Prior SHADOW MODE operations are governed by the contract version in effect at the time of operation.

---

## 10. References

This contract is referenced by:

- All CAL-EXP-* specifications
- All verifier implementations
- All CI workflow configurations
- All governance signal specifications

This contract references:

- STRATCOM GOVERNANCE authority
- Gate registry schema (to be defined)

---

**CANONICAL** — This document is the single authoritative definition of SHADOW MODE.

**Version**: 1.0.0
**Owner**: SHADOW MODE GOVERNANCE AUTHORITY (SMGA)
