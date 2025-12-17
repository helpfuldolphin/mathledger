# SHADOW GRADUATION POLICY

**Document Type**: Binding Governance Policy
**Status**: CANONICAL
**Version**: 1.0.0
**Effective Date**: 2025-12-17
**Owner**: SHADOW MODE GOVERNANCE AUTHORITY (SMGA)
**References**: [SHADOW_MODE_CONTRACT.md](./SHADOW_MODE_CONTRACT.md)

---

## 1. Purpose

This document establishes the ONLY permitted mechanism for transitioning artifacts, evidence, or verification results from SHADOW MODE status to non-SHADOW (canonical/enforced) status.

**Default Policy**: No SHADOW artifact graduates to canonical status unless this document is explicitly invoked and its procedures are fully satisfied.

---

## 2. Definitions

### 2.1 Graduation

**Graduation** is the formal transition of an artifact from SHADOW MODE status to canonical/enforced status, wherein:

1. The artifact becomes citable as authoritative evidence
2. The artifact may be used for claim substantiation
3. The artifact's verification results become binding
4. The artifact's thresholds become enforceable

### 2.2 SHADOW Artifact

A **SHADOW artifact** is any output produced under SHADOW-OBSERVE or SHADOW-GATED conditions, including but not limited to:

- Verification reports
- Audit logs
- Metric snapshots
- Threshold evaluations
- Compliance assessments
- Evidence packets

### 2.3 Canonical Artifact

A **canonical artifact** is any output that:

1. Was produced under non-SHADOW conditions, OR
2. Has completed the graduation procedure defined in Section 4

---

## 3. Forbidden Transitions

### 3.1 Enumeration of Forbidden Implicit Transitions

The following transitions are PROHIBITED:

| ID | Forbidden Transition | Reason |
|----|---------------------|--------|
| F1 | Time-based graduation | Artifacts do not become canonical by age |
| F2 | Reference-based graduation | Citing a SHADOW artifact does not canonicalize it |
| F3 | Accumulation-based graduation | Multiple SHADOW runs do not sum to canonical status |
| F4 | Silence-based graduation | Absence of objection does not constitute approval |
| F5 | Downstream-use graduation | Using SHADOW output in another system does not canonicalize it |
| F6 | Copy-based graduation | Copying SHADOW artifacts to canonical locations does not canonicalize them |
| F7 | Relabel-based graduation | Removing "SHADOW" labels does not change artifact status |
| F8 | Version-based graduation | New versions of SHADOW artifacts remain SHADOW |
| F9 | Authority-assumption graduation | Assuming graduation authority does not grant it |
| F10 | Emergency-based graduation | Urgency does not bypass graduation requirements |

### 3.2 Forbidden Language

The following phrases MUST NOT appear in graduation-related documentation:

- "automatically becomes"
- "implicitly graduates"
- "effectively canonical"
- "treated as canonical"
- "functionally equivalent to canonical"
- "grandfathered in"
- "legacy exception"

---

## 4. Permitted Transition Path

### 4.1 Exclusive Graduation Procedure

The ONLY permitted graduation path is:

```
SHADOW Artifact
      │
      ▼
┌─────────────────────────────────────┐
│ STEP 1: Graduation Request          │
│ - Formal request to SMGA            │
│ - Identifies artifact by hash/ID    │
│ - States intended canonical use     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ STEP 2: Re-Verification             │
│ - Execute verification under        │
│   NON-SHADOW criteria               │
│ - Use canonical thresholds          │
│ - Produce new canonical artifact    │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ STEP 3: Comparison Audit            │
│ - Compare SHADOW result to          │
│   canonical result                  │
│ - Document any discrepancies        │
│ - Discrepancies require resolution  │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ STEP 4: SMGA Approval               │
│ - SMGA reviews comparison           │
│ - SMGA issues graduation decree     │
│ - Decree references this policy     │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ STEP 5: Canonical Registration      │
│ - New artifact (from Step 2)        │
│   registered as canonical           │
│ - Original SHADOW artifact          │
│   remains SHADOW (archived)         │
└─────────────────────────────────────┘
      │
      ▼
Canonical Artifact
```

### 4.2 Critical Requirement: Re-Verification

**The SHADOW artifact itself NEVER graduates.**

Graduation produces a NEW canonical artifact through re-verification. The original SHADOW artifact retains its SHADOW status permanently. This ensures:

1. Canonical artifacts are always produced under canonical conditions
2. SHADOW artifacts remain available for audit comparison
3. No ambiguity exists about artifact provenance

### 4.3 Graduation Decree Format

A valid graduation decree MUST contain:

```yaml
graduation_decree:
  decree_id: "<unique identifier>"
  policy_reference: "SHADOW_GRADUATION_POLICY v1.0.0"
  shadow_artifact:
    id: "<SHADOW artifact identifier>"
    hash: "<SHADOW artifact hash>"
    produced_at: "<ISO-8601>"
  canonical_artifact:
    id: "<canonical artifact identifier>"
    hash: "<canonical artifact hash>"
    produced_at: "<ISO-8601>"
  comparison_result: "MATCH | DIVERGENT"
  divergence_resolution: "<if DIVERGENT, explanation>"
  smga_authority: "<SMGA role holder>"
  effective_date: "<ISO-8601>"
```

---

## 5. Default Policy

### 5.1 Statement

**No SHADOW artifact graduates to canonical status unless:**

1. This document (SHADOW_GRADUATION_POLICY) is explicitly invoked
2. The procedure in Section 4 is fully executed
3. A graduation decree is issued by SMGA
4. The decree is recorded in the graduation registry

### 5.2 Burden of Proof

The burden of proving canonical status rests with the party asserting it. In the absence of a valid graduation decree:

- The artifact is presumed SHADOW
- Claims based on the artifact are presumed non-binding
- Evidence based on the artifact is presumed informational only

### 5.3 Fail-Closed Semantics

When artifact status is ambiguous:

- Treat as SHADOW
- Do not cite as canonical
- Do not use for binding claims
- Initiate status clarification procedure

---

## 6. Threat Model

### 6.1 Exploitation of Undefined Graduation

Undefined or implicit graduation paths create exploitable attack surfaces. An adversary—or an inadvertent process error—could leverage ambiguous graduation semantics to elevate SHADOW artifacts to canonical status without re-verification. This would allow evidence produced under relaxed SHADOW conditions (where failures are non-blocking and thresholds may differ) to be cited as authoritative proof of system properties. Such exploitation could substantiate false claims, bypass verification requirements, or create audit trails that appear valid but lack canonical verification. The fail-closed policy in this document ensures that SHADOW artifacts cannot gain canonical authority through any path other than explicit re-verification under canonical conditions, eliminating the class of vulnerabilities arising from status ambiguity.

---

## 7. Audit Interface

### 7.1 Verification Questions

Auditors MAY ask:

| Question | Expected Answer |
|----------|-----------------|
| Is this artifact canonical? | YES with decree reference, or NO |
| What is the graduation decree ID? | Valid decree ID or "Not graduated" |
| Where is the re-verification evidence? | Path to canonical verification run |
| Was the SHADOW artifact preserved? | YES with archive reference |

### 7.2 Red Flags

Auditors SHOULD flag:

- Canonical claims without graduation decree
- SHADOW artifacts in canonical registries
- Missing re-verification evidence
- Graduation decrees without SMGA signature
- Policy references to versions other than current

---

## 8. Exceptions

### 8.1 No Exceptions Exist

There are no exceptions to this policy.

### 8.2 Emergency Procedures

Emergency conditions do not bypass graduation requirements. If canonical evidence is urgently needed:

1. Execute expedited re-verification under canonical conditions
2. Follow standard graduation procedure
3. Document urgency in graduation decree
4. Urgency does not reduce procedural requirements

---

## 9. Governance

### 9.1 Policy Owner

SHADOW MODE GOVERNANCE AUTHORITY (SMGA) owns this policy.

### 9.2 Amendment Authority

Amendments require:

1. STRATCOM GOVERNANCE resolution
2. SMGA concurrence
3. Published amendment with effective date
4. Version increment

### 9.3 Conflict Resolution

In case of conflict between this policy and other documents:

- This policy governs graduation semantics
- SHADOW_MODE_CONTRACT.md governs SHADOW MODE semantics
- STRATCOM GOVERNANCE resolves unresolved conflicts

---

## 10. References

### 10.1 This Policy References

- [SHADOW_MODE_CONTRACT.md](./SHADOW_MODE_CONTRACT.md)
- STRATCOM GOVERNANCE authority

### 10.2 This Policy Is Referenced By

- All CAL-EXP-* specifications
- All evidence packet specifications
- All claim substantiation procedures
- All audit procedures

---

**CANONICAL** — SHADOW artifacts NEVER graduate implicitly.

**Version**: 1.0.0
**Owner**: SHADOW MODE GOVERNANCE AUTHORITY (SMGA)
