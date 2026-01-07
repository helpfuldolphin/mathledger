# Agent Audit Kit (Lane B Companion)

**Status:** PARKED
**Classification:** Lane B / Forensic / Evidence-only
**Phase:** Architectural companion (not integration)
**Not included in fm.tex**

---

## Summary

Agent Audit Kit (AAK) is a forensic evidence substrate designed for revenue-facing agent deployments. It captures, hashes, and packages agent execution traces for post-hoc analysis and dispute resolution. AAK intentionally reuses MathLedger-compatible primitives (RFC 8785 canonicalization, domain-separated SHA256, hash-chained logs) to ensure evidence portability, but it operates entirely outside the authority-bearing governance layer. AAK produces evidence artifacts only; it never produces verdicts, trust assignments, or compliance certifications.

---

## Relationship to MathLedger

| Aspect | MathLedger (Lane A) | Agent Audit Kit (Lane B) |
|--------|---------------------|--------------------------|
| Purpose | Governance / authority | Forensics / evidence |
| Outputs | VERIFIED / REFUTED / ABSTAINED | Hash-chained logs, replay bundles |
| Trust class | Assigns (MV, PA, ADV, FV) | Never assigns |
| Authority | Produces authority-bearing claims | Produces evidence only |
| U_t / R_t / H_t | Participates | Never participates |

### Shared Primitives

- **RFC 8785 canonical JSON** — deterministic serialization for hash stability
- **Domain-separated SHA256** — prefixed hashing to prevent cross-context collisions
- **Hash-chained event logs** — tamper-evident sequential records
- **Portable evidence bundles** — standalone verification without external dependencies

### Explicit Separation

- **Primitive reuse does NOT imply governance equivalence.** Using the same hashing algorithm does not make AAK outputs authority-bearing.
- **Evidence is not authority.** AAK artifacts may inform human judgment but cannot substitute for verification routes.
- **No trust class leakage.** AAK outputs inherit no trust class; they are raw forensic material.

---

## What AAK Produces

| Artifact | Description |
|----------|-------------|
| Hash-chained event logs | Tamper-evident sequential record of agent actions |
| Replay bundles | Portable packages enabling offline trace reconstruction |
| Heuristic threat flags | Non-authoritative signals for anomaly triage |
| Provenance labels | Synthetic (generated) vs captured (observed) distinction |
| Evidence manifests | Indexed artifact inventories with integrity hashes |

---

## What AAK NEVER Produces

| Prohibition | Rationale |
|-------------|-----------|
| No VERIFIED / REFUTED / ABSTAINED | These are Lane A governance outcomes |
| No trust class assignment | Trust typing is a governance primitive |
| No authority upgrades | Evidence cannot promote itself to authority |
| No compliance certification | Certification requires verification routes |
| No U_t / R_t / H_t participation | Ledger state is Lane A only |

---

## Lane Boundary Rule (Hard)

```
┌─────────────────────────────────────────────────────────────────────┐
│ LANE BOUNDARY RULE                                                  │
│                                                                     │
│ AAK artifacts may be referenced as supplementary evidence,         │
│ but may NEVER be treated as authority-bearing inputs to Lane A     │
│ without an explicit admissibility contract and independent         │
│ verification.                                                       │
│                                                                     │
│ Evidence ≠ Authority.                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Re-open Conditions

A future Phase II bridge between Lane B evidence and Lane A governance might be considered only if:

1. **Deterministic tool verification.** AAK-captured tool invocations can be replayed deterministically with pinned inputs/outputs, enabling MV-route admission.

2. **Formalized verifier routes.** A PA or MV route is defined that accepts AAK replay bundles as input and produces governance-compatible outcomes.

3. **Admissibility contract.** An explicit contract specifies which AAK artifact types, under what conditions, may be admitted as evidence for Lane A verification—with version pinning, hash binding, and failure-mode enumeration.

**Until these conditions are met:**
Lane B remains forensic-only. No bridge. No leakage.

---

## Non-Claims

- This document does not assert that AAK is correct, safe, or complete.
- This document does not assert that AAK outputs are trustworthy.
- This document does not propose integration between AAK and MathLedger.
- This document does not claim that forensic evidence reduces verification cost.
- This document exists solely to clarify architectural separation and prevent future scope confusion.

---

## Status

Awareness only.
Not a claim.
Not an obligation.
Not included in fm.tex.

**Activation condition:**
Only revisit if Phase II explicitly addresses Lane B → Lane A bridging with formalized verifier routes.

**Until then:**
This remains parked as an architectural companion note.
