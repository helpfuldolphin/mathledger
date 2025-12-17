# Response to External Red Team Review (Palantir-style)

**Document ID**: STRATCOM-REDTEAM-RESPONSE-MEMO-2025-12-17
**Classification**: Internal
**Audience**: Technical Architecture Review Board

---

## 1. Acknowledgment

The red team review identified valid deficiencies. The following criticisms are accepted without qualification:

| Issue | Validity | Disposition |
|-------|----------|-------------|
| `verify-core-loop` naming is deceptive | **Valid** | Target renamed; mock/real paths now explicit |
| Lean verification claims not demonstrated by evaluator path | **Valid** | Claims narrowed; manual Lean invocation documented |
| "tamper-evident" overclaims without signing | **Valid** | Language changed to "hash-verified for internal consistency" |
| L5 "independence" not demonstrated | **Valid** | Renamed to L5-lite with single-environment qualifier |
| No explicit H_t formula test | **Valid** | Unit tests added asserting `SHA256(R_t || U_t)` |
| SHADOW MODE semantically undefined | **Valid** | Deferred to Phase 2; acknowledged as governance gap |
| Manifest not cryptographically signed | **Valid** | Deferred to Phase 2; acknowledged as security gap |
| Isolation audit is self-reported | **Valid** | Acknowledged; caveat added to documentation |

The review correctly identified that evaluator-facing claims exceeded demonstrable evidence. The remediation strategy prioritizes claim narrowing over scope expansion.

---

## 2. Remediations Completed (Phase 1)

### 2.1 verify-core-loop Remediation

| Before | After |
|--------|-------|
| `make verify-core-loop` | `make verify-mock-determinism` (explicit mock mode) |
| Implicit Lean claim | `make verify-lean-single PROOF=<path>` (explicit real Lean) |
| `"lean": "enabled"` in mock scenarios | Mock mode forced via `ML_LEAN_MODE=mock` |

**Files modified**: Makefile, EVALUATOR_QUICKSTART.md, EVALUATOR_GUIDE.md, README.md, CAL_EXP_3_INDEX.md, CAL_EXP_3_RATIFICATION_BRIEF.md, core-loop-verification.yml

### 2.2 Claim Narrowing

| Original Claim | Narrowed Claim |
|----------------|----------------|
| "Lean 4 type-checks proofs" | "Lean 4 is configured to type-check proofs; evaluators must manually invoke Lean to verify specific proofs" |
| "H_t is computed correctly" | "H_t is computed per the implementation in `attestation/dual_root.py`" |
| "tamper-evident" | "hash-verified for internal consistency" |
| "L5 (Uplift Replicated)" | "L5-lite (Replicated in controlled single-environment conditions)" |

**Files modified**: CAL_EXP_3_EVIDENCE_PACKET.tex, CAL_EXP_3_RATIFICATION_BRIEF.md

### 2.3 H_t Formula Test

Added explicit unit tests in `tests/test_dual_root_attestation.py`:

```python
class TestH_tFormulaExplicit:
    def test_h_t_formula_known_values(self):
        # Asserts: H_t == SHA256(R_t || U_t) for fixed inputs

    def test_h_t_formula_concatenation_order(self):
        # Asserts: R_t comes first, not U_t

    def test_h_t_formula_frozen_contract(self):
        # Frozen test vector; fails if formula changes
```

**Test result**: 4 tests passing

### 2.4 Distribution Gate

Created `results/first_light/evidence_pack_first_light/README.md` with explicit gate:

> Distribution of this evidence pack is blocked until Phase 1 (STOPSHIP) remediation is complete.

---

## 3. Remaining Governance Work (Phase 2)

The following items are acknowledged as incomplete and scoped for Phase 2:

| Item | Current State | Phase 2 Deliverable | Status |
|------|---------------|---------------------|--------|
| SHADOW MODE definition | Scattered, inconsistent | `SHADOW_MODE_CONTRACT.md` with formal semantics | **COMPLETE** |
| SHADOW graduation policy | Undefined | `SHADOW_GRADUATION_POLICY.md` stating "never without re-verification" | **COMPLETE** |
| Manifest signing | Not implemented | GPG detached signature on `manifest.json` | Pending |
| Path traversal protection | Not implemented | Containment check in `verify_evidence_pack_integrity.py` | Pending |
| Lean fallback enforcement | Partial | Remove `is_lean_available()` from public API | Pending |
| Isolation audit labeling | Self-reported without caveat | Rename to "self-reported isolation audit" | Pending |

**Phase 2 is partially complete. Remaining items are not claimed as resolved.**

---

## 4. Explicit Non-Claims

The following are **NOT asserted** by this project:

| Non-Claim | Reason |
|-----------|--------|
| Lean verification occurs during `make verify-mock-determinism` | Mock mode uses synthetic artifacts |
| Evidence pack is tamper-evident | Hashing without signing is not tamper-evidence |
| L5 represents independent replication | All runs were single-environment with sequential seeds |
| SHADOW MODE is formally defined | Definition is scattered; formal contract pending |
| Isolation audit is externally verified | Audit is self-reported by the harness |
| Manifest cannot be modified post-hoc | No cryptographic binding to identity or timestamp |
| System makes capability claims | No AI capability, intelligence, or generalization claims are made |

---

## 5. Readiness Statement

**Suitable for sandbox evaluation under the following constraints:**

1. No external distribution of evidence packs until Phase 2 complete
2. All evaluator-facing documentation reflects narrowed claims
3. SHADOW MODE semantics must be formalized before production exposure
4. Quarterly re-review required
5. Manifest signing required before defense/intelligence contexts

**Not suitable for:**
- External audit without Phase 2 completion
- Regulatory submission
- Customer-facing deployment
- Any context requiring tamper-evidence or cryptographic provenance

---

**Phase 1 Status**: COMPLETE
**Phase 2 Status**: NOT STARTED
**Sandbox Admission**: CONDITIONAL (constraints above)

---

*Precision > feelings. Execution > argument.*

---

## Addendum (2025-12-17)

The following Phase 2 governance items have been completed and committed:

- `docs/system_law/SHADOW_MODE_CONTRACT.md` — Formal definition of SHADOW MODE semantics, ownership, and subsystem classification (SHADOW-OBSERVE vs SHADOW-GATED)
- `docs/system_law/SHADOW_GRADUATION_POLICY.md` — States that SHADOW proofs never graduate without re-verification against non-SHADOW criteria

The Phase 2 status table above has been updated to reflect these completions.

No external-facing posture changes result from this update; the readiness statement and constraints remain unchanged.
