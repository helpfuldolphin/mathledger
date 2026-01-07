# Audit Artifact Taxonomy (Binding)

**Status:** BINDING
**Effective:** 2026-01-05
**Authority:** Adversarial Closure Engineering (ACE) doctrine

---

## Definitions

**External Audit**: Any audit performed by, or simulating, a hostile, cold-start, or third-party evaluator. The auditor has no privileged access and operates under adversarial assumptions.

**Internal Audit**: Any diagnostic, exploration, or scratch work performed by operators or agents during development. These artifacts support the development process but do not constitute formal adversarial evaluation.

**Closure Matrix**: The authoritative summary artifact that records all BLOCKING, MAJOR, and MINOR findings from external audits, with evidence of resolution.

---

## Classification Rules

### `docs/external_audits/` SHALL contain:

1. All audits produced by a hostile/cold/external evaluator role
2. Both PASS and FAIL outcomes (failures are never hidden)
3. All closure matrices (`CLOSURE_MATRIX_*.md`)
4. Any audit that an acquisition committee, regulator, or third-party reviewer could independently reproduce

### `docs/internal_audits/` SHALL contain:

1. Response documents (`*_response.md`)
2. Operator diagnostics and investigation notes
3. Agent scratch work
4. Any artifact not framed as a formal hostile audit

---

## Invariants

**INV-1: External FAIL Preservation**
External audit FAILs SHALL remain in `docs/external_audits/` permanently. A subsequent PASS does not delete or relocate the prior FAIL. History is append-only.

**INV-2: No Post-Hoc Reclassification**
An artifact classified as an external audit at creation time SHALL NOT be reclassified to internal after the fact. Classification is determined by auditor role at execution time, not by outcome or convenience.

**INV-3: Closure Matrix Authority**
Closure matrices are the only authoritative summary artifacts. Raw audit transcripts provide provenance but do not override closure matrix status.

**INV-4: Evaluator Path Separation**
The public evaluator path (website navigation, FOR_AUDITORS.md) surfaces closure matrices as the primary reference. Full audit transcripts are preserved in the repository for completeness.

---

## Decision Heuristic

```
Q: Could an acquisition committee have run this audit independently?
   YES → external_audits/
   NO  → internal_audits/

Q: Was the auditor operating in a hostile/cold-start capacity?
   YES → external_audits/
   NO  → internal_audits/

Q: Is this a formal gate audit (Gate 2, Gate 3)?
   YES → external_audits/
   NO  → Evaluate based on auditor role
```

---

## Edge Cases

| Scenario | Classification | Rationale |
|----------|----------------|-----------|
| Short audits (one-line FAILs) | external | Length is irrelevant; auditor role determines classification |
| Re-runs after fixes | Both preserved in external | Each run is an independent audit event |
| Multiple FAILs before PASS | All FAILs in external | History is append-only |
| Response documents | internal | Operator artifacts explaining remediation |
| Agent diagnostics | internal | Unless formally declared as hostile audit |

---

## Prohibitions

1. **No deletion** of external audit artifacts
2. **No reclassification** of external audits to internal after the fact
3. **No hiding** of FAIL outcomes
4. **No rewriting** of audit history

---

## Changelog

- 2026-01-05: Initial taxonomy established and enforced
