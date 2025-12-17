# run_shadow_audit.py v0.1 — SCOPE COMPLIANCE

> **STOP.** This PR touches `run_shadow_audit.py` or related files.
> Before proceeding, confirm you have read the **canonical contract**:
> - `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` **(CANONICAL)**
>
> Supporting documents:
> - `docs/system_law/calibration/RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` (enforcement protocol)
> - `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_SPEC.md` (historical reference)

---

## Frozen CLI Check

- [ ] No new arguments added
- [ ] No arguments removed or renamed
- [ ] `--input`, `--output`, `--deterministic`, `--seed`, `--verbose`, `--schema-version` unchanged

## No New Metrics Check

- [ ] No new formulas introduced
- [ ] No new statistical methods introduced
- [ ] No new thresholds or tuning parameters introduced
- [ ] No new metric fields in summary.json (except `_diagnostics.*`)

## SHADOW-Only Semantics Check

- [ ] This code NEVER affects real governance decisions
- [ ] All outputs are advisory/observational only
- [ ] No database writes occur
- [ ] No external network calls occur

## Frozen Invariants Check

| ID | Invariant | Holds? |
|----|-----------|--------|
| FI-001 | SHADOW-only execution | [ ] YES |
| FI-002 | Advisory-only output | [ ] YES |
| FI-003 | Deterministic when flagged | [ ] YES |
| FI-004 | No external network calls | [ ] YES |
| FI-005 | No database writes | [ ] YES |
| FI-006 | Schema version "1.0.0" | [ ] YES |
| FI-007 | Self-verifying manifest | [ ] YES |

## Kill-Switch Review (Reviewer completes)

| # | Question | Answer |
|---|----------|--------|
| 1 | Does this PR add, remove, or rename ANY CLI argument? | [ ] NO |
| 2 | Does this PR introduce ANY new metric, formula, or threshold? | [ ] NO |
| 3 | Does this PR cause ANY effect on real governance decisions? | [ ] NO |
| 4 | Does this PR add ANY external network or database dependency? | [ ] NO |
| 5 | Does this PR change exit code semantics or output file paths? | [ ] NO |

**If ANY answer is YES:** REJECT immediately. Do not negotiate.

## Change Classification

Select ONE:

- [ ] **PATCH (0.1.x):** Bug fix only. No interface changes.
- [ ] **EXPERIMENTAL:** Changes to `_diagnostics`, `--verbose`, or optional plots only.
- [ ] **MINOR (0.2.0):** New optional feature. Migration note required.
- [ ] **MAJOR (1.0.0):** Breaking change. Stakeholder review required.

## Required Artifacts

| Artifact | Provided? |
|----------|-----------|
| All existing tests pass | [ ] YES |
| `pytest tests/scripts/test_run_shadow_audit.py -v` green | [ ] YES |
| MIGRATION.md updated (if MINOR/MAJOR) | [ ] YES / N/A |
| Spec review completed (if touching frozen items) | [ ] YES / N/A |

---

## Author Attestation

I, the PR author, declare:

- [ ] I have read `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` (canonical) in full
- [ ] This PR does not expand v0.1 scope
- [ ] This PR introduces no new science
- [ ] All frozen interfaces and invariants are preserved

**Author:** @_______________
**Date:** _______________

---

## Reviewer Attestation

I, the reviewer, attest:

- [ ] I have verified all checklist items above
- [ ] This PR complies with the v0.1 scope contract
- [ ] No "new science" is introduced
- [ ] Change classification is correct

**Reviewer:** @_______________
**Date:** _______________
**Disposition:** [ ] APPROVED / [ ] REJECTED

**If REJECTED, cite:**
> Rejected per Section ___ of RUN_SHADOW_AUDIT_V0_1_CONTRACT.md because: _______________
