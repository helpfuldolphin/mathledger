# Pilot Evaluation Checklist

**Document Type:** External Reviewer Checklist
**Scope:** SHADOW-OBSERVE Pilot Audit
**Version:** 1.2
**Date:** 2025-12-22

---

## Instructions

This checklist defines binary PASS/FAIL criteria for external reviewers. Each item must be independently verifiable. No interpretive or qualitative judgments are required.

**Scope Boundary:** PASS means cryptographic binding + schema compliance only; it does not evaluate the substantive adequacy, correctness, or appropriateness of governance commitments.

**GCR Commitments:** The governance commitments in `commitment_registry.json` are illustrative placeholders in v0.9.x. This audit validates hash binding and replay integrity, not the normative correctness of commitment content.

**Evaluation command:**
```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
```

---

## Section A: Execution

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| A1 | Demo script exists | `scripts/run_dropin_demo.py` is present | [ ] PASS / [ ] FAIL |
| A2 | Demo executes | Exit code is `0` | [ ] PASS / [ ] FAIL |
| A3 | No runtime errors | No Python exceptions or tracebacks | [ ] PASS / [ ] FAIL |

---

## Section B: Artifact Presence

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| B1 | manifest.json exists | File present in output directory | [ ] PASS / [ ] FAIL |
| B2 | reasoning_root.txt exists | File present in output directory | [ ] PASS / [ ] FAIL |
| B3 | ui_root.txt exists | File present in output directory | [ ] PASS / [ ] FAIL |
| B4 | epoch_root.txt exists | File present in output directory | [ ] PASS / [ ] FAIL |
| B5 | verify.py exists | File present in output directory | [ ] PASS / [ ] FAIL |
| B6 | events/ directory exists | Directory present with .jsonl files | [ ] PASS / [ ] FAIL |
| B7 | governance/ directory exists | Directory present with commitment_registry.json | [ ] PASS / [ ] FAIL |

---

## Section C: Schema Validity

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| C1 | manifest.json parses | Valid JSON; no parse errors | [ ] PASS / [ ] FAIL |
| C2 | Required fields present | `seed`, `attestation`, `governance` keys exist | [ ] PASS / [ ] FAIL |
| C3 | Attestation structure | `reasoning_merkle_root`, `ui_merkle_root`, `composite_attestation_root` present | [ ] PASS / [ ] FAIL |
| C4 | Governance structure | `claim_level`, `f5_codes`, `passed` present | [ ] PASS / [ ] FAIL |
| C5 | Governance registry present | `governance_registry.commitment_registry_sha256` present (v1.1.0+) | [ ] PASS / [ ] FAIL |
| C6 | Artifacts block present | `artifacts` array with `artifact_kind` per entry (v1.1.0+) | [ ] PASS / [ ] FAIL |

---

## Section D: Hash Verification

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| D1 | verify.py executes | Exit code is `0` | [ ] PASS / [ ] FAIL |
| D2 | Composite root verifies | Output contains `[PASS] Composite root verified` | [ ] PASS / [ ] FAIL |
| D3 | Equation holds | `H_t == SHA256(R_t || U_t)` verified by verify.py | [ ] PASS / [ ] FAIL |
| D4 | Registry hash verifies | Output contains `[PASS] Governance registry verified` (v1.1.0+) | [ ] PASS / [ ] FAIL |
| D5 | Artifact kinds valid | Output contains `[PASS] Artifact kinds verified` (v1.1.0+) | [ ] PASS / [ ] FAIL |

---

## Section E: Determinism

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| E1 | Two runs executed | Demo run twice with same seed to separate directories | [ ] PASS / [ ] FAIL |
| E2 | manifest.json identical | `sha256sum` produces same hash for both | [ ] PASS / [ ] FAIL |
| E3 | epoch_root.txt identical | `sha256sum` produces same hash for both | [ ] PASS / [ ] FAIL |

---

## Section F: SHADOW Mode Preservation

| # | Check | PASS Criteria | Result |
|---|-------|---------------|--------|
| F1 | Mode declared | manifest.json contains `"demo_mode": "SHADOW"` | [ ] PASS / [ ] FAIL |
| F2 | No enforcement | Demo does not block, gate, or modify external state | [ ] PASS / [ ] FAIL |
| F3 | Observational only | All outputs are read-only artifacts | [ ] PASS / [ ] FAIL |

---

## Summary

| Section | Items | Passed | Failed |
|---------|-------|--------|--------|
| A: Execution | 3 | ___ | ___ |
| B: Artifact Presence | 7 | ___ | ___ |
| C: Schema Validity | 6 | ___ | ___ |
| D: Hash Verification | 5 | ___ | ___ |
| E: Determinism | 3 | ___ | ___ |
| F: SHADOW Mode | 3 | ___ | ___ |
| **Total** | **27** | ___ | ___ |

---

## Evaluation Outcome

- [ ] **PASS** - All 27 items passed
- [ ] **FAIL** - One or more items failed (list below)

**Failed items:** _______________________________________________

---

## Evaluator Signature

```
Name:       ____________________
Date:       ____________________
Signature:  ____________________
```

---

---

## Notes on v1.1.0 Fields

The following fields were added in schema v1.1.0:

- `governance_registry.commitment_registry_sha256`: Binds run to governance commitments
- `governance_registry.commitment_registry_version`: Registry schema version
- `artifacts[].artifact_kind`: Per-artifact classification (VERIFIED/REFUTED/ABSTAINED/INADMISSIBLE_UPDATE)

**Note:** In v0.9.x, the governance commitments in `commitment_registry.json` are placeholder/illustrative. The mechanism (hash binding) is being audited, not the specific commitment content.

---

*This checklist evaluates artifact integrity only. It does not validate correctness, safety, or compliance. See PILOT_NON_CLAIMS.md for binding constraints.*
