# MathLedger Documentation Index

This index routes readers to the right document for their purpose.

---

## Start Here

| Your role | Start with |
|-----------|------------|
| **New visitor** | [HOW_THE_DEMO_EXPLAINS_ITSELF.md](HOW_THE_DEMO_EXPLAINS_ITSELF.md) |
| **External auditor** | [FOR_AUDITORS.md](FOR_AUDITORS.md) |
| **Pilot evaluator** | [pilot/AUDIT_WALKTHROUGH.md](pilot/AUDIT_WALKTHROUGH.md) |
| **Developer** | [V0_LOCK.md](V0_LOCK.md) then [invariants_status.md](invariants_status.md) |

---

## By Category

### Demo & UI

| Document | Purpose |
|----------|---------|
| [HOW_THE_DEMO_EXPLAINS_ITSELF.md](HOW_THE_DEMO_EXPLAINS_ITSELF.md) | What the demo enforces and refuses |
| [HOW_TO_APPROACH_THIS_DEMO.md](HOW_TO_APPROACH_THIS_DEMO.md) | Framing and expectations |
| [V0_LOCK.md](V0_LOCK.md) | What is and isn't in v0 scope |
| [V0_SYSTEM_BOUNDARY_MEMO.md](V0_SYSTEM_BOUNDARY_MEMO.md) | Formal claims and non-claims |
| [DEMO_SELF_EXPLANATION_UI_PLAN.md](DEMO_SELF_EXPLANATION_UI_PLAN.md) | UI integration point specification |
| [HOSTILE_DEMO_REHEARSAL.md](HOSTILE_DEMO_REHEARSAL.md) | Hostile questions with answers |

### Audit & Verification

| Document | Purpose |
|----------|---------|
| [FOR_AUDITORS.md](FOR_AUDITORS.md) | Cold-start auditor guide |
| [pilot/AUDIT_WALKTHROUGH.md](pilot/AUDIT_WALKTHROUGH.md) | Complete audit procedure |
| [pilot/PILOT_NON_CLAIMS.md](pilot/PILOT_NON_CLAIMS.md) | Binding non-claims |
| [pilot/PILOT_EVALUATION_CHECKLIST.md](pilot/PILOT_EVALUATION_CHECKLIST.md) | 22-item PASS/FAIL checklist |
| [EVIDENCE_PACK_VERIFIER_SPEC.md](EVIDENCE_PACK_VERIFIER_SPEC.md) | Evidence pack verifier specification |
| [AUDIT_TAXONOMY.md](AUDIT_TAXONOMY.md) | Audit classification system |

### External Audit Records

| Document | Purpose |
|----------|---------|
| [external_audits/README.md](external_audits/README.md) | Index of all external audit results |
| [external_audits/CLOSURE_MATRIX_v0.2.13.md](external_audits/CLOSURE_MATRIX_v0.2.13.md) | Latest closure matrix |

### Specifications

| Document | Purpose |
|----------|---------|
| [FOL_FIN_EQ_PHASE3_CLOSURE.md](FOL_FIN_EQ_PHASE3_CLOSURE.md) | FOL_FIN_EQ_v1 sealed specification |
| [FOL_FIN_EQ_PHASE3_EVIDENCE.md](FOL_FIN_EQ_PHASE3_EVIDENCE.md) | Reproduction commands |
| [invariants_status.md](invariants_status.md) | Tier A/B/C invariant classification |
| [ABSTENTION_PRESERVATION_ENFORCEMENT.md](ABSTENTION_PRESERVATION_ENFORCEMENT.md) | Abstention gate specification |
| [UVIL_V0_EXECUTION_PACKET.md](UVIL_V0_EXECUTION_PACKET.md) | UVIL execution specification |

### Releases & Deployment

**Canonical production branch: master**

| Document | Purpose |
|----------|---------|
| [DEPLOY_BY_TAG_DOCTRINE.md](DEPLOY_BY_TAG_DOCTRINE.md) | Non-negotiable deployment rules |
| [VERSION_NUMBER_DOCTRINE.md](VERSION_NUMBER_DOCTRINE.md) | Versioning conventions |
| [RELEASE_CLOSURE_V0.2.1.md](RELEASE_CLOSURE_V0.2.1.md) | v0.2.1 closure document |
| [V0.2.2_SCOPE.md](V0.2.2_SCOPE.md) | v0.2.2 scope constraints |
| [V0.2.10_SMOKE_CHECKLIST.md](V0.2.10_SMOKE_CHECKLIST.md) | v0.2.10 verification checklist |
| [HOSTED_DEMO_GO_CHECKLIST.md](HOSTED_DEMO_GO_CHECKLIST.md) | 60-second deployment verification |
| [FROZEN_VERSION_IMMUTABILITY.md](FROZEN_VERSION_IMMUTABILITY.md) | Archive immutability rules |
| [RELEASE_METADATA_CONTRACT.md](RELEASE_METADATA_CONTRACT.md) | Release metadata specification |
| [RELEASE_METADATA_DIAGNOSTIC.md](RELEASE_METADATA_DIAGNOSTIC.md) | Version mismatch troubleshooting |

### Internal Reference

| Document | Purpose |
|----------|---------|
| [INTERNAL_CHAMPION_BRIEF.md](INTERNAL_CHAMPION_BRIEF.md) | One-page technical summary |
| [DROPIN_DEMO_FREEZE.md](DROPIN_DEMO_FREEZE.md) | Freeze declaration |
| [DROPIN_REPLAY_INSTRUCTIONS.md](DROPIN_REPLAY_INSTRUCTIONS.md) | Third-party verification guide |
| [DEMO_REGRESSION_HARNESS.md](DEMO_REGRESSION_HARNESS.md) | Regression test documentation |
| [DEMO_HOSTING_V0.2.0.md](DEMO_HOSTING_V0.2.0.md) | Hosting runbook |

### Field Manual & Papers

| Document | Purpose |
|----------|---------|
| [field_manual/README.md](field_manual/README.md) | Field Manual doctrinal framing |
| [field_manual/fm.pdf](field_manual/fm.pdf) | Field Manual (compiled) |
| [PAPERS/mathledger_arxiv_preprint.pdf](PAPERS/mathledger_arxiv_preprint.pdf) | arXiv preprint |
| [PAPERS/README.md](PAPERS/README.md) | Papers directory index |

---

## Document Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `V0_*.md` | v0 scope constraints |
| `V0.2.x_*.md` | Version-specific documents |
| `*_CLOSURE.md` | Sealed specifications (do not modify) |
| `*_FREEZE.md` | Freeze declarations |
| `PILOT_*.md` | Pilot evaluation materials |

---

## Not Finding What You Need?

- **Live demo**: [mathledger.ai/demo](https://mathledger.ai/demo)
- **Source**: [github.com/helpfuldolphin/mathledger](https://github.com/helpfuldolphin/mathledger)
- **Paper**: [arXiv preprint](PAPERS/mathledger_arxiv_preprint.pdf)
