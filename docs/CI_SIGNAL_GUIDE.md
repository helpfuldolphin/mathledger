# CI Signal Guide

**Version**: 1.0
**Last Updated**: 2025-12-18

This document explains which CI workflows produce blocking signals vs. informational signals.

---

## Workflow Signal Classification

| Workflow Name | Signal Type | Failure Meaning |
|---------------|-------------|-----------------|
| Core Loop Verification | **Informational** (continue-on-error) | Pre-existing CI environment variability (e.g., dependency setup); does not block merge |
| Critical Files Check | **Blocking** | Required files missing or malformed |
| Uplift Evaluation | **Blocking** | Uplift safety checks failed |
| Shadow Audit Gate | **Blocking** | Unit tests failed; integration tests are informational |
| Shadow Release Gate | **Blocking** | SHADOW MODE contract violations (prohibited phrases, unqualified references) |
| CODEX M Phase X | **Informational** (continue-on-error) | P3/P4 harness tests; some expected failures |
| System Law Index Check | **Blocking** | Governance document structure violations |
| ci.yml | **Inactive on master** | Triggers only on `integrate/ledger-v0.1` branch (legacy) |
| ci-updated.yml | **Inactive on master** | Triggers only on `integrate/ledger-v0.1` branch (legacy) |

---

## Signal Type Definitions

| Type | Meaning | Verifier Action |
|------|---------|-----------------|
| **Blocking** | Failure indicates a regression or violation. | Investigate before proceeding. |
| **Informational** | Failure is logged but does not block. | Note in verification report; not a stop-ship. |
| **Inactive on master** | Workflow triggers on a different branch only. | Ignore on master/tag checkouts. |

---

## How to Identify Signal Type in Workflow Files

- **Blocking**: No `continue-on-error` at job level
- **Informational**: Job-level `continue-on-error: true`
- **Inactive on master**: Workflow file triggers only on different branches (e.g., `integrate/ledger-v0.1`)

---

## What CI Does NOT Prove

Passing CI indicates that documented checks and invariants hold under the tested configurations. It does **not** prove:

- Completeness of verification
- Correctness of external tools (e.g., Lean)
- Absence of logical errors beyond tested cases
- Suitability for production deployment

CI is a regression and contract-enforcement mechanism, not a proof of correctness.

---

## For Second Maintainers

When reviewing CI status after a push:

1. **All Blocking workflows must be green** — Any red here is a stop-ship.
2. **Informational workflows may be yellow/red** — Note in your verification report but do not treat as blocking.
3. **Inactive workflows won't run on master** — These trigger only on other branches; ignore.

---

*This document reflects CI configuration as of v0.9.0-governance-p0 + PR #60.*
