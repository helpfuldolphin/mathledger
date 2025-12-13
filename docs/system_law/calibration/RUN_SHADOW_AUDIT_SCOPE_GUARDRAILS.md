# RUN_SHADOW_AUDIT v0.1 — Scope Guardrails & Enforcement

**Document Version:** 0.1.3
**Status:** ENFORCEMENT PROTOCOL — Active
**Classification:** PR Review Gate
**Date:** 2025-12-12
**Authority:** RUN_SHADOW_AUDIT_V0_1_CONTRACT.md (canonical)
**Maintainers:** Engineering Review Board

---

## Canonical Authority

### Single Source of Truth

| Layer | Document | Path | Role |
|-------|----------|------|------|
| **1. CANONICAL** | Contract | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | Defines scope, interfaces, invariants. **AUTHORITATIVE.** |
| **2. ENFORCEMENT** | Guardrails | `docs/system_law/calibration/RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` | This document. Checklists, rejection playbook, attestations. |
| **3. MECHANISM** | PR Template | `.github/PULL_REQUEST_TEMPLATE/run_shadow_audit.md` | Enforces checklist in GitHub UI. |
| **3. MECHANISM** | CODEOWNERS | `.github/CODEOWNERS` | Forces review by `@mathledger/scope-guardians`. |
| **4. REFERENCE** | Spec (deprecated) | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_SPEC.md` | Historical reference only. Superseded by contract. |
| **4. REFERENCE** | Addendum | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md` | Supplementary clarifications. |

### Change Protocol

**Any change to `run_shadow_audit.py` or its specification MUST:**

1. **Use the designated PR template** — GitHub will prompt for `.github/PULL_REQUEST_TEMPLATE/run_shadow_audit.md`
2. **Pass CODEOWNERS review** — `@mathledger/scope-guardians` must approve
3. **Include version bump** — PATCH/MINOR/MAJOR per Change Control Matrix (Section 5)
4. **Include migration note** — Required for MINOR and MAJOR bumps
5. **Complete all attestations** — Author and reviewer must sign off
6. **Update canonical contract** — For MINOR/MAJOR changes, contract must be versioned

### Version Bump Enforcement

| Change Type | Version Bump | Migration Note | Contract Update |
|-------------|--------------|----------------|-----------------|
| Bug fix | PATCH (0.1.x → 0.1.y) | Not required | Not required |
| Experimental zone | None | Not required | Not required |
| New optional feature | MINOR (0.1.x → 0.2.0) | **REQUIRED** | **REQUIRED** |
| Breaking change | MAJOR (0.x → 1.0.0) | **REQUIRED** | **REQUIRED** |

### Document Hierarchy

```
RUN_SHADOW_AUDIT_V0_1_CONTRACT.md (CANONICAL — Layer 1)
    │
    ├── RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md (ENFORCEMENT — Layer 2)
    │
    ├── .github/PULL_REQUEST_TEMPLATE/run_shadow_audit.md (MECHANISM — Layer 3)
    │
    ├── .github/CODEOWNERS (MECHANISM — Layer 3)
    │
    └── RUN_SHADOW_AUDIT_V0_1_SPEC.md (REFERENCE — Layer 4, deprecated)
```

**Conflict Resolution:** If any document conflicts with the canonical contract, **the contract wins**.

### Branch Protection Setup Checklist

To make CODEOWNERS enforceable, complete these GitHub admin steps:

```
[ ] 1. Create GitHub Team
      Settings → Organization → Teams → New Team
      Name: scope-guardians
      Add members with contract authority

[ ] 2. Enable Branch Protection
      Settings → Branches → Add rule
      Branch pattern: master (or main)

[ ] 3. Require Pull Request Reviews
      ✓ Require a pull request before merging
      ✓ Require approvals: 1 (minimum)
      ✓ Require review from Code Owners

[ ] 4. Require Status Checks
      ✓ Require status checks to pass before merging
      ✓ Require branches to be up to date
      Select: CI workflow, test suite

[ ] 5. Restrict Push Access
      ✓ Restrict who can push to matching branches
      Add: scope-guardians team only (optional, strict mode)

[ ] 6. Verify CODEOWNERS Syntax
      GitHub → Insights → Dependency graph → CODEOWNERS
      Confirm: No syntax errors, team resolves correctly
```

**Verification:** Open test PR touching `scripts/run_shadow_audit.py`. Confirm `@mathledger/scope-guardians` appears as required reviewer.

### How This Contract Is Enforced

| Mechanism | File | What It Blocks |
|-----------|------|----------------|
| **CODEOWNERS** | `.github/CODEOWNERS:34` | PRs to `scripts/run_shadow_audit.py` require `@mathledger/scope-guardians` approval |
| **CODEOWNERS** | `.github/CODEOWNERS:37` | PRs to `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` require `@mathledger/scope-guardians` approval |
| **CI Sentinel** | `tests/ci/test_shadow_audit_sentinel.py` | Blocks if script adds forbidden CLI flags (`--p3-dir`, `--p4-dir`, `--deterministic`) |
| **CI Sentinel** | `tests/ci/test_shadow_audit_sentinel.py:82` | Blocks if `mode="SHADOW"` constant is missing |
| **CI Sentinel** | `tests/ci/test_shadow_audit_sentinel.py:78` | Blocks if `schema_version` is not `"1.0.0"` |
| **CI Sentinel** | `tests/ci/test_shadow_audit_sentinel.py:175` | Blocks if `enforcement=true` appears in source |
| **CI Workflow** | `.github/workflows/shadow-audit-gate.yml` | Runs sentinel + guardrail tests on every PR touching protected paths |
| **Branch Protection** | GitHub Settings | Merge blocked until CI passes AND CODEOWNERS approve |

**Enforcement Chain:** PR → CODEOWNERS assigns reviewer → CI runs sentinel tests → Both must pass → Merge allowed.

---

## 0. Why This Exists

### 0.1 The Stakes

MathLedger is positioned for:
- **Acquisition due diligence** — Technical buyers will audit our governance discipline
- **Defense/NDAA compliance** — Auditors require evidence of controlled, reproducible tooling
- **Frontier lab outreach** — Credibility depends on demonstrable engineering rigor

**Scope creep in shadow audit tooling is an existential credibility risk.**

A single "helpful improvement" that introduces untested science or breaks interface contracts:
- Invalidates prior evidence packs
- Triggers re-certification cycles
- Signals weak engineering governance to evaluators
- Creates audit trail discontinuities that raise red flags

### 0.2 The Discipline

This document exists to make scope creep **structurally difficult**:

| Mechanism | Purpose |
|-----------|---------|
| Frozen interfaces | Downstream consumers (CI, dashboards, auditors) depend on stability |
| No new science | Algorithms require calibration; uncalibrated science is liability |
| SHADOW-only semantics | Auditors must trust that shadow tools cannot affect production |
| Kill-switch checklist | Reviewers have explicit authority to reject without negotiation |
| PR template | Forces authors to confront constraints before writing code |

### 0.3 The Message to Acquirers

> "Our shadow audit tooling has a frozen interface contract. Changes require version bumps with migration notes. No algorithmic changes are permitted without separate calibration specs. Every PR is reviewed against an explicit scope-creep checklist. This is how we operate."

This is the story we tell. This document makes it true.

---

## 1. Reviewer Kill-Switch Checklist

**5 items. Any YES = immediate rejection. No negotiation.**

```markdown
## KILL-SWITCH REVIEW — run_shadow_audit.py

Answer each question. If ANY answer is YES, reject immediately.

| # | Question | Answer |
|---|----------|--------|
| 1 | Does this PR add, remove, or rename ANY CLI argument? | [ ] NO |
| 2 | Does this PR introduce ANY new metric, formula, or threshold? | [ ] NO |
| 3 | Does this PR cause ANY effect on real governance decisions? | [ ] NO |
| 4 | Does this PR add ANY external network or database dependency? | [ ] NO |
| 5 | Does this PR change exit code semantics or output file paths? | [ ] NO |

**If ANY box is YES:** REJECT. Do not request changes. Do not negotiate. Reject.

**Rejection template:**
> This PR is rejected per Kill-Switch item #[N] in RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md.
> The proposed change [brief description] violates the v0.1 scope contract.
> If this functionality is needed, open a spec proposal for v0.2+.
```

---

## 2. PR Template Snippet (Copy-Paste Ready)

This is the **minimal enforcement block**. Copy directly into PR description.

```markdown
## run_shadow_audit.py v0.1 — SCOPE COMPLIANCE

### Frozen CLI Check
- [ ] No new arguments added
- [ ] No arguments removed or renamed
- [ ] `--input`, `--output`, `--deterministic`, `--seed`, `--verbose`, `--schema-version` unchanged

### No New Metrics Check
- [ ] No new formulas introduced
- [ ] No new statistical methods introduced
- [ ] No new thresholds or tuning parameters introduced
- [ ] No new metric fields in summary.json (except `_diagnostics.*`)

### SHADOW-Only Semantics Check
- [ ] This code NEVER affects real governance decisions
- [ ] All outputs are advisory/observational only
- [ ] No database writes occur
- [ ] No external network calls occur

### Classification
- [ ] PATCH: Bug fix only
- [ ] EXPERIMENTAL: `_diagnostics` or `--verbose` only
- [ ] MINOR: New optional feature (migration note required)
- [ ] MAJOR: Breaking change (stakeholder review required)

### Author Attestation
I confirm this PR complies with `RUN_SHADOW_AUDIT_V0_1_SPEC.md` and introduces no scope creep.

**Author:** @_______________  **Date:** _______________
```

---

## 3. Full PR Template Checklist

Copy this ENTIRE block into every PR that touches `scripts/run_shadow_audit.py` or its tests.

```markdown
## PR SCOPE GATE — run_shadow_audit.py v0.1

> **STOP.** Before proceeding, confirm you have read `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_SPEC.md`.

### SECTION A: Frozen Interface Check

| Interface | Status | Notes |
|-----------|--------|-------|
| CLI args (`--input`, `--output`, `--deterministic`, `--seed`, `--verbose`, `--schema-version`) | [ ] UNCHANGED | |
| Output paths (`manifest.json`, `summary.json`, `shadow_log.jsonl`) | [ ] UNCHANGED | |
| Manifest keys (`bundle_id`, `schema_version`, `generated_at`, `artifacts`, `shadow_mode_ok`) | [ ] UNCHANGED | |
| Summary keys (`total_cycles`, `divergence_rate`, `governance_aligned_rate`, `warnings`, `status`) | [ ] UNCHANGED | |
| Exit codes (0=success, 1=warnings, 2=fatal) | [ ] UNCHANGED | |
| Schema version string (`"1.0.0"`) | [ ] UNCHANGED | |

**If ANY box is unchecked:** STOP. This requires v0.2+ planning. Do not merge.

### SECTION B: Frozen Invariant Check

| ID | Invariant | Holds? |
|----|-----------|--------|
| FI-001 | SHADOW-only execution (no real governance effects) | [ ] YES |
| FI-002 | Advisory-only output (no blocking decisions) | [ ] YES |
| FI-003 | Deterministic when `--deterministic --seed N` | [ ] YES |
| FI-004 | No external network calls | [ ] YES |
| FI-005 | No database writes | [ ] YES |
| FI-006 | Schema version immutable at "1.0.0" | [ ] YES |
| FI-007 | Manifest is self-verifying (SHA-256 of artifacts) | [ ] YES |

**If ANY box is unchecked:** STOP. Invariant violation is a blocking defect.

### SECTION C: No New Science Check

Answer each question. If ANY answer is YES, the PR is REJECTED.

| Question | Answer |
|----------|--------|
| Does this PR introduce a new mathematical formula? | [ ] NO |
| Does this PR introduce a new statistical method? | [ ] NO |
| Does this PR introduce a new threshold or tuning parameter? | [ ] NO |
| Does this PR introduce a new heuristic or pattern detector? | [ ] NO |
| Does this PR introduce a new estimation or prediction algorithm? | [ ] NO |
| Does this PR introduce a new metric definition? | [ ] NO |
| Does this PR introduce a new convergence/divergence criterion? | [ ] NO |

**If ANY box shows YES:** STOP. New science requires separate spec + calibration plan.

### SECTION D: Change Classification

Select ONE:

- [ ] **PATCH (0.1.x):** Bug fix only. No interface changes. No new features.
- [ ] **EXPERIMENTAL:** Changes to `_diagnostics`, `--verbose` output, or optional plots only.
- [ ] **MINOR (0.2.0):** New optional feature. Requires migration note.
- [ ] **MAJOR (1.0.0):** Breaking change. Requires migration note + stakeholder review.

### SECTION E: Required Artifacts

| Artifact | Provided? |
|----------|-----------|
| All existing tests pass | [ ] YES |
| New test for changed behavior (if applicable) | [ ] YES or N/A |
| MIGRATION.md updated (if MINOR/MAJOR) | [ ] YES or N/A |
| Spec review completed (if touching frozen items) | [ ] YES or N/A |

### SECTION F: PR Author Declaration

I, the PR author, declare:

- [ ] I have read `RUN_SHADOW_AUDIT_V0_1_SPEC.md` in full
- [ ] This PR does not expand v0.1 scope
- [ ] This PR introduces no new science
- [ ] All frozen interfaces and invariants are preserved

**Author:** _______________
**Date:** _______________

---

### REVIEWER: Do not approve until all boxes are checked.
```

---

## 4. Change Control Matrix

### 4.1 PATCH Changes (0.1.x → 0.1.y)

Allowed without spec review. Requires passing tests only.

| Change | Example | Rationale |
|--------|---------|-----------|
| Fix file handle leak | `with open(...) as f:` instead of `f = open(...)` | Bug fix, no behavior change |
| Fix off-by-one in cycle count | `range(n)` → `range(n+1)` where spec says inclusive | Bug fix aligning to spec |
| Fix typo in error message | `"Faild"` → `"Failed"` | Cosmetic, no semantic change |
| Fix exception type | `except Exception` → `except FileNotFoundError` | More specific, same behavior |
| Fix import order | Reorder imports per PEP8 | Style only |
| Fix docstring | Correct parameter description | Documentation only |
| Add type hints | `def foo(x: int) -> str:` | No runtime behavior change |
| Improve `_diagnostics` timing precision | Use `time.perf_counter()` | Experimental zone |

### 4.2 EXPERIMENTAL Changes (no version bump)

Allowed in experimental zones only. Must not affect manifest/summary/exit codes.

| Change | Example | Constraint |
|--------|---------|------------|
| Add `--verbose` detail | Print cycle-by-cycle progress | Must not change summary.json |
| Add `_diagnostics.memory_peak_mb` | Track memory usage | Underscore-prefixed only |
| Add `_diagnostics.wall_time_s` | Track execution time | Underscore-prefixed only |
| Add optional `--plot` flag | Generate divergence plot | Plot files not in manifest |
| Change log format | Use structured logging | Must not change exit codes |
| Add `--quiet` flag | Suppress stderr | summary.json unchanged |

### 4.3 MINOR Changes (0.1.x → 0.2.0)

Requires migration note. Requires spec review.

| Change | Example | Migration Required |
|--------|---------|-------------------|
| Add optional `--format yaml` | Alternative output format | Document new flag |
| Add optional `--config` | Load settings from file | Document config schema |
| Add new manifest key | `"generator_version": "0.2.0"` | Document new key |
| Add new summary key (non-underscore) | `"artifact_count": 3` | Document new key |
| Deprecate flag | `--verbose` deprecated, use `--log-level` | Deprecation notice |
| Change default (backward compat) | `--format` defaults to `json` | Document default |

### 4.4 MAJOR Changes (0.x → 1.0.0)

Requires stakeholder review. Requires migration plan. Requires announcement.

| Change | Example | Impact |
|--------|---------|--------|
| Remove CLI flag | Remove `--verbose` | Breaks existing scripts |
| Rename CLI flag | `--output` → `--out-dir` | Breaks existing scripts |
| Change exit code meaning | Exit 1 now means "fatal" | Breaks CI pipelines |
| Remove manifest key | Remove `shadow_mode_ok` | Breaks downstream parsers |
| Change manifest key type | `artifacts` from array to object | Breaks downstream parsers |
| Change schema version | `"1.0.0"` → `"2.0.0"` | Requires bundle migration |
| Make optional flag required | `--config` now required | Breaks existing invocations |
| Change summary.json structure | Nest `warnings` under `diagnostics` | Breaks dashboards |

---

## 5. Rejection Playbook

Ten common scope-creep PRs and their rejection rationale.

### 5.1 "Add adaptive divergence threshold"

```
PR: "Compute threshold as mean + 2*std of historical divergence"
```

**REJECTED:** New Science (statistical method + threshold calculation).
**Disposition:** Requires separate spec: `DIVERGENCE_THRESHOLD_CALIBRATION_SPEC.md`.

---

### 5.2 "Detect oscillation patterns"

```
PR: "Add oscillation_detected: true/false to summary based on FFT analysis"
```

**REJECTED:** New Science (signal processing algorithm + new metric).
**Disposition:** Requires separate module with calibration experiments.

---

### 5.3 "Add --threshold flag"

```
PR: "Allow user to set divergence threshold via --threshold 0.3"
```

**REJECTED:** Frozen Interface (new CLI arg) + New Science (threshold semantics).
**Disposition:** v0.2+ planning required. Threshold semantics need spec.

---

### 5.4 "Change exit code 1 meaning"

```
PR: "Exit 1 should mean 'review recommended' not 'warnings present'"
```

**REJECTED:** Frozen Interface (exit code semantics).
**Disposition:** MAJOR version bump required. CI contracts depend on current semantics.

---

### 5.5 "Add convergence_quality metric"

```
PR: "Compute convergence_quality = 1 - (final_divergence / max_divergence)"
```

**REJECTED:** New Science (new metric formula).
**Disposition:** Requires calibration spec defining metric properties and validation.

---

### 5.6 "Improve divergence calculation"

```
PR: "Use exponential moving average instead of simple mean"
```

**REJECTED:** New Science (changed algorithm).
**Disposition:** This changes observable output. Requires calibration comparison.

---

### 5.7 "Add Kalman filter for state estimation"

```
PR: "Use Kalman filter to smooth state estimates before divergence check"
```

**REJECTED:** New Science (estimation algorithm).
**Disposition:** Requires convergence proof and separate integration spec.

---

### 5.8 "Add regime_change_detected field"

```
PR: "Detect regime changes using CUSUM and report in summary"
```

**REJECTED:** New Science (statistical method + heuristic).
**Disposition:** Regime detection requires separate spec with false positive analysis.

---

### 5.9 "Make --config required"

```
PR: "All runs should use config files for reproducibility"
```

**REJECTED:** MAJOR breaking change (new required argument).
**Disposition:** Breaks all existing invocations. Requires v1.0.0 planning.

---

### 5.10 "Add real-time progress webhook"

```
PR: "POST progress updates to --webhook-url during execution"
```

**REJECTED:** Frozen Invariant FI-004 (no external network calls).
**Disposition:** Air-gapped execution is a hard requirement. Webhook would violate.

---

## 6. Reviewer Attestation Block

Reviewers MUST complete this attestation before approving.

```markdown
## REVIEWER ATTESTATION — run_shadow_audit.py

### Pre-Review Confirmation

- [ ] I have read `RUN_SHADOW_AUDIT_V0_1_SPEC.md` within the last 30 days
- [ ] I have reviewed `RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` (this document)
- [ ] I understand the distinction between "plumbing" and "new science"

### PR-Specific Review

- [ ] I have verified ALL Section A boxes (Frozen Interface) are checked UNCHANGED
- [ ] I have verified ALL Section B boxes (Frozen Invariant) are checked YES
- [ ] I have verified ALL Section C boxes (No New Science) are checked NO
- [ ] I have verified the Change Classification in Section D is correct
- [ ] I have verified Required Artifacts in Section E are provided

### Scope Creep Judgment

- [ ] This PR does NOT introduce functionality that belongs in a separate module
- [ ] This PR does NOT introduce "improvements" beyond the stated bug/feature
- [ ] This PR does NOT set precedent for future scope expansion
- [ ] This PR could be explained to an auditor as "wiring and plumbing only"

### Risk Assessment

| Risk | Assessment |
|------|------------|
| Breaks existing scripts | [ ] No risk / [ ] Low risk / [ ] STOP |
| Breaks CI pipelines | [ ] No risk / [ ] Low risk / [ ] STOP |
| Breaks downstream parsers | [ ] No risk / [ ] Low risk / [ ] STOP |
| Introduces untested behavior | [ ] No risk / [ ] Low risk / [ ] STOP |
| Opens door to future scope creep | [ ] No risk / [ ] Low risk / [ ] STOP |

**If ANY "STOP" is selected:** Do not approve. Escalate to spec review.

### Approval

I attest that this PR:
1. Complies with `RUN_SHADOW_AUDIT_V0_1_SPEC.md`
2. Introduces no new science
3. Preserves all frozen interfaces and invariants
4. Is correctly classified (PATCH/EXPERIMENTAL/MINOR/MAJOR)
5. Has appropriate tests and documentation

**Reviewer Name:** _______________
**Reviewer GitHub:** @_______________
**Date:** _______________
**Disposition:** [ ] APPROVED / [ ] REQUEST CHANGES / [ ] REJECTED

### If REJECTED, cite specific section:

> Rejected per Section ___ of RUN_SHADOW_AUDIT_V0_1_SPEC.md because: _______________
```

---

## 7. Smoke-Test Readiness Checklist

This checklist verifies documentation and PR hygiene are in place BEFORE any code is written.

### 7.1 Documentation Readiness

| Item | Status | Location |
|------|--------|----------|
| v0.1 Spec exists | [ ] YES | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_SPEC.md` |
| Scope Guardrails exist | [ ] YES | `docs/system_law/calibration/RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` |
| Spec defines CLI interface | [ ] YES | Spec Section 2.1 |
| Spec defines output schema | [ ] YES | Spec Section 2.1 |
| Spec defines exit codes | [ ] YES | Spec Section 1.3 |
| Spec defines frozen invariants | [ ] YES | Spec Section 2.2 |
| Spec defines "no new science" | [ ] YES | Spec Section 4 |
| Change Control Matrix defined | [ ] YES | Guardrails Section 4 |
| PR template defined | [ ] YES | Guardrails Section 2 |
| Rejection playbook defined | [ ] YES | Guardrails Section 5 |
| Reviewer attestation defined | [ ] YES | Guardrails Section 6 |

### 7.2 PR Hygiene Readiness

| Item | Status | Notes |
|------|--------|-------|
| PR template added to repo | [ ] PENDING | `.github/PULL_REQUEST_TEMPLATE/run_shadow_audit.md` |
| Branch protection requires checklist | [ ] PENDING | Require "APPROVED" attestation |
| CODEOWNERS includes spec maintainers | [ ] PENDING | Require review from spec owners |
| CI runs spec compliance check | [ ] PENDING | Future: automated frozen interface check |

### 7.3 Pre-Implementation Verification

Before writing `run_shadow_audit.py`:

| Question | Answer |
|----------|--------|
| Is the scope clear and bounded? | [ ] YES — container harness only |
| Are all interfaces frozen and documented? | [ ] YES — per Spec Section 2.1 |
| Is "new science" explicitly forbidden? | [ ] YES — per Spec Section 4 |
| Are acceptance criteria testable? | [ ] YES — per Spec Section 1.3 |
| Is the PR review process defined? | [ ] YES — per Guardrails Section 1, 2, 6 |
| Are rejection criteria explicit? | [ ] YES — per Guardrails Section 5 |

### 7.4 Reality Lock Acknowledgment

| Statement | Confirmed |
|-----------|-----------|
| `scripts/run_shadow_audit.py` does NOT exist yet | [ ] YES |
| `tests/scripts/test_run_shadow_audit.py` does NOT exist yet | [ ] YES |
| This document governs FUTURE implementation | [ ] YES |
| No code claims are made in this document | [ ] YES |

---

## 8. Quick Reference Card

Print this for desk reference during PR review.

```
┌─────────────────────────────────────────────────────────────────┐
│           RUN_SHADOW_AUDIT v0.1 — QUICK REFERENCE               │
├─────────────────────────────────────────────────────────────────┤
│ FROZEN CLI ARGS:                                                │
│   --input, --output, --deterministic, --seed,                   │
│   --verbose, --schema-version                                   │
├─────────────────────────────────────────────────────────────────┤
│ FROZEN OUTPUT FILES:                                            │
│   manifest.json, summary.json, shadow_log.jsonl                 │
├─────────────────────────────────────────────────────────────────┤
│ FROZEN EXIT CODES:                                              │
│   0 = success, 1 = warnings, 2 = fatal                          │
├─────────────────────────────────────────────────────────────────┤
│ FROZEN INVARIANTS:                                              │
│   FI-001: SHADOW-only    FI-005: No DB writes                   │
│   FI-002: Advisory-only  FI-006: Schema "1.0.0"                 │
│   FI-003: Deterministic  FI-007: Self-verifying                 │
│   FI-004: No network                                            │
├─────────────────────────────────────────────────────────────────┤
│ NEW SCIENCE = REJECT:                                           │
│   Formula, Algorithm, Threshold, Heuristic, Estimator, Metric   │
├─────────────────────────────────────────────────────────────────┤
│ PLUMBING = ALLOW:                                               │
│   Wiring, Aggregation, Formatting, Validation, Logging, Paths   │
├─────────────────────────────────────────────────────────────────┤
│ EXPERIMENTAL ZONE (no version bump):                            │
│   _diagnostics.*, --verbose output, optional --plot             │
├─────────────────────────────────────────────────────────────────┤
│ VERSION BUMPS:                                                  │
│   PATCH: Bug fix only                                           │
│   MINOR: New optional feature (needs migration note)            │
│   MAJOR: Breaking change (needs stakeholder review)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Verification Appendix

**Required GitHub Setting:** `Settings → Branches → [rule] → Require review from Code Owners`

**3-Step Smoke Verification:**
1. Create branch, add `echo "" >> scripts/run_shadow_audit.py`, commit, push
2. Open PR → Confirm `@mathledger/scope-guardians` appears as required reviewer
3. Confirm PR body prompts for `run_shadow_audit.md` template (or auto-populates)

**PASS:** Merge blocked until scope-guardian approves. **FAIL:** Check CODEOWNERS syntax + branch protection.

---

**Document Hash:** (computed at commit time)
**Canonical Authority:** `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
