# Document Publication Boundary

**Status**: ACTIVE
**Scope**: All repository contributors
**Enforcement**: CI tripwire (`tests/policy/test_repo_leak_hygiene.py`)

---

## Purpose

This document defines what content is appropriate for the public repository versus what must remain in private, gitignored paths. The goal is to ensure the repository can be shared externally (with partners, auditors, or publicly) without risk of leaking sensitive business information.

---

## Publication Zones

### Zone 1: Public Repository (Tracked by Git)

Content that **CAN** be committed to the repository:

| Category | Examples | Location |
|----------|----------|----------|
| Source code | Python modules, Lean proofs, scripts | `backend/`, `scripts/`, `attestation/` |
| Technical documentation | Architecture docs, API specs, governance specs | `docs/system_law/`, `docs/api/` |
| Test suites | Unit tests, integration tests, policy tests | `tests/` |
| CI/CD configuration | GitHub workflows, Makefiles | `.github/workflows/`, `Makefile` |
| Experiment specifications | CAL-EXP designs, claim ladders, failure taxonomies | `docs/system_law/calibration/` |
| Operational runbooks | Technical procedures (non-sensitive) | `docs/repro/` |

**Language guidelines for Zone 1**:
- Technical, factual, neutral
- No financial figures, valuations, or projections
- No negotiation terminology
- No references to specific business partners by name (unless public)
- No internal strategy or positioning language

---

### Zone 2: Private (Gitignored)

Content that **MUST NOT** be committed:

| Category | Examples | Location (gitignored) |
|----------|----------|----------------------|
| Founder execution plans | Daily/weekly execution schedules | `_founder_notes/execution/` |
| Negotiation materials | Term sheets, position summaries, walk-away conditions | `_founder_notes/negotiation/` |
| Internal reviews | Business assessments, partner evaluations | `docs/internal/reviews/` |
| People/HR notes | Candidate assessments, compensation discussions | `_founder_notes/people/` |
| Audit tracking | Vendor communications, engagement details | `_founder_notes/audit/` |
| Financial projections | Valuations, discount models, pricing | `_founder_notes/financial/` |

**These directories are gitignored**:
```
_founder_notes/
_private/
_negotiation/
_internal_strategy/
docs/internal/
```

---

## Banned Terms in Zone 1

The following terms indicate content that belongs in Zone 2:

| Category | Banned Terms |
|----------|--------------|
| Financial | `valuation`, `$` + numbers, `million`, `earnout`, `strike price` |
| Negotiation | `term sheet`, `exclusivity`, `M&A`, `acquirer`, `acquisition price` |
| Strategy | `leverage execution`, `maximize valuation`, `negotiation position` |
| Positioning | `discount removal`, `discount adjusted`, `walk-away condition` |

**Exception**: These terms may appear in:
- This document (defining the policy)
- Test files that verify the policy
- Historical/educational context with explicit disclaimers

---

## Enforcement

### Automated (CI)

The tripwire test `tests/policy/test_repo_leak_hygiene.py`:

1. Scans all git-tracked files
2. Detects banned negotiation terms
3. Fails CI if violations found in Zone 1 paths
4. Verifies `.gitignore` contains required private paths

### Manual (Before Commit)

Before committing any document, ask:

1. **Could this embarrass us if leaked?** → Zone 2
2. **Does it contain dollar amounts?** → Zone 2
3. **Does it reference specific partners/negotiations?** → Zone 2
4. **Is it purely technical/procedural?** → Zone 1
5. **Would we share this with an auditor?** → Zone 1

---

## Remediation

If the CI tripwire fails:

1. **Identify the violation** (file path + line number in test output)
2. **Move content to appropriate Zone 2 path**
3. **Replace with neutral language** if the file must stay in Zone 1
4. **Verify gitignore** includes the destination path
5. **Re-run tests** to confirm clean

### Neutral Language Substitutions

| Sensitive | Neutral |
|-----------|---------|
| "valuation of $X" | "project scope" |
| "discount removal" | "progress on objectives" |
| "maximize valuation" | "build value" |
| "acquirer" | "partner" or "stakeholder" |
| "negotiation position" | "status summary" |
| "term sheet" | "agreement" (if public) or move to Zone 2 |

---

## Directory Structure Reference

```
mathledger/
├── docs/
│   ├── system_law/           # Zone 1: Public technical governance
│   ├── api/                  # Zone 1: Public API documentation
│   ├── repro/                # Zone 1: Public reproducibility docs
│   └── internal/             # Zone 2: GITIGNORED - internal reviews
├── _founder_notes/           # Zone 2: GITIGNORED - all founder private
│   ├── execution/            # Execution plans
│   ├── negotiation/          # Negotiation materials
│   ├── audit/                # Audit tracking
│   ├── people/               # People/HR notes
│   └── financial/            # Financial projections
├── backend/                  # Zone 1: Public source code
├── tests/                    # Zone 1: Public test suites
└── .gitignore                # Enforces Zone 2 exclusions
```

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-17 | Initial policy created |

---

*This policy is enforced by CI. Violations block merge.*
