# SUPERSEDED

> **This document is SUPERSEDED.**
>
> **Canonical Source:** [`RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`](./RUN_SHADOW_AUDIT_V0_1_CONTRACT.md)
>
> Do NOT implement against this spec.

---

# RUN_SHADOW_AUDIT v0.1 — Scope Control Specification (HISTORICAL)

**Document Version:** 0.1.0
**Status:** SUPERSEDED
**Superseded By:** `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` (2025-12-12)

---

## 1. Scope Control Charter

### 1.1 v0.1 Purpose

**Mission:** Provide a CLI harness that orchestrates existing shadow audit subsystems and produces a conformant evidence pack.

**v0.1 is a container, not an engine.** It wires together existing modules; it does not implement new governance logic, metrics, or decision algorithms.

### 1.2 Explicit Boundaries

| IN SCOPE | OUT OF SCOPE |
|----------|--------------|
| CLI argument parsing | New governance algorithms |
| Orchestrating `USLAShadowLogger` | New divergence heuristics |
| Orchestrating `GovernanceEvidencePack` | New metric formulas |
| Writing manifest.json | New threshold calculations |
| Writing summary.json | New pattern detectors |
| Exit code semantics (0/1/2) | New state estimators |
| JSONL output to canonical paths | New calibration procedures |
| Deterministic mode toggle | New visualization code |
| Schema version stamping | Schema version *upgrades* |
| Invoking existing validators | Writing new validators |

### 1.3 Acceptance Criteria

v0.1 is COMPLETE when:

1. **CLI executes:** `python -m scripts.run_shadow_audit --input <path> --output <dir>` produces artifacts
2. **Manifest valid:** `manifest.json` passes `evidence_bundle.schema.json` validation
3. **Summary valid:** `summary.json` contains required fields per Section 4.1
4. **Exit codes:** 0 = success, 1 = audit warnings, 2 = fatal error
5. **Determinism:** Given `--deterministic --seed N`, output SHA-256 is reproducible
6. **Tests pass:** `pytest tests/scripts/test_run_shadow_audit.py -v` green
7. **No new imports:** Only imports from existing `backend/` modules or stdlib

---

## 2. Freeze Matrix

### 2.1 Frozen Interfaces

These interfaces are LOCKED for v0.1. Any change requires v0.2+.

| Interface | Specification | Freeze Reason |
|-----------|---------------|---------------|
| **CLI Arguments** | `--input`, `--output`, `--deterministic`, `--seed`, `--verbose`, `--schema-version` | Consumer contracts |
| **Output Paths** | `{output}/manifest.json`, `{output}/summary.json`, `{output}/shadow_log.jsonl` | External tooling depends on paths |
| **Manifest Keys** | `bundle_id`, `schema_version`, `generated_at`, `artifacts`, `shadow_mode_ok` | Schema contract |
| **Summary Keys** | `total_cycles`, `divergence_rate`, `governance_aligned_rate`, `warnings`, `status` | Dashboard integration |
| **Exit Codes** | 0, 1, 2 (meanings per 1.3) | CI/CD integration |
| **Schema Version** | `"1.0.0"` in manifest | Bundle compatibility |

### 2.2 Frozen Invariants

These invariants MUST hold. Violations are blocking defects.

| ID | Invariant | Rationale |
|----|-----------|-----------|
| **FI-001** | SHADOW-only execution | Never affect real governance decisions |
| **FI-002** | Advisory-only output | Warnings/errors are informational, not blocking |
| **FI-003** | Deterministic when flagged | `--deterministic --seed N` produces identical output bytes |
| **FI-004** | No external network calls | Air-gapped execution capability required |
| **FI-005** | No database writes | Read-only against any persistence layer |
| **FI-006** | Schema version immutable | v0.1 always emits `"schema_version": "1.0.0"` |
| **FI-007** | Manifest is self-verifying | `manifest.json` includes SHA-256 of all artifacts |

### 2.3 Experimental Zones

These areas MAY change without version bump (patch-level changes acceptable):

| Zone | Allowed Changes | Constraint |
|------|-----------------|------------|
| **Diagnostic logging** | Additional `--verbose` output | Must not change summary.json |
| **Optional plots** | New `--plot` flag for visualizations | Plots are advisory, not in manifest |
| **Performance metrics** | Timing/memory stats in summary.json under `_diagnostics` | Underscore-prefixed keys only |
| **Warning messages** | Human-readable warning text | Must not change warning count semantics |
| **Internal refactoring** | Code structure changes | Must not change observable behavior |

---

## 3. Change Control Policy

### 3.1 Version Bump Requirements

| Change Type | Version Bump | Examples |
|-------------|--------------|----------|
| Bug fix (no interface change) | PATCH (0.1.0 → 0.1.1) | Fix off-by-one, fix file handle leak |
| New optional CLI flag | MINOR (0.1.x → 0.2.0) | Add `--format yaml` |
| New required CLI flag | MAJOR (0.x → 1.0) | Breaking: `--config` now required |
| Remove CLI flag | MAJOR | Breaking: `--verbose` removed |
| Add manifest key | MINOR | New optional key in manifest |
| Remove manifest key | MAJOR | Breaking schema change |
| Change exit code semantics | MAJOR | Breaking CI contracts |
| New summary.json field | MINOR (if underscore-prefixed) or MAJOR | `_timing` = minor, `status` = major |

### 3.2 Migration Note Requirements

A `MIGRATION.md` note is REQUIRED for:

- Any MINOR or MAJOR version bump
- Any change to frozen interfaces (even if "backward compatible")
- Any new dependency (even optional)
- Any change to default behavior

Migration note format:
```markdown
## v0.1.x → v0.2.0

### Breaking Changes
- None

### New Features
- `--format` flag accepts `json` (default) or `yaml`

### Migration Steps
1. No action required for existing users
2. To use YAML output: add `--format yaml`
```

### 3.3 Test Requirements

| Change Type | Required Tests |
|-------------|----------------|
| Any code change | Existing tests pass |
| New CLI flag | Unit test for flag parsing + integration test |
| New output field | Schema validation test |
| Bug fix | Regression test proving fix |
| Frozen invariant touch | Explicit invariant assertion test |

**Test file:** `tests/scripts/test_run_shadow_audit.py`

---

## 4. "No New Science" Rule

### 4.1 Definition of "New Science"

**New Science** = any change that introduces novel:
- Mathematical formulas or algorithms
- Statistical methods or heuristics
- Machine learning or estimation procedures
- Threshold values or tuning parameters
- Convergence/divergence detection logic
- Pattern recognition rules

**New Science is FORBIDDEN in v0.1.**

### 4.2 What Constitutes New Science (FORBIDDEN)

| Category | Examples | Why Forbidden |
|----------|----------|---------------|
| **New Metrics** | "Let's add a Lyapunov exponent calculation" | Requires validation, calibration |
| **New Heuristics** | "Detect regime changes using X" | Requires empirical tuning |
| **New Thresholds** | "Block if divergence > 0.3" | Requires calibration experiments |
| **New Estimators** | "Use Kalman filter for state" | Requires convergence proof |
| **New Validators** | "Check semantic consistency with X" | Requires specification |
| **New Formulas** | "Compute adjusted HSS = H * f(rho)" | Requires derivation |

### 4.3 What Constitutes "Plumbing" (ALLOWED)

| Category | Examples | Why Allowed |
|----------|----------|-------------|
| **Wiring** | "Call `USLAShadowLogger.log_cycle()`" | Uses existing interface |
| **Aggregation** | "Sum divergence_count from log entries" | Simple arithmetic on existing data |
| **Formatting** | "Write summary as JSON" | Data transformation only |
| **Validation** | "Check manifest against existing schema" | Uses existing schema |
| **Error handling** | "Catch FileNotFoundError, exit 2" | Standard exception handling |
| **Logging** | "Log cycle count to stderr" | Observability only |
| **Path manipulation** | "Join output_dir with 'manifest.json'" | Stdlib operations |

### 4.4 Decision Flowchart

```
Is this change introducing a formula, algorithm, threshold, or heuristic?
    │
    ├─YES─→ REJECT: "New Science" — requires separate spec and calibration
    │
    └─NO──→ Does this change affect observable output (manifest, summary, exit code)?
              │
              ├─YES─→ Is the affected field in the Freeze Matrix (Section 2.1)?
              │         │
              │         ├─YES─→ REJECT: Frozen interface — requires v0.2+
              │         │
              │         └─NO──→ Is it underscore-prefixed (_diagnostics)?
              │                   │
              │                   ├─YES─→ ALLOW: Experimental zone
              │                   │
              │                   └─NO──→ REJECT: Requires minor version bump
              │
              └─NO──→ ALLOW: Internal plumbing change
```

---

## 5. PR Checklist Template

Copy this checklist into every PR touching `run_shadow_audit.py`:

```markdown
## run_shadow_audit.py PR Checklist

### Scope Creep Guard

- [ ] **No New Science:** This PR does NOT introduce new formulas, algorithms, thresholds, heuristics, or estimation procedures
- [ ] **Plumbing Only:** All changes are wiring, aggregation, formatting, or error handling
- [ ] **Frozen Interfaces:** No changes to CLI args, output paths, manifest keys, summary keys, exit codes, or schema version
- [ ] **Frozen Invariants:** All invariants FI-001 through FI-007 remain satisfied

### If ANY box above is unchecked, STOP and escalate to spec review.

### Change Classification

- [ ] **PATCH:** Bug fix only, no interface changes
- [ ] **MINOR:** New optional feature, experimental zone change
- [ ] **MAJOR:** Breaking change (requires v0.2+ and migration note)

### Required Artifacts

- [ ] Tests pass: `pytest tests/scripts/test_run_shadow_audit.py -v`
- [ ] If MINOR/MAJOR: `MIGRATION.md` updated
- [ ] If new flag: Integration test added
- [ ] If touching frozen invariant: Explicit assertion test added

### Reviewer Attestation

- [ ] I have verified this PR does not expand v0.1 scope
- [ ] I have verified no "new science" is introduced
- [ ] I have verified frozen interfaces are unchanged

**Reviewer:** _______________
**Date:** _______________
```

---

## 6. Summary

| Aspect | v0.1 Stance |
|--------|-------------|
| **Purpose** | Container harness, not engine |
| **Science** | NONE — use existing modules only |
| **Interfaces** | FROZEN — changes require v0.2+ |
| **Invariants** | LOCKED — violations are blocking |
| **Experimental** | Diagnostics, plots, underscore-prefixed fields |
| **Tests** | Required for all changes |
| **Migration** | Required for MINOR/MAJOR |

**Golden Rule:** When in doubt, DON'T. If a change feels like it might be "new science" or "improving" the audit logic, it belongs in a separate module with its own spec and calibration plan — not in v0.1 of the container harness.

---

## 7. Appendix: Example Rejections

| Proposed Change | Rejection Reason |
|-----------------|------------------|
| "Add adaptive threshold calculation" | New Science: formula |
| "Detect oscillation patterns in divergence" | New Science: heuristic |
| "Add `--threshold` CLI flag" | Frozen Interface + New Science |
| "Change exit code 1 to mean 'needs review'" | Frozen Interface: exit codes |
| "Add `convergence_quality` to summary.json" | New Science + non-underscore field |
| "Compute rolling mean of divergence" | New Science: statistical method |
| "Add `--config` as required argument" | Major version bump required |

---

**Document Hash:** (computed at commit time)
**DO NOT MODIFY** frozen sections without version bump and migration note.
