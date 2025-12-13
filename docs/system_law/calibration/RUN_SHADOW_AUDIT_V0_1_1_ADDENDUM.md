# MERGED INTO CANONICAL CONTRACT

> **This addendum has been MERGED into the canonical contract.**
>
> **Canonical Source:** [`RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`](./RUN_SHADOW_AUDIT_V0_1_CONTRACT.md)
>
> All decisions below are now in Section 2-7 of the canonical contract.

---

# RUN_SHADOW_AUDIT v0.1.1 — Addendum (HISTORICAL)

**Document Version:** 0.1.1-ADDENDUM
**Status:** MERGED
**Merged Into:** `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` (2025-12-12)

---

## Historical Note

This addendum resolved divergences between agents P, Q, R, S, T, V. Its decisions are now frozen in the canonical contract.

---

## 2. Final Rulings

### 2.1 CLI Shape: FLAGS ONLY (CONFIRMED FINAL)

**RULING:** The CLI uses flags only. No subcommands.

```
# CORRECT (v0.1 compliant)
python scripts/run_shadow_audit.py --input <dir> --output <dir> --seed 42

# REJECTED (T's subcommand proposal)
python scripts/run_shadow_audit.py audit --input <dir>
python scripts/run_shadow_audit.py status --from <run_id>
```

**Rationale:**
- Simpler shell integration
- No argparse subparser complexity
- CI scripts don't need conditional parsing
- Single entry point = single responsibility

**Frozen Flags (v0.1):**

| Flag | Required | Type | Purpose |
|------|----------|------|---------|
| `--input` | YES | path | Input shadow log directory |
| `--output` | YES | path | Output evidence bundle root |
| `--seed` | NO | int | Deterministic run_id generation |
| `--verbose` | NO | flag | Extended stderr logging |
| `--dry-run` | NO | flag | Validate without executing |

---

### 2.2 Exit Codes: REVISED FINAL MAPPING

**RULING:** Exit codes are revised to align with Unix conventions and CI tooling.

| Code | Name | Semantics | When |
|------|------|-----------|------|
| **0** | SUCCESS | Script completed (success OR warn) | Normal completion, even with warnings |
| **1** | FATAL | Script failed fatally | Missing input, unreadable file, exception |
| **2** | RESERVED | Unused in v0.1 | Reserved for future use (e.g., user interrupt) |

**Divergence Resolution:**

| Prior Spec | Said | v0.1.1 Ruling |
|------------|------|---------------|
| CLAUDE P (original) | 0=success, 1=warn, 2=fatal | **OVERRIDDEN** |
| CLAUDE R | 0=success, 1=warn, 2=fatal | **OVERRIDDEN** |
| CLAUDE Q | 0=success/warn, 2=fatal | **PARTIALLY ADOPTED** (exit 1 for fatal) |

**Rationale:**
- Exit 0 = "script ran to completion" (CI green)
- Exit 1 = "script failed to run" (CI red)
- Warnings are logged in `run_summary.json`, not in exit code
- Simplifies CI: `if exit != 0 then fail`

**Code Pattern:**
```python
def main() -> int:
    try:
        result = run_shadow_audit(args)
        # Warnings go to summary.json, not exit code
        return 0  # Always 0 if we complete
    except FatalError as e:
        log_error(e)
        return 1
    except Exception as e:
        log_crash(e)
        return 1
```

---

### 2.3 Required Artifacts: SIMPLIFIED FINAL LIST

**RULING:** Only TWO files are strictly required. All others are optional.

| File | Required | Purpose |
|------|----------|---------|
| `run_summary.json` | **YES** | Top-level summary with status, warnings, metrics |
| `first_light_status.json` | **YES** | Signal for first_light_status integration |
| `manifest.json` | NO | Optional evidence chain (write if crypto needed) |
| `shadow_log.jsonl` | NO | Optional detailed log (write if input available) |
| `reconciliation.json` | NO | Optional metric reconciliation |
| `_diagnostics.json` | NO | Experimental (verbose mode only) |

**Divergence Resolution:**

| Prior Spec | Required Files | v0.1.1 Ruling |
|------------|----------------|---------------|
| CLAUDE P (original) | 4 (manifest, summary, log, reconciliation) | **SIMPLIFIED** |
| CLAUDE Q | 4 (manifest, summary, log, reconciliation) | **SIMPLIFIED** |

**Rationale:**
- Minimal viable output = 2 files
- CI needs `run_summary.json` for pass/fail
- Dashboards need `first_light_status.json` for signals
- Other files are valuable but not blocking

**Directory Layout (Revised):**

```
{OUTPUT}/
└── {run_id}/
    ├── run_summary.json           # REQUIRED — status, warnings, metrics
    ├── first_light_status.json    # REQUIRED — signal for dashboard
    ├── manifest.json              # OPTIONAL — crypto chain (if needed)
    ├── shadow_log.jsonl           # OPTIONAL — detailed audit trail
    ├── reconciliation.json        # OPTIONAL — metric consistency
    └── _diagnostics.json          # EXPERIMENTAL — verbose only
```

---

### 2.4 `run_summary.json` Schema (FROZEN)

```json
{
  "schema_version": "1.0.0",
  "mode": "SHADOW",
  "run_id": "<string>",
  "generated_at": "<ISO8601>",
  "status": "OK|WARN",
  "exit_code": 0,

  "metrics": {
    "total_cycles": "<int>",
    "divergence_count": "<int>",
    "divergence_rate": "<float|null>"
  },

  "warnings": ["<string>"],
  "warning_count": "<int>",

  "shadow_mode_compliance": {
    "observational_only": true,
    "no_enforcement": true
  },

  "artifacts_written": ["<filename>"]
}
```

---

### 2.5 `first_light_status.json` Schema (FROZEN)

```json
{
  "schema_version": "1.0.0",
  "mode": "SHADOW",
  "run_id": "<string>",
  "generated_at": "<ISO8601>",
  "light": "GREEN|YELLOW|RED",

  "signals": {
    "shadow_audit": {
      "status": "OK|WARN",
      "divergence_rate": "<float|null>",
      "advisory_only": true
    }
  },

  "experimental": {}
}
```

---

## 3. Supersedes Notice

### 3.1 Documents Hereby Deprecated

The following documents are **SUPERSEDED** by `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` and this addendum:

| Document | Location | Status |
|----------|----------|--------|
| `RUN_SHADOW_AUDIT_V0_1_SPEC.md` | `docs/system_law/calibration/` | **DEPRECATED** |
| `SHADOW_AUDIT_ARTIFACT_CONTRACT_V0_1.md` | *(deleted)* | **DELETED** — contained invalid keys, see git history |
| CLAUDE Q artifact schema drafts | (inline in prior conversations) | **SUPERSEDED** |
| CLAUDE T subcommand proposal | (inline in prior conversations) | **REJECTED** |
| CLAUDE S skeleton drafts | (inline in prior conversations) | **SUPERSEDED** |

### 3.2 Canonical Document Chain

**Authoritative for v0.1.x:**

1. `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` — Base contract
2. `RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md` — This addendum (takes precedence on conflicts)

**Reading Order:** Read base contract first, then apply addendum overrides.

### 3.3 Deprecation Action

Deprecated documents SHOULD be:
- Marked with header: `# DEPRECATED — See RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
- NOT deleted (preserve history)
- NOT referenced in new code or specs

---

## 4. Merge Checklist

### PRE-MERGE REQUIREMENTS (All Must Be True)

Before any `run_shadow_audit.py` code lands in `master`:

- [ ] **MC-01: CLI Compliance** — Script accepts exactly `--input`, `--output`, `--seed`, `--verbose`, `--dry-run` (no subcommands, no extra flags)

- [ ] **MC-02: Exit Code Compliance** — Script exits 0 on completion (even with warnings), exits 1 on fatal error only

- [ ] **MC-03: Required Artifacts** — Script writes `run_summary.json` AND `first_light_status.json` to `{output}/{run_id}/`

- [ ] **MC-04: SHADOW Mode Markers** — All JSON outputs contain `"mode": "SHADOW"` at top level

- [ ] **MC-05: Schema Version** — All JSON outputs contain `"schema_version": "1.0.0"`

- [ ] **MC-06: No Enforcement** — No code path modifies governance decisions; `shadow_mode_compliance.no_enforcement` is always `true`

- [ ] **MC-07: Dry-Run Works** — `--dry-run` validates inputs and prints plan without creating files

- [ ] **MC-08: Determinism** — Same `--seed` produces same `run_id` prefix and reproducible output (excluding timestamps)

- [ ] **MC-09: Graceful Degradation** — Empty/missing input produces `run_summary.json` with `status: "WARN"` and exits 0

- [ ] **MC-10: Tests Pass** — `pytest tests/scripts/test_run_shadow_audit.py -v` green with coverage of all 9 items above

---

## 5. Reviewer Attestation Template

```markdown
## run_shadow_audit.py v0.1 Merge Attestation

**PR:** #___
**Reviewer:** _______________
**Date:** _______________

### Checklist Verification

- [ ] MC-01: CLI Compliance verified
- [ ] MC-02: Exit Code Compliance verified
- [ ] MC-03: Required Artifacts verified
- [ ] MC-04: SHADOW Mode Markers verified
- [ ] MC-05: Schema Version verified
- [ ] MC-06: No Enforcement verified
- [ ] MC-07: Dry-Run Works verified
- [ ] MC-08: Determinism verified
- [ ] MC-09: Graceful Degradation verified
- [ ] MC-10: Tests Pass verified

### Attestation

I attest that this PR implements `run_shadow_audit.py v0.1` in compliance with:
- `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
- `RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md`

No scope creep, no new science, no frozen interface violations.

**Signature:** _______________
```

---

## 6. Summary of v0.1.1 Changes

| Aspect | v0.1.0 (Base Contract) | v0.1.1 (This Addendum) |
|--------|------------------------|------------------------|
| Exit codes | 0/1/2 (success/warn/fatal) | **0/1/2 (success-or-warn/fatal/reserved)** |
| Required files | 4 files | **2 files** |
| File names | summary.json | **run_summary.json** |
| Status signal | embedded in summary | **separate first_light_status.json** |

---

**END OF ADDENDUM**

**Document Hash:** (computed at commit time)
**Authority:** CLAUDE P, Canonical Contract Steward
**Effective:** Immediately upon commit
