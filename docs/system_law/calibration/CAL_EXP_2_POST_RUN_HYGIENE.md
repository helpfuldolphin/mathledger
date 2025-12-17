# CAL-EXP-2: Post-Execution Hygiene Checklist

> **Purpose:** Prevent calibration experiment results from leaking into specification documents.
> **Applies To:** Any PR that references CAL-EXP-2 results or conclusions.
> **Authority:** Spec hygiene protocol — enforced by reviewers.

---

## Principle

**CAL-EXP-2 produces DATA, not SPEC.**

- Experiment results describe **what happened**
- Specifications describe **what the system is**
- These are orthogonal — results validate specs, they do not modify them

---

## Pre-Merge Checklist (10 Items)

Reviewers MUST confirm all items before merging any PR that references CAL-EXP-2:

### Section A: Frozen Doc Protection

| # | Check | Yes/No |
|---|-------|--------|
| 1 | **No edits to `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`** — The canonical contract is unchanged. | [ ] |
| 2 | **No edits to `RUN_SHADOW_AUDIT_METRICS_V0_1.md`** — Metric definitions are unchanged. | [ ] |
| 3 | **No edits to `Metric_Versioning_Policy_v0.1.md`** — Anti-laundering rules are unchanged. | [ ] |
| 4 | **No edits to any SUPERSEDED doc** — Historical docs remain untouched. | [ ] |

### Section B: Non-Normative Labeling

| # | Check | Yes/No |
|---|-------|--------|
| 5 | **All CAL-EXP-2 references are marked NON-NORMATIVE** — Any doc mentioning CAL-EXP-2 results includes a `> **NON-NORMATIVE**` or `> **Observational**` banner. | [ ] |
| 6 | **No new requirements derived from results** — CAL-EXP-2 data does not create new "MUST" or "SHALL" statements in any spec. | [ ] |
| 7 | **Results are in `results/` or `audits/` directories** — Raw data and conclusions are NOT placed in spec directories without NON-NORMATIVE markers. | [ ] |

### Section C: Scope Containment

| # | Check | Yes/No |
|---|-------|--------|
| 8 | **No new CLI flags added** — CAL-EXP-2 used existing `--input`, `--output`, `--seed` only. | [ ] |
| 9 | **No new exit codes introduced** — Exit semantics remain 0/1/2 per canonical contract. | [ ] |
| 10 | **No new required artifacts** — `run_summary.json` and `first_light_status.json` remain the only required outputs. | [ ] |

---

## Reviewer Attestation

```
I have verified all 10 items above. This PR does not introduce spec drift from CAL-EXP-2.

Reviewer: _______________
Date: _______________
PR #: _______________
```

---

## Violation Response

If any item is unchecked:

1. **BLOCK the PR** — Do not merge
2. **Identify the violation** — Which frozen doc was touched? Which reference lacks NON-NORMATIVE?
3. **Remediate** — Remove spec changes OR escalate to version bump (v0.2+)
4. **Re-review** — All 10 items must pass before merge

---

## Allowed vs Forbidden Changes

### ALLOWED (Data Recording)

| Change | Location | Example |
|--------|----------|---------|
| Add results file | `results/cal_exp_2/` | `cal_exp_2_divergence_log.jsonl` |
| Update results doc | `docs/system_law/calibration/audits/` | Fill in `CAL_EXP_2_RESULTS_TEMPLATE.md` |
| Add observational note | Demo/test docs | `> CAL-EXP-2 confirmed this on 2025-12-XX` |

### FORBIDDEN (Spec Drift)

| Change | Location | Why Forbidden |
|--------|----------|---------------|
| Add new metric | `RUN_SHADOW_AUDIT_METRICS_V0_1.md` | Metrics are frozen |
| Change exit code meaning | `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | Contract is canonical |
| Add required artifact | Any spec doc | Artifact list is frozen |
| Remove NON-NORMATIVE banner | Any doc with CAL-EXP-2 ref | Results must stay non-normative |

---

## Quick Reference: Frozen Docs

Do NOT edit these for CAL-EXP-2:

```
docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md
docs/system_law/calibration/RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md
docs/system_law/calibration/RUN_SHADOW_AUDIT_METRICS_V0_1.md
docs/system_law/calibration/METRICS_AUDIT_TRUST_BINDING_V0_1.md
docs/system_law/Metric_Versioning_Policy_v0.1.md
docs/system_law/Evidence_Pack_Spec_PhaseX.md
docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_SPEC.md (SUPERSEDED)
docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md (MERGED)
docs/engineering/SHADOW_AUDIT_UX_SPEC_v0.1.md (SUPERSEDED)
```

---

**END OF HYGIENE CHECKLIST**
