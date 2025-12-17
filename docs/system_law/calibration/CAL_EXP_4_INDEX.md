# CAL-EXP-4 Index

**Status**: OPEN (Specification Landed)
**Phase**: CAL-EXP-4 (Phase-II Measurement Integrity Stress)
**Created**: 2025-12-17
**Predecessor**: CAL-EXP-3 (CLOSED — CANONICAL v1.0)

---

## Experiment Classification

**CAL-EXP-4 is Phase-II measurement integrity stress; no capability claims.**

| Property | Value |
|----------|-------|
| Type | Measurement integrity stress test |
| Scope | Stress-test CAL-EXP-3 verifier fail-close behavior |
| Claims permitted | Measurement stability claims only |
| Claims forbidden | Capability, intelligence, generalization, uplift claims |
| Pilot involvement | **FORBIDDEN** |
| Mode | SHADOW (observational only) |

---

## Experiment Summary

| Field | Value |
|-------|-------|
| Experiment ID | CAL-EXP-4 |
| Objective | Detect temporal/variance structure mismatches that invalidate comparability |
| Primary Metric | Temporal comparability (validity condition, NOT a metric) |
| Horizon | Inherits CAL-EXP-3 (1000 cycles, 200 warm-up excluded) |
| Binding Spec | CAL_EXP_4_VARIANCE_STRESS_SPEC.md |
| Implementation Plan | CAL_EXP_4_IMPLEMENTATION_PLAN.md |

---

## Authoritative Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [CAL_EXP_4_VARIANCE_STRESS_SPEC.md](CAL_EXP_4_VARIANCE_STRESS_SPEC.md) | **BINDING** | Charter, definitions, validity conditions |
| [CAL_EXP_4_IMPLEMENTATION_PLAN.md](CAL_EXP_4_IMPLEMENTATION_PLAN.md) | **PROVISIONAL** | Execution machinery, artifact layout |
| [CAL_EXP_4_VERIFIER_PLAN.md](CAL_EXP_4_VERIFIER_PLAN.md) | **PROVISIONAL** | Verifier check specification |
| `scripts/verify_cal_exp_4_run.py` | **READY** | Verifier implementation (52 tests passing) |
| [CAL_EXP_4_READINESS.md](CAL_EXP_4_READINESS.md) | **CURRENT** | Readiness assessment and launch checklist |

---

## Authoritative Source of Truth

| Authority | Source | Notes |
|-----------|--------|-------|
| Contract authority | `CAL_EXP_4_VARIANCE_STRESS_SPEC.md` + `CAL_EXP_4_IMPLEMENTATION_PLAN.md` | Spec and plan define the contract |
| Verifier authority | `scripts/verify_cal_exp_4_run.py` | Verifier defines pass/fail; harness deviations are harness bugs |
| Index authority | This document | Single place to understand binding vs provisional |

**Interpretation rule**: If harness output does not match verifier expectations, the harness is non-conformant. The verifier defines correctness per the contract.

---

## Artifact Contract Block (BINDING)

**Source**: `CAL_EXP_4_IMPLEMENTATION_PLAN.md` Section 2

### Authoritative Run-Dir Shape

```
results/cal_exp_4/<run_id>/
├── run_config.json                          # Extended with variance_profile
├── RUN_METADATA.json                        # Final verdict with F5.x status
├── baseline/
│   ├── cycles.jsonl                         # Per-cycle delta_p values (learning OFF)
│   └── summary.json                         # Baseline arm summary
├── treatment/
│   ├── cycles.jsonl                         # Per-cycle delta_p values (learning ON)
│   └── summary.json                         # Treatment arm summary
├── analysis/
│   ├── uplift_report.json                   # Inherited from CAL-EXP-3
│   └── windowed_analysis.json               # Inherited from CAL-EXP-3
├── validity/
│   ├── toolchain_hash.txt                   # Inherited from CAL-EXP-3
│   ├── corpus_manifest.json                 # Inherited from CAL-EXP-3
│   ├── validity_checks.json                 # Inherited from CAL-EXP-3
│   ├── isolation_audit.json                 # Inherited from CAL-EXP-3
│   ├── temporal_structure_audit.json        # NEW: F5.x temporal checks
│   └── variance_profile_audit.json          # NEW: F5.x variance checks
```

### Required Files (must exist for PASS)

| File | Why Required |
|------|--------------|
| `run_config.json` | Pre-registered seed, windows, variance_profile (F4.3 protection) |
| `RUN_METADATA.json` | Final verdict with F5.x status and claim level |
| `baseline/cycles.jsonl` | Per-cycle delta_p values for control arm |
| `treatment/cycles.jsonl` | Per-cycle delta_p values for learning arm |
| `validity/toolchain_hash.txt` | Toolchain parity proof (F1.1) |
| `validity/corpus_manifest.json` | Corpus identity proof (F1.2) |
| `validity/validity_checks.json` | Aggregate validity (F1.x-F2.x) |
| `validity/isolation_audit.json` | External ingestion negative proof (F2.3) |
| `validity/temporal_structure_audit.json` | **NEW**: Temporal comparability (F5.4) |
| `validity/variance_profile_audit.json` | **NEW**: Variance comparability (F5.1-F5.3) |

### Optional Files (may exist; never required)

| File | Purpose |
|------|---------|
| `baseline/summary.json` | Arm summary statistics |
| `treatment/summary.json` | Arm summary statistics |
| `analysis/uplift_report.json` | Computed delta-delta-p (inherited) |
| `analysis/windowed_analysis.json` | Per-window breakdown (inherited) |

### Untracked Outputs Policy

`results/cal_exp_4/**` remains **untracked** in git. Rationale:
- Run outputs are ephemeral experiment data
- Only code, docs, and schemas are committed
- Results are archived separately for reproducibility audits

### Determinism Policy

| Field | Rule |
|-------|------|
| `timestamp` fields | Allowed (execution/completion time) |
| `generated_at` in audit files | Allowed (ISO8601 timestamp) |
| JSON output | MUST use `sort_keys=True` for deterministic ordering |
| UUIDs in data lines | FORBIDDEN (verifier rejects) |

### No Pilot Involvement Rule

**BINDING**: CAL-EXP-4 MUST NOT:
- Reference any pilot data paths
- Ingest external signals
- Use pilot configuration files
- Include pilot metrics in reporting

Any pilot reference invalidates the experiment.

---

## Drift Prevention Tripwires

### Forbidden Terms (enforced by review + tripwire test)

The following terms MUST NOT appear in CAL-EXP-4 artifacts as metric names or claim language:

| Forbidden Term | Reason |
|----------------|--------|
| `uplift_score` | CAL-EXP-3 already defines uplift; no renaming |
| `intelligence` | Unoperationalized; forbidden per LANGUAGE_CONSTRAINTS |
| `generalization` | Requires OOD evidence; out of scope |
| `capability` | Overreach; CAL-EXP-4 is measurement stress only |
| `improved` / `improvement` | Causal claim without mechanism evidence |
| `delta_delta_p` (as new metric) | CAL-EXP-4 inherits, does not redefine |
| `slope` / `trend` (as metric) | Temporal structure is validity, not metric |

### Forbidden Paths (enforced by review + verifier)

| Forbidden Path Pattern | Reason |
|------------------------|--------|
| `pilot/` | No pilot involvement |
| `external/` | No external ingestion |
| `experiments/pilot*` | No pilot experiments |

### Tripwire Test Location

```
tests/calibration/test_cal_exp_4_drift_guard.py
```

---

## Verifier/Workflow Alignment

**Source**: `CAL_EXP_4_VERIFIER_PLAN.md`

| Requirement | Enforcement |
|-------------|-------------|
| Verifier MUST match artifact contract | Schema validation + field checks |
| Harness deviations are harness bugs | Interpretation rule |
| Pilot references invalidate run | Verifier check |
| F5.x checks extend (not replace) F1.x-F2.x | Inheritance from CAL-EXP-3 verifier |

### Verifier Check Categories

| Category | Checks | Failure Mode |
|----------|--------|--------------|
| Artifact Presence | temporal_structure_audit, variance_profile_audit | FAIL (invalidates) |
| SHADOW Mode | experiment=CAL-EXP-4, enforcement=false | FAIL (invalidates) |
| Toolchain Parity | hash_present, baseline_treatment_match | FAIL (invalidates) |
| Window Alignment | pre_registered, arm_alignment | FAIL (invalidates) |
| Temporal Structure | cycle_count_match, monotonic, coverage | FAIL (invalidates) |
| Variance Profile | ratio_within_threshold, drift_acceptable, compatible | FAIL or CAP |

---

## Execution Gate

### Pre-Execution Requirements

1. `CAL_EXP_4_VARIANCE_STRESS_SPEC.md` merged (DONE)
2. `CAL_EXP_4_IMPLEMENTATION_PLAN.md` merged (DONE)
3. `CAL_EXP_4_VERIFIER_PLAN.md` merged (DONE)
4. Verifier script `scripts/verify_cal_exp_4_run.py` created
5. Tripwire test passing

### Claim Level Requirements (F5.x Capping)

| Condition | Maximum Claim Level | Outcome |
|-----------|---------------------|---------|
| All F1.x-F2.x and F5.x PASS | Per CAL-EXP-3 rules (L4/L5 possible) | PASS |
| F5.1-F5.3 detected (variance/autocorr/window mismatch) | **L3** | CAP |
| F5.4 (temporal audit missing) | **L2** | CAP |
| F5.5 (temporal audit inconclusive) | **L3** | CAP |
| Any F1.x-F2.x FAIL | L0 (run void) | FAIL |

### PASS vs CAP Outcomes

| Outcome | Meaning | Claim Permitted |
|---------|---------|-----------------|
| **PASS** | All checks pass, no capping | Up to L5 (per CAL-EXP-3 replication rules) |
| **CAP** | F5.x triggered, claim capped | Max L3 (delta-delta-p computed but comparability not verified) |
| **FAIL** | F1.x-F2.x violated | L0 (run void, no claims) |

---

## Change Control

Modifications to CAL-EXP-4 artifacts require:
1. Update to this index
2. Explicit rationale
3. STRATCOM approval for semantic changes

---

## Next Expected Commits

### Topologist

- [x] Create `CAL_EXP_4_VARIANCE_STRESS_SPEC.md`
- [x] Create `CAL_EXP_4_IMPLEMENTATION_PLAN.md`
- [x] Define artifact contract (run-dir shape, required files)
- [x] Define F5.x failure taxonomy and thresholds

### Claude R

- [x] Draft `CAL_EXP_4_VERIFIER_PLAN.md`
- [x] Define schemas (`temporal_structure_audit.schema.json`, `variance_profile_audit.schema.json`)
- [ ] Create `scripts/verify_cal_exp_4_run.py` (implementation)

### Claude A (this agent)

- [x] Update this index when spec/plan land
- [x] Populate Artifact Contract Block from plan
- [x] Create tripwire test `tests/calibration/test_cal_exp_4_drift_guard.py`
- [x] Add delta-delta-p/slope tripwire

---

*This index is organizational and traceability only. It does not define new metrics, claims, or pilot logic.*
