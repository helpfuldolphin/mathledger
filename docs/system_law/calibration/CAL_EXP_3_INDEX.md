# CAL-EXP-3 Index

**Status**: CLOSED — CANONICAL (v1.0)
**Phase**: CAL-EXP-3 (Learning Uplift Measurement)
**Created**: 2025-12-13
**Canonized**: 2025-12-14
**Predecessor**: CAL-EXP-2 (CLOSED)

---

## Experiment Summary

| Field | Value |
|-------|-------|
| Experiment ID | CAL-EXP-3 |
| Objective | Measure delta-delta-p (uplift) with learning enabled vs. disabled |
| Primary Metric | delta_p_success (from METRIC_DEFINITIONS.md@v1.1.0) |
| Horizon | 1000 cycles (200 warm-up excluded) |
| Binding Spec | CAL_EXP_3_UPLIFT_SPEC.md |
| Implementation Plan | CAL_EXP_3_IMPLEMENTATION_PLAN.md |

---

## Authoritative Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [CAL_EXP_3_UPLIFT_SPEC.md](CAL_EXP_3_UPLIFT_SPEC.md) | **CANONICAL (v1.0)** | Charter, definitions, validity conditions |
| [CAL_EXP_3_IMPLEMENTATION_PLAN.md](CAL_EXP_3_IMPLEMENTATION_PLAN.md) | **CANONICAL (v1.0)** | Execution machinery, artifact layout |
| [CAL_EXP_3_RATIFICATION_BRIEF.md](CAL_EXP_3_RATIFICATION_BRIEF.md) | **APPROVED** | Canonization evidence and decision record |
| [CAL_EXP_3_AUTHORIZATION.md](CAL_EXP_3_AUTHORIZATION.md) | BINDING | Execution authorization gate |
| [CAL_EXP_3_LANGUAGE_CONSTRAINTS.md](CAL_EXP_3_LANGUAGE_CONSTRAINTS.md) | BINDING | Claim language constraints |

---

## Authoritative Source of Truth

| Authority | Source | Sections |
|-----------|--------|----------|
| Contract authority | `CAL_EXP_3_IMPLEMENTATION_PLAN.md` | Section 4.1 (artifact layout), Section 4.3 (determinism rules), Section 7.1.1 (isolation audit) |
| Verifier authority | `scripts/verify_cal_exp_3_run.py` | MUST match the contract; any deviations are HARNESS bugs, not verifier bugs |

**Interpretation rule**: If harness output does not match verifier expectations, the harness is non-conformant. The verifier defines correctness per the contract.

---

## Artifact Contract Block (BINDING)

**Source**: `CAL_EXP_3_IMPLEMENTATION_PLAN.md` Section 4.1
**Verifier**: `scripts/verify_cal_exp_3_run.py`

### Authoritative Run-Dir Shape

```
results/cal_exp_3/<run_id>/
├── run_config.json
├── RUN_METADATA.json
├── baseline/
│   ├── cycles.jsonl
│   └── summary.json
├── treatment/
│   ├── cycles.jsonl
│   └── summary.json
├── analysis/
│   ├── uplift_report.json
│   └── windowed_analysis.json
├── validity/
│   ├── toolchain_hash.txt
│   ├── corpus_manifest.json
│   ├── validity_checks.json
│   └── isolation_audit.json
└── (optional artifacts below)
```

### Required Files (must exist for PASS)

| File | Why Required |
|------|--------------|
| `run_config.json` | Pre-registered seed, windows, arm configs (F4.3 protection) |
| `RUN_METADATA.json` | Final verdict and claim level assignment |
| `baseline/cycles.jsonl` | Per-cycle delta_p values for control arm (NaN/missing invalidates) |
| `treatment/cycles.jsonl` | Per-cycle delta_p values for learning arm (exact alignment required) |
| `validity/toolchain_hash.txt` | Toolchain parity proof (F1.1 detection) |
| `validity/corpus_manifest.json` | Corpus identity proof (F1.2 detection) |
| `validity/validity_checks.json` | Aggregate validity condition pass/fail |
| `validity/isolation_audit.json` | Negative proof for external ingestion (F2.3) |

### Optional Files (may exist; never required)

| File | Purpose |
|------|---------|
| `baseline/summary.json` | Arm summary statistics |
| `treatment/summary.json` | Arm summary statistics |
| `analysis/uplift_report.json` | Computed uplift with standard error |
| `analysis/windowed_analysis.json` | Per-window breakdown |
| `summary.md` | Human-readable summary |
| `cal_exp_3_plots.png` | Visualization (matplotlib optional) |
| `cal_exp_3_verification_report.json` | Verifier output |

### Untracked Outputs Policy

`results/cal_exp_3/**` remains **untracked** in git. Rationale:
- Run outputs are ephemeral experiment data
- Only code, docs, and schemas are committed
- Results are archived separately for reproducibility audits

### Determinism Policy

| Field | Rule |
|-------|------|
| `timestamp` in `run_config.json` | Allowed (execution time) |
| `timestamp` in `RUN_METADATA.json` | Allowed (completion time) |
| `timestamp` in `cycles.jsonl` | Allowed (per-cycle, auxiliary) |
| `generated_at` in reports | Allowed (generation time) |
| JSON output | MUST use `sort_keys=True` for deterministic ordering |
| UUIDs in cycle lines | FORBIDDEN (verifier rejects) |
| `id`, `uuid`, `run_id` fields in cycle lines | FORBIDDEN |

---

## Verifier/Workflow Alignment

**BINDING CONSTRAINT**: All verifiers, CI workflows, and harness scripts MUST conform to the Artifact Contract above.

| Requirement | Enforcement |
|-------------|-------------|
| Verifiers MUST read from required file paths | `verify_cal_exp_3_run.py` checks file_exists |
| Harness MUST produce `cycles.jsonl` (not `delta_p_trace.jsonl`) | Harness patch required |
| Harness MUST produce root-level `run_config.json` | Harness patch required |
| Harness MUST produce `validity/` directory with all required files | Harness patch required |
| JSON output MUST use `sort_keys=True` | Schema + verifier enforcement |

### Mismatch Checklist (RESOLVED)

**Owner key**: HARNESS = `scripts/run_cal_exp_3_canonical.py` (canonical), VERIFIER = `scripts/verify_cal_exp_3_run.py`

| # | Mismatch | Owner | Status | Resolution |
|---|----------|-------|--------|------------|
| 1 | `delta_p_trace.jsonl` should be `cycles.jsonl` | HARNESS | **RESOLVED** | Canonical producer uses `cycles.jsonl` |
| 2 | `baseline/run_config.json` should be root `run_config.json` | HARNESS | **RESOLVED** | Canonical producer produces root config |
| 3 | Missing `RUN_METADATA.json` | HARNESS | **RESOLVED** | Canonical producer includes claim level |
| 4 | Missing `validity/toolchain_hash.txt` | HARNESS | **RESOLVED** | Canonical producer records toolchain hash |
| 5 | Missing `validity/corpus_manifest.json` | HARNESS | **RESOLVED** | Canonical producer records corpus hash |
| 6 | Missing `validity/validity_checks.json` | HARNESS | **RESOLVED** | Canonical producer runs validity aggregator |
| 7 | Missing `validity/isolation_audit.json` | HARNESS | **RESOLVED** | Canonical producer includes isolation audit |
| 8 | Missing `analysis/uplift_report.json` | HARNESS | **RESOLVED** | Canonical producer computes uplift |
| 9 | JSON not using `sort_keys=True` | HARNESS | **RESOLVED** | Canonical producer uses deterministic JSON |

**Note**: The original harness (`scripts/run_cal_exp_3.py`) is superseded. Use `scripts/run_cal_exp_3_canonical.py` for all future runs.

---

## Dependency Table

| Artifact Filename | Generating Component | Validation | Why |
|-------------------|---------------------|------------|-----|
| `run_config.json` | Harness orchestrator | `verify_cal_exp_3_run.py` | Pre-registration of seed/windows |
| `RUN_METADATA.json` | Claim level assigner | `verify_cal_exp_3_run.py` | Final verdict binding |
| `baseline/cycles.jsonl` | Baseline arm executor | Verifier: NaN/duplicate check | Control arm measurements |
| `treatment/cycles.jsonl` | Treatment arm executor | Verifier: alignment check | Learning arm measurements |
| `validity/toolchain_hash.txt` | Pre-execution recorder | Verifier: hash length | F1.1 parity proof |
| `validity/corpus_manifest.json` | Corpus generator | Verifier: hash present | F1.2 identity proof |
| `validity/validity_checks.json` | Validity checker | Verifier: all_passed | Aggregate validity |
| `validity/isolation_audit.json` | Isolation auditor | Verifier: isolation_passed | F2.3 negative proof |
| `analysis/uplift_report.json` | Uplift analyzer | (optional) | Computed uplift |
| `analysis/windowed_analysis.json` | Windowed analyzer | (optional) | Per-window breakdown |

---

## Canonical Components

| Component | Path | Status |
|-----------|------|--------|
| Canonical producer | `scripts/run_cal_exp_3_canonical.py` | **CANONICAL** |
| Verifier | `scripts/verify_cal_exp_3_run.py` | **CANONICAL** |
| Legacy producer | `scripts/run_cal_exp_3.py` | SUPERSEDED |

---

## Canonization Record

### L5 Achieved

| Run ID | Seed | ΔΔp | Validity | Claim Level |
|--------|------|-----|----------|-------------|
| `cal_exp_3_seed42_20251214_044612` | 42 | +0.0321 | PASS | L4 |
| `cal_exp_3_seed43_20251214_044619` | 43 | +0.0422 | PASS | L4 |
| `cal_exp_3_seed44_20251214_051658` | 44 | +0.0312 | PASS | L4 |

**Collective claim level**: L5 (Uplift Replicated)

### What Is Canon

| Element | Document | Status |
|---------|----------|--------|
| Uplift definition (ΔΔp) | `CAL_EXP_3_UPLIFT_SPEC.md` | CANONICAL |
| Claim strength ladder (L0-L5) | `CAL_EXP_3_UPLIFT_SPEC.md` | CANONICAL |
| Failure taxonomy (F1.x-F4.x) | `CAL_EXP_3_UPLIFT_SPEC.md` | CANONICAL |
| Artifact contract | `CAL_EXP_3_IMPLEMENTATION_PLAN.md` | CANONICAL |
| Verifier checks | `verify_cal_exp_3_run.py` | CANONICAL |

### What Is NOT Canon

| Element | Reason |
|---------|--------|
| Specific ΔΔp numeric values | Seed-dependent |
| Toolchain fingerprint hash | Environment-specific |
| Run IDs | Session-specific |

---

## Change Control

Modifications to CAL-EXP-3 artifacts require:
1. Update to this index
2. Explicit rationale
3. STRATCOM approval for semantic changes

---

*This index is organizational and traceability only. It does not define new metrics, claims, or pilot logic.*
