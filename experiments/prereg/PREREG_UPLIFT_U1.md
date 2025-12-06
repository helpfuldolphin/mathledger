# Preregistration: Uplift Experiment U1

> **STATUS: PREREGISTERED — NOT YET EXECUTED**
>
> This document satisfies the preregistration requirements defined in
> `docs/VSD_PHASE_2.md § Phase II Uplift Evidence Gate`.

> **CANONICAL STATUS: U1 is the canonical first uplift experiment for Phase II.**
>
> - Slice: `slice_medium` (truth-table verification, no Lean)
> - Effect size threshold: **10 percentage points** (0.10 absolute reduction)
> - Lean-enabled uplift slices (`slice_pl_uplift_a/b/c`) are deferred to Phase IIb
> - U1b (Lean-enabled) will only be designed after U1 results are analyzed

**Experiment ID:** `uplift_u1_slice_medium`
**Preregistration Date:** 2025-11-30
**Last Updated:** 2025-11-30 (10pp threshold tightening)
**Author:** Claude A — VSD Governance Architect

---

## 1. Experiment Identification

| Field | Value |
|-------|-------|
| `experiment_id` | `uplift_u1_slice_medium` |
| `slice_name` | `slice_medium` |
| `verifier_mode` | `truth-table-only` (no Lean) |
| `baseline_cycles` | 500 |
| `rfl_cycles` | 500 |
| `seed` | `20251130` (YYYYMMDD of prereg date) |

---

## 2. Hypothesis

**Primary Hypothesis:**
RFL (Reflexive Formal Learning) will reduce the abstention rate relative to baseline on the `slice_medium` curriculum slice, when using truth-table verification only.

**Directional Prediction:**
- Baseline abstention rate: expected in range 0.30–0.70 (based on slice difficulty)
- RFL abstention rate: expected to be lower than baseline

**Null Hypothesis:**
RFL has no effect on abstention rate; any observed difference is due to random variation.

---

## 3. Slice Configuration

From `config/curriculum.yaml`, `slice_medium`:

```yaml
name: slice_medium
params:
  atoms: 5
  depth_max: 7
  breadth_max: 1500
  total_max: 8000
gates:
  coverage:
    ci_lower_min: 0.85
    sample_min: 20
    require_attestation: true
  abstention:
    max_rate_pct: 15.0
    max_mass: 800
  velocity:
    min_pph: 150
    stability_cv_max: 0.12
    window_minutes: 60
  caps:
    min_attempt_mass: 3000
    min_runtime_minutes: 20
    backlog_max: 0.40
```

**Rationale for Slice Selection:**
- `slice_medium` is designated as the "Wide Slice" for uplift experiments
- Parameters (atoms=5, depth_max=7) produce non-trivial abstention without saturating
- Already validated in `config/curriculum.yaml` (lines 125-149)
- No prior experiments on this slice (clean comparison)

---

## 4. Verifier Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `verifier_mode` | `truth-table-only` | Eliminates Lean complexity for first pass |
| `lean_enabled` | `false` | Avoids Lean timeout/flakiness confounds |
| `taut_checker` | `normalization.taut.is_tautology` | Deterministic, fast |
| `timeout_seconds` | `5.0` | Per-statement timeout |

**Why Truth-Table Only:**
1. Deterministic — no external process variability
2. Fast — enables 500+ cycles without timeout issues
3. Sufficient — propositional logic completeness
4. Clean baseline — isolates RFL effect from verifier noise

---

## 5. Success Criteria

### 5.1 Validity Criteria (Must Pass)

These criteria determine if the experiment is **valid** (not degenerate):

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Baseline abstention rate | 0.10 ≤ α_baseline ≤ 0.80 | Avoids trivial (0%) or saturated (100%) |
| Baseline proof attempts | ≥ 200 per run | Statistical power |
| RFL proof attempts | ≥ 200 per run | Statistical power |
| Determinism check | Same seed → same H_t | Reproducibility |

**If validity criteria fail:** Experiment is declared **invalid** (not "failed"). Results are documented but do not constitute uplift evidence.

### 5.2 Uplift Criteria (Determines Outcome)

If validity criteria pass, uplift is assessed by:

| Criterion | Test | Threshold |
|-----------|------|-----------|
| Direction | α_rfl < α_baseline | RFL abstention lower |
| Magnitude | α_baseline - α_rfl ≥ **0.10** | At least **10 percentage points** |
| Significance | Bootstrap 95% CI for (α_baseline - α_rfl) excludes zero | Statistical confidence |

**Effect Size Definition:**
> We define **uplift** as an absolute abstention reduction of at least **0.10 (10 percentage points)**,
> with a 95% bootstrap confidence interval that excludes zero.

**Outcome Classification:**

| Validity | Uplift Criteria | Classification |
|----------|-----------------|----------------|
| ❌ Invalid | N/A | **INVALID** — Not uplift evidence |
| ✅ Valid | Direction wrong (α_rfl ≥ α_baseline) | **NULL** — No uplift detected |
| ✅ Valid | Direction correct but magnitude < 10pp | **NULL** — No uplift detected (trend only) |
| ✅ Valid | Direction correct, magnitude ≥ 10pp, CI excludes 0 | **POSITIVE** — Uplift detected |

---

## 6. Measurement Methodology

### 6.1 Abstention Rate Calculation

```
abstention_rate = abstained_statements / total_proof_attempts
```

Where:
- `abstained_statements`: Count of statements where verifier returned "abstain" or timeout
- `total_proof_attempts`: Count of all statements submitted to verifier

### 6.2 Per-Cycle Logging

Each cycle logs:
```json
{
  "cycle": <int>,
  "seed": <int>,
  "slice_name": "slice_medium",
  "proof_attempts": <int>,
  "successful_proofs": <int>,
  "abstained": <int>,
  "abstention_rate": <float>,
  "H_t": "<64-char hex>"
}
```

### 6.3 Aggregate Statistics

After all cycles:
- Mean abstention rate (baseline vs RFL)
- Standard deviation
- 95% bootstrap CI for difference
- Mann-Whitney U test p-value

---

## 7. Output Artifacts

The experiment will produce:

| Artifact | Path | Description |
|----------|------|-------------|
| Preregistration hash | `experiments/prereg/PREREG_UPLIFT_U1.sha256` | SHA-256 of this file (pre-commit) |
| Baseline log | `results/uplift_u1/baseline_log.jsonl` | 500 cycles, RFL disabled |
| RFL log | `results/uplift_u1/rfl_log.jsonl` | 500 cycles, RFL enabled |
| Baseline attestation | `results/uplift_u1/baseline_attestation.json` | Final H_t for baseline |
| RFL attestation | `results/uplift_u1/rfl_attestation.json` | Final H_t for RFL |
| Statistical summary | `results/uplift_u1/statistical_summary.json` | CI, p-values, outcome |
| Experiment manifest | `results/uplift_u1/experiment_manifest.json` | Links prereg hash to results |

---

## 8. Commitment to Publish

**Regardless of outcome, this experiment's results will be published.**

| Outcome | Action |
|---------|--------|
| INVALID | Document why validity failed; do not claim uplift |
| NULL | Document null result; do not claim uplift |
| POSITIVE | Document uplift with appropriate caveats |

**No selective reporting.** All 500+500 cycles will be included in analysis.

---

## 9. Limitations and Caveats

Even if this experiment shows positive uplift:

1. **Scope**: Results apply only to `slice_medium` with truth-table verification
2. **Generalization**: Does not prove RFL works on harder slices or with Lean
3. **Single experiment**: Replication required before strong claims
4. **Not basis promotion**: Does not automatically promote `basis/` package

---

## 10. Gate Alignment

This preregistration satisfies VSD_PHASE_2.md § Phase II Uplift Evidence Gate:

| Gate Requirement | Satisfied By |
|------------------|--------------|
| Non-degenerate slice | `slice_medium` (atoms=5, depth_max=7) |
| Minimum cycle count | 500 baseline + 500 RFL = 1000 total |
| Baseline comparison | Paired runs with same seed |
| Statistical test | Mann-Whitney + Bootstrap CI |
| Determinism verified | H_t determinism check in validity criteria |
| Preregistration | This document |
| Manifest requirements | Section 7 artifacts |

---

## 11. Execution Checklist

Before running:
- [ ] Compute SHA-256 of this file → `PREREG_UPLIFT_U1.sha256`
- [ ] Verify `slice_medium` exists in `config/curriculum.yaml`
- [ ] Verify truth-table verifier is available
- [ ] Create output directory `results/uplift_u1/`

After running:
- [ ] Verify all artifacts in Section 7 exist
- [ ] Verify determinism (re-run with same seed, compare H_t)
- [ ] Compute statistics
- [ ] Fill in `experiment_manifest.json` with prereg hash
- [ ] Document outcome (INVALID/NULL/POSITIVE)

---

## Appendix A: Output Artifact Schemas

### A.1 statistical_summary.json Schema

```json
{
  "experiment_id": "uplift_u1_slice_medium",
  "baseline": {
    "cycles": 500,
    "total_proof_attempts": "<int>",
    "total_abstained": "<int>",
    "mean_abstention_rate": "<float>",
    "std_abstention_rate": "<float>"
  },
  "rfl": {
    "cycles": 500,
    "total_proof_attempts": "<int>",
    "total_abstained": "<int>",
    "mean_abstention_rate": "<float>",
    "std_abstention_rate": "<float>"
  },
  "comparison": {
    "difference": "<float, baseline - rfl>",
    "difference_pp": "<float, percentage points>",
    "bootstrap_ci_95_lower": "<float>",
    "bootstrap_ci_95_upper": "<float>",
    "bootstrap_replicates": 10000,
    "mann_whitney_u_statistic": "<float>",
    "mann_whitney_p_value": "<float>"
  },
  "validity": {
    "baseline_abstention_in_range": "<bool, 0.10 <= rate <= 0.80>",
    "baseline_proof_attempts_sufficient": "<bool, >= 200>",
    "rfl_proof_attempts_sufficient": "<bool, >= 200>",
    "determinism_verified": "<bool>",
    "all_validity_criteria_met": "<bool>"
  },
  "uplift": {
    "direction_correct": "<bool, rfl < baseline>",
    "magnitude_sufficient": "<bool, diff >= 0.10 (10pp)>",
    "statistically_significant": "<bool, 95% CI excludes 0>",
    "all_uplift_criteria_met": "<bool>"
  },
  "outcome": "INVALID | NULL | POSITIVE",
  "outcome_reason": "<string explaining outcome>",
  "computed_at": "<ISO8601 timestamp>"
}
```

### A.2 experiment_manifest.json Schema

```json
{
  "manifest_version": "1.0",
  "experiment_id": "uplift_u1_slice_medium",
  "preregistration": {
    "prereg_path": "experiments/prereg/PREREG_UPLIFT_U1.md",
    "prereg_hash_sha256": "<64-char hex>",
    "prereg_hash_computed_at": "<ISO8601 timestamp>"
  },
  "execution": {
    "started_at": "<ISO8601 timestamp>",
    "completed_at": "<ISO8601 timestamp>",
    "executor": "<agent or human identifier>",
    "environment": {
      "python_version": "<string>",
      "platform": "<string>",
      "mathledger_commit": "<git commit hash if available>"
    }
  },
  "artifacts": {
    "baseline_log": "results/uplift_u1/baseline_log.jsonl",
    "rfl_log": "results/uplift_u1/rfl_log.jsonl",
    "baseline_attestation": "results/uplift_u1/baseline_attestation.json",
    "rfl_attestation": "results/uplift_u1/rfl_attestation.json",
    "statistical_summary": "results/uplift_u1/statistical_summary.json"
  },
  "determinism_verification": {
    "baseline_H_t": "<64-char hex>",
    "baseline_replay_H_t": "<64-char hex, from re-run>",
    "baseline_determinism_match": "<bool>",
    "rfl_H_t": "<64-char hex>",
    "rfl_replay_H_t": "<64-char hex, from re-run>",
    "rfl_determinism_match": "<bool>"
  },
  "outcome": "INVALID | NULL | POSITIVE",
  "gate_alignment": {
    "satisfies_vsd_phase_2_gate": "<bool>",
    "gate_document": "docs/VSD_PHASE_2.md",
    "gate_section": "Phase II Uplift Evidence Gate"
  }
}
```

---

## Appendix B: Preregistration Hash

To be computed before execution:

```bash
sha256sum experiments/prereg/PREREG_UPLIFT_U1.md > experiments/prereg/PREREG_UPLIFT_U1.sha256
```

**This hash must be recorded before any experimental runs begin.**

---

## Appendix C: Related Documents

- `docs/VSD_PHASE_2.md` — Phase II Uplift Evidence Gate definition
- `docs/canonical_basis_plan.md` — Uplift as input to basis promotion policy
- `config/curriculum.yaml` — Slice configuration (lines 125-149)
- `configs/rfl_experiment_wide_slice.yaml` — RFL runner configuration template
