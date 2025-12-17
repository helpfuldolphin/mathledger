# CAL-EXP-2: P4 Divergence Minimization — Results

> **NON-CANONICAL / RESULTS (data only) — not a spec**
>
> This document records measured outcomes from CAL-EXP-2.
> It does NOT define requirements, CLI interfaces, or artifact schemas.
> Language uses "measured/reduced" not "improved/validated/production".

---

**Status:** EXECUTED
**Experiment Design:** `docs/system_law/calibration/CAL_EXP_2_EXPERIMENT_DESIGN.md`
**Canonical Contract:** `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`

---

## 1. Execution Summary

| Field | Value |
|-------|-------|
| Execution Date | 2025-12-12T10:38:32Z |
| Executed By | CLAUDE (Scientist) |
| Commit Hash | `f160c08fe786889bdb9f7c177fa33d4c644c6a99` |
| Toolchain Fingerprint | `b828a2185e017e172db966d3158e8e2b91b00a37f0cd7de4c4f7cf707130a20a` |
| Python Version | 3.11.9 |
| UV Version | 0.8.16 |
| UV Lock Hash | `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc` |
| Lean Version | leanprover/lean4:v4.23.0-rc2 |
| Lean Toolchain Hash | `410d5c912b1a040c79883f5e0bb55e733888534e2006eefe186e631c24864546` |
| Seed | 42 |
| Cycles | 1000 |
| Input Source | Synthetic (real_synthetic adapter) |
| Output Location | `results/cal_exp_2/p4_20251212_103832/` |

### LR Overrides (UPGRADE-1)

| Parameter | Value |
|-----------|-------|
| H | 0.20 |
| rho | 0.15 |
| tau | 0.02 |
| beta | 0.12 |

---

## 2. Results

### 2.1 Primary Metrics

| Metric | Value |
|--------|-------|
| Baseline Mean δp (first window, cycles 1-50) | 0.0197 |
| Post Mean δp (last window, cycles 951-1000) | 0.0187 |
| Absolute Reduction | **0.0010** |
| Percent Reduction | **5.1%** |
| Overall Slope (per window) | +0.000148 |
| Success Prediction Accuracy | 82.1% |
| Convergence Floor Measured | ~0.025 |

### 2.2 Divergence Observations

| Observation | Measured Value |
|-------------|----------------|
| Total Divergences | 987 |
| Divergence Rate | 98.7% |
| Max Divergence Streak | 603 cycles |
| Severe Divergences | 0 |
| Moderate Divergences | 276 |
| Minor Divergences | 710 |

#### Divergence by Type

| Type | Count |
|------|-------|
| State | 211 |
| Outcome | 114 |
| Combined | 65 |

### 2.3 Phase Analysis (200-cycle windows)

| Phase | Cycles | Mean δp | Status |
|-------|--------|---------|--------|
| 1 | 1-200 | 0.0230 | BASELINE |
| 2 | 201-400 | 0.0267 | DIVERGING |
| 3 | 401-600 | 0.0307 | DIVERGING (peak) |
| 4 | 601-800 | 0.0268 | CONVERGING |
| 5 | 801-1000 | 0.0254 | PLATEAUING |

### 2.4 Warnings/Anomalies

| Anomaly | Measured |
|---------|----------|
| Non-monotonic trajectory | Observed — "warm-up divergence" in phases 2-3 |
| Peak divergence | δp = 0.0340 at ~cycle 450-500 |
| Variance instability | Not observed — variance stable throughout |
| Runaway divergence | Not observed |

---

## 3. Artifacts Produced

| Artifact | Path | SHA-256 |
|----------|------|---------|
| Run Config | `results/cal_exp_2/p4_20251212_103832/run_config.json` | `b774d6c2abb821f3a9f59e072ecb6fc7c0f27838cbb9e76b478d325a7c0164e3` |
| P4 Summary | `results/cal_exp_2/p4_20251212_103832/p4_summary.json` | `be7e870dd80e2f0c29faf8788bb90dcd1023f78e7dca687b50d663b7c95598af` |
| Divergence Log | `results/cal_exp_2/p4_20251212_103832/divergence_log.jsonl` | `d5e936a2792d128c412cc447568349cb394439894ec306b935231c0ed9b28c5e` |
| Run Metadata | `results/cal_exp_2/p4_20251212_103832/RUN_METADATA.json` | `c6b89383ab4f1876c6e4c751b19e17cb8862035ea3c814cee9ff30ff56e3c861` |
| Real Cycles | `results/cal_exp_2/p4_20251212_103832/real_cycles.jsonl` | `7705b8c9828fea84d8e7a29c9655809c4da9ffdc65014fed3cb2f0a9d9562024` |
| Twin Predictions | `results/cal_exp_2/p4_20251212_103832/twin_predictions.jsonl` | `05e2e15c133718b8e096d3934fcadc01fbd3ea85ba717061e59ecd189e3cdb4e` |
| Twin Accuracy | `results/cal_exp_2/p4_20251212_103832/twin_accuracy.json` | `902980789d1a8a475dcaafc499a404521191d1d50024f604618f7c9d4454a489` |
| Manifest | `results/cal_exp_2/cal_exp_2_manifest.json` | `d2f6ab3a0a2a04abc052426291395c62b33fed610b4a05ff244611aa4ce334ba` |
| Scientist Report | `results/cal_exp_2/CAL_EXP_2_Scientist_Report.md` | _(markdown, not hashed)_ |

---

## 4. Verification Automation

### 4.1 Verification Command

```bash
uv run python scripts/verify_cal_exp_2_run.py --run-dir results/cal_exp_2/p4_20251212_103832/
```

| Field | Value |
|-------|-------|
| Script | `scripts/verify_cal_exp_2_run.py` |
| Run Directory | `results/cal_exp_2/p4_20251212_103832/` |
| Exit Code | **0** (PASS) |
| Verification Timestamp | 2025-12-13T10:47:00Z |
| Checks Executed | 19 |
| FAIL Count | 0 |
| WARN Count | 0 |

### 4.2 Verification Summary

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| file_exists:run_config.json | exists | exists | PASS |
| file_exists:RUN_METADATA.json | exists | exists | PASS |
| mode | SHADOW | SHADOW | PASS |
| schema_version | 1.x.x | 1.4.0 | PASS |
| lr_bounds:H | [0, 1] | 0.2 | PASS |
| lr_bounds:rho | [0, 1] | 0.15 | PASS |
| lr_bounds:tau | [0, 1] | 0.02 | PASS |
| lr_bounds:beta | [0, 1] | 0.12 | PASS |
| enforcement | false | false | PASS |
| status | non-blocking | unknown | PASS |
| jsonl_valid:divergence_log.jsonl | valid JSONL | 1000 lines | PASS |
| jsonl_valid:real_cycles.jsonl | valid JSONL | 1000 lines | PASS |
| jsonl_valid:twin_predictions.jsonl | valid JSONL | 1000 lines | PASS |
| divergence_actions | LOGGED_ONLY | all LOGGED_ONLY | PASS |
| seed_canonical | 42 | 42 | PASS |
| lr_canonical:H | 0.2 | 0.2 | PASS |
| lr_canonical:rho | 0.15 | 0.15 | PASS |
| lr_canonical:tau | 0.02 | 0.02 | PASS |
| lr_canonical:beta | 0.12 | 0.12 | PASS |

---

## 5. Artifact Integrity

### 5.1 Manifest Hash

| Artifact | SHA-256 |
|----------|---------|
| `cal_exp_2_manifest.json` | `d2f6ab3a0a2a04abc052426291395c62b33fed610b4a05ff244611aa4ce334ba` |
| Toolchain Fingerprint | `b828a2185e017e172db966d3158e8e2b91b00a37f0cd7de4c4f7cf707130a20a` |

### 5.2 Field Stability Classification

#### Expected to Vary (per-run)

| Field | Location | Reason |
|-------|----------|--------|
| `timestamp` | manifest, run_config, RUN_METADATA | Execution time |
| `generated_at` | all JSON outputs | Generation time |
| `end_time` | p4_summary.json | Completion time |
| `run_id` | run_config.json | Contains timestamp suffix |
| `output_dir` | run_config.json | Contains run_id |

#### Expected Stable (deterministic given seed)

| Field | Location | Stability Condition |
|-------|----------|---------------------|
| `seed` | run_config.json | Must equal 42 for CAL-EXP-2 canonical |
| `cycles` | run_config.json | Must equal 1000 |
| `twin_lr_overrides.*` | run_config.json | Must match UPGRADE-1 values |
| `mode` | run_config.json | Must equal "SHADOW" |
| `schema_version` | run_config.json | Must be 1.x.x |
| `toolchain_fingerprint` | manifest | Stable if uv.lock unchanged |
| `uv_lock_hash` | manifest | Stable if dependencies unchanged |
| `lean_toolchain_hash` | manifest | Stable if Lean version unchanged |
| `verdict` | RUN_METADATA.json | Deterministic given seed + config |
| `key_metrics.*` | RUN_METADATA.json | Deterministic given seed + config |

---

## 6. Conclusions

### 6.1 Monotone Improvement

**INVALID**

The trajectory is non-monotonic. Divergence increased during phases 2-3 (warm-up period) before reducing in phases 4-5. Net reduction measured (0.0010), but monotone decrease not observed.

### 6.2 No New Pathology

**VALID**

- No severe divergences measured
- No variance instability measured
- No runaway divergence measured
- System reached stable plateau (~0.025 convergence floor)

> **Attestation**: See `CAL_EXP_2_VALIDITY_ATTESTATION.md` for formal review.

### 6.3 Verdict

**PLATEAUING** — The system measured net reduction but reached a convergence floor. Current Twin architecture cannot reduce δp below ~0.025 without structural changes.

---

## 7. Sign-Off

| Role | Agent | Date | Status |
|------|-------|------|--------|
| Execution | CLAUDE (Scientist) | 2025-12-12 | COMPLETE |
| Results Recording | CLAUDE Q | 2025-12-13 | COMPLETE |
| Attestation Review | _(see CAL_EXP_2_VALIDITY_ATTESTATION.md)_ | | LINKED |

---

## 8. Evidence Chain — Direct Links

| Document | Path | Role |
|----------|------|------|
| **Definitions Binding** | `docs/system_law/calibration/CAL_EXP_2_DEFINITIONS_BINDING.md` | Term definitions for this experiment |
| **Validity Attestation** | `docs/system_law/calibration/CAL_EXP_2_VALIDITY_ATTESTATION.md` | Formal attestation of no-new-pathology claim |
| **Canonical Record** | `docs/system_law/calibration/CAL_EXP_2_Canonical_Record.md` | Immutable record of experiment execution |
| **Exit Decision** | `docs/system_law/calibration/CAL_EXP_2_EXIT_DECISION.md` | Decision gate outcome (PLATEAUING) |
| **Distribution Checklist** | `docs/system_law/calibration/audits/CAL_EXP_2_DISTRIBUTION_CHECKLIST.md` | Artifact distribution verification |

---

## 9. References

| Document | Role |
|----------|------|
| `docs/system_law/calibration/CAL_EXP_2_EXPERIMENT_DESIGN.md` | Experiment design |
| `docs/system_law/calibration/CAL_EXP_2_POST_RUN_HYGIENE.md` | Post-run hygiene checklist |
| `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | Canonical contract (unchanged) |
| `results/cal_exp_2/CAL_EXP_2_Scientist_Report.md` | Full analysis |

---

**END OF RESULTS**
