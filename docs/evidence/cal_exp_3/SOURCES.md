# CAL-EXP-3 Evidence Packet Sources

**Generated**: 2025-12-14
**Purpose**: Traceability index for Evidence Packet draft v0.1

---

## Authoritative Specification Documents

| Document | Path | Purpose |
|----------|------|---------|
| CAL_EXP_3_UPLIFT_SPEC.md | `docs/system_law/calibration/CAL_EXP_3_UPLIFT_SPEC.md` | Charter, definitions, validity conditions |
| CAL_EXP_3_IMPLEMENTATION_PLAN.md | `docs/system_law/calibration/CAL_EXP_3_IMPLEMENTATION_PLAN.md` | Execution machinery, artifact layout |
| CAL_EXP_3_LANGUAGE_CONSTRAINTS.md | `docs/system_law/calibration/CAL_EXP_3_LANGUAGE_CONSTRAINTS.md` | Claim language constraints |
| CAL_EXP_3_AUTHORIZATION.md | `docs/system_law/calibration/CAL_EXP_3_AUTHORIZATION.md` | Execution authorization gate |
| CAL_EXP_3_INDEX.md | `docs/system_law/calibration/CAL_EXP_3_INDEX.md` | Artifact contract, canonization record |

---

## Canonical Run Artifacts

### Run: cal_exp_3_seed42_20251214_044612

| Artifact | Path |
|----------|------|
| run_config.json | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/run_config.json` |
| RUN_METADATA.json | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/RUN_METADATA.json` |
| uplift_report.json | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/analysis/uplift_report.json` |
| windowed_analysis.json | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/analysis/windowed_analysis.json` |
| baseline/cycles.jsonl | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/baseline/cycles.jsonl` |
| treatment/cycles.jsonl | `results/cal_exp_3/cal_exp_3_seed42_20251214_044612/treatment/cycles.jsonl` |

### Run: cal_exp_3_seed43_20251214_044619

| Artifact | Path |
|----------|------|
| uplift_report.json | `results/cal_exp_3/cal_exp_3_seed43_20251214_044619/analysis/uplift_report.json` |
| RUN_METADATA.json | `results/cal_exp_3/cal_exp_3_seed43_20251214_044619/RUN_METADATA.json` |

### Run: cal_exp_3_seed44_20251214_051658

| Artifact | Path |
|----------|------|
| uplift_report.json | `results/cal_exp_3/cal_exp_3_seed44_20251214_051658/analysis/uplift_report.json` |
| RUN_METADATA.json | `results/cal_exp_3/cal_exp_3_seed44_20251214_051658/RUN_METADATA.json` |

---

## Figures (Staged)

| Figure | Source Path | Staged Path |
|--------|-------------|-------------|
| Delta vs Cycles | `results/cal_exp_3/run_20251214_042346/cal_exp_3_plots.png` | `figures/cal_exp_3_plots_delta_vs_cycles.png` |
| Success vs Cycles | `results/cal_exp_3/run_20251214_042410/cal_exp_3_plots.png` | `figures/cal_exp_3_plots_success_vs_cycles.png` |

---

## Canonical Components

| Component | Path |
|-----------|------|
| Canonical Producer | `scripts/run_cal_exp_3_canonical.py` |
| Verifier | `scripts/verify_cal_exp_3_run.py` |

---

## External References

| Reference | Location |
|-----------|----------|
| METRIC_DEFINITIONS.md@v1.1.0 | `docs/system_law/calibration/METRIC_DEFINITIONS.md` |

---

**SHADOW-OBSERVE** — verification results are non-blocking.
