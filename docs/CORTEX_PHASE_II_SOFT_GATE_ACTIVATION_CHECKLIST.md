# CORTEX Phase II Soft Gate Activation Checklist

**Operation CORTEX: TDA Mind Scanner**
**Phase II: Soft Gating Deployment**

---

## Pre-Activation Requirements

### 1. Phase I Validation Complete
- [ ] Phase I Shadow Mode has run for minimum 1000 cycles
- [ ] HSS correlates positively with verification outcomes (r > 0.3)
- [ ] Cohen's d > 0.8 between good/bad proof DAGs
- [ ] AUC-ROC > 0.80 for HSS-based classification
- [ ] p95 TDA computation latency < 100ms sustained
- [ ] Verify `docs/CORTEX_PHASE_I_ACTIVATION_CHECKLIST.md` complete

### 2. Patch Application
- [ ] Apply or integrate `docs/TDA_RFLRUNNER_LEARNING_RATE_MODULATION.patch`:
  - [ ] `TDAModulationConfig` dataclass added
  - [ ] `_compute_modulated_learning_rate()` method in RFLRunner
  - [ ] New fields in `RunLedgerEntry`: `eta_base`, `eta_eff`, `hss_class`
  - [ ] `get_modulation_stats()` method added
- [ ] Apply or integrate `docs/TDA_U2_PLANNER_REWEIGHTING.patch`:
  - [ ] `TDAPlannerConfig` dataclass added
  - [ ] `tda_planner_config` field in `U2Config`
  - [ ] `_apply_hss_reweighting()` method in U2Runner
  - [ ] `get_reweighting_stats()` method added

### 3. Telemetry Schema Update
- [ ] Verify `experiments/u2/tda_schema_extension.py` updated to v2.0.0:
  - [ ] `TDASoftGateEvent` dataclass present
  - [ ] `TDASoftGateSummaryEvent` dataclass present
  - [ ] `TDAModulationConfigEvent` dataclass present
  - [ ] Helper functions: `softgate_event_to_dict()`, `softgate_summary_to_dict()`
- [ ] Test event serialization:
  ```python
  from experiments.u2.tda_schema_extension import (
      TDASoftGateEvent,
      softgate_event_to_dict,
  )
  event = TDASoftGateEvent(
      cycle=0, slice_name="test", mode="rfl",
      hss=0.7, sns=0.8, pcs=0.6, drs=0.2,
      hss_class="OK", eta_base=0.1, eta_eff=0.1,
      learning_allowed=True, reweighting_applied=True,
      score_delta_mean=0.05, candidates_reweighted=10,
      theta_warn=0.5, theta_block=0.2, lambda_soft=0.3,
      computation_ms=15.0,
  )
  assert "eta_eff" in softgate_event_to_dict(event)
  ```

---

## Configuration Verification

### 4. Learning Rate Modulation Configuration
- [ ] Verify default thresholds:
  ```python
  from experiments.u2.runner import TDAPlannerConfig
  # Or from rfl.runner import TDAModulationConfig

  config = TDAModulationConfig()
  assert config.theta_warn == 0.5
  assert config.theta_block == 0.2
  assert config.lambda_soft == 0.3
  assert config.skip_on_block == True
  ```
- [ ] Verify modulation formula:
  - `f(HSS) = 1.0` when HSS >= 0.5 (OK zone)
  - `f(HSS) = 0.3` when 0.2 <= HSS < 0.5 (WARN zone)
  - `f(HSS) = 0.0` when HSS < 0.2 (SOFT_BLOCK zone)

### 5. Planner Reweighting Configuration
- [ ] Verify default weights:
  ```python
  config = TDAPlannerConfig()
  assert config.base_weight == 0.5  # alpha
  assert config.hss_weight == 0.5   # beta
  assert config.min_score == 0.01
  assert config.max_score == 10.0
  ```
- [ ] Verify reweighting formula:
  - `score' = score * (0.5 + 0.5 * HSS)`
  - When HSS=1.0: score unchanged (multiplied by 1.0)
  - When HSS=0.5: score reduced to 75%
  - When HSS=0.0: score reduced to 50%

### 6. Environment Configuration
- [ ] Set environment for Phase II:
  ```bash
  export MATHLEDGER_TDA_MODE=soft
  ```
- [ ] Verify mode detection:
  ```python
  import os
  assert os.getenv("MATHLEDGER_TDA_MODE") == "soft"
  ```

---

## Smoke Tests

### 7. RFLRunner Learning Rate Modulation Test
- [ ] Run RFL experiment with soft gating:
  ```python
  from rfl.runner import RFLRunner, RFLConfig

  config = RFLConfig(...)
  runner = RFLRunner(config)

  # After running cycles...
  stats = runner.get_modulation_stats()
  assert "hss_class_distribution" in stats
  assert "learning_skipped_count" in stats
  ```
- [ ] Verify modulation statistics recorded:
  - [ ] `OK`, `WARN`, `SOFT_BLOCK` counts present
  - [ ] `learning_skipped_count` >= 0
  - [ ] `eta_history` contains modulation records

### 8. U2Runner Planner Reweighting Test
- [ ] Run U2 experiment with soft gating:
  ```python
  from experiments.u2.runner import U2Runner, U2Config, TDAPlannerConfig
  from backend.tda.runtime_monitor import create_monitor

  config = U2Config(
      experiment_id="softgate_test",
      slice_name="arithmetic_simple",
      mode="rfl",
      total_cycles=50,
      master_seed=42,
      tda_monitor=create_monitor(mode="soft"),
      tda_planner_config=TDAPlannerConfig(enabled=True),
  )
  runner = U2Runner(config)
  # Run cycles...
  stats = runner.get_reweighting_stats()
  assert stats["reweighted_candidates"] > 0
  ```
- [ ] Verify telemetry in `ht_series`:
  - [ ] `tda_reweighting_applied` field present
  - [ ] `current_hss` field present

---

## Validation Analysis

### 9. Run Soft Gate Analysis
- [ ] Execute analysis pipeline:
  ```bash
  python experiments/tda_softgate_analysis.py \
    --baseline-dir results/u2_baseline \
    --softgate-dir results/u2_softgate \
    --output-dir results/tda_phase2_analysis \
    --generate-plots
  ```
- [ ] Review generated report: `results/tda_phase2_analysis/softgate_report.md`
- [ ] Verify acceptance criteria in report:
  - [ ] Success rate delta within acceptable bounds (|delta| < 5%)
  - [ ] Learning rate stability acceptable (std(eta_eff) / mean(eta_eff) < 0.5)
  - [ ] Planner divergence acceptable (KL < 0.5)
  - [ ] Topological correlation positive (r > 0.2)

### 10. Long Run Stress Test
- [ ] Run 2000-cycle stress test:
  ```bash
  python experiments/run_uplift_u2.py \
    --slice arithmetic_simple \
    --mode rfl \
    --cycles 2000 \
    --tda-enabled \
    --tda-mode soft \
    --output-dir results/stress_test_2000
  ```
- [ ] Verify no memory leaks (RSS stable over run)
- [ ] Verify no error accumulation
- [ ] Verify HSS distribution stable (check histogram)

### 11. Slice-by-Slice Validation
- [ ] Run soft gate analysis for each active slice:
  - [ ] `arithmetic_simple`
  - [ ] `arithmetic_medium`
  - [ ] `propositional_basic`
  - [ ] (Add other active slices)
- [ ] Verify per-slice HSS thresholds appropriate
- [ ] Document any slice-specific threshold adjustments needed

---

## Activation

### 12. Final Checklist & Activation
- [ ] All pre-activation requirements complete
- [ ] Patches applied and tested
- [ ] Telemetry schema updated
- [ ] Configuration verified
- [ ] Smoke tests passing
- [ ] Analysis report reviewed
- [ ] Stress test completed
- [ ] No regressions in verification success rate

**Activation Command:**
```bash
# Set environment for Phase II Soft Gating
export MATHLEDGER_TDA_MODE=soft

# Run production experiment with soft gating
python experiments/run_uplift_u2.py \
  --slice arithmetic_simple \
  --mode rfl \
  --cycles 500 \
  --tda-enabled \
  --tda-mode soft \
  --output-dir results/phase2_production
```

---

## Post-Activation Monitoring

### Metrics to Watch
- `tda_modulation_ratio`: Should stabilize around 0.7-0.9
- `tda_soft_block_rate`: Should be < 10%
- `tda_learning_skip_rate`: Should be < 5%
- `success_rate_delta`: Should remain within +/- 5% of baseline

### Alert Conditions
- `soft_block_rate > 15%`: Investigate HSS distribution shift
- `learning_skip_rate > 10%`: Check theta_block threshold
- `success_rate_drop > 10%`: Consider reverting to Phase I
- `computation_latency_p95 > 150ms`: Investigate TDA performance

### Rollback Procedure
```bash
# Immediate rollback to Phase I Shadow Mode
export MATHLEDGER_TDA_MODE=shadow

# Emergency disable (no TDA)
export MATHLEDGER_TDA_MODE=offline
```

---

## Phase II -> Phase III Transition Criteria

Before advancing to Phase III (Hard Gate):

1. **Stability**: Soft gating stable for 5000+ cycles across all slices
2. **Effectiveness**: HSS modulation positively correlates with outcomes
3. **No Regression**: Verification success rate within 2% of Phase I
4. **Coverage**: All active slices validated with soft gating
5. **Documentation**: Phase III migration plan approved

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Integration Issues**: File in `github.com/mathledger/issues`
- **Emergency Rollback**: Set `MATHLEDGER_TDA_MODE=offline`

---

**STRATCOM: CORTEX PHASE II — SOFT GATING ARMED.**
