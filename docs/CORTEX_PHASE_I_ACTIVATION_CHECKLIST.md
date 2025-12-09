# CORTEX Phase I Activation Checklist

**Operation CORTEX: TDA Mind Scanner**
**Phase I: Shadow Mode Deployment**

---

## Pre-Activation Requirements

### 1. Module Installation
- [ ] Verify `backend/tda/` module exists with all 8 submodules:
  - [ ] `__init__.py` (package exports)
  - [ ] `proof_complex.py` (combinatorial complex construction)
  - [ ] `metric_complex.py` (Vietoris-Rips and persistence)
  - [ ] `scores.py` (SNS, PCS, DRS, HSS computation)
  - [ ] `reference_profile.py` (per-slice calibration)
  - [ ] `runtime_monitor.py` (TDAMonitor sidecar)
  - [ ] `backends/protocol.py` (backend abstraction)
  - [ ] `backends/ripser_backend.py` (primary backend)

### 2. Dependency Verification
- [ ] Confirm TDA dependencies in `pyproject.toml`:
  ```bash
  uv pip list | grep -E "(networkx|scipy|ripser|persim)"
  ```
- [ ] Expected versions:
  - networkx >= 3.2.0
  - scipy >= 1.12.0
  - ripser >= 0.6.4
  - persim >= 0.3.2

### 3. Reference Profile Calibration
- [ ] Extract real DAGs:
  ```bash
  python experiments/tda_extract_real_dags.py \
    --output-dir results/tda_real_dags \
    --sample-size 100
  ```
- [ ] Build reference profiles:
  ```bash
  python experiments/tda_build_reference_profiles.py \
    --input-dir results/tda_real_dags \
    --output-dir config/tda_profiles
  ```
- [ ] Verify profiles created:
  - [ ] `config/tda_profiles/profiles/default.json`
  - [ ] `config/tda_profiles/calibration_report.md`

### 4. Synthetic Validation
- [ ] Run synthetic validation harness:
  ```bash
  python experiments/tda_validate_synthetic.py \
    --output-dir results/tda_validation \
    --num-good 50 \
    --num-bad 50 \
    --seed 42
  ```
- [ ] Verify acceptance criteria:
  - [ ] Cohen's d > 0.8 (large effect size)
  - [ ] AUC-ROC > 0.80
  - [ ] Positive HSS-verification correlation

### 5. Predictive Power Analysis
- [ ] Run predictive analysis:
  ```bash
  python experiments/tda_predictive_power.py \
    --input-dir results/tda_validation \
    --output-dir results/tda_predictive_analysis
  ```
- [ ] Review `predictive_report.md`
- [ ] Confirm all acceptance criteria PASSED

---

## U2Runner Integration

### 6. Apply Shadow Mode Patch
- [ ] Review patch: `docs/TDA_U2RUNNER_SHADOW_MODE.patch`
- [ ] Apply or manually integrate TDA hooks:
  - [ ] Add `tda_monitor` field to `U2Config`
  - [ ] Add `_evaluate_tda()` method to `U2Runner`
  - [ ] Add TDA telemetry to `ht_series` records
- [ ] Verify no functional changes to baseline/RFL logic

### 7. U2Runner Smoke Test
- [ ] Run baseline experiment with TDA:
  ```python
  from backend.tda.runtime_monitor import create_monitor

  config = U2Config(
      experiment_id="tda_smoke_test",
      slice_name="arithmetic_simple",
      mode="baseline",
      total_cycles=10,
      master_seed=42,
      tda_monitor=create_monitor(mode="shadow"),
  )
  runner = U2Runner(config)
  # Run cycles...
  ```
- [ ] Verify TDA telemetry in `ht_series`:
  - [ ] `tda_hss` field present
  - [ ] `tda_signal` field present
  - [ ] No runtime errors

---

## RFLRunner Integration

### 8. Apply Shadow Coupling Patch
- [ ] Review patch: `docs/TDA_RFLRUNNER_SHADOW_COUPLING.patch`
- [ ] Apply or manually integrate TDA coupling:
  - [ ] Add `tda_monitor` field to `RFLRunner`
  - [ ] Add `_evaluate_tda_for_attestation()` method
  - [ ] Add TDA fields to `RunLedgerEntry`
- [ ] Verify no changes to policy update logic

### 9. RFLRunner Smoke Test
- [ ] Run RFL experiment with TDA:
  ```python
  runner = RFLRunner(config)
  # After _init_tda_monitor() is called...
  result = runner.run_with_attestation(attestation)
  # Check ledger_entry.tda_hss
  ```
- [ ] Verify TDA telemetry in policy ledger
- [ ] Verify `get_tda_summary()` returns valid stats

---

## Telemetry Verification

### 10. Schema Extension
- [ ] Confirm `experiments/u2/tda_schema_extension.py` exists
- [ ] Verify schema version: `tda-u2-trace-1.0.0`
- [ ] Test event serialization:
  ```python
  from experiments.u2.tda_schema_extension import (
      TDAEvaluationEvent,
      tda_event_to_dict,
  )
  event = TDAEvaluationEvent(cycle=0, ..., hss=0.7, ...)
  assert "hss" in tda_event_to_dict(event)
  ```

### 11. Performance Verification
- [ ] TDA computation < 100ms per evaluation (p95)
- [ ] Error rate < 1%
- [ ] No memory leaks in long-running sessions

---

## Monitoring & Observability

### 12. Metrics Setup
- [ ] Configure Redis metrics (if available):
  - [ ] `ml:metrics:tda:eval_count`
  - [ ] `ml:metrics:tda:eval_latency_ms`
  - [ ] `ml:metrics:tda:block_rate`
  - [ ] `ml:metrics:tda:mean_hss`

### 13. Alert Thresholds
- [ ] Define alert conditions:
  - [ ] `tda_eval_latency_p95 > 100ms`: Investigate performance
  - [ ] `tda_error_rate > 1%`: Check input validation
  - [ ] `tda_mean_hss < 0.3`: Investigate score distribution

---

## Documentation

### 14. Specification Documents
- [ ] `docs/TDA_MIND_SCANNER_SPEC.md` - Complete and up-to-date
- [ ] `docs/TDA_U2RUNNER_INTEGRATION.md` - Integration guidance
- [ ] `tests/tda/` - Test suite with ~78 test cases

### 15. Test Coverage
- [ ] Run TDA test suite:
  ```bash
  pytest tests/tda/ -v
  ```
- [ ] All tests passing
- [ ] Coverage > 80% for `backend/tda/`

---

## Phase I Activation

### 16. Final Checklist
- [ ] All pre-activation requirements complete
- [ ] U2Runner integration tested
- [ ] RFLRunner integration tested
- [ ] Telemetry verified
- [ ] Performance acceptable
- [ ] Documentation complete

### 17. Activation Command
```bash
# Set environment variable to enable TDA Shadow Mode
export MATHLEDGER_TDA_MODE=shadow

# Run experiment with TDA
python experiments/run_uplift_u2.py \
  --slice arithmetic_simple \
  --mode rfl \
  --cycles 100 \
  --tda-enabled
```

### 18. Post-Activation Verification
- [ ] Monitor first 100 cycles for:
  - [ ] No runtime errors
  - [ ] HSS distribution looks reasonable
  - [ ] Computation time within limits
- [ ] Generate initial analysis report:
  ```bash
  python experiments/tda_predictive_power.py \
    --input-dir results/u2_production_data \
    --output-dir results/tda_phase1_analysis
  ```

---

## Phase I → Phase II Transition Criteria

Before advancing to Phase II (Soft Gating):

1. **Correlation Validation**: HSS correlates positively with verification outcomes
2. **Stability**: HSS distributions stable across multiple runs
3. **Performance**: p95 latency < 100ms sustained
4. **Coverage**: Reference profiles built for all active slices
5. **Effect Size**: Cohen's d > 0.8 on production data

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Integration Issues**: File in `github.com/mathledger/issues`
- **Emergency Rollback**: Set `MATHLEDGER_TDA_MODE=offline`

---

**STRATCOM: CORTEX PHASE I DROP READY.**
