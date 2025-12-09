# CORTEX Phase III Activation Checklist

**Operation CORTEX: TDA Mind Scanner**
**Phase III: Hard Gate + Governance Coupling**

---

## Pre-Activation Requirements

### 1. Phase II Validation Complete
- [ ] Phase II Soft Gating stable for 5000+ cycles
- [ ] Verification success rate within 2% of Phase I baseline
- [ ] HSS modulation positively correlates with outcomes
- [ ] All active slices validated with soft gating
- [ ] Verify `docs/CORTEX_PHASE_II_SOFT_GATE_ACTIVATION_CHECKLIST.md` complete
- [ ] Governance signal stable at "HEALTHY" for extended period

### 2. Hard Gate Implementation
- [ ] Apply or integrate `docs/TDA_PHASE3_U2RUNNER_HARD_GATE.patch`:
  - [ ] `ProofOutcome.ABANDONED_TDA` enum value added
  - [ ] `_should_apply_hard_gate()` method in U2Runner
  - [ ] `_evaluate_hard_gate()` method in U2Runner
  - [ ] Hard gate check before cycle execution
  - [ ] `CycleResult` extended with Phase III fields
- [ ] Apply or integrate `docs/TDA_PHASE3_RFLRUNNER_HARD_GATE.patch`:
  - [ ] `ProofOutcome` enum with `ABANDONED_TDA` for RFL
  - [ ] `_should_apply_hard_gate()` method in RFLRunner
  - [ ] `_build_abandoned_result()` method in RFLRunner
  - [ ] Hard gate check in `run_with_attestation()`
  - [ ] `RflResult` extended with `outcome` and `tda_gate_enforced`

### 3. Governance Module Installation
- [ ] Verify `backend/tda/governance.py` installed:
  - [ ] `compute_tda_pipeline_hash()` function
  - [ ] `summarize_tda_for_global_health()` function
  - [ ] `compute_drift_metrics()` function
  - [ ] `generate_drift_report()` function
  - [ ] `extend_attestation_with_tda()` function
- [ ] Test governance module:
  ```python
  from backend.tda.governance import (
      compute_tda_pipeline_hash,
      summarize_tda_for_global_health,
  )
  # Verify imports work
  ```

---

## Schema Verification

### 4. RunLedgerEntry TDA Fields
- [ ] Verify new fields in `RunLedgerEntry`:
  ```python
  # Phase III fields
  tda_outcome: Optional[str] = None  # "OK", "WARN", "BLOCK", "ABANDONED"
  tda_gate_enforced: bool = False
  tda_pipeline_hash: Optional[str] = None
  lean_submission_avoided: bool = False
  policy_update_avoided: bool = False
  ```
- [ ] Test ledger entry creation with TDA fields

### 5. Attestation Schema Extension
- [ ] Verify attestation metadata includes `tda_governance` section:
  ```python
  attestation_metadata = {
      # ... existing fields ...
      "tda_governance": {
          "phase": "III",
          "mode": "hard",
          "pipeline_hash": "<64-char-hex>",
          "summary": {...},
      }
  }
  ```
- [ ] Test `extend_attestation_with_tda()` function

### 6. Drift Report Schema
- [ ] Verify `tda-drift-report-v1` schema fields:
  - [ ] `schema_version`
  - [ ] `generated_at`
  - [ ] `pipeline_hash`
  - [ ] `baseline_period`
  - [ ] `current_period`
  - [ ] `drift_metrics`
  - [ ] `recommendations`
- [ ] Test `generate_drift_report()` produces valid output

---

## Test Suite Verification

### 7. Phase III Unit Tests
- [ ] Run TDA hard gate tests:
  ```bash
  pytest tests/tda/test_phase3_hard_gate.py -v
  ```
- [ ] All tests passing:
  - [ ] `TestHardGateEnforcement` (4 tests)
  - [ ] `TestProofOutcome` (3 tests)
  - [ ] `TestPipelineHash` (4 tests)
  - [ ] `TestGovernanceSummarization` (5 tests)
  - [ ] `TestDriftDetection` (4 tests)
  - [ ] `TestAttestationBinding` (2 tests)
  - [ ] `TestHardGateIntegrationFlow` (3 tests)

### 8. RFL Hard Gate Tests
- [ ] Run RFL TDA tests:
  ```bash
  pytest tests/rfl/test_rfl_tda_hard_gate.py -v
  ```
- [ ] All tests passing:
  - [ ] `TestRflProofOutcome` (2 tests)
  - [ ] `TestRflRunnerHardGateConfig` (3 tests)
  - [ ] `TestAttestationBlocking` (3 tests)
  - [ ] `TestRunLedgerEntryTdaFields` (2 tests)
  - [ ] `TestPolicyUpdatePrevention` (3 tests)
  - [ ] `TestHardGateStatistics` (3 tests)
  - [ ] `TestRflResultPhase3Fields` (2 tests)
  - [ ] `TestGovernanceSummaryIntegration` (2 tests)

### 9. End-to-End Integration Test
- [ ] Run integration test with hard gate:
  ```bash
  MATHLEDGER_TDA_MODE=hard pytest tests/integration/test_tda_hard_gate_e2e.py -v
  ```
- [ ] Verify:
  - [ ] Low-HSS attempts return `ABANDONED_TDA`
  - [ ] No Lean submissions for blocked attempts
  - [ ] No policy updates for blocked attempts
  - [ ] Telemetry records all blocked attempts
  - [ ] Governance summary computed correctly

---

## Configuration Verification

### 10. Environment Configuration
- [ ] Set environment for Phase III:
  ```bash
  export MATHLEDGER_TDA_MODE=hard
  ```
- [ ] Verify mode detection:
  ```python
  import os
  assert os.getenv("MATHLEDGER_TDA_MODE") == "hard"
  ```
- [ ] Optional: Configure drift report output:
  ```bash
  export MATHLEDGER_TDA_DRIFT_REPORT_PATH=results/tda_drift_report.json
  ```

### 11. Fail-Closed Verification
- [ ] Verify `fail_open=False` in HARD mode:
  ```python
  from backend.tda.runtime_monitor import create_monitor
  monitor = create_monitor(mode="hard")
  assert monitor.cfg.fail_open is False
  ```
- [ ] Test that TDA errors result in WARN (not OK) in HARD mode

### 12. Pipeline Hash Binding
- [ ] Verify pipeline hash computed correctly:
  ```python
  from backend.tda.governance import compute_tda_pipeline_hash
  hash_value = compute_tda_pipeline_hash(config, profiles)
  assert len(hash_value) == 64
  ```
- [ ] Verify hash changes when config changes
- [ ] Verify hash included in attestation metadata

---

## Activation

### 13. Final Pre-Activation Checklist
- [ ] All Phase II validation criteria met
- [ ] Hard gate patches applied and tested
- [ ] Governance module installed
- [ ] All schemas verified
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Configuration verified
- [ ] Rollback procedure documented

### 14. Activation Command
```bash
# Set environment for Phase III Hard Gate
export MATHLEDGER_TDA_MODE=hard

# Run production experiment with hard gate
python experiments/run_uplift_u2.py \
  --slice arithmetic_simple \
  --mode rfl \
  --cycles 500 \
  --tda-enabled \
  --tda-mode hard \
  --output-dir results/phase3_production
```

---

## Post-Activation Monitoring

### 15. Key Metrics to Watch
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `tda_block_rate` | Fraction of cycles blocked | > 15% |
| `tda_mean_hss` | Mean HSS over window | < 0.4 |
| `tda_governance_signal` | Aggregate health | "CRITICAL" |
| `lean_submissions_avoided` | Lean calls saved | Informational |
| `policy_updates_avoided` | Policy updates prevented | Informational |

### 16. Alert Conditions
```yaml
alerts:
  - name: tda_high_block_rate
    condition: tda_block_rate > 0.15
    severity: warning
    action: "Investigate HSS distribution shift"

  - name: tda_critical_health
    condition: tda_governance_signal == "CRITICAL"
    severity: critical
    action: "Consider rollback to Phase II"

  - name: tda_drift_critical
    condition: drift_severity == "critical"
    severity: critical
    action: "Immediate review of reference profiles"
```

### 17. Governance Dashboard
- [ ] Monitor governance signal:
  - HEALTHY: Normal operation
  - DEGRADED: Increased monitoring required
  - CRITICAL: Consider rollback
- [ ] Review drift reports weekly:
  ```bash
  python -c "from backend.tda.governance import generate_drift_report; ..."
  ```

---

## Rollback Procedure

### 18. Rollback to Phase II
```bash
# Immediate rollback to Phase II Soft Gating
export MATHLEDGER_TDA_MODE=soft

# Verify mode change
python -c "import os; print(os.getenv('MATHLEDGER_TDA_MODE'))"
```

### 19. Emergency Disable
```bash
# Emergency disable all TDA
export MATHLEDGER_TDA_MODE=offline

# Or completely remove TDA monitor from config
```

### 20. Rollback Criteria
Rollback to Phase II if any of:
- Block rate exceeds 20% sustained
- Governance signal at "CRITICAL" for > 1 hour
- Verification success rate drops > 5% from baseline
- Drift severity at "critical"
- Unexpected production errors

---

## Safety Invariants Verification

### 21. INV-HARD-1: No Lean Submission for Blocked Attempts
- [ ] Verify blocked attempts never reach Lean:
  ```python
  # In test: mock Lean and verify not called when blocked
  ```

### 22. INV-HARD-2: No Policy Update for Blocked Attempts
- [ ] Verify policy weights unchanged on block:
  ```python
  # In test: capture weights before/after blocked attempt
  ```

### 23. INV-HARD-3: Telemetry Completeness
- [ ] Verify all blocked attempts recorded:
  ```python
  # In test: check telemetry contains ABANDONED_TDA entries
  ```

### 24. INV-HARD-4: Pipeline Hash Binding
- [ ] Verify attestation contains pipeline hash when TDA enabled

### 25. INV-HARD-5: Governance Signal Validity
- [ ] Verify signal always in {"HEALTHY", "DEGRADED", "CRITICAL"}

---

## Documentation

### 26. Specification Documents
- [ ] `docs/CORTEX_PHASE_III_HARD_GATE_SPEC.md` - Complete and reviewed
- [ ] `docs/TDA_PHASE3_U2RUNNER_HARD_GATE.patch` - Ready for application
- [ ] `docs/TDA_PHASE3_RFLRUNNER_HARD_GATE.patch` - Ready for application
- [ ] `backend/tda/governance.py` - Installed and tested

### 27. Runbook Updates
- [ ] Update operations runbook with Phase III procedures
- [ ] Document new metrics and dashboards
- [ ] Document rollback procedures
- [ ] Update on-call escalation for TDA critical alerts

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Integration Issues**: File in `github.com/mathledger/issues`
- **Emergency Rollback**: Set `MATHLEDGER_TDA_MODE=soft`
- **Critical Alert**: Page on-call via PagerDuty

---

**STRATCOM: TDA NO LONGER ADVISES — IT GOVERNS.**

**CORTEX PHASE III — HARD GATE ARMED AND READY.**
