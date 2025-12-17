# Phase X P3/P4 Execution Readiness Checklist

**Status:** ACTIVE — Single Source of Truth for Execution Readiness
**Last Updated:** 2025-12-10
**Binding To:** Phase_X_Prelaunch_Review.md

---

## Overview

This document is the **single source of truth** for P3/P4 execution readiness. Each checkbox must be marked PASS before the corresponding phase can be authorized.

Cross-references:
- Pre-Launch Review: `docs/system_law/Phase_X_Prelaunch_Review.md`
- Divergence Metric Spec: `docs/system_law/Phase_X_Divergence_Metric.md`
- P3 Schemas: `docs/system_law/schemas/first_light/`
- P4 Schemas: `docs/system_law/schemas/phase_x_p4/`

---

## Phase X P3: Synthetic First Light

### P3-01: SyntheticStateGenerator Exists and Is Tested

- [ ] **VERIFY** — Unit tests pass for all parameter ranges
- **Files:**
  - Implementation: `backend/topology/first_light/runner.py:SyntheticStateGenerator`
  - Tests: `tests/topology/test_first_light_synthetic.py` (if exists)
- **Condition:** All unit tests pass; generator produces valid state for tau_0 in [0.16, 0.24]
- **Sign-Off:** ________

### P3-02: Δp Engine Produces Bounded Output

- [ ] **VERIFY** — 100-cycle smoke test shows no NaN/Inf
- **Files:**
  - Implementation: `backend/topology/first_light/delta_p_computer.py`
  - Tests: `tests/topology/test_delta_p_computer.py` (if exists)
- **Condition:** `compute()` never returns NaN or Inf for any valid input
- **Sign-Off:** ________

### P3-03: Red-Flag Predicates Defined

- [ ] **MISSING** — Threshold document exists
- **Required Document:** Formal threshold definitions for each red-flag type
- **Current State:** Thresholds are hardcoded in `red_flag_observer.py`
- **Files:**
  - Implementation: `backend/topology/first_light/red_flag_observer.py`
  - Config: `backend/topology/first_light/config.py`
- **Action:** Create formal threshold specification document or verify existing config suffices
- **Sign-Off:** ________

### P3-04: Red-Flag Observer Connected

- [ ] **VERIFY** — Observer receives Δp stream and emits events
- **Files:**
  - Implementation: `backend/topology/first_light/red_flag_observer.py`
  - Integration: `backend/topology/first_light/runner.py:_run_single_cycle`
- **Condition:** Observer is called on every cycle; events are logged
- **Sign-Off:** ________

### P3-05: Pathological Injection Modes Defined

- [ ] **VERIFY** — At least 3 modes: spike, drift, oscillation
- **Files:**
  - Implementation: `backend/topology/first_light/runner.py:SyntheticStateGenerator`
- **Condition:** Generator can be parameterized to produce pathological trajectories
- **Note:** May require adding explicit pathological modes
- **Sign-Off:** ________

### P3-06: Metrics Windowing Implemented

- [ ] **VERIFY** — Aggregates computed over configurable windows
- **Files:**
  - Implementation: `backend/topology/first_light/metrics_window.py`
  - Integration: `backend/topology/first_light/runner.py`
- **Condition:** Window size configurable; rates computed correctly
- **Sign-Off:** ________

### P3-07: Output Schema — Synthetic Raw

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/first_light/first_light_synthetic_raw.schema.json`
- **Output Path:** `results/first_light/{run_id}/synthetic_raw.jsonl`
- **Sign-Off:** Architect (2025-12-10)

### P3-08: Output Schema — Red Flag Matrix

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/first_light/first_light_red_flag_matrix.schema.json`
- **Output Path:** `results/first_light/{run_id}/red_flag_matrix.json`
- **Sign-Off:** Architect (2025-12-10)

### P3-09: Output Schema — Stability Report

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/first_light/first_light_stability_report.schema.json`
- **Output Path:** `results/first_light/{run_id}/stability_report.json`
- **Sign-Off:** Architect (2025-12-10)

### P3-10: Output Schema — Metrics Windows

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/first_light/first_light_metrics_windows.schema.json`
- **Output Path:** `results/first_light/{run_id}/metrics_windows.json`
- **Sign-Off:** Architect (2025-12-10)

### P3-11: 1000-Cycle Harness Exists

- [ ] **VERIFY** — Script can execute 1000 cycles and write outputs
- **Files:**
  - Harness: `scripts/usla_first_light_harness.py` (STUB)
  - Runner: `backend/topology/first_light/runner.py:FirstLightShadowRunner`
- **Condition:** `FirstLightShadowRunner.run()` completes 1000 cycles without error
- **TODO Marker:** `TODO[PhaseX-Harness]` in harness script
- **Sign-Off:** ________

### P3-12: TDA Metrics Integrated

- [ ] **MISSING** — SNS/PCS/DRS/HSS computed per window
- **Files:**
  - Integration Point: `backend/topology/first_light/metrics_window.py`
  - TDA Monitor: `backend/ht/tda_monitor.py` (if exists)
- **TODO Markers:**
  - `TODO[PhaseX-TDA-P3]` in `runner.py`
  - `TODO[PhaseX-TDA-P3]` in `metrics_window.py`
  - `TODO[PhaseX-TDA-P3]` in `usla_first_light_harness.py`
- **Output:** `first_light_tda_metrics.json`
- **Sign-Off:** ________

### P3-13: No External Dependencies

- [ ] **VERIFY** — P3 runs in fully synthetic mode
- **Condition:** No network calls, no database access, no file I/O beyond output writing
- **Sign-Off:** ________

---

## Phase X P4: Real-Runner Shadow Coupling

### P4-01: P3 Completed Successfully (BLOCKING)

- [ ] **BLOCKED** — At least one 1000-cycle P3 run with non-pathological output
- **Condition:** P3 must complete before P4 can begin
- **Evidence Required:** `first_light_stability_report.json` showing non-pathological behavior
- **Sign-Off:** ________

### P4-02: TelemetryProviderInterface Implemented

- [ ] **SKELETON** — `get_snapshot()` returns valid TelemetrySnapshot
- **Files:**
  - Implementation: `backend/topology/first_light/telemetry_adapter.py`
- **Current State:** Raises `NotImplementedError`
- **Sign-Off:** ________

### P4-03: TelemetrySnapshot Schema Frozen

- [x] **DOCUMENTED** — Versioned schema exists (embedded in p4_divergence_log.schema.json)
- **Schema Reference:** `docs/system_law/schemas/phase_x_p4/p4_divergence_log.schema.json` (contains state structure)
- **Note:** May need dedicated TelemetrySnapshot schema
- **Sign-Off:** Architect (2025-12-10)

### P4-04: TwinRunner Implemented

- [ ] **SKELETON** — Initializes from snapshot, produces predictions
- **Files:**
  - Implementation: `backend/topology/first_light/runner_p4.py:TwinRunner`
- **Current State:** Raises `NotImplementedError`
- **Sign-Off:** ________

### P4-05: DivergenceAnalyzer Implemented

- [ ] **SKELETON** — Computes divergence, classifies severity, logs async
- **Files:**
  - Implementation: `backend/topology/first_light/divergence_analyzer.py`
- **TODO Marker:** `TODO[PhaseX-Divergence-Metric]`
- **Current State:** Raises `NotImplementedError`
- **Sign-Off:** ________

### P4-06: Divergence Metric Defined

- [x] **DOCUMENTED** — Formal definition exists
- **Specification:** `docs/system_law/Phase_X_Divergence_Metric.md`
- **Formula:** `divergence = |Δp_real - Δp_twin|`
- **Sign-Off:** Architect (2025-12-10)

### P4-07: Severity Bands Defined

- [x] **DOCUMENTED** — Thresholds specified
- **Specification:** `docs/system_law/Phase_X_Divergence_Metric.md` Section 3
- **Bands:**
  - NONE: < 0.01
  - INFO: 0.01 - 0.05
  - WARN: 0.05 - 0.15
  - CRITICAL: >= 0.15
- **Sign-Off:** Architect (2025-12-10)

### P4-08: Async Logging Verified

- [ ] **UNVERIFIED** — Logging does not block telemetry read path
- **Condition:** Log writes must be non-blocking
- **Sign-Off:** ________

### P4-09: Shadow-Mode Invariant Enforced

- [ ] **VERIFY** — No control paths from P4 to real runner
- **Files:**
  - All P4 modules in `backend/topology/first_light/*_p4.py`
- **Condition:** Code review confirms no write paths to real runner
- **Sign-Off:** ________

### P4-10: Real USLA Runner Available

- [ ] **VERIFY** — Runner can execute and emit telemetry
- **Condition:** Real runner exists and can be observed
- **Sign-Off:** ________

### P4-11: Output Schema — Divergence Log

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/phase_x_p4/p4_divergence_log.schema.json`
- **Output Path:** `results/phase_x_p4/{run_id}/divergence_log.jsonl`
- **Sign-Off:** Architect (2025-12-10)

### P4-12: Output Schema — Twin Trajectory

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/phase_x_p4/p4_twin_trajectory.schema.json`
- **Output Path:** `results/phase_x_p4/{run_id}/twin_trajectory.jsonl`
- **Sign-Off:** Architect (2025-12-10)

### P4-13: Output Schema — Calibration Report

- [x] **DOCUMENTED** — Schema exists
- **Schema:** `docs/system_law/schemas/phase_x_p4/p4_calibration_report.schema.json`
- **Output Path:** `results/phase_x_p4/{run_id}/calibration_report.json`
- **Sign-Off:** Architect (2025-12-10)

### P4-14: TDA Metrics Integrated

- [ ] **MISSING** — SNS/PCS/DRS/HSS for real and twin trajectories
- **TODO Markers:**
  - `TODO[PhaseX-TDA-P4]` in `runner_p4.py`
  - `TODO[PhaseX-TDA-P4]` in `divergence_analyzer.py`
- **Output:** `p4_tda_metrics.json`
- **Sign-Off:** ________

### P4-15: 1000-Cycle Shadow Harness Exists

- [ ] **VERIFY** — Script can observe 1000 real cycles
- **Files:**
  - Runner: `backend/topology/first_light/runner_p4.py:FirstLightShadowRunnerP4`
- **Condition:** Shadow runner can observe 1000 real cycles without error
- **Sign-Off:** ________

---

## Summary Status

### P3 Readiness

| Category | Checkpoints | Passed | Status |
|----------|-------------|--------|--------|
| Implementation Verified | 7 | 0 | BLOCKED |
| Schemas Documented | 4 | 4 | COMPLETE |
| TDA Integration | 1 | 0 | BLOCKED |
| External Dependencies | 1 | 0 | BLOCKED |
| **Total** | **13** | **4** | **NOT READY** |

### P4 Readiness

| Category | Checkpoints | Passed | Status |
|----------|-------------|--------|--------|
| P3 Prerequisite | 1 | 0 | BLOCKED |
| Implementation (Skeleton) | 4 | 0 | BLOCKED |
| Specs Documented | 3 | 3 | COMPLETE |
| Schemas Documented | 3 | 3 | COMPLETE |
| Verification | 4 | 0 | BLOCKED |
| **Total** | **15** | **6** | **NOT READY** |

---

## Execution Readiness Roadmap

### To Flip P3 GO → PASS:

1. **Verify existing tests pass** for SyntheticStateGenerator, DeltaPComputer, RedFlagObserver
2. **Run 100-cycle smoke test** to confirm no NaN/Inf in Δp output
3. **Add pathological injection modes** to SyntheticStateGenerator (if not present)
4. **Wire harness script** to call FirstLightShadowRunner and write all artifacts
5. **Integrate TDA metrics** via TDAMonitor at window boundaries
6. **Execute 1000-cycle run** and verify all output schemas match specs

### To Flip P4 GO → PASS:

1. **Complete P3 successfully** (prerequisite)
2. **Implement TelemetryProviderInterface** with real runner coupling
3. **Implement TwinRunner** prediction logic
4. **Implement DivergenceAnalyzer** with severity classification per spec
5. **Verify async logging** is non-blocking
6. **Execute 1000-cycle shadow run** against real runner
7. **Integrate TDA metrics** for twin vs real comparison

---

*This checklist is the single source of truth for execution readiness.*
*Update status as work progresses. Do not execute P3/P4 until all checkpoints are PASS.*
