# Metrics P5 Migration Plan

**Version:** 1.0.0
**Status:** Planning Draft
**Phase:** P5 Transition (NOT AUTHORIZED)
**Author:** CLAUDE D — Metrics Conformance Layer
**Date:** 2025-12-11

---

## 1. Threshold Migration Table

| Metric | Constant | P3/P4 Value | P5 Value | Migration Notes |
|--------|----------|-------------|----------|-----------------|
| Drift warn | `_DRIFT_WARN_THRESHOLD` | 0.30 | 0.35 | +17% tolerance for real variance |
| Drift critical | `_DRIFT_CRITICAL_THRESHOLD` | 0.70 | 0.75 | Avoid false BLOCK on spikes |
| Success rate warn | `_SUCCESS_RATE_WARN_THRESHOLD` | 80% | 75% | Real proofs harder |
| Success rate critical | `_SUCCESS_RATE_CRITICAL_THRESHOLD` | 50% | 45% | Accommodate complexity |
| Budget warn | `_BUDGET_WARN_THRESHOLD` | 80% | 85% | Real utilization less uniform |
| Budget critical | `_BUDGET_CRITICAL_THRESHOLD` | 95% | 92% | Earlier warning for real loads |
| Abstention warn | (implicit) | 5% | 8% | Real queues more variable |
| Abstention critical | (implicit) | 15% | 18% | Bursty abstention expected |
| Block rate warn | (implicit) | 0.08 | 0.12 | Real derivation paths vary |
| Block rate critical | (implicit) | 0.20 | 0.25 | Higher ceiling for hard proofs |

### 1.1 Threshold Source Location

```
backend/health/metrics_governance_adapter.py:21-26
```

Current constants:
```python
_DRIFT_CRITICAL_THRESHOLD = 0.7
_DRIFT_WARN_THRESHOLD = 0.3
_BUDGET_CRITICAL_THRESHOLD = 95.0
_BUDGET_WARN_THRESHOLD = 80.0
_SUCCESS_RATE_CRITICAL_THRESHOLD = 50.0
_SUCCESS_RATE_WARN_THRESHOLD = 80.0
```

---

## 2. Safe Rollout Plan

### Stage 1: Log-Only (Shadow Comparison)

**Duration:** 2 calibration runs minimum

**Mode:** `METRIC_THRESHOLDS_MODE=MOCK`

| Action | Description |
|--------|-------------|
| Deploy P5 thresholds as secondary | Add `_P5_*` constants alongside existing |
| Dual evaluation | Evaluate both threshold sets on each cycle |
| Log divergence | Emit `threshold_divergence.jsonl` when verdicts differ |
| No behavior change | P3/P4 thresholds remain authoritative |

**Artifacts:**
```
results/metrics_p5_calibration/{run_id}/
├── threshold_divergence.jsonl    # When P3 vs P5 verdicts differ
├── dual_evaluation_summary.json  # Aggregate comparison stats
└── band_position.jsonl           # Where real metrics land in bands
```

**Exit criteria:**
- [ ] ≥100 cycles with dual evaluation
- [ ] P5 thresholds produce ≤5% more BLOCKs than P3
- [ ] No false BLOCK on healthy runs

---

### Stage 2: Comparison Mode (Hybrid)

**Duration:** 5 calibration runs minimum

**Mode:** `METRIC_THRESHOLDS_MODE=HYBRID`

| Action | Description |
|--------|-------------|
| P5 thresholds primary for logging | P5 verdicts logged as "would-be" status |
| P3 thresholds authoritative | P3 verdicts still control `safe_for_promotion` |
| Alert on divergence | Emit WARN-level log when P3/P5 disagree |
| Collect band statistics | Track % of metrics within safe comparison bands |

**Comparison Band Tracking:**
```json
{
  "cycle": 42,
  "success_rate": {"p3": 92.5, "p5": 88.1, "delta": 4.4, "in_band": true},
  "block_rate": {"p3": 0.05, "p5": 0.09, "delta": 0.04, "in_band": true},
  "drift_magnitude": {"p3": 0.22, "p5": 0.28, "delta": 0.06, "in_band": true},
  "all_in_band": true
}
```

**Exit criteria:**
- [ ] ≥500 cycles in HYBRID mode
- [ ] ≥95% of cycles have all metrics within safe bands
- [ ] P5 BLOCK rate within 2% of P3 BLOCK rate
- [ ] Manual review of all P3/P5 verdict divergences

---

### Stage 3: P5 Thresholds Active

**Mode:** `METRIC_THRESHOLDS_MODE=REAL`

| Action | Description |
|--------|-------------|
| P5 thresholds authoritative | P5 verdicts control `safe_for_promotion` |
| P3 thresholds deprecated | P3 constants retained but unused |
| Full telemetry | Real runner metrics drive all decisions |
| Rollback ready | Env flag allows instant revert to HYBRID |

**Activation checklist:**
- [ ] Stage 2 exit criteria met
- [ ] STRATCOM authorization for P5 activation
- [ ] Rollback procedure documented and tested
- [ ] Alert thresholds configured for P5 behavior

---

## 3. Code-Facing Hooks

### 3.1 Environment Flag

```python
# REAL-READY: Threshold mode selector
# Values: MOCK | HYBRID | REAL
METRIC_THRESHOLDS_MODE = os.environ.get("METRIC_THRESHOLDS_MODE", "MOCK")
```

### 3.2 Threshold Registry

```python
# backend/health/metrics_thresholds.py

# REAL-READY: Dual threshold registry
_THRESHOLDS = {
    "MOCK": {
        "drift_warn": 0.30,
        "drift_critical": 0.70,
        "success_rate_warn": 80.0,
        "success_rate_critical": 50.0,
        "budget_warn": 80.0,
        "budget_critical": 95.0,
        "abstention_warn": 5.0,
        "abstention_critical": 15.0,
        "block_rate_warn": 0.08,
        "block_rate_critical": 0.20,
    },
    "REAL": {
        "drift_warn": 0.35,
        "drift_critical": 0.75,
        "success_rate_warn": 75.0,
        "success_rate_critical": 45.0,
        "budget_warn": 85.0,
        "budget_critical": 92.0,
        "abstention_warn": 8.0,
        "abstention_critical": 18.0,
        "block_rate_warn": 0.12,
        "block_rate_critical": 0.25,
    },
}

def get_threshold(name: str, mode: str = None) -> float:
    """
    REAL-READY: Get threshold by name and mode.

    Args:
        name: Threshold name (e.g., "drift_warn")
        mode: Override mode, defaults to METRIC_THRESHOLDS_MODE env

    Returns:
        Threshold value for the specified mode.
    """
    mode = mode or os.environ.get("METRIC_THRESHOLDS_MODE", "MOCK")
    if mode == "HYBRID":
        # HYBRID uses MOCK as authoritative
        mode = "MOCK"
    return _THRESHOLDS[mode][name]

def get_all_thresholds(mode: str = None) -> Dict[str, float]:
    """REAL-READY: Get all thresholds for a mode."""
    mode = mode or os.environ.get("METRIC_THRESHOLDS_MODE", "MOCK")
    if mode == "HYBRID":
        mode = "MOCK"
    return _THRESHOLDS[mode].copy()
```

### 3.3 Dual Evaluation Hook

```python
# REAL-READY: Dual threshold evaluation for HYBRID mode
def evaluate_with_dual_thresholds(
    drift_compass: Dict[str, Any],
    budget_view: Dict[str, Any],
    governance_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate metrics against both MOCK and REAL thresholds.

    Returns:
        Dict with both verdicts and divergence flag.
    """
    mode = os.environ.get("METRIC_THRESHOLDS_MODE", "MOCK")

    if mode != "HYBRID":
        # Single evaluation
        return {
            "mode": mode,
            "verdict": _evaluate_single(drift_compass, budget_view, governance_signal, mode),
            "dual_evaluation": False,
        }

    # HYBRID: evaluate both
    mock_verdict = _evaluate_single(drift_compass, budget_view, governance_signal, "MOCK")
    real_verdict = _evaluate_single(drift_compass, budget_view, governance_signal, "REAL")

    diverges = mock_verdict["status_light"] != real_verdict["status_light"]

    return {
        "mode": "HYBRID",
        "verdict": mock_verdict,  # MOCK is authoritative
        "p5_verdict": real_verdict,  # P5 for logging
        "dual_evaluation": True,
        "diverges": diverges,
        "divergence_detail": {
            "mock_status": mock_verdict["status_light"],
            "real_status": real_verdict["status_light"],
        } if diverges else None,
    }
```

### 3.4 Band Position Logger

```python
# REAL-READY: Safe comparison band position tracking
def log_band_position(
    p3_metrics: Dict[str, float],
    p5_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Log where metrics fall relative to safe comparison bands.

    Safe bands (from Metrics_PhaseX_Spec.md Section 6.3):
    - success_rate: ±15%
    - block_rate: ±0.08
    - abstention_rate: ±5%
    - drift_magnitude: ±0.15
    - budget_utilization: ±10%
    """
    BANDS = {
        "success_rate": 15.0,
        "block_rate": 0.08,
        "abstention_rate": 5.0,
        "drift_magnitude": 0.15,
        "budget_utilization": 10.0,
    }

    positions = {}
    all_in_band = True

    for metric, band in BANDS.items():
        p3_val = p3_metrics.get(metric, 0.0)
        p5_val = p5_metrics.get(metric, 0.0)
        delta = abs(p3_val - p5_val)
        in_band = delta <= band

        positions[metric] = {
            "p3": p3_val,
            "p5": p5_val,
            "delta": round(delta, 4),
            "band": band,
            "in_band": in_band,
        }

        if not in_band:
            all_in_band = False

    return {
        "positions": positions,
        "all_in_band": all_in_band,
        "out_of_band_count": sum(1 for p in positions.values() if not p["in_band"]),
    }
```

### 3.5 Integration Points

| Location | Hook | Purpose |
|----------|------|---------|
| `metrics_governance_adapter.py:29` | `# REAL-READY` | Replace hardcoded thresholds with `get_threshold()` |
| `metrics_governance_adapter.py:_determine_status_light` | `# REAL-READY` | Add dual evaluation path |
| `global_surface.py:build_global_health_surface` | `# REAL-READY` | Attach band position to payload |
| `test_metrics_governance_tile_serializes.py` | `# REAL-READY` | Add mode-parameterized tests |

---

## 4. Smoke-Test Readiness Checklist

### 4.1 Pre-Switch Verification

| # | Check | Command/Action | Pass Criteria |
|---|-------|----------------|---------------|
| 1 | Threshold registry loads | `python -c "from backend.health.metrics_thresholds import get_all_thresholds; print(get_all_thresholds('REAL'))"` | Returns P5 thresholds |
| 2 | MOCK mode baseline | `METRIC_THRESHOLDS_MODE=MOCK pytest tests/ci/test_metrics_governance_tile_serializes.py -v` | All 37 tests pass |
| 3 | HYBRID mode dual eval | `METRIC_THRESHOLDS_MODE=HYBRID pytest tests/ci/test_metrics_p5_dual_evaluation.py -v` | Dual verdicts logged |
| 4 | Band position logging | Run 10 cycles, verify `band_position.jsonl` emitted | File exists, valid JSON |
| 5 | Divergence detection | Inject divergent metrics, verify alert | Log contains "P3/P5 divergence" |

### 4.2 Switch Execution

| # | Step | Command | Verification |
|---|------|---------|--------------|
| 1 | Set HYBRID mode | `export METRIC_THRESHOLDS_MODE=HYBRID` | Echo confirms |
| 2 | Run calibration | `python -m scripts.run_metrics_calibration --cycles 100` | Completes without error |
| 3 | Review divergences | `jq '.diverges' results/metrics_p5_calibration/*/threshold_divergence.jsonl \| sort \| uniq -c` | ≤5% true |
| 4 | Check band positions | `jq '.all_in_band' results/metrics_p5_calibration/*/band_position.jsonl \| sort \| uniq -c` | ≥95% true |
| 5 | Authorize P5 | STRATCOM approval | Documented |
| 6 | Set REAL mode | `export METRIC_THRESHOLDS_MODE=REAL` | Echo confirms |
| 7 | Smoke test | `pytest tests/ci/test_metrics_governance_tile_serializes.py -v` | All tests pass |

### 4.3 Post-Switch Monitoring

| # | Check | Frequency | Alert Threshold |
|---|-------|-----------|-----------------|
| 1 | BLOCK rate | Per run | >10% increase from baseline |
| 2 | False BLOCK | Per run | Any on healthy run |
| 3 | Band drift | Daily | >2 metrics outside band |
| 4 | Threshold divergence | Per cycle | >10% cycles diverging |

### 4.4 Rollback Procedure

```bash
# Immediate rollback to HYBRID
export METRIC_THRESHOLDS_MODE=HYBRID

# Full rollback to MOCK
export METRIC_THRESHOLDS_MODE=MOCK

# Verify rollback
python -c "import os; print(f'Mode: {os.environ.get(\"METRIC_THRESHOLDS_MODE\", \"MOCK\")}')"
```

---

## 5. Test Matrix

| Test | MOCK | HYBRID | REAL | Notes |
|------|------|--------|------|-------|
| `test_stable_drift_maps_to_green` | PASS | PASS | PASS | Threshold-independent |
| `test_drifting_maps_to_yellow` | PASS | PASS | PASS* | *May differ at boundary |
| `test_critical_drift_maps_to_red` | PASS | PASS | PASS | Threshold-independent |
| `test_high_drift_magnitude_maps_to_red` | PASS | PASS | ADJUST | 0.75 vs 0.70 boundary |
| `test_budget_exceeded_maps_to_red` | PASS | PASS | PASS | Threshold-independent |
| `test_warn_path_degraded_fo_health` | PASS | PASS | PASS* | *75% vs 80% boundary |
| `test_block_path_critical_fo_health` | PASS | PASS | PASS* | *45% vs 50% boundary |

Tests marked ADJUST require mode-aware assertions or parameterization.

---

## Appendix A: Threshold Constant Mapping

```python
# Current → P5 mapping for metrics_governance_adapter.py

# Line 21
_DRIFT_CRITICAL_THRESHOLD = get_threshold("drift_critical")  # 0.70 → 0.75

# Line 22
_DRIFT_WARN_THRESHOLD = get_threshold("drift_warn")  # 0.30 → 0.35

# Line 23
_BUDGET_CRITICAL_THRESHOLD = get_threshold("budget_critical")  # 95.0 → 92.0

# Line 24
_BUDGET_WARN_THRESHOLD = get_threshold("budget_warn")  # 80.0 → 85.0

# Line 25
_SUCCESS_RATE_CRITICAL_THRESHOLD = get_threshold("success_rate_critical")  # 50.0 → 45.0

# Line 26
_SUCCESS_RATE_WARN_THRESHOLD = get_threshold("success_rate_warn")  # 80.0 → 75.0
```

---

*End of Migration Plan*

**CLAUDE D: P5 Metrics Migration Plan Ready.**
