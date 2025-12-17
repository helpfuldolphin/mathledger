# Metrics Threshold Migration Patch Notes

**Date:** 2025-12-11
**Author:** CLAUDE D — Metrics Conformance Layer
**Module:** `backend/health/metrics_thresholds.py`

---

## Overview

This patch introduces the P5 threshold migration infrastructure. The `metrics_thresholds.py` module provides:

1. **Dual threshold registry** (MOCK vs REAL)
2. **Environment-based mode selection** (`METRIC_THRESHOLDS_MODE`)
3. **HYBRID mode** for safe transition comparison
4. **Safe comparison band** utilities
5. **Dual evaluation** with divergence detection

---

## Integration Points

### 1. Replace Hardcoded Thresholds in `metrics_governance_adapter.py`

**Current (lines 21-26):**
```python
_DRIFT_CRITICAL_THRESHOLD = 0.7
_DRIFT_WARN_THRESHOLD = 0.3
_BUDGET_CRITICAL_THRESHOLD = 95.0
_BUDGET_WARN_THRESHOLD = 80.0
_SUCCESS_RATE_CRITICAL_THRESHOLD = 50.0
_SUCCESS_RATE_WARN_THRESHOLD = 80.0
```

**Patched:**
```python
# REAL-READY: Import from threshold registry
from backend.health.metrics_thresholds import get_threshold

# REAL-READY: Dynamic threshold access
_DRIFT_CRITICAL_THRESHOLD = get_threshold("drift_critical")
_DRIFT_WARN_THRESHOLD = get_threshold("drift_warn")
_BUDGET_CRITICAL_THRESHOLD = get_threshold("budget_critical")
_BUDGET_WARN_THRESHOLD = get_threshold("budget_warn")
_SUCCESS_RATE_CRITICAL_THRESHOLD = get_threshold("success_rate_critical")
_SUCCESS_RATE_WARN_THRESHOLD = get_threshold("success_rate_warn")
```

**Note:** For module-level constants, thresholds are read at import time. For dynamic mode switching, call `get_threshold()` inside functions.

---

### 2. Add Dual Evaluation to `_determine_status_light()`

**Current logic:**
```python
def _determine_status_light(...) -> str:
    # Single evaluation path
    if drift_mag >= _DRIFT_CRITICAL_THRESHOLD:
        return "RED"
    ...
```

**Patched (HYBRID mode support):**
```python
from backend.health.metrics_thresholds import (
    get_threshold_mode,
    evaluate_with_dual_thresholds,
    MODE_HYBRID,
)

def _determine_status_light(...) -> str:
    # REAL-READY: Check for HYBRID mode
    mode = get_threshold_mode()

    if mode == MODE_HYBRID:
        # Dual evaluation for comparison
        metrics = _extract_metrics_for_evaluation(drift_compass, budget_view, governance_signal)
        result = evaluate_with_dual_thresholds(metrics)

        # Log divergence if detected
        if result["diverges"]:
            _log_threshold_divergence(result)

        # Use authoritative verdict
        return _status_from_verdict(result["verdict"]["status"])

    # Single evaluation (MOCK or REAL)
    ... existing logic with get_threshold() calls ...
```

---

### 3. Add Band Position Logging to Global Health Surface

**In `global_surface.py`:**
```python
from backend.health.metrics_thresholds import log_band_position, get_threshold_mode, MODE_HYBRID

def build_global_health_surface(...) -> Dict[str, Any]:
    payload = {...}

    # REAL-READY: Attach band position in HYBRID mode
    mode = get_threshold_mode()
    if mode == MODE_HYBRID and p3_metrics and p5_metrics:
        band_position = log_band_position(p3_metrics, p5_metrics)
        payload["_debug_band_position"] = band_position

    return payload
```

---

### 4. Environment Configuration

**Add to `.env` or runtime configuration:**
```bash
# P3/P4 mode (default)
METRIC_THRESHOLDS_MODE=MOCK

# Hybrid comparison mode (Stage 2)
METRIC_THRESHOLDS_MODE=HYBRID

# P5 real telemetry mode (Stage 3)
METRIC_THRESHOLDS_MODE=REAL
```

**Docker/CI:**
```yaml
environment:
  - METRIC_THRESHOLDS_MODE=${METRIC_THRESHOLDS_MODE:-MOCK}
```

---

## Test Integration

### Existing Tests

Existing tests in `test_metrics_governance_tile_serializes.py` should continue to pass unchanged (they run in MOCK mode by default).

### New Tests

Add `tests/ci/test_metrics_thresholds.py` to CI pipeline:

```yaml
# In CI config
- name: Run threshold migration tests
  run: pytest tests/ci/test_metrics_thresholds.py -v
```

### Mode-Specific Test Runs

```bash
# Test in MOCK mode (default)
pytest tests/ci/test_metrics_governance_tile_serializes.py

# Test in HYBRID mode
METRIC_THRESHOLDS_MODE=HYBRID pytest tests/ci/test_metrics_thresholds.py

# Test in REAL mode
METRIC_THRESHOLDS_MODE=REAL pytest tests/ci/test_metrics_thresholds.py
```

---

## Rollback Procedure

### Immediate Rollback

```bash
# Revert to MOCK mode
export METRIC_THRESHOLDS_MODE=MOCK

# Or remove the variable entirely (defaults to MOCK)
unset METRIC_THRESHOLDS_MODE
```

### Code Rollback

If the module needs to be disabled entirely, revert to hardcoded constants:

```python
# Revert metrics_governance_adapter.py lines 21-26 to:
_DRIFT_CRITICAL_THRESHOLD = 0.7
_DRIFT_WARN_THRESHOLD = 0.3
# ... etc
```

---

## Migration Checklist

### Stage 1: Log-Only

- [ ] Import `metrics_thresholds` in `metrics_governance_adapter.py`
- [ ] Replace constants with `get_threshold()` calls
- [ ] Add divergence logging in HYBRID mode
- [ ] Deploy with `METRIC_THRESHOLDS_MODE=MOCK`
- [ ] Verify all existing tests pass
- [ ] Monitor for import errors

### Stage 2: Hybrid Comparison

- [ ] Set `METRIC_THRESHOLDS_MODE=HYBRID`
- [ ] Run 100+ cycles
- [ ] Review `threshold_divergence.jsonl` logs
- [ ] Verify ≤5% additional BLOCKs from P5 thresholds
- [ ] Review band position statistics

### Stage 3: P5 Active

- [ ] Obtain STRATCOM authorization
- [ ] Set `METRIC_THRESHOLDS_MODE=REAL`
- [ ] Monitor BLOCK rate for 24 hours
- [ ] Confirm no false positives on healthy runs
- [ ] Archive P3/P4 threshold constants (keep for reference)

---

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `get_threshold(name, mode=None)` | Get single threshold value |
| `get_all_thresholds(mode=None)` | Get all thresholds as dict |
| `get_threshold_pair(name)` | Get both MOCK and REAL values |
| `get_threshold_mode()` | Get current mode from env |
| `evaluate_with_dual_thresholds(metrics)` | Dual evaluation for HYBRID |
| `log_band_position(p3, p5)` | Compare P3/P5 against safe bands |
| `check_in_band(metric, p3, p5)` | Single metric band check |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MODE_MOCK` | `"MOCK"` | P3/P4 synthetic mode |
| `MODE_HYBRID` | `"HYBRID"` | Dual evaluation mode |
| `MODE_REAL` | `"REAL"` | P5 real telemetry mode |
| `ENV_THRESHOLD_MODE` | `"METRIC_THRESHOLDS_MODE"` | Environment variable name |

---

## Files Changed

| File | Change |
|------|--------|
| `backend/health/metrics_thresholds.py` | **NEW** - Threshold registry |
| `tests/ci/test_metrics_thresholds.py` | **NEW** - 47 tests |
| `backend/health/metrics_governance_adapter.py` | **PENDING** - Integration |
| `backend/health/global_surface.py` | **PENDING** - Band logging |

---

*End of Migration Patch Notes*
