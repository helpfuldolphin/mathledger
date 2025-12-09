# CORTEX Phase VI: Auto-Watchdog & Global Health Coupler

**Operation CORTEX: TDA Mind Scanner**
**Phase VI: Auto-Watchdog & Global Health Coupler**

---

## Overview

Phase VI transforms CORTEX from "something you look at" into "something that wakes you up when the substrate starts thinking weird."

Three core capabilities:

1. **Global Health Coupler** — TDA as a first-class tile in `global_health.json`
2. **Autonomous Watchdog Daemon** — Unattended monitoring for cron/CI
3. **Explain This Tile** — Bridge between operator console and global health

---

## 1. Global Health Coupler

### Module: `backend/health/tda_adapter.py`

**Purpose**: Adapter from `TDAGovernanceSnapshot` → global_health tile.

### Usage

```python
from backend.health.tda_adapter import summarize_tda_for_global_health

snapshot = build_governance_console_snapshot(...)
tile = summarize_tda_for_global_health(snapshot.to_dict())

print(tile)
```

### Output Schema (`1.0.0`)

```json
{
    "schema_version": "1.0.0",
    "tda_status": "OK",
    "block_rate": 0.0,
    "mean_hss": 0.75,
    "hss_trend": "STABLE",
    "governance_signal": "OK",
    "notes": ["TDA metrics within normal operating range"]
}
```

### Status Classification Rules

| Status | Condition |
|--------|-----------|
| **ALERT** | `governance_signal == "BLOCK"` OR |
| | `block_rate >= 0.2 AND hss_trend == "DEGRADING"` OR |
| | `golden_alignment == "BROKEN"` |
| **ATTENTION** | `block_rate > 0` (any blocking) OR |
| | `hss_trend == "DEGRADING"` OR |
| | `golden_alignment == "DRIFTING"` OR |
| | `governance_signal == "WARN"` |
| **OK** | All other cases |

### Field Mappings

| Snapshot Field | Tile Field | Transformation |
|----------------|------------|----------------|
| `governance_signal` | `governance_signal` | HEALTHY→OK, DEGRADED→WARN, CRITICAL→BLOCK |
| `hss_trend` | `hss_trend` | Uppercase normalization |
| `block_rate` | `block_rate` | Direct, rounded to 4 decimals |
| `mean_hss` | `mean_hss` | Direct, rounded to 4 decimals |

### Extending Global Health

```python
from backend.health.tda_adapter import extend_global_health_with_tda

global_health = load_existing_global_health()
extended = extend_global_health_with_tda(global_health, tda_snapshot)

# global_health.json now has "tda" section
```

---

## 2. Autonomous Watchdog Daemon

### Script: `scripts/tda_watchdog.py`

**Purpose**: Automated monitor for cron/CI integration.

### Usage

```bash
python scripts/tda_watchdog.py \
    --governance-log "artifacts/tda/governance_runs/*.json" \
    --config config/tda_watchdog.yaml \
    --output artifacts/tda/watchdog_report.json
```

### Exit Codes

| Code | Status | Meaning |
|------|--------|---------|
| 0 | OK | All metrics within bounds |
| 1 | ATTENTION | Elevated metrics, monitoring required |
| 2 | ALERT | Critical condition OR error |

### CI/Cron Integration

```yaml
# GitHub Actions example
- name: TDA Watchdog
  run: |
    python scripts/tda_watchdog.py \
      --governance-log "results/tda/*.json" \
      --config config/tda_watchdog.yaml \
      --output results/watchdog.json
  continue-on-error: false  # Fail on ALERT

# Cron example (crontab)
0 * * * * cd /path/to/mathledger && python scripts/tda_watchdog.py --governance-log "results/*.json" --output /tmp/watchdog.json || notify-admin
```

### Configuration

**File**: `config/tda_watchdog.yaml`

```yaml
schema_version: "1.0.0"

block_rate:
  max_ok: 0.05           # Below 5%: OK
  max_attention: 0.15    # 5-15%: ATTENTION, >15%: ALERT

mean_hss:
  min_ok: 0.6            # Above 0.6: OK
  min_attention: 0.4     # 0.4-0.6: ATTENTION, <0.4: ALERT

hss_trend:
  alert_on_degrading: true

golden_alignment:
  alert_on_drifting: true
  alert_on_broken: true

exception_windows:
  max_active_ok: 0       # Any active window: ATTENTION

signal_strength:
  min_runs_for_strong_signal: 10

combined_rules:
  block_rate_with_degrading_trend:
    enabled: true
    block_rate_threshold: 0.2
```

### Report Schema (`1.0.0`)

```json
{
    "schema_version": "1.0.0",
    "generated_at": "2025-12-09T12:00:00Z",
    "tda_status": "ATTENTION",
    "block_rate": 0.08,
    "mean_hss": 0.55,
    "hss_trend": "DEGRADING",
    "governance_signal": "WARN",
    "recent_runs": 5,
    "signal_strength": "weak",
    "alerts": [
        {
            "code": "TDA_BLOCK_RATE_ELEVATED",
            "severity": "ATTENTION",
            "message": "block_rate=8.00% exceeds max_ok=5.00%"
        },
        {
            "code": "TDA_HSS_TREND_DEGRADING",
            "severity": "ATTENTION",
            "message": "hss_trend=DEGRADING over 50 cycles"
        }
    ],
    "metrics": {
        "cycle_count": 50,
        "warn_rate": 0.05,
        "runs_aggregated": 1
    }
}
```

### Alert Codes

| Code | Severity | Trigger |
|------|----------|---------|
| `TDA_BLOCK_RATE_ELEVATED` | ATTENTION | block_rate > max_ok |
| `TDA_BLOCK_RATE_HIGH` | ALERT | block_rate > max_attention |
| `TDA_MEAN_HSS_LOW` | ATTENTION | mean_hss < min_ok |
| `TDA_MEAN_HSS_CRITICAL` | ALERT | mean_hss < min_attention |
| `TDA_HSS_TREND_DEGRADING` | ATTENTION | hss_trend == DEGRADING |
| `TDA_GOLDEN_ALIGNMENT_DRIFTING` | ATTENTION | golden_alignment == DRIFTING |
| `TDA_GOLDEN_ALIGNMENT_BROKEN` | ALERT | golden_alignment == BROKEN |
| `TDA_EXCEPTION_WINDOW_ACTIVE` | ATTENTION | exception_windows_active > 0 |
| `TDA_WEAK_SIGNAL` | ATTENTION | cycle_count < min_runs |
| `TDA_COMBINED_BLOCK_AND_DEGRADING` | ALERT | block_rate >= threshold + degrading trend |

---

## 3. Explain This Tile

### Script: `scripts/tda_explain_tile.py`

**Purpose**: Human-readable explanation of TDA health tiles.

### Usage

```bash
# Explain TDA tile from global_health.json
python scripts/tda_explain_tile.py --global-health global_health.json

# Explain standalone tile
python scripts/tda_explain_tile.py --tile tda_tile.json

# JSON output
python scripts/tda_explain_tile.py --global-health global_health.json --json-stdout
```

### Output Example

```
============================================================
TDA TILE EXPLANATION
============================================================

Status: ATTENTION

Reason Codes:
  - STATUS_ATTENTION
  - BLOCK_RATE_ELEVATED
  - HSS_HEALTHY
  - HSS_TREND_DEGRADING
  - GOVERNANCE_SIGNAL_WARN

Metrics:
  block_rate: 0.0800
  mean_hss: 0.7200
  hss_trend: DEGRADING
  governance_signal: WARN

Summary:
  TDA health tile indicates ATTENTION condition. Block rate is
  elevated (8.00%). Mean HSS (0.7200) is within healthy range.
  HSS trend is degrading over time.

============================================================
```

### Reason Codes

| Category | Codes |
|----------|-------|
| Status | `STATUS_OK`, `STATUS_ATTENTION`, `STATUS_ALERT` |
| Block Rate | `BLOCK_RATE_ZERO`, `BLOCK_RATE_LOW`, `BLOCK_RATE_ELEVATED`, `BLOCK_RATE_HIGH` |
| HSS | `HSS_HEALTHY`, `HSS_LOW`, `HSS_CRITICAL`, `HSS_UNAVAILABLE` |
| HSS Trend | `HSS_TREND_IMPROVING`, `HSS_TREND_STABLE`, `HSS_TREND_DEGRADING`, `HSS_TREND_UNKNOWN` |
| Governance | `GOVERNANCE_SIGNAL_OK`, `GOVERNANCE_SIGNAL_WARN`, `GOVERNANCE_SIGNAL_BLOCK` |

---

## Test Coverage

### Test Files

| File | Coverage |
|------|----------|
| `tests/tda/test_tda_global_health_adapter.py` | Adapter classification, JSON serialization |
| `tests/tda/test_tda_watchdog.py` | Snapshot loading, alert evaluation, exit codes |

### Run Tests

```bash
# All Phase VI tests
pytest tests/tda/test_tda_global_health_adapter.py tests/tda/test_tda_watchdog.py -v

# Adapter tests only
pytest tests/tda/test_tda_global_health_adapter.py -v

# Watchdog tests only
pytest tests/tda/test_tda_watchdog.py -v
```

---

## Safety Invariants

### INV-VI-1: Pure/Read-Only

- Adapter does not mutate upstream snapshots
- Watchdog reads logs but does not modify them
- All functions are pure and side-effect free

### INV-VI-2: Deterministic

- Same snapshots always produce same report
- Alert order is deterministic
- Notes order is deterministic

### INV-VI-3: Neutral Language

- No "good/bad/excellent/poor" in output
- All notes are structural and descriptive
- Human summaries state facts, not judgments

### INV-VI-4: Graceful Degradation

- Missing fields use sensible defaults
- Corrupt files are skipped, not fatal
- Empty inputs produce valid (empty) reports

---

## Operational Playbook

### Daily Monitoring

```bash
# Run watchdog, alert on non-zero exit
python scripts/tda_watchdog.py \
    --governance-log "results/daily/*.json" \
    --config config/tda_watchdog.yaml \
    --output results/daily_watchdog.json

# Check exit code
echo "Exit code: $?"
```

### Investigating Alerts

```bash
# Step 1: Run watchdog to see current status
python scripts/tda_watchdog.py \
    --governance-log "results/*.json" \
    --output /tmp/watchdog.json

# Step 2: Explain the tile
python scripts/tda_explain_tile.py \
    --tile /tmp/global_health.json

# Step 3: For specific cycle details
python -m scripts.tda_explain_block \
    --run-ledger results/run.json \
    --cycle-id 42
```

### CI Pipeline Integration

```yaml
name: TDA Health Check

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  watchdog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run TDA Watchdog
        id: watchdog
        run: |
          python scripts/tda_watchdog.py \
            --governance-log "results/**/*.json" \
            --config config/tda_watchdog.yaml \
            --output watchdog_report.json
        continue-on-error: true

      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: watchdog-report
          path: watchdog_report.json

      - name: Alert on ALERT Status
        if: steps.watchdog.outcome == 'failure'
        run: |
          cat watchdog_report.json
          exit 1
```

---

## Files Created

| File | Description |
|------|-------------|
| `backend/health/tda_adapter.py` | Global health adapter |
| `config/tda_watchdog.yaml` | Watchdog configuration |
| `scripts/tda_watchdog.py` | Autonomous watchdog daemon |
| `scripts/tda_explain_tile.py` | Tile explanation tool |
| `tests/tda/test_tda_global_health_adapter.py` | Adapter tests |
| `tests/tda/test_tda_watchdog.py` | Watchdog tests |
| `docs/CORTEX_PHASE_VI_WATCHDOG.md` | This document |

---

## Integration with Previous Phases

| Phase | Integration Point |
|-------|-------------------|
| Phase III | Hard gate decisions feed into governance snapshot |
| Phase IV | Golden alignment, exception windows included in status |
| Phase V | `build_governance_console_snapshot()` is the input |
| Phase VI | Output: global health tile, watchdog alerts |

### Data Flow

```
TDAMonitorResult (Phase II)
    │
    ▼
Hard Gate Decision (Phase III)
    │
    ▼
Calibration + Exception Windows (Phase IV)
    │
    ▼
Governance Console Snapshot (Phase V)
    │
    ▼
TDA Health Adapter (Phase VI)
    │
    ├─► global_health.json ["tda" section]
    ├─► Watchdog Report + Exit Codes
    └─► Tile Explanation
```

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Watchdog Alerts**: Configure notification webhook in CI
- **Exit Code 2 (ALERT)**: Immediate investigation required

---

**STRATCOM: PHASE V GAVE US EYES. PHASE VI TURNED THOSE EYES INTO A REFLEX.**

**CORTEX PHASE VI — WATCHDOG ARMED AND READY.**
