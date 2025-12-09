# CORTEX Phase V: Operator Console & Self-Audit Harness

**Operation CORTEX: TDA Mind Scanner**
**Phase V: Operator Console, Self-Audit & Long-Horizon Drift**

---

## Overview

Phase V provides operators, auditors, and meta-learners with complete visibility into TDA hard-gate behavior over time. The mission is threefold:

1. **Single source of truth** — One place to see TDA behavior
2. **Explainability** — Trivial to answer "Why did this get blocked?"
3. **Misconfiguration detection** — Impossible for silent degradation

---

## Core Components

### 1. TDA Governance Console API

**Module**: `backend/tda/governance_console.py`

**Purpose**: Read-only API layer over existing governance primitives (Phases II-IV).

#### TDAGovernanceSnapshot

Immutable snapshot of TDA governance state:

```python
from backend.tda.governance_console import (
    TDAGovernanceSnapshot,
    build_governance_console_snapshot,
)

snapshot = build_governance_console_snapshot(
    tda_results=results,
    hard_gate_decisions=decisions,
    golden_state={"calibration_status": "OK"},
    exception_manager=manager,
)

print(snapshot.to_json())
```

**Output Schema** (`tda-governance-console-1.0.0`):
```json
{
    "schema_version": "tda-governance-console-1.0.0",
    "mode": "hard",
    "cycle_count": 100,
    "block_rate": 0.05,
    "warn_rate": 0.10,
    "mean_hss": 0.72,
    "hss_trend": "stable",
    "golden_alignment": "ALIGNED",
    "exception_windows_active": 0,
    "recent_exceptions": [],
    "governance_signal": "HEALTHY"
}
```

**Field Definitions**:

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | Current hard gate mode (off/shadow/dry_run/hard) |
| `cycle_count` | int | Total cycles evaluated |
| `block_rate` | float | Fraction of cycles blocked |
| `warn_rate` | float | Fraction of cycles warned (not blocked) |
| `mean_hss` | float | Mean Hallucination Stability Score |
| `hss_trend` | string | HSS trend: improving/stable/degrading/unknown |
| `golden_alignment` | string | Calibration status: ALIGNED/DRIFTING/BROKEN/UNKNOWN |
| `exception_windows_active` | int | Count of active exception windows |
| `recent_exceptions` | array | Recent exception window descriptors |
| `governance_signal` | string | Aggregate signal: HEALTHY/DEGRADED/CRITICAL |

---

### 2. Operator CLI: "Explain This Block"

**Script**: `scripts/tda_explain_block.py`

**Purpose**: One-shot tool to explain why the hard gate fired for a specific cycle.

#### Usage

```bash
# Explain cycle 42 in a run ledger
python -m scripts.tda_explain_block \
    --run-ledger artifacts/runs/run_123.json \
    --cycle-id 42

# Write output to file
python -m scripts.tda_explain_block \
    --run-ledger artifacts/runs/run_123.json \
    --cycle-id 42 \
    --output explanation.json
```

#### Output Schema (`tda-block-explanation-1.0.0`)

```json
{
    "schema_version": "tda-block-explanation-1.0.0",
    "run_id": "run_123",
    "cycle_id": 42,
    "tda_mode": "hard",
    "hss": 0.13,
    "scores": {
        "sns": 0.72,
        "pcs": 0.61,
        "drs": 0.48
    },
    "gate_decision": {
        "status": "BLOCK",
        "reason_codes": ["HSS_BELOW_THRESHOLD", "GOLDEN_MISALIGNED"],
        "exception_window_applied": false
    },
    "effects": {
        "lean_submission_avoided": true,
        "policy_update_avoided": true
    },
    "status": "BLOCK"
}
```

#### Reason Codes

| Code | Meaning |
|------|---------|
| `HSS_BELOW_THRESHOLD` | HSS < 0.2 (block threshold) |
| `HSS_BELOW_WARN_THRESHOLD` | 0.2 ≤ HSS < 0.4 (warn threshold) |
| `GOLDEN_MISALIGNED` | Golden set calibration is DRIFTING or BROKEN |
| `EXCEPTION_WINDOW_APPLIED` | Exception window was active |
| `HSS_ACCEPTABLE` | HSS ≥ 0.4 (no issues) |
| `TDA_DATA_MISSING` | No TDA data in ledger entry |
| `CYCLE_NOT_FOUND` | Cycle ID not found in ledger |

#### Robustness

- Never crashes on missing data — emits `status: "UNKNOWN"`
- Supports multiple ledger formats (entries list, cycles dict, results list)
- Uses same decision logic as runtime (no fork)

---

### 3. Long-Horizon Drift Watcher

**Script**: `experiments/tda_longhorizon_drift.py`

**Purpose**: Analyze TDA governance behavior over multiple runs to detect long-term drift.

#### Usage

```bash
# Analyze tiles from glob pattern
python experiments/tda_longhorizon_drift.py \
    --tiles "artifacts/tda/*.json" \
    --output longhorizon_report.json

# Analyze tiles from manifest
python experiments/tda_longhorizon_drift.py \
    --manifest artifacts/tda/manifest.json \
    --output longhorizon_report.json

# Analyze all tiles in directory
python experiments/tda_longhorizon_drift.py \
    --directory artifacts/tda/tiles/ \
    --output longhorizon_report.json
```

#### Output Schema (`tda-longhorizon-drift-1.0.0`)

```json
{
    "schema_version": "tda-longhorizon-drift-1.0.0",
    "runs_analyzed": 37,
    "first_run_timestamp": "2025-12-01T00:00:00Z",
    "last_run_timestamp": "2025-12-09T12:00:00Z",
    "block_rate_trend": "stable",
    "mean_hss_trend": "stable",
    "golden_alignment_trend": "stable",
    "exception_usage": {
        "total_windows": 2,
        "per_run_mean": 0.054,
        "trend": "stable"
    },
    "governance_signal": "OK",
    "recommendations": [],
    "metrics": {
        "block_rate_series": [0.05, 0.06, 0.05, ...],
        "mean_hss_series": [0.72, 0.71, 0.73, ...],
        "block_rate_mean": 0.053,
        "block_rate_std": 0.012,
        "mean_hss_mean": 0.72,
        "mean_hss_std": 0.03
    }
}
```

#### Governance Signals

| Signal | Condition | Action |
|--------|-----------|--------|
| OK | All trends stable | Normal operation |
| ATTENTION | Block rate rising OR HSS degrading | Monitor closely |
| ALERT | Block rate rising AND HSS degrading | Investigate immediately |
| ALERT | Exception usage rising + golden drifting/broken | Recalibration required |

#### Recommendations

The system generates neutral, structural recommendations:

- "Block rate increasing while HSS degrading — investigate TDA calibration"
- "Block rate trend is increasing"
- "Mean HSS trend is degrading"
- "Exception window usage increasing with golden alignment drift — recalibration recommended"
- "Golden alignment broken — immediate review required"
- "Golden alignment drifting — consider recalibration"

---

## Programmatic API

### Build Governance Snapshot

```python
from backend.tda.governance_console import build_governance_console_snapshot

# Compose from existing Phase III/IV primitives
snapshot = build_governance_console_snapshot(
    tda_results=tda_results,       # List of TDAMonitorResult
    hard_gate_decisions=decisions, # List of decision dicts
    golden_state={                 # From evaluate_hard_gate_calibration()
        "calibration_status": "OK",
        "false_block_rate": 0.02,
        "false_pass_rate": 0.01,
    },
    exception_manager=manager,     # ExceptionWindowManager instance
    mode=TDAHardGateMode.HARD,
)

# Export for dashboard
print(snapshot.to_json())
```

### Build Block Explanation

```python
from backend.tda.governance_console import build_block_explanation

explanation = build_block_explanation(
    run_id="run_123",
    cycle_id=42,
    tda_mode=TDAHardGateMode.HARD,
    hss=0.13,
    sns=0.72,
    pcs=0.61,
    drs=0.48,
    block=True,
    warn=False,
    exception_window_applied=False,
    lean_submission_avoided=True,
    policy_update_avoided=True,
    golden_alignment="DRIFTING",
)

print(explanation.to_json())
```

### Build From Ledger Entry

```python
from backend.tda.governance_console import build_block_explanation_from_ledger_entry

# Robust to missing data
explanation = build_block_explanation_from_ledger_entry(
    run_id="run_123",
    cycle_id=42,
    ledger_entry={
        "tda_hss": 0.13,
        "tda_sns": 0.72,
        "tda_pcs": 0.61,
        "tda_drs": 0.48,
        "tda_outcome": "BLOCK",
        "lean_submission_avoided": True,
    },
)
```

### Build Long-Horizon Report

```python
from backend.tda.governance_console import build_longhorizon_drift_report

tiles = [
    {"block_rate": 0.05, "mean_hss": 0.72, "golden_alignment": "ALIGNED"},
    {"block_rate": 0.06, "mean_hss": 0.71, "golden_alignment": "ALIGNED"},
    {"block_rate": 0.08, "mean_hss": 0.68, "golden_alignment": "DRIFTING"},
]

report = build_longhorizon_drift_report(tiles)

if report.governance_signal == "ALERT":
    for rec in report.recommendations:
        print(f"  - {rec}")
```

---

## Test Coverage

### Test Module

`tests/tda/test_phase5_governance_console.py`

### Test Classes

| Class | Coverage |
|-------|----------|
| `TestHSSTrendClassification` | HSS trend detection |
| `TestGoldenAlignmentClassification` | Golden status mapping |
| `TestTDAGovernanceSnapshot` | Snapshot immutability, serialization |
| `TestBuildGovernanceConsoleSnapshot` | Snapshot building from inputs |
| `TestBuildReasonCodes` | Reason code generation |
| `TestBuildBlockExplanation` | Explanation structure |
| `TestBuildBlockExplanationFromLedgerEntry` | Ledger extraction |
| `TestAnalyzeGoldenAlignmentTrend` | Alignment trend analysis |
| `TestTrendClassification` | General trend classification |
| `TestBuildLonghorizonDriftReport` | Drift report building |
| `TestExplainBlockCLI` | CLI smoke tests |
| `TestLonghorizonDriftCLI` | CLI smoke tests |
| `TestDeterministicBehavior` | Reproducibility guarantees |
| `TestEdgeCases` | Boundary conditions |

### Running Tests

```bash
# All Phase V tests
pytest tests/tda/test_phase5_governance_console.py -v

# Specific test class
pytest tests/tda/test_phase5_governance_console.py::TestBuildGovernanceConsoleSnapshot -v
```

---

## Safety Invariants

### INV-V-1: No State Mutation

All console functions are pure and read-only:
- No writes to ledger
- No changes to exception window state
- No side effects in runners

### INV-V-2: Deterministic Output

Same inputs always produce identical outputs:
- Snapshots are deterministic
- Explanations are deterministic
- Drift reports are deterministic (except timestamps)

### INV-V-3: Graceful Degradation

Missing data never causes crashes:
- Missing TDA fields → `status: "UNKNOWN"`
- Missing cycle → `reason_codes: ["CYCLE_NOT_FOUND"]`
- Empty inputs → sane defaults

### INV-V-4: No Normative Language

All output is neutral and structural:
- No "good", "bad", "excellent", "poor"
- Facts only: rates, trends, codes
- Recommendations are actionable, not judgmental

### INV-V-5: Decision Logic Reuse

Explanation uses same logic as runtime:
- `build_block_explanation()` uses same thresholds
- No forked or duplicated decision logic
- Reason codes match actual gate behavior

---

## Schema Versions

| Component | Schema Version |
|-----------|---------------|
| Governance Snapshot | `tda-governance-console-1.0.0` |
| Block Explanation | `tda-block-explanation-1.0.0` |
| Long-Horizon Drift | `tda-longhorizon-drift-1.0.0` |

---

## Operational Playbook

### Daily Health Check

```bash
# Generate governance snapshot
python -c "
from backend.tda.governance_console import build_governance_console_snapshot
# ... build snapshot
print(snapshot.governance_signal)
"

# Run long-horizon analysis
python experiments/tda_longhorizon_drift.py \
    --tiles 'results/tda/*.json' \
    --output results/drift_report.json
```

### Investigating a Block

```bash
# Step 1: Identify the blocked cycle
grep "ABANDONED_TDA" results/run.log

# Step 2: Get structured explanation
python -m scripts.tda_explain_block \
    --run-ledger results/run.json \
    --cycle-id 42 \
    --output investigation/cycle_42_explanation.json

# Step 3: Review explanation
cat investigation/cycle_42_explanation.json | jq '.gate_decision.reason_codes'
```

### Detecting Configuration Drift

```bash
# Weekly drift analysis
python experiments/tda_longhorizon_drift.py \
    --directory results/weekly_tiles/ \
    --output results/weekly_drift.json

# Check governance signal
cat results/weekly_drift.json | jq '.governance_signal'

# If ATTENTION or ALERT, review recommendations
cat results/weekly_drift.json | jq '.recommendations[]'
```

### Audit Evidence Collection

```python
# Collect evidence for audit
from backend.tda.governance_console import (
    build_governance_console_snapshot,
    build_longhorizon_drift_report,
)
from backend.tda.governance import build_tda_hard_gate_evidence_tile

# Current state
snapshot = build_governance_console_snapshot(...)
evidence_tile = build_tda_hard_gate_evidence_tile(snapshot.to_dict())

# Historical trend
drift_report = build_longhorizon_drift_report(tiles)

# Package for audit
audit_package = {
    "snapshot": snapshot.to_dict(),
    "evidence_tile": evidence_tile,
    "drift_report": drift_report.to_dict(),
}
```

---

## Integration with Existing Phases

### Phase III: Hard Gate Enforcement

- Snapshot includes `mode` from Phase III
- Explanation includes `effects.lean_submission_avoided`
- Reason codes reflect actual gate thresholds

### Phase IV: Operator Guardrails

- Snapshot includes `exception_windows_active`
- Explanation includes `exception_window_applied`
- Drift report tracks `exception_usage`
- Golden alignment from `evaluate_hard_gate_calibration()`

### Data Flow

```
TDAMonitorResult (Phase II)
    │
    ▼
Hard Gate Decision (Phase III)
    │
    ▼
Calibration + Alignment (Phase IV)
    │
    ▼
Governance Console Snapshot (Phase V)
    │
    ├─► Operator Dashboard
    ├─► Block Explanation CLI
    └─► Long-Horizon Drift Analysis
```

---

## Files Created

| File | Description |
|------|-------------|
| `backend/tda/governance_console.py` | Console API module |
| `scripts/tda_explain_block.py` | Block explanation CLI |
| `experiments/tda_longhorizon_drift.py` | Drift analysis tool |
| `tests/tda/test_phase5_governance_console.py` | Test suite |
| `docs/CORTEX_PHASE_V_OPERATOR_CONSOLE.md` | This document |

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Phase V Questions**: File issue in repository
- **Governance Signal ALERT**: Investigate immediately

---

**STRATCOM: PHASE III GAVE US TEETH. PHASE IV GAVE US BRAKES.**

**PHASE V GAVE US EYES.**

**CORTEX PHASE V — OPERATOR CONSOLE ARMED AND READY.**
