# CORTEX Phase IV: Operator Guardrails

**Operation CORTEX: TDA Mind Scanner**
**Phase IV: Guardrails for Hard Gate & Operator Ergonomics**

---

## Overview

Phase IV establishes operational guardrails for the TDA hard gate, ensuring that:

1. **The hard gate never silently over-blocks healthy runs**
2. **Operators have clear levers, dry-run modes, and post-hoc audits**
3. **Every gate decision is explainable, calibratable, and reversible**

This phase is designed for external audit readiness. All decisions are traceable, all overrides are logged, and all calibration drift is detectable.

---

## Core Components

### 1. Golden Set Calibration

**Purpose**: Lock in canonical "golden" runs with known-good and known-bad labels to detect calibration drift.

**Module**: `backend/tda/governance.py`

**Key Functions**:
- `LabeledTDAResult` - Data class for labeled golden runs
- `CalibrationResult` - Result of calibration evaluation
- `evaluate_hard_gate_calibration()` - Main calibration function

**Usage**:
```python
from backend.tda.governance import (
    LabeledTDAResult,
    evaluate_hard_gate_calibration,
)

# Define golden set with expected labels
golden_set = [
    LabeledTDAResult(hss=0.85, sns=0.7, pcs=0.6, drs=0.1, expected_label="OK"),
    LabeledTDAResult(hss=0.75, sns=0.8, pcs=0.7, drs=0.05, expected_label="OK"),
    LabeledTDAResult(hss=0.05, sns=0.2, pcs=0.1, drs=0.5, expected_label="BLOCK"),
    LabeledTDAResult(hss=0.12, sns=0.3, pcs=0.2, drs=0.4, expected_label="BLOCK"),
]

# Evaluate calibration
result = evaluate_hard_gate_calibration(golden_set)

print(f"Status: {result.calibration_status}")  # "OK", "DRIFTING", or "BROKEN"
print(f"False Block Rate: {result.false_block_rate:.1%}")
print(f"False Pass Rate: {result.false_pass_rate:.1%}")
```

**Calibration Status Semantics**:
| Status | False Block Rate | False Pass Rate | Action |
|--------|------------------|-----------------|--------|
| OK | < 5% | < 5% | Normal operation |
| DRIFTING | 5-15% | 5-15% | Monitor, consider recalibration |
| BROKEN | > 15% | > 15% | Immediate review required |

**Thresholds** (configurable):
```python
evaluate_hard_gate_calibration(
    golden_runs,
    block_threshold=0.2,                    # HSS < 0.2 triggers block
    false_block_threshold_ok=0.05,          # < 5% for OK status
    false_block_threshold_drifting=0.15,    # < 15% for DRIFTING
    false_pass_threshold_ok=0.05,
    false_pass_threshold_drifting=0.15,
)
```

---

### 2. Hard Gate Modes

**Purpose**: Allow operators to preview hard gate behavior before committing to enforcement.

**Environment Variable**: `MATHLEDGER_TDA_HARD_GATE_MODE`

**Supported Modes**:

| Mode | Value | Behavior |
|------|-------|----------|
| OFF | `off` | TDA completely disabled |
| SHADOW | `shadow` | TDA logs scores, no enforcement |
| DRY_RUN | `dry_run` | TDA logs "would block", no actual blocking |
| HARD | `hard` | Full enforcement (Phase III behavior) |

**Configuration**:
```bash
# Disable TDA entirely
export MATHLEDGER_TDA_HARD_GATE_MODE=off

# Shadow mode (Phase I behavior)
export MATHLEDGER_TDA_HARD_GATE_MODE=shadow

# Dry-run mode - preview what would be blocked
export MATHLEDGER_TDA_HARD_GATE_MODE=dry_run

# Hard enforcement (default)
export MATHLEDGER_TDA_HARD_GATE_MODE=hard
```

**Programmatic Access**:
```python
from backend.tda.governance import TDAHardGateMode

# Read from environment
mode = TDAHardGateMode.from_env()
print(f"Current mode: {mode.value}")

# Manual construction
mode = TDAHardGateMode.DRY_RUN
```

**Decision Logic**:
```python
from backend.tda.governance import evaluate_hard_gate_decision

decision = evaluate_hard_gate_decision(tda_result, mode)

if decision.should_block:
    # Return ABANDONED_TDA
    pass
elif decision.should_log_as_would_block:
    # Log for audit: "would have blocked"
    logger.info(f"[TDA DRY_RUN] Would block: {decision.reason}")
```

---

### 3. Governance Alignment

**Purpose**: Detect when TDA hard gate diverges from other system health signals.

**Module**: `backend/tda/governance.py`

**Key Functions**:
- `GovernanceAlignmentResult` - Alignment evaluation result
- `evaluate_tda_governance_alignment()` - Alignment evaluator

**Alignment Status Semantics**:
| Status | TDA Block Rate | Global Health | Interpretation |
|--------|---------------|---------------|----------------|
| ALIGNED | Any | Any | TDA agrees with other signals |
| TENSION | 10-20% | > 80% OK | TDA blocking more than expected |
| DIVERGENT | > 20% | > 80% OK | Significant disagreement |

**Usage**:
```python
from backend.tda.governance import (
    evaluate_tda_governance_alignment,
    summarize_tda_for_global_health,
)

# Get TDA summary
tda_summary = summarize_tda_for_global_health(tda_results, config)

# Get global health snapshot
global_health = {
    "preflight_ok": True,
    "bundle_ok": True,
    "replay_ok": True,
    "global_ok_fraction": 0.95,
}

# Evaluate alignment
alignment = evaluate_tda_governance_alignment(tda_summary, global_health)

print(f"Alignment: {alignment.alignment_status}")
for note in alignment.notes:
    print(f"  - {note}")
```

**When DIVERGENT**:
- TDA is blocking significantly while other layers report healthy
- Consider activating an exception window
- Review TDA thresholds and reference profiles

---

### 4. Exception Windows

**Purpose**: Time-bounded override of hard gate to dry-run behavior when governance diverges.

**Environment Variable**: `MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS`

**Concept**: When alignment is DIVERGENT, operators can enable an exception window that:
1. Allows N runs to proceed without blocking
2. Logs all "would have blocked" decisions
3. Auto-expires after N runs
4. Returns to HARD mode automatically

**Configuration**:
```bash
# Allow 100 runs in exception window
export MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS=100

# Disable exception windows (default)
export MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS=0
```

**Activation**:
```python
from backend.tda.governance import (
    ExceptionWindowManager,
    get_exception_window_manager,
)

# Get global manager
manager = get_exception_window_manager()

# Activate exception window (requires alignment report reference)
if alignment.alignment_status == "DIVERGENT":
    success = manager.activate(
        reason="DIVERGENT alignment detected, block_rate=25%, global_ok=95%"
    )
    if success:
        logger.info(f"Exception window activated: {manager.runs_remaining} runs")

# Check state
state = manager.get_state()
print(f"Active: {state.active}")
print(f"Runs remaining: {state.runs_remaining}")
print(f"Reason: {state.activation_reason}")
```

**Integration with Decision Logic**:
```python
decision = evaluate_hard_gate_decision(tda_result, mode, exception_manager=manager)

if decision.exception_window_active:
    logger.info(f"[TDA] Exception window: {decision.reason}")
```

**Automatic Expiration**:
- Each call to `evaluate_hard_gate_decision()` consumes one run
- When runs_remaining reaches 0, exception window deactivates
- Subsequent decisions use normal HARD mode behavior

---

### 5. Extended Health Summary (v2)

**Purpose**: Enrich health summary with Phase IV operational context.

**Function**: `summarize_tda_for_global_health_v2()`

**Additional Fields**:
```python
{
    # Phase III fields...
    "cycle_count": 100,
    "block_rate": 0.05,
    "mean_hss": 0.72,

    # Phase IV fields
    "hard_gate_mode": "hard",
    "hard_gate_block_rate": 0.05,
    "hard_gate_exception_window_active": False,
    "hard_gate_exception_runs_remaining": None,
    "hypothetical_block_count": 0,
    "hypothetical_block_rate": 0.0,
}
```

**Usage**:
```python
from backend.tda.governance import summarize_tda_for_global_health_v2

summary = summarize_tda_for_global_health_v2(
    tda_results,
    config,
    hard_gate_mode=TDAHardGateMode.DRY_RUN,
    exception_manager=manager,
    hypothetical_blocks=5,
)
```

---

### 6. Evidence Tile Builder

**Purpose**: Compact, deterministic, neutral-language summary for external audits.

**Function**: `build_tda_hard_gate_evidence_tile()`

**Output Schema** (`tda-evidence-tile-1.0.0`):
```json
{
    "schema_version": "tda-evidence-tile-1.0.0",
    "mode": "hard",
    "block_rate": 0.0512,
    "mean_hss": 0.7234,
    "hss_trend": 0.001234,
    "structural_health": 0.8512,
    "cycle_count": 100,
    "block_count": 5,
    "warn_count": 10,
    "ok_count": 85,
    "exception_active": false,
    "hypothetical_block_count": 0
}
```

**Design Principles**:
- **Deterministic**: Same input always produces same output
- **Neutral Language**: No "good", "bad", "excellent", "poor"
- **Rounded Floats**: Fixed decimal precision for reproducibility
- **Audit-Ready**: Suitable for external consumption

**Usage**:
```python
from backend.tda.governance import build_tda_hard_gate_evidence_tile

tile = build_tda_hard_gate_evidence_tile(tda_summary)

# Include in attestation
attestation["evidence"]["tda_tile"] = tile
```

---

## Operational Playbook

### Scenario 1: Normal Operation

```bash
# Default configuration
export MATHLEDGER_TDA_HARD_GATE_MODE=hard
export MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS=0

# Run experiment
python experiments/run_uplift_u2.py --tda-enabled
```

### Scenario 2: Pre-Deployment Preview

```bash
# Dry-run mode to preview blocking
export MATHLEDGER_TDA_HARD_GATE_MODE=dry_run

# Run experiment
python experiments/run_uplift_u2.py --tda-enabled

# Review logs for "would have blocked" entries
grep "would_block=True" results/experiment.log
```

### Scenario 3: Investigating Over-Blocking

```python
# 1. Run calibration against golden set
result = evaluate_hard_gate_calibration(golden_set)
if result.calibration_status == "BROKEN":
    print("ALERT: Calibration broken")
    for note in result.notes:
        print(f"  {note}")

# 2. Check governance alignment
alignment = evaluate_tda_governance_alignment(tda_summary, global_health)
if alignment.alignment_status == "DIVERGENT":
    print("ALERT: TDA diverging from global health")

# 3. Enable exception window if needed
manager.activate("Investigating over-blocking, calibration=BROKEN")
```

### Scenario 4: Emergency Bypass

```bash
# Immediate soft bypass (keeps logging)
export MATHLEDGER_TDA_HARD_GATE_MODE=dry_run

# Full disable (no TDA at all)
export MATHLEDGER_TDA_HARD_GATE_MODE=off
```

### Scenario 5: Post-Incident Audit

```python
# 1. Generate evidence tile
tile = build_tda_hard_gate_evidence_tile(tda_summary)

# 2. Include in incident report
report = {
    "incident_id": "INC-2025-001",
    "tda_evidence": tile,
    "calibration_result": calibration_result.to_dict(),
    "alignment_result": alignment.to_dict(),
}

# 3. Export for external review
with open("incident_tda_evidence.json", "w") as f:
    json.dump(report, f, indent=2)
```

---

## Test Coverage

### Phase IV Tests

| Test File | Coverage |
|-----------|----------|
| `tests/tda/test_phase4_calibration.py` | Golden set calibration |
| `tests/tda/test_phase4_dry_run.py` | Mode behavior, exception windows |

**Run Tests**:
```bash
# All Phase IV tests
pytest tests/tda/test_phase4_*.py -v

# Calibration only
pytest tests/tda/test_phase4_calibration.py -v

# Dry-run mode only
pytest tests/tda/test_phase4_dry_run.py -v
```

### Key Test Classes

**Calibration Tests**:
- `TestPerfectCalibration` - Baseline OK scenario
- `TestOverBlocking` - High false block rate detection
- `TestUnderBlocking` - High false pass rate detection
- `TestDeterministicBehavior` - Reproducibility guarantees

**Dry-Run Tests**:
- `TestTDAHardGateModeEnum` - Mode enumeration
- `TestTDAHardGateModeFromEnv` - Environment resolution
- `TestHardGateDecisionOffMode` - OFF mode behavior
- `TestHardGateDecisionShadowMode` - SHADOW mode behavior
- `TestHardGateDecisionDryRunMode` - DRY_RUN mode behavior
- `TestHardGateDecisionHardMode` - HARD mode behavior
- `TestExceptionWindowWithHardMode` - Exception window interaction
- `TestExceptionWindowManager` - Window lifecycle
- `TestEvidenceTileGeneration` - Evidence tile properties

---

## Schema Versions

| Component | Schema Version |
|-----------|---------------|
| Calibration Result | `tda-calibration-1.0.0` |
| Evidence Tile | `tda-evidence-tile-1.0.0` |
| Governance Summary | `tda-governance-1.0.0` |
| Drift Report | `tda-drift-report-v1` |

---

## Safety Invariants

### INV-IV-1: No Silent Mode Changes
- Mode is always logged at session start
- Mode changes require environment variable update
- Evidence tile always includes current mode

### INV-IV-2: No Silent Overrides
- Exception windows have explicit activation logs
- Exception window state is included in all summaries
- Auto-expiration is logged

### INV-IV-3: Calibration Traceability
- Golden set is versioned and stored
- Calibration results include notes explaining status
- Threshold parameters are logged

### INV-IV-4: Evidence Tile Determinism
- Same input always produces same tile
- Float values are rounded to fixed precision
- No timestamp or random components in tile

### INV-IV-5: Decision Explainability
- Every HardGateDecision includes human-readable reason
- Reason includes HSS value and decision factors
- Exception window influence is explicit

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATHLEDGER_TDA_HARD_GATE_MODE` | `hard` | Operational mode |
| `MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS` | `0` | Max runs in exception window |

### Calibration Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_threshold` | `0.2` | HSS below this triggers block |
| `false_block_threshold_ok` | `0.05` | Max false block rate for OK |
| `false_block_threshold_drifting` | `0.15` | Max for DRIFTING |
| `false_pass_threshold_ok` | `0.05` | Max false pass rate for OK |
| `false_pass_threshold_drifting` | `0.15` | Max for DRIFTING |

### Alignment Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tension_block_rate_threshold` | `0.1` | Block rate for TENSION |
| `divergent_block_rate_threshold` | `0.2` | Block rate for DIVERGENT |
| `global_ok_threshold` | `0.8` | Global OK fraction for divergence |

---

## Activation Checklist

### Pre-Activation

- [ ] Phase III stable and validated
- [ ] Golden set defined and stored
- [ ] Calibration tests passing
- [ ] Dry-run tests passing
- [ ] Operators trained on mode switching

### Activation

```bash
# 1. Start in DRY_RUN for observation
export MATHLEDGER_TDA_HARD_GATE_MODE=dry_run

# 2. Run test experiment
python experiments/run_uplift_u2.py --tda-enabled --cycles 100

# 3. Review hypothetical blocks
# Check: Is block rate reasonable?
# Check: Are blocked runs truly low-quality?

# 4. Run calibration check
python -c "
from backend.tda.governance import evaluate_hard_gate_calibration
# ... run calibration
"

# 5. If OK, switch to HARD
export MATHLEDGER_TDA_HARD_GATE_MODE=hard
```

### Post-Activation

- [ ] Monitor block rate (alert if > 15%)
- [ ] Monitor calibration status (alert if DRIFTING)
- [ ] Monitor alignment status (alert if DIVERGENT)
- [ ] Generate weekly drift reports
- [ ] Review exception window usage

---

## Contacts & Escalation

- **TDA Module Owner**: See `backend/tda/__init__.py`
- **Phase IV Questions**: File issue in repository
- **Emergency Override**: Set `MATHLEDGER_TDA_HARD_GATE_MODE=off`

---

**STRATCOM: PHASE III GAVE US POWER. PHASE IV GAVE US CONTROL.**

**CORTEX PHASE IV — OPERATOR GUARDRAILS ARMED AND READY.**
