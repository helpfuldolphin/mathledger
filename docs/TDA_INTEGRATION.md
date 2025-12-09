# TDA Integration for Multi-Run Evidence Fusion

## Overview

This document describes the TDA (Topological Data Analysis) governance integration into the Phase II multi-run evidence fusion pipeline. The integration enables detection of inconsistencies between uplift signals and structural quality metrics, blocking promotions when hard gate conditions are violated.

## Purpose

**PHASE II — U2 Uplift Experiments Extension**

The TDA integration extends the evidence fusion pipeline to:
1. Collect governance signals (HSS, block_rate, tda_outcome) from experimental runs
2. Detect inconsistencies between performance uplift and structural quality
3. Block promotions when TDA Hard Gate detects structural risks
4. Provide deterministic reproduction of fusion results

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Evidence Fusion Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  U2Runner    │    │  RFLRunner   │    │  Baseline    │  │
│  │              │    │              │    │   Runs       │  │
│  │  + Hard Gate │    │  + Hard Gate │    │              │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                    │                    │          │
│         └────────────────────┴────────────────────┘          │
│                              │                                │
│                    ┌─────────▼──────────┐                    │
│                    │   Run Entries      │                    │
│                    │   (with TDA)       │                    │
│                    └─────────┬──────────┘                    │
│                              │                                │
│                    ┌─────────▼──────────┐                    │
│                    │ fuse_evidence_     │                    │
│                    │   summaries()      │                    │
│                    └─────────┬──────────┘                    │
│                              │                                │
│                    ┌─────────▼──────────┐                    │
│                    │  Fused Evidence    │                    │
│                    │  + Inconsistencies │                    │
│                    │  + TDA Aggregates  │                    │
│                    └─────────┬──────────┘                    │
│                              │                                │
│                    ┌─────────▼──────────┐                    │
│                    │  Promotion Gate    │                    │
│                    │  (SHADOW/ENFORCE)  │                    │
│                    └────────────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

1. **rfl/evidence_fusion.py** - Core fusion logic
   - `fuse_evidence_summaries()`: Aggregates runs, detects inconsistencies
   - `TDAFields`: Governance signal dataclass
   - `RunEntry`: Extended run ledger with TDA
   - `FusedEvidenceSummary`: Complete fusion result

2. **rfl/hard_gate.py** - Hard gate evaluation
   - `evaluate_hard_gate_decision()`: Evaluates single cycle
   - Computes HSS (Hash Stability Score) traces
   - Computes Δp (policy delta) metrics
   - Returns TDA outcome (PASS/WARN/BLOCK/SHADOW)

3. **rfl/runner_integration.py** - Runner integration helpers
   - `HardGateIntegration`: Stateful integration for runners
   - Helper functions for policy snapshots and event stats
   - Creates RunEntry with TDA fields

4. **scripts/promotion_precheck_tda.py** - CLI tool
   - Phase-safe promotion precheck with TDA
   - Loads and fuses evidence from JSONL logs
   - Blocks promotion on structural risks

## TDA Fields

### HSS (Hash Stability Score)

**Range:** 0.0 (unstable) to 1.0 (stable)

Measures policy stability across cycles by comparing policy hash values:
- **1.0**: Policy unchanged from previous cycle (perfectly stable)
- **0.5**: First cycle (no previous hash to compare)
- **< 1.0**: Policy changed (lower values = more change)

**Thresholds:**
- HSS < 0.3: Structural instability (BLOCK in ENFORCE mode)
- HSS < 0.7: Moderate instability (WARN)

### block_rate

**Range:** 0.0 to 1.0

Fraction of events blocked by the event verification gate:
```
block_rate = blocked_events / total_events
```

**Thresholds:**
- block_rate > 0.95: Critical blocking (BLOCK in ENFORCE mode)
- block_rate > 0.80: High blocking (WARN)

### tda_outcome

**Values:** PASS | WARN | BLOCK | SHADOW | UNKNOWN

Hard gate decision outcome:
- **PASS**: All metrics within acceptable ranges
- **WARN**: Metrics concerning but not critical
- **BLOCK**: Critical structural risk detected (in ENFORCE mode)
- **SHADOW**: Same as BLOCK but in SHADOW mode (log only)
- **UNKNOWN**: TDA data not available

## Hard Gate Modes

### SHADOW Mode (Default)

- **Behavior**: Log all issues but don't block promotion
- **Use case**: Phase II development and validation
- **Exit code**: Always 0 (success) unless infrastructure error

```bash
python scripts/promotion_precheck_tda.py \
  --experiment-id EXP_001 \
  --mode SHADOW
```

### ENFORCE Mode

- **Behavior**: Block promotion on critical TDA failures
- **Use case**: Production gating for Phase II promotions
- **Exit code**: 1 (fail) if TDA Hard Gate blocks

```bash
python scripts/promotion_precheck_tda.py \
  --experiment-id EXP_001 \
  --mode ENFORCE
```

## Inconsistency Types

The fusion pipeline detects the following inconsistencies:

### 1. UPLIFT_WITHOUT_QUALITY
- **Condition**: RFL coverage improved but block_rate increased significantly
- **Severity**: warning
- **Meaning**: Performance uplift may be unreliable due to poor event quality

### 2. DEGRADATION_WITH_GOOD_TDA
- **Condition**: RFL coverage decreased but block_rate improved
- **Severity**: info
- **Meaning**: Performance regression but improved quality (may be acceptable)

### 3. HIGH_BLOCK_RATE
- **Condition**: block_rate > 0.5 for any run
- **Severity**: error
- **Meaning**: Excessive event blocking indicates structural problems

### 4. TDA_STRUCTURAL_RISK
- **Condition**: tda_outcome = BLOCK or WARN
- **Severity**: error (BLOCK) or warning (WARN)
- **Meaning**: Hard gate detected structural risks

### 5. MISSING_TDA_DATA
- **Condition**: TDA fields not populated
- **Severity**: warning
- **Meaning**: Run executed without TDA integration

## Usage Examples

### 1. Integrate Hard Gate into RFLRunner

```python
from rfl.config import RFLConfig
from rfl.runner import RFLRunner
from rfl.runner_integration import (
    integrate_with_rfl_runner,
    create_policy_state_snapshot,
    create_mock_event_stats,
)
from rfl.hard_gate import HardGateMode

# Create runner
config = RFLConfig.from_env()
runner = RFLRunner(config)

# Attach hard gate integration
hard_gate = integrate_with_rfl_runner(runner, mode=HardGateMode.SHADOW)

# In run loop:
for cycle in range(config.derive_steps):
    # ... execute cycle ...
    
    # Evaluate hard gate
    decision = hard_gate.evaluate_cycle(
        cycle=cycle,
        policy_state=create_policy_state_snapshot(
            policy_weights=runner.policy_weights,
            success_count=runner.success_count,
            attempt_count=runner.attempt_count,
        ),
        event_stats=create_mock_event_stats(
            total_events=100,
            blocked_events=5,
        ),
    )
    
    # Log decision
    print(f"Cycle {cycle}: {decision.outcome.value} - {decision.decision_reason}")
```

### 2. Create RunEntry with TDA Fields

```python
# After run completes
performance_metrics = {
    "coverage_rate": 0.75,
    "novelty_rate": 0.5,
    "throughput": 10.0,
    "success_rate": 0.8,
    "abstention_fraction": 0.2,
}

run_entry = hard_gate.create_run_entry_with_tda(
    run_id="run_001",
    experiment_id="EXP_001",
    slice_name="slice_a",
    mode="rfl",
    performance_metrics=performance_metrics,
    cycle_count=100,
)

# Save to JSONL
with open("results/EXP_001_rfl_001.jsonl", "a") as f:
    f.write(json.dumps(run_entry.to_dict()) + "\n")
```

### 3. Fuse Evidence from Multiple Runs

```python
from rfl.evidence_fusion import (
    fuse_evidence_summaries,
    save_fused_evidence,
    RunEntry,
)
from pathlib import Path

# Load run entries from JSONL logs
baseline_runs = []
rfl_runs = []

# ... load from files ...

# Fuse evidence
summary = fuse_evidence_summaries(
    baseline_runs=baseline_runs,
    rfl_runs=rfl_runs,
    experiment_id="EXP_001",
    slice_name="slice_a",
    tda_hard_gate_mode="SHADOW",
)

# Save fused evidence
save_fused_evidence(
    summary,
    Path("artifacts/phase_ii/EXP_001/fused_evidence.json"),
)

# Check promotion decision
if summary.promotion_blocked:
    print(f"Promotion blocked: {summary.promotion_block_reason}")
else:
    print("Promotion allowed")
```

### 4. Run Promotion Pre-Check

```bash
# With auto-discovery of run logs
python scripts/promotion_precheck_tda.py \
  --experiment-id EXP_001 \
  --slice-name slice_a \
  --mode SHADOW \
  --output artifacts/phase_ii/EXP_001/precheck_report.json

# With pre-computed evidence pack
python scripts/promotion_precheck_tda.py \
  --experiment-id EXP_001 \
  --slice-name slice_a \
  --mode ENFORCE \
  --evidence-pack artifacts/phase_ii/EXP_001/fused_evidence.json \
  --output artifacts/phase_ii/EXP_001/precheck_report.json
```

## Determinism Guarantees

### Fusion Hash

Every fused evidence summary includes a `fusion_hash` computed from:
1. Sorted run entries (by run_id)
2. Canonical JSON encoding (sorted keys, no whitespace)
3. SHA256 hash

This ensures:
- **Reproducibility**: Same runs always produce same hash
- **Traceability**: Hash uniquely identifies fusion result
- **Integrity**: Detects any tampering with fusion data

```python
# Fusion hash is deterministic
summary1 = fuse_evidence_summaries(baseline_runs, rfl_runs, ...)
summary2 = fuse_evidence_summaries(baseline_runs, rfl_runs, ...)
assert summary1.fusion_hash == summary2.fusion_hash
```

### Ordering Guarantees

- Run entries sorted by `run_id` before fusion
- Inconsistency reports maintain insertion order
- HSS traces ordered by cycle number
- All dictionaries use sorted keys in JSON output

## Phase Safety

The TDA integration is **Phase II only**:

### Phase I (v0.1.0 - v0.2.0)
- Basic precheck: SPARK hermetic tests + attestation
- No TDA integration
- Use `ops/basis_promotion_precheck.py`

### Phase II (v0.3.0+)
- Extended precheck: TDA fusion + multi-run evidence
- TDA Hard Gate evaluation
- Use `scripts/promotion_precheck_tda.py`

Both prechecks coexist and serve different purposes:
- **Phase I**: First Organism metabolism proof
- **Phase II**: Uplift experiments with quality gates

## Advisory-Only Constraints

Per agent constraints, the TDA integration is **advisory only**:

✅ **Allowed:**
- Collect and aggregate TDA metrics
- Detect inconsistencies between uplift and quality
- Block promotions in ENFORCE mode
- Produce governance signals for decision-making

❌ **Not Allowed:**
- Claim uplift without proper evidence (gates G1-G5)
- Make causal claims about policy effectiveness
- Reinterpret Phase I logs as Phase II evidence

## Testing

Run tests:

```bash
# All TDA tests
uv run pytest tests/rfl/test_evidence_fusion.py -v
uv run pytest tests/rfl/test_hard_gate.py -v
uv run pytest tests/rfl/test_runner_integration.py -v

# Specific test
uv run pytest tests/rfl/test_evidence_fusion.py::TestFuseEvidenceSummaries::test_tda_hard_gate_enforce_mode -v
```

## Future Extensions

### Planned Features
1. Real event verification integration (replace mock stats)
2. Policy delta (Δp) computation with actual parameter tracking
3. Additional inconsistency detection patterns
4. TDA metric visualization (HSS curves, block rate trends)
5. Multi-experiment fusion (aggregate across experiments)

### Integration Points
- Wire hard gate into RFLRunner main loop
- Wire hard gate into U2Runner cycle execution
- Integrate with event verification module
- Add TDA fields to experiment logger schemas

## References

- `rfl/evidence_fusion.py` - Core fusion logic
- `rfl/hard_gate.py` - Hard gate evaluation
- `rfl/runner_integration.py` - Runner integration
- `scripts/promotion_precheck_tda.py` - CLI tool
- `rfl/event_verification.py` - Event gate (source of block_rate)
- `docs/VSD_PHASE_2.md` - Phase II governance
