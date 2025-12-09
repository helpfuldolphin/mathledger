# Cortex: TDA Hard Gate Integration

**STRATCOM: FIRST LIGHT. The Brain is connected.**

## Overview

The **Cortex** is the central decision point that integrates **TDA (Topological Decision Analysis)** hard gates with **Safety SLO (Service Level Objective)** outcomes. It controls whether U2Runner and RFLRunner can proceed with execution.

This is a **BLOCKING** system - when the Cortex says stop, the organism stops.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CORTEX                               │
│            evaluate_hard_gate_decision()                    │
│                                                             │
│  Input: Execution Context (metrics, state)                 │
│  Output: SafetyEnvelope (decision + audit trail)           │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼──────┐        ┌──────▼──────┐
        │     TDA      │        │   Safety    │
        │   Analysis   │        │     SLO     │
        │              │        │             │
        │ - should_block │      │ - status    │
        │ - mode       │        │ - message   │
        │ - reason     │        │ - metadata  │
        └──────────────┘        └─────────────┘
                │                       │
                └───────────┬───────────┘
                            │
                    ┌───────▼────────┐
                    │ SafetyEnvelope │
                    │                │
                    │ decision:      │
                    │  - proceed     │
                    │  - block       │
                    └────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      ┌─────▼─────┐   ┌────▼────┐   ┌─────▼─────┐
      │ RFLRunner │   │U2Runner │   │ Future... │
      └───────────┘   └─────────┘   └───────────┘
```

## Components

### 1. SafetyEnvelope (`backend/safety/envelope.py`)

The envelope that wraps TDA decisions with safety guarantees.

```python
@dataclass
class SafetyEnvelope:
    slo: SafetySLO                # Safety SLO outcome
    tda_should_block: bool        # TDA recommendation
    tda_mode: str                 # "BLOCK", "DRY_RUN", "SHADOW"
    decision: str                 # Final decision: "proceed" or "block"
    audit_trail: Dict[str, Any]   # Complete audit trail for CI
```

### 2. SafetySLO (`backend/safety/envelope.py`)

Safety Service Level Objective tracking.

```python
class SLOStatus(str, Enum):
    PASS = "pass"        # System healthy, proceed
    WARN = "warn"        # Warning condition
    BLOCK = "block"      # Critical issue, must block
    ADVISORY = "advisory"  # DRY_RUN/SHADOW mode
```

### 3. TDA Modes (`backend/tda/modes.py`)

Three operational modes control how decisions affect execution:

- **BLOCK**: Production mode. TDA decisions directly control execution.
- **DRY_RUN**: Advisory mode. Records what TDA would do without blocking.
- **SHADOW**: Hypothetical mode. Records decisions for analysis without affecting execution.

### 4. Cortex Function (`backend/safety/cortex.py`)

The central decision point:

```python
def evaluate_hard_gate_decision(
    context: Dict[str, Any],
    tda_mode: Optional[TDAMode] = None,
    timestamp: Optional[datetime] = None,
) -> SafetyEnvelope:
    """
    Evaluate hard gate decision (the "Cortex").
    
    Returns SafetyEnvelope with decision and complete audit trail.
    """
```

## Integration Points

### RFLRunner Integration

Location: `rfl/runner.py` → `run_with_attestation()`

The Cortex is invoked **after attestation validation** and **before policy updates**:

```python
def run_with_attestation(self, attestation: AttestedRunContext) -> RflResult:
    # ... validation ...
    
    # CORTEX: TDA Hard Gate Decision (BLOCKING)
    if self.config.tda_enabled:
        gate_context = {
            "abstention_rate": attestation.abstention_rate,
            "coverage_rate": attestation.metadata.get("coverage_rate", 0.0),
            "verified_count": attestation.metadata.get("verified_count", 0),
            "cycle_index": self.first_organism_runs_total,
            "composite_root": attestation.composite_root,
            "slice_id": attestation.slice_id,
        }
        
        tda_mode = TDAMode(self.config.tda_mode)
        gate_envelope = evaluate_hard_gate_decision(gate_context, tda_mode)
        
        # Log decision
        logger.info(f"[CORTEX] {gate_envelope.slo.message}")
        
        # Record in attestation for audit
        self.dual_attestation_records.setdefault("gate_decisions", []).append({
            "cycle": self.first_organism_runs_total,
            "envelope": gate_envelope.to_dict(),
        })
        
        # BLOCKING CHECK
        check_gate_decision(gate_envelope)
    
    # ... continue with policy update ...
```

**Configuration** (`rfl/config.py`):
```python
@dataclass
class RFLConfig:
    tda_mode: str = "BLOCK"      # "BLOCK", "DRY_RUN", "SHADOW"
    tda_enabled: bool = True     # Enable/disable gate
```

### U2Runner Integration

Location: `experiments/u2/runner.py` → `run_cycle()`

The Cortex is invoked **after cycle initialization** and **before candidate processing**:

```python
def run_cycle(self, cycle: int, ...) -> CycleResult:
    # ... initialization ...
    
    # CORTEX: TDA Hard Gate Decision (BLOCKING)
    if self.config.tda_enabled:
        gate_context = {
            "abstention_rate": 0.0,
            "coverage_rate": 0.0,
            "verified_count": candidates_processed,
            "cycle_index": cycle,
            "frontier_size": self.frontier.size(),
            "experiment_id": self.config.experiment_id,
            "slice_name": self.config.slice_name,
        }
        
        tda_mode = TDAMode(self.config.tda_mode)
        gate_envelope = evaluate_hard_gate_decision(gate_context, tda_mode)
        
        # Log to trace if available
        if trace_ctx:
            trace_ctx.trace_logger.log_event(
                EventType.DERIVE_SUCCESS if not gate_envelope.is_blocking() else EventType.DERIVE_FAILURE,
                cycle=cycle,
                data={"gate_decision": gate_envelope.decision, ...}
            )
        
        # BLOCKING CHECK
        check_gate_decision(gate_envelope)
    
    # ... process candidates ...
```

**Configuration** (`experiments/u2/runner.py`):
```python
@dataclass
class U2Config:
    tda_mode: str = "BLOCK"      # "BLOCK", "DRY_RUN", "SHADOW"
    tda_enabled: bool = True     # Enable/disable gate
```

## TDA Decision Logic

Current TDA analyzer (`backend/tda/analyzer.py`) blocks execution when:

1. **Critical Abstention Rate**: `abstention_rate > 0.5` (50%)
   - System is failing to produce valid results
   
2. **Poor Coverage**: `coverage_rate < 0.5` after 10+ cycles
   - Search is not covering the problem space
   
3. **No Verified Proofs**: `verified_count == 0` after 5+ cycles
   - System has made no progress

All conditions include confidence scoring for future adaptive behavior.

## Mode Behavior Matrix

| TDA Mode  | should_block=True       | should_block=False     | Execution Effect |
|-----------|------------------------|------------------------|------------------|
| BLOCK     | SLO=BLOCK, decision=block | SLO=PASS, decision=proceed | **BLOCKING** |
| DRY_RUN   | SLO=ADVISORY, decision=proceed | SLO=ADVISORY, decision=proceed | **NON-BLOCKING** |
| SHADOW    | SLO=ADVISORY, decision=proceed | SLO=ADVISORY, decision=proceed | **NON-BLOCKING** |

### Mode Use Cases

- **BLOCK**: Production runs, CI/CD gates, experiment validation
- **DRY_RUN**: Testing new TDA logic without affecting runs
- **SHADOW**: A/B testing, collecting decision data for analysis

## Determinism & Reproducibility

The Cortex is designed for **deterministic CI reproduction**:

1. **Deterministic Timestamps**: Uses `substrate.repro.determinism.deterministic_timestamp()`
2. **Serializable State**: All decisions serialize to JSON via `.to_dict()`
3. **Complete Audit Trail**: Every decision includes full context and reasoning
4. **Idempotent Evaluation**: Same context → same decision (same seed)

Example audit trail:
```json
{
  "timestamp": "1970-01-01T00:00:00Z",
  "tda_decision": {
    "should_block": false,
    "mode": "BLOCK",
    "reason": "TDA: System healthy",
    "confidence": 1.0,
    "metadata": {
      "abstention_rate": 0.15,
      "coverage_rate": 0.88,
      "verified_count": 12,
      "cycle_index": 8
    }
  },
  "context_keys": ["abstention_rate", "coverage_rate", ...],
  "decision_path": "BLOCK -> pass -> proceed"
}
```

## Testing

### Unit Tests (`tests/test_safety_slo.py`)

- SafetyEnvelope creation and serialization
- SafetySLO status handling
- All TDA modes (BLOCK, DRY_RUN, SHADOW)
- Blocking enforcement via `check_gate_decision()`
- Deterministic timestamp verification

### Integration Tests (`tests/test_cortex_integration.py`)

- RFLRunner integration scenarios
- U2Runner integration scenarios
- Mode transitions (BLOCK → DRY_RUN → SHADOW)
- Audit trail completeness
- Error handling

Run tests:
```bash
# Unit tests
python3 -m pytest tests/test_safety_slo.py -v

# Integration tests
python3 -m pytest tests/test_cortex_integration.py -v
```

## Usage Examples

### Basic Usage

```python
from backend.safety import evaluate_hard_gate_decision, check_gate_decision
from backend.tda import TDAMode

# Build context from runner state
context = {
    "abstention_rate": 0.15,
    "coverage_rate": 0.88,
    "verified_count": 12,
    "cycle_index": 8,
}

# Evaluate gate decision
envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)

# Check result
if envelope.is_blocking():
    print(f"BLOCKED: {envelope.slo.message}")
else:
    print(f"PROCEED: {envelope.slo.message}")

# Enforce blocking (raises RuntimeError if blocked)
check_gate_decision(envelope)
```

### Configuration

**Environment Variables** (optional overrides):
```bash
# For RFLRunner
export RFL_TDA_MODE="DRY_RUN"
export RFL_TDA_ENABLED="true"

# For U2Runner
export U2_TDA_MODE="SHADOW"
export U2_TDA_ENABLED="true"
```

**In Code**:
```python
from rfl.config import RFLConfig

config = RFLConfig(
    experiment_id="test_experiment",
    tda_mode="BLOCK",
    tda_enabled=True,
    # ... other config ...
)
```

## Observability

### Logging

The Cortex logs all decisions at INFO level:

```
[CORTEX] TDA Gate Decision: proceed (mode=BLOCK, status=pass)
[CORTEX] PASS: TDA: System healthy
```

For blocked decisions:
```
[CORTEX] TDA Gate Decision: block (mode=BLOCK, status=block)
[CORTEX] BLOCKED: TDA: Critical abstention rate 65.00% > 50%
```

### Metrics

RFLRunner records gate decisions in dual attestation records:

```python
runner.dual_attestation_records["gate_decisions"]
# [
#   {
#     "cycle": 1,
#     "composite_root": "abc...",
#     "envelope": {...}
#   },
#   ...
# ]
```

### Traces

U2Runner logs gate decisions to trace logger:

```python
EventType.DERIVE_SUCCESS  # Gate passed
EventType.DERIVE_FAILURE  # Gate blocked
```

## Future Enhancements

1. **Adaptive TDA**: Learn optimal thresholds from historical data
2. **Multi-Signal Fusion**: Incorporate additional metrics (latency, memory, etc.)
3. **Graduated Blocking**: WARN state that reduces but doesn't stop execution
4. **Custom Analyzers**: Plugin system for domain-specific TDA logic
5. **Cross-Runner Coordination**: Global organism health spanning both runners

## Security & Safety

The Cortex enforces critical safety invariants:

- **Fail-Safe**: Default mode is BLOCK (conservative)
- **Audit Trail**: Every decision is logged with full context
- **Deterministic**: Reproducible for security audits
- **Type-Safe**: Full typing throughout the stack
- **Tested**: Comprehensive unit and integration tests

## References

- Safety SLO Module: `backend/safety/`
- TDA Module: `backend/tda/`
- RFL Integration: `rfl/runner.py`
- U2 Integration: `experiments/u2/runner.py`
- Tests: `tests/test_safety_slo.py`, `tests/test_cortex_integration.py`

---

**STRATCOM: FIRST LIGHT COMPLETE.**

The Brain is connected. The Machine can run. The Organism is awake.

Δp + HSS traces ready for first integrated uplift run.
