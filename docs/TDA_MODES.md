# TDA Mode Reference

Quick reference for Topological Decision Analysis (TDA) operational modes.

## Mode Summary

| Mode | Purpose | Blocks Execution | Use Case |
|------|---------|-----------------|----------|
| **BLOCK** | Production | ✅ Yes | Live experiments, CI gates |
| **DRY_RUN** | Testing | ❌ No | Testing new TDA logic |
| **SHADOW** | Analysis | ❌ No | A/B testing, data collection |

## BLOCK Mode

**Default production mode.**

### Behavior
- When `should_block=True` → **Execution stops** with RuntimeError
- When `should_block=False` → Execution proceeds normally
- SLO Status: `PASS` or `BLOCK`

### Configuration
```python
# RFLRunner
config = RFLConfig(tda_mode="BLOCK", tda_enabled=True)

# U2Runner
config = U2Config(tda_mode="BLOCK", tda_enabled=True)
```

### Example
```python
context = {"abstention_rate": 0.7, "cycle_index": 10}
envelope = evaluate_hard_gate_decision(context, TDAMode.BLOCK)
# envelope.decision = "block"
# envelope.slo.status = SLOStatus.BLOCK

check_gate_decision(envelope)
# Raises: RuntimeError: Hard gate BLOCKED: ...
```

### Use When
- Running production experiments
- Enforcing quality gates in CI/CD
- Protecting against systemic failures

## DRY_RUN Mode

**Advisory mode for testing TDA changes.**

### Behavior
- Records what TDA **would** do without affecting execution
- Always proceeds regardless of `should_block` value
- SLO Status: `ADVISORY`
- Message includes "DRY_RUN: Would block" or "Would pass"

### Configuration
```python
config = RFLConfig(tda_mode="DRY_RUN", tda_enabled=True)
```

### Example
```python
context = {"abstention_rate": 0.7, "cycle_index": 10}  # Would normally block
envelope = evaluate_hard_gate_decision(context, TDAMode.DRY_RUN)
# envelope.decision = "proceed"  ← Always proceeds
# envelope.slo.status = SLOStatus.ADVISORY
# envelope.tda_should_block = True  ← But TDA says would block
# envelope.slo.message = "DRY_RUN: Would block - TDA: Critical abstention..."

check_gate_decision(envelope)
# No error - execution continues
```

### Use When
- Testing new TDA thresholds before deployment
- Debugging TDA decision logic
- Validating TDA behavior on historical data

## SHADOW Mode

**Hypothetical recording for A/B testing.**

### Behavior
- Records hypothetical decisions without affecting execution
- Always proceeds regardless of `should_block` value
- SLO Status: `ADVISORY`
- Message includes "SHADOW: Hypothetical block" or "Hypothetical pass"

### Configuration
```python
config = RFLConfig(tda_mode="SHADOW", tda_enabled=True)
```

### Example
```python
context = {"abstention_rate": 0.7, "cycle_index": 10}
envelope = evaluate_hard_gate_decision(context, TDAMode.SHADOW)
# envelope.decision = "proceed"  ← Always proceeds
# envelope.slo.status = SLOStatus.ADVISORY
# envelope.tda_should_block = True
# envelope.slo.message = "SHADOW: Hypothetical block - ..."

check_gate_decision(envelope)
# No error - execution continues
```

### Use When
- Running A/B tests of TDA versions
- Collecting decision data for analysis
- Comparing TDA algorithms side-by-side

## Comparison Table

| Aspect | BLOCK | DRY_RUN | SHADOW |
|--------|-------|---------|--------|
| **Execution** | Can block | Never blocks | Never blocks |
| **SLO Status** | PASS/BLOCK | ADVISORY | ADVISORY |
| **Decision** | proceed/block | always proceed | always proceed |
| **Message Prefix** | PASS:/BLOCKED: | DRY_RUN: Would... | SHADOW: Hypothetical... |
| **Audit Trail** | Full | Full | Full |
| **CI Safety** | ✅ Enforced | ⚠️ Advisory only | ⚠️ Advisory only |

## Mode Transitions

You can change modes between runs to test TDA behavior:

```python
# Run 1: Production mode
config.tda_mode = "BLOCK"
runner1 = RFLRunner(config)
# ... runner1 can be blocked ...

# Run 2: Test new threshold in DRY_RUN
config.tda_mode = "DRY_RUN"
runner2 = RFLRunner(config)
# ... runner2 never blocked, but records what would happen ...

# Run 3: A/B test SHADOW
config.tda_mode = "SHADOW"
runner3 = RFLRunner(config)
# ... runner3 records hypothetical decisions for analysis ...
```

## Decision Path Examples

### Healthy System (All Modes Pass)

```
Context: abstention_rate=0.1, coverage_rate=0.9, cycle=5

BLOCK Mode:
  TDA: should_block=False
  SLO: status=PASS, message="PASS: TDA: System healthy"
  Decision: proceed
  Effect: ✅ Execution continues

DRY_RUN Mode:
  TDA: should_block=False
  SLO: status=ADVISORY, message="DRY_RUN: Would pass - TDA: System healthy"
  Decision: proceed
  Effect: ✅ Execution continues

SHADOW Mode:
  TDA: should_block=False
  SLO: status=ADVISORY, message="SHADOW: Hypothetical pass - TDA: System healthy"
  Decision: proceed
  Effect: ✅ Execution continues
```

### Degraded System (Different Mode Behaviors)

```
Context: abstention_rate=0.7, coverage_rate=0.5, cycle=15

BLOCK Mode:
  TDA: should_block=True
  SLO: status=BLOCK, message="BLOCKED: TDA: Critical abstention rate 70.00% > 50%"
  Decision: block
  Effect: ❌ RuntimeError raised, execution stops

DRY_RUN Mode:
  TDA: should_block=True
  SLO: status=ADVISORY, message="DRY_RUN: Would block - TDA: Critical abstention..."
  Decision: proceed
  Effect: ✅ Execution continues (but decision recorded)

SHADOW Mode:
  TDA: should_block=True
  SLO: status=ADVISORY, message="SHADOW: Hypothetical block - TDA: Critical abstention..."
  Decision: proceed
  Effect: ✅ Execution continues (hypothetical recorded)
```

## Configuration Best Practices

### Development
```python
tda_mode = "DRY_RUN"    # Test without blocking
tda_enabled = True       # Keep gate active
```

### Testing/CI
```python
tda_mode = "BLOCK"       # Enforce gates
tda_enabled = True       # Required for safety
```

### Production
```python
tda_mode = "BLOCK"       # Enforce gates
tda_enabled = True       # Always enabled
```

### Analysis/Research
```python
tda_mode = "SHADOW"      # Collect decisions
tda_enabled = True       # Keep collecting data
```

## Disabling the Gate

To completely disable TDA (not recommended for production):

```python
tda_enabled = False  # Skips gate evaluation entirely
```

⚠️ **Warning**: Disabling TDA removes safety guarantees. Only use for debugging.

## Environment Variable Overrides

Override mode at runtime:

```bash
# RFLRunner
export RFL_TDA_MODE="DRY_RUN"

# U2Runner  
export U2_TDA_MODE="SHADOW"
```

## Audit Trail for All Modes

All modes produce complete audit trails for reproducibility:

```python
envelope.audit_trail = {
    "timestamp": "1970-01-01T00:00:00Z",
    "tda_decision": {
        "should_block": True,
        "mode": "DRY_RUN",
        "reason": "TDA: Critical abstention rate 70.00% > 50%",
        "confidence": 0.95,
        "metadata": {...}
    },
    "context_keys": ["abstention_rate", "coverage_rate", ...],
    "decision_path": "DRY_RUN -> advisory -> proceed"
}
```

This ensures **deterministic CI reproduction** across all modes.

## Further Reading

- [CORTEX_INTEGRATION.md](./CORTEX_INTEGRATION.md) - Complete integration guide
- `backend/safety/` - Safety SLO implementation
- `backend/tda/` - TDA decision logic
- `tests/test_cortex_integration.py` - Mode transition tests
