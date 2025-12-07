# U2 Safety SLO Engine

## Overview

The U2 Safety SLO Engine provides continuous, scenario-aware safety tracking for U2 experiments. It enforces policy-level Service Level Objectives (SLOs) with typed contracts, making it very hard for any new U2 code path to bypass type and safety checks.

## Architecture

### Core Components

1. **Safety Envelope** (`SafetyEnvelope`)
   - Captures single experiment run results
   - Includes performance metrics, lint issues, warnings
   - Tagged with slice, mode, timestamp, run_id

2. **Safety SLO Timeline** (`SafetySLOTimeline`)
   - Aggregates multiple runs into deterministic timeline
   - Computes rates: perf_ok_rate, lint_issue_rate
   - Provides status counts (OK/WARN/BLOCK)

3. **Scenario Safety Matrix** (`ScenarioSafetyMatrix`)
   - Groups runs by (slice_name, mode) scenarios
   - Tracks per-scenario metrics and worst status
   - Enables slice-level and mode-level analysis

4. **Safety SLO Evaluation** (`SafetySLOEvaluation`)
   - Evaluates overall safety status with configurable thresholds
   - Produces BLOCK/WARN/OK based on policy rules
   - Identifies failing scenarios and reasons

### Type Safety

All components are fully type-annotated and pass strict mypy checks:

```python
# Literal types for safety status
SafetyStatus = Literal["OK", "WARN", "BLOCK"]

# Frozen dataclasses for immutability
@dataclass(frozen=True)
class SafetySLOPoint:
    run_id: str
    slice_name: str
    mode: Literal["baseline", "rfl"]
    safety_status: SafetyStatus
    # ...
```

## Usage

### 1. Running an Experiment with Safety Envelope

```python
from experiments.u2 import (
    U2Config,
    U2SafetyContext,
    run_u2_experiment,
)

# Create configuration
config = U2Config(
    experiment_id="experiment_001",
    slice_name="slice_hard",
    mode="rfl",
    total_cycles=100,
    master_seed=42,
)

# Create safety context
safety_ctx = U2SafetyContext(
    config=config,
    perf_threshold_ms=1000.0,
    max_cycles=100,
    enable_safe_eval=True,
    slice_name="slice_hard",
    mode="rfl",
)

# Run experiment
def execute_candidate(item: str, cycle: int) -> tuple[bool, Any]:
    # Your execution logic
    return (True, {"result": "success"})

envelope = run_u2_experiment(safety_ctx, execute_candidate)

# envelope is a SafetyEnvelope with:
# - safety_status: "OK" | "WARN" | "BLOCK"
# - perf_ok: bool
# - lint_issues: List[str]
# - warnings: List[str]
```

### 2. Building a Safety Timeline

```python
from experiments.u2.safety_slo import build_safety_slo_timeline

# Collect envelopes from multiple runs
envelopes: List[SafetyEnvelope] = [
    # ... your envelopes
]

# Build timeline
timeline = build_safety_slo_timeline(envelopes)

print(f"Total runs: {len(timeline.points)}")
print(f"Performance OK rate: {timeline.perf_ok_rate:.1%}")
print(f"Status counts: {timeline.status_counts}")
```

### 3. Building a Scenario Matrix

```python
from experiments.u2.safety_slo import build_scenario_safety_matrix

# Build matrix from timeline
matrix = build_scenario_safety_matrix(timeline)

print(f"Total scenarios: {len(matrix.cells)}")
for cell in matrix.cells:
    print(f"  {cell.slice_name}:{cell.mode}")
    print(f"    Runs: {cell.runs}")
    print(f"    OK/WARN/BLOCK: {cell.ok_runs}/{cell.warn_runs}/{cell.blocked_runs}")
    print(f"    Worst status: {cell.worst_status}")
```

### 4. Evaluating Safety SLO

```python
from experiments.u2.safety_slo import evaluate_safety_slo

# Evaluate SLO with default thresholds
evaluation = evaluate_safety_slo(matrix)

print(f"Overall status: {evaluation.overall_status}")
print(f"Failing scenarios: {evaluation.failing_scenarios}")
print("Reasons:")
for reason in evaluation.reasons:
    print(f"  - {reason}")

# Exit with error if BLOCK
if evaluation.overall_status == "BLOCK":
    sys.exit(1)
```

## SLO Thresholds

Default thresholds (configurable):

```python
MAX_BLOCK_RATE = 0.05       # ≤ 5% BLOCK across all runs
MAX_WARN_RATE = 0.20        # ≤ 20% WARN across all runs
MAX_PERF_FAILURE_RATE = 0.10  # ≤ 10% performance failures
```

### Evaluation Logic

**BLOCK** status if:
- Any scenario has `blocked_runs / runs > MAX_BLOCK_RATE`, OR
- Global BLOCK rate > `MAX_BLOCK_RATE`

**WARN** status if (and not BLOCK):
- Global WARN rate > `MAX_WARN_RATE`, OR
- Performance failure rate > `MAX_PERF_FAILURE_RATE`

**OK** status otherwise.

## Type-Safe Wrappers

### Safe Expression Evaluation

Never uses `eval()` directly:

```python
from experiments.u2 import safe_eval_expression

# Only evaluates numeric literals
result = safe_eval_expression("123.45")  # OK
result = safe_eval_expression("1e6")     # OK

# Raises ValueError for non-numeric input
safe_eval_expression("1 + 2")      # Error - no expressions
safe_eval_expression("import os")  # Error - no code
```

### Type-Safe Snapshots

```python
from experiments.u2 import (
    U2Snapshot,
    save_u2_snapshot,
    load_u2_snapshot,
)

# Save typed snapshot
snapshot = U2Snapshot(
    config=config,
    cycles_completed=50,
    state_hash="abc123...",
    snapshot_data=snapshot_data,
)
hash = save_u2_snapshot(path, snapshot)

# Load typed snapshot
loaded = load_u2_snapshot(path)
assert loaded.config.experiment_id == "experiment_001"
```

## CI Integration

The U2 Safety Gate workflow (`.github/workflows/u2-safety-gate.yml`) runs on every PR:

1. **Type Safety Checks**
   - Runs mypy with `--strict` on safety modules
   - Ensures no type regressions

2. **Unit Tests**
   - 42 comprehensive tests
   - Covers all SLO functions
   - Tests determinism, type contracts, evaluation logic

3. **Safety SLO Evaluation** (optional)
   - If `artifacts/u2_safety_evaluation.json` exists
   - Evaluates SLO and fails CI if status is BLOCK

## Determinism Guarantees

All operations are deterministic:

- **Timeline building**: Sorts by `(timestamp, run_id)`
- **Matrix building**: Sorts cells by `(slice_name, mode)`
- **Hash computation**: Uses canonical JSON with sorted keys

Same inputs always produce identical outputs, regardless of:
- Input order
- Machine/OS
- Python version (3.11+)

## Testing

Run tests:

```bash
# All U2 tests
pytest tests/experiments/u2/ -v

# Just safety SLO tests
pytest tests/experiments/u2/test_safety_slo.py -v

# Just type contract tests
pytest tests/experiments/u2/test_mypy_contracts.py -v
```

## Files

- `experiments/u2/safety_slo.py` - Core SLO engine
- `experiments/u2/runner.py` - Updated with U2SafetyContext and run_u2_experiment
- `tests/experiments/u2/test_safety_slo.py` - SLO engine tests
- `tests/experiments/u2/test_safety_context.py` - Safety context tests
- `tests/experiments/u2/test_mypy_contracts.py` - Type safety tests
- `.github/workflows/u2-safety-gate.yml` - CI workflow

## Design Principles

1. **Type Safety First**: All public APIs are fully typed with strict mypy compliance
2. **Immutability**: Critical dataclasses are frozen to prevent accidental mutation
3. **Determinism**: All aggregations use stable sorting for reproducibility
4. **Fail-Safe**: Safety violations fail CI, not just warn
5. **Composability**: Each component has single responsibility and clear interface

## Future Enhancements

Potential improvements:

- [ ] Persist timeline to database for historical analysis
- [ ] Dashboard for visualizing scenario matrix over time
- [ ] Alerting on SLO threshold breaches
- [ ] Per-slice custom thresholds
- [ ] Integration with experiment planning system
