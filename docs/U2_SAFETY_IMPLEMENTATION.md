# U2 Safety Envelope & Type-Verified Runner - Implementation Summary

## Overview

This document summarizes the Phase III U2 Safety Envelope and Type-Verified Runner implementation completed by the `sober-refactor` agent.

## What Was Delivered

### 1. U2 Safety Envelope Contract ✅

**Module:** `experiments/u2/safety_envelope.py`

Implemented `build_u2_safety_envelope()` that creates a comprehensive safety contract:

```python
envelope = build_u2_safety_envelope(
    run_config=config,
    perf_stats=perf_stats,
    eval_lint_results=lint_results,
)
# Returns: U2SafetyEnvelope with safety_status: "OK" | "WARN" | "BLOCK"
```

**Features:**
- Schema-versioned safety contract (v1.0.0)
- Safe subset of config (no secrets)
- Performance validation against thresholds
- Evaluation lint issue tracking
- Automatic status determination (OK/WARN/BLOCK)
- Detailed warnings and top issues

**Default Thresholds:**
- Max cycle duration: 5000ms
- Avg cycle duration: 2000ms
- Max eval lint issues: 10

### 2. Type-Verified Runner Entry Surface ✅

**Module:** `experiments/u2/entrypoint.py`

Single, well-documented entry point:

```python
def run_u2_experiment(
    config: U2Config,
    items: List[str],
    execute_fn: Callable[[str, int], Tuple[bool, Any]],
    lint_expressions: Optional[List[str]] = None,
) -> Tuple[List[CycleResult], U2SafetyEnvelope]:
    """
    Run a U2 experiment with type verification and safety monitoring.
    """
```

**Enforces:**
- Complete type annotations everywhere
- No implicit `Any` types
- Strict generic type checking
- Full mypy compliance

**Integrates:**
- U2Runner for cycle execution
- Safe eval for expression linting
- Snapshot system for pause/resume
- Structured logging with event filtering

### 3. CI Type & Safety Gate ✅

**Configuration:** `pyproject.toml`

Strict mypy settings for U2 modules:

```toml
[[tool.mypy.overrides]]
module = "experiments.u2.*"
disallow_untyped_defs = true
disallow_any_generics = true
disallow_untyped_calls = true
# ... more strict checks
```

**Documentation:** `docs/MYPY_CI_GUIDE.md` (8KB comprehensive guide)

**CI Workflow:** `.github/workflows/u2-safety-gate.yml`

Includes:
- Mypy type checking job
- Safety & performance test job
- Safety envelope smoke test
- Fails if status = "BLOCK"

## Architecture

### Module Structure

```
experiments/u2/
├── __init__.py          # Package exports
├── u2_safe_eval.py      # Safe evaluation with lint mode
├── runner.py            # Type-verified U2Runner
├── safety_envelope.py   # Safety contract builder
├── snapshots.py         # State capture/restore
├── logging.py           # Structured trace logging
├── schema.py            # Pydantic event schemas
└── entrypoint.py        # Main entry point
```

### Key Classes

**U2Config** - Typed configuration
```python
@dataclass
class U2Config:
    experiment_id: str
    slice_name: str
    mode: str  # "baseline" or "rfl"
    total_cycles: int
    master_seed: int
    snapshot_interval: int = 0
    ...
```

**CycleResult** - Typed execution result
```python
@dataclass
class CycleResult:
    cycle_index: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: Any
    success: bool
    duration_ms: Optional[float] = None
    ...
```

**U2SafetyEnvelope** - Safety contract
```python
@dataclass
class U2SafetyEnvelope:
    schema_version: str
    config: Dict[str, Any]
    perf_ok: bool
    eval_lint_issues: int
    safety_status: SafetyStatus  # "OK" | "WARN" | "BLOCK"
    perf_stats: Dict[str, Any]
    top_eval_issues: List[str]
    warnings: List[str]
```

## Testing

**67 tests, 100% passing**

### Test Coverage

1. **Safe Evaluation** (19 tests)
   - Arithmetic and comparison expressions
   - Dangerous operation detection (imports, calls, attributes)
   - Safe builtin function calls
   - Variable name restrictions
   - Batch linting

2. **Runner & Safety Envelope** (17 tests)
   - Config validation
   - Baseline and RFL modes
   - Multi-cycle execution
   - Safety envelope with OK/WARN/BLOCK status
   - Performance threshold detection
   - Entry point integration

3. **Snapshots** (21 tests)
   - Save/load/roundtrip
   - Hash verification
   - Corruption detection
   - Latest snapshot discovery
   - Rotation policy

4. **Performance Guardrails** (10 tests)
   - Fast cycle validation
   - Slow cycle warnings
   - Duration tracking
   - Custom threshold support
   - Performance benchmarks
   - Regression detection

## Usage Examples

### Basic Experiment

```python
from experiments.u2.entrypoint import run_u2_experiment
from experiments.u2.runner import U2Config

config = U2Config(
    experiment_id="my_exp",
    slice_name="arithmetic",
    mode="baseline",
    total_cycles=100,
    master_seed=42,
)

def execute(item: str, seed: int) -> Tuple[bool, Any]:
    # Your execution logic
    return True, {"outcome": "VERIFIED"}

results, envelope = run_u2_experiment(
    config=config,
    items=["1+1", "2+2", "3+3"],
    execute_fn=execute,
)

if envelope.safety_status == "BLOCK":
    raise RuntimeError(f"Safety blocked: {envelope.warnings}")

print(f"Completed {len(results)} cycles")
print(f"Safety status: {envelope.safety_status}")
```

### With Expression Linting

```python
results, envelope = run_u2_experiment(
    config=config,
    items=["a", "b", "c"],
    execute_fn=execute,
    lint_expressions=["1+1", "2*3", "max(1,2)"],
)

if envelope.eval_lint_issues > 0:
    print(f"Lint issues: {envelope.top_eval_issues}")
```

### With Snapshots

```python
config = U2Config(
    experiment_id="long_exp",
    slice_name="test",
    mode="rfl",
    total_cycles=1000,
    master_seed=42,
    snapshot_interval=100,  # Save every 100 cycles
    snapshot_dir=Path("./snapshots"),
)

results, envelope = run_u2_experiment(config, items, execute)
```

## Type Safety

All U2 modules pass strict mypy checking:

```bash
$ mypy experiments/u2/ --config-file pyproject.toml
Success: no issues found in 8 source files
```

**Enforced rules:**
- All functions must have type annotations
- No implicit `Any` types
- No untyped function calls
- Strict generic checking
- No implicit re-exports

## Performance

**Benchmark results:**

- 100 baseline cycles: < 1.0 second
- 50 RFL cycles: < 0.5 seconds
- Avg cycle time: < 10ms
- Test suite: < 1 second (67 tests)

**Performance monitoring:**
- Automatic duration tracking per cycle
- Configurable performance thresholds
- Warning generation on threshold breach
- Performance stats in safety envelope

## Integration Points

### With Existing Code

The U2 infrastructure is designed to integrate with `experiments/run_uplift_u2.py`:

```python
from experiments.u2.runner import U2Runner, U2Config, CycleResult
from experiments.u2.snapshots import load_snapshot, save_snapshot
from experiments.u2.logging import U2TraceLogger
from experiments.u2 import schema as trace_schema
```

### Safety Gate in CI

The safety gate automatically runs on:
- Pull requests affecting `experiments/u2/**`
- Pushes to `main` and `copilot/**` branches

**Gate checks:**
1. Mypy type checking (must pass)
2. All U2 tests (must pass)
3. Performance guardrails (must pass)
4. Safety envelope smoke test (status ≠ BLOCK)

## Security Features

### Safe Evaluation

**AST-based static analysis:**
- Blocks imports, function definitions, attribute access
- Allows only safe arithmetic and comparisons
- Whitelist for safe builtin functions
- Detects dangerous operations before execution

**Example:**
```python
from experiments.u2.u2_safe_eval import lint_expression

result = lint_expression("import os")
# result.is_safe = False
# result.issues = ["Dangerous operation: Import"]

result = lint_expression("1 + 1")
# result.is_safe = True
# result.issues = []
```

### Safety Envelope

Automatically evaluates experiment safety:

```python
if envelope.safety_status == "BLOCK":
    # Critical issues detected - do not proceed
elif envelope.safety_status == "WARN":
    # Issues detected - review recommended
else:  # "OK"
    # All checks passed
```

## Files Modified/Created

**Created:**
- `experiments/u2/__init__.py`
- `experiments/u2/u2_safe_eval.py`
- `experiments/u2/runner.py`
- `experiments/u2/safety_envelope.py`
- `experiments/u2/snapshots.py`
- `experiments/u2/logging.py`
- `experiments/u2/schema.py`
- `experiments/u2/entrypoint.py`
- `tests/test_u2_safe_eval.py`
- `tests/test_u2_runner_safety.py`
- `tests/test_u2_snapshots.py`
- `tests/test_u2_perf_guardrails.py`
- `docs/MYPY_CI_GUIDE.md`
- `.github/workflows/u2-safety-gate.yml`

**Modified:**
- `pyproject.toml` (added mypy config and dev dependency)

## Next Steps

The U2 safety infrastructure is complete and ready for use:

1. ✅ All 67 tests passing
2. ✅ Full mypy compliance
3. ✅ CI workflow configured
4. ✅ Documentation complete
5. ✅ Performance benchmarked

**Ready for:**
- Integration with existing U2 experiments
- Additional safety policies as needed
- Extension to other experiment frameworks
- Production deployment

## References

- [MYPY_CI_GUIDE.md](./MYPY_CI_GUIDE.md) - Complete type safety guide
- [U2_PORT_PLAN.md](./U2_PORT_PLAN.md) - Original port plan
- Test files - Usage examples and patterns

## Contact

For questions about this implementation:
- Review the agent instructions in `.github/agents/sober-refactor.md`
- Check test files for usage examples
- Refer to MYPY_CI_GUIDE.md for type safety patterns
