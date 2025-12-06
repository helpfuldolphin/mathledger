# Mypy CI Guide - U2 Type Safety Gate

## Overview

This guide documents the type safety and CI integration for the U2 experiment framework (Phase III). The U2 modules are under **strict type checking** to ensure type safety, prevent runtime errors, and maintain a verifiable safety envelope.

## Type Checking Configuration

### Strict Type Checking for U2 Modules

The `experiments.u2.*` modules are configured with strict mypy settings in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = "experiments.u2.*"
disallow_untyped_defs = true
disallow_any_generics = true
disallow_subclassing_any = true
disallow_untyped_calls = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
no_implicit_reexport = true
strict_equality = true
```

These settings enforce:
- All functions must have type annotations
- No implicit `Any` types
- Strict checking of generics
- No untyped function calls
- No implicit re-exports

## Running Type Checks Locally

### Install mypy

```bash
pip install mypy>=1.8.0
# or with uv
uv pip install mypy>=1.8.0
```

### Check U2 Modules

To check only the U2 modules:

```bash
mypy experiments/u2/
```

### Check Entire Project

To check the entire project:

```bash
mypy .
```

### Expected Output

A successful type check will produce no errors:

```
Success: no issues found in 8 source files
```

If there are type errors, they must be fixed before merging.

## CI Integration

### Safety Gate Pattern

The CI safety gate ensures that:
1. All U2 modules pass strict type checking
2. No new type errors are introduced
3. Safety envelope indicates status is not "BLOCK"
4. Performance guardrails are met

### CI Workflow Example

Create `.github/workflows/u2-safety-gate.yml`:

```yaml
name: U2 Safety Gate

on:
  pull_request:
    paths:
      - 'experiments/u2/**'
      - 'tests/test_u2*.py'
  push:
    branches:
      - main
      - 'copilot/**'

jobs:
  type-check:
    name: Type Safety Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install mypy pytest pyyaml pydantic
      
      - name: Run mypy on U2 modules
        run: |
          mypy experiments/u2/ --strict
      
      - name: Fail if mypy errors
        if: failure()
        run: |
          echo "::error::Type checking failed for U2 modules"
          exit 1

  safety-tests:
    name: Safety & Performance Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pytest pyyaml pydantic
      
      - name: Run U2 tests
        run: |
          pytest tests/test_u2*.py -v
      
      - name: Run performance guardrail tests
        run: |
          pytest tests/test_u2_perf_guardrails.py -v
      
      - name: Check safety envelope
        run: |
          python3 -c "
          from experiments.u2.entrypoint import run_u2_experiment
          from experiments.u2.runner import U2Config
          from experiments.u2.safety_envelope import evaluate_safety_status
          
          # Quick smoke test
          config = U2Config(
              experiment_id='ci_smoke',
              slice_name='test',
              mode='baseline',
              total_cycles=5,
              master_seed=42
          )
          
          def mock_exec(item, seed):
              return True, {'ok': True}
          
          results, envelope = run_u2_experiment(
              config=config,
              items=['a', 'b', 'c'],
              execute_fn=mock_exec
          )
          
          if not evaluate_safety_status(envelope):
              raise RuntimeError(f'Safety envelope BLOCKED: {envelope.warnings}')
          
          print(f'✓ Safety status: {envelope.safety_status}')
          "
```

### Manual CI Check

To manually verify CI compliance before pushing:

```bash
# 1. Run type checks
mypy experiments/u2/

# 2. Run tests
pytest tests/test_u2*.py -v

# 3. Run performance tests (if available)
pytest tests/test_u2_perf_guardrails.py -v
```

## Type Safety Best Practices

### 1. Always Annotate Function Signatures

```python
# ✓ Good
def process_cycle(item: str, seed: int) -> Tuple[bool, Dict[str, Any]]:
    return True, {"result": "ok"}

# ✗ Bad
def process_cycle(item, seed):
    return True, {"result": "ok"}
```

### 2. Use Type Aliases for Complex Types

```python
from typing import TypeAlias

ExecuteFunction: TypeAlias = Callable[[str, int], Tuple[bool, Any]]

def run_cycle(execute_fn: ExecuteFunction) -> CycleResult:
    ...
```

### 3. Avoid `Any` Types

```python
# ✓ Good
def process_result(result: Dict[str, int]) -> int:
    return sum(result.values())

# ✗ Bad
def process_result(result: Any) -> Any:
    return sum(result.values())
```

### 4. Use Dataclasses for Structured Data

```python
from dataclasses import dataclass

@dataclass
class Config:
    experiment_id: str
    total_cycles: int
    seed: int
```

## Safety Envelope Integration

The safety envelope is checked in CI to ensure experiments are safe to run:

### Safety Status Levels

- **OK**: All checks pass, safe to proceed
- **WARN**: Some issues detected, review recommended
- **BLOCK**: Critical issues, must not proceed

### CI Safety Check Example

```python
from experiments.u2.safety_envelope import build_u2_safety_envelope, evaluate_safety_status

# After running experiment
envelope = build_u2_safety_envelope(config, perf_stats, lint_results)

# CI fails if status is BLOCK
if envelope.safety_status == "BLOCK":
    raise RuntimeError(f"Safety envelope BLOCKED: {envelope.warnings}")

# CI warns if status is WARN
if envelope.safety_status == "WARN":
    print(f"WARNING: Safety envelope issues: {envelope.warnings}")
```

## Performance Guardrails

Performance guardrails ensure experiments complete within acceptable time bounds:

### Default Thresholds

- **Max cycle duration**: 5000ms (5 seconds)
- **Avg cycle duration**: 2000ms (2 seconds)
- **Max eval lint issues**: 10

### Custom Thresholds

```python
custom_thresholds = {
    "max_cycle_duration_ms": 3000.0,
    "max_avg_cycle_duration_ms": 1500.0,
    "max_eval_lint_issues": 5,
}

envelope = build_u2_safety_envelope(
    config,
    perf_stats,
    lint_results,
    perf_thresholds=custom_thresholds,
)
```

## Troubleshooting

### Common Type Errors

#### Missing Return Type

```
error: Function is missing a return type annotation
```

**Fix**: Add return type annotation:
```python
def my_function() -> int:
    return 42
```

#### Incompatible Types

```
error: Argument 1 has incompatible type "str"; expected "int"
```

**Fix**: Ensure argument types match the signature:
```python
my_function(42)  # Not my_function("42")
```

#### `Any` Not Allowed

```
error: Returning Any from function declared to return "int"
```

**Fix**: Add explicit type annotations:
```python
def process(data: Dict[str, int]) -> int:  # Not Dict[str, Any]
    return data["value"]
```

### Mypy Cache Issues

If mypy reports stale errors:

```bash
# Clear mypy cache
rm -rf .mypy_cache/

# Re-run mypy
mypy experiments/u2/
```

## Maintenance

### Adding New U2 Modules

When adding new modules to `experiments/u2/`:

1. Ensure all functions have type annotations
2. Run mypy locally: `mypy experiments/u2/`
3. Fix any type errors before committing
4. Add tests for new functionality
5. Update this guide if introducing new patterns

### Updating Type Checking Rules

To update mypy configuration:

1. Edit `[tool.mypy]` section in `pyproject.toml`
2. Run mypy on all affected modules
3. Fix any new errors that surface
4. Document changes in this guide

## References

- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [U2 Safety Envelope Spec](./U2_PORT_PLAN.md)
- [Python Type Checking Guide](https://realpython.com/python-type-checking/)

## Contact

For questions about U2 type safety:
- Review the agent instructions in `.github/agents/sober-refactor.md`
- Check existing U2 modules for examples
- Refer to test files for usage patterns
