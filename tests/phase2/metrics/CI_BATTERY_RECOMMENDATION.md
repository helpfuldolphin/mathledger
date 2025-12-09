# Phase II Metrics Test Battery - CI Recommendation

## Overview

This document provides recommendations for integrating the Phase II Metrics Test Battery into CI workflows.

## Test Battery Statistics

| Metric | Value |
|--------|-------|
| **Total Tests** | 251 |
| **Test Files** | 6 |
| **Deterministic** | Yes (all tests) |
| **External Dependencies** | None |
| **Database Required** | No |
| **Network Required** | No |

## Test Files

| File | Tests | Focus Area |
|------|-------|------------|
| `test_goal_hit.py` | ~40 | Goal hit metric boundary/degenerate cases |
| `test_sparse_density.py` | ~40 | Sparse/density metric validation |
| `test_chain_length.py` | ~45 | Chain length with complex DAGs |
| `test_multi_goal.py` | ~40 | Multi-goal success semantics |
| `test_metric_adapter.py` | ~35 | Adapter routing and type stability |
| `test_metric_integration_replay.py` | ~50 | Cross-function replay determinism |

## Pytest Markers

The test battery uses the following custom markers:

```ini
markers =
    phase2_metrics: marks tests as Phase II metrics battery
    boundary: marks tests for boundary conditions
    degenerate: marks tests for degenerate/edge cases
    large_scale: marks tests with large data volumes
    determinism: marks tests verifying deterministic behavior
    schema: marks tests for schema validation
    cross_slice: marks tests across multiple slice configs
    replay: marks tests verifying replay equivalence
    type_stability: marks tests verifying return types
```

## CI Integration

### Recommended Pytest Command

```bash
# Full battery (recommended for PRs touching metrics)
pytest tests/phase2/metrics/ -v --tb=short -m "phase2_metrics"

# Quick smoke test (for general PRs)
pytest tests/phase2/metrics/ -v --tb=short -m "phase2_metrics and not large_scale"

# Replay determinism verification (for pre-release)
pytest tests/phase2/metrics/ -v --tb=short -m "replay"
```

### pytest.ini Addition

Add to `pytest.ini`:

```ini
[pytest]
markers =
    # ... existing markers ...
    phase2_metrics: marks tests as Phase II metrics battery
    boundary: marks tests for boundary conditions
    degenerate: marks tests for degenerate/edge cases
    large_scale: marks tests with large data volumes
    schema: marks tests for schema validation
    cross_slice: marks tests across multiple slice configs
    replay: marks tests verifying replay equivalence
    type_stability: marks tests verifying return types
```

## Resource Estimates

### Runtime Estimates

| Marker Filter | Approximate Runtime | Tests |
|---------------|---------------------|-------|
| All tests | 5-10 seconds | 251 |
| `-m "not large_scale"` | 3-5 seconds | ~200 |
| `-m "replay"` | 2-3 seconds | ~30 |
| `-m "determinism"` | 1-2 seconds | ~20 |
| `-m "boundary"` | 1-2 seconds | ~25 |

### Memory Usage

- Peak memory: < 100 MB
- No external processes spawned
- No file I/O (except pytest internals)

### CPU Usage

- Single-threaded execution recommended
- Parallel execution supported (`-n auto`) but not required
- No GPU usage

## Test Coverage

The test battery provides coverage for:

### experiments/slice_success_metrics.py

| Function | Coverage |
|----------|----------|
| `compute_goal_hit` | Full (boundary, degenerate, large-scale) |
| `compute_sparse_success` | Full (boundary, degenerate, large-scale) |
| `compute_chain_success` | Full (DAG, cycles, deep chains) |
| `compute_multi_goal_success` | Full (set operations, large sets) |

### Coverage Goals

- **Statement Coverage**: > 95%
- **Branch Coverage**: > 90%
- **Mutation Testing**: Recommended for future validation

## Determinism Guarantees

All tests provide the following guarantees:

1. **PRNG Seeds**: All random operations use fixed seeds from `conftest.py`
2. **No Network**: No external API calls or network dependencies
3. **No Time-Dependency**: No `time.time()` or datetime-based logic
4. **No File System**: No temp file creation or file reads
5. **Idempotent**: Multiple runs produce identical results

## Failure Modes

### Expected Failures

Tests should **never** flake under normal conditions. If a test fails:

1. Check for unintended non-determinism in production code
2. Verify PRNG seeds haven't been modified in conftest.py
3. Check for floating-point precision issues (use `pytest.approx`)

### Not Covered

These scenarios are explicitly **not tested** by this battery:

- Database integration
- Redis/cache integration
- Network latency
- Concurrent execution correctness
- Performance benchmarks (use `pytest-benchmark` separately)

## CI Workflow Example

```yaml
# .github/workflows/metrics-battery.yml
name: Phase II Metrics Battery

on:
  pull_request:
    paths:
      - 'experiments/slice_success_metrics.py'
      - 'tests/phase2/metrics/**'

jobs:
  metrics-battery:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv sync
      - name: Run metrics battery
        run: |
          uv run pytest tests/phase2/metrics/ \
            -v --tb=short \
            -m "phase2_metrics" \
            --junitxml=metrics-battery-results.xml
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: metrics-battery-results
          path: metrics-battery-results.xml
```

## Maintenance

### Adding New Tests

1. Use appropriate markers (`@pytest.mark.phase2_metrics` required)
2. Use `DeterministicGenerator` from conftest.py for random data
3. Assert type stability using helper functions
4. Document any new PRNG seeds in conftest.py

### Updating Existing Tests

1. Do not change PRNG seed values
2. Preserve determinism guarantees
3. Update this document if test counts change significantly

---

**Generated by**: Agent `metrics-engineer-2`  
**Test Count**: 251 tests  
**Last Updated**: Phase II Battery Creation

