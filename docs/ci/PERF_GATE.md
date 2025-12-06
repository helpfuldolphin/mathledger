# Performance Regression Gate

## Overview
The performance regression gate ensures that Modus Ponens performance doesn't regress by more than 10% compared to baseline measurements.

## Components
- `tools/perf/benchmarks.py`: Performance harness that measures Modus Ponens execution time
- `tools/perf/check_regression.py`: CI script that compares current performance to baseline
- `artifacts/perf/baseline.csv`: Baseline performance measurements

## How It Works
1. Loads baseline performance data from `artifacts/perf/baseline.csv`
2. Runs current benchmarks using `benchmarks.py`
3. Extracts 1K atom performance from benchmark output
4. Calculates regression percentage: `((current - baseline) / baseline) * 100`
5. Fails if regression > 10%

## Inputs
- Baseline CSV with columns: dataset_size, avg_time_ms, max_time_ms, derivations, timestamp
- Current codebase with axiom_engine.rules module

## Outputs
- Performance comparison report
- Exit code 0 (pass) or 1 (fail)

## Fail Modes
- Missing baseline.csv file
- Import errors for axiom_engine.rules
- Benchmark subprocess failure
- Unable to parse benchmark output
- Performance regression > 10%

## Environment Requirements
- PYTHONPATH must include backend directory for axiom_engine imports
- artifacts/perf directory must exist
- No network dependencies (uses synthetic datasets)

## Troubleshooting
- Check that benchmarks.py runs locally: `python tools/perf/benchmarks.py run --target modus_ponens`
- Verify baseline exists: `ls -la artifacts/perf/baseline.csv`
- Test regression check: `python tools/perf/check_regression.py`

## Technical Details
The performance gate uses subprocess calls to run benchmarks.py and parse its output. Since unittest.TextTestRunner sends output to stderr by default, the regression checker captures both stdout and stderr to ensure reliable output parsing in CI environments.

## CI Integration
The performance gate runs as part of the CI workflow in `.github/workflows/ci.yml`:
```yaml
- name: Performance regression gate (Modus Ponens <10%)
  run: |
    python tools/perf/check_regression.py
```

## Fork Safety
The performance gate is fork-safe and requires no secrets or external services. All benchmarks use synthetic datasets and mock data structures.
