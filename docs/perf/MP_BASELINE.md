# Modus Ponens Performance Baseline

## Overview

This document describes the performance baseline system for the O(n) Modus Ponens optimization, including methodology, CI integration, and update procedures.

## Baseline Data Source

Performance baselines are generated from synthetic datasets using `tools/perf/benchmarks.py`:

- **100 atoms**: 50 atoms + 50 implications (p1, p1->q1, p2, p2->q2, ...)
- **1K atoms**: 500 atoms + 500 implications  
- **10K atoms**: 5000 atoms + 5000 implications

Each benchmark runs multiple iterations and records average wall-time, maximum time, and derivation count.

## Current Baselines

| Dataset Size | Avg Time (ms) | Max Time (ms) | Derivations | Algorithm |
|--------------|---------------|---------------|-------------|-----------|
| 100 atoms    | 0.50          | 1.0           | 50          | O(n)      |
| 1K atoms     | 131.92        | 200.0         | 500         | O(n)      |
| 10K atoms    | 2068.80       | 3000.0        | 5000        | O(n)      |

## Methodology

### Baseline Generation
```bash
# Generate new baseline (run on clean main/integration branch)
python tools/perf/benchmarks.py baseline
```

This creates `artifacts/perf/baseline.csv` with format:
```csv
dataset_size,avg_time_ms,max_time_ms,derivations,timestamp
100,0.5040,1.2000,50,2024-10-02T10:30:00
1000,131.9178,200.0000,500,2024-10-02T10:30:00
10000,2068.8017,3000.0000,5000,2024-10-02T10:30:00
```

### CI Regression Detection

The CI pipeline runs `tools/perf/check_regression.py` on every PR to `integrate/ledger-v0.1`:

1. Loads baseline from `artifacts/perf/baseline.csv`
2. Runs current benchmarks
3. Compares 1K atom performance (critical threshold)
4. **FAILS PR if >10% regression** at 1K atoms

### Parity Validation

`tools/perf/parity_test.py` proves functional equivalence:

- Small datasets: Direct comparison with legacy O(n²) implementation
- Large datasets: Quadratic extrapolation (legacy times out for 10K+ atoms)
- Scaling analysis: Confirms O(n) vs O(n²) complexity difference

## Update Procedures

### When to Update Baselines

- **Algorithmic improvements**: New optimization that legitimately improves performance
- **Infrastructure changes**: Hardware upgrades, compiler optimizations
- **Major refactoring**: Significant changes to core engine architecture

### How to Update Baselines

1. **Verify improvement is legitimate**:
   ```bash
   python tools/perf/parity_test.py  # Ensure correctness maintained
   python tools/perf/benchmarks.py run --target modus_ponens
   ```

2. **Generate new baseline**:
   ```bash
   python tools/perf/benchmarks.py baseline
   git add artifacts/perf/baseline.csv
   git commit -m "perf: update MP baseline after [improvement description]"
   ```

3. **Document the change**:
   - Update this README with new baseline numbers
   - Include justification for the improvement
   - Reference the PR/commit that delivered the optimization

## Monitoring and Alerts

### CI Integration

The performance gate is integrated into `.github/workflows/ci.yml`:

```yaml
- name: Performance regression gate (Modus Ponens <10%)
  run: python tools/perf/check_regression.py
```

### Local Testing

Before submitting PRs that touch `backend/axiom_engine/rules.py`:

```bash
# Quick performance check
python tools/perf/benchmarks.py run --target modus_ponens

# Full regression check
python tools/perf/check_regression.py

# Parity validation
python tools/perf/parity_test.py
```

## Troubleshooting

### "Baseline file not found"
```bash
python tools/perf/benchmarks.py baseline
```

### "Performance regression detected"
1. Verify the regression is real (run locally)
2. If legitimate regression, investigate the cause
3. If false positive, check for environmental factors
4. Consider updating baseline if improvement is intentional

### "Parity test failed"
This indicates a correctness bug in the optimization. **Do not merge** until functional equivalence is restored.
