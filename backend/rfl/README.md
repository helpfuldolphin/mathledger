# Reflexive Formal Learning (RFL) Framework

**Quantifying MathLedger's Reflexive Metabolism**

The RFL framework measures how MathLedger learns from abstentions through automated 40-run experiments, bootstrap confidence intervals, and empirical evidence curves.

## Mission

Produce empirical evidence that MathLedger exhibits **reflexive metabolism**: self-learning capacity that improves proof generation through systematic exploration and validation.

### Acceptance Criteria

- **Coverage ≥ 92%**: Bootstrap CI lower bound on statement coverage
- **Uplift > 1.0**: Bootstrap CI lower bound on throughput improvement

Verification format:
```
[PASS] Reflexive Metabolism Verified coverage≥0.92 uplift>1
```

## Architecture

### Core Components

1. **Bootstrap Statistics** (`bootstrap_stats.py`)
   - Bias-corrected and accelerated (BCa) confidence intervals
   - Coverage and uplift CI computation
   - Metabolism verification logic

2. **Coverage Tracker** (`coverage.py`)
   - Statement novelty measurement
   - Per-run coverage tracking
   - Cumulative coverage aggregation

3. **Experiment Executor** (`experiment.py`)
   - Single derivation run execution
   - Metrics collection (throughput, depth, success rate)
   - Database integration

4. **40-Run Orchestrator** (`runner.py`)
   - Sequential experiment execution
   - Bootstrap CI computation
   - Metabolism verification
   - Results export

5. **Evidence Visualizer** (`visualizer.py`)
   - Matplotlib evidence curves
   - Coverage/uplift trajectories
   - Bootstrap CI summary plots

6. **Configuration** (`config.py`)
   - Experiment parameters
   - Environment variable loading
   - Validation
7. **Curriculum Planner & Policy Ledger** (`config.py`, `runner.py`)
   - Deterministic curriculum slices (warmup/core/refinement)
   - Reward-shaped policy ledger with symbolic descent tracking
   - Abstention histogram and tolerance guardrails

## Usage

### Quick Test (5 runs)

```bash
python scripts/rfl/rfl_gate.py --quick
```

### Production (40 runs)

```bash
python scripts/rfl/rfl_gate.py --config config/rfl/production.json
```

### From Python

```python
from backend.rfl.config import RFLConfig
from backend.rfl.runner import RFLRunner

config = RFLConfig(
    experiment_id="my_experiment",
    num_runs=40,
    derive_steps=100,
    coverage_threshold=0.92,
    uplift_threshold=1.0
)

runner = RFLRunner(config)
results = runner.run_all()

if runner.metabolism_passed:
    print(f"✓ {runner.metabolism_message}")
else:
    print(f"✗ {runner.metabolism_message}")
```

## Statistical Methods

### Bootstrap Confidence Intervals

The framework uses **BCa (bias-corrected and accelerated)** bootstrap for confidence intervals:

1. **Resampling**: Generate 10,000 bootstrap samples with replacement
2. **Bias Correction**: Compute z₀ = Φ⁻¹(#{θ̂* < θ̂} / B)
3. **Acceleration**: Jackknife-based acceleration parameter â
4. **Adjusted Percentiles**: α₁ = Φ(z₀ + (z₀+zα)/(1-â(z₀+zα)))

BCa is preferred over percentile bootstrap for:
- Ratio estimators (uplift = treatment/baseline)
- Non-symmetric distributions
- Transformation-respecting intervals
- **Dual Attestation**: BCa estimates are required to agree with a percentile-based attestation within `dual_attestation_tolerance`; disagreements force an abstain verdict to preserve rigor.

**Reference**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"

### Coverage Metric

```
coverage_rate = distinct_statements / target_statement_space
```

Per-run coverage rates are aggregated via bootstrap to produce:
- Point estimate: mean coverage
- 95% CI: [ci_lower, ci_upper]

**Acceptance**: ci_lower ≥ 0.92

### Uplift Metric

```
uplift = treatment_throughput / baseline_throughput
```

Baseline = first half of runs, Treatment = second half of runs.

Paired bootstrap computes CI for ratio estimator:
- Point estimate: mean uplift
- 95% CI: [ci_lower, ci_upper]

**Acceptance**: ci_lower > 1.0

### Abstention Discipline

The framework follows **Proof-or-Abstain** protocol:

- **ABSTAIN** when:
  - Insufficient runs (< 2 successful)
  - Zero baseline throughput
  - Schema errors
  - Missing data

- **Abstention tracking** enables self-learning:
  - Identify failure patterns
  - Refine derivation policies
  - Improve sampling strategies

## Output Artifacts

All artifacts saved to `artifacts/rfl/`:

### `rfl_results.json`

Complete experiment results:
```json
{
  "experiment_id": "rfl_001",
  "execution_summary": {
    "total_runs": 40,
    "successful_runs": 40
  },
  "runs": [...],
  "coverage": {
    "aggregate": {...},
    "bootstrap_ci": {
      "point_estimate": 0.9456,
      "ci_lower": 0.9301,
      "ci_upper": 0.9587,
      "method": "BCa_95%"
    }
  },
  "uplift": {
    "bootstrap_ci": {
      "point_estimate": 1.3214,
      "ci_lower": 1.1523,
      "ci_upper": 1.4872,
      "method": "BCa_95%"
    }
  },
  "metabolism_verification": {
    "passed": true,
    "message": "[PASS] Reflexive Metabolism Verified coverage≥0.92 uplift>1",
    "abstention_fraction": 0.083,
    "abstention_tolerance": 0.25
  },
  "abstentions": {
    "histogram": {
      "pending_validation": 42,
      "zero_throughput": 3
    },
    "fraction": 0.083,
    "tolerance": 0.25
  },
  "policy": {
    "summary": {
      "entries": 40,
      "mean_reward": 0.712,
      "mean_symbolic_descent": 0.008,
      "curriculum_counts": {
        "warmup": 8,
        "core": 24,
        "refinement": 8
      }
    },
    "ledger": [
      {
        "run_id": "rfl_prod_run_01",
        "slice_name": "warmup",
        "policy_reward": 0.402,
        "symbolic_descent": 0.612,
        "coverage_rate": 0.781,
        "novelty_rate": 0.421
      }
    ]
  },
  "dual_attestation": {
    "enabled": true,
    "checks": {
      "coverage": {
        "consistent": true,
        "tolerance": 0.005
      },
      "uplift": {
        "resolved_method": "BCa_95%",
        "consistent": true
      }
    }
  }
}
```

### `rfl_coverage.json`

Per-run coverage details:
```json
{
  "baseline_count": 5234,
  "accumulated_count": 8912,
  "runs": [
    {
      "run_id": "run_01",
      "total_statements": 198,
      "distinct_statements": 187,
      "novel_statements": 145,
      "coverage_rate": 0.9450,
      "novelty_rate": 0.7323
    },
    ...
  ],
  "aggregate": {
    "coverage_mean": 0.9456,
    "coverage_std": 0.0123,
    "novelty_mean": 0.6789
  }
}
```

### `rfl_curves.png`

Evidence curves (6 panels):
1. Coverage over runs with bootstrap CI
2. Throughput over runs (baseline vs treatment)
3. Success rate over runs
4. Novelty rate over runs
5. Mean depth over runs
6. Bootstrap CI summary with pass/fail indicators

## CI Integration

### Exit Codes

```
0: PASS - Metabolism verified
1: FAIL - Criteria not met
2: ERROR - System/configuration error
3: ABSTAIN - Insufficient data
```

### GitHub Actions Example

```yaml
name: RFL Metabolism Gate

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  rfl-verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync

      - name: Start infrastructure
        run: docker compose up -d postgres redis

      - name: Run RFL verification
        run: |
          python scripts/rfl/rfl_gate.py --config config/rfl/production.json

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: rfl-results
          path: artifacts/rfl/
```

## Testing

```bash
# Run RFL tests
pytest tests/rfl/ -v

# Run specific test modules
pytest tests/rfl/test_bootstrap_stats.py -v
pytest tests/rfl/test_coverage.py -v
pytest tests/rfl/test_config.py -v

# Run with coverage
pytest tests/rfl/ --cov=backend.rfl --cov-report=html
```

## Performance

Typical timing (40 runs, 100 steps/run):
- Derivation: ~2-3 hours (depends on system performance)
- Bootstrap CI: ~30 seconds (10,000 replicates)
- Visualization: ~5 seconds
- **Total**: ~2-3 hours

Quick test (5 runs, 10 steps/run):
- Derivation: ~15 minutes
- Bootstrap CI: ~3 seconds (1,000 replicates)
- **Total**: ~15 minutes

## Dependencies

Core:
- `numpy` - Array operations and statistics
- `scipy` - Statistical functions (norm.ppf, norm.cdf)
- `matplotlib` - Evidence curve visualization
- `psycopg` - PostgreSQL database access

From existing MathLedger stack:
- `backend.axiom_engine.derive_cli` - Derivation engine
- `backend.logic.canon` - Formula normalization

## References

1. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

2. DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*, 11(3), 189-228.

3. Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494-509.

## Future Enhancements

Potential extensions:
- Multi-system experiments (PL, FOL=, Group, Ring)
- Parallel run execution (multiprocessing)
- Adaptive policy refinement based on abstentions
- Real-time monitoring dashboard
- Cross-epoch metabolism tracking
- Bayesian credible intervals
- Effect size computation (Cliff's Delta, Cohen's d)
- Convergence diagnostics (Gelman-Rubin, Geweke)

## Contact

For questions about RFL framework:
- See `docs/whitepaper.md` for theoretical foundation
- See `docs/API_REFERENCE.md` for API details
- Open issue at https://github.com/helpfuldolphin/mathledger/issues
