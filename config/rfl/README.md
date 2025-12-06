# RFL Configuration Files

This directory contains configuration files for Reflexive Formal Learning (RFL) experiments.

## Configuration Format

RFL experiments are configured via JSON files following the `RFLConfig` schema:

```json
{
  "experiment_id": "rfl_001",
  "num_runs": 40,
  "random_seed": 42,
  "system_id": 1,
  "derive_steps": 50,
  "max_breadth": 200,
  "max_total": 1000,
  "depth_max": 4,
  "bootstrap_replicates": 10000,
  "confidence_level": 0.95,
  "coverage_threshold": 0.92,
  "uplift_threshold": 1.0,
  "artifacts_dir": "artifacts/rfl",
  "results_file": "rfl_results.json",
  "coverage_file": "rfl_coverage.json",
  "curves_file": "rfl_curves.png"
}
```

## Parameters

### Experiment Metadata
- `experiment_id`: Unique identifier for the experiment suite
- `num_runs`: Number of derivation runs (default 40)
- `random_seed`: Random seed for reproducibility

### Derivation Parameters (per run)
- `system_id`: Theory system (1=PL, 2=FOL=, etc.)
- `derive_steps`: Number of derivation steps per run
- `max_breadth`: Max new statements per step
- `max_total`: Max total statements per run
- `depth_max`: Max formula depth

### Statistical Parameters
- `bootstrap_replicates`: Number of bootstrap samples (≥1000 recommended)
- `confidence_level`: Confidence level for CIs (default 0.95 = 95%)

### Acceptance Criteria
- `coverage_threshold`: Minimum coverage CI lower bound (default 0.92)
- `uplift_threshold`: Minimum uplift CI lower bound (default 1.0)

### Output Paths
- `artifacts_dir`: Directory for results artifacts
- `results_file`: Main results JSON filename
- `coverage_file`: Coverage details JSON filename
- `curves_file`: Evidence curves PNG filename

## Presets

### Quick Test (5 runs)
```bash
python scripts/rfl/rfl_gate.py --quick
```

Uses `RFL_QUICK_CONFIG`:
- 5 runs
- 10 steps per run
- 1000 bootstrap replicates

### Production (40 runs)
```bash
python scripts/rfl/rfl_gate.py --config config/rfl/production.json
```

Uses `RFL_PRODUCTION_CONFIG`:
- 40 runs
- 100 steps per run
- 10000 bootstrap replicates

## Environment Variables

Configuration can also be loaded from environment variables:

```bash
export RFL_EXPERIMENT_ID=rfl_nightly
export RFL_NUM_RUNS=40
export DERIVE_STEPS=100
export RFL_COVERAGE_THRESHOLD=0.92
export RFL_UPLIFT_THRESHOLD=1.0

python scripts/rfl/rfl_gate.py
```

See `backend/rfl/config.py:RFLConfig.from_env()` for full list of environment variables.

## Examples

Example configurations are provided in this directory:

- `quick_test.json` - Fast test (5 runs)
- `nightly.json` - Nightly CI run (40 runs, moderate parameters)
- `production.json` - Full production run (40 runs, high parameters)
