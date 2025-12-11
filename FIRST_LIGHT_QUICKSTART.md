# First Light Orchestrator - Quick Start

## What is First Light?

First Light is the coordination agent that ties curriculum, safety, TDA, uplift, and evidence together around the run harness. It provides a single-command way to execute reproducible experiments with full governance tracking.

## Quick Start

### Run Your First Experiment

```bash
# Run a 100-cycle integrated experiment
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 100 \
  --slice arithmetic_simple \
  --mode integrated
```

Expected output:
```
================================================================================
First Light Orchestrator - INTEGRATED Mode
================================================================================
Run ID: fl_integrated_42_1765429080
Seed: 42
Cycles: 100
Slice: arithmetic_simple
Output: first_light_run/fl_integrated_42_1765429080
================================================================================
[Cycle 1/100] Δp_len=-0.0100, HSS=0.2857, verified=5
[Cycle 100/100] Δp_len=-1.0020, HSS=0.3333, verified=4
...
✓ Evidence package valid: Evidence package structurally valid
```

### View Results

```bash
# Get the run directory
RUN_DIR=$(ls -dt first_light_run/* | head -1)

# View main results
cat $RUN_DIR/result.json | python3 -m json.tool | head -30

# View trajectories
cat $RUN_DIR/trajectories.json | python3 -m json.tool | head -20

# View evidence package
cat $RUN_DIR/evidence.json | python3 -m json.tool | head -30
```

### Verify Evidence Package

```bash
# Verify structural integrity
python scripts/first_light_orchestrator.py \
  --verify-evidence \
  --run-dir $RUN_DIR
```

Expected output:
```
Verifying evidence package: first_light_run/fl_integrated_42_1765429080
Building evidence package from: first_light_run/fl_integrated_42_1765429080
Evidence package built: 8 top-level keys
✓ PASS: Evidence package structurally valid
```

## Key Concepts

### Δp (Delta-P) Trajectory

The Δp trajectory tracks changes in the 3-parameter policy weights over time:
- `len`: Preference for formula length (negative = prefer shorter)
- `depth`: Preference for AST depth
- `success`: Weight for success history

In **integrated mode**, these weights are updated based on cycle performance using RFL (Reflexive Formal Learning).

In **baseline mode**, these weights remain at zero (no learning).

### HSS (Harmonic Stability Spectrum) Trajectory

The HSS trajectory tracks the abstention rate over time. This measures what fraction of candidates are abstained (not proven) in each cycle.

A stable HSS (low coefficient of variation) indicates the system has reached equilibrium.

### Governance Envelopes

Each cycle collects a governance envelope with:
- **Curriculum Stability**: Active slice, proof velocity, coverage
- **Safety Metrics**: Abstention rates, threshold compliance
- **TDA Metrics**: Topological data analysis results
- **Telemetry**: Throughput, queue, resource utilization
- **Governance Tiles**: Epistemic, harmonic, drift, semantic metrics

## Common Use Cases

### Baseline vs Integrated Comparison

```bash
# Run baseline (no RFL)
python scripts/first_light_orchestrator.py \
  --seed 42 --cycles 100 --mode baseline

# Run integrated (with RFL)
python scripts/first_light_orchestrator.py \
  --seed 42 --cycles 100 --mode integrated

# Compare final policy weights (baseline should be all zeros)
```

### Determinism Verification

```bash
# Run twice with same seed
python scripts/first_light_orchestrator.py --seed 123 --cycles 50 --mode integrated
sleep 2
python scripts/first_light_orchestrator.py --seed 123 --cycles 50 --mode integrated

# Compare trajectories (should be identical)
RUN1=$(ls -dt first_light_run/fl_integrated_123_* | sed -n 2p)
RUN2=$(ls -dt first_light_run/fl_integrated_123_* | sed -n 1p)
diff $RUN1/trajectories.json $RUN2/trajectories.json
```

### Long-Running Production Experiment

```bash
# 1000-cycle run with detailed governance tracking
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 1000 \
  --slice arithmetic_simple \
  --mode integrated \
  > first_light_production.log 2>&1 &

# Monitor progress
tail -f first_light_production.log
```

## Understanding Output

### Directory Structure

```
first_light_run/fl_integrated_42_1765429080/
├── result.json          # Main results and metadata
├── trajectories.json    # Δp and HSS time series
├── governance.json      # Per-cycle governance envelopes
├── cycles.jsonl         # Raw cycle logs (JSONL format)
└── evidence.json        # Complete evidence package
```

### Key Metrics

- **Total Proofs Verified**: Number of successfully proven candidates
- **Final Abstention Rate**: Fraction of candidates abstained in last cycle
- **Policy Weights**: Final learned policy parameters (integrated mode only)
- **HSS CV (Coefficient of Variation)**: Measure of HSS stability (lower = more stable)
- **Convergence Achieved**: Boolean indicating if system reached stable state

### Stability Criteria

The orchestrator checks for convergence:
- HSS CV < 0.2 (abstention rate is stable)
- Policy weight std < 0.5 (policy has converged)

If both criteria are met, `convergence_achieved` is `true`.

## Troubleshooting

### High Abstention Rate

If HSS remains high (>0.5):
- Increase cycle count to allow policy to converge
- Check that integrated mode is enabled
- Verify slice difficulty is appropriate

### Non-Convergence

If `convergence_achieved` stays `false`:
- Run more cycles (try 1000+ for full convergence)
- Check for stability report details in result.json
- Verify seed is deterministic (not using random external sources)

### Evidence Verification Fails

If `--verify-evidence` returns FAIL:
- Check error message for specific issue
- Verify all artifact files exist
- Re-run experiment if files are corrupted

## Next Steps

- Read the full documentation: [docs/FIRST_LIGHT_ORCHESTRATOR.md](docs/FIRST_LIGHT_ORCHESTRATOR.md)
- Explore the test suite: [tests/integration/test_first_light_orchestrator.py](tests/integration/test_first_light_orchestrator.py)
- Integrate with existing U2/RFL runners for real experiments
- Build custom analysis tools using the evidence package format

## Support

For issues or questions:
1. Check the full documentation in `docs/FIRST_LIGHT_ORCHESTRATOR.md`
2. Review test examples in `tests/integration/test_first_light_orchestrator.py`
3. File an issue in the repository

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: 2024-12-11
