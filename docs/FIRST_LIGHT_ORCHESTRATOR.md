# First Light Orchestrator

## Overview

The First Light Orchestrator is a single-command coordination system that ties together curriculum, safety, TDA (Topological Data Analysis), uplift, and evidence generation around the run harness. It provides a reproducible and deterministic way to execute First Light experiments with full governance tracking.

## Key Features

- **Single Command Execution**: Run complete First Light experiments with one command
- **Deterministic**: Same seed produces identical trajectories
- **Comprehensive Tracking**: Records Δp (policy weight changes) and HSS (abstention rate) trajectories
- **Governance Envelopes**: Collects curriculum, safety, TDA, and telemetry metrics
- **Evidence Packaging**: Automatically generates structured evidence packages
- **Verification Mode**: Validates evidence package structural integrity

## Installation

No additional installation required beyond the base MathLedger dependencies.

## Usage

### Basic Run

Execute a First Light run with default parameters:

```bash
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 1000 \
  --slice arithmetic_simple \
  --mode integrated
```

### Parameters

- `--seed INT`: Random seed for deterministic execution (default: 42)
- `--cycles INT`: Number of cycles to run (default: 1000)
- `--slice STRING`: Curriculum slice name (default: "arithmetic_simple")
- `--mode {baseline|integrated}`: Run mode
  - `baseline`: No RFL policy updates (control)
  - `integrated`: Full RFL with policy learning (treatment)

### Run Modes

#### Baseline Mode

In baseline mode, the orchestrator runs without RFL policy updates. Policy weights remain at zero throughout the run. This provides a control baseline for comparison.

```bash
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 1000 \
  --slice arithmetic_simple \
  --mode baseline
```

#### Integrated Mode

In integrated mode, the orchestrator applies RFL policy updates based on cycle performance. The 3-parameter policy learns to adjust:
- `len`: Preference for formula length
- `depth`: Preference for AST depth
- `success`: Weight for success history

```bash
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 1000 \
  --slice arithmetic_simple \
  --mode integrated
```

### Verification Mode

Verify the structural integrity of an evidence package:

```bash
python scripts/first_light_orchestrator.py \
  --verify-evidence \
  --run-dir first_light_run/fl_integrated_42_1234567890
```

Exit codes:
- `0`: Evidence package is structurally valid
- `1`: Evidence package is invalid or missing files

## Output Structure

Each run creates a directory under `first_light_run/` with the following structure:

```
first_light_run/
└── fl_{mode}_{seed}_{timestamp}/
    ├── result.json          # Main results and metadata
    ├── trajectories.json    # Δp and HSS trajectories
    ├── governance.json      # Governance envelopes per cycle
    ├── cycles.jsonl         # Cycle-by-cycle raw logs
    └── evidence.json        # Complete evidence package
```

### result.json

Contains run metadata, final statistics, and stability report:

```json
{
  "run_id": "fl_integrated_42_1234567890",
  "config": {
    "seed": 42,
    "cycles": 1000,
    "slice_name": "arithmetic_simple",
    "mode": "integrated"
  },
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T00:10:00Z",
  "duration_seconds": 600.0,
  "final_policy_weights": {
    "len": -0.5,
    "depth": 0.25,
    "success": 5.95
  },
  "final_abstention_rate": 0.15,
  "total_proofs_verified": 8500,
  "total_candidates_processed": 10000,
  "stability_report": {
    "hss_mean": 0.15,
    "hss_std": 0.02,
    "hss_cv": 0.13,
    "hss_stable": true,
    "policy_len_std": 0.1,
    "policy_depth_std": 0.05,
    "policy_success_std": 0.2,
    "policy_stable": true,
    "num_cycles": 1000,
    "convergence_achieved": true
  }
}
```

### trajectories.json

Contains Δp (policy weight) and HSS (abstention rate) trajectories:

```json
{
  "delta_p_trajectory": [
    {"len": 0.0, "depth": 0.0, "success": 0.0},
    {"len": -0.01, "depth": 0.005, "success": 0.1},
    ...
  ],
  "hss_trajectory": [0.28, 0.25, 0.22, ...]
}
```

### governance.json

Array of governance envelopes, one per cycle:

```json
[
  {
    "curriculum_stability": {
      "active_slice": "arithmetic_simple",
      "wallclock_minutes": 0.01,
      "proof_velocity_cv": 0.05,
      "coverage_rate": 0.95
    },
    "safety_metrics": {
      "abstention_rate": 0.28,
      "abstention_mass": 2,
      "safety_threshold_met": true
    },
    "tda_metrics": {
      "persistence_diagram": "mock",
      "betti_numbers": [1, 0, 0],
      "bottleneck_distance": 0.1
    },
    "telemetry_metrics": {
      "throughput_proofs_per_hour": 500,
      "queue_backlog": 0.1,
      "resource_utilization": 0.7
    },
    "epistemic_tile": {
      "uncertainty_mass": 0.28,
      "confidence_interval": [0.9, 0.95]
    },
    "harmonic_tile": {
      "oscillation_amplitude": 0.01,
      "phase_coherence": 0.98
    },
    "drift_tile": {
      "concept_drift_score": 0.02,
      "distribution_shift": 0.01
    },
    "semantic_tile": {
      "vocabulary_coverage": 0.95,
      "semantic_density": 0.85
    }
  },
  ...
]
```

### cycles.jsonl

JSONL format with one entry per cycle:

```jsonl
{"cycle": 0, "candidates_processed": 7, "proofs_verified": 5, "abstentions": 2, "abstention_rate": 0.2857, "policy_weights": {"len": 0.0, "depth": 0.0, "success": 0.0}}
{"cycle": 1, "candidates_processed": 8, "proofs_verified": 6, "abstentions": 2, "abstention_rate": 0.25, "policy_weights": {"len": -0.01, "depth": 0.005, "success": 0.2}}
...
```

### evidence.json

Complete evidence package following the Prelaunch spec:

```json
{
  "version": "1.0.0",
  "created_at": "2024-01-01T00:10:00Z",
  "run_metadata": {...},
  "stability_report": {...},
  "trajectories": {...},
  "governance": {...},
  "synthetic_raw_logs": {...},
  "summary": {...}
}
```

## Determinism

The orchestrator guarantees deterministic execution:
- Same seed produces identical Δp trajectories
- Same seed produces identical HSS trajectories
- Same seed produces identical governance envelopes

This is critical for reproducibility and verification.

### Example

```bash
# Run 1
python scripts/first_light_orchestrator.py --seed 123 --cycles 100 --mode integrated
# Produces: first_light_run/fl_integrated_123_T1/

# Run 2 (same seed)
python scripts/first_light_orchestrator.py --seed 123 --cycles 100 --mode integrated
# Produces: first_light_run/fl_integrated_123_T2/

# Verify trajectories are identical
diff \
  first_light_run/fl_integrated_123_T1/trajectories.json \
  first_light_run/fl_integrated_123_T2/trajectories.json
# Should show no differences
```

## Evidence Package Specification

The evidence package follows the Prelaunch specification and includes:

### Required Fields

1. **version**: Schema version (currently "1.0.0")
2. **run_metadata**: Run configuration and timing
3. **stability_report**: Convergence and stability metrics
4. **trajectories**: Δp and HSS time series
5. **governance**: All governance envelopes
6. **summary**: Aggregate statistics

### Validation

Evidence packages are validated for:
- Presence of all required fields
- Trajectory length matching cycle count
- Governance envelope count matching cycle count
- JSON structural integrity

## Integration with Existing Systems

The First Light orchestrator is designed as a coordination layer that can wire together:

- **U2Runner**: For baseline execution and search
- **RFLRunner**: For reflexive formal learning
- **Curriculum Gates**: For slice advancement
- **Safety Gates**: For abstention monitoring
- **TDA Gates**: For topological analysis
- **Telemetry System**: For metrics collection

Currently, the orchestrator simulates these components for demonstration. To integrate with real components, replace the `_run_cycle()` method with actual U2/RFL runner calls.

## API Reference

### FirstLightConfig

Configuration dataclass for First Light runs.

**Fields:**
- `seed: int` - Random seed
- `cycles: int` - Number of cycles
- `slice_name: str` - Curriculum slice
- `mode: str` - "baseline" or "integrated"
- `output_dir: Path` - Output directory (default: "first_light_run")
- `enable_safety_gate: bool` - Enable safety checks (default: True)
- `enable_curriculum_gate: bool` - Enable curriculum checks (default: True)
- `enable_tda_gate: bool` - Enable TDA analysis (default: True)
- `enable_telemetry: bool` - Enable telemetry (default: True)

### FirstLightRunner

Main orchestrator class.

**Methods:**
- `__init__(config: FirstLightConfig)` - Initialize runner
- `run() -> FirstLightResult` - Execute run
- `_run_cycle(cycle: int) -> Dict[str, Any]` - Execute single cycle
- `_collect_governance_envelope(cycle: int, cycle_result: Dict) -> GovernanceEnvelope` - Collect governance metrics
- `_generate_stability_report() -> Dict[str, Any]` - Generate stability report
- `_write_artifacts(result: FirstLightResult) -> None` - Write output files

### Functions

**build_first_light_evidence_package(run_dir: Path) -> Dict[str, Any]**

Load all artifacts from a run directory and build a unified evidence package.

**verify_evidence_package(run_dir: Path) -> Tuple[bool, str]**

Verify structural validity of an evidence package.

Returns:
- `(True, "valid message")` if package is valid
- `(False, "error message")` if package is invalid

## Examples

### Quick Test Run

```bash
# 10-cycle baseline run for testing
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 10 \
  --slice arithmetic_simple \
  --mode baseline
```

### Production Run

```bash
# 1000-cycle integrated run with RFL
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 1000 \
  --slice arithmetic_simple \
  --mode integrated
```

### Determinism Test

```bash
# Run twice with same seed
python scripts/first_light_orchestrator.py --seed 123 --cycles 100 --mode integrated
python scripts/first_light_orchestrator.py --seed 123 --cycles 100 --mode integrated

# Find the two run directories
RUN1=$(ls -dt first_light_run/fl_integrated_123_* | sed -n 2p)
RUN2=$(ls -dt first_light_run/fl_integrated_123_* | sed -n 1p)

# Compare trajectories (should be identical)
diff "$RUN1/trajectories.json" "$RUN2/trajectories.json"
```

### Evidence Verification

```bash
# Run experiment
python scripts/first_light_orchestrator.py \
  --seed 42 \
  --cycles 100 \
  --mode integrated

# Get latest run directory
RUN_DIR=$(ls -dt first_light_run/* | head -1)

# Verify evidence package
python scripts/first_light_orchestrator.py \
  --verify-evidence \
  --run-dir "$RUN_DIR"
```

## Troubleshooting

### Evidence Package Validation Fails

**Symptom**: `--verify-evidence` returns exit code 1

**Possible causes:**
1. Missing artifact files (result.json, trajectories.json, etc.)
2. Corrupted JSON files
3. Trajectory length mismatch
4. Missing required fields

**Solution**: Check error message for specific issue. Re-run experiment if artifacts are corrupted.

### Non-Deterministic Results

**Symptom**: Same seed produces different trajectories

**Possible causes:**
1. External randomness source (should not happen in current implementation)
2. Floating-point non-determinism (extremely rare)
3. Different Python versions or environments

**Solution**: Ensure running in same environment. File bug report if issue persists.

### Out of Memory

**Symptom**: Process killed during long runs

**Possible causes:**
1. Too many cycles
2. Governance data accumulation

**Solution**: Reduce cycle count or implement streaming writes for governance data.

## Future Enhancements

Planned features for future releases:

1. **Real U2/RFL Integration**: Replace simulated cycles with actual U2Runner/RFLRunner calls
2. **Streaming Writes**: Write governance envelopes incrementally for very long runs
3. **Parallel Execution**: Run multiple experiments in parallel
4. **Resume from Checkpoint**: Continue interrupted runs
5. **Advanced Analytics**: Built-in trajectory analysis and visualization
6. **Remote Evidence Upload**: Automatic upload to evidence repository

## License

Part of the MathLedger project. See repository root for license information.
