# Phase II U2 Developer Guide

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> As of Evidence Pack v1, **no uplift claims exist**; all uplift discussion is purely theoretical.
>
> This guide describes the U2 experiment family infrastructure. No experiments have been executed
> and no empirical uplift has been observed.

---

## What U2 Is

U2 refers to the **preregistered asymmetric uplift experiments** designed for Phase II of the MathLedger RFL research program. These experiments are intended to test whether policy-based candidate ordering produces measurable improvements over random baseline exploration in non-degenerate environments.

### Key Characteristics

- **Preregistered**: All experiment parameters, success criteria, and statistical thresholds must be documented before execution
- **Asymmetric**: Unlike Phase I's symmetric negative control, U2 slices introduce environments where learning signal should exist
- **Paired Design**: Each experiment requires both baseline (random ordering) and RFL (policy-driven ordering) runs with identical seeds
- **Deterministic**: All runs must be reproducible given the same seed

### Phase II vs Phase I

| Aspect | Phase I | Phase II U2 |
|--------|---------|-------------|
| Purpose | Negative control (no uplift expected) | Test for measurable uplift |
| Slice Type | Symmetric environment | Asymmetric environments |
| Expected Result | No difference between baseline/RFL | Potential uplift if policy learns |
| Status | Complete | **NOT YET RUN** |

---

## Key Code Modules

The U2 experiment infrastructure consists of the following modules:

### `experiments/run_uplift_u2.py`

The main experiment runner for U2 experiments. Handles:
- Experiment configuration loading
- Seed schedule generation for deterministic runs
- Baseline mode (random shuffle ordering)
- RFL mode (policy-driven ordering)
- Telemetry logging and manifest generation

### `experiments/slice_success_metrics.py`

Pure, deterministic functions for computing slice-specific success metrics:
- `compute_goal_hit()` — Success based on hitting specific target goals
- `compute_sparse_success()` — Success based on minimum verified count
- `compute_chain_success()` — Success based on verified dependency chain length
- `compute_multi_goal_success()` — Success based on verifying a set of required goals

### `experiments/curriculum_loader_v2.py`

*(PHASE II — TO BE IMPLEMENTED)*

Loads curriculum configurations for U2 slices from YAML files.

### `experiments/u2_calibration.py`

*(PHASE II — TO BE IMPLEMENTED)*

Calibration utilities for determining appropriate difficulty parameters for U2 slices.

### `experiments/manifest_verifier.py`

*(PHASE II — TO BE IMPLEMENTED)*

Verifies experiment manifests for integrity and completeness.

---

## How to Run a Small Experiment

> **IMPORTANT**: These commands are provided for documentation purposes only.
> No U2 experiments have been run as of Evidence Pack v1.

### Baseline Run

Run a baseline experiment with random candidate ordering:

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --mode baseline \
  --cycles 20 \
  --seed 12345 \
  --out results/uplift_u2/slice_uplift_goal
```

### Paired RFL Run

Run the paired RFL experiment with the same seed:

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --mode rfl \
  --cycles 20 \
  --seed 12345 \
  --out results/uplift_u2/slice_uplift_goal
```

### Calibration Run

*(PHASE II — NOT YET IMPLEMENTED)*

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --calibration \
  --cycles 50 \
  --seed 12345 \
  --out results/uplift_u2/slice_uplift_goal/calibration
```

### CLI Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--slice` | Slice name (e.g., `slice_uplift_goal`) | Yes |
| `--mode` | Execution mode: `baseline` or `rfl` | Yes |
| `--cycles` | Number of experiment cycles | Yes |
| `--seed` | Random seed for deterministic execution | Yes |
| `--out` | Output directory for results | Yes |
| `--config` | Path to curriculum config (default: `config/curriculum_uplift_phase2.yaml`) | No |

---

## Where Results Go

All U2 experiment outputs are written to structured directories under `results/uplift_u2/`:

```
results/uplift_u2/
└── <slice_name>/
    ├── baseline.jsonl              # Baseline run telemetry (JSONL format)
    ├── rfl.jsonl                   # RFL run telemetry (JSONL format)
    └── manifest.json               # Experiment manifest with hashes
```

### Output Files

#### `baseline.jsonl` / `rfl.jsonl`

Per-cycle telemetry records in JSON Lines format:

```json
{
  "cycle": 0,
  "slice": "slice_uplift_goal",
  "mode": "baseline",
  "seed": 12345,
  "item": "<chosen_item>",
  "result": "<result>",
  "success": true,
  "label": "PHASE II — NOT USED IN PHASE I"
}
```

#### `manifest.json`

Experiment manifest containing:

```json
{
  "label": "PHASE II — NOT USED IN PHASE I",
  "slice": "slice_uplift_goal",
  "mode": "baseline",
  "cycles": 20,
  "initial_seed": 12345,
  "slice_config_hash": "<sha256>",
  "prereg_hash": "<sha256>",
  "ht_series_hash": "<sha256>",
  "deterministic_seed_schedule": [/* seed per cycle */],
  "outputs": {
    "results": "results/uplift_u2/slice_uplift_goal/baseline.jsonl",
    "manifest": "results/uplift_u2/slice_uplift_goal/manifest.json"
  }
}
```

---

## Related Documentation

- `docs/PHASE2_RFL_UPLIFT_PLAN.md` — Overall Phase II uplift plan and slice definitions
- `RFL_UPLIFT_THEORY.md` — Theoretical framework (unverified conjectures)
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` — Preregistration template
- `config/curriculum_uplift_phase2.yaml` — Slice configuration file

---

## Absolute Safeguards

The following safeguards are embedded in the U2 infrastructure:

1. **No Uplift Claims**: All output files are labeled `"PHASE II — NOT USED IN PHASE I"`
2. **Deterministic Execution**: Same seed must produce identical results
3. **Preregistration Required**: Experiments must be preregistered before execution
4. **Paired Design**: Uplift claims require both baseline and RFL runs with matched seeds
5. **Verifiable Feedback Only**: RFL uses only verifiable feedback (no RLHF, no proxy rewards)

---

## Status

| Component | Status |
|-----------|--------|
| `run_uplift_u2.py` | ✅ Implemented |
| `slice_success_metrics.py` | ✅ Implemented |
| `curriculum_loader_v2.py` | ⏳ Not Implemented |
| `u2_calibration.py` | ⏳ Not Implemented |
| `manifest_verifier.py` | ⏳ Not Implemented |
| U2 Experiments | ❌ **NOT RUN** |
| Uplift Evidence | ❌ **NONE** |

---

**PHASE II — NOT PART OF EVIDENCE PACK v1**

*Document created for MathLedger Phase II developer onboarding.*
