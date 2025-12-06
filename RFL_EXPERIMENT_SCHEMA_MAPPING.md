# RFL Experiment Log Schema Mapping

This document defines how existing system telemetry maps to the versioned RFL Experiment Log Schema (`schema_v_rfl_experiment_1.json`) and provides operational recommendations for log management.

## 1. Schema Mapping Strategy

The RFL schema acts as a **unifying layer** over three primary telemetry sources:
1.  **Derivation Engine**: Outputs `DERIVATION_SUMMARY`.
2.  **First Organism (FO)**: Outputs Cryptographic Roots (`H_t`, `R_t`, `U_t`).
3.  **Metrics Cartographer**: Outputs System Performance (Latency, Memory).

### Mapping Table

| Schema Field | Source System | Source Variable/Field | Notes |
| :--- | :--- | :--- | :--- |
| `cycle_index` | RFL Runner | `cycle_seq` / Loop index | Monotonically increasing per experiment. |
| `mode` | RFL Config | `config.mode` | `baseline` or `rfl`. |
| **`config`** | | | |
| `config.experiment_id` | RFL Config | `config.experiment_id` | e.g., "rfl_law_test". |
| `config.slice_id` | Derivation | `slice.name` / `config.slice_id` | e.g., "test-slice". |
| `config.policy_id` | RFL Context | `policy_context.id` | Identifier of the active policy. |
| **`attestation`** | | | |
| `attestation.h_t` | FO / Ledger | `block.composite_attestation_root` | **CRITICAL**: `SHA256(R_t || U_t)`. |
| `attestation.r_t` | FO / Ledger | `block.reasoning_root` | |
| `attestation.u_t` | FO / Ledger | `block.ui_root` | |
| **`metrics.abstention`** | | | |
| `abstention.rate` | Derivation / RFL | `summary.metrics.abstention_rate` | $\alpha_{rate}$ |
| `abstention.mass` | Derivation / RFL | `summary.metrics.n_abstain` (weighted) | $\alpha_{mass}$ |
| `abstention.attempt_mass` | Derivation / RFL | `summary.metrics.n_candidates` | Total candidates processed. |
| `abstention.tolerance` | RFL Config | `config.abstention_tolerance` | $\tau$ |
| **`metrics.derivation`** | | | |
| `derivation.candidates_total` | Derivation | `summary.metrics.n_candidates` | From `DERIVATION_SUMMARY`. |
| `derivation.verified_count` | Derivation | `summary.metrics.n_verified` | From `DERIVATION_SUMMARY`. |
| `derivation.abstained_count` | Derivation | `summary.metrics.n_abstain` | From `DERIVATION_SUMMARY`. |
| `derivation.depth_max` | Derivation | `summary.filtering.depth_max` | (If available in extended summary). |
| **`metrics.performance`** | | | |
| `performance.latency_ms` | Metrics Cartographer | `performance.latency_ms` | From Performance Passport. |
| `performance.memory_mb` | Metrics Cartographer | `performance.memory_mb` | From Performance Passport. |
| **`rfl_law`** | | | |
| `rfl_law.step_id` | RFL Audit | `StepIdComputation.step_id` | Deterministic ID. |
| `rfl_law.symbolic_descent` | RFL Audit | `SymbolicDescentGradient.symbolic_descent` | $\nabla_{sym}$ |
| `rfl_law.policy_reward` | RFL Audit | `SymbolicDescentGradient.policy_reward` | Reward signal. |

## 2. Log Storage Recommendations

### Directory Structure
We recommend a hierarchical structure separating raw artifacts from synthesized experiment logs.

```text
artifacts/
  └── experiments/
      ├── rfl/
      │   ├── {experiment_id}/
      │   │   ├── run_{run_id}/
      │   │   │   ├── experiment_log.jsonl  <-- The Schema File (Append-only)
      │   │   │   ├── performance_passport.json
      │   │   │   └── derivation_summary.json
      │   │   └── latest -> run_{latest_run_id}
      └── baseline/
          └── ...
```

### Justification
*   **`artifacts/experiments/rfl/*`**: Keeps experimental data distinct from production/system logs (`logs/`).
*   **`experiment_log.jsonl`**: Using JSON Lines (JSONL) is preferred over a single JSON array for append-only safety and stream processing ease. Each line is a valid JSON object conforming to `schema_v_rfl_experiment_1.json`.

## 3. Log Management Strategy

Since we cannot use DB/Redis, file management is crucial to prevent bloat.

### Rotation & Compression
1.  **Rotation**: Rotate `experiment_log.jsonl` if it exceeds 100MB or every 10,000 cycles.
    *   Naming: `experiment_log.1.jsonl`, `experiment_log.2.jsonl`.
2.  **Compression**: Post-process completed runs by compressing JSONL files to `.jsonl.gz`.
    *   Trigger this at the end of an experiment batch in the `Makefile` or CI script.

### Summaries
Instead of parsing huge logs for quick checks, generate a `summary_report.md` at the end of each run containing:
*   Total Cycles
*   Final `H_t`
*   Average Abstention Rate
*   Net Symbolic Descent

### Ingestion
For downstream analysis (e.g., Jupyter, dashboards), a simple Python script can load the JSONL files into a Pandas DataFrame:

```python
import pandas as pd
import json

def load_rfl_logs(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)
```

## 4. Integration Points

### `log_first_organism_pass(h_t)`
*   **Current**: Prints to stdout.
*   **Proposed Extension**: Should also append a structured entry to the `experiment_log.jsonl` if an experiment context is active.

### `DERIVATION_SUMMARY`
*   **Current**: Logged as a structured string/dict.
*   **Proposed Extension**: The RFL Runner should capture this object and fold it into the `metrics.derivation` section of the schema before writing the log line.

```