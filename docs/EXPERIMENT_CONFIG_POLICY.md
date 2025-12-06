# Experiment Config Policy

## 1. Overview

This policy defines the taxonomy, naming conventions, and usage of experiment configurations for the Reflexive Metabolism Gate (RFL). The goal is to ensure reproducibility and clear provenance for all experimental runs.

## 2. Config Taxonomy

We categorize configurations into two types:

### A. Canonical Experiments
These are "official" configurations used for critical gates (CI, Release, Audits). They should only be modified via a reviewed PR.

*   **`configs/rfl_experiment_rfl_default.yaml`**: The standard production configuration. Used for release gates and nightly integrity checks. Matches the legacy `production.json`.
*   **`configs/rfl_experiment_high_abstention.yaml`**: A high-rigor mode for critical infrastructure verification. Uses stricter confidence intervals (99%) and higher replication counts.

### B. Exploratory/Dev Experiments
These are lightweight or experimental configurations used for development loops or testing new metrics.

*   **`configs/rfl_experiment_baseline.yaml`**: A fast, low-cost configuration (fewer replicates, percentile method) for local development and pre-commit hooks.

## 3. Naming Convention

All config files in `configs/` must follow this pattern:

```
rfl_experiment_<descriptor>.yaml
```

*   **descriptor**: snake_case description of the experimental condition (e.g., `baseline`, `high_noise`, `ab_test_v2`).

## 4. Usage Recommendation

The RFL runner (`rfl_gate.py`) supports both JSON and YAML configurations.

### Running an Experiment

```bash
# Run the baseline (fast)
python rfl_gate.py configs/rfl_experiment_baseline.yaml

# Run production gate
python rfl_gate.py configs/rfl_experiment_rfl_default.yaml
```

### Resolving Configs

Agents and scripts should accept a `--config` flag (or positional argument) that defaults to `configs/rfl_experiment_rfl_default.yaml` if not specified.

## 5. Provenance

All experiment artifacts (results.json, coverage.json) **MUST** include a copy of the configuration used (or its hash/name) to ensure that any plot or table can be linked back to the exact experimental conditions. The current RFL runner embeds the full config in `results.json`.
