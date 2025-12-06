# Causal Analysis Framework for RFL

**Claude D — The Causal Architect**

## Overview

The causal analysis framework formalizes cause-effect structures in MathLedger's Reflexive Formal Learning (RFL) loop. It constructs causal graphs linking policy updates → abstention shifts → throughput uplifts, applies Do-Calculus for interventional analysis, and simulates counterfactuals to test causality.

## Architecture

### Core Components

```
backend/causal/
├── __init__.py           # Package initialization
├── graph.py              # Causal graph (DAG) data structures
├── variables.py          # Variable extraction (Δabstain, Δproof, Δpolicy)
├── do_calculus.py        # Interventional analysis (do-operator)
├── counterfactual.py     # Counterfactual simulation
├── estimator.py          # Causal coefficient estimation
├── report.py             # Visualization and reporting
└── analyze_cli.py        # CLI tool for running analysis
```

### Tests

```
tests/
├── test_causal_graph.py      # Graph structure tests
├── test_causal_variables.py  # Variable extraction tests
└── test_causal_estimator.py  # Estimation tests
```

## Causal Graph Structure

The RFL causal graph models the following relationships:

```
Policy (policy_hash)
    ↓
    ├─→ Abstention (abstain_pct) ─→ Throughput (proofs_per_sec)
    │                    ↓                ↑
    └─→ Depth (depth_max) ─→ Verification Time (verify_ms_p50)
```

### Causal Mechanisms

1. **Policy → Abstention**: Policy scoring determines which derivations to attempt vs abstain
2. **Policy → Depth**: Policy prioritizes formulas by complexity
3. **Abstention → Verification Time**: Higher abstention → attempt easier proofs → faster verification
4. **Depth → Verification Time**: Deeper formulas require longer verification
5. **Verification Time → Throughput**: Faster verification → higher throughput
6. **Abstention → Throughput**: Abstaining from hard problems → focus resources on solvable proofs

## Variable Extraction

### Run Metrics

```python
RunMetrics(
    run_id: int,
    started_at: datetime,
    ended_at: datetime,
    policy_hash: str,
    abstain_pct: float,
    proofs_success: int,
    proofs_per_sec: float,
    depth_max_reached: int,
    system: str,
    slice_name: str
)
```

### Run Deltas

Deltas represent changes between consecutive runs:

- **Δpolicy**: Binary indicator (0 = same policy, 1 = different)
- **Δabstain**: Change in abstention percentage (percentage points)
- **Δproof**: Change in successful proof count
- **Δthroughput**: Change in proofs/second
- **Δdepth**: Change in maximum depth reached

## Do-Calculus and Interventional Analysis

### The Do-Operator

Pearl's do-operator enables computing interventional distributions:

```
P(Y | do(X=x))
```

This represents the distribution of Y when we intervene to set X=x, breaking incoming causal arrows.

### Example Usage

```python
from backend.causal.do_calculus import intervene

# Intervene to set policy to specific hash
data_counterfactual = intervene(
    variable='policy_hash',
    value='new_policy_xyz',
    data=observational_data,
    causal_graph=graph
)

# Compute Average Treatment Effect
ate = compute_ate(
    treatment_var='policy_hash',
    outcome_var='proofs_per_sec',
    treatment_values=['baseline_policy', 'new_policy'],
    data=data,
    causal_graph=graph
)
```

### Identifiability

The framework checks if causal effects are identifiable from observational data via:

- **Backdoor criterion**: Adjusting for confounders
- **Frontdoor criterion**: Using mediators when backdoor is blocked

## Counterfactual Reasoning

Counterfactuals answer questions like:

> "If we had used policy B instead of policy A, what would throughput have been?"

### Three-Step Algorithm (Pearl)

1. **Abduction**: Infer unobserved factors U from observed data
2. **Action**: Replace structural equation for X with X=x
3. **Prediction**: Compute Y using modified equations and inferred U

### Example

```python
from backend.causal.counterfactual import CounterfactualScenario, CounterfactualEngine

scenario = CounterfactualScenario(
    intervention_var='policy_hash',
    intervention_value='alternative_policy',
    observed_vars={
        'policy_hash': 'baseline_policy',
        'proofs_per_sec': 1.5
    },
    target_var='proofs_per_sec'
)

engine = CounterfactualEngine(causal_graph)
result = engine.simulate(scenario)

print(f"Predicted: {result.predicted_value}")
print(f"Actual: {result.actual_value}")
print(f"Effect: {result.counterfactual_effect}")
```

## Causal Coefficient Estimation

### Methods

1. **OLS (Ordinary Least Squares)**: Default method for continuous variables
2. **Matching**: Propensity score matching for binary treatments
3. **Double ML**: Double machine learning for debiased estimates

### Output Format

```python
CausalCoefficient(
    source_var='abstain_pct',
    target_var='proofs_per_sec',
    coefficient=-0.042,
    std_error=0.008,
    confidence_interval=(-0.058, -0.026),
    p_value=0.001,
    n_observations=50,
    method=EstimationMethod.OLS
)
```

### Pass Messages

Coefficients are reported as:

```
[PASS] Causal Model Stable do(abstain_pct)->proofs_per_sec coeff=-0.042 (p=0.0010)
```

## CLI Tool

### Basic Usage

```bash
# Run analysis on database
python -m backend.causal.analyze_cli --system pl --output reports/causal_analysis.json

# Generate markdown report
python -m backend.causal.analyze_cli --format markdown --output reports/causal.md

# Export Graphviz visualization
python -m backend.causal.analyze_cli --graphviz reports/causal_graph.dot

# Test stability with bootstrap
python -m backend.causal.analyze_cli --bootstrap 100
```

### Options

- `--db-url`: PostgreSQL connection URL (default: from `DATABASE_URL` env)
- `--system`: Filter runs by system (pl, fol, etc.)
- `--min-runs`: Minimum number of runs required (default: 2)
- `--output`: Output file path
- `--format`: Output format (json, markdown, ascii)
- `--graphviz`: Export Graphviz DOT file
- `--method`: Estimation method (ols, matching, dml)
- `--bootstrap`: Number of bootstrap samples for stability testing

### Example Output

```
============================================================
CAUSAL ARCHITECT — RFL Mechanistic Analysis
============================================================

Fetching runs from database (system=pl)...
✓ Loaded 47 runs

Extracting run deltas...
✓ Computed 46 consecutive run deltas

Building RFL causal graph...
✓ Graph structure valid

Preparing data for estimation...
✓ Data prepared: 46 samples

Estimating causal coefficients (method=ols)...
✓ Estimated 6 edge coefficients

CAUSAL MODEL STATUS
============================================================
[PASS] Causal Model Stable do(policy_hash)->abstain_pct coeff=0.123 (p=0.0340)
[PASS] Causal Model Stable do(abstain_pct)->proofs_per_sec coeff=-0.042 (p=0.0010)
[PASS] Causal Model Stable do(abstain_pct)->verify_ms_p50 coeff=-0.312 (p=0.0050)
...
```

## Integration with MathLedger

### Database Schema

The causal analysis extracts data from the `runs` table:

```sql
CREATE TABLE runs (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    system TEXT NOT NULL,
    slice TEXT NOT NULL,
    policy_hash TEXT,
    abstain_pct REAL DEFAULT 0.0,
    proofs_success INTEGER DEFAULT 0,
    proofs_per_sec REAL DEFAULT 0.0,
    depth_max_reached INTEGER DEFAULT 0
);
```

### Workflow Integration

1. **After Derivation Run**: Log metrics to `runs` table
2. **Periodic Analysis**: Run causal analysis CLI (e.g., weekly)
3. **Report Generation**: Generate reports and visualizations
4. **Policy Evaluation**: Use coefficients to evaluate policy changes

## Determinism and Reproducibility

All components use fixed seeds for determinism:

- Variable extraction: Deterministic ordering by timestamp
- Do-Calculus: SeededRNG with `_GLOBAL_SEED = 0`
- Bootstrap: Incremental seeds (`seed + i`)
- Counterfactuals: Fixed latent factor inference

## Validation and Testing

### Graph Validation

```python
from backend.causal.graph import validate_graph

warnings = validate_graph(causal_graph)
if warnings:
    for w in warnings:
        print(f"Warning: {w}")
```

Checks:
- Acyclicity (DAG structure)
- Isolated nodes
- Missing coefficient estimates

### Stability Testing

```python
from backend.causal.estimator import test_causal_stability

stability = test_causal_stability(
    causal_graph,
    data,
    n_bootstrap=100
)

for (source, target), stats in stability.items():
    print(f"{source} → {target}: CV={stats['cv']:.3f}")
```

### Test Suite

```bash
# Run causal analysis tests
pytest tests/test_causal_*.py -v

# Run with coverage
coverage run -m pytest tests/test_causal_*.py
coverage report
```

## Visualization

### ASCII Art

```
============================================================
CAUSAL GRAPH
============================================================

  policy_hash          ──→ abstain_pct         β=0.123*
  policy_hash          ──→ depth_max           β=0.045
  abstain_pct          ──→ verify_ms_p50       β=-0.312**
  abstain_pct          ──→ proofs_per_sec      β=-0.042***
  depth_max            ──→ verify_ms_p50       β=0.234*
  verify_ms_p50        ──↓ proofs_per_sec      β=-0.018**

Significance: *** p<0.001, ** p<0.01, * p<0.05
============================================================
```

### Graphviz Export

```bash
python -m backend.causal.analyze_cli --graphviz graph.dot
dot -Tpng graph.dot -o graph.png
```

### Markdown Report

Generates comprehensive report with:
- Graph structure summary
- Coefficient table
- Key findings
- Empirical delta statistics

## Advanced Features

### Conditional Average Treatment Effect (CATE)

```python
cate = compute_cate(
    treatment_var='policy_hash',
    outcome_var='proofs_per_sec',
    treatment_values=['baseline', 'new'],
    conditioning_var='system',
    conditioning_value='pl',
    data=data,
    causal_graph=graph
)
```

### Sensitivity Analysis

```python
results = sensitivity_analysis(
    scenario=counterfactual_scenario,
    causal_graph=graph,
    param_name='abstention_coefficient',
    param_range=(-0.1, 0.1),
    n_steps=20
)
```

### Bounds Analysis

When point identification fails, compute bounds:

```python
lower, upper = bounds_analysis(
    scenario=counterfactual_scenario,
    causal_graph=graph,
    observational_data=data
)
```

## References

### Theoretical Foundation

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Pearl, J., & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.

### Implementation

- DoWhy: Python library for causal inference (inspiration)
- NetworkX: Graph algorithms
- NumPy/SciPy: Statistical estimation

## Future Work

1. **Structural Equation Learning**: Automatically learn functional forms from data
2. **Time-Series Extensions**: Handle temporal dependencies in run sequences
3. **Multi-System Analysis**: Compare causal structures across logical systems
4. **Automated Policy Optimization**: Use causal model to guide policy updates
5. **Real-Time Monitoring**: Stream causal metrics during derivation runs

## Contact

For questions or contributions related to the causal analysis framework, contact:

- **Claude D** - The Causal Architect
- Integration with **Codex M** (mechanistic RFL) and **Claude B** (nightly orchestration)

---

**Status**: [PASS] Causal framework operational - tracing mechanistic links in RFL loop.
