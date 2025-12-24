# RFL Policy Module - Phase II

This module implements the core RFL (Reflexive Formal Learning) policy update logic with Phase II enhancements:
- Per-feature telemetry and drift snapshots (TASK 1)
- Policy comparison tool (TASK 2)  
- Stress-tested safety guardrails (TASK 3)

## Core Components

### `policy.py` - Policy Update Engine

Implements gradient-free policy updates using verifiable feedback (proof success/failure).

**Key Classes:**
- `PolicyState`: Container for policy weights and metadata
- `PolicyUpdater`: Update engine with safety guards
- `PolicyStateSnapshot`: Telemetry snapshot with optional per-feature stats
- `FeatureTelemetry`: Per-feature diagnostic information

**Safety Guards:**
- **L2 norm clamping**: Prevents unbounded weight vector growth
- **Per-weight clipping**: Prevents individual weights from exploding
- **Deterministic updates**: All randomness via `SeededRNG`

**Example:**
```python
from rfl.policy import PolicyUpdater, init_cold_start, summarize_policy_state

# Initialize policy
feature_names = ["formula_length", "num_atoms", "num_connectives"]
state = init_cold_start(feature_names, seed=42)
updater = PolicyUpdater(learning_rate=0.01, max_weight_norm_l2=10.0, seed=42)

# Update policy
features = {"formula_length": 5.0, "num_atoms": 2.0, "num_connectives": 1.0}
reward = 1.0  # Success
state = updater.update(state, features, reward)

# Get telemetry snapshot
snapshot = summarize_policy_state(state, include_feature_telemetry=True, top_k=5)
print(f"L2 norm: {snapshot.l2_norm}")
print(f"Top features: {snapshot.feature_telemetry.top_k_positive}")
```

### `features.py` - Feature Extraction

Extracts structural features from formulas for policy-guided derivation.

**Features:**
- `formula_length`: Character length
- `num_atoms`: Number of atomic propositions
- `num_connectives`: Total logical connectives
- `num_implications`, `num_conjunctions`, `num_disjunctions`, `num_negations`: Per-operator counts
- `tree_depth`: Nesting depth estimate
- `bias`: Constant feature (always 1.0)

**Example:**
```python
from rfl.policy import extract_features

formula = "(p & q) -> r"
features = extract_features(formula)
print(features.features)
# {'formula_length': 13.0, 'num_atoms': 3.0, 'num_implications': 1.0, ...}
```

### `rewards.py` - Reward Computation

Computes reward signals from derivation outcomes using **only verifiable feedback** (no human preferences, no proxy metrics).

**Reward Scheme:**
- Success: +1.0
- Failure: -1.0
- Abstention: 0.0

**Example:**
```python
from rfl.policy import compute_reward

reward_signal = compute_reward(success=True, abstained=False)
print(reward_signal.reward)  # 1.0
```

### `compare.py` - Policy Comparison Tool

Compares two policy states and highlights meaningful differences.

**Metrics:**
- L2 distance between weight vectors
- L1 distance
- Number of sign flips
- Top-K features by absolute delta

**Example (Python API):**
```python
from rfl.policy import compare_policy_states, init_from_file

state_a = init_from_file("policy_a.json")
state_b = init_from_file("policy_b.json")

result = compare_policy_states(state_a, state_b, top_k=10)
print(f"L2 distance: {result.l2_distance}")
print(f"Sign flips: {result.num_sign_flips}")
print(result.format_summary())
```

**Example (CLI):**
```bash
# Compare two JSON files
python -m rfl.policy --a policy_a.json --b policy_b.json

# Compare with JSONL (use specific index)
python -m rfl.policy --a policy_a.jsonl --b policy_b.jsonl --index-a 5 --index-b 10

# JSON output with top-20 features
python -m rfl.policy --a a.json --b b.json --top-k 20 --json

# Handle mismatched feature sets
python -m rfl.policy --a a.json --b b.json --handle-missing union
```

## Configuration

Configuration file: `config/rfl_policy_phase2.yaml`

```yaml
# Policy Update Parameters
learning_rate: 0.01
max_weight_norm_l2: 10.0  # L2 norm clamping threshold
max_abs_weight: 5.0       # Per-weight clipping threshold

# Feature Telemetry Settings (TASK 1)
feature_telemetry:
  enabled: false          # Enable per-feature telemetry in snapshots
  top_k: 5               # Number of top features to track

# Snapshot Settings
snapshot_interval: 10     # Save policy snapshot every N updates
save_snapshots: true      # Whether to save snapshots to disk

# Initialization
warm_start: false         # Load from checkpoint if available
checkpoint_path: null     # Path to checkpoint file

# Reward Settings
bonus_for_short_proof: false  # Apply chain length penalty
```

## Integration with Experiments

Use `PolicyTelemetryCollector` to integrate telemetry into experiment runs:

```python
from experiments.policy_telemetry_example import PolicyTelemetryCollector

# Load config
collector = PolicyTelemetryCollector.from_config("config/rfl_policy_phase2.yaml")

# During experiment loop
for step in range(num_steps):
    # ... policy update ...
    if collector.should_snapshot(step):
        snapshot = collector.capture_snapshot(policy_state, step)

# Save telemetry
collector.save_to_file("artifacts/policy_snapshots.jsonl")
```

## Telemetry Schema

When `feature_telemetry.enabled: true`, snapshots include:

```json
{
  "step": 100,
  "l2_norm": 3.14159,
  "l1_norm": 5.0,
  "max_abs_weight": 2.5,
  "mean_weight": 0.5,
  "num_features": 10,
  "total_reward": 50.0,
  "feature_telemetry": {
    "top_k_positive": [
      ["formula_length", 2.5],
      ["num_atoms", 1.8],
      ["bias", 1.2]
    ],
    "top_k_negative": [
      ["num_negations", -1.5],
      ["tree_depth", -0.8]
    ],
    "sparsity": 0.8,
    "num_features": 10
  }
}
```

**Backward Compatibility:** Old consumers that don't expect `feature_telemetry` will still work (field is optional).

## Testing

Three test suites validate the implementation:

### `test_policy.py` (22 tests)
- Basic policy operations
- Safety guards (per-weight clipping, L2 norm clamping)
- Determinism
- **Telemetry snapshots (TASK 1)**
- Serialization

### `test_policy_compare.py` (19 tests)
- Distance computations (L1, L2)
- Sign flip detection
- Top-K delta selection
- Mismatched feature set handling
- JSON/JSONL loading
- **CLI functionality (TASK 2)**

### `test_policy_guards_stress.py` (15 tests)
- L2 clamping under high learning rates
- Per-weight clipping under extreme updates
- Combined guardrails under adversarial schedules
- **Determinism under stress (TASK 3)**
- Performance (all tests < 1s)

Run tests:
```bash
pytest tests/rfl/test_policy*.py -v
```

## Determinism Contract

All policy operations are deterministic:
- Same seed → same policy trajectory
- No wall-clock time or external entropy
- All randomness via `SeededRNG`
- Policy state is serializable and replayable

## Sober Truth Invariants

✅ **DO:**
- Use verifiable feedback only (proof success/failure)
- Maintain determinism (SeededRNG for all randomness)
- Document policy update formulas in code comments
- Cover all policy code with determinism tests

❌ **DO NOT:**
- Add human preference signals or proxy rewards
- Claim policy changes produce uplift without experiment evidence
- Break determinism
- Modify experiment logs or attestation artifacts
