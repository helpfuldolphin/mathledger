# Phase II: RFL Uplift Plan

> **STATUS: PHASE II — NOT YET RUN. NO UPLIFT CLAIMS MAY BE MADE.**

## Overview

Phase II introduces four uplift slices designed for environments where policy-based
candidate ordering should produce measurable improvements over random baseline exploration.

### Phase I Recap (Completed)

Phase I established:
- **RFL Infrastructure**: Working loop with baseline vs learned modes
- **Negative Control**: No uplift on `slice_uplift_proto` (symmetric environment)
- **Validation**: Different orderings → different candidates, but same outcomes

Phase I result: The system correctly shows no uplift when the environment is path-invariant.
This is the expected behavior for a negative control.

---

## Phase II Uplift Slices

### 1. `slice_uplift_goal` — Goal-Conditioned Target

**Intuition**: Specific target formula(s) are valuable; many other proofs are distractors.

**Success Metric**:
```
success = (
    any(hash in TARGET_HASHES for hash in verified_hashes)
    AND len(verified) >= 3
)
```

**Parameters**:
- Atoms: 4 (`{p, q, r, s}`)
- Depth: 2–5
- Budget: max 40 candidates/cycle

**Expected Uplift**:
- Baseline: 10–30% goal hit rate
- RFL Target: 40–70% goal hit rate

---

### 2. `slice_uplift_sparse` — Sparse Reward

**Intuition**: Many candidates, few are provable. Baseline wanders; RFL learns to avoid dead zones.

**Success Metric**:
```
success = (verified >= 5) under max_candidates <= 40
density = verified / candidates_tried
```

**Parameters**:
- Atoms: 5 (`{p, q, r, s, t}`)
- Depth: 3–7
- Budget: max 40 candidates/cycle

**Expected Uplift**:
- Baseline: 20–50% success
- RFL Target: 50–80% success

---

### 3. `slice_uplift_tree` — Chain Depth

**Intuition**: Target requires chain of k intermediate lemmas. RFL learns to build useful intermediates.

**Success Metric**:
```
success = (
    target_hash in verified_hashes
    AND proof_depth(target) >= 3
)
```

**Parameters**:
- Atoms: 4
- Depth: 2–6
- Budget: max 30 candidates/cycle

**Expected Uplift**:
- Baseline: 10–30% chain success
- RFL Target: 40–70% chain success

---

### 4. `slice_uplift_dependency` — Multiple Subgoals

**Intuition**: Success requires ALL k sub-goals proved in same cycle. RFL learns to coordinate.

**Success Metric**:
```
success = all(
    count(goal) >= 1
    for goal in REQUIRED_GOALS
)
```

**Parameters**:
- Atoms: 5
- Depth: 2–6
- Budget: max 40 candidates/cycle

**Expected Uplift**:
- Baseline: 5–20% joint success
- RFL Target: 30–60% joint success

---

## Policy Features for Phase II

### Core Feature Groups

For each candidate formula φ:

**Syntactic Features**:
- `len(φ)` — token length
- `depth(φ)` — syntax tree depth
- `num_connectives(φ)` — count of logical connectives

**Goal-Related Features** (for goal/dependency slices):
- `overlap_with_target(φ)` — |atoms(φ) ∩ atoms(target)|
- `is_subformula_of_target(φ)` — boolean
- `goal_distance(φ)` — structural difference from target

**Success History Features**:
- `success_count[h]` — successful cycles including this hash
- `attempt_count[h]` — total cycles with this hash
- `success_rate[h]` — success_count / (attempt_count + ε)

**Chain/Dependency Features** (for tree/dependency slices):
- `depth_in_current_DAG(φ)` — minimal proof depth
- `num_dependents(φ)` — downstream proofs using this formula
- `is_required_goal(φ)` — indicator for target hashes

### Scoring Function

```python
score(φ) = (
    w_len      * f_len(φ)
  + w_depth    * f_depth(φ)
  + w_goal     * f_goal_overlap(φ)
  + w_succ     * f_success_rate(φ)
  + w_chain    * f_chain_depth(φ)
  + w_required * f_is_required_goal(φ)
)
```

- **Baseline mode**: All weights = 0, candidates shuffled randomly
- **RFL mode**: Weights updated based on verifiable success

---

## Uplift Evidence Gate

### Eligibility Requirements

1. **Non-degenerate slice**: Phase II slice (not Phase I negative control)
2. **Preregistration**: `PREREG_UPLIFT_U2.yaml` completed before run
3. **Paired runs**: Identical seed schedule for baseline and RFL
4. **Deterministic protocol**: Hermetic, reproducible execution

### Required Outputs

- `results/uplift_u2_<slice>_baseline.jsonl`
- `results/uplift_u2_<slice>_rfl.jsonl`
- `experiment_manifest.json` with config hash and H_t sequence
- `statistical_summary.json` with Δp, CIs, test results

### Statistical Criteria

- **Primary**: Δp = p_rfl - p_base
- **Confidence Intervals**: 95% Wilson CIs for p_base and p_rfl
- **Hypothesis Test**: Two-proportion z-test or bootstrap
- **Minimum Effect**: Δp ≥ 0.05 with CI excluding 0

---

## Experiment Protocol

### U2 Experiment Family

```bash
# 1. Baseline run
uv run python experiments/run_uplift_u2.py \
  --slice-name=slice_uplift_goal \
  --mode=baseline \
  --cycles=500 \
  --seed=<MDAP_SEED> \
  --out=results/uplift_u2_goal_baseline.jsonl

# 2. RFL run (same seed)
uv run python experiments/run_uplift_u2.py \
  --slice-name=slice_uplift_goal \
  --mode=rfl \
  --cycles=500 \
  --seed=<MDAP_SEED> \
  --out=results/uplift_u2_goal_rfl.jsonl

# 3. Summarize
uv run python experiments/summarize_uplift.py \
  --baseline=results/uplift_u2_goal_baseline.jsonl \
  --rfl=results/uplift_u2_goal_rfl.jsonl \
  --metric=goal_hit
```

---

## Narrative Context

> "Industry is moving from RLHF → RLPF → RL with Verifiable Feedback.
>
> Phase I showed that the RFL infrastructure behaves correctly on a symmetric negative control.
>
> Phase II U2 will test whether policies can actually lower epistemic risk on non-degenerate
> slices, under a preregistered, statistically sound protocol."

---

## Files

- `config/curriculum_uplift_phase2.yaml` — Slice definitions
- `experiments/run_uplift_u2.py` — U2 experiment runner (TO BE IMPLEMENTED)
- `experiments/summarize_uplift.py` — Uplift summarizer (TO BE IMPLEMENTED)
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` — Preregistration template (TO BE IMPLEMENTED)

---

## Status

| Component | Status |
|-----------|--------|
| Phase I Infrastructure | ✅ Complete |
| Phase I Negative Control | ✅ Documented |
| Phase II Slice Definitions | ✅ Designed |
| Phase II Runner (U2) | ⏳ Not Implemented |
| Phase II Preregistration | ⏳ Not Implemented |
| Phase II Experiments | ❌ Not Run |
| Uplift Claims | ❌ None (blocked until U2 completes) |

---

## Auto-Generated Reference Tables

> **Note**: The following sections are auto-generated from source files.
> Do not edit manually. Run `python scripts/doc_sync_phase2.py` to update.

### Curriculum Slices (Auto-Generated)

<!-- BEGIN:AUTOGEN:PHASE2-SLICES -->
| Slice Name | Description | Items Count | Prereg Hash |
|------------|-------------|-------------|-------------|
| `algebra_expansion` | Focuses on algebraic expansion. | 6 | `c157a79031e1c40f...` |
| `arithmetic_simple` | Focuses on basic arithmetic. | 8 | `d8e8fca2dc0f896f...` |
<!-- END:AUTOGEN:PHASE2-SLICES -->

### Preregistered Experiments (Auto-Generated)

<!-- BEGIN:AUTOGEN:PHASE2-PREREG -->
| Experiment ID | Description | Slice Config | Success Metrics |
|---------------|-------------|--------------|-----------------|
| `U2_EXP_001` | Experiment to measure uplift in slice A. | `configs/slice_a.json` | goal_hit, sparse_density |
| `U2_EXP_002` | Experiment to measure uplift in slice B. | `configs/slice_b.json` | chain_success, joint_goal |
<!-- END:AUTOGEN:PHASE2-PREREG -->

### CLI Usage (Auto-Generated)

<!-- BEGIN:AUTOGEN:PHASE2-CLI -->
**CLI Arguments:**

```bash
uv run python experiments/run_uplift_u2.py \
  --config=<value> \
  --cycles=100 \
  --mode=baseline \
  --out=results/ \
  --seed=42 \
  --slice=arithmetic_simple
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config` | str | No | `config/curriculum_uplift_phase2.yaml` | Path to the curriculum config file. |
| `--cycles` | int | Yes | — | Number of experiment cycles to run. |
| `--mode` | str | Yes | — | Execution mode: 'baseline' or 'rfl'. |
| `--out` | str | Yes | — | Output directory for results and manifest files. |
| `--seed` | int | Yes | — | Initial random seed for deterministic execution. |
| `--slice` | str | Yes | — | The experiment slice to run (e.g., 'arithmetic_simple'). |
<!-- END:AUTOGEN:PHASE2-CLI -->

### Success Metrics (Auto-Generated)

<!-- BEGIN:AUTOGEN:PHASE2-METRICS -->
| Metric Function | Parameters | Return Type | Description |
|-----------------|------------|-------------|-------------|
| `compute_chain_success` | `verified_statements, dependency_graph, chain_target_hash, min_chain_length` | `Tuple[bool, float]` | Computes success based on the length of a verified dependency chain ending at a target. |
| `compute_goal_hit` | `verified_statements, target_hashes, min_total_verified` | `Tuple[bool, float]` | Computes success based on hitting a minimum number of specific target goals. |
| `compute_multi_goal_success` | `verified_hashes, required_goal_hashes` | `Tuple[bool, float]` | Computes success based on verifying a set of required goals. |
| `compute_sparse_success` | `verified_count, attempted_count, min_verified` | `Tuple[bool, float]` | Computes success based on a simple minimum count of verified statements. |
<!-- END:AUTOGEN:PHASE2-METRICS -->

### Implementation Status (Auto-Generated)

<!-- BEGIN:AUTOGEN:PHASE2-STATUS -->
| Component | Status |
|-----------|--------|
| Curriculum Slices | ✅ 2 slices defined |
| Preregistration | ✅ 2 experiments |
| U2 Runner | ✅ Implemented |
| Success Metrics | ✅ Implemented |
<!-- END:AUTOGEN:PHASE2-STATUS -->

