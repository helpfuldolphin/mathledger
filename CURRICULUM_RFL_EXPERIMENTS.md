# Curriculum for RFL Experiments

## Overview
This document proposes three experimental slice profiles for the Propositional Logic (PL) system within the First Organism (FO) cycle. These slices are designed to probe the behavior of the Reflexive Formal Learning (RFL) loop across different regimes of complexity, specifically targeting the "abstention boundary" where the prover transitions from certainty to uncertainty.

## Experimental Slice Profiles

### 1. Profile: "Easy FO" (Calibration & Sanity)
**Goal:** Verify baseline system integrity and minimal overhead.
**Configuration:**
- **Atoms:** 3
- **Depth Max:** 3
- **Breadth Max:** 300
- **Total Max:** 1000

**Expected Behavior:**
- **Abstention Rate:** ~0-1% (Near Zero).
- **Runtime:** Very fast (< 5 mins).
- **Proof Velocity:** High (> 300 pph).
- **Success:** > 99% coverage.

**Scientific Interest:**
- Establishing a "clean signal" baseline. Any failure here indicates a fundamental regression in the harness, environment, or basic prover logic, rather than a learning failure.
- Calibration of "overhead" (how much time is spent on setup/teardown vs. actual proving).

---

### 2. Profile: "Medium" (The Zone of Proximal Development)
**Goal:** The ideal training regime where the model faces a mix of solvable and challenging problems.
**Configuration:**
- **Atoms:** 5
- **Depth Max:** 7
- **Breadth Max:** 1500
- **Total Max:** 8000

**Expected Behavior:**
- **Abstention Rate:** 5-15%.
- **Runtime:** Moderate (15-25 mins).
- **Proof Velocity:** Stable (~150-200 pph).
- **Success:** High enough to maintain momentum, but with enough failures to drive gradient updates (if training) or metric variance (if evaluating).

**Scientific Interest:**
- This is the "sweet spot" for RFL. It is slightly harder than the current active slice (`atoms5-depth6`), pushing the boundary of what is easily provable without hitting a wall.
- We expect to see the highest "information gain" here—failures are likely structural rather than just running out of compute (depth/breadth limits).

---

### 3. Profile: "Hard" (The Abstention Cliff)
**Goal:** Force the system into a high-abstention regime to test robustness and failure recovery.
**Configuration:**
- **Atoms:** 7
- **Depth Max:** 12
- **Breadth Max:** 3000
- **Total Max:** 15000

**Expected Behavior:**
- **Abstention Rate:** > 25% (High).
- **Runtime:** Long (> 40 mins).
- **Proof Velocity:** Low & Variable (< 100 pph).
- **Backlog:** Likely to saturate.

**Scientific Interest:**
- **Stress Testing:** Can the RFL gates handle a "failing" slice without crashing the entire pipeline?
- **Signal Extraction:** In a high-noise environment (many failures), can the few successes provide a strong signal for "breakthrough" learning?
- **Resource Bounds:** This slice effectively tests the `caps` gates (attempt mass, backlog max).

---

## Implementation Strategy

### 1. Configuration Updates (`config/curriculum.yaml`)
Add these slices to the `systems: pl: slices` list. To prevent them from blocking the main release train, they can be added *after* the currently active slice or in a separate experimental branch.

```yaml
    - name: experiment-easy-pl
      params:
        atoms: 3
        depth_max: 3
        breadth_max: 300
        total_max: 1000
      gates:
        coverage:
          ci_lower_min: 0.98
          sample_min: 10
          require_attestation: false
        abstention:
          max_rate_pct: 2.0
          max_mass: 50
        velocity:
          min_pph: 250
          stability_cv_max: 0.10
          window_minutes: 30
        caps:
          min_attempt_mass: 500
          min_runtime_minutes: 5
          backlog_max: 0.20

    - name: experiment-medium-pl
      params:
        atoms: 5
        depth_max: 7
        breadth_max: 1500
        total_max: 8000
      gates:
        coverage:
          ci_lower_min: 0.85
          sample_min: 20
          require_attestation: true
        abstention:
          max_rate_pct: 15.0
          max_mass: 800
        velocity:
          min_pph: 150
          stability_cv_max: 0.12
          window_minutes: 60
        caps:
          min_attempt_mass: 3000
          min_runtime_minutes: 20
          backlog_max: 0.40

    - name: experiment-hard-pl
      params:
        atoms: 7
        depth_max: 12
        breadth_max: 3000
        total_max: 15000
      gates:
        coverage:
          ci_lower_min: 0.70  # Relaxed for hard slice
          sample_min: 30
          require_attestation: true
        abstention:
          max_rate_pct: 30.0  # High tolerance
          max_mass: 2000
        velocity:
          min_pph: 80     # Expect slowdown
          stability_cv_max: 0.25
          window_minutes: 90
        caps:
          min_attempt_mass: 5000
          min_runtime_minutes: 40
          backlog_max: 0.60
```

### 2. Helper Functions (`curriculum/gates.py` / `curriculum/config.py`)
For ad-hoc experiments without modifying the YAML, we can add a helper factory in `curriculum/config.py`:

```python
def make_experimental_slice(profile: str = "medium") -> CurriculumSlice:
    if profile == "easy":
        # Return Easy configuration
        ...
    elif profile == "hard":
        # Return Hard configuration
        ...
    # Default/Medium
    ...
```

### 3. Harness Integration
The FO harness (likely driven by `bootstrap_metabolism.py` or `rfl_gate.py`) should accept a `--slice-profile` or `--curriculum-slice` argument that selects one of these specific named slices from the config, bypassing the standard `active_index` progression if meant for a one-off experiment.
