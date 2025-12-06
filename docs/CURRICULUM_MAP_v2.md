# Curriculum Map v2: Complexity vs. Abstention Regimes

This document maps the Propositional Logic (PL) curriculum slices against their expected complexity and abstention behavior. It serves as a visual guide for selecting the appropriate experimental regime for RFL (Reflexive Formal Learning) tests.

## The Complexity-Abstention Landscape

The chart below visualizes where each slice sits on the "difficulty curve." As the structural complexity of the logic problems increases (more atoms, deeper derivation trees), the prover's certainty decreases, leading to higher abstention rates.

```ascii
Abstention Rate (%)
   ^
   |
35+|                                                 [slice_hard]
   |                                               (atoms:7, depth:12)
30+|                                              /
   |                                             /
25+|                                            /
   |                                  [first_organism_pl2_hard]
20+|                                     (atoms:6, depth:8)
   |                                    /
15+|                         [slice_medium]
   |                        (atoms:5, depth:7)
10+|                 [atoms5-depth6]
   |               (Current Active)
 5+|
   | [slice_easy_fo]
 0+|_(atoms:3, depth:3)__________________________________________>
     Low (Trivial)       Medium (Training)       High (Stress)
                     Complexity (Atoms × Depth)
```

## Slice Definitions & Regimes

### 1. Trivial Regime (Calibration)
*   **Slice:** `slice_easy_fo`
*   **Params:** Atoms: 3, Depth: 3
*   **Role:** Sanity check.
*   **Behavior:** The system should solve these instantly with near-zero abstention. Any failure here indicates a broken harness, not a learning failure.

### 2. Training Regime (The "Sweet Spot")
*   **Slices:** `atoms5-depth6` (Active), `slice_medium`
*   **Params:** Atoms: 5, Depth: 6-7
*   **Role:** Core learning zone.
*   **Behavior:** The prover encounters a healthy mix of solvable proofs and challenging edge cases. Abstention rates (5-15%) provide enough "negative signal" to drive gradient updates or metric variance without stalling the pipeline.

### 3. Stress Regime (The "Abstention Cliff")
*   **Slices:** `first_organism_pl2_hard`, `slice_hard`
*   **Params:** Atoms: 6-7, Depth: 8-12
*   **Role:** Robustness testing.
*   **Behavior:** The problem space explodes combinatorially. The prover enters a high-uncertainty state (>20% abstention). Success relies heavily on efficient search heuristics and "intuition." This regime tests the RFL gates' ability to handle failure gracefully (e.g., backpressure, queue management).

## Configuration Reference
*   **File:** `config/curriculum.yaml`
*   **System:** `pl`
*   **Usage:** Select slices via the `active` field or experimental flags in the harness.