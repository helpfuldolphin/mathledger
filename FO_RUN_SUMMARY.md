### FO_RUN_SUMMARY.md

Summary of First Organism (FO) Runner experiments:

Phase I results demonstrate hermetic execution, not uplift.

#### Degenerate RFL Run Notice
The sealed `fo_rfl.jsonl` contains 1000 cycles with 100% abstention. This is not evidence of RFL uplift.

| Run                   | Cycles | Mean Abstention | Mode Check | Cycle Range | Roots Present |
| :-------------------- | :----- | :-------------- | :--------- | :---------- | :------------ |
| results\fo_baseline.jsonl | 1000 | 0.000 | Pass | 0 to 999 (Pass) | Pass |
| results\fo_rfl.jsonl | 1000 | 1.000 | Pass | 0 to 999 (Pass) | Pass |

*Note on baseline:* `fo_baseline.jsonl` represents deterministic stagnation, where the system consistently produces the same (non-abstaining) outcome without RFL intervention.
