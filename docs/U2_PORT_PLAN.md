# U2 Architecture Port Plan

> **STATUS: PHASE II — IMPLEMENTATION PLANNING**
> 
> **Note:** No empirical uplift has been demonstrated yet. This document describes implementation planning for Phase II experiments.

This document outlines the plan for porting the U2 uplift experiment architecture from the `helpfuldolphin/mathledger` GitHub repository into the local MathLedger substrate.

## 1. File-by-File Diff Summary

### `experiments/run_uplift_u2.py`

The GitHub version of this file contains a complete, self-contained simulation of the U2 experiment, including a mock RFL policy and a hermetic verifier. The local version is a more basic script with mock metric functions. The GitHub version is far more detailed and structured.

-   **KEEP**: The overall structure of the GitHub script is excellent and should be adopted. This includes:
    -   `argparse` for command-line interface.
    -   The `U2CycleRunner` class for encapsulating cycle logic.
    -   Manifest generation (`generate_manifest`).
    -   Clear separation of concerns (validation, execution, reporting).
    -   Paired experiment mode (`--pair`).
-   **REPLACE**: The core simulation logic must be replaced with calls to the actual MathLedger substrate (`run_fo_cycles.py`).
    -   The `hermetic_verify` function is a mock and must be replaced.
    -   The `RFLPolicy` class is a mock and needs to be replaced with the real implementation if one exists, or its logic needs to be integrated with the real FO runner.
-   **INTEGRATE**: The script should be modified to call `run_fo_cycles.py` for each candidate formula. The `_run_cycle` method is the primary integration point.

### `experiments/slice_success_metrics.py`

The GitHub version provides a much more robust and formalized set of metric functions compared to the local version.

-   **KEEP**: The function signatures and the `METRIC_DISPATCHER` from the GitHub version should be adopted. The signatures are `(success, metric_value, metrics_dict)`. The dispatcher is a clean way to manage different metrics.
-   **REPLACE**: The local version's simple metric functions should be completely replaced by the more detailed and better-structured functions from the GitHub repository.
-   **INTEGRATE**: The `run_uplift_u2.py` script will use these metric functions via the dispatcher to evaluate the success of each cycle.

### `config/curriculum_uplift_phase2.yaml`

The GitHub version of the curriculum is more detailed and structured for the specific U2 experiments.

-   **KEEP**: The structure of the `slices` in the GitHub version, with `parameters`, `success_metric`, and `formula_pool_entries`, should be adopted. This is the manifest schema the user wants to preserve.
-   **REPLACE**: The local file's simpler `arithmetic_simple` and `algebra_expansion` slices should be updated to the new, more detailed format.
-   **INTEGRATE**: The `run_uplift_u2.py` script will load this config file to configure the experiments.

### `experiments/audit_uplift_u2.py`

This file was specified in the request, but it was not found in either the local codebase or the cloned GitHub repository. No action can be taken on this file.

## 2. Specific Code Blocks to Copy (KEEP)

The following components from the GitHub repository should be brought into the local codebase.

### From `experiments/run_uplift_u2.py`:

-   **Argument Parsing**: The `argparse` setup in `main()` is comprehensive.
    ```python
    parser = argparse.ArgumentParser(description=f"{PHASE_LABEL} - U2 Uplift Runner")
    mg = parser.add_mutually_exclusive_group(required=True)
    mg.add_argument("--mode", choices=sorted(VALID_MODES), help="Single mode")
    mg.add_argument("--pair", action="store_true", help="Paired experiment")
    parser.add_argument("--slice", required=True, choices=sorted(VALID_SLICES))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=MDAP_EPOCH_SEED)
    parser.add_argument("--out", type=str)
    parser.add_argument("--dry-run", action="store_true")
    ```

-   **Manifest Generation**: The `generate_manifest` function is crucial for record-keeping.
    ```python
    def generate_manifest(slice_name: str, mode: str, cycles: int, base_seed: int, output_path: Path, results: List[CycleResult], started_at: str, paired_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # ... function implementation ...
    ```

### From `experiments/slice_success_metrics.py`:

-   **Metric Dispatcher**: The `METRIC_DISPATCHER` and the `compute_metric` function provide a clean interface.
    ```python
    METRIC_DISPATCHER = {
        "goal_hit": goal_hit,
        "density": density,
        "chain_length": chain_length,
        "multi_goal": multi_goal,
    }

    def compute_metric(kind: str, **kwargs) -> Tuple[bool, float, Dict[str, Any]]:
        # ... function implementation ...
    ```

-   **Metric Function Signatures**: All metric functions should follow this pattern:
    ```python
    def goal_hit(...) -> Tuple[bool, float, Dict[str, Any]]:
        # ...
    ```

## 3. Specific Code Blocks to Replace (REPLACE)

The key change is to replace the mock simulation with calls to the real substrate.

### In `experiments/run_uplift_u2.py`:

The `hermetic_verify` function is the mock simulation.

-   **IDENTIFIED MOCK**:
    ```python
    def hermetic_verify(formula_hash: str, formula_pool_entries: List[Dict[str, Any]], rng: random.Random) -> bool:
        role = next((e.get("role", "unknown") for e in formula_pool_entries if e.get("hash") == formula_hash), "unknown")
        probs = {"axiom_k": 0.95, "axiom_s": 0.95, "axiom_contraposition": 0.95, "intermediate": 0.70, "target_peirce": 0.50, "target_transitivity": 0.50, "target": 0.50, "subgoal_1_k": 0.60, "subgoal_2_s": 0.60, "subgoal_3_chained": 0.40, "decoy": 0.30, "unknown": 0.20}
        return rng.random() < probs.get(role, 0.20)
    ```

-   **REPLACEMENT LOGIC**:
    This function should be replaced by a function that calls `run_fo_cycles.py` as a subprocess. The `_run_cycle` method in `U2CycleRunner` should be modified.

    The line to replace is inside `_run_cycle`:
    ```python
    # OLD LINE TO BE REPLACED
    if hermetic_verify(h, self.formula_pool, rng):
        verified.add(h)
    ```

    It should be replaced with something like this:
    ```python
    # NEW INTEGRATION LOGIC
    formula_to_run = next((e["formula"] for e in self.formula_pool if e["hash"] == h), None)
    if formula_to_run:
        # This is a conceptual replacement. The actual command might differ.
        # It assumes run_fo_cycles.py can take a single formula and run one cycle.
        command = [
            sys.executable,
            "run_fo_cycles.py",
            "--cycles", "1",
            "--seed", str(seed),
            "--formulas", formula_to_run
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse output of run_fo_cycles.py to check for verification.
            # This part needs to be implemented based on the actual output.
            # For now, let's assume success if the command runs.
            output_data = json.loads(result.stdout) # Assuming JSON output
            if output_data.get("verified_hashes"):
                 verified.add(h)

    ```
    This assumes `run_fo_cycles.py` can be modified or used to run a single formula for a single cycle.

## 4. Integration Test Commands

To verify the integration, a "10-cycle Calibration Fire" should be run. This will involve running the new `run_uplift_u2.py` script in paired mode for one of the slices.

```bash
# First, ensure the new curriculum is in place at config/curriculum_uplift_phase2.yaml

# Run a 10-cycle paired experiment for the slice_uplift_goal slice.
# This will run both 'baseline' and 'rfl' modes for 10 cycles each.
python experiments/run_uplift_u2.py --pair --slice slice_uplift_goal --cycles 10

# After the run, check the generated manifest file for the results:
# results/uplift_u2_manifest_slice_uplift_goal.json

# Also, check the raw output files:
# results/uplift_u2_slice_uplift_goal_baseline.jsonl
# results/uplift_u2_slice_uplift_goal_rfl.jsonl
```
