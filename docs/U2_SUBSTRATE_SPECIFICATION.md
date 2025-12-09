# PHASE II — U2 Substrate Governance Specification
**Document ID:** U2_SUBSTRATE_SPEC_V1  
**Author:** Gemini C, Substrate Systems Architect  
**Status:** ACTIVE  
**Mandate:** U2 SUBSTRATE GOVERNANCE SPEC ENGINE  

This document provides the formal specification for the Substrate Abstraction Layer (SAL) within the Phase II U2 Uplift Runner experimental harness. All current and future substrate implementations **MUST** adhere to this specification to ensure determinism, reproducibility, and modularity.

---

## 1. Formal ABC and Data Class Definitions

The Substrate Abstraction Layer is defined by an Abstract Base Class (ABC), `Substrate`, and a standardized return object, `SubstrateResult`.

### 1.1. `Substrate` Abstract Base Class

All substrate implementations **MUST** inherit from `experiments.substrate.Substrate` and implement the following method signature precisely.

```python
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SubstrateResult:
    """Standardized result from a substrate execution."""
    success: bool
    result_data: Dict[str, Any]
    verified_hashes: List[str] = field(default_factory=list)
    error: str | None = None

class Substrate(abc.ABC):
    """Abstract Base Class for all substrates."""
    
    @abc.abstractmethod
    def execute(self, item: str, cycle_seed: int) -> SubstrateResult:
        """
        Executes a given item on the substrate.
        Must be deterministic based on the item and cycle_seed.
        """
        raise NotImplementedError
```

### 1.2. `SubstrateResult` Data Class

The `execute` method **MUST** return an instance of `SubstrateResult`. The fields are defined as follows:

- **`success` (bool):** The primary outcome of the execution. `True` indicates that the substrate successfully achieved its goal (e.g., verified a proof). `False` indicates failure.
- **`result_data` (Dict[str, Any]):** A JSON-serializable dictionary containing raw data and metadata from the substrate run. This data is embedded in the manifest for full auditability.
- **`verified_hashes` (List[str]):** A list of unique identifiers (hashes) for all statements/lemmas/theorems successfully verified during the execution. This is the primary input for `goal_hit` and other success metrics.
- **`error` (str | None):** If execution fails non-deterministically (e.g., script crash, timeout), this field **MUST** contain a descriptive error message. It should be `None` on a successful run.

---

## 2. Substrate Error Taxonomy

Errors originating from the substrate layer are categorized for debugging and reporting.

### Configuration and Initialization Errors (SUB-1 to SUB-10)
- **SUB-1:** Substrate script not found (e.g., `run_fo_cycles.py` is not in the expected path).
- **SUB-2:** Invalid Python executable path for a script-based substrate.
- **SUB-3:** Missing permissions to execute a substrate script.

### Execution Errors (SUB-11 to SUB-20)
- **SUB-11:** Substrate process timed out (violation of execution budget).
- **SUB-12:** Substrate process returned a non-zero exit code (`subprocess.CalledProcessError`).
- **SUB-13:** Substrate process terminated by signal (e.g., OOM killer).
- **SUB-14:** Catastrophic internal substrate error (e.g., unhandled exception within the substrate script itself).

### Parsing and Serialization Errors (SUB-21 to SUB-30)
- **SUB-21:** Substrate `stdout` cannot be decoded as `utf-8`.
- **SUB-22:** Substrate `stdout` is not valid JSON (`json.JSONDecodeError`).
- **SUB-23:** Substrate JSON output is missing required fields (e.g., `outcome` or `verified_hashes`).

### Contract and Determinism Violations (SUB-31 to SUB-40)
- **SUB-31:** Substrate produced a different `SubstrateResult` for the same `(item, cycle_seed)` pair during a replay. **This is the most severe error category.**
- **SUB-32:** Substrate produced non-JSON-serializable data in `result_data`.
- **SUB-33:** Substrate exhibited a forbidden side effect (see Section 4).

---

## 3. Determinism Contract

Determinism is the foundational principle of the U2 experimental harness. Every substrate **MUST** guarantee that for a given pair of inputs `(item: str, cycle_seed: int)`, it always produces an identical `SubstrateResult` object.

- **For `MockSubstrate`:** Determinism is guaranteed by seeding all internal pseudorandom operations (e.g., mock hash generation) with a combination of the `cycle_seed` and `hash(item)`.

- **For `FoSubstrate`:** This substrate acts as a deterministic proxy. It guarantees its own determinism by passing the `cycle_seed` to the external script. It transfers the determinism obligation to the external script (`run_fo_cycles.py`), which **MUST** in turn guarantee that its output to `stdout` is based solely on its input arguments.

- **For `LeanSubstrate` (Future):** The `LeanSubstrate` will be obligated to pass the `cycle_seed` to the `lean` process. Any stochastic elements within the Lean tactics or proof search **MUST** be seeded from this value. Any file-based output from Lean **MUST** be written to a temporary, seed-named directory to avoid collisions.

---

## 4. Forbidden Behaviors

A substrate implementation **MUST NOT**:

1.  **Use Global Randomness:** Call any function that relies on a global, unseeded random number generator (e.g., `random.random()`). All stochasticity **MUST** be derived from a `random.Random` instance seeded with the `cycle_seed`.
2.  **Introduce Nondeterministic I/O:** Perform arbitrary network calls or read from file paths that are not explicitly passed in and tracked. Any I/O must be fully determined by the input arguments.
3.  **Produce Unmanaged Side Effects:** Modify any files outside of a temporary, isolated directory. It must not modify the RFL policy, the curriculum configuration, or any other part of the runner's state. Its sole purpose is to execute its task and return a `SubstrateResult`.
4.  **Rely on System-Dependent State:** Depend on system time, environment variables (unless explicitly documented), process IDs, or other volatile system state to generate its result.

---

## 5. Substrate Capability Matrix

| Capability              | MockSubstrate                            | FoSubstrate                              | LeanSubstrate (Future)                   | ExternalSubstrate (Future)             |
| ----------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | -------------------------------------- |
| **Determinism**         | Guaranteed (Seeded `eval`/`hash`)        | Guaranteed (Seeded external script)      | **Required** (Seeded Lean process)       | **Required** (Seeden API call/job)     |
| **Resource Cost**       | Extremely Low                            | Low (Python script overhead)             | High (Lean compiler/interpreter)         | Variable (Network latency, queue time) |
| **Proof Depth**         | None (Simulation)                        | Shallow (Simulated FO logic)             | Deep (Real, verifiable proofs)           | Arbitrary (Depends on endpoint)        |
| **Failure Behavior**    | Python `Exception`                       | `CalledProcessError`, `JSONDecodeError`  | Lean Timeout, Lean Error Output          | HTTP Error, Timeout, API Error Body    |
| **Dependency Guarantees** | Python Built-ins                         | `run_fo_cycles.py` script                | Lean 4 installation + `mathlib`          | Network access + API specification   |

---

## 6. Migration Plan to `LeanSubstrate`

The following is a high-level plan for integrating a live Lean 4 substrate into the U2 harness.

### Phase 1: Interface and Tooling
- **Action:** Finalize the communication method with Lean. A `subprocess`-based approach calling `lean --run MyFile.lean <args>` is recommended for initial simplicity and alignment with `FoSubstrate`.
- **Action:** Define the `stdout` contract for Lean scripts. A Lean script run by the substrate **MUST** print a single JSON object to `stdout` containing the fields required by `SubstrateResult` (`success`, `verified_hashes`, etc.). A helper function or `meta` program within Lean should be developed to facilitate this.

### Phase 2: `LeanSubstrate` Implementation
- **Action:** Implement the `LeanSubstrate` class in `experiments/substrate.py`.
- **Action:** The `execute` method will construct the `lean --run` command, including passing the `cycle_seed` and `item` information.
- **Action:** The method will invoke the subprocess, capture the output, and parse the resulting JSON into a `SubstrateResult` object, including robust error handling for Lean compile/runtime errors.

### Phase 3: Curriculum and Metric Updates
- **Action:** Update `config/curriculum_uplift_phase2.yaml` with new slices containing items that are valid inputs for the `LeanSubstrate` (e.g., Lean file paths or theorem names).
- **Action:** Implement new success metrics in `slice_success_metrics.py` capable of evaluating the output from real Lean proofs (e.g., checking if a specific theorem hash appears in `verified_hashes`).

### Phase 4: Testing and Validation
- **Action:** Create `tests/test_lean_substrate.py`.
- **Action:** Write unit tests that mock `subprocess.run` and provide sample `stdout` from a real Lean process to test the `LeanSubstrate`'s parsing logic.
- **Action:** Write a small number of integration tests that call a real, simple, and fast-running Lean file to verify the end-to-end workflow.

### Phase 5: Benchmarking and Deployment
- **Action:** Run paired experiments (`--pair`) comparing `MockSubstrate` vs. `LeanSubstrate` on a simple curriculum.
- **Action:** This will validate the integration, benchmark the resource cost (time, CPU, memory), and confirm the determinism of the new substrate before it is used for formal RFL experiments.
