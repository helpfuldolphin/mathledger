# Lean Swap Plan: Operational Instructions

**Version**: 1.0.0  
**Author**: Manus-F, Deterministic Planner & Substrate Executor Engineer  
**Date**: 2025-12-06  
**Status**: Final

---

## 1. Overview

This document provides the minimal, precise operational instructions for replacing the MVP `PropositionalVerifier` (which uses truth tables) with the production **Lean 4 theorem prover** inside the `FOSubstrateExecutor`. This is a critical step for enabling the U2 Planner to reason about First-Order Logic.

This plan is designed to be executed with **no changes to the core `FOSubstrateExecutor` pipeline**. The swap is confined to a new `LeanExecutor` class that implements the same interface as the `PropositionalVerifier`.

## 2. Command Protocol

The `LeanExecutor` will interact with the Lean 4 prover via a standard command-line interface. The interaction **MUST** follow this protocol:

1.  **Input**: The executor will write a temporary Lean source file (`.lean`) containing the statement to be proven.
2.  **Execution**: The executor will invoke the `lean` command-line tool as a subprocess, pointing it to the generated source file.
3.  **Output**: The executor will capture the `stdout` and `stderr` from the `lean` process.
4.  **Parsing**: The executor will parse the output to determine the outcome (proof found, timeout, error).

### Lean Source File Template

For each statement, a `.lean` file will be generated using the following template:

```lean
-- U2 Planner Verification Request
-- Statement Hash: {statement_hash}

#check ({statement_normalized})
```

-   `{statement_hash}`: The SHA-256 hash of the normalized statement.
-   `{statement_normalized}`: The statement in Lean-compatible syntax.

### Invocation Command

The `lean` process **MUST** be invoked with a timeout to prevent stalled proofs from blocking the executor.

```bash
timeout {timeout_seconds}s lean {path_to_lean_file}.lean
```

-   `{timeout_seconds}`: The maximum time allowed for the proof search (e.g., 5 seconds).
-   `{path_to_lean_file}`: The path to the generated `.lean` file.

## 3. Error Semantics

The `LeanExecutor` **MUST** map the output of the `lean` process to the `FOSubstrateExecutor`'s existing error taxonomy. This ensures that the U2 runner can handle Lean-specific outcomes without modification.

| Lean Process Outcome | `stdout` / `stderr` Content | Mapped `ExecutionResult` Outcome | Description |
| :--- | :--- | :--- | :--- |
| **Success** | Contains the normalized statement type (e.g., `p → q → p : Prop`) | `"success"` | Lean successfully proved the statement is a tautology. |
| **Failure** | Contains `error: type mismatch` or similar type error | `"failure"` | The statement is not a valid proposition (e.g., ill-formed). |
| **Timeout** | Process exits with a timeout signal (e.g., exit code 124) | `"error"` with `TransientError` | The proof search exceeded the time limit. May succeed with more time. |
| **Syntax Error** | Contains `error: expected ...` | `"error"` with `PermanentError` | The statement has a syntax error and cannot be parsed by Lean. |

### `ExecutionResult` Mapping

-   If the outcome is `"success"`, `is_tautology` **MUST** be set to `true`.
-   If the outcome is `"failure"`, `is_tautology` **MUST** be set to `false`.
-   If the outcome is `"error"`, an appropriate `TransientError` or `PermanentError` **MUST** be raised.

## 4. Tactic Metadata Contract

To provide insight into the proof search, the `LeanExecutor` can optionally extract metadata about the tactics used by Lean. This is **not required for the initial swap** but is specified here for future RFL feature extraction.

If the `lean` command is run with a verbose or debug flag, it may output the sequence of tactics used to find a proof. This output can be parsed to populate a new field in the `ExecutionResult`:

`data.tactic_metadata`: A list of strings representing the tactics used.

**Example**:

```json
"tactic_metadata": ["intro", "exact"]
```

This field will be `null` or an empty list if no tactic information is available. The presence of this metadata will allow future versions of the RFL `FeatureExtractor` to learn which tactics are effective for different types of statements.

## 5. Implementation Plan (No Code)

1.  **Create `LeanExecutor` Class**:
    -   Create a new Python file: `backend/u2/lean_executor.py`.
    -   Define a class `LeanExecutor` with an `execute(statement)` method that matches the interface of `PropositionalVerifier`.

2.  **Implement Command Protocol in `execute` Method**:
    -   Accept a `StatementRecord` object.
    -   Generate the `.lean` source file from the template.
    -   Use Python's `subprocess.run` to invoke the `lean` command with a timeout.
    -   Capture `stdout`, `stderr`, and the return code.

3.  **Implement Error Semantics**:
    -   Parse the captured output to determine the outcome based on the table in Section 3.
    -   Construct and return an `ExecutionResult` object with the correct `outcome`, `is_tautology`, and error type.

4.  **Update `FOSubstrateExecutor`**:
    -   In `fosubstrate_executor.py`, modify the `__init__` method.
    -   Replace the line `self.verifier = PropositionalVerifier()` with `self.verifier = LeanExecutor()`.

**No other changes are required.** The `FOSubstrateExecutor`'s core logic remains untouched, demonstrating the modularity of the design.
