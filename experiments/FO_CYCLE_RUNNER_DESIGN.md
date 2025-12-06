# First Organism Cycle Runner Design

## 1. Overview
The First Organism (FO) Cycle Runner is a standalone harness designed to execute the closed-loop First Organism cycle repeatedly and deterministically. Its primary purpose is to generate longitudinal data on the organism's behavior, specifically the interplay between UI events, curriculum gates, derivation/abstention, dual attestation, and Reflexive Forest Learning (RFL) metabolism.

## 2. Objectives
- **Repetition:** Run `N` cycles back-to-back.
- **Hermeticity:** Ensure each cycle is independent and deterministic, with no leakage of state (except for the explicit RFL policy ledger if enabled).
- **Modes:** Support `baseline` (RFL OFF) and `rfl` (RFL ON) modes.
- **Observability:** Log rich, machine-readable metrics (JSONL) for analysis.

## 3. Architecture

### 3.1. Components
The runner reuses the existing canonical components verified by `tests/integration/test_first_organism.py`:
1.  **UI Event Capture:** `ledger.ui_events`
2.  **Curriculum Gates:** `curriculum.gates`
3.  **Derivation Engine:** `derivation.pipeline`
4.  **Dual Attestation:** `attestation.dual_root` & `ledger.blocking`
5.  **RFL Metabolism:** `rfl.runner`

### 3.2. Data Flow (Per Cycle)
1.  **Setup:** Reset internal stores (`ui_event_store`), set deterministic seeds based on `cycle_index`.
2.  **UI Phase:** Generate a deterministic "User Action" (e.g., `select_statement`) and capture it. This produces $U_t$.
3.  **Gate Phase:** Evaluate curriculum gates for the `first-organism` slice.
4.  **Derivation Phase:** Run the derivation pipeline with First Organism bounds. Produce a candidate (expected to be an abstention).
5.  **Attestation Phase:** Seal the "block" containing the proof (abstention) and the UI event. This computes $R_t$ (Reasoning Root) and $H_t = SHA256(R_t || U_t)$.
6.  **RFL Phase (Conditional):**
    -   If `mode=rfl`: Feed the attested block context ($H_t$) into the `RFLRunner`.
    -   Verify metabolism (policy update, symbolic descent).
7.  **Logging:** Emit a JSON record with all roots and metrics.

### 3.3. Determinism Strategy
To ensure reproducibility without full DB resets, we rely on MDAP (MathLedger Deterministic Artifact Protocol) principles:
-   **Seeds:** The seed for each cycle is derived from `MDAP_EPOCH_SEED + cycle_index`.
-   **IDs:** All event IDs and timestamps are derived from content hashes or the cycle seed.
-   **Isolation:** In-memory stores (like `ui_event_store`) are cleared at the start of each cycle.
-   **Mocking:** Database interactions (loading baselines) are mocked to prevent external state dependency.

## 4. CLI Interface

The script `experiments/run_fo_cycles.py` exposes the following arguments:

```bash
usage: run_fo_cycles.py [-h] [--mode {baseline,rfl}] [--cycles CYCLES] [--out OUT]

Run First Organism cycles.

optional arguments:
  -h, --help            show this help message and exit
  --mode {baseline,rfl}
                        Execution mode (default: baseline)
  --cycles CYCLES       Number of cycles to run (default: 10)
  --out OUT             Output JSONL file path (default: results.jsonl)
```

## 5. Output Schema
Each line in the output JSONL file follows this structure:

```json
{
  "cycle": 0,
  "mode": "rfl",
  "timestamp": "2025-11-27T10:00:00+00:00",
  "roots": {
    "h_t": "a1b2...",
    "r_t": "c3d4...",
    "u_t": "e5f6..."
  },
  "derivation": {
    "candidates": 1,
    "abstained": 1,
    "verified": 0,
    "statement_hash": "1234..."
  },
  "rfl": {
    "executed": true,
    "policy_update": true,
    "symbolic_descent": 0.123,
    "abstention_breakdown": {"lean_failure": 1}
  }
}
```

## 6. Implementation Notes
-   **Imports:** Direct imports from `derivation`, `ledger`, `attestation`, and `rfl` packages.
-   **Mocking:** `unittest.mock.patch` is used to bypass `rfl.runner.load_baseline_from_db`.
-   **Helpers:** Minimal reproduction of `conftest` helpers where necessary to avoid complex test-suite dependencies.

## 7. Future Extensions
-   **Varying Input:** Introduce randomized (but deterministic) variations in the UI event or derivation seeds to stress-test the RFL policy.
-   **Performance Profiling:** Add timing metrics to the log output.
