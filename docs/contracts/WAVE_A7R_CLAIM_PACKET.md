# Wave A7R Claim Packet

## Claim A7R

Claim ID: `ML-A7R-SCENARIO-SUITE`  
Tier: B  
Statement: The A7 pipeline executes a fixed set of published adversarial scenarios and benign controls, producing deterministic governance outcomes and per-step sealed evidence artifacts.

Falsifier:
- CEO fraud sequence does not progress `YELLOW -> RED -> RED`.
- CAC scenario does not trigger both risk and convergence gates to `RED`.
- benign controls produce non-`GREEN` outcomes.
- scenario execution omits evidence artifacts (`run_result`, `evidence_pack`, `aak_bridge_packet`, `gate_evaluation`).

Closure artifact targets:
- `src/mathledger/integration/scenario_suite.py`
- `tests/test_wave_a7_scenario_suite.py`
- `docs/results/scenario_suite/results.json`
- `docs/results/scenario_suite/results.md`

## Invariants Enforced in A7R

1. Scenario provenance binding:
- adversarial fixtures include source document path, section, and line anchor.

2. Deterministic gate evaluation:
- risk and convergence gates are derived from normalized CPF scores and compared to expected outcomes.

3. Replay-ready artifacts:
- each scenario step emits deterministic Lane-A evidence and Lane-B bridge artifacts.

4. Explicit expectation matching:
- per-step outcomes are compared against expected `risk_gate`, `convergence_gate`, and `overall_decision`.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `49 passed`
- Command: `$env:PYTHONPATH='src'; python -m mathledger.integration.scenario_suite --output docs/results/scenario_suite --fail-on-mismatch`
- Result: `Matched steps: 11/11; mismatches=0`
