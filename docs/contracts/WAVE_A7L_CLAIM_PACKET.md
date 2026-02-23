# Wave A7L Claim Packet

## Claim A7L

Claim ID: `ML-A7L-LIVE-MODEL-LAYER`  
Tier: B  
Statement: A thin live-model harness can execute scenario prompts against OpenRouter models, classify raw responses into CPF indicator activations, and route those activations through the existing A7 governance pipeline while emitting replay-ready artifacts.

Falsifier:
- live completion payload cannot be normalized into CPF assessments,
- captured responses cannot be routed into `run_three_lane_experiment`,
- capture/evidence artifacts are not emitted for each step.

Closure artifact targets:
- `src/mathledger/integration/live_model_layer.py`
- `tests/test_wave_a7_live_model_layer.py`

## Invariants Enforced in A7L

1. Thin integration only:
- no extraction/import from legacy governance code paths.

2. Capture discipline:
- request/response/classification events are hash-chained in capture logs.

3. Routing continuity:
- classifier assessments are normalized and fed into A7 unchanged in structure.

4. Replay compatibility:
- each live step still emits Lane-A evidence pack + Lane-B bridge + Lane-C snapshot artifacts.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `51 passed`

## Scope Caveat

This packet validates the live harness mechanics and deterministic routing via mock clients.  
A real OpenRouter execution requires `OPENROUTER_API_KEY` at runtime and produces runtime-dependent model behavior.
