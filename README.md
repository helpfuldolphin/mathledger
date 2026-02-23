# MathLedger

Clean-room MathLedger implementation built through ACE claim promotion waves.

Current implementation scope:
- Wave A0: repository bootstrap and deterministic test harness
- Wave A1: basis genome primitives (deterministic core only)
- Wave A2: governance kernel (trust-class routing, ADV quarantine, typed outcomes)
- Wave A3: UVIL boundary (exploration to authority fail-closed commit firewall)
- Wave A4: evidence pack replay (deterministic U_t/R_t/H_t recomputation + tamper detection)
- Wave A5: AAK bridge (Lane-B references to Lane-A roots without authority escalation)
- Wave A6: CPF adapter (deterministic CPF assessment normalization into snapshot artifacts)
- Wave A7: experiment runner (reproducible three-lane orchestration with boundary checks)
- Wave A7R: scenario suite (published adversarial + benign controls with auditable result tables)
- Wave A7L: live model layer (OpenRouter prompt/response capture + CPF text scoring -> governance routing)

Out of scope in this wave:
- TDA/USLA advisory sidecar (A8+)
- authority promotion of TDA/USLA gating (A9+)

## Quickstart

```bash
cd C:\dev\mathledger
python -m pip install -e ".[dev]"
pytest -q
$env:PYTHONPATH='src'; python -m mathledger.integration.scenario_suite --output docs/results/scenario_suite --fail-on-mismatch
$env:OPENROUTER_API_KEY='<your_key>'
$env:PYTHONPATH='src'; python -m mathledger.integration.live_model_layer --model openai/gpt-4o-mini --default-scenario-limit 3 --output docs/results/live_model_suite
```

## Structure

- `docs/contracts/CLAIM_PROMOTION_REGISTER.md`: canonical claim registry
- `src/mathledger/basis`: deterministic genome primitives
- `src/mathledger/uvil`: UVIL commit boundary and immutable authority artifacts
- `src/mathledger/evidence`: evidence-pack build/replay verifier primitives
- `src/mathledger/integration`: AAK bridge, CPF adapter, and three-lane experiment runner
- `docs/results/scenario_suite/results.md`: scenario-level governance outcome table with replay artifacts
- `docs/results/live_model_suite/`: live-model execution artifacts (when OpenRouter key is configured)
- `docs/presentation/GOVERNANCE_PIPELINE_VALIDATION_CALL_BRIEF.md`: call-ready narrative and caveat framing
- `docs/releases/v0.1.0-governance-validated/`: reproducible release notes + SHA256 manifest
- `tests/test_wave_a1_basis_core.py`: Wave A1 invariants
