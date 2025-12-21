# MathLedger

MathLedger is a governance substrate for verifiable learning systems.
It provides deterministic execution, dual attestation, and fail-closed
governance with replayable audit artifacts.

## Quick Start (10 minutes)

```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
cd demo_output && python verify.py
```

Expected output:
```
[PASS] Composite root verified: H_t == SHA256(R_t || U_t)
```

## What This Demonstrates

- **Deterministic execution**: Same seed produces byte-identical outputs
- **Dual attestation**: R_t (reasoning) + U_t (UI) bound to composite H_t
- **Fail-closed governance**: F5.x predicates trigger claim cap when conditions are out of bounds
- **Independent replayability**: All inputs/outputs captured for external audit

## What This Does NOT Claim

- Capability or learning performance
- Convergence guarantees
- Production readiness

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

No database or external services required. The demo runs offline.

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/DROPIN_REPLAY_INSTRUCTIONS.md` | Third-party verification guide |
| `docs/DROPIN_DEMO_FREEZE.md` | Freeze declaration |
| `docs/INTERNAL_CHAMPION_BRIEF.md` | One-page technical summary |

## License

MIT
