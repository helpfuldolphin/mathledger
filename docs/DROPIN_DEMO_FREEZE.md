# Drop-In Governance Demo — Freeze Declaration

**Classification**: Acquisition Diligence Artifact
**Status**: FROZEN
**Freeze Date**: 2025-12-21
**Commit**: 6631592

---

## Freeze Scope

The following artifacts are frozen and SHALL NOT be modified without explicit authorization:

| Artifact | Path | SHA-256 (first 16) |
|----------|------|-------------------|
| Demo Orchestrator | `scripts/run_dropin_demo.py` | (commit-locked) |
| Replay Instructions | `docs/DROPIN_REPLAY_INSTRUCTIONS.md` | (commit-locked) |
| Sample Bundle | `demo_output/` | (commit-locked) |
| Manifest | `demo_output/manifest.json` | (commit-locked) |

---

## What This Artifact Is

- A **completed** demonstration of MathLedger as a drop-in governance substrate
- **Self-contained**: No database, no external services, runs offline
- **Deterministic**: Same seed produces byte-identical outputs
- **Verifiable**: Includes standalone verification script
- **Documented**: Includes third-party replay instructions (<10 minutes)

---

## What This Artifact Is NOT

- NOT a prototype or work-in-progress
- NOT evolving or subject to iterative development
- NOT dependent on future work for validity
- NOT making claims beyond what is demonstrated

---

## Canonical Verification Command

```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
cd demo_output && python verify.py
```

Expected output:
```
[PASS] Composite root verified: H_t == SHA256(R_t || U_t)
```

---

## Forward-Safe

This artifact is safe to forward internally for:
- Research leadership review
- Governance engineering evaluation
- Acquisition due diligence
- Safety team assessment

---

## Change Control

Any modification to frozen artifacts requires:
1. Explicit authorization with documented rationale
2. New freeze declaration with updated commit hash
3. Re-verification of all determinism properties

---

**FROZEN** — No modifications without authorization.
