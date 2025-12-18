# Determinism Claims

**Version**: 1.0
**Last Updated**: 2025-12-18

This document scopes what MathLedger claims to be deterministic, under what conditions, and what is explicitly not claimed.

---

## Determinism Claims Table

| Claim | Scope | Mode | Verification Command |
|-------|-------|------|---------------------|
| Mock harness produces identical outputs given identical seed | First Organism test harness | Mock (no Lean) | `make verify-mock-determinism` |
| UI event → derivation → seal pipeline is reproducible | Closed-loop path simulation | Mock (no Lean) | `uv run python scripts/verify_first_light_determinism.py --mode mock` |
| `H_t = SHA256(R_t \|\| U_t)` computation is deterministic | Dual-root attestation formula | Both | Unit tests in `tests/test_dual_root_attestation.py` |
| Canonical formula normalization produces identical output | `backend.logic.canon.normalize()` | N/A (pure Python) | Unit tests in `tests/test_canon.py` |
| Evidence pack manifest signing produces identical signatures | Ed25519 detached signatures | N/A (pure Python) | `uv run pytest tests/evidence/test_manifest_signing*.py -v` |

*Note: Verification commands use a fixed internal seed (42) for reproducibility. The seed is hardcoded in the harness scripts; no CLI flag is required.*

---

## Mode Definitions

| Mode | Description |
|------|-------------|
| **Mock** | Simulated Lean verification. No real Lean toolchain required. Tests pipeline determinism without proof checking. |
| **Real Lean** | Actual Lean 4 type-checking. Requires Lean installation. Verifies proofs are accepted by Lean. |
| **N/A** | Pure Python computation with no external dependencies. |

---

## What Is NOT Claimed

The following are explicitly **not** claimed to be deterministic:

| Item | Reason |
|------|--------|
| Wall-clock timestamps in logs | Logging timestamps in `rfl/` are metadata for human readability, not inputs to attestation hashes. |
| Run IDs without explicit seed | When no `--seed` is provided, run IDs include timestamps for uniqueness. |
| Lean compilation time or resource usage | Only Lean outputs (success/failure, stdout) are deterministic inputs; timing is not. |
| Order of unrelated log entries | Log interleaving may vary; only hash-relevant outputs are deterministic. |
| CI workflow execution time | Infrastructure variability is expected. |

---

## Verification Notes

1. **Mock mode tests the harness, not Lean**: Passing `--mode mock` verifies that the Python pipeline produces identical outputs. It does not verify that Lean itself is deterministic. We treat Lean as an external verifier; we do not attempt to prove Lean's determinism. (Lean's determinism is assumed as an external property of the Lean 4 toolchain and is not asserted or tested by MathLedger.)

2. **Real Lean mode requires setup**: The optional `make verify-lean-single` command requires a working Lean 4 installation. First-time setup downloads ~2GB of dependencies and takes 30–60 minutes.

3. **Seed requirement**: Determinism claims apply when an explicit seed is provided. Without a seed, timestamps and random elements may vary.

---

*This document is authoritative for determinism claim scope. Claims not listed here are not made.*
