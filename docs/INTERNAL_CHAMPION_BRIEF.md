# MathLedger — Internal Champion Brief

**Classification**: Internal Forwarding Document
**Audience**: Research leadership, safety teams, governance engineering
**Date**: 2025-12-21

---

## One-Line Summary

MathLedger is a verifiable learning substrate that provides cryptographic attestation and fail-close governance for AI training feedback loops.

---

## What the Demo Proves

A skeptical engineer can verify the following in under 10 minutes:

| Property | Verification |
|----------|--------------|
| **Deterministic execution** | Same seed produces byte-identical outputs |
| **Dual attestation** | Reasoning (R_t) + UI (U_t) streams bound to composite root H_t |
| **Composite integrity** | H_t = SHA256(R_t \|\| U_t), independently verifiable |
| **Fail-close governance** | F5.x predicates trigger claim cap when conditions are out of bounds |
| **Replayability** | Full manifest and event streams captured for audit |

**Run command:**
```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
cd demo_output && python verify.py
```

---

## What This Explicitly Does NOT Claim

| Non-Claim | Reason |
|-----------|--------|
| Capability or learning performance | Outside scope; governance stability only |
| Convergence guarantees | Not tested in this demo |
| Threshold optimality | Thresholds frozen, not evaluated |
| Generalization beyond tested conditions | Finite perturbation set |
| Production readiness | Demo is observational (SHADOW mode) |

---

## Why This Is Hard to Rebuild Correctly

Building a verifiable learning substrate requires solving several non-obvious problems simultaneously:

1. **Dual-root binding**: Separate reasoning and UI event streams must be cryptographically bound without either being forgeable after the fact. Getting the hash construction wrong creates audit gaps.

2. **Determinism enforcement**: Every component (timestamps, UUIDs, random seeds, JSON serialization, floating-point precision) must be controlled. A single source of nondeterminism breaks reproducibility.

3. **Fail-close governance**: Predicates must trigger conservatively under uncertainty. Most systems fail-open (claim validity when unsure). Fail-close requires explicit design for every edge case.

4. **Toolchain fingerprinting**: Reproducibility requires capturing the exact dependency graph (Python, Lean, lock files). Partial fingerprints create hidden variance.

5. **Domain separation**: Hash functions must use domain prefixes to prevent cross-context collisions. Without this, reasoning events could be replayed as UI events.

These constraints compound. A system that gets 4 of 5 correct still has an exploitable gap.

---

## Artifact Reference

| Document | Purpose |
|----------|---------|
| `scripts/run_dropin_demo.py` | Single-command demo orchestrator |
| `docs/DROPIN_REPLAY_INSTRUCTIONS.md` | Third-party replay guide |
| `demo_output/` | Sample output bundle (seed=42) |
| `docs/DROPIN_DEMO_FREEZE.md` | Freeze declaration |

---

## Related Work

| Artifact | Path |
|----------|------|
| arXiv paper | `docs/PAPERS/arxiv/main.tex` |
| Phase II Governance Memo | `docs/system_law/calibration/PHASE_II_GOVERNANCE_STABILITY_MEMO.md` |
| Phase II Diligence One-Pager | `docs/system_law/calibration/PHASE_II_DILIGENCE_ONEPAGER.md` |

---

## Contact

For questions about this artifact, contact the repository maintainer.

---

**Status**: FROZEN — Safe for internal forwarding.
