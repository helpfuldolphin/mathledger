# CAL-EXP-3 Evidence Pack

**Status**: Not included in this repository

---

## What This Directory Contains

This directory is a placeholder for consistency with the arXiv paper (Appendix A).

The CAL-EXP-3 evidence pack (experimental manifests, cryptographic hashes, run artifacts) will be published as **ancillary material** with the arXiv submission or in a separate evidence repository.

---

## What IS Included in This Repository

This repository contains the **drop-in governance demo**, which demonstrates:

- Deterministic execution (same seed produces byte-identical outputs)
- Dual attestation (R_t + U_t bound to composite H_t)
- Fail-closed governance (F5.x predicates trigger claim cap)
- Independent replayability (all inputs/outputs captured for audit)

**To run the demo:**
```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
cd demo_output && python verify.py
```

---

## What IS NOT Included

| Artifact | Reason |
|----------|--------|
| CAL-EXP-3 run manifests | Will be published as ancillary material |
| Phase II governance data | Specification frozen; execution pending |
| Internal experiment logs | Not for public distribution |

---

## Verification

The drop-in demo (`demo_output/`) provides a self-contained verification bundle with:
- `manifest.json`: Complete attestation manifest
- `verify.py`: Standalone verification script
- Event streams and cryptographic roots

This demonstrates the same attestation infrastructure referenced in the paper, using synthetic data.

---

**Document Status**: Stub for arXiv path consistency
