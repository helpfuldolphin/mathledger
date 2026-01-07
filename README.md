# MathLedger

Governance substrate for verifiable learning: deterministic replay, dual attestation, fail-closed claims.

[![Live Demo](https://img.shields.io/badge/demo-mathledger.ai-blue)](https://mathledger.ai/demo)
[![arXiv](https://img.shields.io/badge/arXiv-2601.00816-b31b1b.svg)](https://arxiv.org/abs/2601.00816)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Pipeline

```
                         MathLedger Demo Pipeline
                         ========================

  ┌─────────┐     ┌─────────┐     ┌────────────┐     ┌───────────┐
  │  Input  │ ──▶ │   Run   │ ──▶ │  Artifacts │ ──▶ │ verify.py │
  │ (seed)  │     │ (demo)  │     │  (output/) │     │           │
  └─────────┘     └─────────┘     └────────────┘     └───────────┘
                                                            │
                                                            ▼
                                                     ┌────────────┐
                                                     │ PASS / FAIL│
                                                     └────────────┘

  Determinism guarantee: same seed → byte-identical artifacts → same hash
```

---

## Quick Start

```bash
# Prerequisites: Python 3.11+, uv package manager
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
cd demo_output && python verify.py
```

**Expected output:**
```
[PASS] Composite root verified: H_t == SHA256(R_t || U_t)
```

No database or external services required. The demo runs fully offline.

---

## Project Map

| You want to...                          | Start here                                                                 |
|-----------------------------------------|----------------------------------------------------------------------------|
| Run the demo (2 min)                    | `scripts/run_dropin_demo.py`                                               |
| Understand what you're looking at       | [`docs/HOW_THE_DEMO_EXPLAINS_ITSELF.md`](docs/HOW_THE_DEMO_EXPLAINS_ITSELF.md) |
| Audit artifacts independently           | [`docs/pilot/AUDIT_WALKTHROUGH.md`](docs/pilot/AUDIT_WALKTHROUGH.md)       |
| See the sealed FOL specification        | [`docs/FOL_FIN_EQ_PHASE3_CLOSURE.md`](docs/FOL_FIN_EQ_PHASE3_CLOSURE.md)   |
| Browse all documentation                | [`docs/README.md`](docs/README.md)                                         |
| Try the live hosted demo                | [mathledger.ai/demo](https://mathledger.ai/demo)                           |
| Read the paper                          | [`docs/PAPERS/mathledger_arxiv_preprint.pdf`](docs/PAPERS/mathledger_arxiv_preprint.pdf) |

---

## What Gets Generated

After running the demo, `demo_output/` contains:

```
demo_output/
├── verify.py              # Self-contained verifier (run this)
├── evidence_pack.json     # Complete audit artifact
├── manifest.json          # Execution metadata
├── u_t.txt                # UI Merkle root
├── r_t.txt                # Reasoning Merkle root
├── h_t.txt                # Composite root: SHA256(R_t || U_t)
└── replay_instructions.md # Third-party reproduction steps
```

The `verify.py` script recomputes all hashes locally with no external calls. If replay produces different hashes, the pack is invalid.

---

## What This Is

- **Governance substrate**: separation of exploration from authority at the data model level
- **Deterministic attestation**: same inputs → same hashes, every time
- **Fail-closed claims**: ABSTAINED is a first-class outcome, not a missing value
- **Auditable by design**: evidence packs enable independent replay verification

## What This Is NOT

- A capability demonstration (the demo does very little on purpose)
- A production system (governance substrate only)
- A safety claim (we claim legibility, not alignment)
- A convergence guarantee (learning dynamics are Phase II)

---

## Example Output

Running the boundary demo at [mathledger.ai/demo](https://mathledger.ai/demo) produces:

```
1. ADV (Advisory)    "2 + 2 = 4"  →  ABSTAINED   Excluded from authority stream
2. PA  (Attested)    "2 + 2 = 4"  →  ABSTAINED   Authority-bearing but no validator
3. MV  (Validated)   "2 + 2 = 4"  →  VERIFIED    Arithmetic validator confirmed
4. MV  (False)       "3 * 3 = 8"  →  REFUTED     Arithmetic validator disproved

Conclusion: Same claim text, different trust class → different outcome.
            Same trust class, different truth → VERIFIED vs REFUTED.
```

This demonstrates authority routing, not proof generation.

---

## FOL_FIN_EQ_v1 (First-Order Logic Verification)

Verify first-order logic formulas with equality over finite domains:

```bash
python -m scripts.run_fol_fin_eq_demo --domain z2 --output demo_z2
python demo_z2/verify.py
```

**Expected output:**
```
PASS: All certificates verified
```

**Golden manifest SHA256:** `096ee79e4e20c94fffbc2ec9964dde98f8058cba47a887031085e0800d6d2113`

See [`docs/FOL_FIN_EQ_PHASE3_CLOSURE.md`](docs/FOL_FIN_EQ_PHASE3_CLOSURE.md) for the sealed specification.

---

## Independent Audit (SHADOW-OBSERVE)

For external reviewers conducting artifact verification:

| Document | Purpose |
|----------|---------|
| [`docs/pilot/AUDIT_WALKTHROUGH.md`](docs/pilot/AUDIT_WALKTHROUGH.md) | Complete audit procedure |
| [`docs/pilot/PILOT_NON_CLAIMS.md`](docs/pilot/PILOT_NON_CLAIMS.md) | Binding non-claims |
| [`docs/pilot/PILOT_EVALUATION_CHECKLIST.md`](docs/pilot/PILOT_EVALUATION_CHECKLIST.md) | 22-item PASS/FAIL checklist |
| [`docs/FOR_AUDITORS.md`](docs/FOR_AUDITORS.md) | Cold-start auditor guide |

This audit verifies artifact integrity and determinism only; it makes no correctness, safety, or compliance claims.

---

## Paper / Citation

**MathLedger: A Verifiable Learning Substrate with Ledger-Attested Feedback**

```bibtex
@misc{mathledger2025,
  title={MathLedger: A Verifiable Learning Substrate with Ledger-Attested Feedback},
  author={MathLedger Contributors},
  year={2025},
  eprint={2601.00816},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2601.00816}
}
```

- [arXiv preprint](https://arxiv.org/abs/2601.00816)
- [DOI](https://doi.org/10.48550/arXiv.2601.00816)
- [PDF (local)](docs/PAPERS/mathledger_arxiv_preprint.pdf)

---

## Documentation Index

Full documentation index: [`docs/README.md`](docs/README.md)

| Category | Key Documents |
|----------|---------------|
| **Demo** | [HOW_THE_DEMO_EXPLAINS_ITSELF.md](docs/HOW_THE_DEMO_EXPLAINS_ITSELF.md), [V0_LOCK.md](docs/V0_LOCK.md) |
| **Audit** | [FOR_AUDITORS.md](docs/FOR_AUDITORS.md), [pilot/AUDIT_WALKTHROUGH.md](docs/pilot/AUDIT_WALKTHROUGH.md) |
| **Specs** | [FOL_FIN_EQ_PHASE3_CLOSURE.md](docs/FOL_FIN_EQ_PHASE3_CLOSURE.md), [invariants_status.md](docs/invariants_status.md) |
| **Ops** | [DEPLOY_BY_TAG_DOCTRINE.md](docs/DEPLOY_BY_TAG_DOCTRINE.md), [VERSION_NUMBER_DOCTRINE.md](docs/VERSION_NUMBER_DOCTRINE.md) |

---

## License

MIT
