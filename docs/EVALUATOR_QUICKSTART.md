# Evaluator Quickstart

Verify CAL-EXP-3 / First-Light evidence pack integrity in under 5 minutes.

---

## What This Verifies

1. **Artifact completeness** — P3 (synthetic stability) and P4 (shadow twin) outputs exist
2. **File integrity** — SHA-256 hashes of all artifacts match the manifest
3. **Pack structure** — Required directories and files are present

---

## What This Does NOT Verify

- **Lean proof correctness** — Proofs were checked when P3/P4 harnesses ran, not during this command
- **Scientific claims** — This is infrastructure audit, not claim validation
- **Determinism** — Use `make verify-mock-determinism` separately for determinism checks (uses mock, not real Lean)
- **AI capability** — No benchmark or capability claims are made

All artifacts are marked `mode: SHADOW` (calibration phase, observation-only).

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python --version` |
| uv | any | `uv --version` |

No Docker, database, Redis, or network access required.

---

## One Command

```bash
make evidence-pack
```

---

## Expected Output (Success)

```
============================================================
FINAL VERDICT
============================================================

  VERDICT: PASS
  Evidence pack verified: 17/17 files OK

  The evidence pack is ready for external audit.
  Location: results/first_light/evidence_pack_first_light
  Manifest: results/first_light/evidence_pack_first_light/manifest.json

SHADOW MODE: All artifacts are observation-only.
============================================================
```

Key indicators:
- `VERDICT: PASS` — All files exist and hashes match
- `17/17 files OK` — No missing or corrupted artifacts
- Exit code `0`

---

## Expected Output (Failure)

```
============================================================
FINAL VERDICT
============================================================

  VERDICT: FAIL
  Integrity check failed: 2 missing, 1 mismatched

  Missing files:
    - p3_synthetic/stability_report.json
    - p4_shadow/p4_summary.json
```

Exit code `1` indicates verification failure.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Evidence pack verified (all hashes match) |
| 1 | Verification failed (missing or mismatched files) |
| 2 | Generation failed (P3/P4 source artifacts not found) |

---

## If Verification Passes

No further action required. The evidence pack at `results/first_light/evidence_pack_first_light/` is ready for inspection.

Optional next steps:
- Inspect `manifest.json` for artifact inventory with SHA-256 hashes
- Review `compliance_report.json` for machine-readable verdict
- Run `make verify-mock-determinism` to verify pipeline determinism (mock mode, no Lean)
- Run `make verify-lean-single PROOF=<path>` to verify a specific proof with real Lean

---

## If Verification Fails

1. Check the listed missing/mismatched files
2. If P3/P4 artifacts are missing, run the harnesses:
   ```bash
   uv run python scripts/usla_first_light_harness.py --cycles 1000 --seed 42
   uv run python scripts/usla_first_light_p4_harness.py --cycles 1000 --seed 42
   ```
3. Re-run `make evidence-pack`

For deeper audit, see `docs/system_law/First_Light_External_Verification.md`.
