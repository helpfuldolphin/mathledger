# MathLedger Drop-In Governance Demo — Replay Instructions

**Classification**: External Due Diligence
**Audience**: Third-party engineers evaluating MathLedger as governance substrate
**Time to Complete**: < 10 minutes

---

## Overview

This document provides step-by-step instructions for running and verifying the MathLedger drop-in governance demo. The demo proves that MathLedger can serve as a **drop-in governance substrate** with:

1. **Deterministic execution** — Same seed produces identical outputs
2. **Dual attestation** — Reasoning (R_t) + UI (U_t) streams bound to composite (H_t)
3. **Governance verdicts** — F5.x predicates enforce fail-close behavior
4. **Replayability** — All inputs/outputs captured for independent audit

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.11+ | `python --version` |
| uv | 0.4+ | `uv --version` |
| Git | Any | `git --version` |

No database, Redis, or external services required. The demo operates entirely offline.

---

## Quick Start (< 2 minutes)

```bash
# 1. Clone or navigate to repo
cd /path/to/mathledger

# 2. Run demo with default seed
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/

# 3. Verify outputs
cd demo_output
python verify.py
```

Expected output:
```
[PASS] Composite root verified: H_t == SHA256(R_t || U_t)
[INFO] Seed: 42
[INFO] Claim level: L0
[INFO] F5 codes: ['F5.2', 'F5.3']
```

---

## Verification Steps

### Step 1: Verify Composite Root Integrity

The composite attestation root H_t must equal `SHA256(R_t || U_t)`.

```bash
# View the three roots
cat reasoning_root.txt   # R_t
cat ui_root.txt          # U_t
cat epoch_root.txt       # H_t

# Verify manually (bash)
R_T=$(cat reasoning_root.txt)
U_T=$(cat ui_root.txt)
H_T=$(cat epoch_root.txt)
COMPUTED=$(echo -n "${R_T}${U_T}" | sha256sum | cut -d' ' -f1)
[ "$COMPUTED" = "$H_T" ] && echo "PASS" || echo "FAIL"
```

Or use the included verifier:
```bash
python verify.py
```

### Step 2: Verify Determinism

Run the demo twice with the same seed and diff the outputs:

```bash
# First run
uv run python scripts/run_dropin_demo.py --seed 42 --output run1/

# Second run
uv run python scripts/run_dropin_demo.py --seed 42 --output run2/

# Compare roots (should be identical)
diff run1/reasoning_root.txt run2/reasoning_root.txt
diff run1/ui_root.txt run2/ui_root.txt
diff run1/epoch_root.txt run2/epoch_root.txt

# Compare manifests (excluding timestamps if present)
diff <(jq -S 'del(.generated_at)' run1/manifest.json) \
     <(jq -S 'del(.generated_at)' run2/manifest.json)
```

Expected: No differences.

### Step 3: Verify Governance Predicates

The manifest includes the governance verdict:

```bash
jq '.governance' manifest.json
```

Output:
```json
{
  "claim_level": "L0",
  "f5_codes": ["F5.2", "F5.3"],
  "passed": false,
  "rationale": "Fail-close triggered by: F5.2, F5.3"
}
```

**Interpreting the verdict:**
- `L0` claim level = fail-close triggered (governance working correctly)
- `F5.2` = Variance ratio exceeded threshold
- `F5.3` = Windowed drift exceeded threshold
- `passed: false` = This is **expected behavior** demonstrating fail-safe governance

### Step 4: Inspect Event Streams

The demo generates synthetic reasoning and UI events:

```bash
# View reasoning events
head -3 events/reasoning_events.jsonl | jq .

# View UI events
head -3 events/ui_events.jsonl | jq .
```

These events are deterministically generated from the seed and feed into the attestation construction.

---

## File Reference

| File | Description |
|------|-------------|
| `manifest.json` | Complete demo manifest with attestation, governance, and toolchain info |
| `reasoning_root.txt` | R_t: Merkle root over reasoning/proof events |
| `ui_root.txt` | U_t: Merkle root over UI/human interaction events |
| `epoch_root.txt` | H_t: Composite attestation root = SHA256(R_t \|\| U_t) |
| `events/reasoning_events.jsonl` | Raw reasoning events (JSONL format) |
| `events/ui_events.jsonl` | Raw UI events (JSONL format) |
| `verify.py` | Standalone verification script |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Demo completed successfully (governance pass OR fail-close) |
| 1 | Infrastructure/environment error |
| 2 | Import/dependency error |

**Note**: Exit code 0 indicates successful execution of the demo; governance may still fail-close (L0), which is an expected and correct outcome for this pilot. Governance fail-close is NOT an error.

---

## Command-Line Options

```
usage: run_dropin_demo.py [-h] [--seed SEED] [--output OUTPUT]
                          [--reasoning-events N] [--ui-events N]

Options:
  --seed SEED           Random seed for deterministic execution (default: 42)
  --output OUTPUT       Output directory (default: demo_output/)
  --reasoning-events N  Number of synthetic reasoning events (default: 10)
  --ui-events N         Number of synthetic UI events (default: 5)
```

---

## What This Demonstrates

### 1. Deterministic Execution
Every run with the same seed produces byte-identical outputs. This enables:
- Reproducible audits
- Third-party verification
- Regression testing

### 2. Dual Attestation
The system maintains two independent event streams:
- **Reasoning (R_t)**: Proof verifications, derivations, formal events
- **UI (U_t)**: Human interactions, views, flags, session events

These are bound together via `H_t = SHA256(R_t || U_t)`, creating an unforgeable link between machine reasoning and human interaction.

### 3. Governance Verdicts
The F5.x predicates implement fail-close governance:
- **F5.1**: Toolchain drift detection
- **F5.2**: Variance ratio bounds
- **F5.3**: Windowed drift detection
- **F5.4**: Event count sanity
- **F5.5**: UI stream presence

Any predicate failure triggers fail-close (L0 claim level).

### 4. Replayability
All inputs and outputs are captured in the manifest and event files, enabling:
- Full replay by external parties
- Audit trail for compliance
- Deterministic regression testing

---

## Troubleshooting

### "Missing MathLedger primitive" error
Ensure you're running from the repo root with uv:
```bash
cd /path/to/mathledger
uv run python scripts/run_dropin_demo.py
```

### Different outputs on re-run
Check that you're using the same seed:
```bash
uv run python scripts/run_dropin_demo.py --seed 42
```

### Governance shows "fail-close" or "L0"
This is expected behavior for seed 42. Governance triggering fail-close demonstrates that the fail-safe mechanisms work correctly. Exit code remains 0 because the demo executed successfully.

---

## Further Reading

| Document | Path |
|----------|------|
| Phase II Governance Stability Memo | `docs/system_law/calibration/PHASE_II_GOVERNANCE_STABILITY_MEMO.md` |
| Phase II Diligence One-Pager | `docs/system_law/calibration/PHASE_II_DILIGENCE_ONEPAGER.md` |
| Dual-Root Attestation Module | `attestation/dual_root.py` |
| Toolchain Fingerprinting | `substrate/repro/toolchain.py` |
| Determinism Helpers | `backend/repro/determinism.py` |

---

**Document Status**: Ready for external sharing

*Generated for acquisition due diligence.*
