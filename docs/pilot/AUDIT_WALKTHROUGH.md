# Audit Walkthrough: SHADOW-OBSERVE Pilot Demo

**Document Type:** Audit Procedure
**Scope:** Artifact integrity, determinism, and replayability verification
**Mode:** SHADOW-OBSERVE (observational, non-blocking, non-enforcement)
**Version:** 1.0
**Date:** 2025-12-21

---

## 1. Scope Statement

### 1.1 What Is Being Audited

This walkthrough enables independent verification of:

| Item | Description |
|------|-------------|
| Artifact existence | Demo produces expected files |
| Hash integrity | Composite root equation `H_t = SHA256(R_t || U_t)` verifies |
| Schema compliance | Manifest conforms to declared structure |
| Determinism | Same seed produces identical outputs across runs |
| Replayability | All inputs/outputs captured for external replay |

### 1.2 What Is NOT Being Audited

| Non-Claim | Rationale |
|-----------|-----------|
| Correctness | Demo uses synthetic events; no real proofs evaluated |
| Model safety | No AI model is invoked or assessed |
| Learning behavior | No learning or adaptation occurs |
| Alignment | No alignment claims are made or tested |
| Production readiness | SHADOW mode prohibits deployment posture |
| Legal compliance | This document does not constitute legal advice |
| Performance guarantees | No benchmarks or baselines established |

For authoritative non-claims policy, see: [`PILOT_NON_CLAIMS.md`](./PILOT_NON_CLAIMS.md)

---

## 2. Prerequisites

### 2.1 Environment

| Requirement | Notes |
|-------------|-------|
| Python | 3.11+ required |
| `uv` | Recommended. Fallback: `pip install -e .` from repo root |
| OS | Windows, macOS, Linux (WSL supported) |
| Network | Not required |
| Lean toolchain | Not required (skipped if absent) |

### 2.2 Verification

```bash
python --version   # Expect: Python 3.11+
uv --version       # Expect: uv 0.x.x (if using uv)
```

---

## 3. One-Command Execution

### 3.1 Command

From repository root:

```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
```

**Fallback (without uv):**

```bash
pip install -e .
python scripts/run_dropin_demo.py --seed 42 --output demo_output/
```

### 3.2 Expected Behavior

| Condition | Expected |
|-----------|----------|
| Exit code | `0` (success) |
| Governance verdict | May be PASS or FAIL (fail-close) |
| Fail-close trigger | Expected for seed 42; demonstrates fail-safe behavior |

**Exit code 0 indicates successful execution of the demo; governance may still fail-close (L0), which is an expected and correct outcome for this pilot.** Governance triggering fail-close does NOT indicate an audit failure.

### 3.3 Expected Console Output (Excerpt)

```
============================================================
MathLedger Drop-In Governance Demo
============================================================
Seed: 42
Output: demo_output
Mode: SHADOW (observational only)

[1/5] Capturing toolchain snapshot...
      Skipped (Lean toolchain not present - not required for demo)
[2/5] Generating synthetic events...
      Reasoning events: 10
      UI events: 5
[3/5] Building dual-root attestation...
      R_t: 32f131f7aa68ba2c...
      U_t: 8a93fb4ca505054f...
      H_t: cc9c9d1bd237c2e3...
      Integrity: VALID
[4/5] Evaluating governance predicates...
      Claim level: L0
      F5 codes: ['F5.2', 'F5.3']
      Passed: False
[5/5] Writing outputs...
...
[RESULT] Governance triggered fail-close (claim level: L0)
         This is EXPECTED BEHAVIOR demonstrating fail-safe governance
```

---

## 4. Output Artifacts

After execution, `demo_output/` contains:

| File/Directory | Purpose |
|----------------|---------|
| `manifest.json` | Primary verification artifact; contains all hashes, governance verdict, reproducibility info |
| `reasoning_root.txt` | R_t: Merkle root of reasoning event stream |
| `ui_root.txt` | U_t: Merkle root of UI event stream |
| `epoch_root.txt` | H_t: Composite attestation root |
| `events/reasoning_events.jsonl` | Raw reasoning events (deterministic, replayable) |
| `events/ui_events.jsonl` | Raw UI events (deterministic, replayable) |
| `verify.py` | Standalone verification script (no external dependencies) |

---

## 5. Independent Verification

### 5.1 Composite Root Verification

From inside the output directory:

```bash
cd demo_output
python verify.py
```

**Expected Output:**

```
[PASS] Composite root verified: H_t == SHA256(R_t || U_t)
[INFO] Seed: 42
[INFO] Claim level: L0
[INFO] F5 codes: ['F5.2', 'F5.3']
[INFO] To verify reproducibility, run: uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output_verify/
```

### 5.2 What verify.py Checks

| Check | Description |
|-------|-------------|
| Composite root integrity | Recomputes `SHA256(R_t || U_t)` and compares to stored `H_t` |
| File presence | Reads `reasoning_root.txt`, `ui_root.txt`, `epoch_root.txt` |
| Manifest parsing | Loads and displays manifest metadata |

**PASS criteria:** Computed composite root matches stored `H_t`.

---

## 6. Determinism Verification

### 6.1 Procedure

Run the demo twice with identical seeds to separate output directories:

```bash
uv run python scripts/run_dropin_demo.py --seed 42 --output run1/
uv run python scripts/run_dropin_demo.py --seed 42 --output run2/
```

### 6.2 SHA-256 Hash Verification

Compute SHA-256 hashes of key artifacts:

**Unix/Linux/macOS/WSL:**

```bash
sha256sum run1/manifest.json run2/manifest.json
sha256sum run1/epoch_root.txt run2/epoch_root.txt
```

**Windows (PowerShell):**

```powershell
Get-FileHash run1\manifest.json, run2\manifest.json -Algorithm SHA256
Get-FileHash run1\epoch_root.txt, run2\epoch_root.txt -Algorithm SHA256
```

### 6.3 Expected Hashes (seed=42)

For seed 42, the following SHA-256 hashes are expected. Both runs must produce byte-for-byte identical files.

| Artifact | SHA-256 Hash |
|----------|--------------|
| `manifest.json` | `f5f7d95fd12fa3fb1c2f16400920b4f5f137864d372bc4e496e63b2264ad6312` |
| `epoch_root.txt` | `34a941e522c0521f29f5071ddb051616ff1c6e9960c86abbf260ae9f40faaa7f` |
| `reasoning_root.txt` | `314646adf474bd92a56862cc1fcbe9f13407aa1c037d3fe1eb10edbf135b3918` |
| `ui_root.txt` | `250d98261d1160e39d9380dca3d6f0015ba0e3eda8719f7922617429b89e1f68` |

### 6.4 Diff Comparison

**Unix/Linux/macOS/WSL:**

```bash
diff run1/manifest.json run2/manifest.json && echo "PASS: Identical"
```

**Windows (PowerShell):**

```powershell
fc run1\manifest.json run2\manifest.json
```

**Python (cross-platform):**

```python
import json
m1 = json.load(open("run1/manifest.json"))
m2 = json.load(open("run2/manifest.json"))
assert m1 == m2, "Manifests differ"
print("PASS: Manifests identical")
```

### 6.5 Allowed Differences

**Zero differences are permitted.** All output files must be byte-for-byte identical across runs with the same seed.

**PASS criteria:** SHA-256 hashes match between run1 and run2 for all artifacts.

---

## 7. Failure Modes

### 7.1 Execution Failures (Audit-Relevant)

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `ModuleNotFoundError` | Missing dependencies | Run `uv sync` or `pip install -e .` |
| Exit code 2 | Import failure | Verify running from repo root |
| `uv: command not found` | uv not installed | Install uv or use pip fallback |
| Permission denied | Insufficient write permissions | Check output directory permissions |

### 7.2 Governance Outcomes (Not Audit Failures)

| Outcome | Meaning | Audit Status |
|---------|---------|--------------|
| Claim level L0 | Fail-close triggered | NOT a failure; expected for seed 42 |
| F5.x codes present | Governance predicates fired | Expected behavior |
| `Passed: False` | Governance verdict negative | Demonstrates fail-safe; audit passes |

**Critical distinction:**

- **Execution failure:** Demo script could not run (exit code != 0)
- **Governance fail-close:** Demo ran successfully; governance predicates triggered fail-safe

Only execution failures constitute audit failures.

---

## 8. Audit Conclusion Template

Auditors may use this template to document findings:

```
============================================================
AUDIT NOTE: MathLedger SHADOW-OBSERVE Demo Verification
============================================================

Date:           ____________________
Auditor:        ____________________
Environment:    ____________________
Python version: ____________________
uv version:     ____________________

EXECUTION
---------
Command:        uv run python scripts/run_dropin_demo.py --seed 42 --output demo_output/
Exit code:      ____________________
Governance verdict: ____________________

VERIFICATION CHECKS
-------------------
[ ] verify.py executed successfully
[ ] Composite root verified: H_t == SHA256(R_t || U_t)
[ ] Manifest schema valid (JSON parses, required fields present)
[ ] Determinism verified (two runs produce identical outputs)

HASHES / IDENTIFIERS
--------------------
R_t (reasoning root): ____________________
U_t (UI root):        ____________________
H_t (composite root): ____________________
Seed:                 ____________________

DEVIATIONS
----------
[ ] None observed
[ ] Deviations noted: ____________________

CONCLUSION
----------
[ ] PASS - All artifact integrity and determinism checks passed
[ ] FAIL - Execution or verification failure (describe below)

Notes:
______________________________________________________________________
______________________________________________________________________

Signature: ____________________
```

---

## 9. Reference Documents

| Document | Location |
|----------|----------|
| Pilot Non-Claims | `docs/pilot/PILOT_NON_CLAIMS.md` |
| Pilot Evaluation Checklist | `docs/pilot/PILOT_EVALUATION_CHECKLIST.md` |
| Demo Script | `scripts/run_dropin_demo.py` |
| Dual Attestation Primitives | `attestation/dual_root.py` |

---

## 10. Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2025-12-21 | Initial release |
| 1.1 | 2025-12-21 | Hardened determinism proof with SHA-256 hashes; updated references |

---

*This document describes artifact verification procedures only. It does not constitute validation of correctness, safety, alignment, or legal compliance.*
