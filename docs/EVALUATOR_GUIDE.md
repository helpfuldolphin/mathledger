# Core Loop Verification — Evaluator Guide

Two verification commands are available with distinct purposes.

## Prerequisites

| Requirement | Check Command |
|-------------|---------------|
| Python 3.11+ | `python --version` |
| uv (package manager) | `uv --version` |
| Lean 4 (via elan) | `elan --version` (only for real Lean verification) |

## Quick Start

### Option A: Mock Determinism (No Lean Required)

```bash
# 1. Install dependencies
uv sync

# 2. Run mock determinism verification
make verify-mock-determinism
```

This verifies pipeline determinism using **synthetic artifacts**. It does NOT invoke real Lean.

### Option B: Real Lean Verification (Requires lean-setup)

```bash
# 1. Install dependencies
uv sync

# 2. Build Lean toolchain (first run: ~10-30 min, cached after)
make lean-setup

# 3. Verify a specific proof with real Lean
make verify-lean-single PROOF=backend/lean_proj/ML/Jobs/job_test.lean
```

This invokes **real Lean type-checking** on the specified proof file.

## What It Does

Runs the **real** Lean-backed pipeline **twice** with identical inputs:

```
(statement) → (Lean verifies) → (proof artifacts) → (R_t, U_t) → H_t
```

Compares H_t from both runs. If identical: **deterministic**.

## JSON Output

```json
{
  "deterministic": true,
  "runs": 2,
  "H_t": "073faa69...",
  "mode": "standalone",
  "lean": "enabled"
}
```

| Field | Meaning |
|-------|---------|
| `deterministic` | `true` if all H_t values match |
| `runs` | Number of pipeline executions |
| `H_t` | Composite root: `SHA256(R_t \|\| U_t)` |
| `mode` | Always `standalone` (no DB/Redis/network) |
| `lean` | `enabled` (real Lean), `mock` (synthetic), or `unavailable` |

### Lean-Coupled Fields (only when `lean: "enabled"`)

```json
{
  "lean_verification": {
    "statements": ["p -> p", "p -> (q -> p)", "(p -> q) -> (p -> q)"],
    "all_verified": true,
    "R_t_derivation": {
      "method": "SHA256(concat(sorted(leaf_hashes)))",
      "leaf_formula": "SHA256(lean_source_hash:stdout_hash:stderr_hash:returncode)"
    }
  }
}
```

The presence of `lean_verification` with `stdout_hash`, `stderr_hash` proves real Lean ran.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Deterministic, Lean verified |
| 1 | Non-deterministic (H_t mismatch) |
| 2 | Error (Lean unavailable, import failure, etc.) |

## What This Does NOT Claim

- **Not a formal proof** of mathematical soundness
- **Not exhaustive** — tests 3 statements, 2 runs by default
- **Not a security audit** — determinism != correctness
- **No side effects** — standalone mode, nothing persisted

## Mock Mode

The `make verify-mock-determinism` target explicitly runs in mock mode:

```bash
make verify-mock-determinism
```

Output will show `"lean": "mock"` — synthetic artifacts, not real Lean verification.

**Important**: Mock mode tests pipeline determinism (identical seeds → identical H_t). It does NOT verify Lean proof validity.

## Verbose Output

```bash
uv run python scripts/verify_core_loop.py --runs 5 --verbose --pretty
```

Shows per-run details including R_t, U_t, and artifact hashes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lean Verification                        │
│   statement → job_<hash>.lean → lake build → stdout/stderr  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  R_t = SHA256(concat(sorted([                               │
│    SHA256(lean_source_hash:stdout_hash:stderr_hash:rc)      │
│  ])))                                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  U_t = SHA256(ui_event_payload)                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  H_t = SHA256(R_t || U_t)                                   │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `scripts/verify_core_loop.py` | Main verification script |
| `backend/lean_mode.py` | Lean mode management (mock/dry_run/full) |
| `Makefile` | `verify-mock-determinism` and `verify-lean-single` targets |
| `.github/workflows/core-loop-verification.yml` | CI workflow (runs real Lean) |
