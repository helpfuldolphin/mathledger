# Toolchain Snapshot Specification v1.0

**Status**: SAVE TO REPO: YES
**Rationale**: Defines the ground truth for what constitutes a reproducible toolchain state. Required for audit and replay.

---

## 1. Overview

This specification defines what toolchain components are pinned, hashed, and recorded for CAL-EXP-1 and successor experiments. A run is reproducible if two clean checkouts with identical toolchain inputs produce identical artifact hashes (after stripping documented time-variant keys).

## 2. Toolchain Components

### 2.1 Python Toolchain

| Component | Source | Pinning Mechanism | Hash Target |
|-----------|--------|-------------------|-------------|
| Python version | Runtime | `pyproject.toml` requires-python | Version string |
| Package manager | uv | `--version` output | Version string |
| Dependencies | `uv.lock` | Lock file with SHA-256 per package | Full file hash |

**Current Versions (Baseline)**:
- Python: `>=3.11` (runtime: 3.11.9)
- uv: 0.8.16
- Lock file: `uv.lock` (SHA-256: `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc`)

### 2.2 Lean Toolchain

| Component | Source | Pinning Mechanism | Hash Target |
|-----------|--------|-------------------|-------------|
| Lean version | `lean-toolchain` | Version string file | File contents |
| Lake version | Bundled with Lean | Derived from Lean | N/A (implicit) |
| Mathlib + deps | `lake-manifest.json` | Git commit SHAs | Full file hash |
| Lakefile config | `lakefile.lean` | Checked into repo | Full file hash |

**Current Versions (Baseline)**:
- Lean: `leanprover/lean4:v4.23.0-rc2`
- Lake: 5.0.0-src+ad1a017 (bundled)
- Mathlib: commit `a3e910d1569d6b943debabe63afe6e3a3d4061ff`

### 2.3 Platform Fingerprint

| Component | Source | Recording Method |
|-----------|--------|------------------|
| OS | `platform.platform()` | String in manifest |
| Architecture | `platform.machine()` | String in manifest |
| Hostname | `socket.gethostname()` | String in manifest |

**Note**: Platform fingerprint is recorded for audit but NOT enforced for reproducibility. Cross-platform determinism is a non-goal.

## 3. Toolchain Fingerprint Computation

The **toolchain_fingerprint** is a single SHA-256 hash computed from the concatenation of:

```
toolchain_fingerprint = SHA256(
    SHA256(uv.lock) +
    SHA256(lean-toolchain) +
    SHA256(lake-manifest.json) +
    SHA256(lakefile.lean)
)
```

This fingerprint is recorded in every experiment manifest and must match between runs claiming reproducibility.

### 3.1 Hash Order

Hashes are concatenated in the following canonical order:
1. `uv.lock` (Python dependencies)
2. `lean-toolchain` (Lean version)
3. `lake-manifest.json` (Lean dependencies)
4. `lakefile.lean` (Lean build configuration)

### 3.2 File Paths (Relative to Repo Root)

```
uv.lock                             # Python dependency lock
backend/lean_proj/lean-toolchain    # Lean version pin
backend/lean_proj/lake-manifest.json # Lean dependency lock
backend/lean_proj/lakefile.lean     # Lean build config
```

## 4. Manifest Schema Extension

Every experiment manifest MUST include a `toolchain` block:

```json
{
  "toolchain": {
    "schema_version": "1.0",
    "fingerprint": "<sha256-hex-64-chars>",
    "python": {
      "version": "3.11.9",
      "uv_version": "0.8.16",
      "uv_lock_hash": "<sha256-hex-64-chars>"
    },
    "lean": {
      "version": "leanprover/lean4:v4.23.0-rc2",
      "toolchain_hash": "<sha256-hex-64-chars>",
      "lake_manifest_hash": "<sha256-hex-64-chars>",
      "lakefile_hash": "<sha256-hex-64-chars>"
    },
    "platform": {
      "os": "Windows-10-...",
      "arch": "AMD64",
      "hostname": "..."
    }
  }
}
```

## 5. Verification Protocol

### 5.1 Pre-Run Verification

Before an experiment executes, the harness MUST:

1. Compute `toolchain_fingerprint` from current files
2. Record fingerprint in manifest
3. If comparing to baseline: assert fingerprint matches

### 5.2 Post-Run Verification

A run is reproducible if:

1. `toolchain_fingerprint` matches between runs
2. `prng_seed` is identical
3. Git commit SHA is identical (or diff_sha256 matches for dirty trees)
4. `evidence_merkle_root_normalized` matches (after stripping time-variant keys)

### 5.3 Time-Variant Keys

The following keys are stripped before comparison:
- `timestamp`
- `created_at`
- `updated_at`
- `run_timestamp`

## 6. CI Parity Requirement

The following invariant MUST hold:

```
local_toolchain_fingerprint == ci_toolchain_fingerprint
```

This is enforced by:
1. Recording fingerprint in CI artifacts
2. Local `verify_toolchain_parity.py` script comparing against CI
3. CI workflow failing if fingerprint diverges from locked baseline

## 7. Non-Goals

This specification does NOT address:
- Cross-platform reproducibility (Windows vs Linux)
- Database state reproducibility (covered by fixture hashing)
- Network-dependent operations (experiments must be hermetic)
- Compiled artifact caching (`.lake/` directory is ephemeral)

## 8. Migration Path

### 8.1 Immediate (This PR)

1. Update CAL-EXP-1 harness to compute full `toolchain_fingerprint`
2. Add `toolchain` block to manifest schema
3. Add `substrate/repro/toolchain.py` module for fingerprint computation

### 8.2 Future (Out of Scope)

- Pin mathlib to tagged release instead of `master`
- Add `lake-packages.toml` for deterministic dependency resolution
- Cross-verifier toolchain alignment (Coq, Isabelle)

---

## Appendix A: Baseline Hashes (2025-12-13)

| File | SHA-256 |
|------|---------|
| `uv.lock` | `d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc` |
| `lean-toolchain` | `410d5c912b1a040c79883f5e0bb55e733888534e2006eefe186e631c24864546` |
| `lake-manifest.json` | `f13722c8f13f52ef06e5fc123ba449287887018f2b071ad4da2d8f580045dd3e` |
| `lakefile.lean` | `550736714d3a69ef99fca869f0f0b7e2e5fe81e0a51b621cb5e08baf37c82d30` |
| **Fingerprint** | `b828a2185e017e172db966d3158e8e2b91b00a37f0cd7de4c4f7cf707130a20a` |

## Appendix B: Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-13 | Initial specification |
