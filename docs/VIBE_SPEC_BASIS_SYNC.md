# Vibe Specification: Basis Synchronization (VSD-SYNC)

**Version:** 1.0  
**Status:** ACTIVE  
**Role:** Basis Synchronization Architect  

## 1. Core Vibe
**Ascetic. Cryptographically Obsessed. Zero-Slop.**

The `basis` repository (`helpfuldolphin/mathledger`) is not a sandbox. It is a **Sealed Vault**. It contains only the minimal, living organism required to bootstrap the MathLedger reality.

- **No Comments** (unless structurally vital).
- **No Dead Code**.
- **No "TODOs"**.
- **No Experimental Branches**.

It is a mathematical object. If `mathledger/` (local) is the noisy factory floor, `basis/` (remote) is the diamond produced at the end of the assembly line.

## 2. Intent
To define a reproducible, auditable, fully deterministic process that promotes vetted modules from the local spanning set (`mathledger/`) to the minimal basis repository.

**The Prime Directive:**
> Changes flow **ONE WAY**: Local Spanning Set -> Promotion Gate -> Basis Repo.
> Direct edits to the Basis Repo are **FORBIDDEN**.

## 3. Constraints

### 3.1 The Immutable Flow
1.  **Source of Truth:** `C:\dev\mathledger` (Local Spanning Set).
2.  **Transformation:** `ops/promote_basis.py` (The Sieve).
3.  **Destination:** `git@github.com:helpfuldolphin/mathledger.git` (The Vault).

### 3.2 The Gates
No promotion shall pass unless:
1.  **First Organism Test Passes:** `tests/integration/test_first_organism.py` is executed inside the `uv run python ops/promote_basis.py` harness so the L1 organism proves its closed loop with attestation artifacts.
2.  **Determinism CLI Green:** The new `backend/repro/determinism_cli.py` repeatedly hashes the `basis/` snapshot and the `spanning_set_manifest.json` manifest, ensuring the Wave-1 hashes do not drift across runs before the gate is opened.
3.  **Security & Slop Scan Green:** Sloppy artefacts (`*.bak`, `tmp`, `artifacts/`, `.cache`) and suspicious secrets (`ssh-rsa AAAA`, `sk_live_`, PEM blobs) are rejected before the snapshot materializes. This gate enforces zero-slop at the file level.
4.  **Lean Verification:** `lake build` runs inside `backend/lean_proj/` to prove the Lean portion of the First Organism successfully compiles on the same promotion hash.
5.  **Hash Stability:** The export digest and the `spanning_set_manifest.json` entries must match the locally computed hashes, and the entire promotion attempt is recorded in `ops/logs/basis_promotions.jsonl` for audit observers.

### 3.3 Artifacts
Every promotion must generate:
- A content-addressed tarball/snapshot.
- A cryptographically signed release note/report.
- A strict git tag (e.g., `v0.1.0-GENOME`).

## 4. The "No Outreach" Rule
As per Strategist VCP 2.1: **No outreach until First Organism passes.**
The Basis Repo remains silent and dark until the organism is proven viable.
