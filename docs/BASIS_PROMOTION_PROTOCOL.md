# Basis Promotion Protocol

**Role:** Global Reviewer / Basis Architect  
**Context:** Moving code from Spanning Set (Local) to Basis (Remote/Vault).

## 1. Multi-Agent Decomposition

### Phase 0: Snapshot & Planning
**Agent:** `ops/promote_basis.py` (The Archiver)
- **Action:** Bundles `basis/` directory + `pyproject.toml` + `README` + `docs/` specs and prints the deterministic plan describing the wipe/commit path.
- **Output:** `exports/basis_<hash>.tar.gz` plus a console plan identifying the whitelisted paths.
- **Validation:** Computes SHA-256 of the tarball and fingerprints the manifest before moving to validation.

### Phase 1: Validation (The Gates)
**Agent:** `ops/promote_basis.py` (The Sentinel)
- **Action:**
    1.  Runs `tests/integration/test_first_organism.py` to certify the dual-root chain.
    2.  Executes `backend/repro/determinism_cli.py` to rehash `basis/` and `spanning_set_manifest.json` multiple times for drift resistance.
    3.  Performs security/slop scanning for `.bak`, `tmp`, `artifacts/`, PEM blobs (`sk_live_*`, `ssh-rsa AAAA`, etc.).
    4.  Runs `lake build` under `backend/lean_proj/` to prove Lean verification.
- **Abort Condition:** ANY failure triggers immediate abort and is logged in `ops/logs/basis_promotions.jsonl`. The plan halts so no snapshot is committed.

### Phase 2: Materialization
**Agent:** `ops/promote_basis.py` (The Builder)
- **Action:**
    1.  Clones/Updates target repo (`--target-dir`) and replays the promotion plan (whitelist, snapshot hash).
    2.  **Wipes** non-whitelisted files in the target workspace with plan visibility.
    3.  Unpacks Snapshot into target.
    4.  Verifies integrity (File Count, Hash Match) and leaves the target ready for deterministic reporting.

### Phase 3: Reporting
**Agent:** `ops/promote_basis.py` (The Scribe)
- **Action:** Generates `basis_promotion_release_notes.md`.
- **Content:**
    - Source Commit Hash.
    - Snapshot Hash.
    - Gate Results (First Organism, Determinism CLI, Security Scan, Lean Build) with concise details.
    - Manifest fingerprint and promotion plan notes to keep audits tight.

### Phase 4: Tagging & Push
**Agent:** Human Operator (The Keyholder)
- **Action:**
    - Reviews the release note and `ops/logs/basis_promotions.jsonl` entry for the successful gate vector.
    - Commits any drift-free changes in Target.
    - Tags (e.g., `v0.2.0`) with the recorded hash.
    - Pushes to GitHub ensuring the commit referenced in the plan matches what the Vault will receive.

## 2. Global Reviewer Hooks
Before the "Keyholder" pushes, they must verify:

1.  **Minimality Check:** Does the target contain *only* the basis? (No `tmp/`, no `artifacts/`).
2.  **Hash Identity:** Does `basis_<hash>.tar.gz` match the files currently in the target working tree?
3.  **Green Lights:** Does the log at `ops/logs/basis_promotions.jsonl` show `PASS` for this snapshot hash?
4.  **Gate Traceability:** Does `basis_promotion_release_notes.md` summarize the gate statuses (First Organism, Determinism CLI, Security Scan, Lean Build) and reference the manifest hash?

## 3. Reversibility
If a promotion is found to be defective *after* push:
1.  **Do NOT edit remote.**
2.  Revert local change.
3.  Run Promotion Protocol again to generate a reverting commit.
4.  Push.

## 4. Directory Structure (Target)
The Basis Repo should look like this:
```text
/
├── basis/              # The Python Package
├── docs/               # Vibe & Specs
├── tests/              # Minimal test suite to prove life
├── pyproject.toml      # Minimal deps
├── README.md           # The Manifesto
└── LICENSE
```
Anything else is sludge. Remove it.
