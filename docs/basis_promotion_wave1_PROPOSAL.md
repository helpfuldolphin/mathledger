# Wave 1 Promotion Proposal — First Organism

**Author:** Cursor P (Basis Repo Curator)  
**Status:** PENDING — awaiting First Organism integration test pass  
**Target Tag:** `v0.2.0-FIRST-ORGANISM`

---

## Overview

Wave 0 (`v0.1.0-GENOME`) established the immutable genomic primitives in the basis repo:

- Core types (`Block`, `DualAttestation`, `CurriculumTier`, etc.)
- Canonical normalization (`basis.logic.normalizer`)
- Domain-separated hashing and Merkle toolkit (`basis.crypto.hash`)
- Deterministic block sealing (`basis.ledger.block`)
- Dual attestation primitives (`basis.attestation.dual`)
- Curriculum ladder serialization (`basis.curriculum.ladder`)

Wave 1 promotes the **First Organism**—the minimal living loop that proves the MathLedger protocol can:

1. Receive a UI event
2. Pass through a curriculum gate
3. Execute a derivation step
4. Attempt Lean verification (abstain/fail path acceptable for first organism)
5. Ingest into the ledger with dual-root attestation (R_t, U_t, H_t)
6. Metabolize the sealed attestation via the RFL runner

---

## Gating Condition

Wave 1 promotion is **blocked** until the following integration test passes in the local spanning-set repo:

```
tests/integration/test_first_organism.py::test_first_organism_closed_loop
```

The test must emit:

```
[PASS] First Organism: UI→Curriculum→Derivation→LeanAbstain→LedgerIngestor→DualAttest(H_t)→RFLRunner — organism alive.
```

Additionally:

- **Determinism gate:** Re-running the test produces identical H_t.
- **Security gate:** `runtime_env` enforces required environment variables; no secrets leak.

## VCP-2.2 Hermetic Gate Update

STRATCOM sanctified a new SPARK definition for Wave 1: only the hermetic First Organism closed loop is required. The canonical command is:

```
uv run pytest tests/integration/test_first_organism.py -m first_organism -k "closed_loop_standalone or determinism" -v -s
```

The run must report:

1. `test_first_organism_closed_loop_standalone` = PASS  
2. `test_first_organism_determinism` = PASS  
3. `[PASS] First Organism: ... H_t=...` log line  
4. `artifacts/first_organism/attestation.json` written with matching `H_t`, `R_t`, `U_t` fields  

DB-backed tests (`full_integration`, `chain_integrity`, etc.) are permitted to SKIP with `[SKIP] Database unavailable` messages and no longer gate the promotion. Once the hermetic gate passes, and the canonical precheck report records `"status": "pass"`, STRATCOM authorizes moving to Wave 1 promotion. The existing `ops/basis_promotion_precheck.py` script now implements this hermetic gate.

## VCP-2.2 Promotion Gate: First Organism

The canonical pre-gate is implemented via `ops/basis_promotion_precheck.py` in the spanning-set repo. Execute:

```
uv run python ops/basis_promotion_precheck.py
```

The harness runs `tests/integration/test_first_organism.py::test_first_organism_closed_loop` twice, scans stdout for `[PASS] FIRST ORGANISM` plus the `H_t` log, and blocks if any `[ABSTAIN]` security skips are emitted. After the runs it loads `artifacts/first_organism/attestation.json`, recomputes `H_t = SHA256(R_t || U_t)`, and requires the artifact to be byte-for-byte identical in both executions.

On success the script writes `artifacts/first_organism/basis_precheck_report.json`, a canonical RFC 8785 JSON document with these keys:

```
{
  "status": "pass",
  "reason": "...",
  "ht": "<composite root hex>",
  "rt": "<reasoning root hex>",
  "ut": "<ui root hex>",
  "timestamp_iso": "<UTC>",
  "test_case": "tests/integration/test_first_organism.py::test_first_organism_closed_loop",
  "security_harness_passed": true,
  "determinism_verified": true
}
```

The `ht`/`rt`/`ut` values and SHA-256 digest of this report feed the future Wave 1 `basis_promotion_history.jsonl` entry. The gate forbids Wave 1 promotion until this report exists with `"status": "pass"` and STRATCOM explicitly authorizes the release.

---

## Post-SPARK Invariants (Wave-1 Promotion Requirements)

**Status:** Observer mode — awaiting SPARK + Wide Slice completion

After SPARK (First Organism hermetic test) and Wide Slice (Dyno Chart generation) succeed, the following invariants must hold for Wave-1 promotion:

### 1. FO Hermetic Tests + RFL Determinism

**Requirement:** The First Organism closed-loop test must pass deterministically with RFL metabolism enabled.

**Validation:**
- `test_first_organism_closed_loop_standalone` = PASS (hermetic, no DB required)
- `test_first_organism_determinism` = PASS (identical H_t across runs)
- RFL runner executes deterministically when processing attestation context
- All randomness sourced from deterministic seeds (MDAP_EPOCH_SEED + cycle_index)

**Canonical Command:**
```bash
uv run pytest tests/integration/test_first_organism.py -m first_organism -k "closed_loop_standalone or determinism" -v -s
```

**Expected Output:**
- `[PASS] FIRST ORGANISM ALIVE H_t=<hex>` log line
- No `[ABSTAIN]` security skips
- Identical `H_t` values across multiple runs

### 2. Attestation Artifact Present & Recomputable

**Requirement:** The attestation artifact must exist and be byte-for-byte recomputable.

**Validation:**
- `artifacts/first_organism/attestation.json` exists after SPARK run
- Contains `compositeAttestationRoot` (H_t), `reasoningMerkleRoot` (R_t), `uiMerkleRoot` (U_t)
- `H_t = SHA256(R_t || U_t)` recomputes correctly
- Artifact is RFC 8785 canonical JSON (stable key ordering, no trailing commas)
- `artifacts/first_organism/basis_precheck_report.json` exists with `"status": "pass"`

**Recomputation Check:**
```python
# From ops/basis_promotion_precheck.py
rt, ut, ht, raw = load_attestation()
candidate_ht = hashlib.sha256((rt + ut).encode("ascii")).hexdigest()
assert candidate_ht == ht  # Must match
```

### 3. Dyno Chart Produced for at Least One Wide Slice Run

**Requirement:** At least one Wide Slice experiment must complete and produce valid JSONL logs suitable for Dyno Chart visualization.

**Validation:**
- Wide Slice run executed with `slice_medium` curriculum slice
- Output files exist:
  - `results/fo_baseline_wide.jsonl` (baseline mode, RFL OFF)
  - `results/fo_rfl_wide.jsonl` (RFL mode, RFL ON) — at least one required
- JSONL logs pass validation tests:
  - `pytest -m wide_slice tests/integration/test_wide_slice_logs.py -v`
- Each JSONL line contains required fields for Dyno Chart:
  - `cycle` (int): Cycle number
  - `slice_name` (str): Curriculum slice name (e.g., "slice_medium")
  - `status` (str): Cycle status ("abstain", "verified", "error")
  - `method` or `verification_method` (str): Verification method used
  - `abstention` (bool): Whether cycle abstained
  - `roots.h_t`, `roots.r_t`, `roots.u_t`: Dual-root attestation values
  - `rfl.executed` (bool): Whether RFL metabolism ran
  - `mode` (str): "baseline" or "rfl"

**Canonical Command:**
```bash
# Run Wide Slice experiment
uv run python experiments/run_fo_cycles.py \
  --mode=rfl \
  --cycles=1000 \
  --slice-name=slice_medium \
  --system=pl \
  --out=results/fo_rfl_wide.jsonl

# Validate logs
pytest -m wide_slice tests/integration/test_wide_slice_logs.py -v
```

**Expected Output:**
- JSONL file with ≥100 cycles (recommended: 1000 for statistical stability)
- All validation tests pass
- Logs contain deterministic roots (same inputs → same outputs)

**Dyno Chart Visualization:**
The JSONL logs are consumed by visualization tools (e.g., `experiments/plotting.py`) to produce Dyno Charts showing:
- Cycle-by-cycle status trends (abstention rate, verification success)
- RFL uplift comparison (baseline vs RFL mode)
- Attestation root progression (H_t, R_t, U_t over cycles)
- Policy update effects (when RFL metabolism triggers)

**Note:** The Dyno Chart itself (PNG/PDF visualization) is not required for promotion, but the underlying JSONL data must be valid and complete.

---

## Module Inventory (Post-SPARK Refresh)

**Last Updated:** Post-SPARK observer audit (Cursor O)

### LAW (Ledger / Attestation / Blocking)

| Source Path (spanning set)       | Target Path (basis repo)            | Whitepaper Ref | Backend.* Imports | Status |
|----------------------------------|-------------------------------------|----------------|-------------------|--------|
| `ledger/ingest.py`               | `ledger/ingest.py`                  | §4.2, §5.1     | ✅ None           | ✅ Ready |
| `ledger/blocking.py`             | `ledger/blocking.py`                | §4.1           | ✅ None           | ✅ Ready |
| `ledger/ui_events.py`            | `ledger/ui_events.py`               | §4.3           | ✅ None           | ✅ Ready |
| `attestation/dual_root.py`       | `attestation/dual_root.py`          | §4.2, §4.3     | ✅ None           | ✅ Ready |
| `attestation/__init__.py`        | `attestation/__init__.py`           | §4.2           | ✅ None           | ✅ Ready |

**Key Exports:**
- `LedgerIngestor`, `IngestOutcome`, `StatementRecord`, `ProofRecord`, `BlockRecord`
- `compute_reasoning_root`, `compute_ui_root`, `compute_composite_root`
- `build_reasoning_attestation`, `build_ui_attestation`, `verify_composite_integrity`
- `generate_attestation_metadata`

**Dependency Check:** All LAW modules use canonical imports (`substrate.*`, `normalization.*`, `attestation.*`) with no `backend.*` dependencies. ✅

### ECONOMY (Curriculum / Derivation)

| Source Path (spanning set)       | Target Path (basis repo)            | Whitepaper Ref | Backend.* Imports | Status |
|----------------------------------|-------------------------------------|----------------|-------------------|--------|
| `curriculum/gates.py`            | `curriculum/gates.py`               | §5.2           | ✅ None           | ✅ Ready |
| `curriculum/config.py`           | `curriculum/config.py`              | §5.2           | ✅ None           | ✅ Ready |
| `curriculum/__init__.py`         | `curriculum/__init__.py`            | §5.2           | ✅ None           | ✅ Ready |
| `derivation/pipeline.py`         | `derivation/pipeline.py`            | §3.1           | ✅ None           | ✅ Ready |
| `derivation/axioms.py`           | `derivation/axioms.py`              | §3.1           | ✅ None           | ✅ Ready |
| `derivation/bounds.py`            | `derivation/bounds.py`              | §3.1           | ✅ None           | ✅ Ready |
| `derivation/derive_rules.py`     | `derivation/derive_rules.py`       | §3.1           | ✅ None           | ✅ Ready |
| `derivation/derive_utils.py`    | `derivation/derive_utils.py`       | §3.1           | ⚠️ Check          | ⚠️ Review |
| `derivation/structure.py`        | `derivation/structure.py`           | §3.1           | ✅ None*          | ✅ Ready |
| `derivation/verification.py`    | `derivation/verification.py`       | §3.3           | ✅ None           | ✅ Ready |

**Note:** `derivation/structure.py` contains a comment reference to `backend.logic.canon.normalize` but no actual import (uses `normalization.canon` instead). ✅

**Key Exports:**
- `CurriculumSlice`, `CoverageGateSpec`, `AbstentionGateSpec`, `SliceConfig`
- `check_advancement_gate`, `load_curriculum_config`
- `DerivationPipeline`, `SliceBounds`, `Axiom`, `InferenceRule`
- `make_first_organism_derivation_slice`, `make_first_organism_seed_statements`

**Dependency Check:** All ECONOMY modules use canonical imports. `derivation/derive_utils.py` may need review if it contains DB-specific utilities (e.g., `get_or_create_system_id`). ⚠️

### METABOLISM (RFL Runner)

| Source Path (spanning set)       | Target Path (basis repo)            | Whitepaper Ref | Backend.* Imports | Status |
|----------------------------------|-------------------------------------|----------------|-------------------|--------|
| `rfl/runner.py`                  | `rfl/runner.py`                     | §6.1           | ✅ None           | ✅ Ready |
| `rfl/config.py`                  | `rfl/config.py`                     | §6.1           | ✅ None           | ✅ Ready |
| `rfl/bootstrap_stats.py`         | `rfl/bootstrap_stats.py`            | §6.2           | ✅ None           | ✅ Ready |
| `rfl/coverage.py`                | `rfl/coverage.py`                   | §6.2           | ⚠️ 1 import       | ⚠️ Needs shim |
| `rfl/experiment.py`               | `rfl/experiment.py`                 | §6.2           | ⚠️ 1 import       | ⚠️ Needs shim |
| `rfl/__init__.py`                | `rfl/__init__.py`                   | §6.1           | ✅ None           | ✅ Ready |

**Backend.* Import Issues:**
- `rfl/coverage.py`: Imports `backend.axiom_engine.derive_utils.get_or_create_system_id` (line 18)
- `rfl/experiment.py`: Imports `backend.axiom_engine.derive_utils.get_or_create_system_id` (line 23)

**Resolution Required:**
- Extract `get_or_create_system_id` to canonical namespace (e.g., `derivation/derive_utils.py` or `ledger/utils.py`)
- Update `rfl/coverage.py` and `rfl/experiment.py` to use canonical import
- Ensure function signature and behavior match SPARK/Wide Slice expectations

**Key Exports:**
- `RFLRunner`, `RflResult`, `RunLedgerEntry`
- `RFLConfig`, `CurriculumSlice`
- `CoverageTracker`, `CoverageMetrics`
- `ExperimentResult`
- `compute_coverage_ci`, `compute_uplift_ci`, `verify_metabolism`

**Dependency Check:** Core RFL modules (`runner.py`, `config.py`, `bootstrap_stats.py`) are clean. Coverage and experiment modules need shim extraction. ⚠️

---

## Required Tests

| Test File                                      | Purpose                                      |
|------------------------------------------------|----------------------------------------------|
| `tests/integration/test_first_organism.py`     | End-to-end organism loop                     |
| `tests/attestation/test_dual_root.py`          | Dual-root computation and verification       |
| `tests/curriculum/test_gates.py`               | Curriculum gate determinism                  |
| `tests/rfl/test_runner.py`                     | RFL metabolism assertions                    |

All tests must pass with deterministic outputs (no wall-clock, no randomness, no network unless hermetic).

---

## Required Documentation Updates

| Document                     | Action                                                      |
|------------------------------|-------------------------------------------------------------|
| `BASIS_SPEC.md`              | Add Wave 1 organism description and chain diagram           |
| `docs/ATTESTATION_SPEC.md`   | Expand with dual-root walkthrough referencing `attestation/`|
| `docs/RFL_SPEC.md`           | Document RFL runner contract and metabolism invariants      |
| `docs/PROTOCOL.md`           | Add block lifecycle and ingestion flow                      |

---

## Basis Promotion Script Outline

**Purpose:** Automated script to copy LAW/Economy/Metabolism modules into `basis/` repo after SPARK + Wide Slice succeed.

**Prerequisites:**
- SPARK hermetic tests pass (`ops/basis_promotion_precheck.py` reports `"status": "pass"`)
- Wide Slice Dyno Chart logs validated (`pytest -m wide_slice` passes)
- Attestation artifact present and recomputable
- All `backend.*` imports resolved (shims extracted where needed)

### Step 1: Pre-Promotion Validation

```bash
# Verify SPARK gate
uv run python ops/basis_promotion_precheck.py
# Expected: "status": "pass" in artifacts/first_organism/basis_precheck_report.json

# Verify Wide Slice logs exist and are valid
pytest -m wide_slice tests/integration/test_wide_slice_logs.py -v
# Expected: All tests pass

# Verify no backend.* imports in target modules
grep -r "from backend\." ledger/ attestation/ curriculum/ derivation/ rfl/ || echo "✅ No backend.* imports found"
```

### Step 2: Extract Shim Dependencies

**If needed:** Extract `get_or_create_system_id` from `backend.axiom_engine.derive_utils` to canonical namespace:

```python
# Create or update derivation/derive_utils.py or ledger/utils.py
# Move get_or_create_system_id function
# Update rfl/coverage.py and rfl/experiment.py imports
```

### Step 3: Copy Modules to Basis Repo

**Script Outline:**
```bash
#!/bin/bash
# scripts/promote_wave1_to_basis.sh

BASIS_REPO_DIR="../mathledger_basis_repo"  # Adjust path as needed
SPANNING_SET_ROOT="$(pwd)"

# LAW modules
cp -r ledger/ "$BASIS_REPO_DIR/basis/"
cp -r attestation/ "$BASIS_REPO_DIR/basis/"

# ECONOMY modules
cp -r curriculum/ "$BASIS_REPO_DIR/basis/"
cp -r derivation/ "$BASIS_REPO_DIR/basis/"

# METABOLISM modules
cp -r rfl/ "$BASIS_REPO_DIR/basis/"

# Clean up any experimental residue
find "$BASIS_REPO_DIR/basis/" -name "*.bak" -delete
find "$BASIS_REPO_DIR/basis/" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$BASIS_REPO_DIR/basis/" -name "*.pyc" -delete
```

**Manual Steps (if script not available):**
1. Checkout basis repo: `git clone <basis-repo-url> mathledger_basis_repo`
2. Copy modules preserving directory structure
3. Remove any `.bak`, `__pycache__`, or experimental files
4. Verify imports use canonical paths (no `backend.*`)

### Step 4: Run Tests in Basis Repo

**Test Suite:**
```bash
cd mathledger_basis_repo

# Core basis tests
uv run pytest tests/test_basis_core.py -v

# First Organism reproduction test
# Note: This test should use the attestation fixture from SPARK run
uv run pytest tests/integration/test_first_organism_reproduction.py -v

# FO hermetic test (should match SPARK results)
uv run pytest tests/integration/test_first_organism.py \
  -m first_organism \
  -k "closed_loop_standalone or determinism" \
  -v -s

# Expected: Identical H_t to SPARK run
```

**Test Requirements:**
- `test_basis_core.py`: Validates basis primitives (Wave 0)
- `test_first_organism_reproduction.py`: Reproduces SPARK attestation artifact
  - Loads `artifacts/first_organism/attestation.json` from SPARK run
  - Recomputes H_t, R_t, U_t
  - Verifies byte-for-byte match
- `test_first_organism_closed_loop_standalone`: Hermetic FO test
  - Must produce identical H_t to SPARK run
  - No DB/Redis required

### Step 5: Record Promotion History

**Append to `basis_promotion_history.jsonl`:**
```json
{
  "wave": 1,
  "tag": "v0.2.0-FIRST-ORGANISM",
  "source_commit": "<spanning-set commit SHA>",
  "ht": "<composite root from SPARK>",
  "rt": "<reasoning root from SPARK>",
  "ut": "<ui root from SPARK>",
  "test_suite": "test_first_organism_closed_loop_standalone",
  "determinism_verified": true,
  "security_verified": true,
  "spark_report_sha256": "<SHA256 of basis_precheck_report.json>",
  "wide_slice_logs_validated": true,
  "dyno_chart_produced": true,
  "promoted_at_utc": "<ISO timestamp>",
  "promoted_by": "Cursor P"
}
```

### Step 6: Commit and Tag

```bash
cd mathledger_basis_repo

git add basis/ledger/ basis/attestation/ basis/curriculum/ basis/derivation/ basis/rfl/
git add basis_promotion_history.jsonl

git commit -m "feat: promote Wave 1 (First Organism)

Includes:
- LAW: LedgerIngestor, dual_root, blocking, ui_events
- ECONOMY: curriculum gates, derivation pipeline, verification
- METABOLISM: RFL runner, bootstrap stats, coverage tracker

First Organism H_t: <ht from SPARK>
Source commit: <spanning-set commit>
SPARK report: <spark_report_sha256>
Determinism: verified (identical H_t across runs)
Security: verified (no [ABSTAIN] skips)
Wide Slice: validated (Dyno Chart logs present)"

git tag v0.2.0-FIRST-ORGANISM
git push origin main --tags
```

### Step 7: Update MANIFEST.json

**Regenerate manifest:**
```bash
# Run manifest generator (if available)
python scripts/generate_basis_manifest.py

# Or manually update MANIFEST.json with:
# - New file paths and SHA-256 digests
# - Import graph (no backend.* dependencies)
# - Whitepaper references (§4.2, §5.1, §6.1, etc.)
```

### Step 8: Post-Promotion Verification

```bash
# Verify tag exists
git tag -l v0.2.0-FIRST-ORGANISM

# Verify all modules present
ls -R basis/ledger/ basis/attestation/ basis/curriculum/ basis/derivation/ basis/rfl/

# Verify no backend.* imports
grep -r "from backend\." basis/ || echo "✅ Clean"

# Verify tests pass
uv run pytest tests/test_basis_core.py tests/integration/test_first_organism_reproduction.py -v
```

---

## Promotion Procedure (Manual Checklist)

Once the gating condition is satisfied:

1. **Fetch attestation artifact**
   ```
   artifacts/first_organism/attestation.json
   artifacts/first_organism/basis_precheck_report.json
   ```
   Extract `H_t`, `R_t`, `U_t`, test log excerpts.

2. **Verify Wide Slice logs**
   ```
   results/fo_rfl_wide.jsonl (or fo_baseline_wide.jsonl)
   ```
   Ensure at least one Wide Slice run completed and logs are valid.

3. **Resolve backend.* imports**
   - Extract `get_or_create_system_id` shim if needed
   - Update `rfl/coverage.py` and `rfl/experiment.py` imports

4. **Copy modules into basis repo**
   - Checkout `helpfuldolphin/mathledger` locally (or use existing clone)
   - Copy only the modules listed above into their target paths
   - Ensure no experimental residue, no `.bak`, no spanning-set artifacts

5. **Run validation suite**
   ```bash
   uv run pytest tests/test_basis_core.py tests/integration/test_first_organism_reproduction.py -v
   ```

6. **Record in history**
   Append to `basis_promotion_history.jsonl` (see Step 5 above)

7. **Commit and tag**
   ```bash
   git add .
   git commit -m "feat: promote Wave 1 (First Organism)

   Includes:
   - LAW: LedgerIngestor, dual_root, blocking, ui_events
   - ECONOMY: curriculum gates, derivation core, verification
   - METABOLISM: RFL runner, bootstrap stats

   First Organism H_t: <hash>
   Source commit: <local commit>
   Determinism: verified
   Security: verified
   Wide Slice: validated"

   git tag v0.2.0-FIRST-ORGANISM
   git push origin main --tags
   ```

8. **Update MANIFEST.json**
   Regenerate with new files, SHA-256 digests, import graph, whitepaper refs.

---

## Attestation Implications

- Every promoted module must participate in deterministic attestation.
- `LedgerIngestor` seals blocks with `H_t = SHA256(R_t || U_t)`.
- `RFLRunner` consumes `H_t` and emits `policy_update_applied`, `abstention_mass_delta`.
- All hashes use domain separation per `ATTESTATION_SPEC.md`.

---

## Current Status (Observer Mode)

**Observer:** Cursor O (Basis Wave-1 Observer)  
**Mode:** Observer — awaiting SPARK + Wide Slice completion  
**Last Updated:** Post-SPARK plan synchronization

| Checkpoint                        | Status      | Notes |
|-----------------------------------|-------------|-------|
| Wave 0 (Genome) promoted          | ✅ DONE     | v0.1.0-GENOME in basis repo |
| SPARK hermetic test implemented   | ✅ DONE     | `test_first_organism_closed_loop_standalone` |
| SPARK hermetic test passing       | ⏳ PENDING  | Awaiting `ops/basis_promotion_precheck.py` pass |
| Wide Slice Dyno Chart produced    | ⏳ PENDING  | Awaiting `run_fo_cycles.py --slice-name=slice_medium` |
| Attestation artifact recomputable | ⏳ PENDING  | Awaiting SPARK completion |
| Module inventory refreshed        | ✅ DONE     | Backend.* imports identified (2 shims needed) |
| Backend.* imports resolved        | ⚠️ PARTIAL  | `rfl/coverage.py`, `rfl/experiment.py` need shim extraction |
| Wave 1 modules staged             | ⏳ PENDING  | Blocked until SPARK + Wide Slice succeed |
| Wave 1 promoted to basis repo     | ⏳ PENDING  | Blocked until all gates pass |

**Next Steps:**
1. Execute SPARK hermetic test: `uv run python ops/basis_promotion_precheck.py`
2. Run Wide Slice experiment: `uv run python experiments/run_fo_cycles.py --mode=rfl --slice-name=slice_medium --cycles=1000`
3. Extract `get_or_create_system_id` shim to canonical namespace
4. Update `rfl/coverage.py` and `rfl/experiment.py` imports
5. Proceed with promotion script once all gates pass

---

## Appendix: Module Dependency Graph (Wave 1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FIRST ORGANISM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  UI Event    │───▶│  Curriculum  │───▶│  Derivation  │              │
│  │  (U_t leaf)  │    │    Gate      │    │   Pipeline   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                                       │                       │
│         │                                       ▼                       │
│         │                              ┌──────────────┐                 │
│         │                              │ Lean Verify  │                 │
│         │                              │  (abstain)   │                 │
│         │                              └──────────────┘                 │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                      ┌──────────────┐                 │
│  │  ui_events   │                      │   Ingestor   │                 │
│  │   (U_t)      │─────────────────────▶│  (R_t, U_t)  │                 │
│  └──────────────┘                      └──────────────┘                 │
│                                                │                        │
│                                                ▼                        │
│                                        ┌──────────────┐                 │
│                                        │  dual_root   │                 │
│                                        │ H_t=SHA(R||U)│                 │
│                                        └──────────────┘                 │
│                                                │                        │
│                                                ▼                        │
│                                        ┌──────────────┐                 │
│                                        │  RFL Runner  │                 │
│                                        │ (metabolize) │                 │
│                                        └──────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*This proposal will be executed by Cursor P once the First Organism integration test passes and STRATCOM confirms.*

