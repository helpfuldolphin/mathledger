# Basis Promotion Report (P1 Sprint)

Author: Cursor P (Canonical Repository Curator)  
Scope: Deep audit of critical subsystems for basis promotion readiness.

All verdicts reference the MathLedger whitepaper as the normative specification. Any subsystem marked “modify” requires targeted remediation before entering the canonical basis repository.

---

## backend/crypto/
- **Verdict**: keep (tighten interface and key management)
- **Dependency Graph**:
  - `core.py` → `backend.logic.canon`, `hashlib`, `cryptography.hazmat`
  - `auth.py` → `core.py`, `hashing.py`, FastAPI security helpers
  - `dual_root.py` → `core.py`, `backend.ledger.blockchain`, `backend.crypto.hashing`
  - `handshake.py` → `core.py`, `auth.py`
- **Determinism Verdict**: deterministic for hashing/normalization; key generation requires deterministic seed injection for dual attestation.
- **Whitepaper Alignment**: high; matches cryptographic primitives, but needs explicit dual-root attestation proofs.
- **Required Refactors**:
  - Inject deterministic key derivation path with documented entropy sources.
  - Isolate RFC 8785 canonicalization into shared substrate module.
  - Formalize domain separation constants and publish in `ATTESTATION_SPEC.md`.
- **Notes**: Hashing and Merkle logic already enforce domain separation; verify Ed25519 signing path respects hardware-backed keys and dual attestation logging.

---

## backend/ledger/
- **Verdict**: modify (consolidate versions, remove legacy V4 hierarchy)
- **Dependency Graph**:
  - `blockchain.py` ↔ `backend.crypto.core`, `backend.logic.canon`, `backend.rfl.coverage`
  - `blocking.py` → `backend.orchestrator.app`, `backend.crypto.hashing`
  - `ingest.py` → `backend.generator.propgen`, `backend.models.proof_metadata`
  - `consensus/` → `backend.consensus.harmony`, `backend.ledger.v4`
- **Determinism Verdict**: deterministic ingestion/ordering once inputs are normalized; needs removal of legacy clock-based checkpoints and worker callbacks.
- **Whitepaper Alignment**: partial; mix of V4-era constructs and new harmony consensus; must converge on whitepaper dual-root ledger.
- **Required Refactors**:
  - Collapse `v4/` into canonical `ledger/` namespace or archive it.
  - Replace ad-hoc UI event emission with deterministic state snapshots.
  - Rebuild block sealing pipeline to consume deterministic Merkle witnesses from `backend.crypto`.
- **Notes**: Attestation correctness presently distributed; move block hash attestations into canonical ledger module and ensure Merkle proofs documented and tested.

---

## backend/orchestrator/
- **Verdict**: modify (retain core, strip legacy routes)
- **Dependency Graph**:
  - `app.py` → FastAPI, `backend.orchestrator.parents_routes`, `backend.worker`
  - `parents_routes.py` → `backend.ledger.blockchain`, `backend.rfl.runner`
  - `proof_middleware.py` → `backend.crypto.auth`, `backend.logic.canon`
- **Determinism Verdict**: deterministic request handling when backed by hermetic DB; remove `.bak` copies and clock-based retry jitter.
- **Whitepaper Alignment**: medium; route structure mirrors orchestrator spec but lacks dual attestation handshake enforcement.
- **Required Refactors**:
  - Delete `.bak` files and unify configuration via canonical `config/`.
  - Introduce deterministic queue depth control aligned with curriculum pacing.
  - Add attestation middleware ensuring every request logs dual-root commitments.
- **Notes**: Normalize FastAPI dependency injection to avoid hidden global state.

---

## backend/axiom_engine/
- **Verdict**: modify (retain mathematical kernels, drop exploratory pipelines)
- **Dependency Graph**:
  - `derive_core.py` → `axioms.py`, `structure.py`, `backend.logic.taut`
  - `pipeline.py` → `derive_core.py`, `derive_worker.py`, `backend.frontier.curriculum`
  - CLI modules interact with `scripts/` and `backend.generator`
- **Determinism Verdict**: mixed; base derivation functions pure, but pipeline CLI uses randomness and filesystem caches.
- **Whitepaper Alignment**: partial; theoretical mapping exists but code diverges via experimental heuristics.
- **Required Refactors**:
  - Remove stochastic heuristics or replace with deterministic selection derived from curriculum ladders.
  - Extract canonical inference rules into `substrate/axioms`.
  - Fold worker orchestration into deterministic job queue backed by orchestrator.
- **Notes**: Formalize proof obligations and ensure exported derivations carry cryptographic attestations.

---

## backend/rfl/
- **Verdict**: modify (core logic sound; metrics/visualizer experimental)
- **Dependency Graph**:
  - `runner.py` → `config/rfl`, `backend.frontier.curriculum`, `backend.metrics_reporter`
  - `coverage.py` → `backend.ledger.blockchain`, `backend.crypto.core`
  - `experiment.py` → `backend.generator.propgen`, `backend.rfl.bootstrap_stats`
- **Determinism Verdict**: scheduling deterministic when curriculum inputs fixed; experiments module introduces timeline variance.
- **Whitepaper Alignment**: strong; implements Reflexive Formal Learning loop but lacks dual attestation outputs.
- **Required Refactors**:
  - Remove `experiment.py` exploratory knobs; encode policies in canonical config.
  - Move visualization into docs or tooling outside runtime path.
  - Emit attestation events for coverage transitions.
- **Notes**: Verify normalization correctness by calling `backend.logic.canon` for every curriculum state transition.

---

## backend/logic/
- **Verdict**: keep (normalize interface, delete `.bak`)
- **Dependency Graph**:
  - `canon.py` → `taut.py`, `backend.logic.truthtab`
  - `taut.py` → Python stdlib, provides propositional checks
  - `truthtab.py` → `itertools`, used across ledger/repro
- **Determinism Verdict**: deterministic; pure functions with canonical ordering.
- **Whitepaper Alignment**: high; provides normalization pipeline referenced by ledger and crypto.
- **Required Refactors**:
  - Remove `canon.py.bak`.
  - Publish normalization spec excerpt inside `docs/NORMALIZATION.md`.
  - Add property-based determinism tests into `tests/determinism/`.
- **Notes**: Ensure exported API is limited to canonical normalization primitives and truth-table evaluators.

---

## backend/lean_proj/
- **Verdict**: modify (retain minimal Lean project, drop job dumps)
- **Dependency Graph**:
  - `Main.lean` → `ML/Jobs/*.lean`, `lakefile.lean`, `lean-toolchain`
  - Python entry (`backend/lean_interface.py`) loads compiled artifacts
- **Determinism Verdict**: Lean builds deterministic given pinned toolchain; job output files are non-canonical.
- **Whitepaper Alignment**: high for `Main.lean` and toolchain; job artifacts represent historical span not basis.
- **Required Refactors**:
  - Remove `ML/Jobs` generated proofs; regenerate deterministically on-demand.
  - Lock toolchain version in canonical manifest and create reproducible build script.
  - Integrate Lean runner outputs into attestation log.
- **Notes**: Ensure Lean and Python agree on normalization/curriculum semantics; produce checksums for compiled objects.

---

## backend/repro/
- **Verdict**: keep (promote to `tests/determinism/` helper)
- **Dependency Graph**:
  - `determinism.py` → `backend.logic.canon`, `backend.crypto.core`, environment probes
  - `__init__.py` minimal
- **Determinism Verdict**: deterministic harness verifying reproducibility; collects system info.
- **Whitepaper Alignment**: high; enforces determinism guardrails.
- **Required Refactors**:
  - Refactor script into reusable library with CLI wrapper.
  - Ensure outputs are pure JSON summaries suitable for attestation storage.
  - Integrate with CI gating in `ops/ci/`.
- **Notes**: Extend to cover dual attestation checks (e.g., verifying two independent runs match).

---

## config/
- **Verdict**: modify (keep canonical configs, drop redundant copies)
- **Dependency Graph**:
  - `curriculum.yaml` → `backend.rfl.runner`, `backend.frontier.curriculum`
  - `rfl/*.json` → `backend.rfl`, telemetry
  - `nightly.env` → `run-nightly.ps1`, `ops` scripts
- **Determinism Verdict**: deterministic when pinned; need to eliminate `.bak` or alternative YAML variants.
- **Whitepaper Alignment**: strong but cluttered by backups and historical lanes.
- **Required Refactors**:
  - Delete `curriculum.yaml.bak` and redundant lane files once archived.
  - Document every config key in `docs/RFL_SPEC.md`.
  - Add schema validation under `tests/curriculum/`.
- **Notes**: Ensure configs provide dual attestation toggles and canonical seeds.

---

## scripts/
- **Verdict**: modify (curate deterministic entry points)
- **Dependency Graph**:
  - Mix of migration, patching, telemetry scripts; heavy reliance on `backend` modules.
  - Some PowerShell wrappers call external services.
- **Determinism Verdict**: mixed; many scripts assume mutable environments or network.
- **Whitepaper Alignment**: partial; only migration/test scripts align clearly.
- **Required Refactors**:
  - Keep `run-migrations.py`, canonical nightly scripts; deprecate ad-hoc patchers.
  - Standardize script interface (argparse, deterministic output).
  - Move non-deterministic helpers to archive.
- **Notes**: Provide hashed dependency manifest for each retained script and ensure cryptographic logging.

---

## tests/
- **Verdict**: modify (isolate deterministic suites, archive legacy)
- **Dependency Graph**:
  - `tests/integration` → `backend/orchestrator`, DB fixtures
  - `tests/unit` → `backend.logic`, `backend.crypto`
  - Legacy files `test_v05_*` depend on outdated ledger versions
- **Determinism Verdict**: unit tests deterministic; integration tests rely on external DB/timeouts.
- **Whitepaper Alignment**: needs re-segmentation to match curriculum/attestation focus.
- **Required Refactors**:
  - Create `tests/determinism/` harness using `backend.repro`.
  - Remove `.bak` and disabled tests from canonical set.
- **Notes**: Ensure every promoted test logs attestation vectors and normalization properties.

---

## docs/
- **Verdict**: modify (distill to canonical spec set)
- **Dependency Graph**:
  - Extensive Markdown referencing multiple phases; minimal code dependencies.
- **Determinism Verdict**: not applicable; curation required to avoid conflicting specs.
- **Whitepaper Alignment**: inconsistent; need single source-of-truth docs.
- **Required Refactors**:
  - Extract `ARCHITECTURE.md`, `PROTOCOL.md`, `API.md`, `RFL_SPEC.md`, `ATTESTATION_SPEC.md`.
  - Archive historical phase reports outside basis repo.
  - Cross-link docs with canonical modules and manifest.
- **Notes**: Embed cryptographic invariants and normalization proofs into spec docs.

---

## ops/
- **Verdict**: modify (expand into structured ops toolchain)
- **Dependency Graph**:
  - Currently limited to Markdown; operational scripts live elsewhere (`scripts/`, `sanity.ps1`).
- **Determinism Verdict**: pending; need to formalize CI/infra definitions.
- **Whitepaper Alignment**: minimal; must codify dual-attestation ops flows.
- **Required Refactors**:
  - Create `ops/ci/`, `ops/scripts/`, `ops/infra/` with deterministic pipelines.
  - Document reproducible deployment runbooks.
- **Notes**: Ensure ops flows produce attestable artifacts and respect curriculum pacing.

---

## interface/
- **Verdict**: modify (retain API schemas, drop experimental UI)
- **Dependency Graph**:
  - `interface/api` → `backend.api.schemas`, FastAPI
  - `interface/cli.py` → `backend.orchestrator`, `backend.ledger`
  - UI artifacts depend on Svelte/React experiments (apps/ui, ui/)
- **Determinism Verdict**: API/CLI deterministic; UI experiments non-deterministic due to build pipelines.
- **Whitepaper Alignment**: API aligns; UI layers largely exploratory.
- **Required Refactors**:
  - Promote only schema-defining modules and CLI; create `interface/ui/` only after deterministic build path defined.
  - Ensure API docs sync with `docs/API.md`.
- **Notes**: Integrate interface layer with dual-attestation handshake (client ↔ ledger root) and enforce canonical serialization.

---

## Summary
- Subsystems marked **keep** are ready for direct promotion once minor cleanups land (`backend/logic`, `backend/repro`).
- Subsystems marked **modify** require specific, enumerated refactors (most of the tree).
- No subsystem in this audit is marked immediate **drop**; experimental content inside each will be archived during promotion.

Next actions: implement refactors, validate determinism, and cross-reference with promotion shortlist before initializing the canonical basis repository.

