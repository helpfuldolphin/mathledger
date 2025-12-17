# MathLedger

**A cryptographically attested ledger of formally verified mathematical truths.**

MathLedger automates the generation, derivation, and verification of mathematical statements within bounded axiomatic frameworks. Every statement is derived from axioms, verified in Lean 4, and recorded with cryptographic provenance. The system includes built-in drift and stability monitoring for AI-generated reasoning.

---

## Core Capabilities

### 1. Proof Substrate (Verified Truths)

- **Axiom Engine**: Derives mathematical statements via Modus Ponens and K/S schema instantiation
- **Lean 4 Verification**: Machine-checked proofs with Mathlib integration
- **Statement Ledger**: PostgreSQL-backed storage with SHA-256 hashing and Merkle roots
- **Block Sealing**: Immutable proof records with cryptographic attestation

### 2. Dual-Root Attestation

- **Reasoning Root (R_t)**: Merkle root over proof artifacts and verification events
- **UI Root (U_t)**: Merkle root over user interaction events
- **Composite Root (H_t)**: SHA-256(R_t || U_t) binding machine reasoning to human oversight

### 3. Governance Infrastructure

- **USLA (Unified Stability and Learning Audit)**: Stochastic stability monitoring (H, rho, tau)
- **TDA (Topological Data Analysis)**: Pattern detection for reasoning drift and hallucination
- **Shadow Mode**: Observational governance without enforcement (calibration phase)
- **Evidence Packs**: Audit-grade bundles with schema validation and integrity proofs

### 4. Curriculum System

- **Slice Progression**: Bounded complexity advancement (atoms <= 4, depth <= 4, etc.)
- **Gate Enforcement**: Verified capability prerequisites before advancement
- **RFL (Reflexive Formal Learning)**: Policy updates based on verification outcomes

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (PostgreSQL + Redis)
- `uv` package manager
- [elan](https://github.com/leanprover/elan) (Lean version manager)

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/mathledger.git
cd mathledger
uv sync

# Start infrastructure
docker compose up -d postgres redis

# Run database migrations
python run_all_migrations.py
```

### Lean 4 Setup (Required for Verification)

MathLedger uses Lean 4 with Mathlib for proof verification. First-time setup downloads ~2GB of dependencies and takes 10-30 minutes depending on network speed.

```bash
# Install Lean toolchain and build Mathlib (first time: ~2GB download, 10-30 min)
make lean-setup

# Verify setup succeeded
make lean-check
```

**What `make lean-setup` does:**
1. Installs the pinned Lean version (`v4.23.0-rc2`) via elan
2. Downloads pre-built Mathlib cache (~1.5GB)
3. Builds the MathLedger Lean project

**Cache locations** (safe to delete for clean rebuild):
- `.lake/packages/` - Downloaded dependencies
- `backend/lean_proj/.lake/build/` - Build artifacts

### Verification Commands

After setup, two verification targets are available:

```bash
# 1. Mock determinism verification (no Lean required)
# Verifies: pipeline produces identical H_t across runs with same seed
# Does NOT verify: Lean type-checking of proofs
make verify-mock-determinism

# 2. Real Lean verification (requires lean-setup)
# Verifies: Lean 4 type-checks a specific proof file
make verify-lean-single PROOF=backend/lean_proj/ML/Jobs/job_test.lean
```

### What Mock Determinism Verifies

When you run `make verify-mock-determinism`:

1. **Attestation roots are computed**:
   - `R_t` (reasoning root): Merkle root over synthetic verification artifacts
   - `U_t` (UI root): Merkle root over interaction events
   - `H_t` (composite root): `SHA256(R_t || U_t)`

2. **Determinism is verified**: The same seed produces byte-identical `H_t` across runs. If `deterministic: true` appears in the output, both runs matched exactly.

**What mock mode does NOT prove**: Lean proof validity. Mock mode uses synthetic artifacts to test pipeline determinism without invoking Lean. For real Lean verification, use `make verify-lean-single`.

### What Real Lean Verification Does

When you run `make verify-lean-single PROOF=<path>`:

1. **Lean 4 executes**: The Lean proof assistant type-checks the specified proof file.

2. **Binary verdict**: Lean's kernel accepts or rejects proofs. There is no "soft pass"—either the proof type-checks (exit 0) or it fails (exit non-zero).

**Important**: `verify-lean-single` requires `make lean-setup` to be run first (~2GB download, 10-30 minutes).

### Running Services

```bash
# API Server (localhost:8000)
uv run uvicorn backend.orchestrator.app:app --reload

# Verification Worker
python backend/worker.py

# UI Dashboard (localhost:5173)
cd apps/ui && npm run dev
```

### Running Tests

```bash
# Full test suite
pytest

# Specific markers
pytest -m unit
pytest -m integration
pytest -m first_light

# Determinism verification (mock mode, no Lean required)
make verify-mock-determinism
```

---

## Architecture

```
mathledger/
├── attestation/          # Dual-root attestation (R_t, U_t, H_t)
├── backend/
│   ├── axiom_engine/     # Statement derivation and inference
│   ├── governance/       # Multi-signal governance fusion
│   ├── health/           # USLA/TDA health adapters (60+ modules)
│   ├── ledger/           # Block sealing and persistence
│   ├── lean_proj/        # Lean 4 verification project
│   ├── topology/         # First Light calibration system
│   └── worker.py         # Lean verification worker
├── basis/                # Canonical minimal spanning set
├── curriculum/           # Curriculum gates and enforcement
├── normalization/        # Formula canonicalization
├── rfl/                  # Reflexive Formal Learning engine
└── tests/                # Test suite (integration, health, first_light)
```

---

## Key Documents

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Development guidance and architecture overview |
| `VSD.md` | Vibe Specification Document (canonical structure) |
| `CRITICAL_FILES_MANIFEST.md` | Tier 1/2/3 critical file registry |
| `docs/whitepaper/` | Technical whitepaper (LaTeX source) |
| `docs/FieldManual/` | Operational field manual |

---

## Design Principles

1. **Proof-or-Abstain**: The system never upgrades unverified claims. Statements are either PROVED (verified) or ABSTAIN (not verified).

2. **Deterministic Attestation**: All outputs are reproducible. No timestamps in hashes, seeded PRNG, sorted keys.

3. **Shadow Mode Discipline**: Governance signals are observational-only during calibration. No enforcement until stability is proven.

4. **Cryptographic Provenance**: Every statement, proof, and block has a verifiable hash chain.

5. **Schema Tolerance**: Code adapts to database schema changes without hard-coded column names.

---

## Status

**Current Phase**: First Light Calibration (P3/P4/P5)

- Lean 4 verification: Operational
- Dual-root attestation: Implemented
- USLA/TDA monitoring: Implemented (Shadow Mode)
- Evidence pack generation: Implemented
- AI proof ingestion: Not yet implemented

---

## Contributing

See `CLAUDE.md` for development guidelines. All contributions must:
- Pass CI validation (critical files check, syntax validation)
- Maintain deterministic behavior
- Include appropriate test coverage
- Respect shadow mode discipline

---

## License

[Specify your license here]
