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
