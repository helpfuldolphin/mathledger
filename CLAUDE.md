# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**MathLedger** is a verifiable ledger of mathematical truths. It automates the generation, derivation, and verification of mathematical statements within bounded axiomatic frameworks (propositional logic, first-order logic with equality, equational theories, linear arithmetic). Every statement is derived from axioms, verified in Lean 4, and recorded as a block with cryptographic provenance.

The system consists of:
- **Backend**: Python-based proof factory with axiom engine, Lean verifier, and PostgreSQL/Redis persistence
- **Orchestrator**: FastAPI server exposing metrics, block, and statement endpoints
- **UI**: Svelte-based dashboard for browsing statements and proofs
- **Infrastructure**: Docker containers for PostgreSQL and Redis

## System Architecture

### Core Components

1. **Axiom Engine** (`backend/axiom_engine/`)
   - `derive.py`: Main derivation logic applying Modus Ponens and axiom instantiation
   - `rules.py`: Inference rules (MP, substitution)
   - `policy.py`: Derivation policies and slice management
   - `derive_cli.py`: CLI for running derivations
   - `derive_worker.py`: Background worker for async derivation

2. **Logic Module** (`backend/logic/`)
   - `canon.py`: Canonical formula normalization (right-assoc →, commutative ∧/∨)
   - `taut.py`: Truth table verification for propositional logic
   - `truthtab.py`: Truth table evaluation utilities

3. **Ledger** (`backend/ledger/`)
   - `blockchain.py`: Block construction and Merkle root computation
   - `blocking.py`: Block sealing and persistence

4. **Orchestrator** (`backend/orchestrator/`)
   - `app.py`: FastAPI server with metrics, blocks, statements endpoints
   - `parents_routes.py`: Parent/proof lineage endpoints
   - Authentication via `X-API-Key: devkey` header

5. **Worker** (`backend/worker.py`)
   - Redis queue consumer
   - Lean 4 verification orchestrator
   - Job file generation and cleanup (keeps last 500 files)

6. **Lean Project** (`backend/lean_proj/`)
   - Lean 4 project for proof verification
   - Generated job files in `ML/Jobs/job_*.lean`

### Database Schema

PostgreSQL tables:
- `theories`: Defines logical systems (PL, FOL=, Group, Ring)
- `statements`: `id, hash, text, system_id, depth, normalized_text, created_at`
- `proofs`: `id, statement_id, system_id, method, prover, status, duration_ms, created_at`
- `proof_parents`: DAG edges (proof_id → parent_statement_id)
- `runs`: Execution configs and summary stats
- `blocks`: Block headers with `run_id, system_id, root_hash, proof_count, created_at`

Migrations in `migrations/` are plain SQL files run manually via `run_migration.py`.

### Data Flow

```
Generator → Axiom Engine → Lean Verifier → Normalizer/Hasher → Ledger DB → Block Builder
     ↓           ↓              ↓                ↓                    ↓            ↓
  Formulas   Derivations    Verification      Canonical         Statements    Blocks
                                                 Hash
```

## Common Development Commands

### Environment Setup

Requires Python 3.11+, `uv` package manager, Docker Desktop (PostgreSQL + Redis).

```bash
# Install dependencies
uv sync

# Start infrastructure (PostgreSQL + Redis)
docker compose up -d postgres redis

# Run database migrations
python run_all_migrations.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test markers
pytest -m unit
pytest -m integration
pytest -m slow

# Run tests with coverage
coverage run -m pytest
coverage report
```

### Running Services

```bash
# Start FastAPI server (localhost:8000)
make api
# Or manually:
uv run uvicorn backend.orchestrator.app:app --host 0.0.0.0 --port 8000 --reload

# Start worker process
make worker
# Or manually:
python backend/worker.py

# Start UI dev server (localhost:5173)
cd ui && npm run dev
```

### Derivation

```bash
# Run derivation CLI with custom parameters
uv run python backend/axiom_engine/derive_cli.py --system-id 1 --steps 10 --max-breadth 200 --max-total 1000

# Run nightly derivation (PowerShell)
powershell -File .\scripts\run-nightly.ps1

# Run sanity check
powershell -File .\scripts\sanity.ps1
```

### Database Operations

```bash
# Show database statistics
make db-stats

# Backup database (PowerShell)
powershell -File .\scripts\backup-db.ps1

# Restore database (PowerShell)
powershell -File .\scripts\restore-db.ps1 -BackupPath backups/mathledger-20250101-020000.dump

# Run maintenance (VACUUM ANALYZE)
powershell -File .\scripts\db-maintenance.ps1
```

## Key Development Patterns

### Formula Normalization

All formulas must be canonicalized via `backend.logic.canon.normalize()` before hashing:
- Right-associative implication (→)
- Sorted commutative operands (∧, ∨)
- Alpha-renaming for quantifiers
- Hash computed via SHA-256 of normalized text

### Derivation Engine

The axiom engine (`backend.axiom_engine.derive.py`) operates in steps:
1. Load existing statements from DB
2. Apply axiom schemas via substitution (bounded depth)
3. Apply Modus Ponens: (p, p→q) ⊢ q
4. Enqueue new statements for verification
5. Insert verified proofs into DB
6. Seal block after each run

Derivation policies (`policy.py`) enforce:
- Breadth cap: max new statements per step
- Total cap: max new statements per run
- Depth limits: prevent infinite derivations
- Slice advancement: curriculum progression (atoms ≤4 depth ≤4 → depth ≤5 → atoms ≤5 depth ≤6)

### Lean Verification

Worker generates Lean files in `backend/lean_proj/ML/Jobs/job_<uuid>.lean`:
```lean
import ML.Taut
#eval do
  let result ← tautCheck "(p → p)"
  IO.println result
```

Runs `lake build ML.Jobs.job_<uuid>` and parses stdout for SUCCESS/FAILURE.
Fallback: truth table verification for propositional logic.

### Block Construction

After each derivation run:
1. Collect all successful proofs
2. Compute Merkle root: SHA-256 over sorted proof IDs
3. Insert block header into `blocks` table
4. Link proofs to block via `block_id`

### Schema Tolerance

Code is schema-tolerant to handle evolving database schema:
- Dynamic column detection via `information_schema.columns`
- Flexible success predicates: `success=TRUE`, `status='success'`, etc.
- Graceful fallbacks for missing columns

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://ml:mlpass@localhost:5432/mathledger

# Redis
REDIS_URL=redis://localhost:6379/0
QUEUE_KEY=ml:jobs

# Lean
LEAN_PROJECT_DIR=C:\dev\mathledger\backend\lean_proj

# API
LEDGER_API_KEY=devkey
CORS_ORIGINS=*

# Derivation (see config/nightly.env for full list)
DERIVE_STEPS=300
DERIVE_DEPTH_MAX=4
DERIVE_MAX_BREADTH=500
DERIVE_MAX_TOTAL=2000
```

## API Endpoints

All non-UI endpoints require `X-API-Key: devkey` header.

### Core Endpoints

- `GET /metrics` - System metrics (proof counts, block count, max depth, queue length)
- `GET /blocks/latest` - Latest block header
- `GET /statements?hash=<hex64>` - Statement detail with proofs and parents
- `GET /health` - Health check

### UI Endpoints (no auth required)

- `GET /ui` - Dashboard with metrics and recent statements
- `GET /ui/s/<hash>` - Statement detail page
- `GET /ui/parents` - Parent lineage explorer
- `GET /ui/proofs` - Proof explorer

### Swagger Docs

- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Testing Strategy

Tests are in `tests/` directory:
- `test_canon.py`: Formula canonicalization
- `test_taut.py`: Truth table verification
- `test_derive.py`: Derivation engine
- `test_mp.py`: Modus Ponens inference
- `test_subst.py`: Substitution logic
- `test_worker_fallback.py`: Worker fallback verification
- `integration/`: Integration tests requiring DB/Redis

Use `conftest.py` for shared fixtures (DB sessions, Redis clients).

## Windows-Specific Notes

This codebase is developed on Windows with PowerShell scripts:
- Use `powershell -File .\script.ps1` instead of `./script.sh`
- Paths use backslashes (`C:\dev\mathledger\backend`)
- Makefile uses PowerShell via `powershell -Command`
- Docker Desktop required for PostgreSQL/Redis containers

## Nightly Operations

Production automation via `scripts/run-nightly.ps1`:
1. Health check (services, DB, Redis)
2. Derivation (axiom engine with configured limits)
3. Snapshot export (JSONL exports to `exports/`)
4. Database maintenance (VACUUM ANALYZE, prune old proofs)
5. Progress update (append summary to `docs/progress.md`)

Schedule via Windows Task Scheduler (daily 2 AM recommended).

## Important Files

- `pyproject.toml`: Root Python dependencies (FastAPI, Redis, psycopg, SQLAlchemy)
- `backend/orchestrator/pyproject.toml`: Orchestrator-specific dependencies
- `pytest.ini`: Test configuration
- `docker-compose.yml`: Infrastructure services
- `Makefile`: Common build commands (Windows-friendly)
- `README_ops.md`: Detailed operations guide
- `docs/whitepaper.md`: System architecture and theory
- `docs/API_REFERENCE.md`: API documentation
- `config/nightly.env`: Nightly operation configuration

## Deployment Notes

The system runs as:
1. PostgreSQL + Redis containers (via `docker compose`)
2. FastAPI orchestrator (port 8000)
3. Worker process (background derivation)
4. Optional: Svelte UI (port 5173 dev, built to `ui/dist`)

For production:
- Use Task Scheduler for nightly derivations
- Configure alerting via webhook (Discord/Slack)
- Monitor metrics endpoint for success rates
- Keep backups (automated via `scripts/backup-db.ps1`)

## Git Workflow

Main branch: `integrate/ledger-v0.1`

Current branch: `qa/claudeB-2025-09-27`

Typical workflow:
1. Create feature branch from `integrate/ledger-v0.1`
2. Run tests (`pytest`)
3. Run sanity check (`powershell -File .\scripts\sanity.ps1`)
4. Commit with descriptive messages
5. Create PR to `integrate/ledger-v0.1`
