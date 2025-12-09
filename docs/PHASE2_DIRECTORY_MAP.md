# Phase II Directory Map

> **Agent**: `doc-ops-3` — Directory Cartographer  
> **Status**: PHASE II — STRUCTURE DOCUMENTATION  
> **Last Updated**: 2025-12-06

---

## Overview

This document provides a complete cartography of the MathLedger project directory structure, designating each folder and key files as Phase I (foundation) or Phase II (uplift experiments/research).

### Phase Designation Key

| Symbol | Phase | Description |
|--------|-------|-------------|
| **Ⅰ** | Phase I | Core infrastructure, foundation, negative controls |
| **Ⅱ** | Phase II | Uplift experiments, U2 runtime, research deliverables |
| **Ⅰ/Ⅱ** | Mixed | Contains both Phase I and Phase II components |

---

## Root Directory Structure

```
mathledger/
├── analysis/                  [Ⅱ]  Phase II dynamics analysis
├── artifacts/                 [Ⅰ/Ⅱ] Experiment outputs, figures, attestations
├── attestation/               [Ⅰ]  Dual-root attestation infrastructure
├── backend/                   [Ⅰ/Ⅱ] Core services with Phase II extensions
├── config/                    [Ⅰ/Ⅱ] Configuration files for both phases
├── curriculum/                [Ⅰ]  Curriculum gate definitions
├── derivation/                [Ⅰ]  First Organism derivation specs
├── docs/                      [Ⅰ/Ⅱ] Documentation (non-code only)
├── experiments/               [Ⅰ/Ⅱ] Experiment runners and analysis
├── infra/                     [Ⅰ]  Docker/deployment configs
├── migrations/                [Ⅰ]  Database migrations
├── ops/                       [Ⅰ/Ⅱ] Operational tooling and runbooks
├── results/                   [Ⅰ/Ⅱ] Raw experiment output files
├── rfl/                       [Ⅰ]  RFL core library
├── scripts/                   [Ⅰ/Ⅱ] Operational scripts
├── tests/                     [Ⅰ/Ⅱ] Test suites with Phase II subdirectory
├── tools/                     [Ⅰ]  Verification and audit tools
├── ui/                        [Ⅰ]  Frontend dashboard
└── pyproject.toml             [Ⅰ]  Project dependencies
```

---

## Detailed Directory Mapping

### `/analysis/` — Phase II Dynamics Analysis [Ⅱ]

**Purpose**: Statistical analysis modules for Phase II U2 experiments.

| File | Phase | Purpose |
|------|-------|---------|
| `u2_dynamics.py` | Ⅱ | U2 uplift dynamics analysis |

**Dependencies**: `← backend/metrics/u2_analysis.py`, `← experiments/u2_pipeline.py`

---

### `/artifacts/` — Experiment Outputs [Ⅰ/Ⅱ]

**Purpose**: Durable outputs from experiments, attestations, and figures.

```
artifacts/
├── continuity/                [Ⅰ]  Lineage tracking
├── experiments/               [Ⅰ]  R0 baseline experiments
│   └── rfl/                   [Ⅰ]  RFL experiment outputs
├── figures/                   [Ⅰ/Ⅱ] Generated figures
├── first_organism/            [Ⅰ]  FO attestations
├── guidance/                  [Ⅰ]  Training data exports
├── keys/                      [Ⅰ]  Cryptographic key rotation logs
├── metrics/                   [Ⅰ]  Metrics snapshots
├── perf/                      [Ⅰ]  Performance baselines
├── phase_ii/                  [Ⅱ]  *** PHASE II OUTPUTS ***
│   ├── curriculum_hash_ledger.jsonl
│   ├── fo_series_1/           [Ⅱ]  First Organism series 1
│   │   ├── fo_1000_baseline/
│   │   └── fo_1000_rfl/
│   └── uplift_visualizations/ [Ⅱ]  U2 visualization artifacts
├── policy/                    [Ⅰ]  Policy exports
├── repro/                     [Ⅰ]  Reproducibility attestations
├── rfl/                       [Ⅰ]  RFL validation outputs
├── ship/                      [Ⅰ]  Release gate artifacts
├── u2/                        [Ⅱ]  *** U2 EXPERIMENT OUTPUTS ***
│   ├── analysis/              [Ⅱ]  Statistical summaries
│   └── ENV_A/                 [Ⅱ]  Environment A telemetry
└── wpv5/                      [Ⅰ]  Whitepaper v5 artifacts
```

**Boundary Rule**: Phase II outputs MUST be placed under `artifacts/phase_ii/` or `artifacts/u2/`.

---

### `/attestation/` — Dual-Root Attestation [Ⅰ]

**Purpose**: Cryptographic attestation infrastructure.

| File | Phase | Purpose |
|------|-------|---------|
| `__init__.py` | Ⅰ | Module init |
| `dual_root.py` | Ⅰ | Dual-root hash attestation |
| `README.md` | Ⅰ | Documentation |

---

### `/backend/` — Core Services [Ⅰ/Ⅱ]

**Purpose**: FastAPI services, axiom engine, metrics, and verification.

```
backend/
├── api/                       [Ⅰ]  API schemas
│   └── schemas.py             [Ⅰ]  Pydantic models
├── audit/                     [Ⅰ]  Audit tools
│   ├── mirror_auditor.py      [Ⅰ]  Mirror auditor
│   └── verify_phase_x_blocks.py [Ⅰ] Block verification
├── axiom_engine/              [Ⅰ]  Core derivation engine
│   ├── derive_core.py         [Ⅰ]  Core derivation logic
│   ├── derive_rules.py        [Ⅰ]  Inference rules
│   ├── derive.py              [Ⅰ]  Main derive module
│   ├── policy.py              [Ⅰ]  Policy scoring (RFL integration)
│   └── verification.py        [Ⅰ]  Proof verification
├── basis/                     [Ⅰ]  Canonical basis
├── bridge/                    [Ⅰ]  Bridge layer
├── causal/                    [Ⅰ]  Causal inference
├── consensus/                 [Ⅰ]  Harmony consensus
├── crypto/                    [Ⅰ]  Cryptographic primitives
├── dag/                       [Ⅰ]  Proof DAG
├── fol_eq/                    [Ⅰ]  First-order logic equality
├── frontier/                  [Ⅰ]  Curriculum frontier
├── generator/                 [Ⅰ]  Proposition generator
├── governance/                [Ⅰ]  Governance validation
├── ht/                        [Ⅰ]  H_t invariant checker
├── integration/               [Ⅰ]  Integration layer
├── lean_proj/                 [Ⅰ]  Lean 4 proofs
├── ledger/                    [Ⅰ]  Blockchain ledger
├── logic/                     [Ⅰ]  Logic canonicalization
├── metrics/                   [Ⅰ/Ⅱ] Metrics collection
│   ├── first_organism_telemetry.py [Ⅰ]
│   ├── fo_analytics.py        [Ⅰ]
│   ├── fo_feedback.py         [Ⅰ]
│   ├── fo_schema.py           [Ⅰ]
│   ├── statistical.py         [Ⅱ]  Phase II statistical analysis
│   └── u2_analysis.py         [Ⅱ]  *** PHASE II U2 ANALYSIS ***
├── models/                    [Ⅰ]  Data models
├── orchestrator/              [Ⅰ]  Job orchestration
├── phase_ix/                  [Ⅰ]  Phase IX attestation
├── promotion/                 [Ⅰ/Ⅱ] Basis promotion
│   └── u2_evidence.py         [Ⅱ]  U2 evidence collection
├── repro/                     [Ⅰ]  Reproducibility
├── rfl/                       [Ⅰ]  RFL backend support
├── runner/                    [Ⅱ]  *** PHASE II RUNNERS ***
│   └── u2_runner.py           [Ⅱ]  U2 experiment runner
├── security/                  [Ⅰ/Ⅱ] Security enforcement
│   └── u2_security.py         [Ⅱ]  U2 security policies
├── telemetry/                 [Ⅱ]  *** PHASE II TELEMETRY ***
│   └── u2_schema.py           [Ⅱ]  U2 telemetry schema
├── testing/                   [Ⅰ]  Hermetic testing
├── tools/                     [Ⅰ]  Backend utilities
├── ui/                        [Ⅰ]  Backend UI templates
├── verification/              [Ⅰ]  *** VERIFICATION TOOLS ONLY ***
└── worker.py                  [Ⅰ]  Worker process
```

**Boundary Rules**:
1. `backend/verification/` MUST contain ONLY verification tools
2. Phase II modules MUST be in: `metrics/`, `runner/`, `telemetry/`, `promotion/`, `security/`

---

### `/config/` — Configuration [Ⅰ/Ⅱ]

**Purpose**: YAML/JSON configuration files.

| File | Phase | Purpose |
|------|-------|---------|
| `allblue_lanes.json` | Ⅰ | AllBlue lane config |
| `causal/default.json` | Ⅰ | Causal defaults |
| `curriculum.yaml` | Ⅰ | Phase I curriculum |
| `curriculum_uplift_phase2.yaml` | Ⅱ | *** PHASE II SLICES *** |
| `first_organism.env` | Ⅰ | FO environment |
| `first_organism.env.template` | Ⅰ | FO env template |
| `nightly.env` | Ⅰ | Nightly env |
| `rfl/*.json` | Ⅰ | RFL configs |

**Dependency Arrow**: `config/curriculum_uplift_phase2.yaml` → `experiments/run_uplift_u2.py`

---

### `/curriculum/` — Curriculum Gates [Ⅰ]

**Purpose**: Curriculum slice definitions and gate logic.

| File | Phase | Purpose |
|------|-------|---------|
| `__init__.py` | Ⅰ | Module init |
| `config.py` | Ⅰ | Curriculum config loading |
| `gates.py` | Ⅰ | Gate success predicates |
| `README.md` | Ⅰ | Documentation |

---

### `/derivation/` — Derivation Specs [Ⅰ]

**Purpose**: First Organism derivation definitions.

| File | Phase | Purpose |
|------|-------|---------|
| `axioms.py` | Ⅰ | Axiom definitions |
| `bounds.py` | Ⅰ | Bounds checking |
| `derive_rules.py` | Ⅰ | Derivation rules |
| `derive_utils.py` | Ⅰ | Utilities |
| `first_organism_*.yaml` | Ⅰ | FO slice configs |
| `pipeline.py` | Ⅰ | Derivation pipeline |
| `structure.py` | Ⅰ | Structure analysis |
| `verification.py` | Ⅰ | Verification logic |

---

### `/docs/` — Documentation [Ⅰ/Ⅱ]

**Purpose**: Non-code documentation artifacts.

**Boundary Rule**: `docs/` MUST contain ONLY non-code artifacts (markdown, PDFs, images).

```
docs/
├── architecture/              [Ⅰ]  Architecture docs
├── audits/                    [Ⅰ]  Audit reports
├── ci/                        [Ⅰ]  CI documentation
├── evidence/                  [Ⅰ]  Evidence packs
├── fleet/                     [Ⅰ]  Fleet docs
├── integration/               [Ⅰ]  Integration guides
├── methods/                   [Ⅰ]  Methodology docs
├── ops/                       [Ⅰ]  Operations guides
├── perf/                      [Ⅰ/Ⅱ] Performance docs
│   ├── PHASE2_OPTIMIZATIONS.md [Ⅱ]
│   └── PHASE3_OPTIMIZATIONS.md [Ⅰ]
├── prereg/                    [Ⅱ]  *** PHASE II PREREGISTRATIONS ***
├── progress/                  [Ⅰ]  Progress tracking
├── qa/                        [Ⅰ]  QA reports
├── repro/                     [Ⅰ]  Reproducibility docs
├── security/                  [Ⅰ]  Security docs
├── workflows/                 [Ⅰ]  Workflow docs
├── PHASE2_DIRECTORY_MAP.md    [Ⅱ]  *** THIS DOCUMENT ***
├── PHASE2_RFL_UPLIFT_PLAN.md  [Ⅱ]  Phase II uplift plan
└── ... (other docs)
```

---

### `/experiments/` — Experiment Runners [Ⅰ/Ⅱ]

**Purpose**: Experiment execution and analysis.

```
experiments/
├── prereg/                    [Ⅰ/Ⅱ] Preregistration files
│   ├── PREREG_UPLIFT_U1.json  [Ⅰ]
│   ├── PREREG_UPLIFT_U1.md    [Ⅰ]
│   └── PREREG_UPLIFT_U2.yaml  [Ⅱ]  *** PHASE II PREREG ***
├── rfl/                       [Ⅰ]  RFL experiment support
│   ├── analysis/              [Ⅰ]
│   ├── configs/               [Ⅰ]
│   ├── derive_wrapper.py      [Ⅰ]
│   ├── run_experiment.py      [Ⅰ]
│   └── setup_db.py            [Ⅰ]
├── results/                   [Ⅰ]  Local results (deprecated)
├── synthetic_uplift/          [Ⅱ]  *** PHASE II SYNTHETIC DATA ***
│   ├── generate_synthetic_logs.py [Ⅱ]
│   ├── synthetic_slices.yaml  [Ⅱ]
│   └── tests/                 [Ⅱ]
├── run_fo_cycles.py           [Ⅰ]  First Organism cycles
├── run_uplift_u1.py           [Ⅰ]  U1 uplift runner
├── run_uplift_u2.py           [Ⅱ]  *** PHASE II U2 RUNNER ***
├── u2_cross_slice_analysis.py [Ⅱ]  *** PHASE II ANALYSIS ***
├── u2_pipeline.py             [Ⅱ]  *** PHASE II PIPELINE ***
├── curriculum_hash_ledger.py  [Ⅱ]  Hash ledger
├── curriculum_loader_v2.py    [Ⅱ]  V2 loader
├── ... (analysis scripts)
└── PHASE1_RFL_SUMMARY.md      [Ⅰ]  Phase I summary
```

**Boundary Rules**:
1. Phase II runtime MUST be prefixed with `u2_` or in `synthetic_uplift/`
2. Phase II preregistrations MUST be in `prereg/` with `U2` suffix

---

### `/infra/` — Infrastructure [Ⅰ]

**Purpose**: Docker and deployment configuration.

| File | Phase | Purpose |
|------|-------|---------|
| `docker-compose.yml` | Ⅰ | Docker services |
| `env.example` | Ⅰ | Environment template |

---

### `/migrations/` — Database Migrations [Ⅰ]

**Purpose**: SQL schema migrations.

All migrations are Phase I infrastructure.

---

### `/ops/` — Operations [Ⅰ/Ⅱ]

**Purpose**: Operational tooling and runbooks.

```
ops/
├── audit/                     [Ⅰ]  Audit logs
├── checks/                    [Ⅰ]  Health checks
├── first_organism/            [Ⅰ]  FO deployment
├── infra/                     [Ⅰ]  Infrastructure scripts
├── logs/                      [Ⅰ/Ⅱ] Operation logs
│   └── u2_compliance.jsonl    [Ⅱ]  U2 compliance log
├── microagents/               [Ⅰ]  Agent templates
├── microtasks/                [Ⅰ]  Task queues
├── scripts/                   [Ⅰ]  Operational scripts
├── tools/                     [Ⅰ]  Spark tools
└── README.md                  [Ⅰ]  Operations guide
```

---

### `/results/` — Raw Outputs [Ⅰ/Ⅱ]

**Purpose**: Raw JSONL experiment outputs.

**Naming Convention**:
- `uplift_u1_*.jsonl` — Phase I U1 outputs
- `uplift_u2_*.jsonl` — Phase II U2 outputs
- `fo_*.jsonl` — First Organism outputs
- `debug_*.jsonl` — Development outputs

---

### `/rfl/` — RFL Core Library [Ⅰ]

**Purpose**: Reflexive Feedback Loop core implementation.

| File | Phase | Purpose |
|------|-------|---------|
| `__init__.py` | Ⅰ | Module init |
| `audit.py` | Ⅰ | RFL audit |
| `bootstrap_stats.py` | Ⅰ | Bootstrap statistics |
| `config.py` | Ⅰ | RFL configuration |
| `coverage.py` | Ⅰ | Coverage tracking |
| `experiment_logging.py` | Ⅰ | Experiment logging |
| `experiment.py` | Ⅰ | Experiment runner |
| `metrics_logger.py` | Ⅰ | Metrics logging |
| `provenance.py` | Ⅰ | Provenance tracking |
| `runner.py` | Ⅰ | RFL runner |

---

### `/scripts/` — Operational Scripts [Ⅰ/Ⅱ]

**Purpose**: CLI scripts and automation.

```
scripts/
├── db/                        [Ⅰ]  Database scripts
├── rfl/                       [Ⅰ]  RFL scripts
├── sql/                       [Ⅰ]  SQL scripts
├── telemetry/                 [Ⅰ/Ⅱ] Telemetry scripts
│   └── uplift_eval.py         [Ⅱ]  U2 evaluation
├── build_u2_evidence_dossier.py [Ⅱ] *** PHASE II ***
├── proof_dag_u2_audit.py      [Ⅱ]  U2 DAG audit
├── validate_u2_environment.py [Ⅱ]  U2 env validation
├── verify_directory_structure.py [Ⅱ] *** DIRECTORY LINTER ***
├── run-migrations.py          [Ⅰ]  Migration runner
├── sanity.ps1                 [Ⅰ]  Sanity check
└── ... (other scripts)
```

---

### `/tests/` — Test Suites [Ⅰ/Ⅱ]

**Purpose**: Pytest test suites.

```
tests/
├── audit/                     [Ⅰ]  Audit tests
├── devxp/                     [Ⅰ]  DevX tests
├── env/                       [Ⅱ]  Environment tests
│   └── test_validate_u2_environment.py [Ⅱ]
├── experiments/               [Ⅰ/Ⅱ] Experiment tests
├── fixtures/                  [Ⅰ]  Test fixtures
├── frontier/                  [Ⅰ]  Frontier tests
├── governance/                [Ⅰ]  Governance tests
├── helpers/                   [Ⅰ]  Test helpers
├── ht/                        [Ⅰ]  H_t tests
├── integration/               [Ⅰ]  Integration tests
├── interop/                   [Ⅰ]  Interop tests
├── metrics/                   [Ⅱ]  Metrics tests
├── perf/                      [Ⅰ]  Performance tests
├── phase2/                    [Ⅱ]  *** PHASE II TESTS ONLY ***
│   ├── __init__.py            [Ⅱ]
│   └── metrics/               [Ⅱ]
│       ├── conftest.py        [Ⅱ]
│       ├── test_goal_hit.py   [Ⅱ]
│       └── test_sparse_density.py [Ⅱ]
├── plugins/                   [Ⅰ]  Pytest plugins
├── qa/                        [Ⅰ]  QA tests
├── rfl/                       [Ⅰ]  RFL tests
├── statistical/               [Ⅰ]  Stats tests
├── unit/                      [Ⅰ]  Unit tests
├── test_directory_structure.py [Ⅱ] *** STRUCTURE TESTS ***
└── conftest.py                [Ⅰ]  Shared fixtures
```

**Boundary Rule**: Phase II tests MUST be in `tests/phase2/` subdirectory.

---

### `/tools/` — Verification Tools [Ⅰ]

**Purpose**: Audit, verification, and analysis tools.

```
tools/
├── ci/                        [Ⅰ]  CI tools
├── ci-local/                  [Ⅰ]  Local CI tools
├── crypto/                    [Ⅰ]  Crypto verification
├── devin_e_toolbox/           [Ⅰ]  Agent toolbox
├── docs/                      [Ⅰ]  Documentation tools
├── hermetic/                  [Ⅰ]  Hermetic test tools
├── perf/                      [Ⅰ]  Performance tools
├── repro/                     [Ⅰ]  Reproducibility tools
├── simulation/                [Ⅰ]  Simulation tools
└── ... (verification scripts)
```

---

### `/ui/` — Frontend Dashboard [Ⅰ]

**Purpose**: Svelte dashboard for ledger visualization.

```
ui/
├── package.json               [Ⅰ]  NPM dependencies
└── src/
    └── lib/
        └── mathledger-client.js [Ⅰ] API client
```

---

## Dependency Graph

```
                    ┌─────────────────┐
                    │   config/       │
                    │ curriculum*.yaml│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────┐ ┌─────────────────┐
    │  experiments/   │ │ backend/│ │    rfl/         │
    │  run_uplift_*.py│ │ metrics/│ │   runner.py     │
    └────────┬────────┘ └────┬────┘ └────────┬────────┘
             │               │               │
             └───────────────┼───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    results/     │
                    │ uplift_*.jsonl  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────┐ ┌─────────────────┐
    │   artifacts/    │ │analysis/│ │     tests/      │
    │   phase_ii/     │ │u2_*.py  │ │    phase2/      │
    └─────────────────┘ └─────────┘ └─────────────────┘
```

---

## Phase II Specific Paths

### Must Contain ONLY Phase II Code

| Directory | Contents |
|-----------|----------|
| `tests/phase2/` | Phase II test suites |
| `artifacts/phase_ii/` | Phase II experiment outputs |
| `artifacts/u2/` | U2 telemetry and analysis |
| `experiments/synthetic_uplift/` | Synthetic uplift data |

### Phase II Files (Outside Designated Directories)

Files with Phase II content in mixed directories:

| File | Phase | Rationale |
|------|-------|-----------|
| `backend/metrics/u2_analysis.py` | Ⅱ | U2 analysis module |
| `backend/metrics/statistical.py` | Ⅱ | Phase II statistics |
| `backend/runner/u2_runner.py` | Ⅱ | U2 runner |
| `backend/telemetry/u2_schema.py` | Ⅱ | U2 telemetry schema |
| `backend/security/u2_security.py` | Ⅱ | U2 security |
| `backend/promotion/u2_evidence.py` | Ⅱ | U2 evidence |
| `config/curriculum_uplift_phase2.yaml` | Ⅱ | Phase II slices |
| `experiments/run_uplift_u2.py` | Ⅱ | U2 experiment runner |
| `experiments/u2_*.py` | Ⅱ | U2 analysis modules |
| `experiments/prereg/PREREG_UPLIFT_U2.yaml` | Ⅱ | U2 preregistration |
| `scripts/validate_u2_environment.py` | Ⅱ | U2 env validation |
| `scripts/build_u2_evidence_dossier.py` | Ⅱ | U2 evidence |
| `scripts/proof_dag_u2_audit.py` | Ⅱ | U2 DAG audit |

---

## Boundary Enforcement Rules

### Rule 1: Phase II Isolation

Phase II code MUST NOT be imported by Phase I modules.

```python
# FORBIDDEN in Phase I code:
from backend.metrics.u2_analysis import ...
from experiments.run_uplift_u2 import ...
from tests.phase2 import ...
```

### Rule 2: Phase II Labels

All Phase II files MUST contain a phase marker in the first 50 lines:

```python
# PHASE II — U2 UPLIFT EXPERIMENT
```

Or in YAML/Markdown:
```yaml
# PHASE II — U2 UPLIFT EXPERIMENT
```

### Rule 3: Directory Purity

| Directory | Allowed Contents |
|-----------|------------------|
| `tests/phase2/` | ONLY Phase II tests |
| `backend/verification/` | ONLY verification tools |
| `docs/` | ONLY non-code artifacts |
| `artifacts/phase_ii/` | ONLY Phase II outputs |

### Rule 4: Naming Conventions

Phase II files SHOULD follow naming patterns:
- `u2_*.py` — U2 specific modules
- `*_phase2.yaml` — Phase II configs
- `PREREG_UPLIFT_U2.*` — U2 preregistrations
- `uplift_u2_*.jsonl` — U2 results

---

## Linter Integration

The directory structure is enforced by:

```bash
# Run structure linter
uv run python scripts/verify_directory_structure.py

# Run structure tests
uv run pytest tests/test_directory_structure.py -v
```

### CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Verify Directory Structure
  run: uv run python scripts/verify_directory_structure.py --strict
```

---

## Changelog

| Date | Change | Agent |
|------|--------|-------|
| 2025-12-06 | Initial cartography | doc-ops-3 |

---

## References

- [PHASE2_RFL_UPLIFT_PLAN.md](./PHASE2_RFL_UPLIFT_PLAN.md) — Phase II uplift methodology
- [CONTRIBUTING.md](./CONTRIBUTING.md) — Contribution guidelines
- [ARCHITECTURE.md](./ARCHITECTURE.md) — System architecture


