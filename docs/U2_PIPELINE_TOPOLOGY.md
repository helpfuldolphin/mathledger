# U2 Pipeline Topology Specification

## PHASE II — NOT RUN IN PHASE I

This document specifies the complete execution pipeline topology for U2 uplift experiments, including node semantics, edge constraints, failure modes, and CI integration.

**Version:** 1.0.0
**Author:** CLAUDE_B (Workflow Pipeline Architect)
**Status:** PHASE II DESIGN — NOT YET EXECUTED
**Date:** 2025-12-06

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Directed Graph Specification](#directed-graph-specification)
3. [Node Semantics](#node-semantics)
4. [Edge Constraints](#edge-constraints)
5. [Failure Routing Table](#failure-routing-table)
6. [Pipeline Failure Mode Atlas](#pipeline-failure-mode-atlas)
7. [CI Workflow Template](#ci-workflow-template)
8. [Synthetic Worked Example](#synthetic-worked-example)

---

## Pipeline Overview

The U2 pipeline orchestrates Phase II uplift experiments across four asymmetric slices. It enforces determinism, governance compliance, and rigorous failure handling.

### Pipeline Principles

| Principle | Enforcement |
|-----------|-------------|
| **Determinism** | All runs seeded with `MDAP_SEED + cycle_index` |
| **Governance** | `VSD_PHASE_2_uplift_gate` checked before execution |
| **Isolation** | Each slice runs in isolated environment |
| **Auditability** | Every stage produces checksummed artifacts |
| **Fail-Safe** | Failures captured, never hidden |

### Pipeline Stages (High-Level)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  GATE       │───►│  PREPARE    │───►│  EXECUTE    │───►│  ANALYZE    │
│  CHECK      │    │  STAGE      │    │  STAGE      │    │  STAGE      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  SEAL       │◄───│  PACK       │◄───│  AUDIT      │◄───│  SUMMARIZE  │
│  STAGE      │    │  STAGE      │    │  STAGE      │    │  STAGE      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Directed Graph Specification

### Complete Pipeline DAG

```
                              ┌─────────────────────────────────────────────────────────────┐
                              │                    U2 PIPELINE DAG                          │
                              │              PHASE II — NOT RUN IN PHASE I                  │
                              └─────────────────────────────────────────────────────────────┘

                                                    START
                                                      │
                                                      ▼
                                          ┌─────────────────────┐
                                          │    N01: GATE_CHECK  │
                                          │    [Gatekeeper]     │
                                          └──────────┬──────────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    │ GATE_PASS      │                │ GATE_FAIL
                                    ▼                │                ▼
                          ┌─────────────────────┐    │      ┌─────────────────────┐
                          │  N02: PREREG_VERIFY │    │      │  N99: ABORT_GATE    │
                          │  [Validator]        │    │      │  [Terminator]       │
                          └──────────┬──────────┘    │      └─────────────────────┘
                                     │               │
                                     ▼               │
                          ┌─────────────────────┐    │
                          │  N03: CURRICULUM    │    │
                          │      LOAD           │    │
                          │  [Loader]           │    │
                          └──────────┬──────────┘    │
                                     │               │
                                     ▼               │
                          ┌─────────────────────┐    │
                          │  N04: DRY_RUN       │    │
                          │  [Validator]        │    │
                          └──────────┬──────────┘    │
                                     │               │
                                     ▼               │
                          ┌─────────────────────┐    │
                          │  N05: MANIFEST_INIT │    │
                          │  [Initializer]      │    │
                          └──────────┬──────────┘    │
                                     │               │
         ┌───────────────────────────┼───────────────────────────────┐
         │                           │                               │
         │              PARALLEL SLICE EXECUTION                     │
         │                           │                               │
         ▼                           ▼                               ▼
┌─────────────────┐       ┌─────────────────┐            ┌─────────────────┐
│ N10: SLICE_GOAL │       │ N11: SLICE_SPARSE│           │ N12: SLICE_TREE │
│ [Runner]        │       │ [Runner]         │           │ [Runner]        │
└────────┬────────┘       └────────┬─────────┘           └────────┬────────┘
         │                         │                              │
         │                         │                              │
         ▼                         ▼                              ▼
┌─────────────────┐       ┌─────────────────┐            ┌─────────────────┐
│ N20: EVAL_GOAL  │       │ N21: EVAL_SPARSE │           │ N22: EVAL_TREE  │
│ [Evaluator]     │       │ [Evaluator]      │           │ [Evaluator]     │
└────────┬────────┘       └────────┬─────────┘           └────────┬────────┘
         │                         │                              │
         │                         │                              │
         └─────────────────────────┼──────────────────────────────┘
                                   │
                                   │         ┌─────────────────┐
                                   │         │ N13: SLICE_DEP  │
                                   │         │ [Runner]        │
                                   │         └────────┬────────┘
                                   │                  │
                                   │                  ▼
                                   │         ┌─────────────────┐
                                   │         │ N23: EVAL_DEP   │
                                   │         │ [Evaluator]     │
                                   │         └────────┬────────┘
                                   │                  │
                                   ▼                  │
                          ┌─────────────────────┐     │
                          │  N30: SYNC_BARRIER  │◄────┘
                          │  [Synchronizer]     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  N40: CHAIN_ANALYZE │
                          │  [Analyzer]         │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  N41: DIAGNOSTICS   │
                          │  [Diagnostician]    │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  N50: STAT_SUMMARY  │
                          │  [Summarizer]       │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  N60: AUDIT         │
                          │  [Auditor]          │
                          └──────────┬──────────┘
                                     │
                          ┌──────────┴──────────┐
                          │ AUDIT_PASS          │ AUDIT_FAIL
                          ▼                     ▼
                ┌─────────────────────┐  ┌─────────────────────┐
                │  N70: PACK_BUILD    │  │  N98: QUARANTINE    │
                │  [Packager]         │  │  [Isolator]         │
                └──────────┬──────────┘  └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  N80: ATTESTATION   │
                │  [Attester]         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  N90: SEAL          │
                │  [Sealer]           │
                └──────────┬──────────┘
                           │
                           ▼
                         FINISH
```

### Node Registry

| Node ID | Name | Type | Parallelizable | Critical |
|---------|------|------|----------------|----------|
| N01 | GATE_CHECK | Gatekeeper | No | Yes |
| N02 | PREREG_VERIFY | Validator | No | Yes |
| N03 | CURRICULUM_LOAD | Loader | No | Yes |
| N04 | DRY_RUN | Validator | No | Yes |
| N05 | MANIFEST_INIT | Initializer | No | Yes |
| N10 | SLICE_GOAL | Runner | Yes | Yes |
| N11 | SLICE_SPARSE | Runner | Yes | Yes |
| N12 | SLICE_TREE | Runner | Yes | Yes |
| N13 | SLICE_DEP | Runner | Yes | Yes |
| N20 | EVAL_GOAL | Evaluator | Yes | Yes |
| N21 | EVAL_SPARSE | Evaluator | Yes | Yes |
| N22 | EVAL_TREE | Evaluator | Yes | Yes |
| N23 | EVAL_DEP | Evaluator | Yes | Yes |
| N30 | SYNC_BARRIER | Synchronizer | No | Yes |
| N40 | CHAIN_ANALYZE | Analyzer | No | No |
| N41 | DIAGNOSTICS | Diagnostician | No | No |
| N50 | STAT_SUMMARY | Summarizer | No | Yes |
| N60 | AUDIT | Auditor | No | Yes |
| N70 | PACK_BUILD | Packager | No | Yes |
| N80 | ATTESTATION | Attester | No | Yes |
| N90 | SEAL | Sealer | No | Yes |
| N98 | QUARANTINE | Isolator | No | N/A |
| N99 | ABORT_GATE | Terminator | No | N/A |

---

## Node Semantics

### Node Type Definitions

#### Type: Gatekeeper

**Purpose:** Enforce preconditions before pipeline execution.

```python
class Gatekeeper:
    """
    Validates governance gates and preconditions.
    MUST pass before any experiment execution.
    """

    def check(self) -> GateResult:
        """
        Returns:
            GateResult with status PASS/FAIL and reasons
        """

    def get_blocking_conditions(self) -> List[str]:
        """List conditions that would block execution."""
```

**N01: GATE_CHECK Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Environment variables, file system state |
| **Output** | `gate_check_result.json` |
| **Success** | All gates pass |
| **Failure** | Route to N99: ABORT_GATE |
| **Timeout** | 30 seconds |
| **Retries** | 0 (no retry on gate failure) |

**Checks Performed:**
1. `VSD_PHASE_2_uplift_gate` condition met
2. Evidence Pack v1 sealed and present
3. No Phase I logs in output directory
4. Preregistration file exists
5. Required environment variables set

---

#### Type: Validator

**Purpose:** Verify configuration and artifact integrity.

```python
class Validator:
    """
    Validates configurations, schemas, and artifact integrity.
    Prevents invalid experiments from starting.
    """

    def validate(self, target: Path) -> ValidationResult:
        """
        Returns:
            ValidationResult with errors/warnings
        """

    def get_schema(self) -> JSONSchema:
        """Return the validation schema."""
```

**N02: PREREG_VERIFY Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | `experiments/prereg/PREREG_UPLIFT_U2.yaml` |
| **Output** | `prereg_validation.json` |
| **Success** | YAML valid, hash matches (if frozen), sections present |
| **Failure** | Route to N99: ABORT_GATE |
| **Timeout** | 10 seconds |
| **Retries** | 0 |

**N04: DRY_RUN Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Curriculum config, evaluator registry |
| **Output** | `dry_run_report.json` |
| **Success** | All slices load, all evaluators resolve |
| **Failure** | Route to N99: ABORT_GATE |
| **Timeout** | 60 seconds |
| **Retries** | 1 |

---

#### Type: Loader

**Purpose:** Load and parse configuration files.

```python
class Loader:
    """
    Loads configuration from YAML/JSON files.
    Validates against Pydantic schemas.
    """

    def load(self, path: Path) -> Config:
        """Load and validate configuration."""

    def get_slices(self) -> List[SliceConfig]:
        """Return all slice configurations."""
```

**N03: CURRICULUM_LOAD Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | `config/curriculum_uplift_phase2.yaml` |
| **Output** | In-memory `CurriculumConfig` object |
| **Success** | All 4 slices parsed and validated |
| **Failure** | Route to N99: ABORT_GATE |
| **Timeout** | 15 seconds |
| **Retries** | 2 |

---

#### Type: Initializer

**Purpose:** Initialize artifacts and directories.

```python
class Initializer:
    """
    Creates output directories and initial manifest.
    Sets up logging infrastructure.
    """

    def initialize(self, output_dir: Path) -> InitResult:
        """Create directories and initial files."""

    def create_manifest(self) -> Manifest:
        """Create experiment manifest with metadata."""
```

**N05: MANIFEST_INIT Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Output directory path, experiment config |
| **Output** | `experiment_manifest.json` (initial) |
| **Success** | Directories created, manifest written |
| **Failure** | Route to N99: ABORT_GATE |
| **Timeout** | 10 seconds |
| **Retries** | 3 |

---

#### Type: Runner

**Purpose:** Execute uplift experiments on slices.

```python
class Runner:
    """
    Executes baseline and RFL runs on a single slice.
    Produces JSONL logs with per-cycle results.
    """

    def run_baseline(self, config: SliceConfig, seed: int) -> RunResult:
        """Execute baseline (random) run."""

    def run_rfl(self, config: SliceConfig, seed: int) -> RunResult:
        """Execute RFL (policy-scored) run."""

    def get_logs(self) -> Tuple[Path, Path]:
        """Return paths to baseline and RFL logs."""
```

**N10-N13: SLICE_* Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | `SliceConfig`, base seed, cycle count |
| **Output** | `{slice}_baseline.jsonl`, `{slice}_rfl.jsonl` |
| **Success** | Both runs complete, logs valid |
| **Failure** | Partial results saved, error logged |
| **Timeout** | 30 minutes per slice |
| **Retries** | 1 (with checkpoint resume) |

**Parallelism Rules:**
- N10, N11, N12, N13 MAY run in parallel
- Each runner has isolated output directory
- No shared mutable state between runners
- Seed schedule ensures determinism regardless of execution order

---

#### Type: Evaluator

**Purpose:** Evaluate success metrics per cycle.

```python
class Evaluator:
    """
    Evaluates cycle results against success criteria.
    Computes per-slice success/failure classification.
    """

    def evaluate(self, log_path: Path, config: SuccessMetricConfig) -> EvalResult:
        """Evaluate all cycles in log file."""

    def get_success_rate(self) -> float:
        """Return overall success rate."""
```

**N20-N23: EVAL_* Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | JSONL log files, success metric config |
| **Output** | `{slice}_eval_result.json` |
| **Success** | All cycles evaluated |
| **Failure** | Partial evaluation with error markers |
| **Timeout** | 5 minutes per slice |
| **Retries** | 2 |

---

#### Type: Synchronizer

**Purpose:** Barrier synchronization for parallel stages.

```python
class Synchronizer:
    """
    Waits for all parallel stages to complete.
    Collects results and validates completeness.
    """

    def wait_all(self, node_ids: List[str]) -> SyncResult:
        """Wait for all specified nodes to complete."""

    def collect_results(self) -> Dict[str, NodeResult]:
        """Collect results from all synchronized nodes."""
```

**N30: SYNC_BARRIER Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Completion signals from N20-N23 |
| **Output** | `sync_barrier_result.json` |
| **Success** | All 4 evaluators completed |
| **Failure** | Timeout or missing results |
| **Timeout** | 5 minutes (after last evaluator) |
| **Retries** | 0 |

---

#### Type: Analyzer

**Purpose:** Deep analysis of experiment results.

```python
class Analyzer:
    """
    Performs specialized analysis (chain depth, etc.).
    Produces analysis artifacts for diagnostics.
    """

    def analyze(self, data: AnalysisInput) -> AnalysisResult:
        """Perform analysis and return results."""
```

**N40: CHAIN_ANALYZE Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | `slice_uplift_tree` results, proof DAG |
| **Output** | `chain_depth_stats.json` |
| **Success** | Chain analysis complete |
| **Failure** | Degrade gracefully, continue pipeline |
| **Timeout** | 10 minutes |
| **Retries** | 1 |

---

#### Type: Diagnostician

**Purpose:** Generate diagnostic summaries.

```python
class Diagnostician:
    """
    Aggregates per-cycle metrics into slice-level diagnostics.
    Generates comparison tables and trend data.
    """

    def generate_diagnostics(self, eval_results: List[EvalResult]) -> DiagnosticsReport:
        """Generate comprehensive diagnostics report."""
```

**N41: DIAGNOSTICS Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | All evaluation results |
| **Output** | `diagnostics_report.json` |
| **Success** | Diagnostics generated |
| **Failure** | Degrade gracefully, continue pipeline |
| **Timeout** | 5 minutes |
| **Retries** | 2 |

---

#### Type: Summarizer

**Purpose:** Compute statistical summaries.

```python
class Summarizer:
    """
    Computes statistical metrics per PREREG_UPLIFT_U2.yaml.
    Generates confidence intervals and p-values.
    """

    def generate_summary(self, baseline: Path, rfl: Path) -> StatisticalSummary:
        """Generate statistical summary for slice pair."""

    def aggregate_summaries(self, summaries: List[StatisticalSummary]) -> AggregatedSummary:
        """Aggregate summaries across all slices."""
```

**N50: STAT_SUMMARY Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Diagnostics report, all log files |
| **Output** | `statistical_summary.json` |
| **Success** | All metrics computed with CIs |
| **Failure** | Fail-fast, route to N98: QUARANTINE |
| **Timeout** | 10 minutes |
| **Retries** | 1 |

---

#### Type: Auditor

**Purpose:** Validate experiment integrity.

```python
class Auditor:
    """
    Audits experiment for compliance and integrity.
    Checks determinism, governance, and completeness.
    """

    def audit(self, experiment_dir: Path) -> AuditResult:
        """Perform comprehensive audit."""

    def check_determinism(self, log1: Path, log2: Path) -> bool:
        """Verify two runs with same seed are identical."""
```

**N60: AUDIT Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | All experiment artifacts |
| **Output** | `audit_report.json` |
| **Success** | All audit checks pass |
| **Failure** | Route to N98: QUARANTINE |
| **Timeout** | 15 minutes |
| **Retries** | 0 (audit failures are definitive) |

**Audit Checks:**
1. Preregistration hash unchanged
2. Manifest schema valid
3. Log file checksums match
4. No Phase I log contamination
5. Determinism spot-check (10 cycles replayed)
6. Statistical summary completeness

---

#### Type: Packager

**Purpose:** Build Evidence Pack v2.

```python
class Packager:
    """
    Assembles Evidence Pack v2 from experiment artifacts.
    Computes checksums and validates structure.
    """

    def build(self, experiment_dir: Path, output_dir: Path) -> PackResult:
        """Build Evidence Pack v2."""

    def validate_structure(self) -> bool:
        """Validate pack structure against schema."""
```

**N70: PACK_BUILD Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | All validated artifacts |
| **Output** | `evidence_pack_v2/` directory |
| **Success** | Pack built and validated |
| **Failure** | Fail-fast, route to N98: QUARANTINE |
| **Timeout** | 10 minutes |
| **Retries** | 2 |

---

#### Type: Attester

**Purpose:** Generate attestation document.

```python
class Attester:
    """
    Generates cryptographic attestation for Evidence Pack.
    Records builder identity, timestamps, and hashes.
    """

    def attest(self, pack_dir: Path) -> Attestation:
        """Generate attestation for pack."""

    def sign(self, attestation: Attestation) -> SignedAttestation:
        """Sign attestation (if signing enabled)."""
```

**N80: ATTESTATION Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Evidence Pack v2 directory |
| **Output** | `attestation.json` |
| **Success** | Attestation generated with all hashes |
| **Failure** | Fail-fast, route to N98: QUARANTINE |
| **Timeout** | 5 minutes |
| **Retries** | 1 |

**CRITICAL CONSTRAINT:**
> Attestation (N80) MUST NOT run in parallel with uplift metric computation (N50).
> Attestation requires finalized metrics. Parallel execution would create race condition.

---

#### Type: Sealer

**Purpose:** Finalize and seal Evidence Pack.

```python
class Sealer:
    """
    Seals Evidence Pack v2 as immutable.
    Generates final manifest with seal timestamp.
    """

    def seal(self, pack_dir: Path) -> SealResult:
        """Seal the Evidence Pack."""

    def verify_seal(self, pack_dir: Path) -> bool:
        """Verify seal integrity."""
```

**N90: SEAL Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Attested Evidence Pack v2 |
| **Output** | `seal.json`, updated `manifest.json` |
| **Success** | Pack sealed and verified |
| **Failure** | Fail-fast (seal failure is critical) |
| **Timeout** | 5 minutes |
| **Retries** | 0 (seal is one-shot) |

---

#### Type: Isolator

**Purpose:** Quarantine failed experiments.

```python
class Isolator:
    """
    Quarantines failed experiments for investigation.
    Preserves all artifacts with failure metadata.
    """

    def quarantine(self, experiment_dir: Path, reason: str) -> QuarantineResult:
        """Move experiment to quarantine with metadata."""
```

**N98: QUARANTINE Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Failed experiment directory, failure reason |
| **Output** | `quarantine/{experiment_id}/` with `failure_report.json` |
| **Success** | Experiment quarantined |
| **Failure** | Log error, continue (best-effort) |
| **Timeout** | 5 minutes |
| **Retries** | 1 |

---

#### Type: Terminator

**Purpose:** Clean abort of pipeline.

```python
class Terminator:
    """
    Cleanly aborts pipeline execution.
    Logs reason and cleans up partial state.
    """

    def abort(self, reason: str) -> AbortResult:
        """Abort pipeline with reason."""
```

**N99: ABORT_GATE Semantics:**

| Aspect | Specification |
|--------|---------------|
| **Input** | Gate failure reason |
| **Output** | `abort_report.json` |
| **Success** | Clean abort completed |
| **Failure** | Force exit with error code |
| **Timeout** | 30 seconds |
| **Retries** | 0 |

---

## Edge Constraints

### Edge Constraint Matrix

| From | To | Type | Constraint |
|------|-----|------|------------|
| N01 | N02 | Sequential | GATE_PASS required |
| N01 | N99 | Failure | GATE_FAIL triggers |
| N02 | N03 | Sequential | Validation must pass |
| N03 | N04 | Sequential | Load must succeed |
| N04 | N05 | Sequential | Dry-run must pass |
| N05 | N10-N13 | Fan-out | Parallel allowed |
| N10 | N20 | Sequential | Run completes before eval |
| N11 | N21 | Sequential | Run completes before eval |
| N12 | N22 | Sequential | Run completes before eval |
| N13 | N23 | Sequential | Run completes before eval |
| N20-N23 | N30 | Barrier | All must complete |
| N30 | N40 | Sequential | Barrier releases |
| N40 | N41 | Sequential | Analysis before diagnostics |
| N41 | N50 | Sequential | Diagnostics before summary |
| N50 | N60 | Sequential | Summary before audit |
| N60 | N70 | Conditional | AUDIT_PASS required |
| N60 | N98 | Failure | AUDIT_FAIL triggers |
| N70 | N80 | Sequential | Pack before attest |
| N80 | N90 | Sequential | Attest before seal |

### Critical Edge Constraints

#### Constraint C1: No Skip Edges

> **No node may bypass its predecessors.**

```
INVALID: N05 ──────────────────► N50 (skips execution)
INVALID: N01 ──────────────────► N60 (skips all stages)
INVALID: N10 ──────────────────► N30 (skips evaluation)
```

**Enforcement:** Pipeline orchestrator validates DAG before execution.

#### Constraint C2: No Parallel Hazard (Attestation/Metrics)

> **N80 (ATTESTATION) must not run in parallel with N50 (STAT_SUMMARY).**

```
INVALID PARALLEL EXECUTION:

    ┌─────────────────┐     ┌─────────────────┐
    │ N50: STAT_SUMMARY│     │ N80: ATTESTATION │
    │ (still running)  │ ║ ║ │ (started early)  │
    └─────────────────┘     └─────────────────┘
                         ▲
                         │
                    RACE CONDITION!
                    Attestation may hash
                    incomplete metrics.
```

**Enforcement:** Strict edge N50 → N60 → N70 → N80 with no shortcuts.

#### Constraint C3: Barrier Synchronization

> **N30 (SYNC_BARRIER) must wait for ALL of N20, N21, N22, N23.**

```
VALID:
    N20 ─────┐
    N21 ─────┼────► N30
    N22 ─────┤
    N23 ─────┘

INVALID (partial sync):
    N20 ─────┐
    N21 ─────┼────► N30 (N22, N23 still running!)
```

**Enforcement:** Barrier counts expected inputs before releasing.

#### Constraint C4: Failure Isolation

> **Slice failures must not corrupt other slices.**

```
VALID:
    N10 (FAIL) ─► Error logged, N11/N12/N13 continue

INVALID:
    N10 (FAIL) ─► N11/N12/N13 abort (cascade failure)
```

**Enforcement:** Each runner has isolated error handling.

#### Constraint C5: Quarantine Before Retry

> **Failed experiments must be quarantined before any retry.**

```
VALID:
    N60 (AUDIT_FAIL) ─► N98 (QUARANTINE) ─► [new experiment]

INVALID:
    N60 (AUDIT_FAIL) ─► N01 (retry same data)
```

**Enforcement:** Quarantine clears experiment ID for fresh start.

---

## Failure Routing Table

### Failure Categories

| Category | Severity | Response | Retry |
|----------|----------|----------|-------|
| **GATE** | Critical | Abort pipeline | No |
| **CONFIG** | Critical | Abort pipeline | No |
| **RUNTIME** | High | Quarantine, log | Maybe |
| **ANALYSIS** | Medium | Degrade, continue | Yes |
| **AUDIT** | High | Quarantine | No |
| **PACK** | High | Quarantine | Maybe |
| **TRANSIENT** | Low | Retry | Yes |

### Failure Routing Matrix

| Node | Failure Type | Route To | Retry? | Degrade? |
|------|--------------|----------|--------|----------|
| N01 | Gate check failed | N99 | No | No |
| N02 | Prereg invalid | N99 | No | No |
| N02 | Prereg hash mismatch | N99 | No | No |
| N03 | YAML parse error | N99 | No | No |
| N03 | Schema validation error | N99 | No | No |
| N04 | Slice load failed | N99 | Yes (1) | No |
| N04 | Evaluator not found | N99 | No | No |
| N05 | Directory creation failed | N99 | Yes (3) | No |
| N05 | Manifest write failed | N99 | Yes (3) | No |
| N10-N13 | Timeout | N30 (partial) | Yes (1) | Yes |
| N10-N13 | Verification error | Continue | No | Yes |
| N10-N13 | Out of memory | N98 | No | No |
| N20-N23 | Eval error | Continue | Yes (2) | Yes |
| N30 | Timeout waiting | N98 | No | No |
| N30 | Missing results | N98 | No | No |
| N40 | Analysis failed | N41 | Yes (1) | Yes |
| N41 | Diagnostics failed | N50 | Yes (2) | Yes |
| N50 | Stats computation failed | N98 | Yes (1) | No |
| N60 | Audit failed | N98 | No | No |
| N70 | Pack build failed | N98 | Yes (2) | No |
| N80 | Attestation failed | N98 | Yes (1) | No |
| N90 | Seal failed | N98 | No | No |

### Degrade-Gracefully Paths

When a non-critical node fails, the pipeline can continue with degraded output:

```
N40 (CHAIN_ANALYZE) fails:
    ├── Log warning
    ├── Set chain_analysis_available = false in manifest
    └── Continue to N41

N41 (DIAGNOSTICS) fails:
    ├── Log warning
    ├── Set diagnostics_available = false in manifest
    └── Continue to N50 with partial data

Slice runner partial failure:
    ├── Save completed cycles to log
    ├── Mark slice as PARTIAL in manifest
    └── Continue with other slices
```

### Fail-Fast Paths

When a critical node fails, the pipeline must abort:

```
N01 (GATE_CHECK) fails:
    └── IMMEDIATE ABORT ─► N99

N02 (PREREG_VERIFY) fails:
    └── IMMEDIATE ABORT ─► N99

N60 (AUDIT) fails:
    └── QUARANTINE ─► N98

N90 (SEAL) fails:
    └── QUARANTINE ─► N98 (seal is irreversible)
```

---

## Pipeline Failure Mode Atlas

### PHASE II — NOT RUN IN PHASE I

This section catalogs 80+ failure conditions grouped by topology region.

### Failure Code Format

```
TOPO-{Region}-{Category}-{Number}
```

| Region | Description |
|--------|-------------|
| A | Gate and Validation (N01-N05) |
| B | Slice Execution (N10-N13) |
| C | Evaluation (N20-N23) |
| D | Synchronization (N30) |
| E | Analysis (N40-N41) |
| F | Statistics (N50) |
| G | Audit (N60) |
| H | Packaging (N70-N80) |
| I | Sealing (N90) |
| Z | Infrastructure |

---

### TOPO-A: Gate and Validation Failures

#### TOPO-A-GATE-001: Governance gate not satisfied

| Field | Value |
|-------|-------|
| **Condition** | `VSD_PHASE_2_uplift_gate` returns false |
| **Detection** | `python experiments/ci/check_gate.py` |
| **Severity** | Critical |
| **Mitigation** | Verify Evidence Pack v1 sealed, check prerequisites |
| **Retry** | No |
| **Exit Code** | 101 |

#### TOPO-A-GATE-002: Evidence Pack v1 not found

| Field | Value |
|-------|-------|
| **Condition** | `docs/evidence/EVIDENCE_PACK_V1_AUDIT_CURSOR_O.md` missing |
| **Detection** | `test -f docs/evidence/EVIDENCE_PACK_V1_AUDIT_CURSOR_O.md` |
| **Severity** | Critical |
| **Mitigation** | Complete Phase I and seal Evidence Pack v1 |
| **Retry** | No |
| **Exit Code** | 102 |

#### TOPO-A-GATE-003: Phase I logs in output directory

| Field | Value |
|-------|-------|
| **Condition** | `fo_rfl*.jsonl` or `fo_baseline*.jsonl` found in output dir |
| **Detection** | `python experiments/ci/check_phase1_contamination.py` |
| **Severity** | Critical |
| **Mitigation** | Use clean output directory for Phase II |
| **Retry** | No |
| **Exit Code** | 103 |

#### TOPO-A-GATE-004: Required environment variable missing

| Field | Value |
|-------|-------|
| **Condition** | `MDAP_SEED`, `DATABASE_URL`, or `REDIS_URL` not set |
| **Detection** | `python experiments/ci/check_env.py` |
| **Severity** | Critical |
| **Mitigation** | Set required environment variables |
| **Retry** | No |
| **Exit Code** | 104 |

#### TOPO-A-GATE-005: Experiment already running

| Field | Value |
|-------|-------|
| **Condition** | Lock file present in output directory |
| **Detection** | `test -f results/uplift_u2/.lock` |
| **Severity** | Critical |
| **Mitigation** | Wait for existing run or clear stale lock |
| **Retry** | No |
| **Exit Code** | 105 |

#### TOPO-A-PREREG-001: Preregistration file not found

| Field | Value |
|-------|-------|
| **Condition** | `experiments/prereg/PREREG_UPLIFT_U2.yaml` missing |
| **Detection** | `test -f experiments/prereg/PREREG_UPLIFT_U2.yaml` |
| **Severity** | Critical |
| **Mitigation** | Create preregistration document |
| **Retry** | No |
| **Exit Code** | 111 |

#### TOPO-A-PREREG-002: Preregistration YAML syntax error

| Field | Value |
|-------|-------|
| **Condition** | YAML parser raises exception |
| **Detection** | `python -c "import yaml; yaml.safe_load(open('...'))"` |
| **Severity** | Critical |
| **Mitigation** | Fix YAML syntax errors |
| **Retry** | No |
| **Exit Code** | 112 |

#### TOPO-A-PREREG-003: Preregistration hash mismatch

| Field | Value |
|-------|-------|
| **Condition** | SHA-256 differs from frozen hash after experiments started |
| **Detection** | `python experiments/ci/verify_prereg.py` |
| **Severity** | Critical |
| **Mitigation** | Cannot modify preregistration after start; abort |
| **Retry** | No |
| **Exit Code** | 113 |

#### TOPO-A-PREREG-004: Required preregistration section missing

| Field | Value |
|-------|-------|
| **Condition** | `protocol`, `success_metric`, or `hypothesis` missing |
| **Detection** | `python experiments/ci/verify_prereg.py --check-sections` |
| **Severity** | Critical |
| **Mitigation** | Add missing sections to preregistration |
| **Retry** | No |
| **Exit Code** | 114 |

#### TOPO-A-PREREG-005: Invalid preregistration version

| Field | Value |
|-------|-------|
| **Condition** | `version` field not recognized |
| **Detection** | `python experiments/ci/verify_prereg.py --check-version` |
| **Severity** | Critical |
| **Mitigation** | Update to supported version format |
| **Retry** | No |
| **Exit Code** | 115 |

#### TOPO-A-CURR-001: Curriculum file not found

| Field | Value |
|-------|-------|
| **Condition** | `config/curriculum_uplift_phase2.yaml` missing |
| **Detection** | `test -f config/curriculum_uplift_phase2.yaml` |
| **Severity** | Critical |
| **Mitigation** | Create curriculum file |
| **Retry** | No |
| **Exit Code** | 121 |

#### TOPO-A-CURR-002: Curriculum YAML syntax error

| Field | Value |
|-------|-------|
| **Condition** | YAML parser raises exception |
| **Detection** | `python -c "import yaml; yaml.safe_load(open('...'))"` |
| **Severity** | Critical |
| **Mitigation** | Fix YAML syntax errors |
| **Retry** | No |
| **Exit Code** | 122 |

#### TOPO-A-CURR-003: Slice schema validation failed

| Field | Value |
|-------|-------|
| **Condition** | Pydantic validation error on slice config |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Fix schema violations in curriculum |
| **Retry** | No |
| **Exit Code** | 123 |

#### TOPO-A-CURR-004: Unknown success_metric.kind

| Field | Value |
|-------|-------|
| **Condition** | Metric kind not in `[goal_hit, density, chain_length, multi_goal]` |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Use valid success metric kind |
| **Retry** | No |
| **Exit Code** | 124 |

#### TOPO-A-CURR-005: Duplicate slice name

| Field | Value |
|-------|-------|
| **Condition** | Two slices have same `name` field |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Rename duplicate slice |
| **Retry** | No |
| **Exit Code** | 125 |

#### TOPO-A-CURR-006: Invalid target hash format

| Field | Value |
|-------|-------|
| **Condition** | Target hash not valid 64-char hex string |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Fix hash format (64 hex chars) |
| **Retry** | No |
| **Exit Code** | 126 |

#### TOPO-A-CURR-007: Missing required slice parameter

| Field | Value |
|-------|-------|
| **Condition** | `atoms`, `depth_min`, `depth_max`, etc. missing |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Add missing parameters |
| **Retry** | No |
| **Exit Code** | 127 |

#### TOPO-A-CURR-008: Slice parameter out of range

| Field | Value |
|-------|-------|
| **Condition** | `depth_min > depth_max` or `atoms < 1` |
| **Detection** | `python experiments/ci/validate_curriculum_schema.py` |
| **Severity** | Critical |
| **Mitigation** | Fix parameter ranges |
| **Retry** | No |
| **Exit Code** | 128 |

#### TOPO-A-DRY-001: Curriculum loader initialization failed

| Field | Value |
|-------|-------|
| **Condition** | `CurriculumLoaderV2` constructor raises exception |
| **Detection** | `python experiments/run_uplift_u2.py --dry-run` |
| **Severity** | Critical |
| **Mitigation** | Fix curriculum loader code |
| **Retry** | Yes (1) |
| **Exit Code** | 131 |

#### TOPO-A-DRY-002: Evaluator factory missing evaluator

| Field | Value |
|-------|-------|
| **Condition** | `SuccessEvaluatorFactory.create(kind)` raises KeyError |
| **Detection** | `python experiments/run_uplift_u2.py --dry-run` |
| **Severity** | Critical |
| **Mitigation** | Register missing evaluator in factory |
| **Retry** | No |
| **Exit Code** | 132 |

#### TOPO-A-DRY-003: Dry-run timeout

| Field | Value |
|-------|-------|
| **Condition** | Dry-run exceeds 60 second timeout |
| **Detection** | `timeout 60 python experiments/run_uplift_u2.py --dry-run` |
| **Severity** | High |
| **Mitigation** | Investigate performance issue |
| **Retry** | Yes (1) |
| **Exit Code** | 133 |

#### TOPO-A-INIT-001: Output directory creation failed

| Field | Value |
|-------|-------|
| **Condition** | `mkdir -p` fails (permissions, disk full) |
| **Detection** | Check exit code of directory creation |
| **Severity** | Critical |
| **Mitigation** | Check permissions, disk space |
| **Retry** | Yes (3) |
| **Exit Code** | 141 |

#### TOPO-A-INIT-002: Manifest write failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot write `experiment_manifest.json` |
| **Detection** | Check file creation success |
| **Severity** | Critical |
| **Mitigation** | Check permissions, disk space |
| **Retry** | Yes (3) |
| **Exit Code** | 142 |

#### TOPO-A-INIT-003: Lock file creation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot create `.lock` file |
| **Detection** | Check file creation success |
| **Severity** | High |
| **Mitigation** | Check permissions |
| **Retry** | Yes (3) |
| **Exit Code** | 143 |

---

### TOPO-B: Slice Execution Failures

#### TOPO-B-RUN-001: Baseline run timeout

| Field | Value |
|-------|-------|
| **Condition** | Baseline run exceeds 30 minute timeout |
| **Detection** | Timeout wrapper around runner |
| **Severity** | High |
| **Mitigation** | Check for infinite loops, reduce cycle count |
| **Retry** | Yes (1) with checkpoint |
| **Exit Code** | 201 |

#### TOPO-B-RUN-002: RFL run timeout

| Field | Value |
|-------|-------|
| **Condition** | RFL run exceeds 30 minute timeout |
| **Detection** | Timeout wrapper around runner |
| **Severity** | High |
| **Mitigation** | Check policy scoring performance |
| **Retry** | Yes (1) with checkpoint |
| **Exit Code** | 202 |

#### TOPO-B-RUN-003: Seed initialization failed

| Field | Value |
|-------|-------|
| **Condition** | Random seed setup raises exception |
| **Detection** | Exception during `random.seed()` |
| **Severity** | Critical |
| **Mitigation** | Check seed value validity |
| **Retry** | No |
| **Exit Code** | 203 |

#### TOPO-B-RUN-004: Out of memory during run

| Field | Value |
|-------|-------|
| **Condition** | MemoryError raised during execution |
| **Detection** | Exception handler catches MemoryError |
| **Severity** | Critical |
| **Mitigation** | Reduce formula pool size, increase memory |
| **Retry** | No |
| **Exit Code** | 204 |

#### TOPO-B-RUN-005: Log file write failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot write to JSONL log file |
| **Detection** | IOError during log append |
| **Severity** | Critical |
| **Mitigation** | Check disk space, permissions |
| **Retry** | Yes (1) |
| **Exit Code** | 205 |

#### TOPO-B-RUN-006: Verification subsystem failed

| Field | Value |
|-------|-------|
| **Condition** | Truth table verifier raises exception |
| **Detection** | Exception during `taut_check()` |
| **Severity** | High |
| **Mitigation** | Check formula normalization |
| **Retry** | No |
| **Exit Code** | 206 |

#### TOPO-B-RUN-007: Policy scoring failed

| Field | Value |
|-------|-------|
| **Condition** | RFL policy scorer raises exception |
| **Detection** | Exception during `score_candidates()` |
| **Severity** | High |
| **Mitigation** | Check policy model, candidate format |
| **Retry** | Yes (1) |
| **Exit Code** | 207 |

#### TOPO-B-RUN-008: Candidate generation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot generate candidate formulas |
| **Detection** | Empty candidate list |
| **Severity** | High |
| **Mitigation** | Check formula pool configuration |
| **Retry** | No |
| **Exit Code** | 208 |

#### TOPO-B-RUN-009: Checkpoint save failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot save checkpoint for resume |
| **Detection** | IOError during checkpoint write |
| **Severity** | Medium |
| **Mitigation** | Check disk space |
| **Retry** | Continue without checkpoint |
| **Exit Code** | 209 |

#### TOPO-B-RUN-010: Checkpoint load failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot load checkpoint for resume |
| **Detection** | IOError or corrupt checkpoint |
| **Severity** | Medium |
| **Mitigation** | Start fresh run |
| **Retry** | Start from beginning |
| **Exit Code** | 210 |

#### TOPO-B-RUN-011: Invalid cycle result format

| Field | Value |
|-------|-------|
| **Condition** | Cycle result doesn't match schema |
| **Detection** | Schema validation on cycle result |
| **Severity** | Medium |
| **Mitigation** | Check runner output format |
| **Retry** | No |
| **Exit Code** | 211 |

#### TOPO-B-RUN-012: Zero candidates tried

| Field | Value |
|-------|-------|
| **Condition** | Cycle completed with 0 candidates |
| **Detection** | `candidates_tried == 0` in result |
| **Severity** | Medium |
| **Mitigation** | Check formula pool, budget settings |
| **Retry** | No |
| **Exit Code** | 212 |

#### TOPO-B-SLICE-001: slice_uplift_goal formula pool empty

| Field | Value |
|-------|-------|
| **Condition** | Goal slice has no formulas to try |
| **Detection** | Empty `formula_pool_entries` |
| **Severity** | Critical |
| **Mitigation** | Add formulas to curriculum |
| **Retry** | No |
| **Exit Code** | 221 |

#### TOPO-B-SLICE-002: slice_uplift_sparse too dense

| Field | Value |
|-------|-------|
| **Condition** | Verification rate > 90% (not sparse) |
| **Detection** | Post-hoc analysis of logs |
| **Severity** | Warning |
| **Mitigation** | Redesign slice for more sparsity |
| **Retry** | N/A (warning only) |
| **Exit Code** | 0 (warning) |

#### TOPO-B-SLICE-003: slice_uplift_tree no chain possible

| Field | Value |
|-------|-------|
| **Condition** | Target chain not derivable from axioms |
| **Detection** | Post-hoc analysis of proof DAG |
| **Severity** | Critical |
| **Mitigation** | Verify target formula derivability |
| **Retry** | No |
| **Exit Code** | 223 |

#### TOPO-B-SLICE-004: slice_uplift_dependency subgoal unreachable

| Field | Value |
|-------|-------|
| **Condition** | One or more subgoals not provable |
| **Detection** | Post-hoc analysis of logs |
| **Severity** | Critical |
| **Mitigation** | Verify subgoal derivability |
| **Retry** | No |
| **Exit Code** | 224 |

---

### TOPO-C: Evaluation Failures

#### TOPO-C-EVAL-001: Log file not found

| Field | Value |
|-------|-------|
| **Condition** | Expected JSONL log file missing |
| **Detection** | `test -f {slice}_baseline.jsonl` |
| **Severity** | Critical |
| **Mitigation** | Check runner completed successfully |
| **Retry** | No |
| **Exit Code** | 301 |

#### TOPO-C-EVAL-002: Log file empty

| Field | Value |
|-------|-------|
| **Condition** | Log file has 0 bytes |
| **Detection** | `wc -l {slice}_baseline.jsonl` |
| **Severity** | Critical |
| **Mitigation** | Check runner execution |
| **Retry** | No |
| **Exit Code** | 302 |

#### TOPO-C-EVAL-003: Log file corrupted

| Field | Value |
|-------|-------|
| **Condition** | JSONL contains invalid JSON lines |
| **Detection** | JSON parse error on log lines |
| **Severity** | High |
| **Mitigation** | Check for truncation, disk errors |
| **Retry** | Yes (re-run slice) |
| **Exit Code** | 303 |

#### TOPO-C-EVAL-004: Cycle count mismatch

| Field | Value |
|-------|-------|
| **Condition** | Log has fewer cycles than expected |
| **Detection** | Compare log line count vs config |
| **Severity** | High |
| **Mitigation** | Check for early termination |
| **Retry** | No (mark as PARTIAL) |
| **Exit Code** | 304 |

#### TOPO-C-EVAL-005: Missing required field in cycle

| Field | Value |
|-------|-------|
| **Condition** | Cycle result missing `verified`, `success`, etc. |
| **Detection** | Schema validation on cycle |
| **Severity** | High |
| **Mitigation** | Check runner output format |
| **Retry** | No |
| **Exit Code** | 305 |

#### TOPO-C-EVAL-006: Invalid hash in goal_hit evaluation

| Field | Value |
|-------|-------|
| **Condition** | Verified hash not valid hex64 |
| **Detection** | Hash format validation |
| **Severity** | Medium |
| **Mitigation** | Check hash computation |
| **Retry** | No |
| **Exit Code** | 306 |

#### TOPO-C-EVAL-007: Evaluator timeout

| Field | Value |
|-------|-------|
| **Condition** | Evaluation exceeds 5 minute timeout |
| **Detection** | Timeout wrapper |
| **Severity** | High |
| **Mitigation** | Check for very large logs |
| **Retry** | Yes (2) |
| **Exit Code** | 307 |

#### TOPO-C-EVAL-008: Success rate computation failed

| Field | Value |
|-------|-------|
| **Condition** | Division by zero or NaN result |
| **Detection** | Math validation |
| **Severity** | Medium |
| **Mitigation** | Check for empty cycles |
| **Retry** | No |
| **Exit Code** | 308 |

---

### TOPO-D: Synchronization Failures

#### TOPO-D-SYNC-001: Barrier timeout

| Field | Value |
|-------|-------|
| **Condition** | Not all evaluators complete in 5 minutes |
| **Detection** | Barrier wait timeout |
| **Severity** | Critical |
| **Mitigation** | Check for hung evaluators |
| **Retry** | No |
| **Exit Code** | 401 |

#### TOPO-D-SYNC-002: Missing evaluator result

| Field | Value |
|-------|-------|
| **Condition** | One or more evaluators didn't report |
| **Detection** | Result count < expected |
| **Severity** | Critical |
| **Mitigation** | Check evaluator logs |
| **Retry** | No |
| **Exit Code** | 402 |

#### TOPO-D-SYNC-003: Inconsistent result format

| Field | Value |
|-------|-------|
| **Condition** | Evaluator results have different schemas |
| **Detection** | Schema comparison |
| **Severity** | High |
| **Mitigation** | Standardize evaluator output |
| **Retry** | No |
| **Exit Code** | 403 |

#### TOPO-D-SYNC-004: Duplicate result received

| Field | Value |
|-------|-------|
| **Condition** | Same evaluator reports twice |
| **Detection** | Result deduplication |
| **Severity** | Warning |
| **Mitigation** | Take first result, log warning |
| **Retry** | N/A |
| **Exit Code** | 0 (warning) |

---

### TOPO-E: Analysis Failures

#### TOPO-E-CHAIN-001: Proof DAG not available

| Field | Value |
|-------|-------|
| **Condition** | Cannot load proof DAG for chain analysis |
| **Detection** | DAG load failure |
| **Severity** | Medium |
| **Mitigation** | Degrade gracefully, skip chain analysis |
| **Retry** | Yes (1) |
| **Exit Code** | 501 (degraded) |

#### TOPO-E-CHAIN-002: Cyclic DAG detected

| Field | Value |
|-------|-------|
| **Condition** | Proof DAG contains cycles |
| **Detection** | Cycle detection during traversal |
| **Severity** | Critical |
| **Mitigation** | Fix DAG construction code |
| **Retry** | No |
| **Exit Code** | 502 |

#### TOPO-E-CHAIN-003: Chain depth computation timeout

| Field | Value |
|-------|-------|
| **Condition** | BFS/DFS exceeds 10 minute timeout |
| **Detection** | Timeout wrapper |
| **Severity** | Medium |
| **Mitigation** | Optimize traversal, reduce DAG size |
| **Retry** | Yes (1) |
| **Exit Code** | 503 (degraded) |

#### TOPO-E-CHAIN-004: No chains found

| Field | Value |
|-------|-------|
| **Condition** | max_chain_length = 0 for all formulas |
| **Detection** | Post-analysis check |
| **Severity** | Warning |
| **Mitigation** | Check if any proofs succeeded |
| **Retry** | N/A |
| **Exit Code** | 0 (warning) |

#### TOPO-E-DIAG-001: Diagnostics generation failed

| Field | Value |
|-------|-------|
| **Condition** | `SliceDiagnostics.generate()` raises exception |
| **Detection** | Exception handler |
| **Severity** | Medium |
| **Mitigation** | Degrade gracefully |
| **Retry** | Yes (2) |
| **Exit Code** | 511 (degraded) |

#### TOPO-E-DIAG-002: Wilson CI computation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot compute Wilson score interval |
| **Detection** | Math exception |
| **Severity** | Medium |
| **Mitigation** | Fall back to normal approximation |
| **Retry** | No |
| **Exit Code** | 512 (degraded) |

#### TOPO-E-DIAG-003: Bootstrap CI computation failed

| Field | Value |
|-------|-------|
| **Condition** | Bootstrap resampling fails |
| **Detection** | Exception during bootstrap |
| **Severity** | Medium |
| **Mitigation** | Report point estimate only |
| **Retry** | Yes (1) |
| **Exit Code** | 513 (degraded) |

---

### TOPO-F: Statistics Failures

#### TOPO-F-STAT-001: Statistical summary generation failed

| Field | Value |
|-------|-------|
| **Condition** | `StatisticalSummaryU2.generate()` raises exception |
| **Detection** | Exception handler |
| **Severity** | Critical |
| **Mitigation** | Check input data format |
| **Retry** | Yes (1) |
| **Exit Code** | 601 |

#### TOPO-F-STAT-002: Insufficient samples for z-test

| Field | Value |
|-------|-------|
| **Condition** | n < 30 for normal approximation |
| **Detection** | Sample size check |
| **Severity** | Warning |
| **Mitigation** | Use exact test or report limitation |
| **Retry** | N/A |
| **Exit Code** | 0 (warning) |

#### TOPO-F-STAT-003: Effect size computation failed

| Field | Value |
|-------|-------|
| **Condition** | Cohen's h cannot be computed |
| **Detection** | Math exception |
| **Severity** | Low |
| **Mitigation** | Report as N/A |
| **Retry** | No |
| **Exit Code** | 603 (degraded) |

#### TOPO-F-STAT-004: Output schema validation failed

| Field | Value |
|-------|-------|
| **Condition** | `statistical_summary.json` doesn't match schema |
| **Detection** | JSON Schema validation |
| **Severity** | High |
| **Mitigation** | Fix summary generator |
| **Retry** | No |
| **Exit Code** | 604 |

#### TOPO-F-STAT-005: Missing slice in summary

| Field | Value |
|-------|-------|
| **Condition** | One slice not represented in summary |
| **Detection** | Slice count check |
| **Severity** | Critical |
| **Mitigation** | Check all evaluations completed |
| **Retry** | No |
| **Exit Code** | 605 |

---

### TOPO-G: Audit Failures

#### TOPO-G-AUDIT-001: Preregistration modified after start

| Field | Value |
|-------|-------|
| **Condition** | Hash differs from frozen value |
| **Detection** | `verify_prereg.py` |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — scientific integrity violation |
| **Retry** | No |
| **Exit Code** | 701 |

#### TOPO-G-AUDIT-002: Manifest schema invalid

| Field | Value |
|-------|-------|
| **Condition** | `experiment_manifest.json` fails validation |
| **Detection** | JSON Schema validation |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — fix manifest generator |
| **Retry** | No |
| **Exit Code** | 702 |

#### TOPO-G-AUDIT-003: Log file checksum mismatch

| Field | Value |
|-------|-------|
| **Condition** | Recorded checksum differs from computed |
| **Detection** | Checksum verification |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — possible tampering |
| **Retry** | No |
| **Exit Code** | 703 |

#### TOPO-G-AUDIT-004: Phase I log contamination detected

| Field | Value |
|-------|-------|
| **Condition** | References to `fo_rfl*.jsonl` found |
| **Detection** | Content scan |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — remove contamination |
| **Retry** | No |
| **Exit Code** | 704 |

#### TOPO-G-AUDIT-005: Determinism verification failed

| Field | Value |
|-------|-------|
| **Condition** | Replay produces different results |
| **Detection** | 10-cycle replay test |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — fix non-determinism |
| **Retry** | No |
| **Exit Code** | 705 |

#### TOPO-G-AUDIT-006: Statistical summary incomplete

| Field | Value |
|-------|-------|
| **Condition** | Missing required metrics |
| **Detection** | Field presence check |
| **Severity** | High |
| **Mitigation** | QUARANTINE — regenerate summary |
| **Retry** | No |
| **Exit Code** | 706 |

#### TOPO-G-AUDIT-007: Timestamp anomaly detected

| Field | Value |
|-------|-------|
| **Condition** | Timestamps not monotonic or in future |
| **Detection** | Timestamp validation |
| **Severity** | Medium |
| **Mitigation** | QUARANTINE — investigate clock issues |
| **Retry** | No |
| **Exit Code** | 707 |

#### TOPO-G-AUDIT-008: Missing required artifact

| Field | Value |
|-------|-------|
| **Condition** | Expected file not present |
| **Detection** | File existence check |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — check pipeline |
| **Retry** | No |
| **Exit Code** | 708 |

---

### TOPO-H: Packaging Failures

#### TOPO-H-PACK-001: Evidence Pack directory creation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot create `evidence_pack_v2/` |
| **Detection** | Directory creation error |
| **Severity** | Critical |
| **Mitigation** | Check permissions, disk space |
| **Retry** | Yes (2) |
| **Exit Code** | 801 |

#### TOPO-H-PACK-002: Artifact copy failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot copy file to pack |
| **Detection** | Copy operation error |
| **Severity** | Critical |
| **Mitigation** | Check permissions, disk space |
| **Retry** | Yes (2) |
| **Exit Code** | 802 |

#### TOPO-H-PACK-003: Checksum computation failed

| Field | Value |
|-------|-------|
| **Condition** | SHA-256 computation fails |
| **Detection** | Hash operation error |
| **Severity** | High |
| **Mitigation** | Check file readability |
| **Retry** | Yes (1) |
| **Exit Code** | 803 |

#### TOPO-H-PACK-004: Pack structure validation failed

| Field | Value |
|-------|-------|
| **Condition** | Pack doesn't match expected structure |
| **Detection** | Structure validation |
| **Severity** | High |
| **Mitigation** | QUARANTINE — fix builder |
| **Retry** | No |
| **Exit Code** | 804 |

#### TOPO-H-ATTEST-001: Attestation generation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot create `attestation.json` |
| **Detection** | File creation error |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — check attester |
| **Retry** | Yes (1) |
| **Exit Code** | 811 |

#### TOPO-H-ATTEST-002: Attestation hash computation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot compute Merkle root |
| **Detection** | Hash computation error |
| **Severity** | Critical |
| **Mitigation** | Check file listing |
| **Retry** | Yes (1) |
| **Exit Code** | 812 |

#### TOPO-H-ATTEST-003: Attestation timestamp invalid

| Field | Value |
|-------|-------|
| **Condition** | Timestamp not ISO 8601 |
| **Detection** | Format validation |
| **Severity** | Medium |
| **Mitigation** | Fix timestamp format |
| **Retry** | Yes (1) |
| **Exit Code** | 813 |

---

### TOPO-I: Sealing Failures

#### TOPO-I-SEAL-001: Seal generation failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot create `seal.json` |
| **Detection** | File creation error |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — critical failure |
| **Retry** | No (seal is one-shot) |
| **Exit Code** | 901 |

#### TOPO-I-SEAL-002: Seal verification failed

| Field | Value |
|-------|-------|
| **Condition** | Seal hash doesn't verify |
| **Detection** | Hash verification |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — integrity failure |
| **Retry** | No |
| **Exit Code** | 902 |

#### TOPO-I-SEAL-003: Manifest finalization failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot update manifest with seal |
| **Detection** | File update error |
| **Severity** | Critical |
| **Mitigation** | QUARANTINE — check permissions |
| **Retry** | No |
| **Exit Code** | 903 |

---

### TOPO-Z: Infrastructure Failures

#### TOPO-Z-INFRA-001: Database connection failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot connect to PostgreSQL |
| **Detection** | Connection error |
| **Severity** | Critical |
| **Mitigation** | Check DATABASE_URL, network |
| **Retry** | Yes (3) with backoff |
| **Exit Code** | 991 |

#### TOPO-Z-INFRA-002: Redis connection failed

| Field | Value |
|-------|-------|
| **Condition** | Cannot connect to Redis |
| **Detection** | Connection error |
| **Severity** | High |
| **Mitigation** | Check REDIS_URL, network |
| **Retry** | Yes (3) with backoff |
| **Exit Code** | 992 |

#### TOPO-Z-INFRA-003: Disk space exhausted

| Field | Value |
|-------|-------|
| **Condition** | Disk usage > 95% |
| **Detection** | Disk check before operations |
| **Severity** | Critical |
| **Mitigation** | Free disk space |
| **Retry** | No |
| **Exit Code** | 993 |

#### TOPO-Z-INFRA-004: Memory exhausted

| Field | Value |
|-------|-------|
| **Condition** | Available memory < 100MB |
| **Detection** | Memory check |
| **Severity** | Critical |
| **Mitigation** | Free memory, restart services |
| **Retry** | No |
| **Exit Code** | 994 |

#### TOPO-Z-INFRA-005: Python import failed

| Field | Value |
|-------|-------|
| **Condition** | Required module not found |
| **Detection** | ImportError |
| **Severity** | Critical |
| **Mitigation** | Install dependencies |
| **Retry** | No |
| **Exit Code** | 995 |

#### TOPO-Z-INFRA-006: File permissions denied

| Field | Value |
|-------|-------|
| **Condition** | Cannot read/write required files |
| **Detection** | PermissionError |
| **Severity** | Critical |
| **Mitigation** | Fix file permissions |
| **Retry** | No |
| **Exit Code** | 996 |

#### TOPO-Z-INFRA-007: Network timeout

| Field | Value |
|-------|-------|
| **Condition** | Network operation times out |
| **Detection** | Timeout exception |
| **Severity** | High |
| **Mitigation** | Check network connectivity |
| **Retry** | Yes (3) with backoff |
| **Exit Code** | 997 |

#### TOPO-Z-INFRA-008: Process killed (SIGKILL)

| Field | Value |
|-------|-------|
| **Condition** | Process terminated by signal |
| **Detection** | Exit code 137 |
| **Severity** | Critical |
| **Mitigation** | Check OOM killer, resource limits |
| **Retry** | Maybe (investigate first) |
| **Exit Code** | 998 |

---

### Failure Mode Summary

| Region | Count | Critical | High | Medium | Low |
|--------|-------|----------|------|--------|-----|
| TOPO-A | 24 | 20 | 3 | 1 | 0 |
| TOPO-B | 16 | 8 | 5 | 3 | 0 |
| TOPO-C | 8 | 3 | 4 | 1 | 0 |
| TOPO-D | 4 | 2 | 1 | 0 | 1 |
| TOPO-E | 7 | 1 | 0 | 6 | 0 |
| TOPO-F | 5 | 2 | 1 | 0 | 2 |
| TOPO-G | 8 | 6 | 1 | 1 | 0 |
| TOPO-H | 7 | 4 | 2 | 1 | 0 |
| TOPO-I | 3 | 3 | 0 | 0 | 0 |
| TOPO-Z | 8 | 5 | 3 | 0 | 0 |
| **Total** | **90** | **54** | **20** | **13** | **3** |

---

## CI Workflow Template

### Multi-Slice Orchestration Workflow

```yaml
# .ci/u2_pipeline_orchestration.yaml
# PHASE II — NOT RUN IN PHASE I
# CI workflow for U2 uplift experiment orchestration

name: U2 Pipeline Orchestration

triggers:
  workflow_dispatch:
    inputs:
      slices:
        description: 'Slices to run (comma-separated or "all")'
        required: true
        default: 'all'
      cycles:
        description: 'Number of cycles per slice'
        required: true
        default: '500'
      seed:
        description: 'Base seed (hex)'
        required: true
        default: '0xDEADBEEF'
      dry_run:
        description: 'Dry run only (no execution)'
        required: false
        default: 'false'
  schedule:
    # Nightly at 2 AM UTC (if enabled)
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  MDAP_SEED: ${{ inputs.seed || '0xDEADBEEF' }}
  CYCLES: ${{ inputs.cycles || '500' }}
  OUTPUT_BASE: 'results/uplift_u2'

# ============================================================================
# STAGE 1: GATE CHECK
# ============================================================================
jobs:
  gate_check:
    name: "N01: Gate Check"
    runs-on: ubuntu-latest
    outputs:
      gate_passed: ${{ steps.gate.outputs.passed }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install pyyaml pydantic

      - name: Check governance gate
        id: gate
        run: |
          python experiments/ci/check_gate.py
          echo "passed=true" >> $GITHUB_OUTPUT

      - name: Gate failure
        if: failure()
        run: |
          echo "::error::Governance gate check failed. See logs for details."
          exit 101

# ============================================================================
# STAGE 2: VALIDATION
# ============================================================================
  validation:
    name: "N02-N04: Validation"
    needs: gate_check
    if: needs.gate_check.outputs.gate_passed == 'true'
    runs-on: ubuntu-latest
    outputs:
      validation_passed: ${{ steps.validate.outputs.passed }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install pyyaml pydantic pytest

      - name: Verify preregistration (N02)
        run: |
          python experiments/ci/verify_prereg.py

      - name: Load curriculum (N03)
        run: |
          python experiments/ci/validate_curriculum_schema.py

      - name: Dry run all slices (N04)
        run: |
          python experiments/run_uplift_u2.py --dry-run --all-slices

      - name: Validation complete
        id: validate
        run: |
          echo "passed=true" >> $GITHUB_OUTPUT

# ============================================================================
# STAGE 3: MANIFEST INITIALIZATION
# ============================================================================
  manifest_init:
    name: "N05: Manifest Init"
    needs: validation
    if: needs.validation.outputs.validation_passed == 'true'
    runs-on: ubuntu-latest
    outputs:
      manifest_id: ${{ steps.init.outputs.manifest_id }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Initialize manifest
        id: init
        run: |
          MANIFEST_ID=$(python experiments/ci/init_manifest.py \
            --output-dir ${{ env.OUTPUT_BASE }} \
            --seed ${{ env.MDAP_SEED }} \
            --cycles ${{ env.CYCLES }})
          echo "manifest_id=$MANIFEST_ID" >> $GITHUB_OUTPUT

      - name: Upload initial manifest
        uses: actions/upload-artifact@v4
        with:
          name: manifest-initial
          path: ${{ env.OUTPUT_BASE }}/experiment_manifest.json

# ============================================================================
# STAGE 4: SLICE EXECUTION (MATRIX)
# ============================================================================
  slice_execution:
    name: "N10-N13: Run ${{ matrix.slice }}"
    needs: manifest_init
    if: inputs.dry_run != 'true'
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false  # Allow other slices to complete
      matrix:
        slice:
          - slice_uplift_goal
          - slice_uplift_sparse
          - slice_uplift_tree
          - slice_uplift_dependency
    outputs:
      goal_status: ${{ steps.run.outputs.status }}
      sparse_status: ${{ steps.run.outputs.status }}
      tree_status: ${{ steps.run.outputs.status }}
      dependency_status: ${{ steps.run.outputs.status }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Download manifest
        uses: actions/download-artifact@v4
        with:
          name: manifest-initial
          path: ${{ env.OUTPUT_BASE }}

      - name: Run slice (${{ matrix.slice }})
        id: run
        timeout-minutes: 35
        run: |
          python experiments/run_uplift_u2.py \
            --slice ${{ matrix.slice }} \
            --cycles ${{ env.CYCLES }} \
            --seed ${{ env.MDAP_SEED }} \
            --output-dir ${{ env.OUTPUT_BASE }}/${{ matrix.slice }}
          echo "status=success" >> $GITHUB_OUTPUT

      - name: Handle failure
        if: failure()
        run: |
          echo "status=failed" >> $GITHUB_OUTPUT

      - name: Upload slice results
        uses: actions/upload-artifact@v4
        with:
          name: results-${{ matrix.slice }}
          path: ${{ env.OUTPUT_BASE }}/${{ matrix.slice }}

# ============================================================================
# STAGE 5: EVALUATION (MATRIX)
# ============================================================================
  slice_evaluation:
    name: "N20-N23: Eval ${{ matrix.slice }}"
    needs: slice_execution
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        slice:
          - slice_uplift_goal
          - slice_uplift_sparse
          - slice_uplift_tree
          - slice_uplift_dependency
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Download slice results
        uses: actions/download-artifact@v4
        with:
          name: results-${{ matrix.slice }}
          path: ${{ env.OUTPUT_BASE }}/${{ matrix.slice }}

      - name: Evaluate slice (${{ matrix.slice }})
        timeout-minutes: 10
        run: |
          python experiments/evaluate_slice.py \
            --slice ${{ matrix.slice }} \
            --input-dir ${{ env.OUTPUT_BASE }}/${{ matrix.slice }} \
            --output-dir ${{ env.OUTPUT_BASE }}/${{ matrix.slice }}

      - name: Upload evaluation results
        uses: actions/upload-artifact@v4
        with:
          name: eval-${{ matrix.slice }}
          path: ${{ env.OUTPUT_BASE }}/${{ matrix.slice }}/*_eval_result.json

# ============================================================================
# STAGE 6: SYNCHRONIZATION BARRIER
# ============================================================================
  sync_barrier:
    name: "N30: Sync Barrier"
    needs: slice_evaluation
    runs-on: ubuntu-latest
    outputs:
      all_complete: ${{ steps.sync.outputs.complete }}
    steps:
      - name: Verify all evaluations complete
        id: sync
        run: |
          echo "All slice evaluations completed"
          echo "complete=true" >> $GITHUB_OUTPUT

# ============================================================================
# STAGE 7: ANALYSIS
# ============================================================================
  analysis:
    name: "N40-N41: Analysis"
    needs: sync_barrier
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Download all evaluation results
        uses: actions/download-artifact@v4
        with:
          pattern: eval-*
          path: ${{ env.OUTPUT_BASE }}
          merge-multiple: true

      - name: Run chain analysis (N40)
        continue-on-error: true  # Degrade gracefully
        run: |
          python experiments/chain_analyzer.py \
            --input-dir ${{ env.OUTPUT_BASE }}/slice_uplift_tree \
            --output ${{ env.OUTPUT_BASE }}/chain_depth_stats.json

      - name: Generate diagnostics (N41)
        run: |
          python experiments/generate_diagnostics.py \
            --input-dir ${{ env.OUTPUT_BASE }} \
            --output ${{ env.OUTPUT_BASE }}/diagnostics_report.json

      - name: Upload analysis results
        uses: actions/upload-artifact@v4
        with:
          name: analysis
          path: |
            ${{ env.OUTPUT_BASE }}/chain_depth_stats.json
            ${{ env.OUTPUT_BASE }}/diagnostics_report.json

# ============================================================================
# STAGE 8: STATISTICAL SUMMARY
# ============================================================================
  statistical_summary:
    name: "N50: Statistical Summary"
    needs: analysis
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e . scipy statsmodels

      - name: Download all results
        uses: actions/download-artifact@v4
        with:
          pattern: '*'
          path: ${{ env.OUTPUT_BASE }}
          merge-multiple: true

      - name: Generate statistical summary
        run: |
          python experiments/statistical_summary_u2.py \
            --input-dir ${{ env.OUTPUT_BASE }} \
            --output ${{ env.OUTPUT_BASE }}/statistical_summary.json

      - name: Upload statistical summary
        uses: actions/upload-artifact@v4
        with:
          name: statistical-summary
          path: ${{ env.OUTPUT_BASE }}/statistical_summary.json

# ============================================================================
# STAGE 9: AUDIT
# ============================================================================
  audit:
    name: "N60: Audit"
    needs: statistical_summary
    runs-on: ubuntu-latest
    outputs:
      audit_passed: ${{ steps.audit.outputs.passed }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: ${{ env.OUTPUT_BASE }}
          merge-multiple: true

      - name: Run audit
        id: audit
        run: |
          python experiments/ci/audit_experiment.py \
            --experiment-dir ${{ env.OUTPUT_BASE }}
          echo "passed=true" >> $GITHUB_OUTPUT

      - name: Handle audit failure
        if: failure()
        run: |
          echo "::error::Audit failed. Experiment will be quarantined."
          python experiments/ci/quarantine.py \
            --experiment-dir ${{ env.OUTPUT_BASE }} \
            --reason "Audit failed"

# ============================================================================
# STAGE 10: EVIDENCE PACK BUILD
# ============================================================================
  pack_build:
    name: "N70-N80: Pack & Attest"
    needs: audit
    if: needs.audit.outputs.audit_passed == 'true'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: ${{ env.OUTPUT_BASE }}
          merge-multiple: true

      - name: Build Evidence Pack v2 (N70)
        run: |
          python experiments/evidence_pack_v2_builder.py \
            --input-dir ${{ env.OUTPUT_BASE }} \
            --output-dir ${{ env.OUTPUT_BASE }}/evidence_pack_v2

      - name: Generate attestation (N80)
        run: |
          python experiments/ci/generate_attestation.py \
            --pack-dir ${{ env.OUTPUT_BASE }}/evidence_pack_v2

      - name: Upload Evidence Pack v2
        uses: actions/upload-artifact@v4
        with:
          name: evidence-pack-v2
          path: ${{ env.OUTPUT_BASE }}/evidence_pack_v2

# ============================================================================
# STAGE 11: SEAL
# ============================================================================
  seal:
    name: "N90: Seal"
    needs: pack_build
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Download Evidence Pack v2
        uses: actions/download-artifact@v4
        with:
          name: evidence-pack-v2
          path: ${{ env.OUTPUT_BASE }}/evidence_pack_v2

      - name: Seal Evidence Pack (N90)
        run: |
          python experiments/ci/seal_pack.py \
            --pack-dir ${{ env.OUTPUT_BASE }}/evidence_pack_v2

      - name: Upload sealed pack
        uses: actions/upload-artifact@v4
        with:
          name: evidence-pack-v2-sealed
          path: ${{ env.OUTPUT_BASE }}/evidence_pack_v2

      - name: Summary
        run: |
          echo "## U2 Pipeline Complete" >> $GITHUB_STEP_SUMMARY
          echo "- **Cycles**: ${{ env.CYCLES }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Seed**: ${{ env.MDAP_SEED }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Status**: SEALED" >> $GITHUB_STEP_SUMMARY
```

### Slice-Specific Job Matrix Configuration

```yaml
# Matrix configuration for slice-specific settings
matrix_config:
  slice_uplift_goal:
    timeout_minutes: 20
    max_candidates: 40
    evaluator: goal_hit

  slice_uplift_sparse:
    timeout_minutes: 25
    max_candidates: 40
    evaluator: density

  slice_uplift_tree:
    timeout_minutes: 22
    max_candidates: 30
    evaluator: chain_length
    requires_chain_analysis: true

  slice_uplift_dependency:
    timeout_minutes: 27
    max_candidates: 40
    evaluator: multi_goal
```

---

## Synthetic Worked Example

### 4-Slice Parallel Execution Trace

This section provides a complete synthetic execution trace of the U2 pipeline with all four slices running in parallel.

#### Execution Parameters

```yaml
experiment_id: "u2_exp_20251206_001"
base_seed: 0xDEADBEEF
cycles: 500
slices:
  - slice_uplift_goal
  - slice_uplift_sparse
  - slice_uplift_tree
  - slice_uplift_dependency
start_time: "2025-12-06T02:00:00Z"
```

#### Timeline

```
TIME        NODE          STATUS    DETAILS
──────────────────────────────────────────────────────────────────────────────
T+00:00:00  N01:GATE      START     Checking VSD_PHASE_2_uplift_gate...
T+00:00:02  N01:GATE      PASS      All gates passed
T+00:00:02  N02:PREREG    START     Verifying PREREG_UPLIFT_U2.yaml...
T+00:00:03  N02:PREREG    PASS      Hash: 7a3f2b1c... (matches frozen)
T+00:00:03  N03:CURR      START     Loading curriculum_uplift_phase2.yaml...
T+00:00:04  N03:CURR      PASS      4 slices loaded
T+00:00:04  N04:DRY       START     Dry-run validation...
T+00:00:08  N04:DRY       PASS      All slices validated
T+00:00:08  N05:INIT      START     Initializing manifest...
T+00:00:09  N05:INIT      PASS      Manifest ID: u2_exp_20251206_001
T+00:00:09  ─────────────────────── PARALLEL EXECUTION BEGINS ───────────────
T+00:00:09  N10:GOAL      START     slice_uplift_goal baseline...
T+00:00:09  N11:SPARSE    START     slice_uplift_sparse baseline...
T+00:00:09  N12:TREE      START     slice_uplift_tree baseline...
T+00:00:09  N13:DEP       START     slice_uplift_dependency baseline...
│
│           ┌──────────────────────────────────────────────────────────────┐
│           │                    PARALLEL EXECUTION                        │
│           │                                                              │
│           │   N10:GOAL ────────────────────────────────────────►│        │
│           │   (baseline)     14:32                               │        │
│           │                  ────────────────────────────────────►│       │
│           │                  (rfl)        14:28                   │       │
│           │                                                       │       │
│           │   N11:SPARSE ──────────────────────────────────────►│        │
│           │   (baseline)     18:45                               │        │
│           │                  ────────────────────────────────────►│       │
│           │                  (rfl)        19:12                   │       │
│           │                                                       │       │
│           │   N12:TREE ────────────────────────────────────────►│        │
│           │   (baseline)     16:55                               │        │
│           │                  ────────────────────────────────────►│       │
│           │                  (rfl)        17:03                   │       │
│           │                                                       │       │
│           │   N13:DEP ─────────────────────────────────────────►│        │
│           │   (baseline)     20:14                               │        │
│           │                  ────────────────────────────────────►│       │
│           │                  (rfl)        21:33                   │       │
│           └──────────────────────────────────────────────────────────────┘
│
T+00:29:00  N10:GOAL      DONE      Baseline: 500 cycles, RFL: 500 cycles
T+00:34:00  N12:TREE      DONE      Baseline: 500 cycles, RFL: 500 cycles
T+00:37:57  N11:SPARSE    DONE      Baseline: 500 cycles, RFL: 500 cycles
T+00:41:47  N13:DEP       DONE      Baseline: 500 cycles, RFL: 500 cycles
T+00:41:47  ─────────────────────── PARALLEL EXECUTION ENDS ─────────────────
T+00:41:47  N20:EVAL_G    START     Evaluating slice_uplift_goal...
T+00:41:47  N21:EVAL_S    START     Evaluating slice_uplift_sparse...
T+00:41:47  N22:EVAL_T    START     Evaluating slice_uplift_tree...
T+00:41:47  N23:EVAL_D    START     Evaluating slice_uplift_dependency...
T+00:42:12  N20:EVAL_G    DONE      Success rate: baseline=0.23, rfl=0.41
T+00:42:18  N22:EVAL_T    DONE      Success rate: baseline=0.18, rfl=0.35
T+00:42:25  N21:EVAL_S    DONE      Success rate: baseline=0.31, rfl=0.52
T+00:42:35  N23:EVAL_D    DONE      Success rate: baseline=0.08, rfl=0.24
T+00:42:35  N30:SYNC      START     Waiting for all evaluators...
T+00:42:35  N30:SYNC      DONE      All 4 evaluators complete
T+00:42:35  N40:CHAIN     START     Analyzing chain depths...
T+00:43:42  N40:CHAIN     DONE      Max depth: 3, Mean: 1.7
T+00:43:42  N41:DIAG      START     Generating diagnostics...
T+00:44:15  N41:DIAG      DONE      Diagnostics report generated
T+00:44:15  N50:STAT      START     Computing statistical summary...
T+00:45:30  N50:STAT      DONE      All metrics computed with CIs
T+00:45:30  N60:AUDIT     START     Running audit checks...
T+00:46:45  N60:AUDIT     DONE      All 8 checks passed
T+00:46:45  N70:PACK      START     Building Evidence Pack v2...
T+00:47:30  N70:PACK      DONE      Pack built, 47 files
T+00:47:30  N80:ATTEST    START     Generating attestation...
T+00:47:45  N80:ATTEST    DONE      Attestation complete, Merkle root: 8f2a...
T+00:47:45  N90:SEAL      START     Sealing Evidence Pack...
T+00:47:50  N90:SEAL      DONE      Pack sealed successfully
T+00:47:50  ─────────────────────── PIPELINE COMPLETE ────────────────────────
```

#### Results Summary

```json
{
  "experiment_id": "u2_exp_20251206_001",
  "status": "COMPLETE",
  "duration_minutes": 47.83,
  "slices": {
    "slice_uplift_goal": {
      "baseline_success_rate": 0.232,
      "rfl_success_rate": 0.414,
      "delta_p": 0.182,
      "delta_p_ci": [0.121, 0.243],
      "p_value": 0.0001,
      "cohens_h": 0.39,
      "outcome": "POSITIVE_RESULT"
    },
    "slice_uplift_sparse": {
      "baseline_success_rate": 0.312,
      "rfl_success_rate": 0.524,
      "delta_p": 0.212,
      "delta_p_ci": [0.156, 0.268],
      "p_value": 0.00001,
      "cohens_h": 0.43,
      "outcome": "POSITIVE_RESULT"
    },
    "slice_uplift_tree": {
      "baseline_success_rate": 0.178,
      "rfl_success_rate": 0.352,
      "delta_p": 0.174,
      "delta_p_ci": [0.108, 0.240],
      "p_value": 0.0002,
      "cohens_h": 0.40,
      "outcome": "POSITIVE_RESULT"
    },
    "slice_uplift_dependency": {
      "baseline_success_rate": 0.082,
      "rfl_success_rate": 0.238,
      "delta_p": 0.156,
      "delta_p_ci": [0.098, 0.214],
      "p_value": 0.0003,
      "cohens_h": 0.45,
      "outcome": "POSITIVE_RESULT"
    }
  },
  "aggregate": {
    "mean_delta_p": 0.181,
    "all_slices_positive": true,
    "min_effect_size": 0.39
  },
  "evidence_pack": {
    "version": "v2",
    "file_count": 47,
    "total_size_mb": 12.4,
    "merkle_root": "8f2a3b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    "sealed_at": "2025-12-06T02:47:50Z"
  }
}
```

#### Parallel Execution Gantt Chart

```
         0    5   10   15   20   25   30   35   40   45   50 (minutes)
         │    │    │    │    │    │    │    │    │    │    │
GATE     ██
PREREG   █
CURR     █
DRY      ████
INIT     █
         │
GOAL     │████████████████████████████░░░░░░░░░░░░░░░░░░░░│
         │     baseline      │      rfl        │
SPARSE   │██████████████████████████████████████░░░░░░░░░░│
         │        baseline        │       rfl         │
TREE     │██████████████████████████████████░░░░░░░░░░░░░░│
         │       baseline       │      rfl        │
DEP      │██████████████████████████████████████████░░░░░░│
         │          baseline          │       rfl          │
         │                                                 │
EVAL_G   │                                        █
EVAL_S   │                                        █
EVAL_T   │                                        █
EVAL_D   │                                        █
SYNC     │                                        █
CHAIN    │                                         █
DIAG     │                                          █
STAT     │                                          ██
AUDIT    │                                           ██
PACK     │                                            █
ATTEST   │                                             █
SEAL     │                                             █
         └─────────────────────────────────────────────────
                                                    47:50
```

#### Log Excerpts

**N10: slice_uplift_goal runner log (excerpt):**

```jsonl
{"cycle": 0, "mode": "baseline", "seed": 3735928559, "candidates_tried": 38, "verified": 7, "goal_hits": 0, "success": false, "timing_ms": 1823}
{"cycle": 1, "mode": "baseline", "seed": 3735928560, "candidates_tried": 40, "verified": 9, "goal_hits": 1, "success": true, "timing_ms": 1756}
{"cycle": 2, "mode": "baseline", "seed": 3735928561, "candidates_tried": 39, "verified": 6, "goal_hits": 0, "success": false, "timing_ms": 1891}
...
{"cycle": 499, "mode": "baseline", "seed": 3735929058, "candidates_tried": 40, "verified": 8, "goal_hits": 0, "success": false, "timing_ms": 1802}
{"cycle": 0, "mode": "rfl", "seed": 3735928559, "candidates_tried": 40, "verified": 12, "goal_hits": 1, "success": true, "timing_ms": 1934, "policy_scores": [0.92, 0.87, 0.84, ...]}
{"cycle": 1, "mode": "rfl", "seed": 3735928560, "candidates_tried": 38, "verified": 11, "goal_hits": 2, "success": true, "timing_ms": 1867, "policy_scores": [0.91, 0.89, 0.86, ...]}
...
```

**N60: Audit report (excerpt):**

```json
{
  "audit_id": "audit_20251206_024645",
  "experiment_id": "u2_exp_20251206_001",
  "checks": [
    {"name": "prereg_hash_unchanged", "status": "PASS", "details": "Hash matches frozen value"},
    {"name": "manifest_schema_valid", "status": "PASS", "details": "Schema validation passed"},
    {"name": "log_checksums_match", "status": "PASS", "details": "All 8 log files verified"},
    {"name": "no_phase1_contamination", "status": "PASS", "details": "No Phase I references found"},
    {"name": "determinism_verified", "status": "PASS", "details": "10-cycle replay matched"},
    {"name": "statistical_summary_complete", "status": "PASS", "details": "All required fields present"},
    {"name": "timestamps_valid", "status": "PASS", "details": "All timestamps monotonic"},
    {"name": "artifacts_complete", "status": "PASS", "details": "All 47 expected files present"}
  ],
  "overall_status": "PASS",
  "completed_at": "2025-12-06T02:46:45Z"
}
```

---

## Appendix: Quick Reference

### Exit Code Reference

| Range | Category | Description |
|-------|----------|-------------|
| 0 | Success | Pipeline completed successfully |
| 1-99 | Reserved | Standard Unix exit codes |
| 100-149 | TOPO-A | Gate/Validation failures |
| 200-249 | TOPO-B | Slice execution failures |
| 300-349 | TOPO-C | Evaluation failures |
| 400-449 | TOPO-D | Synchronization failures |
| 500-549 | TOPO-E | Analysis failures |
| 600-649 | TOPO-F | Statistics failures |
| 700-749 | TOPO-G | Audit failures |
| 800-849 | TOPO-H | Packaging failures |
| 900-949 | TOPO-I | Sealing failures |
| 990-999 | TOPO-Z | Infrastructure failures |

### Node Quick Reference

```
N01: GATE_CHECK      - Governance gate validation
N02: PREREG_VERIFY   - Preregistration integrity
N03: CURRICULUM_LOAD - Load slice configurations
N04: DRY_RUN         - Configuration validation
N05: MANIFEST_INIT   - Initialize experiment
N10: SLICE_GOAL      - Run goal slice
N11: SLICE_SPARSE    - Run sparse slice
N12: SLICE_TREE      - Run tree slice
N13: SLICE_DEP       - Run dependency slice
N20: EVAL_GOAL       - Evaluate goal slice
N21: EVAL_SPARSE     - Evaluate sparse slice
N22: EVAL_TREE       - Evaluate tree slice
N23: EVAL_DEP        - Evaluate dependency slice
N30: SYNC_BARRIER    - Wait for all evaluators
N40: CHAIN_ANALYZE   - Chain depth analysis
N41: DIAGNOSTICS     - Generate diagnostics
N50: STAT_SUMMARY    - Statistical summary
N60: AUDIT           - Audit experiment
N70: PACK_BUILD      - Build Evidence Pack
N80: ATTESTATION     - Generate attestation
N90: SEAL            - Seal Evidence Pack
N98: QUARANTINE      - Quarantine failed experiment
N99: ABORT_GATE      - Abort pipeline
```

---

## Section 10: Degraded Pipeline Modes

**PHASE II — NOT RUN IN PHASE I**
**No uplift claims are made.**

This section defines three named pipeline degradation modes that govern how the U2 pipeline responds to partial failures while maintaining scientific integrity.

### 10.1 Mode Definitions

#### FULL_PIPELINE Mode

**Description**: Complete pipeline execution with all nodes operational.

```
Mode: FULL_PIPELINE
Governance Label: OK
Artifact Admissibility: ALL artifacts admissible for analysis
Δp Computation: PERMITTED — full statistical analysis valid
Evidence Pack Status: COMPLETE
```

| Property | Value |
|----------|-------|
| All validation nodes | Must pass |
| All slice runners | Must complete |
| All evaluators | Must compute metrics |
| Sync barrier | Must synchronize |
| All analyzers | Must generate reports |
| Audit stage | Must pass |
| Attestation | Full cryptographic seal |

**Entry Criteria**: All N01-N05 validation stages pass.
**Exit Criteria**: N90:SEAL completes with exit code 0.

#### DEGRADED_ANALYSIS Mode

**Description**: One or more slices failed, but remaining slices provide valid data for partial analysis.

```
Mode: DEGRADED_ANALYSIS
Governance Label: WARN
Artifact Admissibility: Per-slice artifacts for successful slices ONLY
Δp Computation: RESTRICTED — only for slices with complete baseline+RFL pairs
Evidence Pack Status: PARTIAL
```

| Node Type | Failure Tolerance |
|-----------|-------------------|
| Gatekeeper (N01-N02) | **HARD-FAIL** — no tolerance |
| Loader (N03) | **HARD-FAIL** — no tolerance |
| Validator (N04-N05) | **HARD-FAIL** — no tolerance |
| Slice Runners (N10-N13) | Up to 2 slices may fail |
| Evaluators (N20-N23) | Must pass for all running slices |
| Sync Barrier (N30) | Operates on available slices |
| Analyzers (N40-N41) | Must pass for available data |
| Summarizer (N50) | Generates partial summary |
| Auditor (N60) | Must pass with degraded scope |
| Packager (N70) | Produces partial Evidence Pack |
| Attester (N80) | Marks pack as DEGRADED |
| Sealer (N90) | Seals with degradation flag |

**Critical Constraint**: A slice pair (baseline + RFL) must BOTH complete for that slice's Δp to be computed. Mixed completion is FORBIDDEN.

**Entry Criteria**: N01-N05 pass; at least 2 slice pairs complete successfully.
**Exit Criteria**: N90:SEAL completes with governance label WARN.

**Allowed Failure Combinations**:
```
DEGRADED_ANALYSIS requires ≥2 successful slice pairs:

Valid combinations (4 choose 2 = 6):
- goal + sparse (tree failed, dep failed)
- goal + tree (sparse failed, dep failed)
- goal + dep (sparse failed, tree failed)
- sparse + tree (goal failed, dep failed)
- sparse + dep (goal failed, tree failed)
- tree + dep (goal failed, sparse failed)

Invalid combinations (single slice):
- goal only
- sparse only
- tree only
- dep only
```

#### EVIDENCE_ONLY Mode

**Description**: Critical failures prevent statistical analysis, but raw evidence is preserved for forensic review.

```
Mode: EVIDENCE_ONLY
Governance Label: DO NOT USE
Artifact Admissibility: Raw logs and traces ONLY — no computed metrics
Δp Computation: FORBIDDEN — no statistical claims permitted
Evidence Pack Status: FORENSIC
```

| Node Type | Status |
|-----------|--------|
| Gatekeeper (N01-N02) | May have failed |
| Loader (N03) | May have failed |
| Validator (N04-N05) | May have failed |
| Slice Runners (N10-N13) | Fewer than 2 pairs complete |
| Evaluators (N20-N23) | May have failed |
| Sync Barrier (N30) | Bypassed |
| Analyzers (N40-N41) | Skipped |
| Summarizer (N50) | Generates failure summary only |
| Auditor (N60) | Forensic audit mode |
| Packager (N70) | Preserves raw evidence |
| Attester (N80) | Marks pack as EVIDENCE_ONLY |
| Sealer (N90) | Quarantine seal |

**Entry Criteria**: Any of:
- Validation nodes (N01-N05) fail
- Fewer than 2 slice pairs complete
- Critical infrastructure failure (TOPO-Z)
- Corruption detected in any artifact

**Exit Criteria**: Evidence preserved in quarantine with forensic seal.

### 10.2 Mode Transition Diagram

```
                                    ┌─────────────────────────────────────────┐
                                    │              FULL_PIPELINE              │
                                    │           Governance: OK                │
                                    │         All artifacts valid             │
                                    └─────────────────────────────────────────┘
                                                       │
                                                       │ 1-2 slice pairs fail
                                                       │ (validation still passed)
                                                       ▼
                                    ┌─────────────────────────────────────────┐
                                    │           DEGRADED_ANALYSIS             │
                                    │          Governance: WARN               │
                                    │     Per-slice artifacts only            │
                                    └─────────────────────────────────────────┘
                                                       │
                                                       │ 3+ slice pairs fail OR
                                                       │ validation failure OR
                                                       │ corruption detected
                                                       ▼
                                    ┌─────────────────────────────────────────┐
                                    │            EVIDENCE_ONLY                │
                                    │        Governance: DO NOT USE           │
                                    │       Raw evidence preserved            │
                                    └─────────────────────────────────────────┘
```

**Irreversibility**: Mode transitions are ONE-WAY. A pipeline cannot recover from DEGRADED_ANALYSIS to FULL_PIPELINE or from EVIDENCE_ONLY to any other mode.

### 10.3 Mode Detection Logic

```python
def determine_pipeline_mode(validation_status, slice_results, integrity_checks):
    """
    Determine the current pipeline degradation mode.

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """
    # Rule 1: Validation failure → EVIDENCE_ONLY
    if not validation_status.all_passed:
        return PipelineMode.EVIDENCE_ONLY

    # Rule 2: Corruption detected → EVIDENCE_ONLY
    if any(not check.passed for check in integrity_checks):
        return PipelineMode.EVIDENCE_ONLY

    # Count successful slice pairs (both baseline AND RFL must succeed)
    successful_pairs = sum(
        1 for s in slice_results
        if s.baseline_completed and s.rfl_completed
    )

    # Rule 3: Fewer than 2 pairs → EVIDENCE_ONLY
    if successful_pairs < 2:
        return PipelineMode.EVIDENCE_ONLY

    # Rule 4: All 4 pairs successful → FULL_PIPELINE
    if successful_pairs == 4:
        return PipelineMode.FULL_PIPELINE

    # Rule 5: 2-3 pairs successful → DEGRADED_ANALYSIS
    return PipelineMode.DEGRADED_ANALYSIS
```

### 10.4 Artifact Admissibility Matrix

| Artifact Type | FULL_PIPELINE | DEGRADED_ANALYSIS | EVIDENCE_ONLY |
|---------------|---------------|-------------------|---------------|
| Raw execution logs | ✓ Admissible | ✓ Admissible | ✓ Admissible |
| Slice metrics JSON | ✓ Admissible | ✓ Per-slice only | ✗ Not admissible |
| Δp computations | ✓ Admissible | ✓ Successful pairs | ✗ FORBIDDEN |
| Statistical summary | ✓ Admissible | ⚠ Partial | ✗ Not admissible |
| Cohen's h values | ✓ Admissible | ✓ Successful pairs | ✗ Not admissible |
| Wilson CIs | ✓ Admissible | ✓ Successful pairs | ✗ Not admissible |
| Chain analysis | ✓ Admissible | ⚠ Available slices | ✗ Not admissible |
| Dependency graphs | ✓ Admissible | ⚠ Available slices | ✗ Not admissible |
| Attestation document | ✓ Full seal | ⚠ Degraded seal | ✗ Forensic seal |
| Evidence Pack | ✓ Complete | ⚠ Partial | ✗ Quarantined |

**Legend**: ✓ = Fully admissible | ⚠ = Conditionally admissible | ✗ = Not admissible/Forbidden

---

## Section 11: Topology Health Matrix

**PHASE II — NOT RUN IN PHASE I**
**No uplift claims are made.**

This section defines health signals, failure patterns, and recovery actions for each node type in the pipeline topology.

### 11.1 Health Signal Definitions

Each node type exposes the following health signals:

| Signal | Type | Description |
|--------|------|-------------|
| `heartbeat` | Boolean | Node is responsive |
| `progress` | Float [0.0-1.0] | Completion percentage |
| `memory_ok` | Boolean | Memory within limits |
| `disk_ok` | Boolean | Disk space within limits |
| `latency_ok` | Boolean | Processing time within bounds |
| `integrity_ok` | Boolean | No corruption detected |
| `dependencies_ok` | Boolean | Upstream dependencies healthy |

### 11.2 Node Type Health Matrix

#### Gatekeeper Nodes (N01, N02)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Response time | < 5s | 5-15s | > 15s |
| Memory usage | < 256MB | 256-512MB | > 512MB |
| Gate file access | Immediate | 1 retry | 2+ retries |
| Hash verification | < 1s | 1-3s | > 3s |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| GK-001 | Gate file missing | File not deployed | Deploy gate file |
| GK-002 | Hash mismatch | File corrupted/modified | Restore from VCS |
| GK-003 | Permission denied | ACL misconfiguration | Fix file permissions |
| GK-004 | Timeout on read | Disk I/O issue | Check storage health |
| GK-005 | Invalid JSON/YAML | Syntax error in gate file | Fix syntax, re-deploy |

**Hard-Fail Conditions**: ALL failures in Gatekeeper nodes trigger EVIDENCE_ONLY mode.

#### Validator Nodes (N04, N05)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Validation time | < 30s | 30-60s | > 60s |
| Schema errors | 0 | 1-2 (warnings) | 3+ |
| Config parsing | < 1s | 1-5s | > 5s |
| Dry-run cycles | All pass | 1 warning | Any failure |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| VD-001 | Schema validation fail | Config schema mismatch | Update config to schema |
| VD-002 | Missing required field | Incomplete configuration | Add required fields |
| VD-003 | Invalid seed value | Seed out of range | Use valid MDAP_SEED |
| VD-004 | Cycle count mismatch | n_cycles inconsistent | Align cycle counts |
| VD-005 | Dry-run assertion fail | Logic error in config | Debug configuration |

**Hard-Fail Conditions**: ALL validation failures trigger EVIDENCE_ONLY mode.

#### Loader Nodes (N03)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Load time | < 10s | 10-30s | > 30s |
| Memory for config | < 128MB | 128-256MB | > 256MB |
| Slice count | 4 | 3 (degraded) | < 3 |
| YAML parse time | < 500ms | 500ms-2s | > 2s |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| LD-001 | YAML parse error | Invalid YAML syntax | Fix YAML file |
| LD-002 | Missing slice config | Slice not defined | Add slice definition |
| LD-003 | Circular reference | Config references loop | Break circular refs |
| LD-004 | Memory exhaustion | Config too large | Optimize config size |
| LD-005 | File not found | Path incorrect | Fix config path |

**Hard-Fail Conditions**: ALL loader failures trigger EVIDENCE_ONLY mode.

#### Runner Nodes (N10-N13)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Cycle completion | 100% | 90-99% | < 90% |
| Memory per cycle | < 2GB | 2-4GB | > 4GB |
| Cycle time | < 5min | 5-10min | > 10min |
| Proof generation | > 0 | 0 (warn) | Error state |
| Verifier calls | All succeed | 1 retry | 2+ retries |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| RN-001 | Zero proofs generated | Empty formula pool | Expand formula pool |
| RN-002 | Cycle timeout | Verifier hanging | Increase timeout/retry |
| RN-003 | Memory exhaustion | Large derivation tree | Reduce breadth |
| RN-004 | Verifier crash | Lean process failure | Restart verifier |
| RN-005 | Seed mismatch | Non-deterministic init | Fix seed propagation |
| RN-006 | Partial cycle | Interrupted execution | Invalidate + retry |
| RN-007 | Baseline/RFL divergence | Different cycle counts | Align counts |

**Soft-Fail Conditions**: Runner failures trigger DEGRADED_ANALYSIS if ≥2 pairs complete.

**Critical Invariant**:
```
INVARIANT: For any slice S, if baseline(S) fails, then RFL(S) MUST NOT contribute to Δp.
INVARIANT: For any slice S, if RFL(S) fails, then baseline(S) MUST NOT contribute to Δp.
```

#### Evaluator Nodes (N20-N23)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Evaluation time | < 2min | 2-5min | > 5min |
| Metric completeness | 100% | 90-99% | < 90% |
| JSON output size | < 10MB | 10-50MB | > 50MB |
| Computation errors | 0 | 1-2 | 3+ |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| EV-001 | Missing metric field | Evaluator bug | Fix evaluator code |
| EV-002 | NaN in computation | Division by zero | Add zero-check |
| EV-003 | Metric out of range | Computation error | Validate bounds |
| EV-004 | JSON serialization fail | Complex object | Simplify output |
| EV-005 | Log file unreadable | Encoding issue | Fix log encoding |

**Soft-Fail Conditions**: Evaluator failures for a slice invalidate that slice pair only.

#### Sync Barrier Node (N30)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Wait time | < 1min | 1-5min | > 5min |
| Slice count | 4 | 2-3 | < 2 |
| Memory during sync | < 512MB | 512MB-1GB | > 1GB |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| SY-001 | Indefinite wait | Upstream deadlock | Timeout and degrade |
| SY-002 | Incomplete slice set | Missing evaluators | Check evaluator status |
| SY-003 | Memory spike | Large result sets | Stream results |

**Mode Transition**: If < 2 slices available at sync, transition to EVIDENCE_ONLY.

#### Analyzer Nodes (N40-N41)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Analysis time | < 5min | 5-10min | > 10min |
| Chain computation | Complete | Partial | Failed |
| Graph generation | Success | Degraded | Failed |
| Memory usage | < 4GB | 4-8GB | > 8GB |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| AN-001 | Chain depth overflow | Unbounded recursion | Add depth limit |
| AN-002 | Graph cycle detected | Corrupted proof DAG | Flag and skip |
| AN-003 | Memory exhaustion | Large DAG | Batch processing |
| AN-004 | Missing dependencies | Incomplete proof data | Report missing data |

**Soft-Fail Conditions**: Analyzer failures produce partial reports in DEGRADED_ANALYSIS.

#### Summarizer Node (N50)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Summary time | < 2min | 2-5min | > 5min |
| Statistical checks | All pass | 1 warning | Any failure |
| Output completeness | 100% | 80-99% | < 80% |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| SM-001 | CI computation fail | Invalid input data | Validate inputs |
| SM-002 | Cohen's h undefined | Zero variance | Report as N/A |
| SM-003 | Z-test failure | Sample size issue | Report limitations |
| SM-004 | Summary incomplete | Missing slice data | Note missing slices |

**Critical Invariant**:
```
INVARIANT: No statistical summary may include Δp values computed from incomplete slice pairs.
INVARIANT: If slice S failed, summary MUST NOT include Δp(S) or any derived statistics.
```

#### Auditor Node (N60)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Audit time | < 5min | 5-10min | > 10min |
| Checks passed | 100% | 95-99% | < 95% |
| Violations found | 0 | 1-2 (minor) | 3+ or major |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| AU-001 | Checksum mismatch | File corruption | Flag corrupt files |
| AU-002 | Missing audit trail | Incomplete logging | Add audit events |
| AU-003 | Timestamp anomaly | Clock skew | Flag and report |
| AU-004 | Duplicate entries | Retry artifacts | Deduplicate |
| AU-005 | Phase I reference | Contamination | EVIDENCE_ONLY mode |

**Hard-Fail on Corruption**: Any integrity failure triggers EVIDENCE_ONLY.

#### Packager Node (N70)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Pack time | < 5min | 5-10min | > 10min |
| Pack size | < 100MB | 100-500MB | > 500MB |
| File count | Complete | 90-99% | < 90% |
| Compression ratio | > 50% | 30-50% | < 30% |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| PK-001 | Missing artifacts | Upstream failure | Include available only |
| PK-002 | Compression failure | Corrupt data | Skip compression |
| PK-003 | Disk space exhausted | Insufficient storage | Clean temp files |
| PK-004 | Manifest incomplete | Missing metadata | Generate partial manifest |

**Soft-Fail Conditions**: Packager produces partial pack in DEGRADED_ANALYSIS.

#### Attester Node (N80)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Attestation time | < 1min | 1-3min | > 3min |
| Signature valid | Yes | N/A | No |
| Hash computation | < 10s | 10-30s | > 30s |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| AT-001 | Signature failure | Key issue | Use backup key |
| AT-002 | Hash mismatch | Concurrent modification | Lock files |
| AT-003 | Timestamp service down | External dependency | Use local timestamp |

**Attestation Modes by Pipeline Mode**:

| Pipeline Mode | Attestation Type | Governance Label |
|---------------|------------------|------------------|
| FULL_PIPELINE | Full cryptographic seal | OK |
| DEGRADED_ANALYSIS | Degraded seal with warnings | WARN |
| EVIDENCE_ONLY | Forensic seal (no claims) | DO NOT USE |

#### Sealer Node (N90)

| Health Metric | OK Threshold | WARN Threshold | FAIL Threshold |
|---------------|--------------|----------------|----------------|
| Seal time | < 1min | 1-3min | > 3min |
| Final validation | Pass | N/A | Fail |
| Archive creation | Success | Retry once | Failed |

**Observable Failure Patterns**:

| Pattern ID | Symptoms | Root Cause | Recovery Action |
|------------|----------|------------|-----------------|
| SL-001 | Archive creation fail | Disk full | Clean and retry |
| SL-002 | Final hash mismatch | Late modification | Recompute from source |
| SL-003 | Metadata incomplete | Upstream failure | Seal with available |

### 11.3 Health Aggregation Rules

```python
def aggregate_pipeline_health(node_health_map):
    """
    Aggregate health signals from all nodes into pipeline health.

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    """
    # Critical nodes that trigger immediate EVIDENCE_ONLY
    critical_nodes = {'N01', 'N02', 'N03', 'N04', 'N05'}

    # Check critical nodes first
    for node_id in critical_nodes:
        if node_health_map[node_id].status == 'FAIL':
            return PipelineHealth(
                mode=PipelineMode.EVIDENCE_ONLY,
                governance_label='DO_NOT_USE',
                failed_nodes=[node_id]
            )

    # Count failed slice pairs
    slice_pairs = [
        ('N10', 'N20'),  # goal
        ('N11', 'N21'),  # sparse
        ('N12', 'N22'),  # tree
        ('N13', 'N23'),  # dep
    ]

    failed_pairs = []
    for runner, evaluator in slice_pairs:
        runner_health = node_health_map.get(runner)
        evaluator_health = node_health_map.get(evaluator)

        # A pair fails if EITHER baseline OR RFL fails
        if (runner_health and runner_health.status == 'FAIL') or \
           (evaluator_health and evaluator_health.status == 'FAIL'):
            failed_pairs.append((runner, evaluator))

    # Determine mode based on failed pairs
    if len(failed_pairs) >= 3:
        return PipelineHealth(
            mode=PipelineMode.EVIDENCE_ONLY,
            governance_label='DO_NOT_USE',
            failed_nodes=[n for pair in failed_pairs for n in pair]
        )
    elif len(failed_pairs) >= 1:
        return PipelineHealth(
            mode=PipelineMode.DEGRADED_ANALYSIS,
            governance_label='WARN',
            failed_nodes=[n for pair in failed_pairs for n in pair]
        )
    else:
        return PipelineHealth(
            mode=PipelineMode.FULL_PIPELINE,
            governance_label='OK',
            failed_nodes=[]
        )
```

---

## Section 12: CI Degradation Policy

**PHASE II — NOT RUN IN PHASE I**
**No uplift claims are made.**

This section defines hard-fail versus soft-fail semantics for CI pipeline stages.

### 12.1 The Cardinal Rule

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  CARDINAL RULE: No node may fail in such a way that Δp is computed from    │
│  partial or corrupted data.                                                 │
│                                                                             │
│  This rule is ABSOLUTE and takes precedence over all soft-fail policies.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Interpretation**:
1. If a baseline run fails, the corresponding RFL run's Δp MUST be excluded
2. If an RFL run fails, the corresponding baseline run's Δp MUST be excluded
3. If corruption is detected in ANY artifact, NO Δp values may be computed from that artifact
4. A "partial" Δp is NEVER valid — either a complete pair exists, or no Δp is reported

### 12.2 Hard-Fail Stages

Hard-fail stages MUST pass for the pipeline to continue. Failure results in immediate transition to EVIDENCE_ONLY mode.

| Stage | CI Job | Exit Codes | Consequence of Failure |
|-------|--------|------------|------------------------|
| Governance Gate | `gate-check` | 100-109 | Pipeline halts, EVIDENCE_ONLY |
| Preregistration Verify | `prereg-verify` | 110-119 | Pipeline halts, EVIDENCE_ONLY |
| Curriculum Load | `curriculum-load` | 120-129 | Pipeline halts, EVIDENCE_ONLY |
| Dry Run | `dry-run` | 130-139 | Pipeline halts, EVIDENCE_ONLY |
| Manifest Init | `manifest-init` | 140-149 | Pipeline halts, EVIDENCE_ONLY |
| Integrity Check | `integrity-check` | 700-749 | Pipeline halts, EVIDENCE_ONLY |
| Phase I Contamination | `contamination-check` | 750 | Pipeline halts, EVIDENCE_ONLY |

**CI Configuration for Hard-Fail**:

```yaml
# Hard-fail stages - MUST pass
hard_fail_stages:
  gate-check:
    continue-on-error: false
    fail-fast: true
    exit_code_range: [100, 149]
    on_failure: evidence_only_mode

  prereg-verify:
    continue-on-error: false
    fail-fast: true
    on_failure: evidence_only_mode

  curriculum-load:
    continue-on-error: false
    fail-fast: true
    on_failure: evidence_only_mode

  dry-run:
    continue-on-error: false
    fail-fast: true
    on_failure: evidence_only_mode

  manifest-init:
    continue-on-error: false
    fail-fast: true
    on_failure: evidence_only_mode

  integrity-check:
    continue-on-error: false
    fail-fast: true
    on_failure: evidence_only_mode
```

### 12.3 Soft-Fail Stages

Soft-fail stages may fail without halting the entire pipeline. The pipeline degrades gracefully based on failure count.

| Stage | CI Job | Exit Codes | Degradation Behavior |
|-------|--------|------------|----------------------|
| Slice Runner (goal) | `run-slice-goal` | 200-209 | Exclude goal slice pair |
| Slice Runner (sparse) | `run-slice-sparse` | 210-219 | Exclude sparse slice pair |
| Slice Runner (tree) | `run-slice-tree` | 220-229 | Exclude tree slice pair |
| Slice Runner (dep) | `run-slice-dep` | 230-239 | Exclude dep slice pair |
| Evaluator (goal) | `eval-goal` | 300-309 | Exclude goal slice pair |
| Evaluator (sparse) | `eval-sparse` | 310-319 | Exclude sparse slice pair |
| Evaluator (tree) | `eval-tree` | 320-329 | Exclude tree slice pair |
| Evaluator (dep) | `eval-dep` | 330-339 | Exclude dep slice pair |
| Chain Analyzer | `chain-analyze` | 500-509 | Partial chain report |
| Diagnostician | `diagnostics` | 510-519 | Partial diagnostics |
| Summarizer | `stat-summary` | 520-529 | Partial summary (valid slices only) |

**CI Configuration for Soft-Fail**:

```yaml
# Soft-fail stages - may fail with degradation
soft_fail_stages:
  slice-runners:
    continue-on-error: true
    fail-fast: false
    jobs:
      run-slice-goal:
        exit_code_range: [200, 209]
        on_failure: mark_slice_failed
      run-slice-sparse:
        exit_code_range: [210, 219]
        on_failure: mark_slice_failed
      run-slice-tree:
        exit_code_range: [220, 229]
        on_failure: mark_slice_failed
      run-slice-dep:
        exit_code_range: [230, 239]
        on_failure: mark_slice_failed

    # Degradation threshold
    min_successful: 2  # At least 2 slice pairs must succeed
    on_threshold_breach: evidence_only_mode

  evaluators:
    continue-on-error: true
    fail-fast: false
    # Evaluator must succeed for its runner's output to be valid
    depends_on: corresponding_runner
    on_failure: invalidate_slice_pair

  analyzers:
    continue-on-error: true
    fail-fast: false
    on_failure: produce_partial_report
```

### 12.4 Degradation Threshold Matrix

| Failed Slice Pairs | Pipeline Mode | CI Job Status | Governance Label |
|--------------------|---------------|---------------|------------------|
| 0 | FULL_PIPELINE | ✓ Success | OK |
| 1 | DEGRADED_ANALYSIS | ⚠ Unstable | WARN |
| 2 | DEGRADED_ANALYSIS | ⚠ Unstable | WARN |
| 3 | EVIDENCE_ONLY | ✗ Failed | DO NOT USE |
| 4 | EVIDENCE_ONLY | ✗ Failed | DO NOT USE |

### 12.5 Slice Pair Failure Propagation

When a slice pair fails, the failure MUST propagate to exclude ALL derived computations:

```
Slice Pair Failure Propagation Graph:

run-slice-X fails
        │
        ├──► eval-X skipped (no input)
        │
        ├──► Δp(X) excluded from summary
        │
        ├──► Cohen's h(X) excluded
        │
        ├──► Wilson CI(X) excluded
        │
        ├──► Chain analysis for X excluded
        │
        └──► Evidence Pack marks X as FAILED

eval-X fails (even if run-slice-X succeeded)
        │
        ├──► Δp(X) excluded from summary
        │
        ├──► Cohen's h(X) excluded
        │
        ├──► Wilson CI(X) excluded
        │
        └──► Evidence Pack marks X as INCOMPLETE
```

### 12.6 CI Workflow with Degradation Handling

```yaml
# U2 Pipeline CI with Degradation Handling
# PHASE II — NOT RUN IN PHASE I
# No uplift claims are made.

name: u2-pipeline-degraded

on:
  workflow_dispatch:
    inputs:
      allow_degraded:
        description: 'Allow DEGRADED_ANALYSIS mode'
        type: boolean
        default: true

env:
  PIPELINE_MODE: FULL_PIPELINE  # Initial assumption
  FAILED_SLICES: ""

jobs:
  # ============================================================
  # HARD-FAIL STAGES
  # ============================================================

  validation:
    runs-on: ubuntu-latest
    outputs:
      validation_passed: ${{ steps.validate.outputs.passed }}
    steps:
      - uses: actions/checkout@v4

      - name: Gate Check
        id: gate
        run: |
          python scripts/u2_gate_check.py
        # No continue-on-error - HARD FAIL

      - name: Prereg Verify
        run: python scripts/u2_prereg_verify.py

      - name: Curriculum Load
        run: python scripts/u2_curriculum_load.py

      - name: Dry Run
        run: python scripts/u2_dry_run.py

      - name: Manifest Init
        run: python scripts/u2_manifest_init.py

      - name: Set Validation Output
        id: validate
        run: echo "passed=true" >> $GITHUB_OUTPUT

  # ============================================================
  # SOFT-FAIL STAGES (Slice Execution)
  # ============================================================

  slice-goal:
    needs: validation
    runs-on: ubuntu-latest
    continue-on-error: true  # SOFT FAIL
    outputs:
      status: ${{ steps.run.outcome }}
    steps:
      - uses: actions/checkout@v4
      - name: Run Goal Slice
        id: run
        run: |
          python scripts/u2_run_slice.py --slice goal

  slice-sparse:
    needs: validation
    runs-on: ubuntu-latest
    continue-on-error: true  # SOFT FAIL
    outputs:
      status: ${{ steps.run.outcome }}
    steps:
      - uses: actions/checkout@v4
      - name: Run Sparse Slice
        id: run
        run: |
          python scripts/u2_run_slice.py --slice sparse

  slice-tree:
    needs: validation
    runs-on: ubuntu-latest
    continue-on-error: true  # SOFT FAIL
    outputs:
      status: ${{ steps.run.outcome }}
    steps:
      - uses: actions/checkout@v4
      - name: Run Tree Slice
        id: run
        run: |
          python scripts/u2_run_slice.py --slice tree

  slice-dep:
    needs: validation
    runs-on: ubuntu-latest
    continue-on-error: true  # SOFT FAIL
    outputs:
      status: ${{ steps.run.outcome }}
    steps:
      - uses: actions/checkout@v4
      - name: Run Dependency Slice
        id: run
        run: |
          python scripts/u2_run_slice.py --slice dep

  # ============================================================
  # DEGRADATION CHECK
  # ============================================================

  degradation-check:
    needs: [slice-goal, slice-sparse, slice-tree, slice-dep]
    runs-on: ubuntu-latest
    outputs:
      pipeline_mode: ${{ steps.check.outputs.mode }}
      successful_slices: ${{ steps.check.outputs.successful }}
      failed_slices: ${{ steps.check.outputs.failed }}
    steps:
      - name: Evaluate Degradation
        id: check
        run: |
          # Count successful slices
          SUCCESSFUL=0
          FAILED=""

          if [ "${{ needs.slice-goal.outputs.status }}" == "success" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
          else
            FAILED="${FAILED}goal,"
          fi

          if [ "${{ needs.slice-sparse.outputs.status }}" == "success" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
          else
            FAILED="${FAILED}sparse,"
          fi

          if [ "${{ needs.slice-tree.outputs.status }}" == "success" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
          else
            FAILED="${FAILED}tree,"
          fi

          if [ "${{ needs.slice-dep.outputs.status }}" == "success" ]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
          else
            FAILED="${FAILED}dep,"
          fi

          # Determine mode
          if [ $SUCCESSFUL -eq 4 ]; then
            MODE="FULL_PIPELINE"
          elif [ $SUCCESSFUL -ge 2 ]; then
            MODE="DEGRADED_ANALYSIS"
          else
            MODE="EVIDENCE_ONLY"
          fi

          echo "mode=$MODE" >> $GITHUB_OUTPUT
          echo "successful=$SUCCESSFUL" >> $GITHUB_OUTPUT
          echo "failed=$FAILED" >> $GITHUB_OUTPUT

          echo "Pipeline Mode: $MODE"
          echo "Successful Slices: $SUCCESSFUL"
          echo "Failed Slices: $FAILED"

  # ============================================================
  # MODE-CONDITIONAL STAGES
  # ============================================================

  evaluators:
    needs: [degradation-check, slice-goal, slice-sparse, slice-tree, slice-dep]
    if: needs.degradation-check.outputs.pipeline_mode != 'EVIDENCE_ONLY'
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        slice: [goal, sparse, tree, dep]
    steps:
      - uses: actions/checkout@v4

      - name: Check Slice Status
        id: check
        run: |
          FAILED="${{ needs.degradation-check.outputs.failed_slices }}"
          if [[ "$FAILED" == *"${{ matrix.slice }}"* ]]; then
            echo "skip=true" >> $GITHUB_OUTPUT
            echo "Skipping ${{ matrix.slice }} - slice failed"
          else
            echo "skip=false" >> $GITHUB_OUTPUT
          fi

      - name: Evaluate Slice
        if: steps.check.outputs.skip != 'true'
        run: |
          python scripts/u2_evaluate.py --slice ${{ matrix.slice }}

  analysis:
    needs: [degradation-check, evaluators]
    if: needs.degradation-check.outputs.pipeline_mode != 'EVIDENCE_ONLY'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Chain Analysis
        run: |
          python scripts/u2_chain_analyze.py \
            --exclude "${{ needs.degradation-check.outputs.failed_slices }}"

      - name: Statistical Summary
        run: |
          python scripts/u2_stat_summary.py \
            --mode ${{ needs.degradation-check.outputs.pipeline_mode }} \
            --exclude "${{ needs.degradation-check.outputs.failed_slices }}"

  # ============================================================
  # EVIDENCE PRESERVATION (Always Runs)
  # ============================================================

  preserve-evidence:
    needs: [degradation-check]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Collect Evidence
        run: |
          python scripts/u2_collect_evidence.py \
            --mode ${{ needs.degradation-check.outputs.pipeline_mode }}

      - name: Upload Evidence
        uses: actions/upload-artifact@v4
        with:
          name: evidence-pack-${{ github.run_id }}
          path: artifacts/evidence/

  # ============================================================
  # FINAL SEAL
  # ============================================================

  seal:
    needs: [degradation-check, evaluators, analysis, preserve-evidence]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Determine Governance Label
        id: label
        run: |
          MODE="${{ needs.degradation-check.outputs.pipeline_mode }}"
          case $MODE in
            FULL_PIPELINE)
              echo "label=OK" >> $GITHUB_OUTPUT
              echo "attestation=full" >> $GITHUB_OUTPUT
              ;;
            DEGRADED_ANALYSIS)
              echo "label=WARN" >> $GITHUB_OUTPUT
              echo "attestation=degraded" >> $GITHUB_OUTPUT
              ;;
            EVIDENCE_ONLY)
              echo "label=DO_NOT_USE" >> $GITHUB_OUTPUT
              echo "attestation=forensic" >> $GITHUB_OUTPUT
              ;;
          esac

      - name: Generate Attestation
        run: |
          python scripts/u2_attestation.py \
            --mode ${{ needs.degradation-check.outputs.pipeline_mode }} \
            --governance-label ${{ steps.label.outputs.label }} \
            --attestation-type ${{ steps.label.outputs.attestation }}

      - name: Seal Evidence Pack
        run: |
          python scripts/u2_seal.py \
            --mode ${{ needs.degradation-check.outputs.pipeline_mode }}

      - name: Report Final Status
        run: |
          echo "========================================"
          echo "U2 Pipeline Complete"
          echo "========================================"
          echo "Mode: ${{ needs.degradation-check.outputs.pipeline_mode }}"
          echo "Governance: ${{ steps.label.outputs.label }}"
          echo "Successful Slices: ${{ needs.degradation-check.outputs.successful_slices }}"
          echo "Failed Slices: ${{ needs.degradation-check.outputs.failed_slices }}"
          echo "========================================"
```

### 12.7 Degradation Decision Tree

```
                                    ┌─────────────────────────────────┐
                                    │        Pipeline Started         │
                                    └─────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌─────────────────────────────────┐
                                    │    Validation Stages Pass?      │
                                    └─────────────────────────────────┘
                                           │                 │
                                          Yes               No
                                           │                 │
                                           ▼                 ▼
                              ┌────────────────────┐  ┌─────────────────┐
                              │   Run All Slices   │  │  EVIDENCE_ONLY  │
                              │    (parallel)      │  │  (hard-fail)    │
                              └────────────────────┘  └─────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────────────────────┐
                              │      Count Successful Slice Pairs      │
                              └────────────────────────────────────────┘
                                    │           │           │
                                   4           2-3         0-1
                                    │           │           │
                                    ▼           ▼           ▼
                          ┌─────────────┐ ┌────────────┐ ┌─────────────┐
                          │FULL_PIPELINE│ │ DEGRADED_  │ │EVIDENCE_ONLY│
                          │   (OK)      │ │ ANALYSIS   │ │(DO NOT USE) │
                          │             │ │  (WARN)    │ │             │
                          └─────────────┘ └────────────┘ └─────────────┘
                                    │           │           │
                                    ▼           ▼           ▼
                              ┌─────────────────────────────────────────┐
                              │           Compute Δp Values             │
                              ├─────────────────────────────────────────┤
                              │ FULL: All 4 Δp values                   │
                              │ DEGRADED: Only successful pairs' Δp     │
                              │ EVIDENCE: NO Δp values (forbidden)      │
                              └─────────────────────────────────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────────┐
                              │           Seal Evidence Pack            │
                              ├─────────────────────────────────────────┤
                              │ FULL: Full cryptographic seal           │
                              │ DEGRADED: Degraded seal with warnings   │
                              │ EVIDENCE: Forensic seal (quarantine)    │
                              └─────────────────────────────────────────┘
```

### 12.8 Governance Label Assignment

| Condition | Governance Label | Evidence Pack Status | Permitted Claims |
|-----------|------------------|----------------------|------------------|
| All validation passes + all 4 slice pairs succeed | **OK** | COMPLETE | Full Δp claims, statistical significance |
| All validation passes + 2-3 slice pairs succeed | **WARN** | PARTIAL | Δp claims for successful pairs only |
| Any validation fails | **DO NOT USE** | FORENSIC | NO claims permitted |
| Fewer than 2 slice pairs succeed | **DO NOT USE** | FORENSIC | NO claims permitted |
| Any corruption detected | **DO NOT USE** | FORENSIC | NO claims permitted |

### 12.9 CI Status Reporting

```yaml
# Status reporting configuration
status_reporting:
  on_full_pipeline:
    github_status: success
    slack_color: good
    message: "U2 Pipeline: FULL_PIPELINE (OK) - All slices succeeded"

  on_degraded_analysis:
    github_status: neutral  # Shows as yellow
    slack_color: warning
    message: "U2 Pipeline: DEGRADED_ANALYSIS (WARN) - {failed_slices} failed"

  on_evidence_only:
    github_status: failure
    slack_color: danger
    message: "U2 Pipeline: EVIDENCE_ONLY (DO NOT USE) - Critical failure"
```

---

*Generated by: CLAUDE_B (Workflow Pipeline Architect / Health & Degradation Spec)*
*Phase: II*
*Status: PHASE II — NOT RUN IN PHASE I*
*Date: 2025-12-06*
