# Curriculum Drift → Governance Signal Specification

**Document ID**: `CUR-GOV-SPEC-V1`  
**Author**: MANUS-E, Curriculum Integrity Engineer  
**Status**: DRAFT  
**Date**: 2025-12-10

---

## 1. Overview

This document defines the formal mapping between detected curriculum drift events and the governance signals they produce within the MathLedger RFL (Reflexive Formal Learning) infrastructure. The primary goal is to create a deterministic, fail-closed system that preserves the scientific validity of all experimental runs by preventing uncontrolled changes to the curriculum.

This specification is enforced by the `CurriculumDriftSentinel` and the `NormalizedMetrics` schema validator.

## 2. Drift Categories & Governance Signals

All drift events are categorized and mapped to one of two governance signals: **BLOCK** or **WARN**. The default signal for any unrecognized drift is **BLOCK**.

| Drift Category | Trigger Condition | Governance Signal | Enforcement Mechanism |
| :--- | :--- | :--- | :--- |
| **ContentDrift** | `CurriculumSystem.fingerprint()` changes | **BLOCK** | `RFLRunner` raises `CurriculumDriftError` before run | 
| **SchemaDrift** | `CurriculumSystem.version` changes | **BLOCK** | `RFLRunner` raises `CurriculumDriftError` before run | 
| **SliceCountDrift** | `len(CurriculumSystem.slices)` changes | **BLOCK** | `RFLRunner` raises `CurriculumDriftError` before run | 
| **MetricSchemaDrift** | Metric payload fails v1 schema validation | **BLOCK** / **WARN** | `NormalizedMetrics.from_raw()` raises `CurriculumDriftError` or logs to `stderr` |

### 2.1. BLOCK Signal

A **BLOCK** signal is a fatal, run-terminating event. It indicates that a non-recoverable drift has occurred, which would invalidate the results of the experiment. 

- **Action**: The `RFLRunner` (or other orchestration agent) MUST immediately halt execution.
- **Artifact**: A `drift_report.json` MUST be generated in the run's artifact directory, detailing the violations.
- **Rationale**: Prevents the system from wasting compute resources on a scientifically invalid experiment. Ensures that all data in the run ledger is associated with a valid, fingerprinted curriculum.

### 2.2. WARN Signal

A **WARN** signal is a non-blocking event used for discovery and migration. It indicates that a drift has been detected, but the system is configured to proceed with the run.

- **Action**: The `RFLRunner` MUST log the drift violation to `stderr` and continue execution.
- **Applicability**: This signal is exclusively used by the `NormalizedMetrics` validator when `METRIC_SCHEMA_ENFORCEMENT_MODE` is set to `log_only`.
- **Rationale**: Enables the safe discovery of non-compliant metric payloads in production without disrupting existing pipelines. This is a critical tool for backward-compatible migration.

---

## 3. Provenance & Invariant Mapping

This section defines how drift detection artifacts are integrated into the broader governance and provenance system.

### 3.1. Fingerprint Deltas → Governance Provenance

The curriculum fingerprint is the cornerstone of provenance. It provides an immutable, content-addressable link between an experimental run and the exact curriculum configuration used.

- **Ledger Requirement**: Every `RunLedgerEntry` MUST contain the `curriculum_fingerprint` of the curriculum used for that run.
- **On Drift**: When a **BLOCK** signal is issued due to `ContentDrift`, the generated `drift_report.json` MUST contain both the `expected_fingerprint` and the `observed_fingerprint`.
- **Analysis**: This allows for precise post-mortem analysis. By comparing the two fingerprints, a `curriculum_differ.py` utility can be used to generate a human-readable report detailing the exact changes (e.g., parameter modifications, gate threshold adjustments) that caused the drift.

**Formal Process**:
1. `RFLRunner` initializes, loading `CurriculumSystem` and computing a baseline fingerprint (`F_base`).
2. `F_base` is stored in the `CurriculumDriftSentinel`.
3. Before each experimental phase, the runner re-computes the fingerprint (`F_current`).
4. If `F_current != F_base`, a **BLOCK** signal is issued.
5. The `RunLedgerEntry` for this run is populated with `F_base`.

### 3.2. Curriculum Version → Schema Invariants

The `version` field within the `curriculum.yaml` file serves as a guard for the curriculum's structural schema, not its content. A change in this version number implies a change in the fundamental invariants of the curriculum system itself.

- **Definition**: The `version` field (e.g., `version: 2`) represents the version of the curriculum *schema*, not the curriculum *content*.
- **Trigger**: Any change to this integer (e.g., `2` → `3`) will trigger a `SchemaDrift` violation and a **BLOCK** signal.
- **Governance Rule**: A change in the curriculum `version` MUST be accompanied by a corresponding update to the schema validation logic in `backend/frontier/curriculum.py`. It signifies that the shape of slices, gates, or system configuration has been altered in a backward-incompatible way.
- **Example Invariants Guarded by `version`**:
  - The required set of gate specifications (e.g., `coverage`, `abstention`, `velocity`, `caps`).
  - The data types of gate parameters (e.g., `ci_lower_min` must be a float).
  - The presence of required fields like `name` and `params` in a slice.

Changing the `version` is a high-ceremony event that indicates a developer has intentionally modified the curriculum processing system itself. The `SchemaDrift` check ensures that old runners cannot accidentally run on new, incompatible curriculum schemas.

---

## 4. Metric Schema Enforcement Modes

The `MetricSchemaDrift` category has a unique, configurable governance model to facilitate safe migration.

| `METRIC_SCHEMA_ENFORCEMENT_MODE` | Governance Signal | Behavior |
| :--- | :--- | :--- |
| `permissive` (Default) | NONE | Ignores schema drift. Uses `_first_available()` logic for maximum backward compatibility. | 
| `log_only` | **WARN** | Logs schema discrepancies to `stderr` but does not block execution. Returns the permissively-parsed result. | 
| `strict` | **BLOCK** | Enforces canonical v1 schema using `_get_metric_by_path()`. Raises `CurriculumDriftError` on any missing path. | 

This tiered model allows the system to be gradually hardened, moving from a state of no enforcement to full, fail-closed governance without disrupting production workflows.
