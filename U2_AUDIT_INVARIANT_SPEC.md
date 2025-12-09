# MathLedger Phase II - U2 Audit Invariant Specification

> **STATUS: DRAFT**
> **ID: U2_AUDIT_INVARIANT_SPEC.md**
>
> This document provides the formal specification for all invariants checked by the Phase II U2 DAG Audit System. It defines the required inputs, output semantics, and error-state behaviors for each invariant.

## 1. Overview

The U2 DAG Audit system is designed to provide observational checks on the structural and logical integrity of the Proof DAG during Phase II uplift experiments. The invariants are categorized into three groups:

*   **Chain Dependency (`CD`):** Invariants governing the structure and temporal ordering of derivation chains.
*   **Multi-Goal (`MG`):** Invariants governing proof sets in experiments with multiple concurrent goals.
*   **DAG Evolution (`EV`):** Invariants governing the isolation and transactional integrity of the DAG during experimentation.

## 2. Chain Dependency Invariants

### INV-P2-CD-1: Chain Completeness

| | |
|---|---|
| **Name** | Chain Completeness |
| **Severity** | `CRITICAL` |
| **Description** | Ensures every U2-generated proof chain has an unbroken ancestry back to a registered axiom or a previously-verified statement from the main DAG. |
| **Required Inputs**| - `u2_proof_parents` table for the experiment.<br>- Set of `u2_statements` hashes to be checked.<br>- Set of valid root hashes (axioms + pre-experiment verified statements). |
| **Output Semantics**| **PASS**: All checked statements successfully trace back to a valid root.<br>**FAIL**: One or more statements are part of an orphaned chain with no valid root ancestor. |
| **Error States** | - **Missing `u2_proof_parents`**: The check cannot run; returns `ERROR` status.<br>- **Missing Root Set**: The check cannot validate roots; returns `ERROR` status. |

### INV-P2-CD-2: Chain Dependency Ordering

| | |
|---|---|
| **Name** | Chain Dependency Ordering |
| **Severity** | `CRITICAL` |
| **Description** | Within a single U2 cycle, a statement cannot depend on another statement derived later in the same cycle. |
| **Required Inputs**| - `u2_statements` for the experiment, with `cycle_number` and a deterministic intra-cycle derivation order.<br>- `u2_proof_parents` edges for the experiment. |
| **Output Semantics**| **PASS**: All intra-cycle dependencies respect the derivation order (parent order < child order).<br>**FAIL**: One or more edges exhibit a temporal violation where `parent_order >= child_order`. |
| **Error States** | - **Missing Derivation Order**: Cannot determine temporal sequence; returns `ERROR` status.<br>- **Missing Cycle Numbers**: Cannot scope the check; returns `ERROR` status. |

### INV-P2-CD-3: Cross-Cycle Dependency Bounds

| | |
|---|---|
| **Name** | Cross-Cycle Dependency Bounds |
| **Severity** | `WARNING` |
| **Description** | Dependencies spanning multiple cycles must not exceed a configured maximum distance (`MAX_CROSS_CYCLE_DISTANCE`). |
| **Required Inputs**| - `u2_statements` with `cycle_number` for all statements.<br>- `u2_proof_parents` edges.<br>- `MAX_CROSS_CYCLE_DISTANCE` configuration parameter. |
| **Output Semantics**| **PASS**: All cross-cycle dependencies are within the allowed distance.<br>**WARN**: One or more edges span a cycle distance greater than the configured maximum. |
| **Error States** | - **Missing Cycle Numbers**: Cannot calculate distance; returns `ERROR` status. |

## 3. Multi-Goal Proof Set Invariants

### INV-P2-MG-1: Goal Attribution Completeness

| | |
|---|---|
| **Name** | Goal Attribution Completeness |
| **Severity** | `ERROR` |
| **Description** | Every derived statement in a multi-goal experiment must be attributed to at least one goal from the experiment's goal set. |
| **Required Inputs**| - `u2_statements` for the experiment.<br>- `u2_goal_attributions` table.<br>- Experiment metadata indicating if it is a multi-goal experiment. |
| **Output Semantics**| **PASS**: All derived statements have at least one entry in the attributions table.<br>**FAIL**: One or more derived statements are found with no corresponding goal attribution.<br>**SKIP**: The experiment is not a multi-goal experiment. |
| **Error States** | - **Missing `u2_goal_attributions` table**: Cannot verify attributions; returns `ERROR` status. |

### INV-P2-MG-2: Goal Conflict Detection

| | |
|---|---|
| **Name** | Goal Conflict Detection |
| **Severity** | `WARNING` |
| **Description** | Flags statements that contribute to multiple, logically conflicting goals. |
| **Required Inputs**| - `u2_goal_attributions` table.<br>- Experiment manifest containing goal definitions (e.g., target formulas).<br>- Logic for determining goal conflicts (e.g., `is_negation(formula1, formula2)`). |
| **Output Semantics**| **PASS**: No statements are attributed to conflicting goals.<br>**WARN**: One or more statements are attributed to goals that are determined to be in conflict.<br>**SKIP**: The experiment is not a multi-goal experiment. |
| **Error States** | - **Missing Goal Definitions**: Cannot determine conflicts; returns `ERROR` status. |

### INV-P2-MG-3: Goal Progress Monotonicity

| | |
|---|---|
| **Name** | Goal Progress Monotonicity |
| **Severity** | `OBSERVATIONAL` |
| **Description** | Progress toward each goal (e.g., max proof chain depth) must be non-decreasing across cycles. |
| **Required Inputs**| - `u2_statements` with cycle numbers.<br>- `u2_proof_parents` to calculate chain depth.<br>- `u2_goal_attributions` to associate statements with goals. |
| **Output Semantics**| **OBSERVE**: The check runs and records the metrics. The status is always `OBSERVE` as this is not a pass/fail condition but a metric to be monitored. The output contains the history of progress for each goal. |
| **Error States** | - **Missing Cycle Numbers**: Cannot track progress over time; returns `ERROR` status. |

## 4. DAG Evolution Constraints

### INV-P2-EV-1: Experiment Isolation

| | |
|---|---|
| **Name** | Experiment Isolation |
| **Severity** | `CRITICAL` |
| **Description** | U2 experiment DAG modifications must occur in isolated "shadow" tables and not be merged to the main DAG until experiment completion and validation. |
| **Required Inputs**| - `u2_experiments` table with experiment status.<br>- `u2_proof_parents` table to check for premature `merged_to_main` flags.<br>- `proof_parents` (main table) to check for illicit direct writes. |
| **Output Semantics**| **PASS**: Experiment is not completed and no edges are marked as merged, OR experiment is completed/validated and merge is proceeding correctly.<br>**FAIL**: Edges from a running experiment are found in the main `proof_parents` table, or are marked `merged_to_main=TRUE` prematurely. |
| **Error States** | - **Inability to access both shadow and main tables**: Cannot verify isolation; returns `ERROR`. |

### INV-P2-EV-2: Concurrent Experiment Non-Interference

| | |
|---|---|
| **Name** | Concurrent Experiment Non-Interference |
| **Severity** | `CRITICAL` |
| **Description** | Concurrent U2 experiments must not share any intermediate (derived) statements. |
| **Required Inputs**| - `u2_statements` table for all experiments.<br>- `u2_experiments` table to identify start/end times and find overlapping runs. |
| **Output Semantics**| **PASS**: No derived statement hashes are shared between experiments running concurrently.<br>**FAIL**: One or more derived statement hashes are found in two or more concurrent experiments.<br>**SKIP**: No concurrent experiments were found. |
| **Error States** | - **Missing Timestamps**: Cannot determine concurrency; returns `ERROR` status. |

### INV-P2-EV-3: DAG Snapshot Consistency

| | |
|---|---|
| **Name** | DAG Snapshot Consistency |
| **Severity** | `CRITICAL` |
| **Description** | The baseline DAG state, captured in a snapshot at the start of an experiment, must remain immutable. |
| **Required Inputs**| - `u2_dag_snapshots` table containing the baseline Merkle hash and counts for the experiment.<br>- Access to the main `statements` and `proof_parents` tables to re-calculate the Merkle hash at the snapshot timestamp. |
| **Output Semantics**| **PASS**: The re-calculated Merkle hash of the baseline state matches the hash stored in the snapshot.<br>**FAIL**: The hashes do not match, indicating the baseline DAG was mutated after the snapshot was taken. |
| **Error States** | - **Missing Snapshot**: Cannot verify consistency; returns `ERROR` status. |

### INV-P2-EV-4: Rollback Capability

| | |
|---|---|
| **Name** | Rollback Capability |
| **Severity** | `CRITICAL` |
| **Description** | An experiment marked as "rolled_back" must have no remaining artifacts in the U2 shadow tables. For active experiments, ensures no external dependencies would prevent a clean rollback. |
| **Required Inputs**| - `u2_experiments` table with experiment status.<br>- `u2_statements`, `u2_proof_parents`, `u2_goal_attributions` tables. |
| **Output Semantics**| **PASS**: For a 'rolled_back' experiment, no artifacts are found. For an active experiment, no blocking foreign key constraints exist.<br>**FAIL**: For a 'rolled_back' experiment, orphaned artifacts (statements, edges) are found.<br>**WARN**: For an active experiment, a potential future rollback issue is detected. |
| **Error States** | - **Incomplete experiment status**: Cannot determine correct check to run; returns `ERROR`. |

### INV-P2-EV-5: Slice-Scoped Mutations

| | |
|---|---|
| **Name** | Slice-Scoped Mutations |
| **Severity** | `ERROR` |
| **Description** | All statements derived within an experiment must conform to the constraints (e.g., max atoms, max depth, permitted connectives) defined in the experiment's slice configuration. |
| **Required Inputs**| - `u2_statements` table for the experiment, including statement text.<br>- Experiment manifest or `u2_experiments` table containing the `slice_config`.<br>- A formula parser to analyze statement text. |
| **Output Semantics**| **PASS**: All derived statements conform to the slice constraints.<br>**FAIL**: One or more statements violate the constraints (e.g., too many atoms, uses a forbidden connective). |
| **Error States** | - **Missing Slice Configuration**: Cannot verify constraints; returns `ERROR`.<br>- **Formula Parsing Failure**: Cannot check a statement; reports a per-statement error. |

---

## 5. Audit Completeness Report Schema

The Audit Completeness Report is a meta-report generated by the audit system *before* the main audit runs. Its purpose is to verify that the audit itself can be performed completely and correctly. It checks for the availability of all required data sources and system prerequisites.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "U2 Audit Completeness Report",
  "description": "A report verifying the availability of all prerequisites and data sources required for a U2 DAG audit.",
  "type": "object",
  "properties": {
    "report_metadata": {
      "type": "object",
      "properties": {
        "report_id": { "type": "string", "format": "uuid" },
        "timestamp": { "type": "string", "format": "date-time" },
        "experiment_id": { "type": "string" },
        "auditor_version": { "type": "string" }
      },
      "required": ["report_id", "timestamp", "experiment_id"]
    },
    "completeness_status": {
      "type": "string",
      "enum": ["READY", "INCOMPLETE", "ERROR"],
      "description": "Overall status of audit readiness."
    },
    "data_source_status": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source_name": { "type": "string", "description": "e.g., u2_statements, manifest.json" },
          "status": { "type": "string", "enum": ["FOUND", "NOT_FOUND", "ACCESS_DENIED", "SCHEMA_MISMATCH"] },
          "details": { "type": "string", "description": "Additional context, e.g., path to file or name of missing table." }
        },
        "required": ["source_name", "status"]
      }
    },
    "prerequisite_checks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "check_name": { "type": "string", "description": "e.g., DB_EXTENSION_PG_TRGM, SCHEMA_VERSION" },
          "status": { "type": "string", "enum": ["MET", "NOT_MET", "UNKNOWN"] },
          "expected_value": { "type": "string" },
          "actual_value": { "type": "string" }
        },
        "required": ["check_name", "status"]
      }
    },
    "invariant_coverage": {
        "type": "object",
        "properties": {
            "total_invariants_in_spec": { "type": "integer" },
            "invariants_to_be_run": { "type": "array", "items": {"type": "string"} },
            "invariants_skipped_due_to_missing_data": { "type": "array", "items": {"type": "string"} }
        },
        "required": ["total_invariants_in_spec", "invariants_to_be_run"]
    }
  },
  "required": ["report_metadata", "completeness_status", "data_source_status", "prerequisite_checks", "invariant_coverage"]
}
```
