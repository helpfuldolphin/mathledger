# U2 Telemetry Consistency Contract

**Document ID:** `U2_TCC_v1.0`
**Author:** Gemini H, Telemetry Data-Rigor Engineer
**Status:** ACTIVE
**Mandate:** PHASE II — TELEMETRY CONSISTENCY ENFORCEMENT ENGINE

---

## 1. Overview

This document establishes the canonical consistency rules for the U2 telemetry pipeline. The schemas (`u2-cycle-v1`, `u2-summary-v1`) define structural validity; this contract defines **semantic and relational integrity**.

Adherence to this contract is mandatory for all data producers (U2 Runners) and consumers (Metrics Ingesters, Governance Gates). The rules herein form the basis for the Telemetry Drift Detection engine.

## 2. Core Principles

- **Trust but Verify:** The pipeline assumes producers *intend* to be correct, but consumers *must* verify consistency.
- **Immutable Provenance:** There must be an unbroken, verifiable chain from the low-level cycle events to the high-level summary conclusions.
- **Strict Schema Adherence:** No extra-schematic data is permitted. The schema is the ground truth.

---

## 3. Field-Level Consistency Rules

These rules define the mathematical and logical relationships between the `u2-cycle-v1` event stream and the `u2-summary-v1` artifact. Any violation of these rules indicates a critical data integrity failure.

### 3.1. Run ID Consistency

- **Rule:** The `baseline_run_id` and `rfl_run_id` fields in a summary record *must* correspond to the `run_id`s of the cycle events used to generate it.
- **Verification Logic:**
  ```
  // Let S be a summary record
  // Let C_base be the set of all cycle events where run_id == S.baseline_run_id
  // Let C_rfl be the set of all cycle events where run_id == S.rfl_run_id
  
  assert C_base is not empty
  assert C_rfl is not empty
  assert for all c in C_base, c.mode == "baseline"
  assert for all c in C_rfl, c.mode == "rfl"
  ```

### 3.2. Slice Consistency

- **Rule:** The `slice` field in a summary record must be identical to the `slice` field in every single cycle event that contributed to it.
- **Verification Logic:**
  ```
  // Let S be a summary record
  // Let C_all be the union of cycle events from S.baseline_run_id and S.rfl_run_id
  
  assert for all c in C_all, c.slice == S.slice
  ```

### 3.3. Cycle Count (`n_cycles`) Consistency

- **Rule:** The `n_cycles` values in a summary record must exactly equal the total count of cycle events for their respective runs.
- **Verification Logic:**
  ```
  // Let S be a summary record
  
  assert count(events where run_id == S.baseline_run_id) == S.n_cycles.baseline
  assert count(events where run_id == S.rfl_run_id) == S.n_cycles.rfl
  ```

### 3.4. Success Probability (`p_base`, `p_rfl`) Consistency

- **Rule:** The success probabilities (`p_base`, `p_rfl`) in a summary record must be the exact result of the calculation over the corresponding cycle events. The definition of a successful cycle is `success == true`.
- **Verification Logic:**
  ```
  // Let S be a summary record
  
  // Calculate p_base from source
  let base_success_count = count(events where run_id == S.baseline_run_id and success == true)
  let p_base_calculated = base_success_count / S.n_cycles.baseline
  
  // Calculate p_rfl from source
  let rfl_success_count = count(events where run_id == S.rfl_run_id and success == true)
  let p_rfl_calculated = rfl_success_count / S.n_cycles.rfl
  
  // Assert equality within a small epsilon for floating point comparisons
  assert abs(p_base_calculated - S.p_base) < 1e-9
  assert abs(p_rfl_calculated - S.p_rfl) < 1e-9
  ```

### 3.5. Uplift Delta (`delta`) Consistency

- **Rule:** The `delta` value in a summary record must be the exact difference between its `p_rfl` and `p_base` fields.
- **Verification Logic:**
  ```
  // Let S be a summary record
  let delta_calculated = S.p_rfl - S.p_base
  assert abs(delta_calculated - S.delta) < 1e-9
  ```

---

## 4. Timestamp Generation and Ordering Rules

To ensure auditability and prevent data corruption, timestamp generation must be deterministic and verifiable.

- **Rule 4.1: Format and Precision:** All `ts` fields MUST be strings formatted according to **ISO 8601 UTC** with microsecond precision (e.g., `YYYY-MM-DDTHH:MM:SS.ffffffZ`).

- **Rule 4.2: Strict Monotonicity:** Within the scope of a single `run_id`, timestamps MUST be strictly monotonic and correlate with the `cycle` index.
- **Verification Logic:**
  ```
  // Let C be the set of cycle events for a single run, sorted by cycle index
  
  assert for i from 0 to len(C)-2:
      parse(C[i].ts) < parse(C[i+1].ts)
  ```

---

## 5. Schema Adherence (Allowed/Forbidden Fields)

- **Rule 5.1: No Undeclared Fields:** Any received data packet (cycle event or summary) that contains fields not explicitly defined in the canonical schema for its type and version (`u2-cycle-v1`, `u2-summary-v1`) is **INVALID**. Such packets must be rejected and must trigger a schema drift alert.

- **Rule 5.2: All Required Fields:** Any packet missing one or more required fields from its canonical schema is **INVALID** and must be rejected.

---

## Appendix A: Telemetry Drift Detection Rules

This appendix defines a set of automated monitoring rules designed to enforce the contract and detect data drift in real-time.

| Drift Type | Rule Definition | Alert Condition |
| :--- | :--- | :--- |
| **Schema Drift** | For every incoming event, verify that `set(event.keys())` is a subset of the allowed keys in the schema definition. | An event contains a key not defined in the canonical schema. |
| **Cardinality Drift** | Maintain a historical set of observed values for all `enum` fields (`slice`, `mode`). | A new value appears that is not in the schema's `enum` list (e.g., `slice` appears as `"U2_env_F"`). |
| **Type Drift** | For every incoming event, verify the data type of each field (e.g., `isinstance(event.cycle, int)`). | The type of any field does not match its schema definition. |
| **Monotonicity Violation** | In the ingestion buffer, before final commit, verify that timestamps for a given run are strictly increasing with the cycle index. | `ts` of `cycle[N+1]` is less than or equal to `ts` of `cycle[N]`. |
| **Statistical Drift** | Maintain a 90-day rolling mean and standard deviation for key metrics (e.g., `duration_seconds`). | A new hourly-average value for a metric is more than 5 standard deviations away from the rolling mean. |
| **Consistency Drift** | As a nightly or batch job, re-calculate a sample of summary records from their source cycle logs for the past 24 hours. | The calculated `p_base`, `p_rfl`, or `delta` does not match the value stored in the summary record (outside a float epsilon). |
| **Provenance Failure** | When a summary is received, query the telemetry data store to ensure that its `baseline_run_id` and `rfl_run_id` exist. | A summary record references `run_id`s for which no cycle events have been ingested. |
