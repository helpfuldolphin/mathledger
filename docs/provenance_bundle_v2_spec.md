# Provenance Bundle v2 Specification

**Version**: 2.0.0  
**Author**: Manus-F, Deterministic Planner & Substrate Executor Engineer  
**Date**: 2025-12-06  
**Status**: Final

---

## 1. Overview

This document specifies the v2 format for the U2 Provenance Bundle. This version extends the MVP bundle with enhanced cryptographic commitments, slice-level metadata for RFL (Reinforcement Learning from Feedback) experiments, and new replay invariants required for the P4 analysis framework.

The Provenance Bundle is a self-contained, cryptographically verifiable package that encapsulates all artifacts, data, and metadata required to reproduce a U2 planner experiment and verify its outcomes.

## 2. Bundle Structure

The v2 bundle remains a single JSON file, but with an expanded structure. The top-level object contains the following key sections:

| Section | Description |
| :--- | :--- |
| `bundle_header` | Core metadata and the dual-hash commitment. |
| `slice_metadata` | Detailed configuration for the specific experiment slice. |
| `manifest` | A list of all files in the bundle with their hashes. |
| `hashes` | A collection of all cryptographic hashes for verification. |
| `p4_replay_invariants` | Expected outcomes for P4-level replay verification. |

### Example JSON Structure:

```json
{
  "bundle_header": { ... },
  "slice_metadata": { ... },
  "manifest": { ... },
  "hashes": { ... },
  "p4_replay_invariants": { ... }
}
```

## 3. Dual-Hash Commitment

The v2 bundle introduces a dual-hash commitment in the `bundle_header` to provide stronger integrity guarantees. This separates the verification of the bundle's *content* from its *metadata*.

1.  **`content_merkle_root`**: The Merkle root of all files listed in the `manifest`. This hash commits to the exact content of every artifact (traces, logs, etc.). It is identical to the `merkle_root` in the v1 bundle.
2.  **`metadata_hash`**: A SHA-256 hash of the canonical JSON representation of the `bundle_header` and `slice_metadata` sections (with the `metadata_hash` field itself excluded). This hash commits to the experiment's configuration and parameters.

| Field | Type | Description |
| :--- | :--- | :--- |
| `content_merkle_root` | String | The Merkle root of all file contents. |
| `metadata_hash` | String | The SHA-256 hash of the bundle's metadata sections. |

This dual-hash system allows verifiers to independently confirm the integrity of the experimental setup and the resulting data.

## 4. Slice-Level Metadata

The `slice_metadata` section is expanded to capture the full configuration of an RFL experiment slice.

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `slice_name` | String | Unique name for the experiment slice. | `"rfl_policy_v2"` |
| `master_seed` | String | The master seed for the PRNG. | `"0xmaster_v2"` |
| `total_cycles` | Integer | The total number of cycles executed. | `1000` |
| `policy_config` | Object | Configuration of the search policy used. | `{"name": "rfl", "version": "2.1"}` |
| `feature_set_version` | String | Version of the feature set used for RFL. | `"v1.2.0"` |
| `executor_config` | Object | Configuration of the FOSubstrateExecutor. | `{"name": "lean", "version": "4.0"}` |
| `budget_config` | Object | Budget limits for the experiment. | `{"max_time_s": 3600}` |

## 5. Replay Invariants for P4

The `p4_replay_invariants` section defines the expected hash values for verifying a replay within the P4 framework. It extends the v1 invariants with checks for RFL feedback determinism.

| Field | Type | Description |
| :--- | :--- | :--- |
| `trace_hash` | String | The canonical hash of the merged execution trace. |
| `final_frontier_hash` | String | The canonical hash of the state of the frontier at the end of the experiment. |
| `per_cycle_trace_hashes` | Object | A mapping from cycle number to the canonical hash of that cycle's trace events. |
| **`rfl_feedback_hash`** | **String** | **(New)** The canonical hash of the generated RFL feedback file. This ensures the feature extraction and aggregation process is deterministic. |
| **`policy_evolution_hash`** | **String** | **(New)** The hash of the final trained policy weights. This verifies that the policy learning process is deterministic. |

### P4 Replay Verification Process:

A P4-compliant replay verifier **MUST**:
1.  Re-run the experiment using the parameters from `slice_metadata`.
2.  Re-generate all artifacts, including the execution trace and RFL feedback.
3.  Re-train the policy based on the generated feedback.
4.  Compute the five invariant hashes from the replayed artifacts.
5.  Compare the computed hashes against the values in `p4_replay_invariants`.
6.  The replay is only considered successful if **all five hashes match**.

## 6. Full Provenance Bundle v2 Schema

```json
{
  "bundle_header": {
    "bundle_version": "2.0.0",
    "experiment_id": "exp_abc123",
    "timestamp_utc": "2025-12-06T18:00:00Z",
    "content_merkle_root": "sha256:abc...",
    "metadata_hash": "sha256:def..."
  },
  "slice_metadata": {
    "slice_name": "rfl_policy_v2",
    "master_seed": "0xmaster_v2",
    "total_cycles": 1000,
    "policy_config": {
      "name": "rfl",
      "version": "2.1",
      "learning_rate": 0.01
    },
    "feature_set_version": "v1.2.0",
    "executor_config": {
      "name": "lean",
      "version": "4.0"
    },
    "budget_config": {
      "max_time_s": 3600,
      "max_statements": 1000000
    }
  },
  "manifest": {
    "total_files": 3,
    "total_bytes": 102400,
    "files": [
      {
        "path": "traces/worker_0.jsonl",
        "sha256": "sha256:123...",
        "size_bytes": 51200
      },
      {
        "path": "feedback/feedback.json",
        "sha256": "sha256:456...",
        "size_bytes": 25600
      },
      {
        "path": "policies/final_policy.bin",
        "sha256": "sha256:789...",
        "size_bytes": 25600
      }
    ]
  },
  "hashes": {
    "trace_hash": "sha256:aaa...",
    "final_frontier_hash": "sha256:bbb...",
    "rfl_feedback_hash": "sha256:ccc...",
    "policy_evolution_hash": "sha256:ddd..."
  },
  "p4_replay_invariants": {
    "expected_trace_hash": "sha256:aaa...",
    "expected_final_frontier_hash": "sha256:bbb...",
    "expected_rfl_feedback_hash": "sha256:ccc...",
    "expected_policy_evolution_hash": "sha256:ddd...",
    "expected_per_cycle_trace_hashes": {
      "0": "sha256:cycle0...",
      "1": "sha256:cycle1..."
    }
  }
}
```
