# Replay Evidence Spine — Artifact Contract

This document specifies the schema and usage of the `replay_governance_snapshot.json` artifact, which is the central output of the replay governance orchestrator.

## Schema: `replay_governance_snapshot.json`

The snapshot file contains a single JSON object with the following top-level fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `artifact_version` | String | The semantic version of this JSON schema. E.g., "1.0.0". |
| `run_id` | String | A unique identifier for the orchestrator run, typically from the CI environment (e.g., GitHub Actions run ID). |
| `timestamp_utc` | String | ISO 8601 timestamp of when the snapshot was generated. |
| `radar_status` | String | The final governance verdict. Enum: "STABLE", "UNSTABLE". |
| `determinism_rate` | Float | The percentage of replay runs that were deterministic. A value between 0.0 and 100.0. |
| `promotion_eval` | Object | An object containing the evaluation for promoting the current candidate. |
| `components` | Array | A list of objects, each representing a component that was part of the replay analysis. |

### `promotion_eval` Object Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `verdict` | String | The final promotion decision. Enum: "BLOCK", "promotion_ok". |
| `reasons` | Array | A list of strings, providing human-readable justifications for the verdict. |

### `components` Array Object Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | String | The unique name of the component (e.g., "basis_alpha", "manus_d"). |
| `determinism_rate` | Float | The component-specific determinism rate. |
| `drift_metric` | Float | A metric quantifying the component's drift from the established baseline. |
| `is_blocking` | Boolean | `true` if this component's state is the primary reason for a "BLOCK" verdict. |

## Artifact Usage Mapping

| Schema Field | CI Usage | Global Health Monitoring | Final Evidence Pack |
| :--- | :--- | :--- | :--- |
| `run_id` | ✅ | ✅ | ✅ |
| `timestamp_utc` | ✅ | ✅ | ✅ |
| `radar_status` | ✅ (Primary gate) | ✅ (Dashboard metric) | ✅ (Core verdict) |
| `determinism_rate` | ✅ (PR comment) | ✅ (Dashboard metric) | ✅ (Core metric) |
| `promotion_eval.verdict` | ✅ (Primary gate) | ❌ | ✅ |
| `promotion_eval.reasons`| ✅ (PR comment) | ❌ | ✅ |
| `components` | ✅ (For debug) | ✅ (Per-component health) | ✅ (Detailed breakdown) |

## Example Snippets

### Example 1: STABLE + promotion_ok

This scenario represents a healthy run where the replay gate passes and the code changes are cleared for promotion.

```json
{
  "artifact_version": "1.0.0",
  "run_id": "github-actions-run-12345",
  "timestamp_utc": "2025-12-10T22:00:00Z",
  "radar_status": "STABLE",
  "determinism_rate": 99.8,
  "promotion_eval": {
    "verdict": "promotion_ok",
    "reasons": [
      "Determinism rate (99.8%) is above threshold (99.5%)",
      "All component drift metrics are within tolerance."
    ]
  },
  "components": [
    {
      "name": "basis_alpha",
      "determinism_rate": 100.0,
      "drift_metric": 0.001,
      "is_blocking": false
    }
  ]
}
```

### Example 2: UNSTABLE + BLOCK

This scenario represents a failing run where significant drift or non-determinism was detected, blocking the change.

```json
{
  "artifact_version": "1.0.0",
  "run_id": "github-actions-run-67890",
  "timestamp_utc": "2025-12-10T23:30:00Z",
  "radar_status": "UNSTABLE",
  "determinism_rate": 85.2,
  "promotion_eval": {
    "verdict": "BLOCK",
    "reasons": [
      "Determinism rate (85.2%) is below threshold (99.5%)",
      "Component 'manus_d' drift metric (0.87) exceeds ceiling (0.10)"
    ]
  },
  "components": [
    {
      "name": "basis_alpha",
      "determinism_rate": 100.0,
      "drift_metric": 0.002,
      "is_blocking": false
    },
    {
      "name": "manus_d",
      "determinism_rate": 70.4,
      "drift_metric": 0.87,
      "is_blocking": true
    }
  ]
}
```
