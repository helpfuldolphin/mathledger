# Global Health Tile: "Substrate"

This document specifies the schema for the `substrate` entry within the `global_health.json` file. This tile provides a real-time summary of the stability and integrity of the underlying compute substrate.

## Schema

| Field                       | Type            | Description                                                                                             |
| --------------------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| `status_light`              | String          | A single-word status indicator. See mapping rules below.                                                |
| `headline`                  | String          | A concise, human-readable summary of the substrate state.                                               |
| `identity_stability_index`  | Float           | A score from 0.0 to 1.0 indicating confidence in substrate integrity. 1.0 is a perfect match.           |
| `drift_flags`               | Array of String | A list of specific, machine-readable flags indicating the nature of any detected drift.                 |
| `last_attestation_timestamp`| String (ISO 8601)| The timestamp of the last successful attestation measurement.                                           |
| `attestation_provider`      | String          | The service or technology that performed the attestation (e.g., "AWS_NITRO", "AZURE_MAA").             |
| `details_url`               | String          | A link to a detailed report or dashboard for further analysis.                                          |

## Status-Light Mapping Rules

The `status_light` is determined by the `identity_stability_index` and the severity of `drift_flags`.

-   **GREEN**: `identity_stability_index` >= 0.95. The substrate is stable and matches its expected identity signature.
-   **YELLOW**: `identity_stability_index` is between 0.75 and 0.95. Minor, non-critical drift has been detected (e.g., non-security patches).
-   **RED**: `identity_stability_index` < 0.75. Significant drift from the identity baseline has occurred, indicating a potential security or stability risk.
-   **BLOCK**: A critical drift flag has been raised, or the stability index is below a hard failure threshold (e.g., < 0.5). **This status corresponds to a failing CI gate.**

---

## Example JSON Snippets

### Example 1: Fully Stable Substrate

This represents a healthy, verified substrate with no drift. The CI gate is clear.

```json
{
  "substrate": {
    "status_light": "GREEN",
    "headline": "Substrate stable, identity verified.",
    "identity_stability_index": 1.0,
    "drift_flags": [],
    "last_attestation_timestamp": "2025-12-10T14:00:00Z",
    "attestation_provider": "INTERNAL_VIRT_HOST",
    "details_url": "/docs/reports/substrate/latest.html"
  }
}
```

### Example 2: BROKEN (BLOCK) Substrate

This represents a critical failure. The substrate identity has been compromised or has undergone unauthorized changes. This state **must** block deployments and trigger an immediate security alert. This is the view intended for critical infrastructure oversight (e.g., DoD, frontier labs).

```json
{
  "substrate": {
    "status_light": "BLOCK",
    "headline": "CRITICAL: Substrate identity mismatch. CI GATE BLOCKED.",
    "identity_stability_index": 0.42,
    "drift_flags": [
      "KERNEL_VERSION_MISMATCH",
      "UNEXPECTED_FILESYSTEM_HASH",
      "SYSTEM_LIBRARY_PATCHED"
    ],
    "last_attestation_timestamp": "2025-12-10T18:30:00Z",
    "attestation_provider": "INTERNAL_VIRT_HOST",
    "details_url": "/docs/reports/substrate/critical-alert-20251210-1830.html"
  }
}
```
