# Evidence Bundle Schemas

This directory centralizes the JSON Schemas for the evidence artifacts that ship with the Pre-Launch Review. Each artifact key maps to a concrete schema file that downstream services can use to validate payloads before they land in long-term storage.

## Artifact Index

| Artifact key | Description | Schema | Status |
| --- | --- | --- | --- |
| `first_light_synthetic_raw` | Raw capture of the P3 first-light synthetic experiment, including capture metadata and the unprocessed payload blob. | `first_light_synthetic_raw.schema.json` | Draft 07 (JSONL items) |
| `first_light_red_flag_matrix` | Structured roll-up of any first-light red flags observed during P3, bucketed by subsystem. | `first_light_red_flag_matrix.schema.json` | Draft 07 (envelope) |
| `stability_report` | Human-in-the-loop stability write-up aggregating sustained run telemetry and subjective review notes. | _TBD (`stability_report.schema.json`)_ | Not yet authored |
| `red_flag_matrix` | Canonical cross-phase red flag report (non first-light specific) summarizing gating blockers. | _TBD (`red_flag_matrix.schema.json`)_ | Not yet authored |
| `p4_divergence_log` | Per-run divergence journal for P4 that captures root cause metadata and mitigation steps. | `p4_divergence_log.schema.json` | Draft 07 (JSONL items) |
| `p3_pathology` | Optional annotation describing synthetic pathology stress used in P3 runs (manifest evidence/governance blocks). | `p3_pathology.schema.json` | Draft 07 (optional) |

## First Light Artifact Map

Use the following mappings when validating First Light bundles:

| Artifact file | Schema | Notes |
| --- | --- | --- |
| `synthetic_raw.jsonl` | `first_light_synthetic_raw.schema.json` | JSONL per-cycle records; validator treats the file as an array of entries. |
| `red_flag_matrix.json` | `first_light_red_flag_matrix.schema.json` | JSON summary roll-up for the matching P3 run. |
| `divergence_log.jsonl` | `p4_divergence_log.schema.json` | JSONL shadow-run divergence records for P4. |

Sample payloads for quick validation live under `results/test_harness/fl_20251211_042514_seed42/` (P3 synthetic) and `results/test_p4/p4_20251211_043607/` (P4 divergence).

## Naming and Conventions

- Schemas target JSON Schema Draft 07 and live beside the README for easy discovery.
- Artifact keys match the payload identifiers used in the evidence bundle manifest.
- Required properties should be reserved for data that is emitted by *every* evidence producer; optional envelope fields belong under `properties` but outside `required`.

## Validation Helper

Use `tools/evidence_schema_check.py` to quickly sanity check a payload file against its schema before distributing a review bundle:

```powershell
uv run python tools/evidence_schema_check.py `
  results\test_harness\fl_20251211_042514_seed42\synthetic_raw.jsonl `
  schemas\evidence\first_light_synthetic_raw.schema.json
```

The helper emits a boolean result and prints any validation errors to stderr. Once the helper is wired into CI we can keep these artifacts consistent across services.
