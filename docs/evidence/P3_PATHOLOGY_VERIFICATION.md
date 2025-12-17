# P3 Pathology Annotations — External Verification Lens

Pathology metadata is **TEST-ONLY** and optional. When a synthetic First-Light run injects a pathology, the harness surfaces it so external verifiers can understand the stress applied without treating it as a gating signal.

## Where the data lives
- `stability_report.json`: `pathology` and `pathology_params` record the selected stress and resolved parameters.
- `manifest.json` (Evidence Pack): `evidence.data.p3_pathology` carries the raw annotation; `governance.p3_pathology` mirrors it with a short expected-effects narrative for auditors.
- `first_light_status.json`: `pathology_used`/`pathology_type` flags reflect the latest P3 run.

## Schema
- `schemas/evidence/p3_pathology.schema.json` (Draft-07) defines the optional structure for both evidence and governance blocks.
- All fields are optional; missing entries imply no pathology (`none`).

## How to interpret
- `type`/`pathology`: one of `spike`, `drift`, `oscillation`.
- `magnitude` and `params`: resolved values used in the harness (e.g., spike magnitude, index; drift slope; oscillation amplitude/period).
- `expected_effects`: human-readable note describing the intended stress (e.g., “Sharp H spike injected as P5 stress probe”).

## Comparing runs
- Treat pathology runs as stress tests: they are not baseline performance and should not be compared directly to non-pathology runs.
- When diffing runs, normalize on `pathology` + key params to ensure like-for-like comparison (e.g., same spike magnitude and index).
- Absence of `p3_pathology` means the run is standard P3 synthetic without forced stress.

## Gating stance
- Pathology annotations are observational; they **must not** be used to gate promotion or compliance checks. They are provided solely to make P5-style stress probes visible to auditors.
