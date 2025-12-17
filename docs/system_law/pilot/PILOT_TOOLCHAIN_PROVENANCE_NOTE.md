# Pilot Toolchain Provenance Note

**Status**: CANONICAL
**Scope**: Descriptive documentation only

---

## What Pilot Provenance Binds

Pilot provenance records the **ingestion-time environment state** for an artifact:

| Bound Element | Description |
|---------------|-------------|
| `uv_lock_hash` | SHA-256 of Python dependency lock file at ingestion |
| `toolchain_fingerprint` | Combined hash of all toolchain files at ingestion |
| `ingestion_timestamp` | UTC timestamp when artifact was processed |
| `toolchain_snapshot` | Python version, uv version, Lean version at ingestion |

Pilot provenance answers: *"What was the toolchain state when this artifact was ingested?"*

---

## What Pilot Provenance Does NOT Bind

Pilot provenance explicitly **does not** establish:

| Not Bound | Explanation |
|-----------|-------------|
| Experimental validity | Pilot artifacts have not undergone CAL-EXP validation |
| Reproducibility | No claim that re-running produces identical results |
| CAL-EXP parity | Pilot provenance is not comparable to experiment provenance |
| Behavioral guarantees | Environment binding does not imply behavioral properties |
| Correctness | Toolchain state does not validate artifact content |

The required disclaimer states this explicitly:

> PILOT PROVENANCE: Binds artifact to toolchain state at ingestion. NOT experiment provenance. Does not imply experimental validity, reproducibility, or parity with CAL-EXP runs.

---

## Usage

### Generate manifest for a pilot artifact

```bash
python scripts/pilot_toolchain_hook.py --artifact-id pilot-001
```

### Generate manifest with source artifact hashing

```bash
python scripts/pilot_toolchain_hook.py --artifact-id pilot-001 --source path/to/artifact.json
```

### Output manifest to stdout as JSON

```bash
python scripts/pilot_toolchain_hook.py --artifact-id pilot-001 --json
```

### Specify custom output directory

```bash
python scripts/pilot_toolchain_hook.py --artifact-id pilot-001 --output manifests/pilot/
```

---

## Common Misread

### Misread

> "Pilot provenance with matching `toolchain_fingerprint` means the artifact has the same validity as CAL-EXP artifacts."

### Correction

Pilot provenance records **when** and **under what toolchain** an artifact was ingested. It does not establish **what** the artifact does or whether it behaves correctly. A matching fingerprint between pilot and CAL-EXP artifacts indicates shared toolchain state, not shared validation status.

Pilot artifacts require separate validation pathways to establish experimental claims.

---

## Schema Reference

- Schema: `schemas/pilot_toolchain_manifest.schema.json`
- Example: `docs/system_law/calibration/audits/pilot_toolchain_manifest_example.json`
- Generator: `scripts/pilot_toolchain_hook.py`

---

## Provenance Level Comparison

| Level | Purpose | Validation Claim |
|-------|---------|------------------|
| `full` | Experiment provenance | All hashes captured at runtime |
| `partial` | Backfilled experiment provenance | Inferred from git evidence |
| `pilot` | Ingestion-time binding | None |

---

## Appendix: Operator Checklist

### Minimum Operator Checklist for Every Ingestion

Before considering a pilot artifact ingested, verify the manifest contains:

- [ ] `schema_version` equals `"1.0.0"`
- [ ] `disclaimer` matches verbatim (see below)
- [ ] `uv_lock_hash` present and non-null
- [ ] `toolchain_fingerprint` present and non-null
- [ ] `source.hash` present if `--source` was provided

### Required Disclaimer (Verbatim)

The `disclaimer` field must contain this exact text:

```json
{
  "disclaimer": "PILOT PROVENANCE: Binds artifact to toolchain state at ingestion. NOT experiment provenance. Does not imply experimental validity, reproducibility, or parity with CAL-EXP runs."
}
```

### Example Manifest Snippet

```json
{
  "schema_version": "1.0.0",
  "artifact_type": "pilot",
  "artifact_id": "pilot-001",
  "ingestion_timestamp": "2025-12-14T04:31:53.258514Z",
  "provenance_level": "pilot",
  "disclaimer": "PILOT PROVENANCE: Binds artifact to toolchain state at ingestion. NOT experiment provenance. Does not imply experimental validity, reproducibility, or parity with CAL-EXP runs.",
  "uv_lock_hash": "d088f20824a5bbc4cd1bf5f02d34a6758752363f417bed1a99970773b8dacfdc",
  "toolchain_fingerprint": "b828a2185e017e172db966d3158e8e2b91b00a37f0cd7de4c4f7cf707130a20a",
  "source": {
    "path": "path/to/artifact.json",
    "hash": "abc123...",
    "size_bytes": 1024
  }
}
```

### Validation Command

To verify a manifest meets checklist requirements:

```bash
python -c "
import json, sys
m = json.load(open(sys.argv[1]))
assert m['schema_version'] == '1.0.0', 'schema_version mismatch'
assert 'NOT experiment provenance' in m['disclaimer'], 'disclaimer mismatch'
assert m['uv_lock_hash'], 'uv_lock_hash missing'
assert m['toolchain_fingerprint'], 'toolchain_fingerprint missing'
print('PASS: Manifest meets operator checklist')
" path/to/manifest.json
```
