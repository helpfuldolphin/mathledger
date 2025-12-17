# Pilot External Ingest - SHADOW MODE

**Status**: PILOT (CAL-EXP-3 SCOPE)
**Schema Version**: 1.0.0
**Mode**: SHADOW (LOGGED_ONLY)

---

## How External Logs Enter SHADOW MODE

This document describes how external parties can supply log artifacts that are
wrapped into a First Light evidence pack WITHOUT changing any existing schema.

### Contract

| Property | Value |
|----------|-------|
| Mode | SHADOW |
| Action | LOGGED_ONLY |
| Gating | NONE |
| Schema Changes | NONE |
| CAL-EXP-2 Interference | NONE |

---

## Quick Start

### 1. Prepare External Log (JSON)

External logs must have a `log_type` field:

```json
{
  "log_type": "runtime_metrics",
  "timestamp": "2025-12-13T10:00:00Z",
  "entries": [
    {"metric": "cpu_usage", "value": 45.2},
    {"metric": "memory_mb", "value": 1024}
  ],
  "metadata": {
    "source": "pilot_system_a"
  }
}
```

### 2. Ingest and Validate

```python
from pathlib import Path
from backend.health.pilot_external_ingest_adapter import (
    ingest_external_log,
    wrap_for_evidence_pack,
    attach_to_manifest,
    copy_to_evidence_pack,
    PilotIngestResult,
)

# Ingest external log
result = ingest_external_log(Path("external_metrics.json"))

if result["result"] == PilotIngestResult.SUCCESS:
    print(f"Ingested: {result['source_type']}")
    print(f"SHA256: {result['sha256']}")
    print(f"Mode: {result['mode']}")  # Always "SHADOW"
```

### 3. Attach to Evidence Pack

```python
import json

# Load existing manifest
with open("evidence_pack/manifest.json") as f:
    manifest = json.load(f)

# Wrap and attach
entry = wrap_for_evidence_pack(result, Path("external_metrics.json"))
new_manifest = attach_to_manifest(manifest, [entry])

# Copy file to evidence pack
copy_to_evidence_pack(
    Path("external_metrics.json"),
    Path("evidence_pack"),
    target_subdir="external",
)

# Save updated manifest
with open("evidence_pack/manifest.json", "w") as f:
    json.dump(new_manifest, f, indent=2)
```

---

## Supported Formats

| Format | Extension | Source Type |
|--------|-----------|-------------|
| JSON | `.json` | EXTERNAL_JSON |
| JSONL | `.jsonl` | EXTERNAL_JSONL |

---

## Schema Requirements

### Minimal (Required)

```json
{
  "log_type": "string"  // REQUIRED: Identifies log category
}
```

### Recognized (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `log_type` | string | Log category identifier |
| `timestamp` | string | ISO 8601 timestamp |
| `entries` | array | Log entries |
| `metadata` | object | Source metadata |
| `source` | string | Source system identifier |
| `version` | string | Log schema version |

Unrecognized fields produce warnings but do not fail validation.

---

## Manifest Structure

External logs are attached under `governance.external_pilot`:

```json
{
  "governance": {
    "p5_calibration": { ... },           // Existing - UNTOUCHED
    "rtts_validation_reference": { ... }, // Existing - UNTOUCHED
    "external_pilot": {                   // NEW - Pilot section only
      "schema_version": "1.0.0",
      "mode": "SHADOW",
      "action": "LOGGED_ONLY",
      "entries": [
        {
          "path": "external/metrics.json",
          "sha256": "abc123...",
          "pilot_metadata": {
            "source_type": "EXTERNAL_JSON",
            "extraction_source": "EXTERNAL_PILOT",
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "action": "LOGGED_ONLY",
            "ingested_at": "2025-12-13T10:00:00Z"
          }
        }
      ],
      "entry_count": 1,
      "invalid_count": 0,
      "warnings": []
    }
  }
}
```

---

## Evidence Pack Tree

After ingestion, the evidence pack structure:

```
evidence_pack/
├── manifest.json          # Updated with external_pilot section
├── governance/
│   └── rtts_validation.json  # Existing - UNTOUCHED
├── external/              # NEW - Pilot artifacts only
│   └── metrics.json       # Copied external log
├── p3/                    # Existing - UNTOUCHED
└── p4/                    # Existing - UNTOUCHED
```

---

## Integrity Verification

Optional SHA256 verification:

```python
from backend.health.pilot_external_ingest_adapter import (
    compute_file_sha256,
    ingest_external_log,
    PilotIngestResult,
)

# Compute expected hash
expected_sha256 = compute_file_sha256(Path("external_metrics.json"))

# Ingest with integrity check
result = ingest_external_log(
    Path("external_metrics.json"),
    expected_sha256=expected_sha256,
)

if result["result"] == PilotIngestResult.INTEGRITY_MISMATCH:
    print("File modified since hash computed!")
```

---

## Result Codes

| Code | Meaning |
|------|---------|
| `SUCCESS` | Ingestion successful |
| `SCHEMA_INVALID` | Missing required `log_type` field |
| `FILE_NOT_FOUND` | Source file does not exist |
| `PARSE_ERROR` | JSON parsing failed |
| `INTEGRITY_MISMATCH` | SHA256 verification failed |

---

## Non-Interference Guarantees

The pilot adapter is designed for strict non-interference:

1. **No CAL-EXP-2 imports**: Does not import any CAL-EXP-2 frozen modules
2. **No schema changes**: Existing manifest fields are never modified
3. **No metric creation**: Does not create new governance metrics
4. **Additive only**: Only adds `governance.external_pilot` section
5. **Independent versioning**: Uses separate schema version (1.0.0)

### Verification

```python
# Verify non-interference
assert new_manifest["governance"]["rtts_validation_reference"] == \
       original_manifest["governance"]["rtts_validation_reference"]
assert new_manifest["governance"]["p5_calibration"] == \
       original_manifest["governance"]["p5_calibration"]
```

---

## Smoke Test

One-liner to verify pilot ingestion works:

```bash
uv run python -c "
from backend.health.pilot_external_ingest_adapter import (
    ingest_external_log, wrap_for_evidence_pack, attach_to_manifest,
    PilotIngestResult
)
from pathlib import Path; import tempfile, json
with tempfile.TemporaryDirectory() as t:
    p = Path(t)
    log = p / 'test.json'
    log.write_text(json.dumps({'log_type': 'test', 'entries': []}))
    r = ingest_external_log(log)
    assert r['result'] == PilotIngestResult.SUCCESS
    assert r['mode'] == 'SHADOW'
    e = wrap_for_evidence_pack(r, log)
    m = attach_to_manifest({'governance': {}}, [e])
    assert 'external_pilot' in m['governance']
    print(f'SMOKE: result={r[\"result\"]} mode={r[\"mode\"]} entry_count={m[\"governance\"][\"external_pilot\"][\"entry_count\"]}')"
```

Expected output:
```
SMOKE: result=SUCCESS mode=SHADOW entry_count=1
```

---

## Scope Fence

| Constraint | Binding |
|------------|---------|
| Cannot gate | MUST NEVER be used for gating decisions |
| Cannot create metrics | No new metrics emitted |
| Cannot modify CAL-EXP-2 | Frozen code paths untouched |
| Shadow only | All operations LOGGED_ONLY |

---

*This document is part of CAL-EXP-3 pilot scope. Subject to revision.*
