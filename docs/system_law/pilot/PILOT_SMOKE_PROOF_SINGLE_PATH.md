# PILOT SMOKE PROOF — Single Path

**Status:** REFERENCE
**Mode:** SHADOW (observational only)
**Purpose:** Technical debrief / follow-up demonstration

---

## What This Demonstrates

| Demonstrates | Does NOT Demonstrate |
|--------------|----------------------|
| Code path execution | Validation of any kind |
| Ingestion mechanics | System correctness |
| Manifest attachment | Learning or adaptation |
| File copy behavior | External-ready artifacts |
| Determinism (timestamp-stripped) | Gating readiness |

This is a **single-path smoke proof** for internal technical verification.
It exercises the pilot external ingest adapter under SHADOW MODE.

---

## Copy/Paste Command Sequence

Run from repository root:

```bash
uv run python -c "
import json
import tempfile
from pathlib import Path
from backend.health.pilot_external_ingest_adapter import (
    ingest_external_log,
    wrap_for_evidence_pack,
    attach_to_manifest,
    copy_to_evidence_pack,
    PilotIngestResult,
)

with tempfile.TemporaryDirectory() as t:
    tmp = Path(t)

    # Create sample external log (JSON)
    log_json = tmp / 'sample_external.json'
    log_json.write_text(json.dumps({
        'log_type': 'pilot_demo',
        'entries': [{'event': 'smoke', 'value': 1}]
    }))

    # Create dummy evidence pack
    pack = tmp / 'evidence_pack'
    pack.mkdir()

    # Ingest
    r = ingest_external_log(log_json)

    # Wrap for evidence pack
    e = wrap_for_evidence_pack(r, log_json)

    # Attach to manifest
    m = attach_to_manifest({'governance': {}}, [e])

    # Copy to evidence pack
    copied = copy_to_evidence_pack(log_json, pack, 'external')

    # Output
    entry_count = m['governance']['external_pilot']['entry_count']
    mode = m['governance']['external_pilot']['mode']
    action = m['governance']['external_pilot']['action']
    print(f'SMOKE: result={r[\"result\"]} mode={mode} action={action} entries={entry_count}')
"
```

---

## Expected Output

```
SMOKE: result=SUCCESS mode=SHADOW action=LOGGED_ONLY entries=1
```

Single line. Neutral language. No assertions about system state.

---

## Non-Interference Statement

The pilot ingestion adapter **must not touch**:

| Path | Constraint |
|------|------------|
| `p3/` | No reads, no writes |
| `p4/` | No reads, no writes |
| `governance/` (existing files) | No modifications |
| `results/cal_exp_*` | No writes |
| CAL-EXP harness scripts | No imports, no modifications |

All pilot outputs are isolated under:
- `evidence_pack/external/` (file copies)
- `manifest.governance.external_pilot` (metadata)

---

## Determinism Statement

Repeated execution produces **byte-identical outputs** after stripping time-related keys:

- `timestamp`
- `ingested_at`
- `created_at`

This is **content-level determinism**, not correctness verification.

Verification:
```python
def strip_time(d):
    if isinstance(d, dict):
        return {k: strip_time(v) for k, v in d.items()
                if 'time' not in k.lower() and k != 'ingested_at'}
    elif isinstance(d, list):
        return [strip_time(x) for x in d]
    return d

# Two runs → strip_time(run1) == strip_time(run2)
```

---

## Related Documents

| Document | Path |
|----------|------|
| Pilot Index | [PILOT_INDEX.md](PILOT_INDEX.md) |
| Pilot Authorization | [PILOT_AUTHORIZATION.md](PILOT_AUTHORIZATION.md) |
| Pilot Contract Posture | [PILOT_CONTRACT_POSTURE.md](PILOT_CONTRACT_POSTURE.md) |
| Pilot Toolchain Schema | [pilot_toolchain_manifest.schema.json](../../../schemas/pilot_toolchain_manifest.schema.json) |
| Pilot Provenance Note | [PILOT_TOOLCHAIN_PROVENANCE_NOTE.md](PILOT_TOOLCHAIN_PROVENANCE_NOTE.md) |

---

## Troubleshooting

- **`ModuleNotFoundError`**: Run from repository root with `uv run python -c "..."`. Ensure `uv sync` has been executed.
- **`PilotIngestResult.SCHEMA_INVALID`**: External log missing required `log_type` field.
- **Empty `external_pilot` section**: Check that `wrap_for_evidence_pack()` returned `valid: True` before passing to `attach_to_manifest()`.

---

*SHADOW MODE — observational only. This document is for technical reference, not external communication.*
