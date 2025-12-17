# Shadow Audit: Three-Command Demo

> **NON-NORMATIVE** — This is a quick-start guide, not a specification.
> Canonical CLI/artifact definitions: `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`

**Mode:** SHADOW (advisory only, never blocking)

---

## Prerequisites

```bash
uv sync
```

Generate test input (once):

```bash
python scripts/usla_first_light_harness.py --seed 42 --cycles 100 --output results/demo_input/
```

---

## The Three Commands

### 1. Dry-Run

Validate inputs without writing files.

```bash
python scripts/run_shadow_audit.py \
  --input results/demo_input/ \
  --output results/demo_output/ \
  --dry-run
```

Exit 0 = completed (validated, no files written).

### 2. Minimal Run

Produce two-file evidence bundle.

```bash
python scripts/run_shadow_audit.py \
  --input results/demo_input/ \
  --output results/demo_output/
```

Exit 0 = completed.

### 3. Deterministic Run

Produce output with fixed run_id prefix for audit-grade reproducibility.

```bash
python scripts/run_shadow_audit.py \
  --input results/demo_input/ \
  --output results/demo_output/ \
  --seed 42
```

Exit 0 = completed. JSON content is byte-stable after stripping time keys.

---

## Expected Directory Tree

```
results/demo_output/
└── sha_42_YYYYMMDD_HHMMSS/
    ├── run_summary.json
    └── first_light_status.json
```

Two files. `--seed 42` fixes the `sha_42_` prefix; timestamp reflects execution time.

---

## How to Verify Locally

```bash
python scripts/run_shadow_audit.py --input results/demo_input/ --output results/run_a/ --seed 99
python scripts/run_shadow_audit.py --input results/demo_input/ --output results/run_b/ --seed 99
```

```
results/run_a/sha_99_*/
├── run_summary.json
└── first_light_status.json

results/run_b/sha_99_*/
├── run_summary.json
└── first_light_status.json
```

```python
import json
from pathlib import Path

STRIP_KEYS = {'generated_at', 'timestamp'}

def normalize(p, fname):
    f = next(Path(p).glob('sha_99_*')) / fname
    d = json.loads(f.read_text())
    for k in STRIP_KEYS:
        d.pop(k, None)
    return json.dumps(d, sort_keys=True)

assert normalize('results/run_a', 'run_summary.json') == normalize('results/run_b', 'run_summary.json')
assert normalize('results/run_a', 'first_light_status.json') == normalize('results/run_b', 'first_light_status.json')
print('Byte-identical.')
```

---

*Canonical: `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`*
