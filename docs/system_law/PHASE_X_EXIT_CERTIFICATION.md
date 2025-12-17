# PHASE X EXIT CERTIFICATION

**Document Version:** 1.0.0
**Status:** CANONICAL
**Owner:** CLAUDE V (Gatekeeper)

---

## 1. Test Suite Requirements

### 1.1 Command

```bash
uv run pytest \
    tests/ci/test_shadow_audit_sentinel.py \
    tests/ci/test_shadow_audit_guardrails.py \
    tests/integration/test_shadow_audit_e2e.py \
    -v --tb=short
```

### 1.2 Pass Criteria

| Metric | Required |
|--------|----------|
| Passed | 21 |
| Failed | 0 |
| Skipped | 0 |

**FAIL if any test fails or is skipped.**

---

## 2. Demo Output Requirements

### 2.1 Command

```bash
# Create input
mkdir -p results/phase_x_verify/input
echo '{"_header":true,"mode":"SHADOW","schema_version":"1.0.0"}' > results/phase_x_verify/input/shadow_log.jsonl

# Run demo
uv run python scripts/run_shadow_audit.py \
    --input results/phase_x_verify/input \
    --output results/phase_x_verify/output \
    --seed 42
```

### 2.2 Required Artifacts

| File | Location | Required |
|------|----------|----------|
| `run_summary.json` | `results/phase_x_verify/output/sha_42_*/` | YES |
| `first_light_status.json` | `results/phase_x_verify/output/sha_42_*/` | YES |

### 2.3 Verification

```bash
# Both files must exist
ls results/phase_x_verify/output/sha_42_*/run_summary.json
ls results/phase_x_verify/output/sha_42_*/first_light_status.json

# Exit code must be 0
echo "Exit: $?"
```

**FAIL if either file is missing or exit code is non-zero.**

---

## 3. Invariant Requirements

### 3.1 JSON Field Invariants

Every output JSON file MUST contain:

| Field | Required Value | Check Command |
|-------|----------------|---------------|
| `mode` | `"SHADOW"` | `grep '"mode": "SHADOW"' <file>` |
| `schema_version` | `"1.0.0"` | `grep '"schema_version": "1.0.0"' <file>` |

### 3.2 Shadow Mode Compliance

`run_summary.json` MUST contain:

```json
"shadow_mode_compliance": {
  "observational_only": true,
  "no_enforcement": true
}
```

### 3.3 Verification Command

```bash
uv run python -c "
import json
from pathlib import Path

out = list(Path('results/phase_x_verify/output').glob('sha_42_*'))[0]

# Check run_summary.json
rs = json.loads((out / 'run_summary.json').read_text())
assert rs['mode'] == 'SHADOW', 'mode != SHADOW'
assert rs['schema_version'] == '1.0.0', 'schema_version != 1.0.0'
assert rs['shadow_mode_compliance']['no_enforcement'] == True, 'enforcement enabled'

# Check first_light_status.json
fl = json.loads((out / 'first_light_status.json').read_text())
assert fl['mode'] == 'SHADOW', 'first_light mode != SHADOW'
assert fl['schema_version'] == '1.0.0', 'first_light schema != 1.0.0'

print('INVARIANTS: PASS')
"
```

**FAIL if any assertion fails.**

---

## 4. CLI Invariants

### 4.1 Required Flags

```bash
uv run python scripts/run_shadow_audit.py --help
```

Output MUST contain:

| Flag | Type |
|------|------|
| `--input` | Required |
| `--output` | Required |
| `--seed` | Optional |
| `--verbose` / `-v` | Optional |
| `--dry-run` | Optional |

### 4.2 Forbidden Flags

Output MUST NOT contain:

| Flag | Reason |
|------|--------|
| `--p3-dir` | Non-canonical |
| `--p4-dir` | Non-canonical |
| `--output-dir` | Non-canonical |
| `--deterministic` | Non-canonical |

### 4.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Ran to completion (success or warnings) |
| 1 | Fatal error (missing input, crash) |

**FAIL if forbidden flags are present or exit codes deviate.**

---

## 5. Pass/Fail Determination

### PASS (All must be true)

- [ ] Test suite: 21 passed, 0 failed, 0 skipped
- [ ] Demo command exits 0
- [ ] `run_summary.json` exists with correct invariants
- [ ] `first_light_status.json` exists with correct invariants
- [ ] `mode="SHADOW"` in all outputs
- [ ] `schema_version="1.0.0"` in all outputs
- [ ] `shadow_mode_compliance.no_enforcement=true`
- [ ] CLI uses canonical flags only
- [ ] No forbidden flags in `--help` output

### FAIL (Any of these)

- Any test fails or is skipped
- Demo command exits non-zero
- Required artifacts missing
- `mode` is not `"SHADOW"`
- `schema_version` is not `"1.0.0"`
- `no_enforcement` is not `true`
- Forbidden CLI flags present

---

## 6. One-Command Verification

```bash
uv run pytest \
    tests/ci/test_shadow_audit_sentinel.py \
    tests/ci/test_shadow_audit_guardrails.py \
    tests/integration/test_shadow_audit_e2e.py \
    -v --tb=short && echo "PHASE X: PASS" || echo "PHASE X: FAIL"
```

**Output must end with:** `PHASE X: PASS`

---

## 7. Certification Statement

```
PHASE X EXIT CERTIFICATION

Date: _______________
Verifier: _______________

Test Suite:     [ ] 21 passed, 0 failed
Demo Artifacts: [ ] run_summary.json present
                [ ] first_light_status.json present
Invariants:     [ ] mode=SHADOW
                [ ] schema_version=1.0.0
                [ ] no_enforcement=true
CLI:            [ ] Canonical flags only

Result: [ ] PASS  [ ] FAIL

Signature: _______________
```

---

**END OF CERTIFICATION**
