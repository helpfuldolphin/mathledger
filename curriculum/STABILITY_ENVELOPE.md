# Curriculum Stability Envelope

## Overview

The Curriculum Stability Envelope provides forward-looking consistency guarantees for curriculum changes. It extends the curriculum drift radar by introducing automated fingerprinting, invariant validation, and promotion guards.

## Features

### 1. Curriculum Fingerprinting

Compute canonical fingerprints of curriculum systems for comparison and change detection:

```python
from curriculum import compute_fingerprint, compute_fingerprint_diff
from curriculum.gates import load

# Load and fingerprint a curriculum
system = load("pl")
fingerprint = compute_fingerprint(system)

# Compare two fingerprints
diff = compute_fingerprint_diff(fingerprint_a, fingerprint_b)

# Check what changed
if diff.has_changes:
    print(f"Changed slices: {diff.changed_slices}")
    print(f"Parameter diffs: {diff.param_diffs}")
    print(f"Gate diffs: {diff.gate_diffs}")
```

### 2. Invariant Validation

Validate curriculum invariants automatically:

```python
from curriculum import validate_curriculum_invariants
from curriculum.gates import load

system = load("pl")
report = validate_curriculum_invariants(system)

if report.valid:
    print("✅ All invariants validated")
else:
    print("❌ Validation failed:")
    for error in report.errors:
        print(f"  - {error}")
```

**Validated Invariants:**
- Slice naming constraints (slug-safe, no whitespace, max length)
- Parameter intervals (positive depth, breadth, total_max)
- Gate threshold bounds (coverage CI in (0,1], velocity > 0, etc.)
- Gate threshold monotonicity (warnings for increasing difficulty)

### 3. Promotion Guard

Block promotions that violate stability constraints:

```python
from curriculum import evaluate_curriculum_stability, validate_curriculum_invariants
from curriculum.gates import load

# Load current and proposed curriculums
current_system = load("pl")
proposed_system = load("pl")  # After modifications

# Compute fingerprints
current_fp = compute_fingerprint(current_system)
proposed_fp = compute_fingerprint(proposed_system)

# Validate invariants
invariants = validate_curriculum_invariants(proposed_system)

# Evaluate stability
stability = evaluate_curriculum_stability(
    current_fp,
    proposed_fp,
    invariants,
    max_slice_changes=3,
    max_gate_change_pct=10.0
)

if stability.allow_promotion:
    print(f"✅ Safe to promote: {stability.reason}")
else:
    print(f"❌ Promotion blocked: {stability.reason}")
```

**Promotion Blocked When:**
- More than N slices changed at once (default: 3)
- Gate thresholds changed by >10%
- Any slice was removed or renamed
- Invariant regressions occurred

## CLI Usage

### Validate Invariants

```bash
python -m curriculum.cli validate-invariants --system pl
```

Output:
```
Curriculum Invariant Validation: pl
Valid: True

Warnings:
  ⚠️  Slice 'slice_b' has higher coverage CI than previous slice

✅ All invariants validated successfully
```

### Check Stability Envelope

```bash
python -m curriculum.cli stability-envelope \
  --system pl \
  --baseline baseline_fp.json \
  --save-fingerprint current_fp.json \
  --max-slice-changes 3 \
  --max-gate-change-pct 10.0
```

Output:
```
Curriculum Stability Envelope: pl
Allow Promotion: True
Reason: Curriculum changes within stability envelope (2 slices changed)
Fingerprint Changes: 2

📝 Fingerprint saved to current_fp.json

✅ Curriculum is stable for promotion
```

### Diff Fingerprints

```bash
python -m curriculum.cli diff-fingerprint before.json after.json --json
```

Output:
```
Fingerprint Diff: before.json -> after.json

Changed Slices: slice_a, slice_b

Parameter Diffs:
  slice_a:
    atoms: 3 -> 4
    depth_max: 4 -> 5

Gate Diffs:
  slice_b:
    coverage.ci_lower_min: 0.9 -> 0.92

JSON Output:
{
  "changed_slices": ["slice_a", "slice_b"],
  "param_diffs": {...},
  "gate_diffs": {...}
}
```

## Integration with CI

Add to your CI pipeline to enforce curriculum stability:

```yaml
# .github/workflows/curriculum-stability.yml
name: Curriculum Stability Check

on:
  pull_request:
    paths:
      - 'config/curriculum.yaml'
      - 'config/curriculum_*.yaml'

jobs:
  stability:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Validate Curriculum Invariants
        run: |
          python -m curriculum.cli validate-invariants --system pl
        
      - name: Check Stability Envelope
        run: |
          # Download baseline fingerprint from main branch
          git fetch origin main
          git show origin/main:artifacts/curriculum_baseline.json > baseline.json
          
          # Generate current fingerprint and check stability
          python -m curriculum.cli stability-envelope \
            --system pl \
            --baseline baseline.json \
            --save-fingerprint current.json \
            --max-slice-changes 3 \
            --max-gate-change-pct 10.0
          
          # Upload current fingerprint for comparison
          # (In real CI, you'd upload this as an artifact)
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_curriculum_stability_envelope.py -v
```

Tests cover:
- Fingerprint computation and normalization
- Fingerprint diffing (added/removed/changed slices)
- Invariant validation (naming, intervals, thresholds)
- Promotion guard logic
- CLI integration and exit codes
- Mixed drift + invariant violations

## API Reference

### `compute_fingerprint(system: CurriculumSystem) -> Dict[str, Any]`

Compute a canonical fingerprint of a curriculum system with sorted slices, normalized parameters, and sorted gate specifications.

### `compute_fingerprint_diff(a: Dict, b: Dict) -> FingerprintDiff`

Compute the difference between two fingerprints, detecting:
- Added/removed slices
- Parameter changes
- Gate threshold changes
- Invariant-level changes

### `validate_curriculum_invariants(system: CurriculumSystem) -> CurriculumInvariantReport`

Validate curriculum invariants including:
- Slice naming constraints
- Parameter interval monotonicity
- Gate threshold bounds and monotonicity

### `evaluate_curriculum_stability(...) -> PromotionStabilityReport`

Evaluate curriculum stability for promotion decisions, blocking promotions that exceed change thresholds or violate invariants.

## Design Principles

1. **Canonical Normalization**: All data structures are normalized (sorted keys, consistent dtypes) to ensure stable fingerprints
2. **Forward-Looking**: Validates proposed changes before they're committed
3. **Fail-Safe**: Blocks promotions when stability constraints are violated
4. **Observable**: Provides detailed reports of what changed and why promotion was blocked
5. **Composable**: Functions can be used independently or chained together

## Future Enhancements

- [ ] Automatic baseline fingerprint updates on successful promotions
- [ ] Integration with curriculum version control
- [ ] Historical stability trend analysis
- [ ] Automated rollback on stability violations
- [ ] Slack/email notifications for blocked promotions
