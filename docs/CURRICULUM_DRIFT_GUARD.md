# Curriculum Drift Guard — Phase II

**Owner:** `curriculum-architect` agent

The Curriculum Drift Guard is a defensive mechanism that ensures Phase II curriculum configurations remain sound, introspectable, and resistant to silent drift. It provides schema versioning, structural validation, fingerprinting, and drift detection for `config/curriculum_uplift_phase2.yaml`.

## Overview

The Drift Guard consists of three components:

1. **Schema Version & Structural Validator** — Enforces curriculum structure and schema version
2. **Curriculum Fingerprint Generator** — Produces stable cryptographic fingerprints for drift detection
3. **Drift Checker** — Compares current curriculum against expected fingerprints

## Quick Start

### List All Slices

```bash
python -m experiments.curriculum_loader_v2 --list-slices
```

Output:
```
Schema Version: phase2-v1
Slice Count: 4

Slices:
  - slice_uplift_dependency (metric: multi_goal_success)
  - slice_uplift_goal (metric: goal_hit)
  - slice_uplift_sparse (metric: sparse_success)
  - slice_uplift_tree (metric: chain_success)
```

### Show Slice Details

```bash
python -m experiments.curriculum_loader_v2 --show-slice slice_uplift_goal
```

### Show All Success Metrics

```bash
python -m experiments.curriculum_loader_v2 --show-metrics
```

### Generate Curriculum Fingerprint

```bash
# Human-readable output
python -m experiments.curriculum_loader_v2 --fingerprint

# JSON output
python -m experiments.curriculum_loader_v2 --fingerprint --json
```

### Check for Drift

```bash
# Save baseline fingerprint
python -m experiments.curriculum_loader_v2 --fingerprint --json > baseline.json

# Later, check if curriculum has drifted
python -m experiments.curriculum_loader_v2 --check-against baseline.json
# Exit code: 0 = no drift, 1 = drift detected
```

## Component 1: Schema Version & Structural Validator

### Schema Version

`config/curriculum_uplift_phase2.yaml` now includes a `schema_version` field:

```yaml
schema_version: "phase2-v1"
version: 2.1.0

slices:
  slice_uplift_goal:
    # ...
```

**Allowed versions:** `phase2-v1`

### Structural Validation

The validator checks:

- ✅ Schema version is present and allowed
- ✅ Required top-level keys exist (`version`, `slices`)
- ✅ Each slice has required fields (`description`, `parameters`, `success_metric`)
- ✅ Success metric has `kind` field (must match function in `slice_success_metrics.py`)
- ✅ No malformed fields (wrong types, missing required nested keys)

**Error Messages:**

All validation errors are clear and actionable:

```
Curriculum validation failed:
  - Slice 'slice_uplift_goal': missing required field 'success_metric'
  - Slice 'slice_uplift_sparse': 'success_metric.kind' must be non-empty string, got: None
```

### Usage in Code

```python
from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    CurriculumValidationError,
)

try:
    loader = CurriculumLoaderV2.from_default_phase2_config()
    
    # Access slices
    slice_obj = loader.get_slice('slice_uplift_goal')
    print(f"Slice: {slice_obj.name}")
    print(f"Metric: {slice_obj.success_metric.kind}")
    
except FileNotFoundError as e:
    print(f"Config not found: {e}")
except CurriculumValidationError as e:
    print(f"Validation failed:\n{e}")
```

## Component 2: Curriculum Fingerprint Generator

### What is a Curriculum Fingerprint?

A fingerprint is a stable, cryptographic summary of the curriculum configuration:

```json
{
  "schema_version": "phase2-v1",
  "slice_count": 4,
  "metric_kinds": [
    "chain_success",
    "goal_hit",
    "multi_goal_success",
    "sparse_success"
  ],
  "hash": "5af5c1acaad92601fee9ae3c228cc44aa26d0c82d6e34463beff949a971fb025"
}
```

**Properties:**

- **Deterministic:** Same curriculum → same fingerprint hash across runs
- **Canonical:** Slice order doesn't matter (internally sorted before hashing)
- **Sensitive:** Any change to slice parameters, metrics, or structure changes the hash
- **Stable:** SHA-256 hash is cryptographically secure

### What Changes the Hash?

✅ **Changes that update the hash:**

- Adding/removing slices
- Changing slice parameters (atoms, depth, etc.)
- Changing success metric thresholds
- Changing success metric kinds
- Changing formula pool entries
- Changing budget constraints

❌ **Changes that DON'T update the hash:**

- YAML comments
- Field ordering in YAML (canonical representation is used)
- Whitespace or formatting

### Usage in Code

```python
from experiments.curriculum_loader_v2 import (
    CurriculumLoaderV2,
    compute_curriculum_fingerprint,
)

loader = CurriculumLoaderV2.from_default_phase2_config()
fingerprint = compute_curriculum_fingerprint(loader)

print(f"Schema: {fingerprint.schema_version}")
print(f"Slices: {fingerprint.slice_count}")
print(f"Metrics: {fingerprint.metric_kinds}")
print(f"Hash: {fingerprint.hash}")

# Save for later comparison
import json
with open('fingerprint.json', 'w') as f:
    json.dump(fingerprint.to_dict(), f, indent=2)
```

## Component 3: Drift Checker

### What is Drift?

Drift occurs when the curriculum configuration changes in unexpected ways. The drift checker compares:

- Schema version
- Slice count
- Set of metric kinds used
- Full cryptographic hash of all slice details

### Drift Detection

```python
from experiments.curriculum_loader_v2 import (
    CurriculumFingerprint,
    check_drift,
)

# Load expected fingerprint (e.g., from preregistration)
with open('expected_fingerprint.json', 'r') as f:
    expected = CurriculumFingerprint.from_dict(json.load(f))

# Compute current fingerprint
current = compute_curriculum_fingerprint(loader)

# Check for drift
report = check_drift(current, expected)

if report.matches:
    print("✓ No drift detected")
else:
    print("✗ Drift detected:")
    for diff in report.differences:
        print(f"  - {diff}")
```

### Example Drift Reports

**No drift:**
```
✓ Fingerprints match — no drift detected
```

**Slice count changed:**
```
✗ Fingerprints differ:
  - slice_count: expected 4, got 5
  - hash: expected 5af5c1acaad926..., got 7b2f8e9c1d3a45...
```

**Metric kind added:**
```
✗ Fingerprints differ:
  - metric_kinds added: ['new_metric']
  - hash: expected 5af5c1acaad926..., got 7b2f8e9c1d3a45...
```

**Only thresholds changed:**
```
✗ Fingerprints differ:
  - hash mismatch: structure same, but thresholds or details changed
```

## CI/CD Integration

### Preregistration Workflow

1. **Lock curriculum config** before experiments:

```bash
# Generate baseline fingerprint
python -m experiments.curriculum_loader_v2 --fingerprint --json \
  > experiments/prereg/curriculum_fingerprint_u2.json

# Commit to preregistration
git add experiments/prereg/curriculum_fingerprint_u2.json
git commit -m "Preregister curriculum fingerprint for U2"
```

2. **Verify no drift before running experiments:**

```bash
# In CI or before experiment launch
python -m experiments.curriculum_loader_v2 \
  --check-against experiments/prereg/curriculum_fingerprint_u2.json

# Exit code 0 = safe to proceed
# Exit code 1 = drift detected, halt
```

### Evidence Pack Integration

Include curriculum fingerprint in Evidence Packs:

```json
{
  "experiment_id": "u2_run_20250101",
  "curriculum_fingerprint": {
    "schema_version": "phase2-v1",
    "slice_count": 4,
    "metric_kinds": ["goal_hit", "sparse_success", "chain_success", "multi_goal_success"],
    "hash": "5af5c1acaad92601fee9ae3c228cc44aa26d0c82d6e34463beff949a971fb025"
  }
}
```

### GitHub Actions Example

```yaml
name: Curriculum Drift Check

on:
  pull_request:
    paths:
      - 'config/curriculum_uplift_phase2.yaml'

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install pyyaml
      
      - name: Check curriculum drift
        run: |
          python -m experiments.curriculum_loader_v2 \
            --check-against experiments/prereg/curriculum_fingerprint_u2.json
```

## API Reference

### Dataclasses

#### `SuccessMetricSpec`

```python
@dataclass(frozen=True)
class SuccessMetricSpec:
    kind: str                           # Metric function name
    parameters: Dict[str, Any]          # Metric parameters
    target_hashes: Optional[Set[str]]   # Target formula hashes (optional)
```

#### `UpliftSlice`

```python
@dataclass(frozen=True)
class UpliftSlice:
    name: str                           # Slice identifier
    description: str                    # Human-readable description
    parameters: Dict[str, Any]          # Slice parameters
    success_metric: SuccessMetricSpec   # Success metric spec
    uplift: Dict[str, Any]              # Uplift metadata
    budget: Dict[str, Any]              # Budget constraints
    formula_pool_entries: List[str]     # Initial formula pool
```

#### `CurriculumFingerprint`

```python
@dataclass
class CurriculumFingerprint:
    schema_version: str    # Schema version
    slice_count: int       # Number of slices
    metric_kinds: List[str]  # Sorted unique metric kinds
    hash: str              # SHA-256 hash of canonical representation
```

### Classes

#### `CurriculumLoaderV2`

```python
class CurriculumLoaderV2:
    """Phase II curriculum loader with validation and introspection."""
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "CurriculumLoaderV2":
        """Load from YAML file with validation."""
    
    @classmethod
    def from_default_phase2_config(cls) -> "CurriculumLoaderV2":
        """Load from config/curriculum_uplift_phase2.yaml."""
    
    def get_slice(self, slice_name: str) -> Optional[UpliftSlice]:
        """Get a slice by name."""
    
    def list_slice_names(self) -> List[str]:
        """Get sorted list of slice names."""
    
    def get_metric_kinds(self) -> Set[str]:
        """Get set of all metric kinds used."""
```

### Functions

#### `validate_curriculum_structure(raw_config: Dict[str, Any]) -> None`

Validates curriculum structure. Raises `CurriculumValidationError` if invalid.

#### `compute_curriculum_fingerprint(loader: CurriculumLoaderV2) -> CurriculumFingerprint`

Computes stable fingerprint for drift detection.

#### `check_drift(current: CurriculumFingerprint, expected: CurriculumFingerprint) -> DriftReport`

Compares fingerprints and reports differences.

## Testing

Run curriculum drift guard tests:

```bash
# All tests (45 total)
python -m pytest tests/test_curriculum_loader_v2.py tests/test_curriculum_fingerprint.py -v

# Schema validation tests
python -m pytest tests/test_curriculum_loader_v2.py::test_validate_accepts_correct_schema_version -v

# Fingerprint tests
python -m pytest tests/test_curriculum_fingerprint.py::test_fingerprint_hash_is_deterministic -v

# Drift detection tests
python -m pytest tests/test_curriculum_fingerprint.py::test_check_drift_only_hash_differs -v
```

## Guardrails

### Sober Truth Compliance

- ❌ Does NOT claim curriculum produces uplift — only validates structure
- ❌ Does NOT reference Phase I configurations
- ❌ Does NOT modify or write to curriculum files (read-only)
- ✅ Maintains clear Phase II labeling
- ✅ Flags any configuration that lacks preregistration cross-reference
- ✅ Ensures all curriculum changes are explicit and trackable

### Security

- All inputs validated (schema version, field types, etc.)
- No arbitrary code execution (YAML safe_load only)
- Clear error messages prevent information leakage
- CodeQL security scan: **0 alerts**

## Troubleshooting

### `CurriculumValidationError: Missing required field 'schema_version'`

**Solution:** Add `schema_version: "phase2-v1"` to the top of `config/curriculum_uplift_phase2.yaml`.

### `Unsupported schema_version 'phase2-v2'`

**Solution:** Only `phase2-v1` is currently supported. Check spelling and case.

### `FileNotFoundError: Curriculum config not found`

**Solution:** Ensure `config/curriculum_uplift_phase2.yaml` exists in the repository root.

### Drift detected but changes were intentional

**Solution:** This is expected! Update the baseline fingerprint after intentional changes:

```bash
# Regenerate baseline after intentional changes
python -m experiments.curriculum_loader_v2 --fingerprint --json > baseline.json
git add baseline.json
git commit -m "Update curriculum baseline after parameter tuning"
```

## Future Work

- [ ] Support for `phase2-v2` schema (with deprecation warnings for v1)
- [ ] Automated curriculum diff tool (show which slices changed)
- [ ] Integration with Evidence Pack manifest generation
- [ ] Curriculum history tracking (Git-based lineage)
- [ ] Cross-validation with `experiments/slice_success_metrics.py` implementations

## See Also

- `config/curriculum_uplift_phase2.yaml` — primary curriculum config
- `experiments/slice_success_metrics.py` — success metric implementations
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` — preregistration cross-reference
- `docs/PHASE2_RFL_UPLIFT_PLAN.md` — curriculum design document
