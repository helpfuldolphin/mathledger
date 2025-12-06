# Phase II U2 Developer Guide

> **STATUS: PHASE II — NOT YET RUN. NO UPLIFT CLAIMS MAY BE MADE.**

This guide covers developer workflows and tooling for Phase II U2 uplift experiments.

## Table of Contents

1. [Curriculum Loading](#curriculum-loading)
2. [Calibration-First Workflow](#calibration-first-workflow)
3. [Verbose Cycles Debugging](#verbose-cycles-debugging)
4. [Testing](#testing)

---

## Curriculum Loading

### Overview

Phase II experiments use `curriculum_loader_v2` for deterministic curriculum loading. This ensures:

- **Reproducible ordering**: Same slice → same item order across runs
- **Clear error handling**: Missing/malformed configs produce explicit errors
- **Format flexibility**: Supports YAML, JSON, and JSONL

### Usage

#### Load Curriculum Programmatically

```python
from experiments.curriculum_loader_v2 import CurriculumLoader

# Initialize loader
loader = CurriculumLoader("config/curriculum_uplift_phase2.yaml")

# Load items for a slice
items = loader.load_for_slice("slice_uplift_goal")
print(f"Loaded {len(items)} items")

# Get full slice config
config = loader.get_slice_config("slice_uplift_goal")
print(config["description"])

# List available slices
slices = loader.list_slices()
print(f"Available slices: {', '.join(slices)}")
```

#### Curriculum File Format

The curriculum config follows this structure:

```yaml
version: "2.1.0"

slices:
  slice_name:
    description: "Slice description"
    parameters:
      atoms: 4
      depth_min: 2
      depth_max: 5
      # ... other parameters
    
    formula_pool_entries:
      - "p"
      - "q"
      - "p->q"
      # ... more formulas
```

#### Error Handling

The loader provides specific exceptions:

- **`CurriculumNotFoundError`**: Config file or slice not found
- **`CurriculumFormatError`**: Malformed YAML/JSON or missing required fields
- **`CurriculumLoaderError`**: General loading error

```python
from experiments.curriculum_loader_v2 import (
    CurriculumLoader,
    CurriculumNotFoundError,
    CurriculumFormatError,
)

try:
    loader = CurriculumLoader("config/curriculum.yaml")
    items = loader.load_for_slice("my_slice")
except CurriculumNotFoundError as e:
    print(f"Slice not found: {e}")
except CurriculumFormatError as e:
    print(f"Config malformed: {e}")
```

### Integration with run_uplift_u2.py

The curriculum loader is automatically integrated into `run_uplift_u2.py`. Items are loaded deterministically when you specify a `--slice`:

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --cycles 100 \
  --seed 42 \
  --mode baseline \
  --out results/test_run \
  --config config/curriculum_uplift_phase2.yaml
```

If the slice is not found, the runner falls back to test items with a warning.

---

## Calibration-First Workflow

### Overview

Calibration verifies that your experiment setup is deterministic before running expensive uplift comparisons. The `--require-calibration` flag enforces this as a prerequisite.

### Calibration Summary Format

Calibration produces a `calibration_summary.json` file:

```json
{
  "slice_name": "slice_uplift_goal",
  "determinism_verified": true,
  "schema_valid": true,
  "replay_hash": "abc123...",
  "baseline_hash": "abc123...",
  "metadata": {}
}
```

**Required fields:**
- `determinism_verified`: Did replay match original run?
- `schema_valid`: Did output conform to expected schema?

### Running with Calibration Requirement

To enforce calibration check before running:

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --cycles 500 \
  --seed 42 \
  --mode rfl \
  --out results/rfl_run \
  --require-calibration
```

**Exit codes:**
- `0`: Success
- `1`: General error (e.g., config not found)
- `2`: Calibration missing or invalid (with `--require-calibration`)

### Calibration Workflow

1. **Run baseline calibration**:
   ```bash
   # Placeholder: actual calibration runner to be implemented
   # For now, manually create calibration_summary.json
   mkdir -p results/uplift_u2/calibration/slice_uplift_goal
   cat > results/uplift_u2/calibration/slice_uplift_goal/calibration_summary.json <<EOF
   {
     "slice_name": "slice_uplift_goal",
     "determinism_verified": true,
     "schema_valid": true,
     "replay_hash": "abc123",
     "baseline_hash": "abc123"
   }
   EOF
   ```

2. **Run main experiment with calibration check**:
   ```bash
   python experiments/run_uplift_u2.py \
     --slice slice_uplift_goal \
     --cycles 500 \
     --seed 42 \
     --mode rfl \
     --out results/rfl_run \
     --require-calibration
   ```

3. **If calibration fails**, the runner exits with code 2:
   ```
   ERROR: Calibration invalid for slice 'slice_uplift_goal'
          determinism check failed
          Re-run calibration to fix.
   ```

### Programmatic Calibration Validation

You can validate calibration in Python:

```python
from pathlib import Path
from experiments.u2_calibration import (
    validate_calibration,
    CalibrationNotFoundError,
    CalibrationInvalidError,
)

calib_dir = Path("results/uplift_u2/calibration")
slice_name = "slice_uplift_goal"

try:
    summary = validate_calibration(calib_dir, slice_name, require_valid=True)
    print(f"Calibration valid: {summary.is_valid()}")
except CalibrationNotFoundError:
    print("Calibration not found - run calibration first")
except CalibrationInvalidError as e:
    print(f"Calibration invalid: {e}")
```

---

## Verbose Cycles Debugging

### Overview

`--verbose-cycles` enables enhanced, machine-parseable cycle-by-cycle logging. Useful for:
- Debugging ordering differences between baseline and RFL
- Tracking policy updates
- Inspecting per-cycle metrics

### Basic Usage

Enable verbose cycles:

```bash
python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --cycles 10 \
  --seed 42 \
  --mode baseline \
  --out results/debug \
  --verbose-cycles
```

**Default output format**:
```
VERBOSE: cycle=1 mode=baseline success=true item=p
VERBOSE: cycle=2 mode=baseline success=false item=q->r
VERBOSE: cycle=3 mode=baseline success=true item=p|~p
```

### Configurable Fields

Customize which fields appear via the `U2_VERBOSE_FIELDS` environment variable:

```bash
export U2_VERBOSE_FIELDS="cycle,mode,success,label,item_hash_prefix"

python experiments/run_uplift_u2.py \
  --slice slice_uplift_goal \
  --cycles 10 \
  --seed 42 \
  --mode rfl \
  --out results/debug \
  --verbose-cycles
```

**Output with custom fields**:
```
VERBOSE: cycle=1 mode=rfl success=true label=PHASE_II item_hash_prefix=abc12345
VERBOSE: cycle=2 mode=rfl success=false label=PHASE_II item_hash_prefix=def67890
```

### Available Fields

| Field | Description | Example |
|-------|-------------|---------|
| `cycle` | Cycle number (1-indexed) | `cycle=1` |
| `mode` | Experiment mode | `mode=baseline` |
| `success` | Whether cycle succeeded | `success=true` |
| `item` | Formula chosen | `item=p->q` |
| `label` | Phase label | `label=PHASE_II` |
| `slice` | Slice name | `slice=slice_uplift_goal` |
| `seed` | Cycle seed | `seed=42` |
| `result` | Result dictionary (string) | `result={'outcome': 'VERIFIED'}` |
| `item_hash_prefix` | First 8 chars of item hash | `item_hash_prefix=abc12345` |

### Machine Parsing

Verbose output uses `key=value` pairs for easy parsing:

```python
import subprocess

proc = subprocess.run(
    ["python", "experiments/run_uplift_u2.py", "--verbose-cycles", ...],
    capture_output=True,
    text=True,
)

for line in proc.stdout.splitlines():
    if line.startswith("VERBOSE:"):
        data = {}
        parts = line.split("VERBOSE: ")[1].split()
        for part in parts:
            key, value = part.split("=", 1)
            data[key] = value
        print(f"Cycle {data['cycle']}: success={data['success']}")
```

### Field Selection Examples

**Minimal (cycle and outcome only)**:
```bash
export U2_VERBOSE_FIELDS="cycle,success"
```

**Extended (full context)**:
```bash
export U2_VERBOSE_FIELDS="cycle,mode,success,item,label,slice,seed"
```

**Debug mode (with hashes)**:
```bash
export U2_VERBOSE_FIELDS="cycle,success,item_hash_prefix,result"
```

---

## Testing

### Running Tests

Run all Phase II U2 tests:

```bash
python -m pytest tests/experiments/ tests/devxp/ -v
```

Run specific test suites:

```bash
# Curriculum loader tests
python -m pytest tests/experiments/test_curriculum_loader_v2.py -v

# Calibration integration tests
python -m pytest tests/devxp/test_u2_calibration_integration.py -v

# Verbose cycles tests
python -m pytest tests/devxp/test_verbose_cycles_configurable.py -v
```

### Test Markers

Tests use these markers:
- `@pytest.mark.unit` — Fast, hermetic unit tests
- `@pytest.mark.integration` — Tests requiring full system setup

### Test Coverage

Current test coverage for Phase II U2 components:

- **curriculum_loader_v2**: 19 tests, 100% pass
- **u2_calibration**: 9 tests, 100% pass (4 skipped - require full u2.runner module)
- **verbose_formatter**: 16 tests, 100% pass

### Writing New Tests

Follow existing patterns:

```python
import pytest
from experiments.curriculum_loader_v2 import CurriculumLoader

@pytest.mark.unit
def test_my_feature():
    """Test description."""
    # Arrange
    loader = CurriculumLoader("config/test.yaml")
    
    # Act
    result = loader.load_for_slice("test_slice")
    
    # Assert
    assert len(result) > 0
```

---

## References

- **Curriculum Config**: `config/curriculum_uplift_phase2.yaml`
- **Phase II Plan**: `docs/PHASE2_RFL_UPLIFT_PLAN.md`
- **U2 Port Plan**: `docs/U2_PORT_PLAN.md`
- **Runner**: `experiments/run_uplift_u2.py`
- **Calibration Module**: `experiments/u2_calibration.py`
- **Verbose Formatter**: `experiments/verbose_formatter.py`

---

## Version History

- **v1.0** (2025-12-06): Initial guide
  - Curriculum loader integration
  - Calibration-first workflow
  - Verbose cycles debugging
