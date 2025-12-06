# Mypy CI Guide for Experiments Package

This document describes the mypy type checking setup for the `experiments/` package.

## Overview

The `experiments/` package now enforces strict type checking via mypy to ensure:
- Type safety in experimental code
- Early detection of type-related bugs
- Better IDE support and code navigation
- Documentation through types

## Configuration

Mypy configuration is located in `pyproject.toml` under `[tool.mypy]`.

### Checked Modules

The following modules have strict type checking enabled:
- `experiments.u2.*` - All U2 runner modules
- `experiments.u2_safe_eval` - Safe evaluation module
- `experiments.u2_calibration` - Calibration utilities
- `experiments.run_uplift_u2` - Main uplift runner script

### Type Checking Rules

For the experiments package, we enforce:
- `disallow_untyped_defs` - All functions must have type annotations
- `disallow_incomplete_defs` - Function signatures must be complete
- `warn_return_any` - Warn when returning `Any` from typed function
- `no_implicit_optional` - Explicitly mark `Optional` types
- `warn_redundant_casts` - Catch unnecessary type casts
- `warn_unused_ignores` - Detect unused `# type: ignore` comments

### Ignored Modules

The following modules are ignored due to missing type stubs:
- `rfl.*` - RFL package (legacy)
- `backend.*` - Backend services (legacy)
- Other internal packages without type annotations

## Running Mypy Locally

To run mypy on the experiments package:

```bash
# Check specific files
python3 -m mypy experiments/run_uplift_u2.py experiments/u2_safe_eval.py experiments/u2/*.py

# Check entire experiments directory
python3 -m mypy experiments/

# Install missing type stubs automatically
python3 -m mypy --install-types experiments/
```

## CI Integration

### GitHub Actions Workflow

Add the following step to your CI workflow (`.github/workflows/ci.yml`):

```yaml
- name: Type Check with mypy
  run: |
    pip install mypy types-PyYAML
    python3 -m mypy experiments/run_uplift_u2.py experiments/u2_safe_eval.py experiments/u2/*.py
```

### Pre-commit Hook

To run mypy before each commit, add to `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.9.0
  hooks:
    - id: mypy
      files: ^experiments/
      additional_dependencies:
        - types-PyYAML
```

## Type Stub Dependencies

Required type stub packages:
- `types-PyYAML` - For YAML parsing
- Additional stubs may be needed as dependencies evolve

Install via:
```bash
pip install types-PyYAML
```

## Common Type Issues and Solutions

### Issue: "Function is missing a return type annotation"

```python
# Bad
def my_function(x: int):
    return x * 2

# Good
def my_function(x: int) -> int:
    return x * 2
```

### Issue: "Missing type parameters for generic type"

```python
# Bad
def get_items() -> set:
    return set()

# Good
def get_items() -> set[str]:
    return set()
```

### Issue: "Returning Any from function"

```python
# Bad
def load_config() -> Dict[str, Any]:
    return yaml.safe_load(f)  # yaml.safe_load returns Any

# Good
def load_config() -> Dict[str, Any]:
    config: Dict[str, Any] = yaml.safe_load(f)
    return config
```

### Issue: "Library stubs not installed"

```bash
# Install missing stubs
python3 -m pip install types-<library-name>

# Example
python3 -m pip install types-PyYAML
```

## Adjusting Thresholds

If mypy rules are too strict for legacy code:

1. Add module-specific overrides in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = "experiments.legacy_module"
disallow_untyped_defs = false
```

2. Use `# type: ignore` comments sparingly:

```python
result = legacy_function()  # type: ignore[no-untyped-call]
```

## Updating This Guide

When CI hardware or Python versions change:
1. Update the `python_version` in `[tool.mypy]`
2. Test locally before updating CI
3. Document changes in this guide

## Resources

- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Type Hints PEP 484](https://peps.python.org/pep-0484/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
