# MathLedger Metrics V1 Linter Implementation

## Summary
Implements a comprehensive metrics linter for MathLedger with edge case validation and ASCII-only error messages for stable CI/CD integration.

## Changes
- **New**: `backend/tools/metrics_linter.py` - Core linter implementation
- **New**: `tests/qa/test_metrics_lint_v1.py` - Comprehensive test suite with edge cases
- **New**: `docs/metrics_linter_message_spec.md` - Message specification document

## Features
- ✅ Empty file handling
- ✅ Single legacy line detection
- ✅ Merkle hash length validation (must be 64 chars)
- ✅ Wall minutes type validation (numeric only, no booleans)
- ✅ Seed type validation (string or integer only)
- ✅ ASCII-only error messages
- ✅ Stable message format for CI/CD
- ✅ Comprehensive edge case coverage

## Edge Cases Tested
1. **Empty file** - Returns `EMPTY_FILE` error
2. **Single legacy line** - Detects legacy format indicators
3. **Merkle length ≠64** - Validates hash field lengths
4. **Wall minutes not numeric** - Rejects booleans, strings, lists
5. **Seed wrong type** - Only accepts strings and integers
6. **Missing required fields** - Validates complete structure
7. **Type errors** - Validates field types
8. **Range validation** - Checks numeric ranges
9. **Nested validation** - Deep structure validation

## Test Results
```bash
$env:NO_NETWORK="true"; $env:PYTHONPATH=(Get-Location).Path
pytest -q tests\qa\test_metrics_lint_v1.py
..................                                                       [100%]
18 passed in 0.06s
```

## Message Format
All messages follow `CATEGORY: Description` format:
- `ERROR:` - Validation failures
- `WARNING:` - Non-critical issues
- `MISSING_FIELD:` - Required fields missing
- `TYPE_ERROR:` - Incorrect field types
- `RANGE_ERROR:` - Values outside valid ranges
- `MERKLE_LENGTH_ERROR:` - Hash length validation
- `WALL_MINUTES_TYPE_ERROR:` - Wall minutes type validation
- `SEED_TYPE_ERROR:` - Seed type validation

## Usage
```python
from backend.tools.metrics_linter import lint_metrics

is_valid, errors, warnings = lint_metrics(metrics_data)
```

## CLI Usage
```bash
python -m backend.tools.metrics_linter metrics.json
python -m backend.tools.metrics_linter metrics.json --warnings-as-errors
```

## Acceptance Criteria Met
- ✅ All linter tests pass
- ✅ Messages are ASCII-only
- ✅ PR body includes repro steps
- ✅ Edge cases comprehensively covered
- ✅ Stable message format for CI/CD
