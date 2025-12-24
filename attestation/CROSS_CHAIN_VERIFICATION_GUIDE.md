# Cross-Chain Attestation Verification Guide

This guide demonstrates how to use the cross-chain attestation verifier for Phase III → Phase IV evidence chain governance.

## Quick Start

### Using the CI Guard Script

The simplest way to verify attestation chains is via the CI guard script:

```bash
# Basic verification
python scripts/ci_attestation_guard.py artifacts/

# Strict schema checking
python scripts/ci_attestation_guard.py artifacts/ --strict-schema

# Fail on warnings (useful for CI/CD)
python scripts/ci_attestation_guard.py artifacts/ --fail-on-warnings

# Quiet mode (only exit code)
python scripts/ci_attestation_guard.py artifacts/ --quiet
```

### Exit Codes

- **0 (PASS)**: All attestation chains valid
- **1 (PARTIAL)**: Non-critical warnings detected (hash drift, schema drift, timestamp issues)
- **2 (FAIL)**: Critical issues detected (chain discontinuities, duplicates, dual-root mismatches)
- **3 (CRITICAL)**: System-level failure (missing artifacts, crashes)

## Programmatic Usage

### Basic Verification

```python
from attestation.cross_chain_verifier import CrossChainVerifier

# Create verifier
verifier = CrossChainVerifier()

# Verify manifests
manifests = [
    {
        'experiment_id': 'EXP_001',
        'manifest_version': '1.0',
        'timestamp_utc': '2025-01-01T00:00:00Z',
        'reasoning_merkle_root': 'abc...',
        'ui_merkle_root': 'def...',
        'composite_attestation_root': 'ghi...',
    },
    # ... more manifests
]

result = verifier.verify_chain(manifests)

# Check results
if result.is_valid:
    print(f"✓ All {result.total_experiments} experiments verified")
else:
    print(f"✗ Found {len(result.chain_discontinuities)} discontinuities")
    print(result.summary())
```

### Verifying Artifacts Directory

```python
from pathlib import Path
from attestation.cross_chain_verifier import CrossChainVerifier

verifier = CrossChainVerifier()
result = verifier.verify_artifacts_directory(
    Path('artifacts/phase_ii'),
    manifest_pattern='**/attestation.json'
)

print(result.summary())
```

### Strict Schema Checking

```python
from attestation.cross_chain_verifier import CrossChainVerifier

# Enable strict schema checking (flag extra fields)
verifier = CrossChainVerifier(strict_schema=True)
result = verifier.verify_chain(manifests)

for drift in result.schema_drifts:
    print(f"Schema drift in {drift.experiment_id}:")
    if drift.missing_fields:
        print(f"  Missing: {drift.missing_fields}")
    if drift.extra_fields:
        print(f"  Extra: {drift.extra_fields}")
```

## Detected Issues

### Critical Issues (Block Chain Validation)

These issues cause `result.is_valid` to return `False`:

1. **Chain Discontinuities**: Broken `prev_hash` links
2. **Duplicate Experiment IDs**: Same ID used multiple times
3. **Dual-Root Mismatches**: H_t ≠ SHA256(R_t || U_t)

### Warnings (Non-Blocking)

These issues set `result.has_warnings` but don't block validation:

1. **Hash Drift**: Configuration hashes differ across runs with same experiment ID
2. **Schema Drift**: Missing required fields or extra fields (in strict mode)
3. **Timestamp Violations**: Timestamps not monotonically increasing

## Example: Complete Validation Workflow

```python
from pathlib import Path
from attestation.cross_chain_verifier import CrossChainVerifier

def validate_experiment_chain(artifacts_dir: Path) -> int:
    """
    Validate attestation chain and return exit code.
    
    Returns:
        0 = PASS, 1 = PARTIAL, 2 = FAIL, 3 = CRITICAL
    """
    if not artifacts_dir.exists():
        print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
        return 3  # CRITICAL
    
    verifier = CrossChainVerifier(strict_schema=False)
    
    try:
        result = verifier.verify_artifacts_directory(artifacts_dir)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return 3  # CRITICAL
    
    # Print summary
    print(result.summary())
    
    # Determine exit code
    if not result.is_valid:
        return 2  # FAIL
    elif result.has_warnings:
        return 1  # PARTIAL
    else:
        return 0  # PASS

# Usage
exit_code = validate_experiment_chain(Path('artifacts/'))
```

## Expected Manifest Schema

### Required Fields

```json
{
  "experiment_id": "string",
  "manifest_version": "string",
  "timestamp_utc": "ISO 8601 timestamp"
}
```

### Optional Fields (Recommended)

```json
{
  "run_index": "integer",
  "prev_hash": "SHA-256 hex (for chain linking)",
  "reasoning_merkle_root": "SHA-256 hex (R_t)",
  "ui_merkle_root": "SHA-256 hex (U_t)",
  "composite_attestation_root": "SHA-256 hex (H_t)",
  "provenance": {
    "git_commit": "string",
    "user": "string"
  },
  "configuration": {
    "snapshot": "object"
  },
  "artifacts": {
    "logs": ["array of paths"],
    "data": ["array of paths"]
  },
  "results": {
    "status": "string",
    "duration_seconds": "number"
  }
}
```

### Dual-Root Invariant

If all three dual-root fields are present, the verifier checks:

```
H_t = SHA256(R_t || U_t)
```

Where `||` denotes concatenation of hex strings as ASCII bytes.

## CI/CD Integration

### GitHub Actions

```yaml
- name: Verify Attestation Chain
  run: |
    python scripts/ci_attestation_guard.py artifacts/ --fail-on-warnings
```

### Pre-commit Hook

```bash
#!/bin/bash
python scripts/ci_attestation_guard.py artifacts/ --quiet
exit $?
```

## Troubleshooting

### Chain Discontinuity

**Problem**: `expected prev_hash=abc..., got def...`

**Solution**: 
1. Check if manifests are in correct chronological order
2. Verify prev_hash computation matches `_compute_manifest_hash()`
3. Ensure no manifests were deleted or reordered

### Dual-Root Mismatch

**Problem**: `H_t does not match SHA256(R_t || U_t)`

**Solution**:
1. Recompute composite root: `hashlib.sha256(f"{r_t}{u_t}".encode('ascii')).hexdigest()`
2. Check if R_t and U_t are correct 64-char hex strings
3. Verify dual-root computation uses canonical `compute_composite_root()` function

### Schema Drift

**Problem**: `missing fields: ['timestamp_utc']`

**Solution**:
1. Add missing required fields to manifest
2. Ensure manifest follows expected schema
3. Check if manifest generation script is up to date

## Reference

See `tests/test_cross_chain_attestation.py` for comprehensive examples of all verification scenarios.
