# Uplift Verified Pipeline - Dual-Attestation Documentation

## Overview

The Uplift Verified Pipeline extends the basic uplift gate with dual-attestation capabilities, integrating performance passport verification with uplift calculations to produce composite Uplift × Verification badges.

## Architecture

### Components

1. **Uplift Gate Core** (`scripts/uplift_gate.py`)
   - FOL and PL-2 system uplift calculations
   - Regression-floor rules implementation
   - Badge generation with shields.io compatibility

2. **Performance Passport Integration**
   - Merkle hash generation from performance metrics
   - Export verification from performance passports
   - Dual-attestation linkage

3. **CI Integration** (`.github/workflows/ci.yml`)
   - Automated uplift-omega job execution
   - Verified badge artifact generation
   - Auto-commit functionality with [skip ci]

## Dual-Attestation Process

### Merkle Hash Generation

#### merkle_perf
Generated from uplift performance metrics:
- FOL uplift ratio, baseline mean, guided mean
- PL-2 uplift ratio, baseline mean, guided mean
- Uses deterministic merkle_root function from blockchain.py

#### merkle_export
Generated from performance passport data:
- Cartographer identifier
- Run ID
- Overall status
- Fallback to deterministic hash if passport unavailable

### Regression-Floor Rules

| Uplift Ratio | Status | Exit Code | Description |
|--------------|--------|-----------|-------------|
| ≥ 1.25x      | PASS   | 0         | Meets performance threshold |
| 1.0x - 1.25x | WARNING| 3         | Positive but below threshold |
| < 1.0x       | FAIL   | 1         | Performance regression |

## Artifacts Generated

### Badges
- `fol_perf_badge.json` - FOL system uplift metrics
- `pl2_perf_badge.json` - PL-2 system uplift metrics  
- `uplift_badge.json` - Combined dual uplift metrics
- `uplift_verified_badge.json` - Composite Uplift × Verification badge

### Reports
- `uplift_verified_summary.md` - Human-readable summary
- `uplift_verified_summary.json` - Machine-readable summary with merkle hashes

## Usage

### Local Testing
```bash
python scripts/uplift_gate.py \
  --fol-csv artifacts/wpv5/fol_ab.csv \
  --pl2-csv artifacts/wpv5/pl2_ab.csv \
  --passport-path performance_passport.json \
  --output-dir artifacts/badges
```

### CI Integration
The uplift-omega job automatically runs on pushes to integrate/ledger-v0.1 and generates verified badges with auto-commit functionality.

## Verification Strategy

1. **Performance Passport Validation**
   - Checks for passport availability
   - Extracts verification data
   - Generates fallback hashes if needed

2. **Uplift Calculation Verification**
   - Validates CSV data integrity
   - Applies regression-floor rules
   - Generates deterministic metrics

3. **Badge Integrity**
   - Shields.io compatibility validation
   - JSON schema compliance
   - Merkle hash verification

## Troubleshooting

### Common Issues
- **Missing performance passport**: System generates fallback merkle_export hash
- **CSV data issues**: Check artifacts/wpv5/ directory for proper data format
- **CI failures**: Verify uplift-omega job logs for specific error details

### Exit Codes
- 0: PASS - All systems meet performance thresholds
- 1: FAIL - Performance regression detected
- 2: ERROR - System error (missing data, etc.)
- 3: WARNING - Positive performance but below threshold

## Attestation Linkage

The dual-attestation pipeline creates verifiable links between:

1. **Performance Metrics** → **Merkle Performance Hash**
   - Deterministic hash from uplift calculations
   - Includes both FOL and PL-2 system metrics
   - Enables verification of performance claims

2. **Performance Passport** → **Merkle Export Hash**
   - Links to external performance validation
   - Includes cartographer and run metadata
   - Provides audit trail for performance testing

3. **Composite Badge** → **Verified Uplift Status**
   - Combines uplift metrics with verification hashes
   - Provides single source of truth for performance status
   - Enables automated decision-making in CI/CD pipelines

## Integration with MathLedger Architecture

The verified uplift pipeline integrates seamlessly with the broader MathLedger verification architecture:

- Uses the same `merkle_root` function as the blockchain module
- Follows the same deterministic hashing patterns
- Maintains consistency with proof verification systems
- Supports the global doctrine of mechanical honesty

## Future Enhancements

- Integration with additional performance passport sources
- Support for multi-dimensional performance metrics
- Enhanced regression detection algorithms
- Real-time performance monitoring dashboards
