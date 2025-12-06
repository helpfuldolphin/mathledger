# MathLedger Verification Suite

Devin C - The Verifier

Mission: Verify every claim made by any Codex or Cursor. Enable verification by writing universal checkers - one-liners that anyone can run locally.

## Overview

The MathLedger verification suite provides comprehensive validation of all system claims including hash integrity, Merkle root computation, database consistency, metrics schema compliance, and proof parent relationships.

## Quick Start

### One-Liner Verification

```bash
# Run all checks (offline mode - no database required)
python tools/verify_all.py --offline

# Run all checks (online mode - requires DATABASE_URL)
python tools/verify_all.py

# Run specific check
python tools/verify_all.py --check hash

# Using shell wrapper
./scripts/verify.sh --offline
```

### PowerShell

```powershell
# Run all checks (offline mode)
.\scripts\verify.ps1 -Offline

# Run specific check
.\scripts\verify.ps1 -Check hash
```

## Verification Checks

### 1. Hash Integrity

Verifies that hash computation is deterministic and consistent.

**What it checks:**
- Same input always produces same hash
- Normalization is consistent
- Hash computation is repeatable

**Command:**
```bash
python tools/verify_all.py --check hash --offline
```

**Expected output:**
```
[PASS] hash_integrity: Hash computation is deterministic
```

### 2. Merkle Root Computation

Verifies that Merkle root computation is deterministic and follows the correct algorithm.

**What it checks:**
- Deterministic computation (same inputs = same output)
- Empty case handling
- Single element handling
- Multi-element tree construction

**Command:**
```bash
python tools/verify_all.py --check merkle --offline
```

**Expected output:**
```
[PASS] merkle_root: Merkle root computation verified
```

### 3. File Existence

Verifies that all critical files exist and have not drifted.

**What it checks:**
- Core backend modules (derive.py, rules.py, blockchain.py, canon.py)
- API server (app.py)
- Validation tools (metrics_lint_v1.py)
- Documentation (progress.md, README_ops.md)

**Command:**
```bash
python tools/verify_all.py --check files --offline
```

**Expected output:**
```
[PASS] file_existence: 8/8 critical files present
```

### 4. Metrics Schema Validation

Verifies that metrics files conform to V1 schema.

**What it checks:**
- V1 schema compliance
- No mixed schema feeds
- Required fields present
- Field types correct

**Command:**
```bash
python tools/verify_all.py --check metrics --offline
```

**Expected output:**
```
[PASS] metrics_schema: All metrics files valid
```

### 5. Normalization Idempotence

Verifies that normalization is idempotent (normalize(normalize(x)) == normalize(x)).

**What it checks:**
- Idempotence property holds for all test formulas
- Normalization is stable

**Command:**
```bash
python tools/verify_all.py --check normalization --offline
```

**Expected output:**
```
[PASS] normalization_idempotence: All normalizations idempotent
```

### 6. Database Integrity (Online Only)

Verifies database schema and data integrity.

**What it checks:**
- Critical tables exist (statements, proofs, blocks, proof_parents)
- Data counts are reasonable
- Latest block exists and has valid Merkle root format
- Merkle root format is correct (64 hex chars or 32 bytes)

**Command:**
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5433/mathledger"
python tools/verify_all.py --check database
```

**Expected output:**
```
[PASS] database_integrity: Database OK: 1234 statements, 567 proofs, 89 blocks
```

### 7. API Parity (Online Only)

Verifies API endpoints return consistent data with database.

**What it checks:**
- API module is importable
- (Future: Full API parity checks require running server)

**Command:**
```bash
python tools/verify_all.py --check api --offline
```

**Expected output:**
```
[PASS] api_parity: API module importable
```

### 8. Proof Parent Relationships (Online Only)

Verifies proof parent relationships are valid.

**What it checks:**
- No orphaned parent references
- All parent hashes exist in statements table
- Parent relationship count

**Command:**
```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5433/mathledger"
python tools/verify_all.py --check parents
```

**Expected output:**
```
[PASS] proof_parents: All 123 parent relationships valid
```

## Specialized Verification Tools

### Merkle Root Verification

Standalone tool for verifying Merkle roots in blocks.

```bash
# Verify all blocks (latest 10)
python tools/verify_merkle.py

# Verify specific block
python tools/verify_merkle.py --block 123

# Verify latest block only
python tools/verify_merkle.py --latest

# Verify more blocks
python tools/verify_merkle.py --limit 50
```

**Example output:**
```
Verifying 10 blocks...

[PASS] Block 1: Merkle root verified (45 statements)
[PASS] Block 2: Merkle root verified (67 statements)
[PASS] Block 3: Merkle root verified (89 statements)
...

[PASS] VERIFIED: All 10 blocks have valid Merkle roots
```

### Hash Collision Detection

Standalone tool for detecting hash collisions in statements.

```bash
# Check all statements
python tools/verify_hash_collisions.py

# Check random sample
python tools/verify_hash_collisions.py --sample 1000

# Verify hash recomputation
python tools/verify_hash_collisions.py --recompute 100
```

**Example output:**
```
============================================================
MathLedger Hash Integrity Verification
============================================================

Checking all statements...
Analyzed 5432 statements
Unique hashes: 5432

[PASS] VERIFIED: No hash collisions detected

Verifying hash recomputation for 100 statements...
[PASS] VERIFIED: All 100 hashes match recomputed values

============================================================
[PASS] VERIFIED: ALL HASH INTEGRITY CHECKS PASSED
============================================================
```

## Exit Codes

All verification tools use consistent exit codes:

- **0**: [PASS] VERIFIED: ALL CLAIMS HOLD
- **1**: [FAIL] One or more verification checks failed
- **2**: [ERROR] Fatal error during verification (missing dependencies, connection failure, etc.)

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Verification

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install psycopg
      - name: Run offline verification
        run: python tools/verify_all.py --offline
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python tools/verify_all.py --offline --check hash --check normalization
exit $?
```

## Troubleshooting

### ModuleNotFoundError: No module named 'backend'

The verification scripts automatically add the project root to PYTHONPATH. If you still see this error, manually set PYTHONPATH:

```bash
export PYTHONPATH=/path/to/mathledger:$PYTHONPATH
python tools/verify_all.py --offline
```

### Database connection failed

Ensure DATABASE_URL is set and PostgreSQL is running:

```bash
export DATABASE_URL="postgresql://ml:mlpass@localhost:5433/mathledger"
docker ps | grep postgres
```

### psycopg not installed

Install psycopg for database checks:

```bash
pip install psycopg[binary]
```

## Verification Philosophy

The verification suite follows these principles:

1. **Proof-or-Abstain**: No speculative answers without verifiable proof
2. **Determinism**: All computations must be deterministic and repeatable
3. **Idempotence**: Running verification multiple times produces same results
4. **Offline-First**: Core checks work without database or network
5. **One-Liner**: Anyone can run verification with a single command
6. **Clear Output**: Pass/fail status is immediately obvious
7. **Actionable Errors**: Failures include fix suggestions

## What Else Could Fail Silently?

The verification suite is designed to catch silent failures. Consider these scenarios:

1. **Hash collisions**: Different statements producing same hash
2. **Non-deterministic normalization**: Same input producing different normalized forms
3. **Merkle root drift**: Stored Merkle roots not matching recomputed values
4. **Orphaned references**: Proof parents pointing to non-existent statements
5. **Schema drift**: Metrics files mixing V1 and legacy formats
6. **File corruption**: Critical files missing or modified
7. **Database inconsistency**: Counts don't match, blocks out of sequence

The verification suite checks all of these and more.

## Extending the Verification Suite

To add a new verification check:

1. Add a method to the `Verifier` class in `tools/verify_all.py`:

```python
def verify_my_check(self) -> VerificationResult:
    """Verify my custom check."""
    self.log("Checking my custom property...")
    
    try:
        # Perform verification logic
        passed = True
        details = {"key": "value"}
        
        message = "Check passed" if passed else "Check failed"
        return VerificationResult("my_check", passed, message, details)
    except Exception as e:
        return VerificationResult("my_check", False, f"Error: {str(e)}")
```

2. Add the check to the `run_all_checks` method:

```python
checks = [
    # ... existing checks ...
    ("mycheck", self.verify_my_check),
]
```

3. Test the new check:

```bash
python tools/verify_all.py --check mycheck --offline
```

## Verification Seal

When all checks pass, the verification suite outputs:

```
============================================================
VERIFICATION SUMMARY: 8/8 checks passed
============================================================

[PASS] VERIFIED: ALL CLAIMS HOLD
```

This seal indicates that all claims about the MathLedger system have been verified and hold true.

## Support

For issues or questions about verification:

1. Check this documentation
2. Run with `--verbose` flag for detailed output
3. Check individual verification tool documentation
4. Review error messages for fix suggestions
5. Create GitHub issue with verification output

## Verification V2 - Audit Sync Edition

Universal Verifier V2 extends the base verification suite with audit harness integration and AllBlue Gate compatibility.

### New Features in V2

1. **Signed Verification Results**: Deterministic SHA-256 signatures for tamper detection
2. **Audit Harness Integration**: Schema cross-validation with Cursor N audit harness
3. **AllBlue Gate Compatibility**: Exit code mapping for CI ingestion
4. **Deterministic JSON Output**: Structured results in `artifacts/audit/`

### V2 Usage

```bash
# Run with audit sync
python tools/verify_all_v2.py --offline --audit-sync

# Specify custom output path
python tools/verify_all_v2.py --offline --output artifacts/audit/verification_summary.json

# Verbose mode with audit sync
python tools/verify_all_v2.py --offline --verbose --audit-sync
```

### V2 Output Files

The V2 verifier generates three files in `artifacts/audit/`:

1. **verification_summary.json** - Signed verification results with audit metadata
2. **verification_summary.md** - CI-friendly markdown summary
3. **exit_code_map.json** - AllBlue ingestion mapping

### V2 JSON Schema

```json
{
  "run_id": "48f24ba9a1ccdd90",
  "timestamp": "2025-10-31T20:53:56.225664+00:00",
  "version": "v2",
  "verifier": "Devin C - Universal Verifier V2",
  "mode": "offline",
  "checks": {
    "total": 8,
    "passed": 8,
    "failed": 0
  },
  "exit_code": 0,
  "exit_code_map": {
    "status": "PASS",
    "description": "VERIFIED: ALL CLAIMS HOLD",
    "allblue_status": "green",
    "ci_emoji": "✅"
  },
  "results": [...],
  "audit_metadata": {
    "schema_version": "v2.0",
    "audit_sync_enabled": true,
    "compliance_tags": ["RC", "ME", "IVL"],
    "acquisition_narrative": "Reliability & Correctness verification with audit trail"
  },
  "signature": "7bc127b5c7edbcc9a5191b879046e7382b85482931576032851f38517952cb9f"
}
```

### Exit Code Map for AllBlue

The exit code map provides structured status information for CI systems:

```json
{
  "0": {
    "status": "PASS",
    "description": "VERIFIED: ALL CLAIMS HOLD",
    "allblue_status": "green",
    "ci_emoji": "✅"
  },
  "1": {
    "status": "FAIL",
    "description": "One or more verification checks failed",
    "allblue_status": "red",
    "ci_emoji": "❌"
  },
  "2": {
    "status": "ERROR",
    "description": "Fatal error during verification",
    "allblue_status": "yellow",
    "ci_emoji": "⚠️"
  }
}
```

### Audit Harness Validation

V2 includes schema cross-validation with the Audit Harness (Cursor N):

- Validates required fields (run_id, timestamp, version, checks, etc.)
- Verifies schema version compatibility (v2.0)
- Checks signature integrity to detect tampering
- Ensures compliance tags are present

### CI Integration

The V2 verifier integrates with GitHub Actions via `.github/workflows/verification-v2.yml`:

```yaml
- name: Run Universal Verifier V2 (Offline)
  run: |
    uv run python tools/verify_all_v2.py \
      --offline \
      --verbose \
      --audit-sync \
      --output artifacts/audit/verification_summary.json

- name: Display Verification Summary
  run: |
    cat artifacts/audit/verification_summary.md >> $GITHUB_STEP_SUMMARY
```

### Signature Verification

To verify the signature of a verification summary:

```python
import json
import hashlib

with open('artifacts/audit/verification_summary.json') as f:
    data = json.load(f)

signature = data.pop('signature')
canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
computed = hashlib.sha256(canonical.encode()).hexdigest()

if signature == computed:
    print("✅ Signature valid - data has not been tampered with")
else:
    print("❌ Signature invalid - data may have been modified")
```

### Compliance Tags

V2 results include strategic differentiator tags:

- **[RC]** - Reliability & Correctness
- **[ME]** - Metrics & Evidence
- **[IVL]** - Integration & Validation Layer

These tags align with the Strategic PR Template requirements and acquisition narrative framework.

## Verification V3.5 - Telemetry Analytics & Drift Alerting

Universal Verifier V3.5 adds telemetry analytics with rolling statistics, drift detection, and trend visualization.

### New Features in V3.5

1. **Telemetry Trends Collection**: Historical telemetry stored in `artifacts/audit/telemetry_trends.jsonl` (RFC 8785 canonical format)
2. **Rolling Statistics**: Computes mean ± sigma over last N runs (default: 10)
3. **Drift Alerting**: Triggers alert when coverage drops >10% below mean (configurable threshold)
4. **ASCII Trend Chart**: Visual bar chart showing historical trends
5. **GitHub Job Summary**: Automatic telemetry analytics in CI job summary

### V3.5 Usage

```bash
# Run with default settings (window=10, threshold=0.9)
python tools/verify_all_v3.py --offline --audit-sync

# Custom trend window (last 20 runs)
python tools/verify_all_v3.py --offline --audit-sync --trend-window 20

# Custom drift threshold (15% below mean)
python tools/verify_all_v3.py --offline --audit-sync --drift-threshold 0.85

# Combined custom settings
python tools/verify_all_v3.py --offline --audit-sync --trend-window 15 --drift-threshold 0.88
```

### V3.5 CLI Flags

- `--trend-window <int>`: Rolling window size (N runs) for telemetry analytics (default: 10)
- `--drift-threshold <float>`: Drift threshold multiplier relative to mean (default: 0.9 => 10% below mean)

### V3.5 Output Example

```
============================================================
TELEMETRY ANALYTICS V3.5 (window=10, threshold=0.9)
============================================================
Rolling Statistics (last 6 runs):
  Mean (mu): 87.5%
  Std Dev (sigma): 0.0%
  Range: [87.5%, 87.5%]

[PASS] Telemetry Health Stable (mu=87.5%, sigma=0.0%)

Telemetry Trend (last 6 runs + current):
R01  87.5% [#################---]
R02  87.5% [#################---]
R03  87.5% [#################---]
R04  87.5% [#################---]
R05  87.5% [#################---]
R06  87.5% [#################---]
NOW  87.5% [*****************---]
============================================================
```

### V3.5 Drift Detection

When coverage drops >10% below mean (or custom threshold):

```
[ALERT] Coverage drift detected: 75.0% < 78.75% (mu=87.5%, drift=14.3%)
```

When coverage is stable:

```
[PASS] Telemetry Health Stable (mu=87.5%, sigma=0.0%)
```

### V3.5 GitHub Job Summary

In CI environments, telemetry analytics are automatically appended to `$GITHUB_STEP_SUMMARY`:

```markdown
## Telemetry Analytics (V3.5)

Runs: 6 | mu=87.5% | sigma=0.0% | Range=[87.5%, 87.5%]

[PASS] Telemetry Health Stable (mu=87.5%, sigma=0.0%)
```

### V3.5 Telemetry Trends JSONL

Historical telemetry is stored in `artifacts/audit/telemetry_trends.jsonl` with RFC 8785 canonical format:

```json
{"checks_passed":7,"checks_total":8,"coverage_pct":87.5,"run_id":"fca26b69d2daf3ee","telemetry_hash":"9e28e25ce0b3117ae12c0ceadd974f7fe82483b65af3e64fdfe0b1a4e7592834","timestamp":"2025-11-02T18:49:41.720089+00:00"}
```

Each line is a complete JSON object with sorted keys, compact format, and ASCII-only characters.

### V3.5 Pass-Lines

- `[PASS] Telemetry Health Stable (mu=X%, sigma=Y%)` - Coverage stable within threshold
- `[ALERT] Coverage drift detected: X% < Y% (mu=Z%, drift=W%)` - Coverage dropped >threshold below mean
- `[PASS] Verifier Sync v3.5 [signature]` - Verification complete with signature

## Verification V3.6 - Predictive Analytics Uplift

Universal Verifier V3.6 adds predictive drift analytics with exponential-weighted moving averages and 3-run forecasting.

### New Features in V3.6

1. **Exponential-Weighted Moving Average (EWMA)**: Computes EWMA with alpha=0.3 for trend smoothing
2. **3-Run Drift Forecast**: Predicts coverage for next 3 runs using EWMA + trend slope
3. **Drift Probability**: Calculates probability of drift in forecasted runs
4. **Forecast Storage**: Stores forecasts in `artifacts/audit/telemetry_forecast.jsonl` (RFC 8785)
5. **Predictive Job Summary**: Appends forecast analytics to GitHub job summary

### V3.6 Usage

```bash
# Enable predictive analytics (requires >=3 historical runs)
python tools/verify_all_v3.py --offline --audit-sync --predict-drift

# Combined with custom parameters
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --trend-window 15 --drift-threshold 0.85
```

### V3.6 CLI Flags

- `--predict-drift`: Enable predictive drift analytics with 3-run forecast (requires >=3 historical runs)

### V3.6 Output Example

```
============================================================
PREDICTIVE ANALYTICS V3.6
============================================================
EWMA (alpha=0.3): 87.5%
3-Run Drift Forecast: 87.5%, 87.5%, 87.5%

[PASS] No Drift Predicted (prob=0.0%, mu=87.5%, sigma=0.0%)
============================================================
```

### V3.6 Drift Prediction

When drift probability >0% (one or more forecasts below threshold):

```
[ALERT] Drift Predicted (prob=33.3%, mu=87.5%, sigma=2.5%)
```

When no drift predicted:

```
[PASS] No Drift Predicted (prob=0.0%, mu=87.5%, sigma=0.0%)
```

### V3.6 GitHub Job Summary

In CI environments, predictive analytics are automatically appended to `$GITHUB_STEP_SUMMARY`:

```markdown
## Predictive Analytics (V3.6)

3-Run Drift Forecast: 87.5%, 87.5%, 87.5%

[PASS] No Drift Predicted (prob=0.0%, mu=87.5%, sigma=0.0%)
```

### V3.6 Forecast JSONL

Forecasts are stored in `artifacts/audit/telemetry_forecast.jsonl` with RFC 8785 canonical format:

```json
{"drift_probability":0.0,"ewma":87.5,"forecasts":[87.5,87.5,87.5],"run_id":"426dcc99e9e5cef5","stats":{"count":3,"mean":87.5,"std_dev":0.0},"threshold":78.75,"timestamp":"2025-11-02T20:07:28.957595+00:00"}
```

Each line contains:
- `ewma`: Exponential-weighted moving average (alpha=0.3)
- `forecasts`: Array of 3 predicted coverage values
- `drift_probability`: Percentage probability of drift (0-100%)
- `threshold`: Drift threshold (mean * drift_threshold)
- `stats`: Rolling statistics (mean, std_dev, count)
- `run_id`: Unique run identifier
- `timestamp`: ISO 8601 timestamp

### V3.6 Algorithm Details

**EWMA Calculation:**
```
ewma[0] = coverage[0]
ewma[i] = alpha * coverage[i] + (1 - alpha) * ewma[i-1]
```

**Trend Slope:**
```
slope = (coverage[-1] - coverage[-3]) / 3
```

**Forecast:**
```
forecast[i] = ewma + (slope * i)  for i in [1, 2, 3]
```

**Drift Probability:**
```
drift_count = sum(1 for f in forecasts if f < threshold)
drift_probability = (drift_count / 3) * 100
```

### V3.6 Determinism

Predictive analytics are deterministic for identical input data:
- Same historical trends → same EWMA
- Same EWMA + trend slope → same forecasts
- Same forecasts + threshold → same drift probability
- RFC 8785 canonical JSON ensures reproducible serialization

**Determinism Test:**
```bash
# Run twice and compare outputs
python tools/verify_all_v3.py --offline --audit-sync --predict-drift > run1.txt
python tools/verify_all_v3.py --offline --audit-sync --predict-drift > run2.txt
diff <(grep "EWMA\|Forecast\|Drift Predicted" run1.txt) <(grep "EWMA\|Forecast\|Drift Predicted" run2.txt)
# Should show no differences (except timestamps/run_ids)
```

### V3.6 Pass-Lines

- `[PASS] No Drift Predicted (prob=X%, mu=Y%, sigma=Z%)` - No forecasted drift
- `[ALERT] Drift Predicted (prob=X%, mu=Y%, sigma=Z%)` - Drift forecasted in next 3 runs
- `[PASS] Verifier Sync v3.6 [signature]` - Verification complete with signature

## Verification V3.6 - Forecast Accuracy Tracker + Alerting

Universal Verifier V3.6 adds forecast accuracy validation, tracking MAE/error%, and alerting on high error or drift probability.

### New Features in V3.6 Accuracy Tracker

1. **Forecast Accuracy Validation**: Validates EWMA forecasts against actual coverage
2. **MAE & Error Percentage**: Computes Mean Absolute Error and percentage error
3. **Alerting Thresholds**: Alerts when error_pct > 10% or drift_probability >= 50%
4. **Accuracy Storage**: Stores accuracy metrics in `artifacts/audit/forecast_accuracy.jsonl` (RFC 8785)
5. **Job Summary Integration**: Appends accuracy metrics to GitHub job summary

### V3.6 Accuracy Tracker Usage

```bash
# Enable forecast validation (auto-enabled with --predict-drift)
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --validate-forecast

# Explicit validation without prediction
python tools/verify_all_v3.py --offline --audit-sync --validate-forecast

# Combined with custom parameters
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --validate-forecast --trend-window 10 --drift-threshold 0.9
```

### V3.6 CLI Flags

- `--validate-forecast`: Validate forecast accuracy against actuals (auto-enabled with --predict-drift)

### V3.6 Accuracy Output Example

```
============================================================
FORECAST ACCURACY VALIDATION
============================================================
Predicted: 87.5%
Actual: 87.5%
MAE: 0.0%
Error: 0.0%

[PASS] Forecast Accuracy ok error_pct=0.0%
============================================================
```

### V3.6 Alerting Scenarios

**High Error Alert (error_pct > 10%):**

```
============================================================
FORECAST ACCURACY VALIDATION
============================================================
Predicted: 100.0%
Actual: 87.5%
MAE: 12.5%
Error: 14.29%

[ABSTAIN] Forecast Alert reason=high_error(>14.3%)
============================================================
```

**High Drift Probability Alert (drift_prob >= 50%):**

```
============================================================
FORECAST ACCURACY VALIDATION
============================================================
Predicted: 75.0%
Actual: 87.5%
MAE: 12.5%
Error: 14.29%

[ABSTAIN] Forecast Alert reason=drift_pred(>=66.7%)
============================================================
```

### V3.6 GitHub Job Summary with Accuracy

In CI environments, accuracy metrics are automatically appended to `$GITHUB_STEP_SUMMARY`:

```markdown
## Predictive Analytics (V3.6)

3-Run Drift Forecast: 87.5%, 87.5%, 87.5%

[PASS] No Drift Predicted (prob=0.0%, mu=87.5%, sigma=0.0%)

Forecast Accuracy: MAE=0.0%, Error=0.0%
[PASS] Forecast Accuracy ok error_pct=0.0%
```

### V3.6 Accuracy JSONL

Accuracy metrics are stored in `artifacts/audit/forecast_accuracy.jsonl` with RFC 8785 canonical format:

```json
{"actual":87.5,"alert":false,"alert_reason":null,"error_pct":0.0,"forecast_run_id":"baac3a80ce7b198a","mae":0.0,"predicted":87.5,"run_id":"94ae11133c22d6f3","timestamp":"2025-11-02T21:59:42.939592+00:00"}
```

Each line contains:
- `predicted`: Forecasted coverage from previous run
- `actual`: Actual coverage from current run
- `mae`: Mean Absolute Error (|predicted - actual|)
- `error_pct`: Percentage error ((mae / actual) * 100)
- `alert`: Boolean indicating if alert triggered
- `alert_reason`: Reason for alert (high_error or drift_pred)
- `forecast_run_id`: Run ID of forecast being validated
- `run_id`: Current run ID
- `timestamp`: ISO 8601 timestamp

### V3.6 Accuracy Algorithm Details

**MAE Calculation:**
```
mae = |predicted - actual|
```

**Error Percentage:**
```
error_pct = (mae / actual) * 100
```

**Alert Logic:**
```python
if error_pct > 10:
    alert = True
    alert_reason = f"high_error(>{error_pct:.1f}%)"
elif drift_probability >= 50:
    alert = True
    alert_reason = f"drift_pred(>={drift_probability}%)"
else:
    alert = False
```

### V3.6 Accuracy Determinism

Accuracy validation is deterministic for identical inputs:
- Same predicted + actual → same MAE
- Same MAE + actual → same error_pct
- Same error_pct + drift_prob → same alert status
- RFC 8785 canonical JSON ensures reproducible serialization

### V3.6 Accuracy Pass-Lines

- `[PASS] Forecast Accuracy ok error_pct=X%` - Error within acceptable threshold
- `[ABSTAIN] Forecast Alert reason=high_error(>X%)` - Error exceeds 10% threshold
- `[ABSTAIN] Forecast Alert reason=drift_pred(>=X%)` - Drift probability >= 50%

## Verification V3.6 - Configurable Thresholds + Rolling Trend

Universal Verifier V3.6 adds configurable thresholds for accuracy and drift alerts, plus rolling trend analysis over the last 10 validations.

### New Features in V3.6 Thresholds + Trend

1. **Configurable Accuracy Threshold**: `--accuracy-threshold` flag (default: 10.0)
2. **Configurable Drift Alert Threshold**: `--drift-alert-threshold` flag (default: 50)
3. **Rolling Trend Analysis**: Computes mean, std dev, min, max over last 10 validations
4. **Trend Display**: Shows accuracy trend statistics (requires >=3 historical records)
5. **Job Summary Integration**: Appends trend metrics to GitHub job summary

### V3.6 Thresholds + Trend Usage

```bash
# Use default thresholds (accuracy=10.0%, drift=50%)
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --validate-forecast

# Custom accuracy threshold (more sensitive)
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --accuracy-threshold 5.0

# Custom drift alert threshold (less sensitive)
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --drift-alert-threshold 70

# Combined custom thresholds
python tools/verify_all_v3.py --offline --audit-sync --predict-drift --accuracy-threshold 8.0 --drift-alert-threshold 40
```

### V3.6 CLI Flags

- `--accuracy-threshold <float>`: Accuracy error threshold percent (default: 10.0)
- `--drift-alert-threshold <int>`: Drift probability alert threshold percent (default: 50)

### V3.6 Thresholds Output Example

```
[PASS] Forecast Thresholds accuracy=10.0% drift=50%

============================================================
FORECAST ACCURACY VALIDATION
============================================================
Predicted: 87.5%
Actual: 87.5%
MAE: 0.0%
Error: 0.0%

[PASS] Forecast Accuracy ok error_pct=0.0%
============================================================
```

### V3.6 Trend Analysis Output Example

```
============================================================
ACCURACY TREND ANALYSIS (last 5 validations)
============================================================
Accuracy Trend: mean=7.43% std=6.42% min=0.0% max=14.29%
[PASS] Forecast Trend last10 mean=7.43% std=6.42%
============================================================
```

**Note:** Trend analysis only displays when >=3 historical accuracy records exist in `artifacts/audit/forecast_accuracy.jsonl`.

### V3.6 GitHub Job Summary with Trend

In CI environments, trend metrics are automatically appended to `$GITHUB_STEP_SUMMARY`:

```markdown
## Predictive Analytics (V3.6)

3-Run Drift Forecast: 87.5%, 87.5%, 87.5%

[PASS] No Drift Predicted (prob=0.0%, mu=87.5%, sigma=0.0%)

Forecast Accuracy: MAE=0.0%, Error=0.0%
[PASS] Forecast Accuracy ok error_pct=0.0%

Accuracy Trend (last 5): mean=7.43%, std=6.42%, min=0.0%, max=14.29%
[PASS] Forecast Trend last10 mean=7.43% std=6.42%
```

### V3.6 Trend Algorithm Details

**Load Accuracy History:**
```python
# Load last 10 accuracy records from forecast_accuracy.jsonl
accuracy_history = verifier.load_accuracy_history(limit=10)
```

**Compute Trend Statistics:**
```python
error_pcts = [record['error_pct'] for record in accuracy_history]

mean = sum(error_pcts) / len(error_pcts)
variance = sum((x - mean) ** 2 for x in error_pcts) / len(error_pcts)
std_dev = variance ** 0.5

trend = {
    'count': len(error_pcts),
    'mean': round(mean, 2),
    'std_dev': round(std_dev, 2),
    'min': round(min(error_pcts), 2),
    'max': round(max(error_pcts), 2)
}
```

### V3.6 Threshold Behavior

**Accuracy Threshold:**
- Alert triggered when: `error_pct > accuracy_threshold`
- Default: 10.0% (alerts on errors > 10%)
- Lower values = more sensitive (alert sooner)
- Higher values = less sensitive (tolerate larger errors)

**Drift Alert Threshold:**
- Alert triggered when: `drift_probability >= drift_alert_threshold`
- Default: 50% (alerts when majority of forecasts below threshold)
- Lower values = more sensitive (alert on lower drift probability)
- Higher values = less sensitive (tolerate higher drift probability)

### V3.6 Thresholds + Trend Pass-Lines

- `[PASS] Forecast Thresholds accuracy=X% drift=Y%` - Thresholds configured
- `[PASS] Forecast Trend last10 mean=X% std=Y%` - Trend statistics computed

### V3.6 Backward Compatibility

- Default thresholds unchanged (accuracy=10.0%, drift=50%)
- Existing behavior preserved when flags not specified
- Trend analysis optional (only displays when >=3 records)
- All existing pass-lines and output formats maintained

---

Devin C - The Verifier: Focus on truth, not noise. Wonder only about "what else could fail silently?"
