# SPARK Verification Tools

**Purpose**: Tools for Cursor J (SPARK Execution Verifier) to verify First Organism execution and attestation integrity.

---

## Tools Overview

### 1. `parse_spark_pass.py`

Extracts `H_t` from SPARK run log by parsing the canonical PASS line.

**Usage**:
```bash
python ops/tools/parse_spark_pass.py [log_file_path]
```

**Default log file**: `ops/logs/SPARK_run_log.txt`

**Output**: Prints `H_t=<short-hex>` to stdout, or exits with non-zero status if not found.

**Example**:
```bash
$ python ops/tools/parse_spark_pass.py
H_t=01e5056e567b
```

---

### 2. `verify_first_organism_attestation.py`

Verifies attestation JSON integrity by:
1. Validating structure (R_t, U_t, H_t present and valid)
2. Recomputing H_t = SHA256(R_t || U_t)
3. Verifying recomputed H_t matches stored H_t

**Usage**:
```bash
python experiments/verify_first_organism_attestation.py [attestation_file]
```

**Default attestation file**: `artifacts/first_organism/attestation.json`

**Exit codes**:
- `0`: All verifications passed
- `1`: Verification failed (with detailed error message)

**Example**:
```bash
$ python experiments/verify_first_organism_attestation.py
✅ Attestation verification PASSED
   File: artifacts/first_organism/attestation.json
   R_t: a8dc5b2c7778ce38f72e63ecc4b7a9b010969c018d3d7cafff12bf6d85400336
   U_t: 8c11ea1e67666dd3f14a12cdf475a2d7f7c801037f3d273ccca069b1fa703359
   H_t: 01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2
   Recomputed H_t: 01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2
   ✓ H_t matches recomputed value
```

---

## Verification Workflow

### Step 1: Parse PASS Line from Log

```bash
python ops/tools/parse_spark_pass.py ops/logs/SPARK_run_log.txt
```

This extracts the short H_t (12 chars) from the log file.

### Step 2: Verify Attestation Integrity

```bash
python experiments/verify_first_organism_attestation.py
```

This validates the full attestation JSON and recomputes H_t.

### Step 3: Cross-Reference

The short H_t from the log should match the first 12 characters of the full H_t in the attestation JSON.

---

## Integration

These tools are designed to be used by:
- **Cursor J**: SPARK Execution Verifier (post-run verification)
- **CI/CD**: Automated verification in pipelines
- **Auditors**: Manual verification of attestation integrity
- **Downstream**: Wave 1 and Dyno Chart consumers

---

## Schema Documentation

See `ops/tools/SPARK_ATTESTATION_SCHEMA.md` for complete schema documentation.

---

## Dependencies

- Python 3.11+
- `basis.attestation.dual.composite_root` function
- Standard library: `json`, `pathlib`, `re`, `sys`

---

**Last Updated**: 2025-01-18  
**Maintainer**: Cursor J (SPARK Execution Verifier)

