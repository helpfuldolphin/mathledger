# Attestation Auditor - Evidence Pack Integrity System

**Phase II — NOT USED IN PHASE I**

## Overview

The attestation-auditor provides cryptographic integrity verification for uplift_u2 Evidence Packs. It ensures tamper-evident attestation through SHA-256 hashing of all artifacts and comprehensive structural validation.

## Components

### 1. Core Hash Utilities (`experiments/manifest_verifier.py`)

Provides cryptographic primitives for artifact verification:

```python
from experiments.manifest_verifier import compute_artifact_hash, verify_artifact_hash

# Compute SHA-256 hash of any file
hash_value = compute_artifact_hash(Path("manifest.json"))
# Returns: "a6357b2709c93398cf9ee19933932b079f470c0f55912c278a2031d0fddd0961"

# Verify a file matches expected hash
is_valid = verify_artifact_hash(Path("manifest.json"), expected_hash)
```

**Key Functions:**
- `compute_artifact_hash(path: Path) -> str` - SHA-256 hash of file
- `verify_artifact_hash(path: Path, expected_hash: str) -> bool` - Verify hash match
- `hash_string(data: str) -> str` - Hash string content

### 2. Single Experiment Auditor (`experiments/audit_uplift_u2.py`)

Audits individual uplift_u2 experiment directories for integrity:

```bash
# Basic usage
python experiments/audit_uplift_u2.py results/uplift_u2/EXP_001

# Generate reports
python experiments/audit_uplift_u2.py results/uplift_u2/EXP_001 \
  --output-json audit.json \
  --output-md audit.md
```

**Checks Performed:**
1. ✅ Manifest exists and is valid JSON
2. ✅ Required manifest fields present (slice, mode, cycles)
3. ✅ Baseline log file exists and is non-empty
4. ✅ RFL log file exists (if applicable)
5. ✅ Cycle counts match between manifest and log files
6. ✅ ht_series hash verification (if present)
7. ✅ All artifacts hashed for tamper-evident reporting

**Exit Codes:**
- `0` - PASS: All checks passed
- `1` - FAIL: Structural or cryptographic failures
- `2` - MISSING: Required artifacts missing or incomplete

**Output Formats:**
- **JSON Report**: Machine-readable with artifact_hashes block
- **Markdown Report**: Human-readable with hash table and findings

### 3. Multi-Experiment Auditor (`experiments/audit_uplift_u2_all.py`)

Recursively discovers and audits all experiments in a directory tree:

```bash
# Audit all experiments
python experiments/audit_uplift_u2_all.py results/uplift_u2

# Generate aggregated reports
python experiments/audit_uplift_u2_all.py results/uplift_u2 \
  --output-json multi_audit.json \
  --output-md multi_audit.md
```

**Features:**
- Recursive experiment discovery (searches for `manifest.json`)
- Aggregated statistics across all experiments
- Findings categorization (log_file, cycle_count, hash_mismatch, etc.)
- Exit code semantics: 0 (all pass), 1 (failures), 2 (missing)

**Output:**
```
============================================================
MULTI-AUDIT SUMMARY
============================================================
Total Experiments: 5
Passed:            4
Failed:            1
Missing:           0
============================================================
Overall Status: FAIL (one or more experiments failed)
```

### 4. CI Entry Point (`experiments/audit_ci_entry.py`)

CI-friendly wrapper with minimal output and exit code forwarding:

```bash
# Basic usage (defaults to results/uplift_u2)
python experiments/audit_ci_entry.py

# Custom directory
python experiments/audit_ci_entry.py path/to/experiments

# With report output
python experiments/audit_ci_entry.py results/uplift_u2 \
  --output-json ci_audit.json \
  --output-md ci_audit.md
```

**CI Output:**
```
============================================================
CI EVIDENCE PACK AUDIT
============================================================
Target: results/uplift_u2

[1/3] Auditing: results/uplift_u2/exp_001
    Status: PASS, Findings: 0
[2/3] Auditing: results/uplift_u2/exp_002
    Status: PASS, Findings: 0
[3/3] Auditing: results/uplift_u2/exp_003
    Status: PASS, Findings: 0

============================================================
Summary: 3 experiment(s) audited
  ✅ Passed: 3
============================================================
```

## GitHub Actions Integration

Add this to `.github/workflows/audit.yml`:

```yaml
name: Audit Evidence Packs

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  audit:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    
    - name: Audit Evidence Packs
      run: |
        PYTHONPATH=$PWD python experiments/audit_ci_entry.py results/uplift_u2
```

## Report Formats

### JSON Report Structure

```json
{
  "experiment_dir": "/path/to/experiment",
  "status": "PASS",
  "findings": [],
  "artifact_hashes": {
    "manifest.json": "a6357b27...",
    "uplift_u2_test_baseline.jsonl": "1a813b80...",
    "uplift_u2_test_rfl.jsonl": "2e60ace6..."
  }
}
```

### Markdown Report Sample

```markdown
# Uplift U2 Experiment Audit Report

## Summary
- **Experiment Directory**: `/path/to/experiment`
- **Status**: **PASS**
- **Findings**: 0

## Artifact Hashes (SHA-256)

These cryptographic hashes provide tamper-evident verification of all
key artifacts in this Evidence Pack.

| Artifact | Hash |
|----------|------|
| `manifest.json` | `a6357b2709c93398cf9ee19933932b079f470c0f55912c278a2031d0fddd0961` |
| `uplift_u2_test_baseline.jsonl` | `1a813b80eb53fcad115767c33834a36e9df649c5ef56343e40d7844ad90a0a6a` |

## Findings

✅ No issues found. All integrity checks passed.
```

## Testing

Comprehensive test coverage across 36 tests:

```bash
# Run all attestation tests
uv run pytest tests/attestation/ -v

# Run specific test suites
uv run pytest tests/attestation/test_artifact_hashes.py -v
uv run pytest tests/attestation/test_multi_audit.py -v
uv run pytest tests/attestation/test_ci_entry.py -v
```

**Test Categories:**
- **Artifact Hashes** (13 tests): Hash computation, verification, determinism
- **Multi-Audit** (17 tests): Discovery, aggregation, exit codes
- **CI Entry** (6 tests): Exit code forwarding, output formatting

## Security & Design Principles

### Read-Only Behavior
- ✅ Never modifies any artifacts on disk
- ✅ Only reads and reports findings
- ✅ Phase I artifacts are sealed evidence

### Deterministic Operation
- ✅ Same files always produce same hashes
- ✅ No randomness in hash computation
- ✅ Reproducible audit results

### Security Hardening
- ✅ No network access
- ✅ No git operations
- ✅ No secret handling
- ✅ CodeQL verified (0 vulnerabilities)

### Tamper-Evident Reporting
- ✅ SHA-256 hashing of all artifacts
- ✅ Hashes embedded in audit reports
- ✅ Any modification changes hash

## Common Use Cases

### 1. Pre-Publication Verification

Before publishing experiment results, verify integrity:

```bash
python experiments/audit_uplift_u2.py results/uplift_u2/U2_EXP_001 \
  --output-json evidence_pack_attestation.json \
  --output-md evidence_pack_attestation.md
```

### 2. Batch Verification

Verify all experiments in a release:

```bash
python experiments/audit_uplift_u2_all.py results/uplift_u2 \
  --output-json release_audit.json \
  --output-md release_audit.md
```

### 3. CI/CD Pipeline

Automated verification on every commit:

```bash
python experiments/audit_ci_entry.py results/uplift_u2
# Exit code determines pass/fail
```

### 4. Post-Experiment Validation

After running an experiment, verify output integrity:

```bash
python experiments/audit_uplift_u2.py results/uplift_u2/latest_experiment
# Check exit code: 0=success, 1=failure, 2=missing
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'experiments'`

**Solution**: Set PYTHONPATH or use uv run:
```bash
# Option 1: Set PYTHONPATH
PYTHONPATH=/path/to/repo python experiments/audit_uplift_u2.py ...

# Option 2: Use uv run (recommended)
uv run python experiments/audit_uplift_u2.py ...
```

**Issue**: Exit code 2 (MISSING) but files exist

**Solution**: Check log file naming conventions. The auditor expects:
- Baseline: `uplift_u2_{slice_name}_baseline.jsonl`
- RFL: `uplift_u2_{slice_name}_rfl.jsonl`

**Issue**: Cycle count mismatch

**Solution**: Verify that:
1. Manifest declares correct number of cycles
2. Log file has exactly that many non-empty lines
3. No truncated or incomplete experiments

## Implementation Details

### Hash Computation

Uses SHA-256 with chunked reading for memory efficiency:

```python
sha256_hash = hashlib.sha256()
with open(path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
return sha256_hash.hexdigest()
```

### Experiment Discovery

Uses `rglob` for recursive manifest.json search:

```python
for manifest_path in root_dir.rglob("manifest.json"):
    experiment_dir = manifest_path.parent
    experiments.append(experiment_dir)
```

### Exit Code Logic

```python
if failed > 0:
    sys.exit(1)  # FAIL: Integrity violations
elif missing > 0:
    sys.exit(2)  # MIXED: Missing artifacts
else:
    sys.exit(0)  # PASS: All clean
```

## Future Enhancements

Potential extensions (not in current scope):

1. **Preregistration Cross-Check**: Verify manifest hashes match prereg declarations
2. **Merkle Tree Construction**: Build cryptographic proof trees for evidence chains
3. **Signature Verification**: Add public-key signature support
4. **Delta Audits**: Detect changes between successive audits
5. **Policy Enforcement**: Configurable gate requirements (G1-G5)

## References

- Original specification: Problem statement for attestation-auditor
- Related: `experiments/manifest.py` - Manifest generation
- Related: `experiments/run_uplift_u2.py` - Experiment runner
- Security: CodeQL analysis results (0 vulnerabilities)

---

**Maintainer**: attestation-auditor agent
**Status**: Production-ready, all tests passing
**Last Updated**: 2025-12-06
