# Attestation Auditor Documentation

**Agent:** `attestation-auditor`  
**Phase:** III - Evidence Chain Ledger & CI Hard Gate

## Overview

The attestation auditor provides comprehensive integrity verification for experiment attestation artifacts. It detects mismatches between declared and actual hashes, empty or truncated JSONL files, missing parent references, and manifest-to-preregistration inconsistencies.

This implementation is **read-only**, **deterministic**, and designed for CI/CD integration.

## Architecture

### Core Modules

#### 1. `attestation/manifest_verifier.py`
Provides SHA-256 hash utilities and manifest verification primitives.

**Key Functions:**
- `compute_sha256_file(filepath)` - Hash a file
- `compute_sha256_string(data)` - Hash a string
- `compute_sha256_json(data)` - Hash JSON with canonical serialization
- `verify_manifest_file_hash(manifest_path, expected_hash)` - Verify manifest integrity
- `load_and_verify_json(filepath)` - Load and validate JSON files

#### 2. `attestation/audit_uplift_u2.py`
Single experiment auditing with comprehensive artifact checks.

**Key Functions:**
- `audit_experiment(experiment_dir, repo_root, prereg_hash)` - Audit one experiment
- `render_audit_json(result)` - Generate JSON report
- `render_audit_markdown(result)` - Generate Markdown report

**Detects:**
- Empty or zero-line JSONL files
- Hash mismatches (manifest vs actual file hash)
- Missing artifact files
- Preregistration hash ≠ manifest declared hash

#### 3. `attestation/audit_uplift_u2_all.py`
Multi-experiment sweep and aggregation.

**Key Functions:**
- `audit_all_experiments(experiments_dir, repo_root, experiment_pattern, prereg_hashes)` - Audit multiple experiments
- `aggregate_audit_summary(results)` - Generate aggregate statistics
- `render_aggregate_json(results)` - JSON aggregate report
- `render_aggregate_markdown(results)` - Markdown aggregate report

#### 4. `attestation/evidence_chain.py` (Phase III)
Evidence chain ledger construction and CI gate evaluation.

**Key Functions:**
- `build_evidence_chain_ledger(audit_results)` - Build evidence chain ledger
- `evaluate_evidence_chain_for_ci(ledger)` - CI hard gate evaluation
- `render_evidence_chain_section(ledger)` - Markdown evidence chain section

**Ledger Schema:**
```json
{
  "schema_version": "1.0",
  "experiment_count": 2,
  "experiments": [
    {
      "id": "EXP_001",
      "status": "PASS",
      "artifact_hashes": {
        "path/to/artifact": "sha256_hash"
      },
      "report_path": "path/to/manifest.json"
    }
  ],
  "global_status": "PASS",
  "ledger_hash": "sha256_of_canonical_ledger_body"
}
```

#### 5. `attestation/audit_ci_entry.py`
CLI entry point for CI/CD integration.

**Modes:**
- `single` - Audit one experiment
- `multi` - Audit multiple experiments
- `evidence-chain` - Build ledger and evaluate CI gate

## Usage

### Single Experiment Audit

```bash
python3 -m attestation.audit_ci_entry \
  --mode single \
  --experiment-dir experiments/EXP_001 \
  --repo-root . \
  --format json \
  --output audit_report.json
```

### Multi-Experiment Audit

```bash
python3 -m attestation.audit_ci_entry \
  --mode multi \
  --experiments-dir experiments/ \
  --repo-root . \
  --pattern "EXP_*" \
  --format markdown \
  --output audit_report.md
```

### Evidence Chain Ledger (CI Gate)

```bash
python3 -m attestation.audit_ci_entry \
  --mode evidence-chain \
  --experiments-dir experiments/ \
  --repo-root . \
  --format json \
  --output evidence_ledger.json \
  --exit-code
```

**Exit Codes:**
- `0` - All experiments PASS (global_status == "PASS")
- `1` - Partial success (global_status == "PARTIAL")
- `2` - Failures detected (global_status == "FAIL")

## CI Integration

### GitHub Actions Example

```yaml
name: Evidence Chain Audit

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Evidence Chain Audit
        run: |
          python3 -m attestation.audit_ci_entry \
            --mode evidence-chain \
            --experiments-dir experiments/ \
            --format json \
            --output evidence_ledger.json \
            --exit-code
      
      - name: Upload Evidence Ledger
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: evidence-ledger
          path: evidence_ledger.json
```

## Testing

The implementation includes 27+ comprehensive tests covering:

1. **Hash Utilities** (9 tests)
   - String, bytes, JSON, and file hashing
   - Deterministic behavior
   - Error handling

2. **Single Experiment Auditing** (5 tests)
   - Valid experiments
   - Empty logs
   - Hash mismatches
   - JSON/Markdown rendering

3. **Multi-Experiment Auditing** (4 tests)
   - Batch processing
   - Aggregate statistics
   - Report generation

4. **Evidence Chain Ledger** (8 tests)
   - Ledger construction
   - CI gate evaluation
   - Deterministic hashing
   - Markdown rendering

5. **Integration** (1 test)
   - Full workflow from experiments to ledger

### Running Tests

```bash
# Run all attestation auditor tests
python3 -m pytest tests/test_attestation_auditor.py -v

# Run specific test class
python3 -m pytest tests/test_attestation_auditor.py::TestEvidenceChainLedger -v

# Run with coverage
python3 -m pytest tests/test_attestation_auditor.py --cov=attestation --cov-report=html
```

## Security & Invariants

### Sober Truth Guardrails

- ❌ **Do NOT** "correct" Phase I attestation files — they are sealed evidence
- ❌ **Do NOT** fabricate hashes or attestation metadata
- ❌ **Do NOT** interpret audit findings as uplift evidence
- ❌ **Do NOT** approve manifests that fail integrity checks
- ✅ **DO** flag any manifest claiming uplift without G1-G5 gate passage
- ✅ **DO** report empty results as potential experiment failures

### Preserved Invariants

1. **Read-Only Operations** - Never modifies attestation files
2. **Deterministic Hashing** - SHA-256 with canonical JSON serialization
3. **Complete Audit Trail** - All issues are logged and reported
4. **Cryptographic Binding** - Evidence chain ledger hash binds all experiments

## Evidence Chain Ledger Details

### Ledger Hash Calculation

The `ledger_hash` is computed as:

```
SHA-256(canonical_json(ledger_body))
```

Where `ledger_body` contains:
- `schema_version`
- `experiment_count`
- `experiments` (sorted by ID)
- `global_status`

The canonical JSON uses:
- Sorted keys (`sort_keys=True`)
- Compact separators (`separators=(',', ':')`)
- No whitespace

### Global Status Determination

```
if all experiments == "PASS":
    global_status = "PASS"
elif any experiment == "FAIL":
    global_status = "FAIL"
else:
    global_status = "PARTIAL"
```

### Artifact Hash Aggregation

For each experiment, artifact hashes are collected from:
- Log files (JSONL, CSV, etc.)
- Figure files (PNG, SVG, etc.)
- Manifest itself
- Any other declared artifacts

These are included in the ledger under `artifact_hashes` as a dictionary mapping paths to SHA-256 hashes.

## Evidence Pack Markdown Section

The `render_evidence_chain_section()` function generates a neutral, factual Markdown table:

```markdown
## Evidence Chain

The following table lists all experiments in the evidence chain.
All hashes are SHA-256. The ledger_hash can be used as a single
attestation fingerprint for this entire evidence pack.

| Experiment ID | Status | Manifest Hash | Evidence Hash |
|---------------|--------|---------------|---------------|
| `EXP_001` | ✓ PASS | `abc12345...` | `def67890...` |
| `EXP_002` | ✗ FAIL | `N/A` | `ghi24680...` |

**Ledger Hash:** `full_sha256_hash_here`

This ledger hash serves as a cryptographic fingerprint of the entire evidence chain.
It is computed as SHA-256 over the canonical JSON representation of the ledger body
(experiments, statuses, and artifact hashes in sorted order).
```

## Future Extensions

Potential enhancements (not part of Phase III):

1. **Preregistration Loader** - Automatic loading of prereg hashes from YAML
2. **Gate Validation** - Explicit G1-G5 gate requirement checks
3. **Proof DAG Verification** - Parent-child proof consistency
4. **Block Merkle Roots** - Verify roots match declared values
5. **Time-Based Audits** - Detect timestamp inconsistencies
6. **Differential Audits** - Compare two evidence chains

## References

- **Agent Definition:** `.github/agents/attestation-auditor.md`
- **Test Suite:** `tests/test_attestation_auditor.py`
- **SHA-256 Standard:** FIPS 180-4
- **JSON Canonicalization:** RFC 8785 (via sorted keys)

---

**Last Updated:** 2025-12-06  
**Version:** 1.0 (Phase III Complete)
