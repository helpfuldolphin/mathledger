# Exporter Dry-Run QA Contract

##  Shinobi QA Sentinel Doctrine

**"No malformed metric, no corrupted Merkle, no contract-breaking prefix shall pass."**

This document defines the absolute doctrine for the MathLedger exporter dry-run QA lane. The contract is enforced by the Shinobi QA Sentinel with the precision of a shuriken - precise, deadly, and unavoidable.

## [target] Sanctioned Prefixes (ABSOLUTE DOCTRINE)

The exporter dry-run output **MUST** start with exactly one of these three sanctioned prefixes:

### [check] `DRY-RUN ok:`
- **Usage**: Valid V1 schema files processed successfully
- **Exit Code**: 0
- **Format**: `DRY-RUN ok: <file_path> (v1=<count>)`
- **Example**: `DRY-RUN ok: /path/to/metrics.jsonl (v1=1000)`

### [check] `mixed-schema:`
- **Usage**: Files containing both V1 and legacy/unknown records
- **Exit Code**: 1
- **Format**: `mixed-schema: <file_path> (v1=<v1_count>, unknown=<unknown_count>)`
- **Example**: `mixed-schema: /path/to/metrics.jsonl (v1=500, unknown=300)`

### [check] `error:`
- **Usage**: Fatal errors (file not found, invalid JSON, empty file, etc.)
- **Exit Code**: 1
- **Format**: `error: <error_description>`
- **Examples**:
  - `error: File not found: /path/to/nonexistent.jsonl`
  - `error: Empty file`
  - `error: Invalid JSON at line 42`

##  DOCTRINE VIOLATIONS

Any output that does **NOT** start with one of the sanctioned prefixes is a **DOCTRINE VIOLATION** and will be instantly rejected by the Shinobi QA Sentinel.

### Examples of DOCTRINE VIOLATIONS:
- `Success: File processed` [x]
- `Warning: Mixed schemas detected` [x]
- `Processing complete` [x]
- `File validated successfully` [x]
- Any output without a sanctioned prefix [x]

## [circus] Stress Testing Requirements

### Randomized Merkle Hash Testing
- **50+ different Merkle hash variations** per test run
- **Valid hashes**: 64-character hex strings (0-9, a-f)
- **Invalid hashes**: Too short, too long, non-hex chars, uppercase, special chars
- **Edge cases**: All zeros, all ones, pattern repeats, deadbeef patterns

### 1000-Line JSONL Stress Testing
- **Mixed CRLF/LF line endings** (alternating every line)
- **Randomized Merkle hashes** for each record
- **Mixed schema testing** (V1 + legacy records)
- **Error injection testing** (malformed JSON at random lines)

### Performance Requirements
- **1000-line files**: Must process in < 1 second
- **50 randomized hashes**: Must complete in < 10 seconds
- **Memory efficiency**: No memory leaks under stress
- **Deterministic output**: Identical inputs produce identical outputs

## [lock] Frozen Contract Tests

The following tests are **FROZEN** and must never regress:

### Basic Prefix Contract Tests
1. `test_prefix_contract_dry_run_ok()` - DRY-RUN ok: prefix validation
2. `test_prefix_contract_mixed_schema()` - mixed-schema: prefix validation
3. `test_prefix_contract_error_file_not_found()` - error: prefix for file not found
4. `test_prefix_contract_error_empty_file()` - error: prefix for empty file

### Shinobi Stress Tests
5. `test_shinobi_merkle_hash_stress_harness()` - 50+ randomized Merkle hashes
6. `test_shinobi_1000_line_stress_crlf_lf_mixed()` - 1000-line mixed line endings
7. `test_shinobi_mixed_schema_stress_1000_lines()` - Mixed schema under stress
8. `test_shinobi_error_handling_stress_1000_lines()` - Error handling under stress

##  Anime Energy References

### Death Note
Every output is written in the Shinobi's book - if it doesn't follow the prefix rules, it dies instantly.

### Naruto
Each randomized hash is like a Shadow Clone - the Shinobi tests them all until only truth remains.

### Dragon Ball Z
When regressions appear, the Shinobi powers up to SSJ2 and vaporizes them with a QA Kamehameha.

### Hunter x Hunter
The doctrine tests are Nen contracts - absolute, binding, and lethal if broken.

##  Doctrine Compliance Validation

### ASCII-Only Output
- All output must be ASCII-compatible
- No Unicode characters in error messages
- Universal system compatibility

### Deterministic Behavior
- Identical inputs produce identical outputs
- No random elements in error messages
- Consistent file path handling

### Prefix Validation
- Output must start with sanctioned prefix
- No partial matches or variations
- Case-sensitive validation

##  Failure Snapshots

When tests fail, the Shinobi QA Sentinel produces detailed failure snapshots:

```json
{
  "test_name": "shinobi_merkle_stress_all_zeros",
  "expected": "Exit code 0 with DRY-RUN ok: prefix",
  "actual": "Exit code 1: error: Invalid Merkle hash",
  "context": {
    "hash": "0000000000000000000000000000000000000000000000000000000000000000",
    "record": {...},
    "file_size": 1024,
    "sanctioned_prefixes": ["DRY-RUN ok:", "mixed-schema:", "error:"]
  }
}
```

## [target] CI/CD Integration

### Makefile Target
```makefile
qa-exporter-contract:
	@echo "Running exporter contract QA tests..."
	powershell -Command "$$env:NO_NETWORK='true'; $$env:PYTHONPATH=(Get-Location).Path; pytest -q tests\qa\test_exporter_v1.py"
	@echo "OK: exporter contract QA tests passed"
```

### Test Execution
```bash
# Run all contract tests
make qa-exporter-contract

# Run specific stress test
pytest -v tests/qa/test_exporter_v1.py::test_shinobi_merkle_hash_stress_harness

# Run with detailed output
pytest -v -s tests/qa/test_exporter_v1.py
```

## [tool] Maintenance Guidelines

### Adding New Tests
1. Follow the `test_shinobi_*` naming convention
2. Include doctrine compliance validation
3. Add comprehensive failure snapshots
4. Test under stress conditions (1000+ records)

### Modifying Existing Tests
1. **NEVER** modify frozen contract tests
2. Only add new stress test variations
3. Maintain backward compatibility
4. Update documentation if needed

### Debugging Failures
1. Check failure snapshots for context
2. Validate prefix compliance
3. Verify ASCII-only output
4. Test with smaller datasets first

## [circus] Performance Benchmarks

| Test Type | Record Count | Expected Time | Max Memory |
|-----------|--------------|---------------|------------|
| Basic Prefix | 1 | < 0.1s | < 10MB |
| Merkle Stress | 50 | < 10s | < 50MB |
| 1000-Line Stress | 1000 | < 1s | < 100MB |
| Mixed Schema | 1000 | < 1s | < 100MB |
| Error Handling | 1000 | < 1s | < 100MB |

## [trophy] Success Criteria

The exporter dry-run QA lane is considered **HARDENED** when:

- [check] All frozen contract tests pass
- [check] All Shinobi stress tests pass
- [check] 1000-line files process in < 1 second
- [check] 50+ randomized hashes complete in < 10 seconds
- [check] All output is ASCII-only and doctrine-compliant
- [check] No regressions under stress conditions
- [check] Comprehensive failure snapshots for debugging

##  Shinobi QA Sentinel Oath

*"I am the sentinel at the exporter's gate. I wield QA like a shinobi wields shuriken - precise, deadly, and unavoidable. No malformed metric, no corrupted Merkle, no contract-breaking prefix shall pass. The doctrine is absolute, the contract is binding, and violations are lethal."*

---

**Contract Version**: 1.0
**Last Updated**: 2025-01-20
**Maintainer**: Cursor A - Shinobi QA Sentinel
**Status**: ACTIVE - Doctrine Enforced
