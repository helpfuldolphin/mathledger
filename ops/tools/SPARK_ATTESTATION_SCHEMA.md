# SPARK Attestation Schema

**Purpose**: Document the structure and validation rules for First Organism attestation artifacts used in SPARK verification.

**Audience**: Cursor J (SPARK Execution Verifier), auditors, and downstream consumers (Wave 1, Dyno Chart).

---

## Attestation Artifact Location

**File Path**: `artifacts/first_organism/attestation.json`

**Log File Path**: `ops/logs/SPARK_run_log.txt`

---

## JSON Schema

### Required Fields

| Field | Type | Format | Description |
|-------|------|--------|-------------|
| `R_t` | string | 64-char hex | Reasoning merkle root (proof/reasoning events) |
| `U_t` | string | 64-char hex | UI merkle root (human interaction events) |
| `H_t` | string | 64-char hex | Composite attestation root = SHA256(R_t \|\| U_t) |

### Optional Fields (Audit Trail)

| Field | Type | Description |
|-------|------|-------------|
| `statement_hash` | string | Hash of the statement that triggered the run |
| `mdap_seed` | integer | Deterministic seed for reproducibility |
| `run_id` | string | Unique identifier for this run |
| `run_timestamp_iso` | string | ISO 8601 timestamp |
| `run_timestamp_unix` | integer | Unix timestamp |
| `version` | string | Attestation format version (e.g., "1.0.0") |
| `environment_mode` | string | Execution mode (standalone, integrated, etc.) |
| `chain_status` | string | Blockchain integration status |
| `components` | object | Dictionary mapping component names to implementations |
| `block_id` | string \| null | Associated block ID (if integrated) |
| `proof_id` | string \| null | Associated proof ID (if integrated) |
| `statement_id` | string \| null | Associated statement ID (if integrated) |
| `slice_name` | string | Curriculum slice name |
| `is_synthetic` | boolean | Whether this is a synthetic/test run |
| `mdap_metadata` | object | Additional MDAP metadata |
| `extra` | object | Additional custom fields |

---

## Validation Rules

### 1. Structure Validation

- All required fields (`R_t`, `U_t`, `H_t`) must be present
- All root fields must be valid 64-character hexadecimal strings
- Hex strings must contain only characters `0-9` and `a-f` (case-insensitive)

### 2. Integrity Validation

**H_t Invariant**: The stored `H_t` must equal the recomputed value:

```
H_t = SHA256(R_t || U_t)
```

Where:
- `||` denotes ASCII string concatenation
- `R_t` and `U_t` are concatenated as-is (no padding, no separators)
- The result is a 64-character lowercase hexadecimal string

### 3. Implementation

The canonical recomputation function is:
- **Module**: `basis.attestation.dual`
- **Function**: `composite_root(reasoning: HexDigest, ui: HexDigest) -> HexDigest`

---

## PASS Line Format

The SPARK run must emit a canonical PASS line in the log:

```
[PASS] FIRST ORGANISM ALIVE H_t=<short-hex>
```

Where:
- `<short-hex>` is the first 12 characters of `H_t` (for readability)
- The full `H_t` (64 chars) is stored in `attestation.json`

**Example**:
```
[PASS] FIRST ORGANISM ALIVE H_t=01e5056e567b
```

Corresponds to full `H_t`:
```
01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2
```

---

## Verification Tools

### 1. Log Parser

**Script**: `ops/tools/parse_spark_pass.py`

**Usage**:
```bash
python ops/tools/parse_spark_pass.py [log_file_path]
```

**Output**: Prints `H_t=<short-hex>` or exits with non-zero status if not found.

**Default log file**: `ops/logs/SPARK_run_log.txt`

### 2. Attestation Verifier

**Script**: `experiments/verify_first_organism_attestation.py`

**Usage**:
```bash
python experiments/verify_first_organism_attestation.py [attestation_file]
```

**Verification Steps**:
1. Load and parse JSON
2. Validate structure (required fields present, valid hex format)
3. Recompute `H_t = SHA256(R_t || U_t)`
4. Verify recomputed `H_t` matches stored `H_t`

**Exit codes**:
- `0`: All verifications passed
- `1`: Verification failed (with detailed error message)

**Default attestation file**: `artifacts/first_organism/attestation.json`

---

## Example Attestation

```json
{
  "statement_hash": "0c90faf28890f9bf1883806f0adbbc433f26f87a75849099ff1dec519aa00679",
  "R_t": "a8dc5b2c7778ce38f72e63ecc4b7a9b010969c018d3d7cafff12bf6d85400336",
  "U_t": "8c11ea1e67666dd3f14a12cdf475a2d7f7c801037f3d273ccca069b1fa703359",
  "H_t": "01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2",
  "mdap_seed": 1296318800,
  "run_id": "run-285fabed4280",
  "run_timestamp_iso": "2025-01-18T05:01:12+00:00",
  "run_timestamp_unix": 1737176472,
  "block_id": null,
  "proof_id": null,
  "statement_id": null,
  "version": "1.0.0",
  "environment_mode": "standalone",
  "chain_status": "ready",
  "slice_name": "first-organism-slice",
  "is_synthetic": false,
  "mdap_metadata": {},
  "extra": {
    "test": "standalone"
  },
  "components": {
    "derivation": "axiom_engine",
    "ledger": "LedgerIngestor",
    "attestation": "attestation.dual_root",
    "rfl": "RFLRunner"
  }
}
```

**Verification**:
```python
from basis.attestation.dual import composite_root

R_t = "a8dc5b2c7778ce38f72e63ecc4b7a9b010969c018d3d7cafff12bf6d85400336"
U_t = "8c11ea1e67666dd3f14a12cdf475a2d7f7c801037f3d273ccca069b1fa703359"
H_t = composite_root(R_t, U_t)
# H_t == "01e5056e567ba57e90a6721281aa253bf6db34a4fa6c80bc10601d04783f59d2" ✓
```

---

## Integration with SPARK Workflow

1. **SPARK Run** → Executes First Organism test
2. **PASS Line** → Emitted to stdout/log: `[PASS] FIRST ORGANISM ALIVE H_t=<short-hex>`
3. **Log Capture** → Written to `ops/logs/SPARK_run_log.txt`
4. **Attestation Write** → `artifacts/first_organism/attestation.json` created
5. **Cursor J Verification**:
   - Parse log file → Extract `H_t` from PASS line
   - Load attestation → Validate structure
   - Recompute `H_t` → Verify integrity
   - Report status → Ready for Wave 1 / Dyno Chart

---

## References

- **Dual Attestation**: `basis/attestation/dual.py`
- **Test Integration**: `tests/integration/test_first_organism.py`
- **SPARK Runner**: `scripts/run_spark_closed_loop.ps1`
- **Log Helper**: `tests/integration/conftest.py::log_first_organism_pass()`

---

**Last Updated**: 2025-01-18  
**Maintainer**: Cursor J (SPARK Execution Verifier)

