# Governance Module: The Lawkeeper

**Claude F - The Lawkeeper of Provenance**

## Overview

The Governance module provides cryptographic validation of MathLedger's provenance seals and governance chains. It ensures:

1. **Dual-roots (R_t, U_t) integrity**: Merkle proof verification for sealed blocks
2. **Governance chain threading**: Attestation lineage (prev_signature → signature)
3. **Determinism enforcement**: All operations meet reproducibility thresholds
4. **Zero-tolerance validation**: Broken seals are immediately detected and reported

## Architecture

```
Attestation History → Governance Chain (signature lineage)
                              ↓
                        Lawkeeper Validator
                              ↓
                     [PASS] or [FAIL] verdict

Blocks Table → Declared Roots (Merkle roots + prev_hash)
                              ↓
                        Lawkeeper Validator
                              ↓
                     [PASS] or [FAIL] verdict
```

## Components

### 1. Validator (`validator.py`)

Core validation engine that verifies:

- **Governance threading**: Each attestation correctly references the previous signature
- **Root threading**: Each block correctly references the previous block hash
- **Dual-root integrity**: Merkle roots are valid SHA-256 hashes
- **Determinism scores**: All operations meet the 95% threshold

```python
from backend.governance import LawkeeperValidator

validator = LawkeeperValidator(
    governance_path=Path("artifacts/governance/governance_chain.json"),
    roots_path=Path("artifacts/governance/declared_roots.json"),
    verbose=True
)

lawful = validator.adjudicate()
# Returns: True if all validations pass, False otherwise
```

### 2. Safety Gate (`safety_gate.py`) - Phase X Neural Link

Surfaces safety gate decisions into observability systems:

```python
from backend.governance import (
    SafetyEnvelope,
    SafetyGateStatus,
    build_safety_gate_summary_for_first_light,
    build_safety_gate_tile_for_global_health,
    attach_safety_gate_to_evidence,
)

# Build envelope from gate decisions
envelope = SafetyEnvelope(
    final_status=SafetyGateStatus.PASS,
    total_decisions=100,
    blocked_cycles=0,
    advisory_cycles=2,
    decisions=[...]
)

# Add to First Light summary
first_light["safety_gate_summary"] = build_safety_gate_summary_for_first_light(envelope)

# Add to global health dashboard
health = build_global_health_surface(tiles, safety_envelope=envelope)

# Attach to evidence pack
evidence = attach_safety_gate_to_evidence(evidence, envelope)
```

See `docs/SAFETY_GATE_INTEGRATION.md` for complete integration guide.

### 3. Exporter (`export.py`)

Generates governance artifacts from source data:

```bash
# Export from attestation history
python backend/governance/export.py \
  --attestation-dir artifacts/repro/attestation_history \
  --governance-output artifacts/governance/governance_chain.json

# Export from database
python backend/governance/export.py \
  --db-url $DATABASE_URL \
  --roots-output artifacts/governance/declared_roots.json

# Export from JSON
python backend/governance/export.py \
  --blocks-json exports/blocks.json \
  --roots-output artifacts/governance/declared_roots.json
```

## Data Structures

### Governance Chain (`governance_chain.json`)

```json
{
  "version": "1.0.0",
  "exported_at": "2025-11-04T18:33:18Z",
  "entry_count": 3,
  "entries": [
    {
      "signature": "3d4af18e9501e98e548f0935fdb8f1de4ff6a4d07b565028da16d3dba0eec77b",
      "prev_signature": "cf05906e9bc3a6c446f307504d092864fcea78a453b8fd3278b3e5dbf02040d5",
      "timestamp": "2025-11-01T03:51:13.890908+00:00",
      "status": "CLEAN",
      "determinism_score": 100,
      "version": "1.0.0",
      "replay_success": true
    }
  ]
}
```

### Declared Roots (`declared_roots.json`)

```json
{
  "version": "1.0.0",
  "exported_at": "2025-11-04T18:33:30Z",
  "block_count": 3,
  "roots": [
    {
      "block_number": 1,
      "root_hash": "a1b2c3d4e5f6...",
      "prev_hash": "",
      "statement_count": 10,
      "sealed_at": "2025-11-01T00:00:00+00:00"
    }
  ]
}
```

## Usage

### CLI Validation

```bash
# Full validation
uv run python backend/governance/validator.py \
  --governance artifacts/governance/governance_chain.json \
  --roots artifacts/governance/declared_roots.json

# Quiet mode (errors only)
uv run python backend/governance/validator.py \
  --governance artifacts/governance/governance_chain.json \
  --roots artifacts/governance/declared_roots.json \
  --quiet
```

### Python API

```python
from pathlib import Path
from backend.governance import LawkeeperValidator

validator = LawkeeperValidator(
    governance_path=Path("artifacts/governance/governance_chain.json"),
    roots_path=Path("artifacts/governance/declared_roots.json"),
    verbose=True
)

# Load data
governance = validator.load_governance_chain()
roots = validator.load_declared_roots()

# Individual validations
validator.validate_governance_threading(governance)
validator.validate_determinism_scores(governance)
validator.validate_root_threading(roots)
validator.validate_dual_roots(roots)

# Full adjudication
lawful = validator.adjudicate()

if not lawful:
    print(f"Violations: {validator.errors}")
```

## Validation Rules

### Governance Chain

1. **Threading**: `entry[i].prev_signature == entry[i-1].signature`
2. **Signature format**: 64-character hex SHA-256
3. **Determinism**: `determinism_score >= 95`
4. **Status**: Must be "CLEAN" or "PROOF"

### Root Chain

1. **Threading**: `block[i].prev_hash == SHA256(BLCK || block[i-1])`
2. **Merkle root format**: 64-character hex SHA-256
3. **Statement count**: Non-negative integer
4. **Sealed timestamp**: Valid ISO 8601 format

## Domain Separation

The validator uses the centralized `backend.crypto.hashing` module with domain-separated hashing:

- **BLCK (0x03)**: Block header hashing
- **ROOT (0x07)**: Root attestation hashing
- **LEAF (0x00)**: Merkle tree leaf nodes
- **NODE (0x01)**: Merkle tree internal nodes

This prevents second preimage attacks (CVE-2012-2459 type).

## Testing

```bash
# Run governance tests
pytest tests/test_governance.py -v

# Run with coverage
pytest tests/test_governance.py --cov=backend.governance
```

Test coverage includes:
- Valid/invalid governance chains
- Valid/invalid root chains
- Determinism score validation
- Missing file handling
- Full adjudication workflows

## Integration with Nightly Operations

Add to `scripts/run-nightly.ps1`:

```powershell
# Export governance artifacts
python backend/governance/export.py `
  --attestation-dir artifacts/repro/attestation_history `
  --db-url $env:DATABASE_URL `
  --governance-output artifacts/governance/governance_chain.json `
  --roots-output artifacts/governance/declared_roots.json

# Validate provenance seals
$lawfulResult = python backend/governance/validator.py `
  --governance artifacts/governance/governance_chain.json `
  --roots artifacts/governance/declared_roots.json

if ($LASTEXITCODE -ne 0) {
    Write-Error "Lawkeeper validation failed - provenance seals broken"
    exit 1
}
```

## Exit Codes

- **0**: All validations passed (lawful)
- **1**: One or more validations failed (unlawful)

## Output Format

### Verbose Mode

```
⚖️  ============================================================
⚖️  LAWKEEPER INVOKED — Adjudicating Provenance Seals
⚖️  ============================================================
⚖️  Loaded governance chain: 3 entries
⚖️  Loaded declared roots: 5 blocks
⚖️  Validating governance chain threading...
⚖️  [PASS] Governance chain integrity OK [entries=3]
⚖️  Validating determinism scores...
⚖️  [PASS] Determinism scores validated [threshold>=95]
⚖️  Validating root chain threading...
⚖️  [PASS] Root chain integrity OK [blocks=5]
⚖️  Validating dual-root structure...
⚖️  [PASS] Dual-root structure validated [blocks=5]
⚖️  ============================================================
⚖️  VERDICT: [LAWFUL] All provenance seals validated
⚖️  ============================================================
```

### Error Output

```
❌ VIOLATION: Chain break at index 2: expected prev=abc123..., got def456...
❌ VIOLATION: Invalid signature format at index 3: xyz...
❌ VIOLATION: Determinism score below threshold at index 1: score=80, threshold=95
⚖️  VERDICT: [UNLAWFUL] 3 violations detected
```

## Future Enhancements

1. **Merkle proof verification**: Validate individual statement inclusion proofs
2. **Multi-system validation**: Cross-validate chains across logical systems (PL, FOL=, Ring)
3. **Incremental validation**: Fast-path validation for new entries only
4. **Automated remediation**: Suggest fixes for broken chains
5. **Webhook alerting**: Notify on validation failures

## Security Considerations

- **Domain separation**: All hashing uses proper domain tags to prevent collisions
- **Canonical serialization**: JSON serialized with `sort_keys=True` for determinism
- **Immutable chains**: Once sealed, governance entries cannot be modified
- **Audit trail**: All validations are logged with timestamps
- **Zero-trust**: Every entry is independently verified

## References

- `backend/crypto/hashing.py`: Cryptographic primitives
- `backend/ledger/blockchain.py`: Block construction
- `backend/ledger/blocking.py`: Block sealing
- `artifacts/repro/`: Attestation history
- `tests/test_governance.py`: Validation test suite

---

**Lawkeeper invoked — adjudicating provenance seals.**

*Judicial calm; zero speculation.*
