# TDA Pipeline Hash Attestation

## Overview

The TDA (Testing/Derivation/Analysis) Pipeline Hash Attestation system extends MathLedger's attestation chain to include cryptographic binding of experiment configuration, enabling:

1. **Configuration Integrity**: Cryptographically hash all TDA pipeline parameters
2. **Drift Detection**: Detect and flag configuration changes across experiment runs
3. **Hard Gate Binding**: Cryptographically bind gate decisions (e.g., `ABANDONED_TDA`)
4. **Chain Verification**: Verify complete attestation chains with TDA consistency

## Architecture

### Components

```
attestation/
├── tda_pipeline.py          # TDA config hashing and divergence detection
├── chain_verifier.py         # Attestation chain verifier with TDA binding
├── experiment_integration.py # RFL/U2 runner integration helpers
└── dual_root.py             # Existing dual-root attestation (R_t, U_t, H_t)

scripts/
└── verify_attestation_chain.py  # CLI tool for verification
```

### TDA Configuration Hash

The TDA pipeline hash is computed from:

- **Derivation Bounds**: `max_breadth`, `max_depth`, `max_total`
- **Verifier Settings**: `verifier_tier`, `verifier_timeout`, `verifier_budget`
- **Curriculum Slice**: `slice_id`, `slice_config_hash`
- **Abstention Strategy**: `abstention_strategy`
- **Gate Specifications**: Optional gate configurations

**Hash Algorithm**: SHA-256 of RFC 8785 canonicalized JSON

### Experiment Block Structure

Each experiment run produces an attestation block:

```json
{
  "run_id": "run_001",
  "experiment_id": "U2_EXP_001",
  
  "R_t": "<reasoning_merkle_root>",
  "U_t": "<ui_merkle_root>",
  "H_t": "<composite_root>",
  
  "tda_pipeline_hash": "<tda_config_hash>",
  "tda_config": {
    "max_breadth": 100,
    "max_depth": 50,
    "max_total": 1000,
    "verifier_tier": "tier1",
    "verifier_timeout": 10.0,
    "verifier_budget": null,
    "slice_id": "slice_a",
    "slice_config_hash": "abc123...",
    "abstention_strategy": "conservative"
  },
  
  "gate_decisions": {
    "G1": "PASS",
    "G2": "ABANDONED_TDA"
  },
  
  "prev_block_hash": "<previous_block_hash>",
  "block_number": 0,
  "block_hash": "<this_block_hash>"
}
```

## Exit Codes

The verification CLI (`scripts/verify_attestation_chain.py`) uses these exit codes:

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | All verifications passed |
| 1 | Integrity Failure | Missing fields, invalid structure, or bad hashes |
| 2 | Merkle Mismatch | Dual-root attestation (H_t) doesn't match recomputed value |
| 3 | Chain Linkage Broken | `prev_block_hash` doesn't match previous block |
| **4** | **TDA-Ledger Divergence** | **TDA configuration drift detected** |

**Exit Code 4** is the primary addition enabling CI to detect configuration drift.

## Usage

### Integration with RFL Runner

```python
from attestation.experiment_integration import create_rfl_attestation_block, save_attestation_block

# After RFL run completes
block = create_rfl_attestation_block(
    run_id=f"run_{run_index:03d}",
    experiment_id=config.experiment_id,
    reasoning_events=proof_events,
    ui_events=ui_events,
    rfl_config=config.to_dict(),
    gate_decisions=gate_decisions,
    prev_block_hash=prev_block_hash,
    block_number=run_index,
)

# Save to artifacts
output_path = artifacts_dir / f"run_{run_index:03d}" / "attestation.json"
save_attestation_block(block, output_path)
```

### Integration with U2 Runner

```python
from attestation.experiment_integration import create_u2_attestation_block, save_attestation_block

# After U2 run completes
block = create_u2_attestation_block(
    run_id=f"run_{cycle_id:03d}",
    experiment_id=experiment_id,
    reasoning_events=cycle_results,
    ui_events=ui_events,
    u2_config=u2_config,
    gate_decisions=gate_decisions,
    prev_block_hash=prev_block_hash,
    block_number=cycle_id,
)

# Save to artifacts
output_path = output_dir / f"run_{cycle_id:03d}" / "attestation.json"
save_attestation_block(block, output_path)
```

### CLI Verification

```bash
# Verify attestation chain (permissive mode - warns on drift)
python scripts/verify_attestation_chain.py artifacts/phase_ii/U2_EXP_001/

# Strict mode (fails on any TDA drift)
python scripts/verify_attestation_chain.py --strict-tda artifacts/phase_ii/U2_EXP_001/

# Verbose output
python scripts/verify_attestation_chain.py --strict-tda --verbose artifacts/
```

### CI Integration

Add to your CI pipeline:

```yaml
- name: Verify Attestation Chain
  run: |
    python scripts/verify_attestation_chain.py \
      --strict-tda \
      artifacts/experiment_output/
  # Exit code 4 = TDA drift detected
```

## Hard Gate Decisions

Hard Gate decisions (e.g., `ABANDONED_TDA`) are cryptographically bound via inclusion in the block hash:

```
block_hash = SHA256({
  "run_id": ...,
  "composite_root": ...,
  "tda_pipeline_hash": ...,
  "gate_decisions": {...},  # Included here!
  "block_number": ...
})
```

This ensures gate decisions cannot be altered without invalidating the block hash and breaking the chain.

### Example Gate Decisions

```json
{
  "G1_COVERAGE": "PASS",
  "G2_UPLIFT": "PASS",
  "G3_MANIFEST": "PASS",
  "G4_HERMETIC": "ABANDONED_TDA",
  "G5_VELOCITY": "PASS"
}
```

The `ABANDONED_TDA` decision indicates a gate was not evaluated due to TDA constraints, and this decision is cryptographically sealed in the attestation.

## Verification Algorithm

1. **Load blocks** from attestation directory
2. **Verify each block** internally:
   - Dual-root integrity: `H_t == SHA256(R_t || U_t)`
   - TDA hash: `tda_pipeline_hash == compute_tda_pipeline_hash(tda_config)`
3. **Verify chain linkage**:
   - `block[i].prev_block_hash == block[i-1].compute_block_hash()`
4. **Detect TDA drift**:
   - Compare `tda_config` between consecutive blocks
   - Report divergent fields
5. **Return result**:
   - Exit 0: Success
   - Exit 4: TDA divergence (in strict mode)

## Testing

Run tests:

```bash
# Unit tests
python -m pytest tests/test_tda_pipeline_attestation.py -v

# Integration test (if available)
python scripts/verify_attestation_chain.py /tmp/test_attestation_chain
```

## Security Guarantees

### Integrity

- **Tamper Detection**: Any modification to TDA config, gate decisions, or attestation roots invalidates the block hash
- **Chain Continuity**: Missing or reordered blocks are detected via `prev_block_hash` verification
- **Configuration Binding**: TDA pipeline parameters are cryptographically sealed

### Reproducibility

- **Deterministic Hashing**: RFC 8785 canonicalization ensures deterministic JSON serialization
- **Version Tracking**: TDA config hash changes with any parameter modification
- **Audit Trail**: Complete configuration history is preserved in the attestation chain

## Future Extensions

1. **Multi-Experiment Chains**: Link attestation chains across experiment series
2. **Consensus Rules**: Define when TDA drift is acceptable (e.g., bug fixes)
3. **Automatic Remediation**: Tools to migrate attestation chains after legitimate config changes
4. **Real-Time Monitoring**: Dashboard for tracking TDA consistency across concurrent runs

## References

- **Dual-Root Attestation**: `attestation/dual_root.py`
- **RFC 8785**: JSON Canonicalization Scheme (JCS)
- **Exit Code Standards**: POSIX exit codes (0=success, 1-255=failure types)
