# Cryptographic Roadmap: Ed25519 Signatures & Merkle Witnesses

## Overview

This document outlines the roadmap for transitioning the AllBlue Witnessed Epoch system from SHA256-based witness signatures to asymmetric Ed25519 cryptography with Merkle tree integration.

**Current State:** Witness signatures use SHA256 hashing with domain-specific prefixes (e.g., `sha256("verification_gate:{hash}:{H_t}")`).

**Target State:** Witness signatures use Ed25519 keypairs with non-repudiation, key rotation, and Merkle tree verification.

## Motivation

### Limitations of Current SHA256 Approach

1. **No Non-Repudiation:** SHA256 hashes can be recomputed by anyone with access to lane artifacts. There is no cryptographic proof that a specific witness generated the signature.

2. **No Key Rotation:** SHA256 hashing has no concept of key rotation. If a witness is compromised, there is no mechanism to revoke and replace keys.

3. **No Partial Verification:** Current system requires all three witness signatures to verify the epoch seal. Merkle trees would enable efficient partial verification.

4. **No Tamper Detection:** If lane artifacts are modified after signature generation, the system cannot detect tampering without recomputing all hashes.

### Benefits of Ed25519 + Merkle Trees

1. **Non-Repudiation:** Ed25519 signatures provide cryptographic proof that a specific private key signed the epoch hash.

2. **Key Rotation:** Keypairs can be rotated on a schedule (quarterly, annually) with versioned keys.

3. **Partial Verification:** Merkle proofs enable verification of individual witnesses without requiring all three signatures.

4. **Tamper Detection:** Signatures become invalid if lane artifacts are modified after signing.

5. **Audit Trail:** Public keys can be published and archived for long-term verification.

## Phase 1: Ed25519 Keypair Infrastructure

### Objectives

- Generate Ed25519 keypairs for each witness lane
- Implement secure key storage strategy
- Add dual-mode verification (SHA256 + Ed25519)
- Maintain backward compatibility with SHA256-only mode

### Deliverables

#### 1.1 Keypair Generation Script

**File:** `scripts/generate_witness_keypairs.py`

**Functionality:**
- Generate Ed25519 keypairs for three witnesses (verification-gate, hermetic-matrix, perf-gate)
- Output keypairs in PEM format (private key) and hex format (public key)
- Store private keys securely (environment variables or secrets manager)
- Store public keys in `config/witness_pubkeys.json`

**Usage:**
```bash
python3 scripts/generate_witness_keypairs.py --output-dir artifacts/keys/
```

**Output:**
```
artifacts/keys/verification_gate.priv  # PEM-encoded private key
artifacts/keys/hermetic_matrix.priv
artifacts/keys/perf_gate.priv
config/witness_pubkeys.json            # RFC 8785 canonicalized public keys
```

**Public Key Format (RFC 8785):**
```json
{
  "hermetic_matrix": "a1b2c3d4e5f6...",
  "perf_gate": "f6e5d4c3b2a1...",
  "verification_gate": "1a2b3c4d5e6f..."
}
```

#### 1.2 Key Storage Strategy

**Option A: Environment Variables (Development)**
```bash
export VERIFICATION_GATE_PRIVKEY="$(cat artifacts/keys/verification_gate.priv)"
export HERMETIC_MATRIX_PRIVKEY="$(cat artifacts/keys/hermetic_matrix.priv)"
export PERF_GATE_PRIVKEY="$(cat artifacts/keys/perf_gate.priv)"
```

**Option B: GitHub Secrets (CI)**
- Store private keys as GitHub repository secrets
- Access via `${{ secrets.VERIFICATION_GATE_PRIVKEY }}`
- Rotate keys quarterly using GitHub UI

**Option C: Secrets Manager (Production)**
- AWS Secrets Manager, HashiCorp Vault, or Azure Key Vault
- Automatic rotation with versioning
- Audit logging for all key access

**Recommendation:** Start with Option A for development, migrate to Option B for CI, and Option C for production.

#### 1.3 Dual-Mode Signature Generation

**File:** `scripts/generate_allblue_epoch_seal.py` (extend)

**New Function:** `compute_witness_signatures_ed25519()`

**Logic:**
```python
def compute_witness_signatures_ed25519(lane_results, H_t_hash):
    """
    Compute witness signatures using Ed25519 keypairs.
    Falls back to SHA256 if private keys unavailable.
    """
    witnesses = {}
    
    # Verification Gate
    if 'triple_hash' in lane_results['lanes']:
        lane = lane_results['lanes']['triple_hash']
        if lane['status'] == 'PASS' and lane['hash']:
            data = f"verification_gate:{lane['hash']}:{H_t_hash or 'ABSTAIN'}"
            
            # Try Ed25519 signature
            privkey = os.getenv('VERIFICATION_GATE_PRIVKEY')
            if privkey:
                sig = ed25519_sign(privkey, data.encode('utf-8'))
                witnesses['verification_gate_sig'] = sig.hex()
            else:
                # Fallback to SHA256
                sig = hashlib.sha256(data.encode('utf-8')).hexdigest()
                witnesses['verification_gate_sig'] = f"sha256:{sig}"
        else:
            witnesses['verification_gate_sig'] = f"ABSTAIN:{lane['status_message']}"
    
    # Similar logic for hermetic_matrix and perf_gate
    return witnesses
```

**Signature Format:**
- Ed25519: `ed25519:<128-hex-signature>`
- SHA256 (fallback): `sha256:<64-hex-hash>`
- ABSTAIN: `ABSTAIN:<reason>`

#### 1.4 Dual-Mode Signature Verification

**File:** `scripts/generate_allblue_epoch_seal.py` (extend)

**New Function:** `verify_witness_signatures_ed25519()`

**Logic:**
```python
def verify_witness_signatures_ed25519(witnesses, lane_results, H_t_hash, pubkeys):
    """
    Verify witness signatures using Ed25519 public keys.
    Falls back to SHA256 verification if signature is SHA256 format.
    """
    verification_results = {
        "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
        "overall_status": "PASS",
        "witnesses": {}
    }
    
    # Verification Gate
    if 'triple_hash' in lane_results['lanes']:
        lane = lane_results['lanes']['triple_hash']
        if lane['status'] == 'PASS' and lane['hash']:
            data = f"verification_gate:{lane['hash']}:{H_t_hash or 'ABSTAIN'}"
            actual_sig = witnesses.get('verification_gate_sig', '')
            
            if actual_sig.startswith('ed25519:'):
                # Ed25519 verification
                sig_hex = actual_sig.split(':', 1)[1]
                pubkey_hex = pubkeys.get('verification_gate')
                if pubkey_hex:
                    valid = ed25519_verify(pubkey_hex, data.encode('utf-8'), sig_hex)
                    if valid:
                        verification_results['witnesses']['verification_gate'] = {
                            "status": "PASS",
                            "signature_type": "ed25519",
                            "pubkey_fingerprint": pubkey_hex[:16],
                            "lane": "triple_hash"
                        }
                    else:
                        verification_results['witnesses']['verification_gate'] = {
                            "status": "FAIL",
                            "reason": "signature_verification_failed",
                            "signature_type": "ed25519",
                            "lane": "triple_hash"
                        }
                        verification_results['overall_status'] = "FAIL"
                else:
                    verification_results['witnesses']['verification_gate'] = {
                        "status": "ABSTAIN",
                        "reason": "pubkey_not_found",
                        "lane": "triple_hash"
                    }
                    verification_results['overall_status'] = "ABSTAIN"
            
            elif actual_sig.startswith('sha256:'):
                # SHA256 fallback verification
                sig_hex = actual_sig.split(':', 1)[1]
                expected = hashlib.sha256(data.encode('utf-8')).hexdigest()
                if sig_hex == expected:
                    verification_results['witnesses']['verification_gate'] = {
                        "status": "PASS",
                        "signature_type": "sha256",
                        "lane": "triple_hash"
                    }
                else:
                    verification_results['witnesses']['verification_gate'] = {
                        "status": "FAIL",
                        "reason": "hash_mismatch",
                        "signature_type": "sha256",
                        "expected": expected,
                        "actual": sig_hex,
                        "lane": "triple_hash"
                    }
                    verification_results['overall_status'] = "FAIL"
            
            elif actual_sig.startswith('ABSTAIN:'):
                verification_results['witnesses']['verification_gate'] = {
                    "status": "ABSTAIN",
                    "reason": actual_sig.split(':', 1)[1],
                    "lane": "triple_hash"
                }
                verification_results['overall_status'] = "ABSTAIN"
    
    # Similar logic for hermetic_matrix and perf_gate
    return verification_results
```

#### 1.5 Pass-Lines Extension

**New Pass-Lines:**
```
[PASS] Epoch Seal <sha256>
[PASS] Witnessed Epoch <sha256>
[PASS] Witnesses Verified (ed25519)
[PASS] Witnesses Verified (sha256)  # Fallback mode
```

**Granular ABSTAIN:**
```
[ABSTAIN] missing witness (lane=verification_gate, reason=triple_hash_not_verified)
[ABSTAIN] missing pubkey (lane=hermetic_matrix, reason=pubkey_not_in_config)
```

**Granular FAIL:**
```
[FAIL] witness verification failed (lane=perf_gate, reason=signature_verification_failed, type=ed25519)
[FAIL] witness verification failed (lane=verification_gate, reason=hash_mismatch, type=sha256)
```

### Acceptance Criteria

- [ ] Keypair generation script creates valid Ed25519 keypairs
- [ ] Public keys stored in RFC 8785 canonicalized format
- [ ] Private keys stored securely (environment variables or secrets manager)
- [ ] Dual-mode signature generation works (Ed25519 + SHA256 fallback)
- [ ] Dual-mode signature verification works (Ed25519 + SHA256 fallback)
- [ ] Pass-lines distinguish between Ed25519 and SHA256 modes
- [ ] ABSTAIN when private keys unavailable (not FAIL)
- [ ] Backward compatibility maintained (SHA256-only mode still works)

## Phase 2: Key Rotation Infrastructure

### Objectives

- Implement key versioning scheme
- Add rotation cadence (quarterly recommended)
- Maintain rotation log for audit trail
- Support multiple active key versions during rotation window

### Deliverables

#### 2.1 Key Versioning Scheme

**Format:** `{witness_name}_v{version}_{timestamp}`

**Example:**
```
verification_gate_v1_20251101
verification_gate_v2_20260201
hermetic_matrix_v1_20251101
hermetic_matrix_v2_20260201
```

**Public Key Config (Versioned):**
```json
{
  "hermetic_matrix": {
    "active_version": "v2",
    "keys": {
      "v1": {
        "pubkey": "a1b2c3d4e5f6...",
        "created": "2025-11-01T00:00:00Z",
        "expires": "2026-02-01T00:00:00Z",
        "status": "expired"
      },
      "v2": {
        "pubkey": "f6e5d4c3b2a1...",
        "created": "2026-02-01T00:00:00Z",
        "expires": "2026-05-01T00:00:00Z",
        "status": "active"
      }
    }
  },
  "perf_gate": { ... },
  "verification_gate": { ... }
}
```

#### 2.2 Rotation Cadence

**Recommended Schedule:** Quarterly (every 3 months)

**Rotation Windows:**
- Q1: February 1 - February 7 (7-day overlap)
- Q2: May 1 - May 7
- Q3: August 1 - August 7
- Q4: November 1 - November 7

**Overlap Period:** During rotation window, both old and new keys are valid. This allows in-flight epochs to complete with old keys while new epochs use new keys.

#### 2.3 Rotation Log

**File:** `artifacts/keys/rotation_log.jsonl`

**Format (RFC 8785 JSONL):**
```json
{"timestamp":"2025-11-01T00:00:00Z","witness":"verification_gate","action":"create","version":"v1","pubkey_fingerprint":"1a2b3c4d5e6f","expires":"2026-02-01T00:00:00Z"}
{"timestamp":"2026-02-01T00:00:00Z","witness":"verification_gate","action":"rotate","old_version":"v1","new_version":"v2","pubkey_fingerprint":"6f5e4d3c2b1a","expires":"2026-05-01T00:00:00Z"}
{"timestamp":"2026-02-07T00:00:00Z","witness":"verification_gate","action":"expire","version":"v1","reason":"rotation_complete"}
```

#### 2.4 Rotation Script

**File:** `scripts/rotate_witness_keys.py`

**Usage:**
```bash
python3 scripts/rotate_witness_keys.py \
  --witness verification_gate \
  --output-dir artifacts/keys/ \
  --overlap-days 7
```

**Functionality:**
1. Generate new Ed25519 keypair (version N+1)
2. Update `config/witness_pubkeys.json` with new key
3. Mark old key as "expiring" (not "expired" yet)
4. Append rotation event to `artifacts/keys/rotation_log.jsonl`
5. After overlap period, mark old key as "expired"

#### 2.5 Verification with Key Versions

**Logic:**
- Check witness signature version (embedded in signature or metadata)
- Look up corresponding public key version in config
- Verify signature using versioned public key
- ABSTAIN if key version not found or expired

### Acceptance Criteria

- [ ] Key versioning scheme implemented
- [ ] Rotation script generates new keypairs and updates config
- [ ] Rotation log records all key lifecycle events
- [ ] Overlap period allows both old and new keys to be valid
- [ ] Verification supports multiple active key versions
- [ ] Expired keys rejected with ABSTAIN status

## Phase 3: Merkle Tree Integration

### Objectives

- Build Merkle tree over witness signatures
- Include Merkle root in witnessed epoch seal
- Generate Merkle proofs for individual witness verification
- Enable efficient partial verification without full witness set

### Deliverables

#### 3.1 Merkle Tree Construction

**File:** `scripts/generate_allblue_epoch_seal.py` (extend)

**New Function:** `build_witness_merkle_tree()`

**Logic:**
```python
def build_witness_merkle_tree(witnesses):
    """
    Build Merkle tree over witness signatures.
    Leaves: sha256(witness_name + ":" + signature)
    Nodes: sha256(left_hash + right_hash)
    """
    leaves = []
    for witness_name in sorted(witnesses.keys()):
        sig = witnesses[witness_name]
        leaf_data = f"{witness_name}:{sig}"
        leaf_hash = hashlib.sha256(leaf_data.encode('utf-8')).hexdigest()
        leaves.append({
            "witness": witness_name,
            "signature": sig,
            "leaf_hash": leaf_hash
        })
    
    # Build tree (assuming 3 witnesses, need 4 leaves for balanced tree)
    # Pad with empty leaf if needed
    while len(leaves) < 4:
        leaves.append({
            "witness": "empty",
            "signature": "EMPTY",
            "leaf_hash": hashlib.sha256(b"EMPTY").hexdigest()
        })
    
    # Level 1: Leaf hashes
    level1 = [leaf['leaf_hash'] for leaf in leaves]
    
    # Level 2: Intermediate nodes
    level2 = [
        hashlib.sha256((level1[0] + level1[1]).encode('utf-8')).hexdigest(),
        hashlib.sha256((level1[2] + level1[3]).encode('utf-8')).hexdigest()
    ]
    
    # Level 3: Root
    merkle_root = hashlib.sha256((level2[0] + level2[1]).encode('utf-8')).hexdigest()
    
    return {
        "merkle_root": merkle_root,
        "leaves": leaves,
        "intermediate_nodes": level2,
        "tree_height": 3
    }
```

#### 3.2 Merkle Proof Generation

**File:** `scripts/generate_witness_merkle_proof.py` (new)

**Usage:**
```bash
python3 scripts/generate_witness_merkle_proof.py \
  --fleet-state artifacts/allblue/fleet_state.json \
  --witness verification_gate \
  --output artifacts/allblue/merkle_proof_verification_gate.json
```

**Proof Format (RFC 8785):**
```json
{
  "merkle_root": "a1b2c3d4e5f6...",
  "witness": "verification_gate",
  "leaf_hash": "1a2b3c4d5e6f...",
  "proof_path": [
    {"position": "right", "hash": "f6e5d4c3b2a1..."},
    {"position": "left", "hash": "6f5e4d3c2b1a..."}
  ],
  "tree_height": 3
}
```

#### 3.3 Merkle Proof Verification

**File:** `scripts/verify_witness_merkle_proof.py` (new)

**Usage:**
```bash
python3 scripts/verify_witness_merkle_proof.py \
  --proof artifacts/allblue/merkle_proof_verification_gate.json \
  --expected-root a1b2c3d4e5f6...
```

**Logic:**
```python
def verify_merkle_proof(proof, expected_root):
    """
    Verify Merkle proof by recomputing root from leaf and proof path.
    """
    current_hash = proof['leaf_hash']
    
    for node in proof['proof_path']:
        if node['position'] == 'right':
            current_hash = hashlib.sha256((current_hash + node['hash']).encode('utf-8')).hexdigest()
        else:
            current_hash = hashlib.sha256((node['hash'] + current_hash).encode('utf-8')).hexdigest()
    
    return current_hash == expected_root
```

**Pass-Lines:**
```
[PASS] Merkle Proof Verified (witness=verification_gate, root=<sha256>)
[FAIL] Merkle Proof Failed (witness=verification_gate, expected=<sha256>, actual=<sha256>)
```

#### 3.4 Witnessed Epoch Seal with Merkle Root

**Extended Fleet State:**
```json
{
  "epoch_hash": "...",
  "witnessed_epoch_hash": "...",
  "witnesses": {
    "verification_gate_sig": "ed25519:...",
    "hermetic_matrix_sig": "ed25519:...",
    "perf_gate_sig": "ed25519:..."
  },
  "merkle_tree": {
    "merkle_root": "a1b2c3d4e5f6...",
    "tree_height": 3,
    "leaf_count": 3
  }
}
```

**Witnessed Epoch Hash Computation (Extended):**
```python
witnessed_epoch_hash = sha256(RFC8785({
  "epoch_core": epoch_hash,
  "witnesses": witnesses,
  "merkle_root": merkle_root
}))
```

### Acceptance Criteria

- [ ] Merkle tree construction over witness signatures
- [ ] Merkle root included in witnessed epoch seal
- [ ] Merkle proof generation for individual witnesses
- [ ] Merkle proof verification script works correctly
- [ ] Partial verification possible without full witness set
- [ ] Pass-lines distinguish Merkle proof verification

## Phase 4: Integration & Testing

### Objectives

- Integrate Ed25519 + Merkle into CI workflows
- Test key rotation in staging environment
- Validate backward compatibility with SHA256 mode
- Document migration path from SHA256 to Ed25519

### Deliverables

#### 4.1 CI Workflow Integration

**File:** `.github/workflows/allblue-epoch-seal.yml` (extend)

**New Steps:**
```yaml
- name: Load Witness Private Keys
  env:
    VERIFICATION_GATE_PRIVKEY: ${{ secrets.VERIFICATION_GATE_PRIVKEY }}
    HERMETIC_MATRIX_PRIVKEY: ${{ secrets.HERMETIC_MATRIX_PRIVKEY }}
    PERF_GATE_PRIVKEY: ${{ secrets.PERF_GATE_PRIVKEY }}
  run: |
    # Keys loaded from secrets, available to epoch seal generation

- name: Generate Epoch Seal (Ed25519 Mode)
  run: |
    python3 scripts/generate_allblue_epoch_seal.py \
      --config config/allblue_lanes.json \
      --output-dir artifacts/allblue \
      --rfcsign \
      --signature-mode ed25519

- name: Verify Merkle Proofs
  run: |
    for witness in verification_gate hermetic_matrix perf_gate; do
      python3 scripts/generate_witness_merkle_proof.py \
        --fleet-state artifacts/allblue/fleet_state.json \
        --witness $witness \
        --output artifacts/allblue/merkle_proof_$witness.json
      
      python3 scripts/verify_witness_merkle_proof.py \
        --proof artifacts/allblue/merkle_proof_$witness.json \
        --expected-root $(jq -r '.merkle_tree.merkle_root' artifacts/allblue/fleet_state.json)
    done
```

#### 4.2 Key Rotation Testing

**Staging Environment:**
1. Generate initial keypairs (v1)
2. Run epoch seal generation with v1 keys
3. Rotate keys to v2 (with 7-day overlap)
4. Run epoch seal generation with v2 keys
5. Verify both v1 and v2 signatures during overlap
6. Expire v1 keys after overlap
7. Verify v1 signatures rejected after expiration

#### 4.3 Backward Compatibility Testing

**Test Cases:**
1. **SHA256-only mode:** No private keys available, fallback to SHA256 hashing
2. **Mixed mode:** Some witnesses use Ed25519, others use SHA256
3. **Ed25519-only mode:** All witnesses use Ed25519 signatures
4. **Verification with missing pubkeys:** ABSTAIN when pubkey not in config
5. **Verification with expired keys:** ABSTAIN when key version expired

#### 4.4 Migration Documentation

**File:** `docs/CRYPTO_MIGRATION.md`

**Contents:**
1. **Pre-Migration Checklist:** Backup existing artifacts, test keypair generation
2. **Migration Steps:** Generate keypairs, update CI secrets, enable Ed25519 mode
3. **Rollback Procedure:** Revert to SHA256 mode if issues detected
4. **Verification:** Confirm Ed25519 signatures in epoch registry
5. **Post-Migration:** Monitor for signature verification failures

### Acceptance Criteria

- [ ] CI workflow loads private keys from GitHub secrets
- [ ] Epoch seal generation works in Ed25519 mode
- [ ] Merkle proof generation and verification in CI
- [ ] Key rotation tested in staging environment
- [ ] Backward compatibility verified (SHA256 fallback works)
- [ ] Migration documentation complete

## Security Considerations

### Private Key Protection

1. **Never Commit Private Keys:** Add `*.priv` to `.gitignore`
2. **Encrypt at Rest:** Use secrets manager encryption
3. **Rotate Regularly:** Quarterly rotation recommended
4. **Audit Access:** Log all private key access
5. **Principle of Least Privilege:** Only CI workflows need private keys

### Public Key Distribution

1. **Commit Public Keys:** Public keys in `config/witness_pubkeys.json` are safe to commit
2. **Verify Integrity:** Check SHA256 hash of public key config file
3. **Archive Old Keys:** Keep expired public keys for historical verification
4. **Publish Fingerprints:** Include public key fingerprints in documentation

### Signature Verification

1. **Always Verify:** Never trust signatures without verification
2. **Check Expiration:** Reject signatures from expired keys
3. **Validate Merkle Proofs:** Verify Merkle proofs against root
4. **Fail Closed:** ABSTAIN or FAIL when verification uncertain

## Performance Considerations

### Ed25519 Performance

- **Signing:** ~50,000 signatures/second (single core)
- **Verification:** ~20,000 verifications/second (single core)
- **Key Generation:** ~10,000 keypairs/second

**Impact:** Negligible performance impact compared to CI workflow overhead.

### Merkle Tree Performance

- **Construction:** O(n) where n = number of witnesses (3 in our case)
- **Proof Generation:** O(log n) = O(1) for 3 witnesses
- **Proof Verification:** O(log n) = O(1) for 3 witnesses

**Impact:** Negligible performance impact.

### Storage Overhead

- **Ed25519 Signature:** 64 bytes (128 hex characters)
- **SHA256 Hash:** 32 bytes (64 hex characters)
- **Merkle Proof:** ~128 bytes per witness

**Impact:** Minimal storage overhead (~200 bytes per epoch).

## Rollout Timeline

### Month 1: Phase 1 (Ed25519 Infrastructure)

- Week 1: Keypair generation script
- Week 2: Dual-mode signature generation
- Week 3: Dual-mode signature verification
- Week 4: Testing and documentation

### Month 2: Phase 2 (Key Rotation)

- Week 1: Key versioning scheme
- Week 2: Rotation script and log
- Week 3: Verification with key versions
- Week 4: Testing and documentation

### Month 3: Phase 3 (Merkle Trees)

- Week 1: Merkle tree construction
- Week 2: Merkle proof generation
- Week 3: Merkle proof verification
- Week 4: Testing and documentation

### Month 4: Phase 4 (Integration & Migration)

- Week 1: CI workflow integration
- Week 2: Staging environment testing
- Week 3: Production migration
- Week 4: Monitoring and validation

## Success Metrics

- [ ] 100% of epoch seals use Ed25519 signatures (no SHA256 fallback)
- [ ] Zero signature verification failures in production
- [ ] Key rotation completes successfully every quarter
- [ ] Merkle proof verification works for all witnesses
- [ ] Backward compatibility maintained (SHA256 mode still works)
- [ ] Documentation complete and accurate

## References

- **RFC 8032:** Edwards-Curve Digital Signature Algorithm (EdDSA)
- **RFC 8785:** JSON Canonicalization Scheme (JCS)
- **RFC 6962:** Certificate Transparency (Merkle Tree Specification)
- **NIST FIPS 186-5:** Digital Signature Standard (DSS)

## Contact

For questions about this roadmap:
- `docs/ci/ALLBLUE_MANUAL_APPLY.md` - Current implementation
- `scripts/generate_allblue_epoch_seal.py` - Epoch seal generator
- `config/witness_pubkeys.json` - Public key configuration
