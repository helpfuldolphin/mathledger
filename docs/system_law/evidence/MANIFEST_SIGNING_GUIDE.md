# Manifest Signing Guide

**Document Type**: Operator Guide
**Status**: CANONICAL
**Version**: 1.0.0
**Date**: 2025-12-17

---

## 1. Purpose

This document describes the manifest signing workflow for evidence packs. Signing provides **provenance binding**: cryptographic proof that a specific manifest was produced by a holder of the signing key.

### What Signing Provides

| Property | Provided By |
|----------|-------------|
| Internal consistency | Hash verification (existing) |
| Provenance binding | Ed25519 signature (this feature) |
| Tamper detection | Signature verification fails if manifest modified |

### What Signing Does NOT Provide

| Property | Reason |
|----------|--------|
| Timestamp authority | No trusted timestamping service integrated |
| Identity verification | Public key binding to identity is out of scope |
| Content validation | Signing does not verify artifact correctness |

---

## 2. Key Management

### 2.1 Key Generation

Generate a new Ed25519 keypair:

```bash
python scripts/generate_signing_keypair.py --output-dir keys/ --name manifest_signing
```

This produces:
- `keys/manifest_signing_private.pem` - **KEEP SECRET**
- `keys/manifest_signing_public.pem` - Safe to distribute

### 2.2 Key Storage

| Environment | Private Key Storage | Public Key Storage |
|-------------|--------------------|--------------------|
| Development | Local directory (gitignored) | Repository or local |
| CI/CD | Secrets manager (GitHub Secrets, Vault) | Repository |
| Production | HSM or encrypted vault | Repository |

**CRITICAL**: Never commit private keys to version control.

### 2.3 Recommended .gitignore

Add to `.gitignore`:

```
# Signing keys
keys/*_private.pem
*.private.pem
```

---

## 3. Signing Workflow

### 3.1 Sign a Manifest

```bash
python scripts/sign_manifest.py \
    --manifest results/first_light/evidence_pack_first_light/manifest.json \
    --key keys/manifest_signing_private.pem
```

Output:
```
Manifest: results/first_light/evidence_pack_first_light/manifest.json
Signature: results/first_light/evidence_pack_first_light/manifest.json.sig
Signature size: 64 bytes
Status: SIGNED
```

### 3.2 Verify a Signature

```bash
python scripts/verify_manifest_signature.py \
    --manifest results/first_light/evidence_pack_first_light/manifest.json \
    --pubkey keys/manifest_signing_public.pem
```

Output (valid):
```
Manifest: results/first_light/evidence_pack_first_light/manifest.json
Signature: results/first_light/evidence_pack_first_light/manifest.json.sig
Public key: keys/manifest_signing_public.pem
Status: VERIFIED
```

Output (invalid):
```
Manifest: results/first_light/evidence_pack_first_light/manifest.json
Signature: results/first_light/evidence_pack_first_light/manifest.json.sig
Public key: keys/manifest_signing_public.pem
Status: INVALID
The signature does not match the manifest content.
```

### 3.3 Integrated Pipeline

The `generate_and_verify_evidence_pack.py` script supports signing:

```bash
# Generate, verify, and sign
python scripts/generate_and_verify_evidence_pack.py \
    --sign \
    --signing-key keys/manifest_signing_private.pem

# Verify existing pack with signature verification
python scripts/generate_and_verify_evidence_pack.py \
    --verify-only \
    --verify-signature \
    --pubkey keys/manifest_signing_public.pem
```

---

## 4. Exit Codes

### sign_manifest.py

| Exit Code | Meaning |
|-----------|---------|
| 0 | Signature created successfully |
| 1 | Signing failed (missing files, invalid key) |
| 2 | Configuration error (missing arguments) |

### verify_manifest_signature.py

| Exit Code | Meaning |
|-----------|---------|
| 0 | Signature verified successfully |
| 1 | Signature verification failed |
| 2 | Configuration error (missing files, invalid key format) |

---

## 5. File Formats

### 5.1 Signature File

The signature file (`manifest.json.sig`) contains raw 64-byte Ed25519 signature data (binary, not base64).

### 5.2 Key Files

Keys are stored in PEM format:

**Private Key** (`*_private.pem`):
```
-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEI...
-----END PRIVATE KEY-----
```

**Public Key** (`*_public.pem`):
```
-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEA...
-----END PUBLIC KEY-----
```

---

## 6. Cryptographic Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Ed25519 |
| Signature size | 64 bytes |
| Public key size | 32 bytes |
| Private key size | 32 bytes |
| Security level | ~128-bit |

Ed25519 is a modern, deterministic signature scheme with:
- No random number generation during signing
- Fast verification
- Small signatures
- Resistance to side-channel attacks

---

## 7. Organizational Key Setup

For organizational use (multiple signers, key rotation):

### 7.1 Development Keys

Generate per-developer keys for local testing:

```bash
python scripts/generate_signing_keypair.py --output-dir ~/.mathledger/keys --name dev_$(whoami)
```

### 7.2 CI/CD Keys

1. Generate organizational key:
   ```bash
   python scripts/generate_signing_keypair.py --output-dir /secure/location --name ci_signing
   ```

2. Store private key in secrets manager:
   - GitHub: Repository Settings > Secrets > `MANIFEST_SIGNING_KEY`
   - GitLab: CI/CD Settings > Variables > `MANIFEST_SIGNING_KEY`

3. Commit public key to repository:
   ```bash
   cp /secure/location/ci_signing_public.pem keys/ci_signing_public.pem
   git add keys/ci_signing_public.pem
   git commit -m "Add CI signing public key"
   ```

### 7.3 Key Rotation

When rotating keys:

1. Generate new keypair
2. Publish new public key
3. Re-sign existing evidence packs with new key
4. Update verification workflows to accept new public key
5. After transition period, revoke old key

---

## 8. Threat Model

### 8.1 What Signing Protects Against

| Threat | Protection |
|--------|------------|
| Post-hoc manifest modification | Signature verification fails |
| Manifest substitution | Different manifest produces different signature |
| Accidental corruption | Invalid signature detected |

### 8.2 What Signing Does NOT Protect Against

| Threat | Reason |
|--------|--------|
| Private key compromise | Attacker can forge signatures |
| Man-in-the-middle (key distribution) | Public key verification is out of scope |
| Signing malicious content | Signing only binds provenance, not validity |

---

## 9. SHADOW MODE Contract

Per SHADOW_MODE_CONTRACT.md:

- Signing scripts use standard exit codes
- Output is neutral and non-evaluative
- Signing failures do not block operations (SHADOW-OBSERVE)
- Verification results are informational unless explicitly gated

---

## 10. References

- [SHADOW_MODE_CONTRACT.md](../SHADOW_MODE_CONTRACT.md)
- [Evidence_Pack_Spec_PhaseX.md](../Evidence_Pack_Spec_PhaseX.md)
- [Ed25519 RFC 8032](https://tools.ietf.org/html/rfc8032)

---

**Version**: 1.0.0
**Owner**: STRATCOM Security Engineering

