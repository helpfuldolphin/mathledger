# Manifest Signing Smoke Test Checklist

**Document Type**: Smoke Test Checklist
**Status**: CANONICAL
**Version**: 1.0.0
**Date**: 2025-12-17

---

## Pre-Requisites

- [ ] Python 3.11+ available
- [ ] `cryptography` package installed (`uv sync`)
- [ ] Working directory is repository root

---

## 1. Key Generation

### Test: Generate Keypair

```bash
python scripts/generate_signing_keypair.py --output-dir tmp_keys --name smoke_test
```

**Expected**:
- [ ] Exit code 0
- [ ] `tmp_keys/smoke_test_private.pem` exists
- [ ] `tmp_keys/smoke_test_public.pem` exists
- [ ] Output includes "SECURITY NOTICE"

### Test: Reject Overwrite Without Force

```bash
python scripts/generate_signing_keypair.py --output-dir tmp_keys --name smoke_test
```

**Expected**:
- [ ] Exit code 2
- [ ] Output includes "Use --force to overwrite"

---

## 2. Signing

### Test: Sign Sample Manifest

Create test manifest:
```bash
echo '{"test": "data"}' > tmp_keys/manifest.json
```

Sign:
```bash
python scripts/sign_manifest.py --manifest tmp_keys/manifest.json --key tmp_keys/smoke_test_private.pem
```

**Expected**:
- [ ] Exit code 0
- [ ] `tmp_keys/manifest.json.sig` exists
- [ ] Signature file is 64 bytes
- [ ] Output includes "Status: SIGNED"

### Test: Sign Missing Manifest

```bash
python scripts/sign_manifest.py --manifest nonexistent.json --key tmp_keys/smoke_test_private.pem
```

**Expected**:
- [ ] Exit code 1
- [ ] Output includes "Manifest not found"

### Test: Sign With Missing Key

```bash
python scripts/sign_manifest.py --manifest tmp_keys/manifest.json --key nonexistent.pem
```

**Expected**:
- [ ] Exit code 1
- [ ] Output includes "Private key not found"

---

## 3. Verification

### Test: Verify Valid Signature

```bash
python scripts/verify_manifest_signature.py --manifest tmp_keys/manifest.json --pubkey tmp_keys/smoke_test_public.pem
```

**Expected**:
- [ ] Exit code 0
- [ ] Output includes "Status: VERIFIED"

### Test: Verify After Tampering

Tamper with manifest:
```bash
echo '{"test": "tampered"}' > tmp_keys/manifest.json
```

Verify:
```bash
python scripts/verify_manifest_signature.py --manifest tmp_keys/manifest.json --pubkey tmp_keys/smoke_test_public.pem
```

**Expected**:
- [ ] Exit code 1
- [ ] Output includes "Status: INVALID"

### Test: Verify Missing Signature

```bash
rm tmp_keys/manifest.json.sig
python scripts/verify_manifest_signature.py --manifest tmp_keys/manifest.json --pubkey tmp_keys/smoke_test_public.pem
```

**Expected**:
- [ ] Exit code 1
- [ ] Output includes "Signature not found" or "SIGNATURE_MISSING"

### Test: Verify With Wrong Key

Generate different key and verify:
```bash
python scripts/generate_signing_keypair.py --output-dir tmp_keys --name wrong_key --force
echo '{"test": "data"}' > tmp_keys/manifest.json
python scripts/sign_manifest.py --manifest tmp_keys/manifest.json --key tmp_keys/smoke_test_private.pem
python scripts/verify_manifest_signature.py --manifest tmp_keys/manifest.json --pubkey tmp_keys/wrong_key_public.pem
```

**Expected**:
- [ ] Exit code 1
- [ ] Output includes "Status: INVALID"

---

## 4. Integrated Pipeline

### Test: Generate, Sign, Verify (Requires Evidence Pack)

If evidence pack exists at `results/first_light/evidence_pack_first_light`:

```bash
python scripts/generate_and_verify_evidence_pack.py \
    --verify-only \
    --sign \
    --signing-key tmp_keys/smoke_test_private.pem \
    --verify-signature \
    --pubkey tmp_keys/smoke_test_public.pem
```

**Expected**:
- [ ] PHASE 4: MANIFEST SIGNING appears
- [ ] SIGNING: SIGNED
- [ ] PHASE 5: SIGNATURE VERIFICATION appears
- [ ] SIGNATURE VERIFICATION: VERIFIED

---

## 5. Unit Tests

```bash
python -m pytest tests/evidence/test_manifest_signing.py -v
```

**Expected**:
- [ ] All 11 tests pass

---

## 6. Cleanup

```bash
rm -rf tmp_keys
```

---

## Smoke Test Result

| Section | Status |
|---------|--------|
| Key Generation | [ ] PASS / [ ] FAIL |
| Signing | [ ] PASS / [ ] FAIL |
| Verification | [ ] PASS / [ ] FAIL |
| Integrated Pipeline | [ ] PASS / [ ] FAIL / [ ] SKIP |
| Unit Tests | [ ] PASS / [ ] FAIL |

**Overall**: [ ] PASS / [ ] FAIL

**Tested By**: _______________
**Date**: _______________

---

**Version**: 1.0.0

