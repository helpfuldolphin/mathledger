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

## 6. End-to-End Integration Tests

```bash
python -m pytest tests/evidence/test_manifest_signing_e2e.py -v
```

**Expected Output**:
```
tests/evidence/test_manifest_signing_e2e.py::TestManifestSigningE2E::test_e2e_sign_verify_tamper_workflow PASSED
tests/evidence/test_manifest_signing_e2e.py::TestManifestSigningE2E::test_e2e_single_bit_flip_detected PASSED
tests/evidence/test_manifest_signing_e2e.py::TestManifestSigningE2E::test_e2e_missing_signature_fails PASSED
tests/evidence/test_manifest_signing_e2e.py::TestManifestSigningE2E::test_e2e_signature_not_transferable PASSED
tests/evidence/test_manifest_signing_e2e.py::TestKeyGeneration::test_e2e_generate_keypair_via_script PASSED
tests/evidence/test_manifest_signing_e2e.py::TestKeyGeneration::test_e2e_generated_keypair_works_for_signing PASSED
tests/evidence/test_manifest_signing_e2e.py::TestFailClose::test_verification_fails_on_corrupted_signature PASSED
tests/evidence/test_manifest_signing_e2e.py::TestFailClose::test_verification_fails_with_truncated_signature PASSED

8 passed
```

**Key Tests**:
| Test | Behavior Verified |
|------|-------------------|
| `test_e2e_sign_verify_tamper_workflow` | Full workflow: sign → verify → tamper → verify fails |
| `test_e2e_single_bit_flip_detected` | Even 1-bit change detected |
| `test_e2e_missing_signature_fails` | Fail-close on missing signature |
| `test_e2e_signature_not_transferable` | Signature bound to specific manifest |

---

## 7. Verify .gitignore Blocks Private Keys

```bash
git check-ignore -v keys/test_private.pem tmp_keys/anything *.private.pem
```

**Expected Output**:
```
.gitignore:15:keys/*_private.pem    keys/test_private.pem
.gitignore:17:tmp_keys/             tmp_keys/anything
.gitignore:16:*.private.pem         test.private.pem
```

- [ ] All private key patterns are blocked

---

## 8. Cleanup

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
| Unit Tests (11 tests) | [ ] PASS / [ ] FAIL |
| E2E Integration Tests (8 tests) | [ ] PASS / [ ] FAIL |
| .gitignore Verification | [ ] PASS / [ ] FAIL |

**Overall**: [ ] PASS / [ ] FAIL

**Tested By**: _______________
**Date**: _______________

---

**Version**: 1.1.0

