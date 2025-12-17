# Evaluator 10-Minute Check

**Time**: ~10 minutes
**Prerequisites**: Python 3.11+, Git

This guide lets you verify the core integrity claims in under 10 minutes.

---

## Step 1: Clone and Install (2 min)

```bash
git clone https://github.com/helpfuldolphin/mathledger.git
cd mathledger
```

Install dependencies (uses `uv` package manager):

```bash
# If you don't have uv installed:
pip install uv

# Install project dependencies:
uv sync
```

---

## Step 2: Run Mock Determinism Check (2 min)

This verifies the core loop produces identical outputs across runs:

```bash
make verify-mock-determinism
```

**Expected output**:
```
Mock determinism verification PASSED
All hashes match across runs
```

If you don't have `make`, run directly:

```bash
uv run python scripts/verify_first_light_determinism.py --mode mock
```

---

## Step 3: Generate a Signing Key (1 min)

Create a test keypair for signing:

```bash
uv run python scripts/generate_signing_keypair.py --output-dir tmp_keys --name eval_test
```

**Expected output**:
```
Ed25519 keypair generated successfully.

Private key: tmp_keys/eval_test_private.pem
Public key:  tmp_keys/eval_test_public.pem

SECURITY NOTICE:
  - NEVER commit the private key to version control
```

---

## Step 4: Create and Sign a Test Manifest (2 min)

Create a simple manifest:

```bash
echo '{"schema_version": "1.0.0", "mode": "SHADOW", "files": []}' > tmp_keys/manifest.json
```

Sign it:

```bash
uv run python scripts/sign_manifest.py \
    --manifest tmp_keys/manifest.json \
    --key tmp_keys/eval_test_private.pem
```

**Expected output**:
```
Manifest: tmp_keys/manifest.json
Signature: tmp_keys/manifest.json.sig
Signature size: 64 bytes
Status: SIGNED
```

Verify the signature:

```bash
uv run python scripts/verify_manifest_signature.py \
    --manifest tmp_keys/manifest.json \
    --pubkey tmp_keys/eval_test_public.pem
```

**Expected output**:
```
Manifest: tmp_keys/manifest.json
Signature: tmp_keys/manifest.json.sig
Public key: tmp_keys/eval_test_public.pem
Status: VERIFIED
```

---

## Step 5: Tamper and Verify Failure (2 min)

Now modify one character in the manifest:

```bash
# Change "1.0.0" to "1.0.1"
sed -i 's/1.0.0/1.0.1/' tmp_keys/manifest.json
```

On Windows (PowerShell):
```powershell
(Get-Content tmp_keys/manifest.json) -replace '1.0.0','1.0.1' | Set-Content tmp_keys/manifest.json
```

Verify again — this time it should FAIL:

```bash
uv run python scripts/verify_manifest_signature.py \
    --manifest tmp_keys/manifest.json \
    --pubkey tmp_keys/eval_test_public.pem
```

**Expected output**:
```
Manifest: tmp_keys/manifest.json
Signature: tmp_keys/manifest.json.sig
Public key: tmp_keys/eval_test_public.pem
Status: INVALID
The signature does not match the manifest content.
```

**Exit code**: 1 (failure)

---

## Step 6: Cleanup (30 sec)

```bash
rm -rf tmp_keys
```

---

## What You Just Verified

| Check | What It Proves |
|-------|----------------|
| Mock determinism | Core loop produces reproducible outputs |
| Sign + Verify | Cryptographic signatures work correctly |
| Tamper detection | Any modification invalidates the signature |

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `make verify-mock-determinism` | Check deterministic outputs |
| `uv run python scripts/generate_signing_keypair.py --output-dir DIR --name NAME` | Generate keypair |
| `uv run python scripts/sign_manifest.py --manifest FILE --key PRIVATE_KEY` | Sign manifest |
| `uv run python scripts/verify_manifest_signature.py --manifest FILE --pubkey PUBLIC_KEY` | Verify signature |

---

## Troubleshooting

**"uv not found"**: Install with `pip install uv`

**"make not found"**: Run the Python commands directly (shown in each section)

**Verification fails unexpectedly**: Ensure you're in the repository root directory

---

## Next Steps

- Read `docs/EVALUATOR_GUIDE.md` for full verification procedures
- Run `make verify-lean-single PROOF=<path>` to verify a specific Lean proof
- Review `docs/system_law/SHADOW_MODE_CONTRACT.md` for governance semantics

