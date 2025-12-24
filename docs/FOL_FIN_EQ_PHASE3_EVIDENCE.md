# FOL_FIN_EQ_v1 Phase 3 Evidence Pack

This document provides reproducible commands for generating and verifying
FOL_FIN_EQ_v1 evidence packs.

## Reproduction Command

```bash
uv run python -m scripts.run_fol_fin_eq_demo --domain z2 --output demo_z2
```

## Expected File Tree

```
demo_z2/
  certificates/
    z2_associativity_formula.json
    z2_identity_formula.json
    z2_inverse_formula.json
  manifest.json
  verify.py
```

## Golden Manifest SHA256

```
096ee79e4e20c94fffbc2ec9964dde98f8058cba47a887031085e0800d6d2113
```

Compute with:
```bash
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('demo_z2/manifest.json').read_bytes()).hexdigest())"
```

## Verification Command

```bash
python demo_z2/verify.py
```

Expected output:
```
PASS: All certificates verified
```
Exit code: 0

## Determinism Guarantee

Running the reproduction command twice produces byte-identical output:
```bash
uv run python -m scripts.run_fol_fin_eq_demo --domain z2 --output demo_a
uv run python -m scripts.run_fol_fin_eq_demo --domain z2 --output demo_b
diff -r demo_a demo_b  # Empty (no differences)
```

## Artifact Commit Policy

Evidence packs are NOT committed to the repository. Rationale:
1. They are fully reproducible from source (deterministic)
2. Committing generated artifacts creates merge conflicts
3. The manifest hash above serves as the canonical fingerprint

To verify a release, regenerate and compare the manifest hash.

## Phase 3 Modules

| Module | Purpose |
|--------|---------|
| `normalization/fol_certificate.py` | Certificate generation + hashing |
| `governance/fol_schema_validator.py` | Fail-closed schema validation |
| `scripts/run_fol_fin_eq_demo.py` | Evidence pack generation |

## Verification Semantics

verify.py validates in order:
1. Manifest schema (required fields, logic_fragment, verification_strategy)
2. Certificate schema (required fields, valid status enum, conditional fields)
3. Certificate hash (domain-separated SHA256 with DOMAIN_FOL_CERT)

All checks must pass. Any failure returns exit code 1.
