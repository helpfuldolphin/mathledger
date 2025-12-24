# FOL_FIN_EQ_v1 Phase 3 Closure

**Status**: CLOSED
**Date**: 2025-12-24
**Golden Manifest SHA256**: `096ee79e4e20c94fffbc2ec9964dde98f8058cba47a887031085e0800d6d2113`

## Reproduction Command

```bash
python -m scripts.run_fol_fin_eq_demo --domain z2 --output demo_z2
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('demo_z2/manifest.json').read_bytes()).hexdigest())"
```

## Expected Output Tree

```
demo_z2/
├── certificates/
│   └── *.json  (one certificate per formula)
├── manifest.json
└── verify.py
```

## Phase 3 Scope

Phase 3 implements the certificate generation, schema validation, and evidence pack
toolchain for FOL_FIN_EQ_v1 (finite-domain first-order logic with equality). The
scope includes: `normalization/fol_certificate.py` (certificate generation + domain-
separated hashing), `governance/fol_schema_validator.py` (fail-closed schema validation),
`scripts/run_fol_fin_eq_demo.py` (deterministic evidence pack generation), and the
embedded standalone `verify.py` (stdlib-only verification script).

## Explicit Non-Claims

- **No SMT/SAT**: Verification is exhaustive enumeration only
- **No optimizations**: No early termination heuristics beyond spec-mandated short-circuits
- **No predicates**: Only functions + equality (no uninterpreted predicates)
- **Determinism required**: Byte-identical output for identical inputs

## Change Policy

Any change that alters the golden manifest hash requires Phase 3.1 with a new
fragment version identifier (e.g., `FOL_FIN_EQ_v1.1`).
