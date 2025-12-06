# Normalization

Canonicalization logic, encoding pipelines, and hash preparation belong here:

- Normalize ASTs into canonical byte representations
- Maintain reversible mappings (Unicode, proof terms, metadata)
- Provide deterministic hashing utilities consumed by ledger and attestation layers

All modules must be pure and idempotent to guarantee re-execution invariance.

