# Ledger

This package manages the canonical ledger surface:

- Block and Merkle tree builders
- Persistence adapters (Postgres schemas, migrations)
- CLI and service entrypoints that materialize ledger state

Every module must preserve deterministic replay and emit data traceable to attestation manifests.

