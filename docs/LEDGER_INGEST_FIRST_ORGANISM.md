# First Organism Ingestion Contract

**Status**: Active
**Version**: 1.0
**Target**: First Organism (v1)

## Overview

This document defines the strict ingestion and sealing semantics for the **First Organism**. The backend guarantees that every sealed block adheres to the **Dual-Root Attestation** protocol as defined in the Whitepaper (§4.2).

## Core Semantics

### 1. Deduplication
- **Statements** are deduplicated by their **Canonical Hash** (`SHA256(normalize(content))`).
- **Proofs** are deduplicated by the tuple `(statement_id, prover, proof_hash)`.
  - `proof_hash` is `SHA256(canonical_json(payload))`.

### 2. Block Sealing (Dual-Root)
Every block is sealed with two distinct Merkle roots bound by a composite root:

$$
H_t = \text{SHA256}(R_t \parallel U_t)
$$

Where:
- **$R_t$ (Reasoning Root)**: Merkle root of the sorted list of proof hashes in the block.
- **$U_t$ (UI Root)**: Merkle root of the sorted list of UI event hashes.
- **$H_t$ (Composite Root)**: Cryptographic binding of the two streams.

### 3. Atomicity
- Ingestion of statements, proofs, and block sealing occurs within a **single database transaction**.
- This ensures that no partial blocks or orphaned proofs exist in the canonical chain.

## Helper Interface

For testing and verification, the `ledger.first_organism` module exposes:

```python
def ingest_and_seal_for_first_organism(
    cur: Cursor,
    result: Dict[str, Any],
    ui_events: Sequence[str]
) -> SealedBlock
```

### SealedBlock Structure
```python
@dataclass
class SealedBlock:
    reasoning_root: str  # R_t
    ui_root: str         # U_t
    composite_root: str  # H_t
    block_id: int
    sequence: int
    timestamp: str       # ISO 8601
```

## Verification Procedure

To verify the contract for any block:
1. **Fetch** the proofs and UI events for the block.
2. **Recompute** $R_t$ from the sorted proof hashes.
3. **Recompute** $U_t$ from the sorted UI event strings.
4. **Assert** that the stored $H_t$ equals $\text{SHA256}(R_t \parallel U_t)$.
5. **Assert** that the stored $R_t$ and $U_t$ match the recomputed values.

## Usage in Tests

```python
from ledger.first_organism import ingest_and_seal_for_first_organism

def test_contract(cur):
    outcome = ingest_and_seal_for_first_organism(cur, proof, events)
    # Assertions...
```

## Running the Verification Test

The contract test in `tests/test_first_organism_ledger.py` is gated on a live Postgres instance that has been migrated via `scripts/run-migrations.py`.
- Export `DATABASE_URL` or `DATABASE_URL_TEST` pointing to the First Organism database. See `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format.
- The test will **skip** (rather than fail) when the database is unreachable or migrations cannot run, so make sure the instance is running before enabling the integration suite.
