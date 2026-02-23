# Wave A6 Claim Packet

## Claim A6

Claim ID: `ML-A6-CPF-ADAPTER`  
Tier: B  
Statement: CPF assessment outputs normalize into deterministic snapshot artifacts suitable for experiment contexts.

Falsifier:
- identical CPF payloads produce different snapshot IDs
- normalization is not idempotent on already-normalized data

Closure artifact targets:
- `src/mathledger/integration/cpf_adapter.py`
- `tests/test_wave_a5_a6_a7_integration.py`

## Invariants Enforced in A6

1. Deterministic snapshot identity:
- `snapshot_id` derives from canonical normalized indicator payload.

2. Fail-closed schema checks:
- invalid score/confidence/indicator formats are rejected.

3. Idempotent normalization:
- re-normalizing normalized indicator payloads preserves snapshot identity.

4. Canonical serialization:
- snapshot output has stable compact JSON encoding for replay.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `45 passed`
