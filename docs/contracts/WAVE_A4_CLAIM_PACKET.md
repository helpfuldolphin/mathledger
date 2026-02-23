# Wave A4 Claim Packet

## Claim A4

Claim ID: `ML-A4-EVIDENCE-REPLAY`  
Tier: A  
Statement: `U_t`, `R_t`, and `H_t` replay recompute exactly from evidence-pack artifacts.

Falsifier:
- unchanged evidence pack replays to different roots
- tampered evidence pack replays to the same roots
- ADV artifacts influence `R_t` authority root

Closure artifact targets:
- `src/mathledger/evidence/replay.py`
- `tests/test_wave_a4_evidence_replay.py`

## Invariants Enforced in A4

1. Deterministic replay:
- same evidence inputs recompute identical `U_t`, `R_t`, `H_t`.

2. Composite contract:
- `H_t` is computed only as `SHA256(R_t || U_t)`.

3. ADV quarantine in replay:
- ADV artifacts may exist in the evidence bundle but are excluded from `R_t`.

4. Fail-closed schema contract:
- malformed evidence packs raise typed `EvidenceReplayViolation`.

5. Tamper visibility:
- mutation of UVIL payloads, reasoning payloads, or stored roots yields replay mismatch.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `39 passed`
