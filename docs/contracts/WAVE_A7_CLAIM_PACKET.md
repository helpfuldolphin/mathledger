# Wave A7 Claim Packet

## Claim A7

Claim ID: `ML-A7-EXPERIMENT-RUNNER`  
Tier: B  
Statement: A deterministic three-lane runner executes Lane-A authority, Lane-B forensic bridge, and Lane-C CPF context with explicit boundary assertions.

Falsifier:
- same run inputs produce different run IDs or Lane-A roots
- Lane-B roots diverge from Lane-A evidence without detection
- Lane-C snapshot ID drifts under deterministic recomputation

Closure artifact targets:
- `src/mathledger/integration/experiment_runner.py`
- `tests/test_wave_a5_a6_a7_integration.py`

## Invariants Enforced in A7

1. Reproducible execution:
- identical inputs produce identical run IDs and Lane-A attestation roots.

2. Lane-A integrity:
- runner requires evidence replay verification pass before publishing run artifact.

3. Lane-B boundary:
- AAK bridge is validated against Lane-A roots and forbidden authority fields.

4. Lane-C determinism:
- CPF snapshot IDs are rechecked from normalized indicators and epoch.

5. Boundary tamper detection:
- root-reference tampering or snapshot-ID drift raises fail-closed violations.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `45 passed`
