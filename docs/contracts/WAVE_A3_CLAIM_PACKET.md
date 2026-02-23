# Wave A3 Claim Packet

## Claim A3

Claim ID: `ML-A3-UVIL-BOUNDARY`  
Tier: A  
Statement: Exploration identifiers never enter committed authority identity paths.

Falsifier:
- `proposal_id` influences `claim_id`, `committed_partition_id`, or `event_id`
- committed payloads contain exploration keys (`proposal_id`, `draft_id`, `exploration_id`)

Closure artifact targets:
- `src/mathledger/uvil/models.py`
- `src/mathledger/uvil/boundary.py`
- `tests/test_wave_a3_uvil_boundary.py`

## Invariants Enforced in A3

1. Explicit commit content:
- committed IDs derive only from edited claims and commit metadata.

2. Exploration firewall:
- exploration keys are rejected in commit inputs and forbidden in committed payloads.

3. Fail-closed parsing:
- unknown claim input keys and empty claim text are rejected.

4. Deterministic commit artifacts:
- identical edited claims + epoch + user fingerprint produce identical IDs.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `30 passed`
