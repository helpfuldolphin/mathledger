# Wave A2 Claim Packet

## Claim A2

Claim ID: `ML-A2-GOV-KERNEL`  
Tier: A  
Statement: Trust class routing and typed abstention preservation are structurally enforced.

Falsifier:
- `ADV` artifact enters authority-bearing reasoning stream
- authority artifact lacks `validation_outcome`
- `validation_outcome` is null or invalid

Closure artifact targets:
- `src/mathledger/governance/trust_class.py`
- `src/mathledger/governance/routing.py`
- `src/mathledger/governance/abstention.py`
- `tests/test_wave_a2_governance.py`

## Invariants Enforced in A2

1. ADV quarantine:
- advisory claims must not enter authority stream.

2. Typed outcome preservation:
- every authority artifact must carry explicit `VERIFIED|REFUTED|ABSTAINED`.

3. Fail-closed behavior:
- missing or invalid trust class/outcome raises explicit violations.

4. Deterministic authority leaves:
- authority payloads serialize to canonical deterministic JSON leaves.

