# Wave A5 Claim Packet

## Claim A5

Claim ID: `ML-A5-AAK-BRIDGE`  
Tier: A  
Statement: Lane-B AAK references may point to Lane-A roots, but cannot escalate authority semantics.

Falsifier:
- Lane-B payload injects authority-bearing keys (`trust_class`, `validation_outcome`, etc.)
- Lane-B references diverge from Lane-A `U_t`, `R_t`, `H_t`

Closure artifact targets:
- `src/mathledger/integration/aak_bridge.py`
- `tests/test_wave_a5_a6_a7_integration.py`

## Invariants Enforced in A5

1. Reference-only bridge:
- Lane-B psych context links to Lane-A roots without carrying governance route fields.

2. No authority escalation:
- forbidden authority keys are blocked fail-closed.

3. Root-link integrity:
- `reasoning_root_ref` and `ui_root_ref` must match verified Lane-A evidence roots.

4. Deterministic bridge identity:
- bridge packet computes stable `lane_b_reference_id` from canonical payload.

## Local Closure Evidence

- Date: 2026-02-23
- Command: `cd C:\dev\mathledger && pytest -q`
- Result: `45 passed`
