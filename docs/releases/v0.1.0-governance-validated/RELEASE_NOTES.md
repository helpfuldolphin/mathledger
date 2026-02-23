# Release Notes: v0.1.0-governance-validated

Date: 2026-02-23  
Scope: Governance pipeline validation checkpoint

## Included

1. Deterministic three-lane scenario suite execution (`9 scenarios`, `11 steps`).
2. Published adversarial fixtures:
   1. CISO firewall exception (Silicon Psyche 5.2.1)
   2. Emergency SSH bypass (Silicon Psyche 5.2.2)
   3. Fight-flight SOC scenario (Silicon Psyche 5.2.3)
   4. CEO fraud 3-step sequence (integration paper 6.1)
   5. CAC double-bind (integration paper 6.2)
3. Benign controls (`4`) yielding `GREEN` decisions.
4. Sealed per-step artifacts:
   1. `run_result.json`
   2. `evidence_pack.json`
   3. `aak_bridge_packet.json`
   4. `gate_evaluation.json`
   5. `step_result.json`

## Validation Results

1. `pytest -q` -> `49 passed`
2. Scenario suite expectations -> `11/11 matched`, `0 mismatches`
3. CEO fraud progression -> `YELLOW -> RED -> RED`
4. CAC scenario -> `RED` via both risk and convergence gates
5. Benign controls -> all `GREEN`

## Caveat

This checkpoint validates governance correctness given deterministic indicator activations. It does not yet validate live model response classification into CPF indicator scores.

## Reproducibility Artifact

File-level manifest with SHA256 hashes:
- [`release_manifest.json`](/C:/dev/mathledger/docs/releases/v0.1.0-governance-validated/release_manifest.json)

## Git Tag Status

No `.git` metadata exists in this workspace path (`C:\dev\mathledger`), so a cryptographic git tag cannot be created from this environment snapshot.
