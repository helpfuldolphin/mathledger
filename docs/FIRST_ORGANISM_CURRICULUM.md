---
title: FIRST ORGANISM Curriculum Slice
---

# First Organism Curriculum Slice

**Author**: Cursor G — Curriculum Gatekeeper  
**Purpose**: Provide a traceable, gate-guarded curriculum entry for the First Organism integration test (PL → FOL → Equational → LIA ladder).

## Ladder Context

The MathLedger ladder progresses:

1. **PL (Propositional Logic)** – atoms/depth slices covering core propositional proof patterns.  
2. **FOL (First-Order Logic)** – quantifiers, predicates, and reflexive reasoning; the proof space begins to require instantiation and background knowledge.  
3. **Equational** – equality-rich statements where rewriting is dominant.  
4. **Arithmetic (QF-LIA/LRA)** – numeric domains with linear integer/real arithmetic; the hardest slices rely on numeric solvers and heavy search.

Each production slice enforces four gate families (coverage, abstention, velocity, caps) backed by canonical telemetry. The First Organism test cannot yet perform the full ladder, so it leverages a bespoke “trial” slice that sits within this hierarchy without skipping any gate.

## make_first_organism_slice()

`make_first_organism_slice()` returns a `CurriculumSlice` with:

- **Coverage gate**: high CI lower bound (≥0.915) that will fail given a single small derivation run.  
- **Abstention gate**: rate bound (≤18 %) and mass limit (≤640) that the test can satisfy even while the verifier abstains.  
- **Velocity gate**: modest proofs/hour (≥160 pph) with a relaxed CV (≤0.10) to let the test progress in a noisy environment.  
- **Caps gate**: ensures enough attempt mass (>2400), runtime (>20 min), and manageable backlog (<36 %) before ratcheting.

These thresholds intentionally permit **exactly one trial run**: the slice passes every gate except coverage, proving the gate engine was consulted but halting advancement until real coverage accumulates.

## build_first_organism_metrics()

This helper synthesizes a normalized metrics payload whose:

- Coverage CI sits below the requirement (failure).  
- Abstention rate/mass, velocity, backlog, and runtime all satisfy the gate to keep focus on the coverage failure.  
- Provenance contains an attestation hash, which is later reflected in the dual-root handshake (`Hₜ`) consumed by the RFL runner.

The integration test asserts the resulting `GateVerdict`:

1. Reports coverage failure with the correct gate reason.  
2. Lists passing entries for abstention, velocity, and caps.  
3. Includes an audit payload containing the attestation hash that must match the ledger/dual-root view later in the closed loop.

## Alignment with the Ladder

This slice is acceptable because it:

- Lives at the base of the ladder (PL-like depth 5, atoms 4).  
- Honors the same gate disciplines as production slices, so the test can prove the gate-checking plumbing without accelerating the curriculum.  
- Records the failure mode (coverage) that prevents ratcheting, ensuring the ladder stays locked until the organism demonstrates actual statistical coverage in later waves.

Future waves can reuse this canonical slice to bootstrap higher-level integrations once the organism demonstrates reproducible coverage, then graduate to the next slice (FOL) by crafting new presets with higher thresholds.

---

## Related Tests

- **Unit tests**: `tests/frontier/test_curriculum_gates.py`
  - `test_first_organism_slice_allows_run_but_holds_ratchet()` exercises the helper slice and metrics, asserting:
    - Coverage gate fails (CI below threshold).
    - Abstention, velocity, and caps gates pass.
    - Audit payload carries the expected attestation hash.
- **Integration test (forthcoming)**: `tests/integration/test_first_organism.py`
  - `test_first_organism_closed_loop()` wires the slice into the full UI → gate → derivation → Lean → dual-root → RFL chain, proving the curriculum gate is not stubbed.

---

## Determinism Guarantees

- `should_ratchet` uses a deterministic timestamp helper when no explicit `now` is provided, ensuring audit logs are reproducible across runs.
- `build_first_organism_metrics()` accepts explicit parameter overrides so integration tests can inject seeded values and assert exact gate outcomes.

---

## Summary Table

| Gate        | Threshold (First Organism) | Expected Outcome |
|-------------|----------------------------|------------------|
| Coverage    | CI ≥ 0.915, sample ≥ 16    | **Fail**         |
| Abstention  | rate ≤ 18 %, mass ≤ 640    | Pass             |
| Velocity    | ≥ 160 pph, CV ≤ 0.10       | Pass             |
| Caps        | mass ≥ 2400, runtime ≥ 20m, backlog ≤ 0.36 | Pass |

This configuration ensures the First Organism test exercises all four gate evaluations while preventing accidental ratcheting until statistical coverage is proven.

