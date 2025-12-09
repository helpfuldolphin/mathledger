# Phase II Terminology Mapping Table

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.

This document provides the canonical mapping between documentation terms,
code terms, and governance terms as defined in VSD_PHASE_2.md and
PREREG_UPLIFT_U2.yaml.

## Governance Safeguards

- All terminology must match this canonical mapping
- Violations are flagged by the documentation consistency scanner
- CI gates enforce terminology consistency

## Slice Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `atoms4-depth4` | 4-atom 4-depth slice, atoms4-depth4 | atoms4-depth4, atoms4_depth4 | config/curriculum.yaml |
| `atoms4-depth5` | 4-atom 5-depth slice, atoms4-depth5 | atoms4-depth5, atoms4_depth5 | config/curriculum.yaml |
| `atoms5-depth6` | 5-atom 6-depth slice, atoms5-depth6 | atoms5-depth6, atoms5_depth6 | config/curriculum.yaml |
| `first_organism_pl2_hard` | FO hard slice, PL2 hard, first_organism_pl2_hard | first_organism_pl2_hard | config/curriculum.yaml |
| `slice_debug_uplift` | debug slice, debug uplift slice, slice_debug_uplift | SLICE_DEBUG_UPLIFT, slice_debug_uplift | config/curriculum.yaml |
| `slice_easy_fo` | Easy, FO easy slice, easy slice, ... | SLICE_EASY_FO, slice_easy_fo | config/curriculum.yaml |
| `slice_hard` | Hard, hard slice, slice_hard | SLICE_HARD, slice_hard | config/curriculum.yaml |
| `slice_medium` | Medium, Wide Slice, medium slice, ... | SLICE_MEDIUM, slice_medium | config/curriculum.yaml |
| `slice_uplift_proto` | proto slice, slice_uplift_proto, uplift proto slice | SLICE_UPLIFT_PROTO, slice_uplift_proto | config/curriculum.yaml |

## Metric Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `abstention_mass` | abstention mass, abstention_mass, alpha_mass, ... | abstention_count, abstention_mass, alpha_mass | docs/RFL_LAW.md |
| `abstention_rate` | abstention rate, abstention_rate, alpha_rate, ... | abstention_fraction, abstention_rate, alpha_rate | docs/RFL_LAW.md |
| `chain_success` | chain success, chain-success, chain_success | CHAIN_SUCCESS, chainSuccess, chain_success | VSD_PHASE_2.md |
| `coverage_rate` | coverage, coverage rate, coverage_rate | ci_lower_min, coverage, coverage_rate | config/curriculum.yaml |
| `goal_hit` | goal hit, goal hit rate, goal-hit, ... | GOAL_HIT, goalHit, goal_hit | VSD_PHASE_2.md |
| `joint_goal` | joint goal, joint-goal, joint_goal | JOINT_GOAL, jointGoal, joint_goal | VSD_PHASE_2.md |
| `max_depth` | depth_max, max_depth, maximum depth | MAX_DEPTH, depth_max, max_depth | experiments/METRICS_DEFINITION.md |
| `sparse_density` | sparse density, sparse_density, sparsity | SPARSE_DENSITY, sparseDensity, sparse_density | VSD_PHASE_2.md |
| `throughput` | pph, proofs per hour, throughput, ... | min_pph, proofs_per_hour, throughput | experiments/METRICS_DEFINITION.md |

## Mode Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `baseline` | BFS baseline, Baseline, baseline, ... | BASELINE, baseline, mode_baseline | experiments/METRICS_DEFINITION.md |
| `rfl` | RFL, Reflexive Formal Learning, reflexive mode, ... | RFL, mode_rfl, rfl | docs/RFL_LAW.md |

## Phase Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `PHASE_I` | Phase 1, Phase I, PHASE I, ... | PHASE1, PHASE_I, phase_1 | VSD_PHASE_2.md |
| `PHASE_II` | Phase 2, Phase II, PHASE II, ... | PHASE2, PHASE_II, phase_2 | VSD_PHASE_2.md |
| `PHASE_III` | Phase 3, Phase III, PHASE III, ... | PHASE3, PHASE_III, phase_3 | VSD_PHASE_2.md |

## Symbol Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `H_t` | H(t), H_t, composite attestation root, ... | H_t, attestation_root, composite_root, ... | docs/RFL_LAW.md |
| `R_t` | R(t), R_t, Reasoning Merkle root, ... | R_t, reasoning_merkle_root, reasoning_root | docs/RFL_LAW.md |
| `U_t` | U(t), U_t, UI Merkle root, ... | U_t, ui_merkle_root, ui_root | docs/RFL_LAW.md |
| `abstention_tolerance` | abstention tolerance, tau, tolerance threshold, ... | abstention_tolerance, tau, tolerance | docs/RFL_LAW.md |
| `step_id` | deterministic step identifier, step ID, step_id | STEP_ID, stepId, step_id | docs/RFL_LAW.md |
| `symbolic_descent` | descent gradient, nabla_sym, symbolic descent, ... | descent_gradient, nabla_sym, symbolic_descent | docs/RFL_LAW.md |

## Concept Terms

| Canonical Name | Doc Variants | Code Variants | Source |
|----------------|--------------|---------------|--------|
| `First_Organism` | FO, First Organism, first organism, ... | FO, FirstOrganism, first_organism | docs/FIRST_ORGANISM.md |
| `attestation` | Attestation, attestation, attested run | AttestedRunContext, attestation, attested | docs/ATTESTATION_SPEC.md |
| `curriculum_slice` | Curriculum Slice, curriculum slice, slice | CurriculumSlice, curriculum_slice, slice_cfg | config/curriculum.yaml |
| `dual_attestation` | dual attestation, dual root attestation, dual-attestation | DUAL_ATTESTATION, dual_attestation, dual_root | VSD_PHASE_2.md |
| `ledger_entry` | RunLedgerEntry, ledger entry, run entry | RunLedgerEntry, ledger_entry, policy_ledger | docs/RFL_LAW.md |

## Term Descriptions

### Slice Descriptions

- **atoms4-depth4**: Intermediate slice (atoms=4, depth=4)
- **atoms4-depth5**: Intermediate slice (atoms=4, depth=5)
- **atoms5-depth6**: Intermediate slice (atoms=5, depth=6)
- **first_organism_pl2_hard**: First Organism PL2 hard slice (atoms=6, depth=8)
- **slice_debug_uplift**: Debug slice for microscopic uplift experiments (atoms=2, depth=2)
- **slice_easy_fo**: Easy slice for First Organism testing (atoms=3, depth=3)
- **slice_hard**: Hard slice for stress testing (atoms=7, depth=12)
- **slice_medium**: Wide Slice for RFL uplift experiments (atoms=5, depth=7)
- **slice_uplift_proto**: Medium-hard slice for uplift experiments (atoms=3, depth=4)

### Metric Descriptions

- **abstention_mass**: Abstention mass = raw abstention count
- **abstention_rate**: Abstention rate = abstentions / total_attempts
- **chain_success**: Success rate across a chain of dependent tasks
- **coverage_rate**: Coverage metric for curriculum gate
- **goal_hit**: Rate of achieving the primary objective
- **joint_goal**: Rate of achieving a composite goal of multiple objectives
- **max_depth**: Maximum derivation depth of verified statements
- **sparse_density**: Measure of the efficiency of the solution path
- **throughput**: Proofs per hour throughput metric

### Mode Descriptions

- **baseline**: Baseline mode using random/BFS derivation
- **rfl**: Reflexive Formal Learning mode with learned policy

### Phase Descriptions

- **PHASE_I**: Phase I - foundational experiments (no modification allowed)
- **PHASE_II**: Phase II - Operation Asymmetry governance
- **PHASE_III**: Phase III - post-audit escalation phase

### Symbol Descriptions

- **H_t**: Composite attestation root = SHA256(R_t || U_t)
- **R_t**: Reasoning Merkle root over proof artifacts
- **U_t**: UI Merkle root over human interaction artifacts
- **abstention_tolerance**: Abstention tolerance threshold (default 0.10)
- **step_id**: Deterministic step identifier (64-char hex)
- **symbolic_descent**: Symbolic descent = -(α_rate - τ)

### Concept Descriptions

- **First_Organism**: The first production metabolic cycle harness
- **attestation**: Cryptographic verification of derivation results
- **curriculum_slice**: A contiguous run interval with fixed derivation policy
- **dual_attestation**: Require two independent statistical checks to agree
- **ledger_entry**: Structured curriculum ledger entry for a single RFL run

---

Generated by `scripts/doc_sync_scanner.py`

Last updated: See git commit history

