# CAL-EXP-2 Index

**Status**: CLOSED
**Closure Date**: 2025-12-13
**Directive**: STRATCOM Phase Closure

---

## Experiment Summary

| Field | Value |
|-------|-------|
| Experiment ID | CAL-EXP-2 |
| Horizon | 1000 cycles |
| Verdict | PLATEAUING |
| Convergence Floor | dp ~ 0.025 |
| Next Action | UPGRADE-2 (structural) |

---

## Archived Attestations

| Document | Purpose | Status |
|----------|---------|--------|
| [CAL_EXP_2_Canonical_Record.md](CAL_EXP_2_Canonical_Record.md) | Primary experimental record | CANONICAL |
| [CAL_EXP_2_EXPERIMENT_DESIGN.md](CAL_EXP_2_EXPERIMENT_DESIGN.md) | Experimental design specification | FROZEN |
| [CAL_EXP_2_DEFINITIONS_BINDING.md](CAL_EXP_2_DEFINITIONS_BINDING.md) | Metric definitions binding | FROZEN |
| [CAL_EXP_2_LANGUAGE_CONSTRAINTS.md](CAL_EXP_2_LANGUAGE_CONSTRAINTS.md) | Language/terminology constraints | FROZEN |
| [CAL_EXP_2_GO_NO_GO.md](CAL_EXP_2_GO_NO_GO.md) | Go/No-Go decision record | FROZEN |
| [CAL_EXP_2_VALIDITY_ATTESTATION.md](CAL_EXP_2_VALIDITY_ATTESTATION.md) | Validity attestation | FROZEN |
| [CAL_EXP_2_FREEZE_ATTESTATION.md](CAL_EXP_2_FREEZE_ATTESTATION.md) | Freeze attestation | FROZEN |
| [CAL_EXP_2_POST_RUN_HYGIENE.md](CAL_EXP_2_POST_RUN_HYGIENE.md) | Post-run hygiene record | FROZEN |
| [CAL_EXP_2_EXIT_DECISION.md](CAL_EXP_2_EXIT_DECISION.md) | Exit decision record | FROZEN |

---

## Agent Attestations

All agents (A-O) have issued final attestations and are STANDING DOWN.

| Agent | Role | Final Status |
|-------|------|--------------|
| CLAUDE A | P5 Replay Signal (v1.3.0 freeze) | STANDING DOWN |
| CLAUDE C | P5 Diagnostic Harness | STANDING DOWN |
| CLAUDE D | Metrics Dual Eval Annex v1.2.0 (78 tests) | STANDING DOWN |
| CLAUDE N | NVR Run-Dir Shape + Non-Interference | STANDING DOWN |
| ALL (A-O) | CAL-EXP-2 Phase Closure | STANDING DOWN |
| CLAUDE L | Metric Integrity (no_metric_laundering.md) | STANDING DOWN |
| CLAUDE G | Warning Neutrality Helper (banned words FINAL) | STANDING DOWN |
| CLAUDE H | RTTS Status Adapter (v1.2.0) | STANDING DOWN |
| CLAUDE H | Manifest Hook + Integrity Check | STANDING DOWN |
| CLAUDE H | Warning Normal Form + Frozen Enums | STANDING DOWN |

---

## Code Paths (FROZEN by Convention)

The following code paths are associated with CAL-EXP-2 and are FROZEN:

- backend/health/rtts_status_adapter.py - RTTS status extraction (v1.2.0)
- scripts/build_first_light_evidence_pack.py - Manifest hook for rtts_validation_reference
- tests/scripts/test_generate_first_light_status_rtts_signal.py - RTTS tests (30 passing)
- scripts/usla_first_light_p4_harness.py - P5 diagnostic emission
- backend/health/p5_divergence_interpreter.py - P5 interpreter
- scripts/generate_p5_divergence_real_report.py - P5 report generator (metric_versioning, true_divergence_vector_v1)
- tests/health/test_p5_diagnostic_harness_integration.py - P5 tests (67 passing)
- tests/health/test_p5_divergence_interpreter.py - Interpreter tests (21 passing)
- tests/topology/first_light/test_p5_divergence_pipeline_integration.py - Metric versioning tests (46 passing)
- backend/topology/first_light/noise_vs_reality_integration.py - NVR core module
- tests/integration/test_noise_vs_reality_non_interference.py - NVR non-interference (22 tests)
- tests/helpers/warning_neutrality.py - Warning neutrality helper (banned words FINAL)
- tests/helpers/test_warning_neutrality.py - Tripwire tests (11 passing)
- scripts/generate_first_light_status.py:extract_p5_replay_signal() - P5 replay signal (v1.3.0)
- scripts/generate_first_light_status.py:generate_warnings() - Warning cap precedence (v1.3.0)
- backend/health/replay_governance_adapter.py:P5_DETERMINISM_* - Threshold constants (v1.3.0)
- backend/health/replay_governance_adapter.py:compute_driver_codes() - Driver codes (v1.3.0)
- tests/first_light/test_p5_replay_wiring_integration.py - P5 replay wiring tests (40 passing)
- backend/health/metrics_dual_eval_annex.py - Metrics dual eval annex (v1.2.0)
- tests/ci/test_metrics_dual_eval_annex.py - Metrics dual eval tests (78 passing)
- docs/system_law/Metrics_PhaseX_Spec.md - Appendix D golden bundle + section 5.5 advisory axis
- docs/system_law/How_To_Read_Divergence_Metrics.md - Advisory interpretation note

---

## Scope Fence (Binding)

P5 Diagnostic Panel constraints (from P5_Divergence_Diagnostic_Panel_Spec.md lines 60-72):

- **Cannot gate** - MUST NEVER be used for gating decisions
- **Cannot influence CAL-EXP-3 acceptance** - not part of divergence minimization acceptance criteria
- **Cannot mutate divergence semantics** - RECONCILIATION VIEW; NOT a metric authority

---

## Transition to CAL-EXP-3

**Prerequisites satisfied:**
- CAL-EXP-2 canonical record complete
- All attestations archived
- Scope fences locked
- All agents standing down

**Ready for:** CAL-EXP-3 (uplift / external signal ingestion)

---

*This index is FROZEN. No modifications permitted without a new experiment ID (CAL-EXP-3+) and explicit STRATCOM directive.*
| CLAUDE E | Identity Preflight Non-Interference (14 isolation tests) | STANDING DOWN |
