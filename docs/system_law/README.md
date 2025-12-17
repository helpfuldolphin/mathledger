# System Law Documentation Index

This directory contains the formal specifications and doctrine documents for MathLedger's USLA (Unified System Law Abstraction) and substrate-level cognitive governance infrastructure.

## Phase X Series: Shadow Experiment Architecture

The Phase X series establishes the shadow observation infrastructure for USLA validation.

| Document | Status | Description |
|----------|--------|-------------|
| [Phase_X_Integration_Spec_v1.0.md](Phase_X_Integration_Spec_v1.0.md) | Implemented | USLA integration architecture (P0-P1) |
| [Phase_X_P2_Spec.md](Phase_X_P2_Spec.md) | Implemented | Digital Twin and divergence monitoring |
| [Phase_X_P3_Spec.md](Phase_X_P3_Spec.md) | Implemented | First-Light shadow experiment with synthetic data |
| [Phase_X_P4_Spec.md](Phase_X_P4_Spec.md) | **Design Freeze** | Real runner shadow coupling (stubs only) |
| [Phase_X_Prelaunch_Review.md](Phase_X_Prelaunch_Review.md) | **Design Freeze** | P3/P4 Go/No-Go criteria + Evidence Package definition |
| [Phase_X_Divergence_Metric.md](Phase_X_Divergence_Metric.md) | **Specification** | Formal divergence metric and severity bands |
| [Phase_X_P3P4_TODO.md](Phase_X_P3P4_TODO.md) | **Active** | Execution readiness checklist with file bindings |
| [Real_Telemetry_Topology_Spec.md](Real_Telemetry_Topology_Spec.md) | **Specification** | P5 Real telemetry topology, validation, and acceptance envelope |
| [Phase_X_P5_Implementation_Blueprint.md](Phase_X_P5_Implementation_Blueprint.md) | **Blueprint** | P5 implementation plan for Codex/Manus execution fleets |
| [Evidence_Pack_Spec_PhaseX.md](Evidence_Pack_Spec_PhaseX.md) | **Specification** | Whitepaper evidence pack bundle format and schemas |

## TDA Mind Scanner Integration

| Document | Status | Description |
|----------|--------|-------------|
| [TDA_PhaseX_Binding.md](TDA_PhaseX_Binding.md) | **Specification** | TDA metrics (SNS, PCS, DRS, HSS) Phase X binding |

## USLA Core Documents

| Document | Status | Description |
|----------|--------|-------------|
| [USLA_v0.1.md](USLA_v0.1.md) | Reference | USLA specification v0.1 |
| [canonical_update_operator.md](canonical_update_operator.md) | Reference | USLA update operator formal definition |
| [intervention_surface_map.md](intervention_surface_map.md) | Reference | Control surface mapping |

## Governance Fusion Layer

| Document | Status | Description |
|----------|--------|-------------|
| [Global_Governance_Fusion_PhaseX.md](Global_Governance_Fusion_PhaseX.md) | **Specification** | Global governance signal fusion layer for Phase X |
| [First_Light_Auditor_Guide.md](First_Light_Auditor_Guide.md) | **Reference** | Auditor guide for interpreting status + alignment files |
| [GGFL_P5_Pattern_Test_Plan.md](GGFL_P5_Pattern_Test_Plan.md) | **Specification** | P5 divergence pattern test cases and validation criteria |

## Governance Signal Documents

| Document | Status | Description |
|----------|--------|-------------|
| [Replay_Safety_Governance_Law.md](Replay_Safety_Governance_Law.md) | Reference | Replay Safety governance signal semantics and mapping rules |
| [NCI_PhaseX_Spec.md](NCI_PhaseX_Spec.md) | **Specification** | NCI consistency laws and governance signal integration |

## Calibration Campaign

Experimental records from the Twin calibration campaign.

| Document | Status | Description |
|----------|--------|-------------|
| [calibration/METRIC_DEFINITIONS.md](calibration/METRIC_DEFINITIONS.md) | **Frozen** | Canonical metric semantics (v1.1.0) |
| [calibration/CAL_EXP_2_Canonical_Record.md](calibration/CAL_EXP_2_Canonical_Record.md) | **Canonical** | CAL-EXP-2 long-window convergence (1000 cycles) |
| [calibration/UPGRADE_2_DRAFT.md](calibration/UPGRADE_2_DRAFT.md) | **Provisional** | UPGRADE-2 Predictive Twin design (pending CAL-EXP-3) |
| [calibration/AUDITOR_CHECKLIST.md](calibration/AUDITOR_CHECKLIST.md) | **Reference** | Auditor smoke-test checklist (CHK-001 to CHK-008) |

See also: `docs/calibration/CAL_EXP_1_UPGRADE_1_VERDICT.md` for UPGRADE-1 validation.

### Scope Lock Assertion

**Canonical vs Provisional**: Documents marked **Canonical** or **Frozen** represent validated experimental records or ratified definitions. They may receive additive updates but their core findings are immutable. Documents marked **Provisional** are design drafts pending experimental validation.

**Authorization Requirements**: Implementation of any design in a **Provisional** document requires explicit STRATCOM authorization. Provisional documents define hypothesis targets, not acceptance thresholds. All metric semantics must reference `METRIC_DEFINITIONS.md` as the canonical authority. No acceptance gates may be introduced outside of ratified specification documents.

## Document Status Legend

- **Reference**: Stable specification document
- **Implemented**: Design complete, implementation active
- **Design Freeze**: Design complete, implementation stubs only (awaiting authorization)
- **Canonical**: Validated experimental record (immutable)
- **Provisional**: Design draft pending experimental validation
- **Frozen**: Ratified definition document (additive changes only)
- **Draft**: Under active development

## Phase X Implementation Status

### P3: First-Light Shadow Experiment (IMPLEMENTED)

- Synthetic data generation via `SyntheticStateGenerator`
- `FirstLightShadowRunner` with 50-1000 cycle shadow execution
- Red-flag observation layer (logging only, no enforcement)
- Delta-p computation for learning curve analysis
- JSONL log schemas for cycles, red-flags, metrics, summary
- 82 tests passing

### P4: Real Runner Shadow Coupling (DESIGN FREEZE)

- Design specification complete
- Skeleton modules created with `NotImplementedError` stubs
- Read-only adapter interface defined
- Divergence analysis architecture specified
- **Implementation NOT authorized** - awaiting explicit activation

### P5: Real Telemetry Topology (SPECIFICATION)

- Formal specification of real vs mock telemetry discrimination
- Manifold constraints: boundedness, continuity, correlation structure
- Noise envelope specification with detection heuristics
- Divergence pattern taxonomy with semantic mapping
- Twin warm-start calibration blueprint
- P5 Acceptance Envelope with Go/No-Go criteria
- **Implementation NOT authorized** - specification only

## SHADOW MODE Contract

All Phase X code operates under the SHADOW MODE contract:

| Invariant | Description |
|-----------|-------------|
| No governance modification | USLA outputs never influence real decisions |
| No abort enforcement | Red-flags are logged, never acted upon |
| Observational only | All outputs are for analysis/logging |
| Read-only coupling | P4 adapter never writes to USLAIntegration |

## Directory Structure

```
docs/system_law/
├── README.md                           # This index
├── USLA_v0.1.md                        # Core USLA specification
├── canonical_update_operator.md        # Update operator formal model
├── intervention_surface_map.md         # Control surface mapping
├── Phase_X_Integration_Spec_v1.0.md    # P0-P1 integration
├── Phase_X_P2_Spec.md                  # P2 digital twin
├── Phase_X_P3_Spec.md                  # P3 synthetic shadow
├── Phase_X_P4_Spec.md                  # P4 real coupling (design)
├── Phase_X_Prelaunch_Review.md         # P3/P4 Go/No-Go + Evidence Package
├── Phase_X_Divergence_Metric.md        # Divergence metric formal definition
├── Phase_X_P3P4_TODO.md                # Execution readiness checklist
├── Real_Telemetry_Topology_Spec.md     # P5 RTTS: Real telemetry validation
├── Phase_X_P5_Implementation_Blueprint.md  # P5 Implementation Blueprint
├── calibration/                        # Twin calibration campaign
│   ├── METRIC_DEFINITIONS.md           # Canonical metric semantics (FROZEN v1.1.0)
│   ├── CAL_EXP_2_Canonical_Record.md   # CAL-EXP-2 canonical record (CANONICAL)
│   ├── UPGRADE_2_DRAFT.md              # UPGRADE-2 design draft (PROVISIONAL)
│   └── AUDITOR_CHECKLIST.md            # Auditor smoke-test checklist (REFERENCE)
└── schemas/
    ├── evidence_pack/                  # Evidence Pack bundle schemas (NEW)
    │   ├── evidence_bundle.schema.json
    │   ├── tda_metrics.schema.json
    │   ├── compliance_narrative.schema.json
    │   └── audit_attestation.schema.json
    ├── first_light/                    # P3 output schemas
    │   ├── first_light_synthetic_raw.schema.json
    │   ├── first_light_red_flag_matrix.schema.json
    │   ├── first_light_stability_report.schema.json
    │   └── first_light_metrics_windows.schema.json
    ├── nci/                            # NCI governance schemas
    │   ├── nci_director_panel.schema.json
    │   └── nci_governance_signal.schema.json
    ├── phase_x/                        # Phase X governance fusion schemas
    │   └── governance_signal_unified.schema.json
    ├── phase_x_p4/                     # P4 output schemas
    │   ├── p4_divergence_log.schema.json
    │   ├── p4_twin_trajectory.schema.json
    │   ├── p4_calibration_report.schema.json
    │   └── p4_divergence_distribution.schema.json  # (NEW)
    ├── replay_safety/                  # Replay Safety governance schemas
    │   └── replay_safety_governance_signal.schema.json
    └── tda/                            # TDA Mind Scanner schemas
        ├── tda_metrics_p3.schema.json  # P3 stability envelope TDA metrics
        └── tda_metrics_p4.schema.json  # P4 divergence context TDA metrics
```
