# Topology/Bundle Phase X Requirements

---

> **STATUS: DESIGN SPECIFICATION — DOCTRINE BINDING DOCUMENT**
>
> This document defines topology and bundle governance requirements for Phase X integration.
> Binds structural TDA invariants to USLA governance layer through shadow observation.
>
> **SHADOW MODE ONLY. NO GOVERNANCE MODIFICATION. NO ABORT ENFORCEMENT.**

---

**Version**: 1.2.0
**Date**: 2025-12-11
**Status**: Design Specification (P5 Reality Adapter Implemented)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Topology Invariants for P3 Synthetic Runs](#2-topology-invariants-for-p3-synthetic-runs)
3. [Topology + Bundle Requirements for P4 Structural Shadow-Coupling](#3-topology--bundle-requirements-for-p4-structural-shadow-coupling)
4. [GovernanceSignal Codes for Topology/Bundle States](#4-governancesignal-codes-for-topologybundle-states)
5. [TODO Anchors for Integration](#5-todo-anchors-for-integration)
6. [Topology Modes → Phase X Structural Implications](#6-topology-modes--phase-x-structural-implications)
7. [Schema References](#7-schema-references)
8. [Authorization Gates](#8-authorization-gates)
9. [P5 Expectations and Interpretation Guidance](#9-p5-expectations-and-interpretation-guidance)
10. [P5 Reality Adapter Wiring Plan](#10-p5-reality-adapter-wiring-plan)

---

## 1. Overview

### 1.1 Purpose

This document establishes the formal requirements for integrating topology-aware governance and bundle provenance into the Phase X shadow experiment framework. The integration follows three principles:

1. **Structural Invariants**: TDA-derived invariants constrain the topology manifold
2. **Bundle Governance**: Provenance bundles provide cryptographic evidence chains
3. **Shadow Observation**: All topology signals are observational (no enforcement in Phase X)

### 1.2 Relationship to Phase X Phases

| Phase | Topology Role | Bundle Role |
|-------|--------------|-------------|
| **P3** (Synthetic) | Synthetic topology metrics from `SyntheticStateGenerator` | Bundle provenance for synthetic traces |
| **P4** (Real Coupling) | Real TDA metrics via `TDAAdapter` observation | Real runner bundle attestation |
| **P5+** (Future) | Active topology governance | Bundle-gated policy updates |

### 1.3 SHADOW MODE Contract

All topology and bundle signals in Phase X operate under SHADOW MODE:

| Invariant | Description |
|-----------|-------------|
| **TOPO-INV-01** | Topology signals NEVER modify governance decisions |
| **TOPO-INV-02** | Bundle attestation NEVER gates real execution |
| **TOPO-INV-03** | TDA metrics logged only, never enforced |
| **TOPO-INV-04** | All outputs include `"mode": "SHADOW"` |

---

## 2. Topology Invariants for P3 Synthetic Runs

### 2.1 Structural TDA Invariants

During P3 synthetic validation runs, the following topology invariants MUST be observable:

#### 2.1.1 Betti Number Bounds

```
Invariant: β₀(state_space) ∈ [β₀_min, β₀_max]
           β₁(state_space) ∈ [β₁_min, β₁_max]

Where:
  β₀ = Connected components (should be 1 for healthy operation)
  β₁ = 1-dimensional holes (topological cycles)

P3 Observation Thresholds (LOGGING only):
  β₀_min = 1, β₀_max = 1   (single connected component expected)
  β₁_min = 0, β₁_max = 3   (limited topological complexity)
```

#### 2.1.2 Persistence Diagram Stability

```
Invariant: Bottleneck(PD_t, PD_t-1) < ε_persistence

Where:
  PD_t = Persistence diagram at cycle t
  Bottleneck = Bottleneck distance metric
  ε_persistence = 0.15 (P3 observation threshold)

Violation (LOGGED only):
  TOPO-PERSIST-DRIFT: Persistence diagram instability detected
```

#### 2.1.3 Safe Region Topology

```
Invariant: Ω_topology = convex_hull(safe_states)

Requirements:
  - Ω must remain simply connected (β₁(Ω) = 0)
  - State trajectory should have bounded curvature
  - Exit duration bounded: consecutive_outside_Ω < max_omega_exit
```

### 2.2 Synthetic State Topology Metrics

The `SyntheticStateGenerator` in P3 MUST produce states that allow topology observation:

```python
@dataclass
class SyntheticTopologyMetrics:
    """Topology metrics derived from synthetic state generation."""

    # Betti numbers (0-dimensional and 1-dimensional)
    betti_0: int                    # Connected components
    betti_1: int                    # Holes/cycles

    # Persistence
    persistence_entropy: float      # Entropy of persistence diagram
    max_persistence: float          # Maximum persistence interval
    persistence_stability: float    # Bottleneck distance from previous

    # Safe region metrics
    omega_convexity: float          # Convexity measure of Ω occupancy
    omega_boundary_distance: float  # Distance to Ω boundary

    # Curvature
    trajectory_curvature: float     # Local curvature of state path
    curvature_variance: float       # Variance of curvature over window

    # Observation status
    observation_mode: str = "SHADOW"  # Always "SHADOW" in P3
```

### 2.3 P3 Topology Log Schema Reference

Synthetic topology observations are logged to:
- `results/first_light/{run_id}/topology_metrics.jsonl`

Schema: See Section 7.1 (`topology_drift_compass.schema.json`)

---

## 3. Topology + Bundle Requirements for P4 Structural Shadow-Coupling

### 3.1 Real TDA Metrics Integration

P4 extends P3 by observing real topology metrics from actual runner execution:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    P4 Topology Shadow Integration                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                      REAL EXECUTION PATH                              │   │
│   │   U2Runner / RFLRunner → USLAIntegration → TDAAdapter (existing)     │   │
│   └─────────────────────────────────┬────────────────────────────────────┘   │
│                                     │                                        │
│                          ╔══════════╧══════════╗                             │
│                          ║   READ-ONLY FENCE   ║                             │
│                          ╚══════════╤══════════╝                             │
│                                     │                                        │
│   ┌─────────────────────────────────┼────────────────────────────────────┐   │
│   │                    SHADOW OBSERVATION PATH                            │   │
│   │                                                                       │   │
│   │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │   │
│   │   │ TopologyAdapter │    │ BundleAdapter   │    │ DivergenceCorr  │ │   │
│   │   │   (READ-ONLY)   │    │   (READ-ONLY)   │    │   elator        │ │   │
│   │   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │   │
│   │            │                      │                      │          │   │
│   │            └──────────────────────┼──────────────────────┘          │   │
│   │                                   │                                  │   │
│   │                          ┌────────▼────────┐                        │   │
│   │                          │ TopologyBundle  │                        │   │
│   │                          │  JointView      │                        │   │
│   │                          │  (LOGGED)       │                        │   │
│   │                          └─────────────────┘                        │   │
│   │                                                                       │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Bundle Provenance Requirements

#### 3.2.1 Bundle Structure for P4

Each P4 cycle observation MUST be associable with a provenance bundle:

```python
@dataclass
class TopologyBundleReference:
    """Reference to provenance bundle for P4 observation."""

    # Bundle identification
    bundle_id: str                          # SHA-256 of bundle manifest
    bundle_timestamp: str                   # ISO8601 creation time

    # Source files
    source_manifest_hash: str               # Hash of source file manifest
    trace_files: List[str]                  # List of trace file paths

    # Topology snapshot
    topology_snapshot_hash: str             # Hash of TDA metrics at bundle time
    betti_numbers: Tuple[int, int]          # (β₀, β₁) at bundle creation

    # Governance correlation
    governance_cycle: int                   # Correlated P4 cycle
    governance_aligned: bool                # Alignment status at bundle time

    # Observation mode
    observation_mode: str = "SHADOW"        # Always "SHADOW" in P4
```

#### 3.2.2 Bundle-Topology Correlation

P4 observations correlate topology metrics with bundle provenance:

| Correlation | Description | Log Field |
|-------------|-------------|-----------|
| **Cycle-Bundle** | Which bundle covers this cycle | `bundle_id` |
| **Topology-Hash** | TDA metrics hash at observation | `topology_snapshot_hash` |
| **Divergence-Bundle** | Bundle associated with divergence | `divergence_bundle_ref` |

### 3.3 Structural Shadow-Coupling Contract

```
STRUCTURAL COUPLING INVARIANTS:

1. Topology Observation Independence
   - Real TDA adapter provides metrics independently of shadow
   - Shadow reads TDA state but NEVER modifies
   - Topology drift detected but not corrected in P4

2. Bundle Attestation Independence
   - Bundles created by real execution path
   - Shadow reads bundle hashes but NEVER creates
   - Bundle validation logged but not enforced

3. Cross-Correlation Read-Only
   - Topology ↔ Bundle correlation is observational
   - Divergence ↔ Topology cross-analysis is logged
   - No feedback from correlation to real execution
```

### 3.4 P4 Topology-Divergence Cross-Correlation

When P4 divergence occurs, correlate with topology state:

```json
{
  "schema": "p4-topology-divergence-correlation/1.0.0",
  "cycle": 142,
  "divergence_severity": "WARN",

  "topology_at_divergence": {
    "betti_0": 1,
    "betti_1": 2,
    "persistence_stability": 0.12,
    "omega_boundary_distance": 0.05,
    "trajectory_curvature": 0.23
  },

  "bundle_at_divergence": {
    "bundle_id": "sha256:abc123...",
    "governance_aligned": false
  },

  "correlation_hypothesis": "TOPOLOGY_DRIFT_PRECEDES_DIVERGENCE",
  "action": "LOGGED_ONLY"
}
```

---

## 4. GovernanceSignal Codes for Topology/Bundle States

### 4.1 Topology Governance Signal Codes

The following signal codes extend the existing governance signal vocabulary:

#### 4.1.1 Topology Health Signals

| Code | Severity | Condition | Action |
|------|----------|-----------|--------|
| `TOPO-OK-001` | OK | Betti numbers within bounds | Continue |
| `TOPO-OK-002` | OK | Persistence diagram stable | Continue |
| `TOPO-OK-003` | OK | Safe region simply connected | Continue |
| `TOPO-WARN-001` | WARN | β₁ approaching upper bound | **LOG ONLY** |
| `TOPO-WARN-002` | WARN | Persistence drift > 0.10 | **LOG ONLY** |
| `TOPO-WARN-003` | WARN | Trajectory curvature spike | **LOG ONLY** |
| `TOPO-CRIT-001` | CRITICAL | β₀ > 1 (disconnected components) | **LOG ONLY** |
| `TOPO-CRIT-002` | CRITICAL | Persistence collapse | **LOG ONLY** |
| `TOPO-CRIT-003` | CRITICAL | Ω topology violation | **LOG ONLY** |

#### 4.1.2 Bundle Governance Signal Codes

| Code | Severity | Condition | Action |
|------|----------|-----------|--------|
| `BNDL-OK-001` | OK | Bundle hash valid | Continue |
| `BNDL-OK-002` | OK | Manifest complete | Continue |
| `BNDL-WARN-001` | WARN | Bundle age > threshold | **LOG ONLY** |
| `BNDL-WARN-002` | WARN | Trace coverage incomplete | **LOG ONLY** |
| `BNDL-CRIT-001` | CRITICAL | Bundle hash mismatch | **LOG ONLY** |
| `BNDL-CRIT-002` | CRITICAL | Manifest missing files | **LOG ONLY** |

#### 4.1.3 Cross-Correlation Signal Codes

| Code | Severity | Condition | Action |
|------|----------|-----------|--------|
| `XCOR-OK-001` | OK | Topology-bundle aligned | Continue |
| `XCOR-WARN-001` | WARN | Topology drift predicts divergence | **LOG ONLY** |
| `XCOR-WARN-002` | WARN | Bundle-topology temporal mismatch | **LOG ONLY** |
| `XCOR-CRIT-001` | CRITICAL | Topology-bundle-divergence triple fault | **LOG ONLY** |

### 4.2 Governance Signal Structure

```python
@dataclass
class TopologyGovernanceSignal:
    """Governance signal for topology/bundle states."""

    # Signal identification
    schema_version: str = "1.0.0"
    signal_type: str = "topology_bundle"

    # Status
    status: str                     # OK, WARN, BLOCK
    governance_status: str          # Mirrors status for harmonization
    governance_alignment: str       # ALIGNED, TENSION, DIVERGENT

    # Topology status
    topology_status: str            # OK, WARN, CRIT
    topology_codes: List[str]       # List of TOPO-* codes active

    # Bundle status
    bundle_status: str              # OK, WARN, CRIT
    bundle_codes: List[str]         # List of BNDL-* codes active

    # Cross-correlation
    correlation_status: str         # OK, WARN, CRIT
    correlation_codes: List[str]    # List of XCOR-* codes active

    # Governance flags (P4 SHADOW: always permissive)
    conflict: bool = False
    safe_for_policy_update: bool = True   # SHADOW: always True
    safe_for_promotion: bool = True       # SHADOW: always True

    # Diagnostic
    reasons: List[str]              # Prefixed reason strings
    timestamp: str                  # ISO8601

    # Observation mode
    observation_mode: str = "SHADOW"
```

### 4.3 Signal Fusion Logic

Topology governance signals fuse with existing signals:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Governance Signal Fusion                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Safety    │     │   Radar     │     │  Topology   │                  │
│   │   Signal    │     │   Signal    │     │   Signal    │                  │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              │                                               │
│                     ┌────────▼────────┐                                     │
│                     │  Fusion Engine  │                                     │
│                     │                 │                                     │
│                     │  Priority:      │                                     │
│                     │  1. Safety      │                                     │
│                     │  2. Topology    │                                     │
│                     │  3. Radar       │                                     │
│                     │                 │                                     │
│                     │  SHADOW: Log    │                                     │
│                     │  all, enforce   │                                     │
│                     │  none           │                                     │
│                     └────────┬────────┘                                     │
│                              │                                               │
│                     ┌────────▼────────┐                                     │
│                     │  Unified Signal │                                     │
│                     │  (evidence pack)│                                     │
│                     └─────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. TODO Anchors for Integration

### 5.1 Global Alignment View Integration

```python
# TODO [TOPOLOGY-GLOBAL-001]: Wire topology signals into global alignment view
#
# Location: backend/health/global_alignment_view.py (to be created)
#
# Requirements:
#   1. TopologyGovernanceSignal must contribute to global alignment score
#   2. Betti number bounds visible in global health surface
#   3. Persistence stability metric exposed
#
# Integration Pattern:
#   - TopologyHealthAdapter.get_alignment_contribution() -> AlignmentContribution
#   - Register adapter in GlobalAlignmentViewRegistry
#
# Dependencies:
#   - topology_drift_compass.schema.json
#   - TopologyGovernanceSignal dataclass
#
# Phase X Status: SHADOW (observation only)
```

### 5.2 P4 Divergence Cross-Correlation

```python
# TODO [TOPOLOGY-P4-DIVERGENCE-001]: Implement topology-divergence cross-correlation
#
# Location: backend/topology/first_light/topology_divergence_correlator.py (to be created)
#
# Requirements:
#   1. When DivergenceAnalyzer detects divergence, capture topology state
#   2. Compute correlation hypothesis (does topology drift precede divergence?)
#   3. Log correlation analysis to divergence.jsonl extension
#
# Integration Pattern:
#   class TopologyDivergenceCorrelator:
#       def correlate(
#           self,
#           divergence: DivergenceSnapshot,
#           topology: TopologyMetrics
#       ) -> TopologyDivergenceCorrelation
#
# Dependencies:
#   - DivergenceAnalyzer (P4)
#   - TDAAdapter or TopologyAdapter
#   - p4_divergence_log.schema.json (extend with topology fields)
#
# Phase X Status: SHADOW (analysis only, no enforcement)
```

### 5.3 Topology Cross-Correlation Analysis

```python
# TODO [TOPOLOGY-XCOR-001]: Implement triple cross-correlation (topology-bundle-divergence)
#
# Location: backend/topology/cross_correlation_engine.py (to be created)
#
# Requirements:
#   1. Track topology metrics over time window
#   2. Track bundle provenance over same window
#   3. Track divergence events
#   4. Compute pairwise and triple correlations
#
# Analysis Outputs:
#   - Does topology instability predict divergence?
#   - Do bundle anomalies correlate with topology changes?
#   - Triple fault detection: topology + bundle + divergence simultaneous
#
# Integration Pattern:
#   class CrossCorrelationEngine:
#       def analyze_window(
#           self,
#           topology_history: List[TopologyMetrics],
#           bundle_history: List[BundleSnapshot],
#           divergence_history: List[DivergenceSnapshot]
#       ) -> CrossCorrelationReport
#
# Dependencies:
#   - topology_bundle_joint_view.schema.json
#   - XCOR-* signal codes
#
# Phase X Status: SHADOW (analysis only)
```

### 5.4 Director Panel Integration

```python
# TODO [TOPOLOGY-DIRECTOR-001]: Wire topology/bundle into Director Panel view
#
# Location: backend/topology/director_panel_adapter.py (to be created)
#
# Requirements:
#   1. Expose topology mode (see Section 6)
#   2. Expose bundle health status
#   3. Expose cross-correlation summary
#   4. Real-time streaming for dashboard integration
#
# Director Panel Tiles:
#   - Topology Mode indicator (STABLE, DRIFT, CRITICAL)
#   - Bundle Chain status (VALID, WARN, BROKEN)
#   - Cross-Correlation Health (ALIGNED, TENSION, FAULT)
#
# Integration Pattern:
#   class TopologyDirectorAdapter:
#       def get_panel_state(self) -> DirectorPanelState
#       def subscribe(self) -> AsyncIterator[DirectorPanelUpdate]
#
# Dependencies:
#   - topology_bundle_director_panel.schema.json
#   - WebSocket or SSE streaming infrastructure
#
# Phase X Status: SHADOW (read-only dashboard)
```

---

## 6. Topology Modes → Phase X Structural Implications

### 6.1 Topology Mode Definitions

| Mode | Definition | Betti Bounds | Persistence | Ω Status |
|------|------------|--------------|-------------|----------|
| **STABLE** | Nominal topology | β₀=1, β₁≤1 | Drift < 0.05 | Inside, convex |
| **DRIFT** | Minor topology deviation | β₀=1, β₁≤2 | 0.05 ≤ Drift < 0.15 | Inside, minor boundary approach |
| **TURBULENT** | Significant topology change | β₀=1, β₁≤3 | 0.15 ≤ Drift < 0.25 | Boundary or recent exit |
| **CRITICAL** | Topology invariant violation | β₀>1 or β₁>3 | Drift ≥ 0.25 | Outside Ω or non-convex |

### 6.2 Mapping Table: Topology Modes → Phase X Implications

| Topology Mode | P3 Synthetic | P4 Real Coupling | P5+ Future (NOT AUTHORIZED) |
|---------------|--------------|------------------|----------------------------|
| **STABLE** | ✅ Normal synthetic generation | ✅ Full observation, no alerts | Would allow policy updates |
| **DRIFT** | ⚠️ Log drift metrics | ⚠️ Log drift, correlate with divergence | Would trigger cautious mode |
| **TURBULENT** | ⚠️ Log turbulence, hypothetical abort | ⚠️ Log turbulence, escalate alert level | Would pause policy updates |
| **CRITICAL** | 🔴 Log critical, record hypothetical abort | 🔴 Log critical, maximum alert (no action) | Would block all updates |

### 6.3 Structural Implications Detail

#### 6.3.1 STABLE Mode Implications

```
P3 (Synthetic):
  - SyntheticStateGenerator produces states with β₀=1, β₁≤1
  - Persistence diagrams should show minimal drift
  - This is the expected baseline mode for healthy runs
  - Success criteria: Maintain STABLE for ≥90% of cycles

P4 (Real Coupling):
  - TDA adapter reads real metrics confirming STABLE
  - Shadow twin predictions should align with real topology
  - Divergence analysis baseline established
  - Bundle provenance should correlate cleanly

Structural Consequence:
  - Safe region Ω is well-defined and simply connected
  - State trajectories have bounded curvature
  - Governance signals are harmonized
```

#### 6.3.2 DRIFT Mode Implications

```
P3 (Synthetic):
  - Synthetic generator may produce controlled drift scenarios
  - Test topology observer response to gradual degradation
  - Verify drift detection and logging
  - Hypothetical: Would trigger increased monitoring

P4 (Real Coupling):
  - Real TDA metrics show drift from baseline
  - Correlate: Does topology drift precede P4 divergence?
  - Cross-correlation engine activates detailed logging
  - Bundle provenance checked for anomalies

Structural Consequence:
  - Safe region Ω may be approaching boundary complexity
  - Trajectory curvature increasing
  - Early warning signal for potential instability
```

#### 6.3.3 TURBULENT Mode Implications

```
P3 (Synthetic):
  - Synthetic stress tests: Can system detect turbulence?
  - Log detailed topology metrics during turbulence
  - Record hypothetical abort points
  - Measure recovery trajectory if turbulence resolves

P4 (Real Coupling):
  - Real execution showing significant topology variation
  - High correlation expected with P4 divergence events
  - Bundle provenance may show gaps or inconsistencies
  - Maximum logging verbosity activated

Structural Consequence:
  - Safe region Ω has increasing complexity
  - Potential for disconnected state space regions
  - Governance signals may show tension
```

#### 6.3.4 CRITICAL Mode Implications

```
P3 (Synthetic):
  - Synthetic extreme scenarios: Force topology violations
  - Verify critical detection and logging
  - Record precise hypothetical abort point
  - Ensure logging does not fail under critical conditions

P4 (Real Coupling):
  - Real execution has topology invariant violation
  - Immediate correlation with all divergence events
  - Full bundle audit trail captured
  - SHADOW MODE: Log everything, enforce nothing

Structural Consequence:
  - Safe region Ω has topological defect (hole, disconnection)
  - State trajectory may be outside recoverable region
  - Governance signals show DIVERGENT alignment
  - Evidence pack captures maximum detail for post-hoc analysis
```

### 6.4 Mode Transition Matrix

| From \ To | STABLE | DRIFT | TURBULENT | CRITICAL |
|-----------|--------|-------|-----------|----------|
| **STABLE** | Normal | TOPO-WARN-001 | Rare (skip) | TOPO-CRIT-* |
| **DRIFT** | Recovery | Persist | TOPO-WARN-002 | TOPO-CRIT-* |
| **TURBULENT** | Recovery | Partial recovery | Persist | TOPO-CRIT-* |
| **CRITICAL** | Full recovery | Partial recovery | Degraded recovery | Persist |

---

## 7. Schema References

### 7.1 Schema Locations

| Schema | Path | Purpose |
|--------|------|---------|
| `topology_drift_compass` | `docs/system_law/schemas/topology/topology_drift_compass.schema.json` | P3/P4 topology drift observation |
| `topology_bundle_joint_view` | `docs/system_law/schemas/topology/topology_bundle_joint_view.schema.json` | Combined topology-bundle snapshot |
| `topology_bundle_director_panel` | `docs/system_law/schemas/topology/topology_bundle_director_panel.schema.json` | Director panel state |

### 7.2 Schema Binding to Code

| Schema | Binding File | Status |
|--------|-------------|--------|
| `topology_drift_compass` | `backend/topology/drift_compass.py` | TODO |
| `topology_bundle_joint_view` | `backend/topology/joint_view.py` | TODO |
| `topology_bundle_director_panel` | `backend/topology/director_panel_adapter.py` | TODO |

---

## 8. Authorization Gates

### 8.1 Current Authorization (Phase X Design)

| Capability | Authorized | Notes |
|------------|------------|-------|
| Requirements document | ✅ Yes | This document |
| JSON schemas | ✅ Yes | Schema definitions only |
| TODO anchors | ✅ Yes | Integration placeholders |
| Mapping tables | ✅ Yes | Documentation only |

### 8.2 NOT Authorized (Requires Future Phase)

| Capability | Phase Required |
|------------|----------------|
| TopologyDivergenceCorrelator implementation | P4 execution authorization |
| CrossCorrelationEngine implementation | P4 execution authorization |
| DirectorPanelAdapter implementation | P4 execution authorization |
| Active topology governance | P5+ (not designed) |
| Bundle-gated policy enforcement | P5+ (not designed) |

---

## 9. P5 Expectations and Interpretation Guidance

### 9.1 Expected Signal Behavior: Real Telemetry vs. Mock

Once P4 transitions from synthetic/mock telemetry to real runner telemetry, the topology/bundle signal characteristics are expected to shift in predictable ways:

| Signal Aspect | P3/P4 Mock Behavior | P5 Real Telemetry Expectation |
|---------------|---------------------|-------------------------------|
| **XCOR-\* codes** | Frequent due to synthetic timing jitter and mock bundle generation | Reduced frequency; real bundles have consistent timing and authentic provenance |
| **bundle_stability** | May fluctuate (ATTENTION) due to mock trace coverage gaps | Should stabilize to VALID for core slices with mature instrumentation |
| **topology_mode** | May show DRIFT during synthetic stress tests | Should remain STABLE for well-characterized slices; DRIFT only on genuine anomalies |
| **cross_system_consistency** | Often false due to mock-to-mock alignment challenges | Should be true for slices with end-to-end real provenance |
| **XCOR-WARN-002** (temporal mismatch) | Common artifact of mock clock skew | Should be rare; real systems have synchronized timestamps |

**Key Hypothesis for P5 Validation:**

When real telemetry replaces mock data, we expect:
1. **Noise reduction**: XCOR-WARN-* codes should drop by 50-70% for mature slices
2. **Stability convergence**: Core arithmetic/logic slices should maintain `bundle_stability=VALID` for >95% of cycles
3. **Correlation signal improvement**: `topology_predicts_divergence` confidence should increase as real topology metrics correlate with genuine divergence events (not mock artifacts)
4. **Triple fault rarity**: XCOR-CRIT-001 (triple fault) should be genuinely rare, not a mock artifact

If these expectations are not met, it indicates either:
- Instrumentation gaps in the real telemetry pipeline
- Genuine structural issues requiring investigation
- Calibration drift between P4 shadow twin and real execution

### 9.2 Interpreting Topology/Bundle Tiles When Multiple Systems Are Active

When replay governance, telemetry governance, RTTS (Real-Time Telemetry Stream), and topology/bundle tiles are all active simultaneously, interpretation requires understanding the layered signal hierarchy:

**Signal Priority and Independence:**
The topology/bundle tile operates independently of replay and telemetry tiles—it does not consume their outputs, nor do they consume its outputs. Each tile observes a different facet of system health: replay governance monitors determinism and reproducibility, telemetry governance monitors metric consistency and drift, and topology/bundle monitors structural invariants and provenance chain integrity. When all three report GREEN/OK, the system is healthy across all observable dimensions. When they diverge, the divergence itself is diagnostic: a GREEN topology tile with a RED replay tile suggests the structural manifold is intact but execution reproducibility has degraded. Conversely, RED topology with GREEN replay suggests the proof structure is sound but the underlying state space geometry has shifted.

**Cross-Tile Correlation Patterns:**
Auditors should watch for correlation patterns that indicate systemic issues rather than isolated anomalies. If topology/bundle shows TENSION while replay shows WARN, investigate whether bundle provenance gaps are causing replay hash mismatches. If RTTS reports high latency while topology shows DRIFT, the drift may be an artifact of delayed metric propagation rather than genuine topological instability. The topology/bundle tile's `conflict_codes` field is particularly useful here: XCOR-WARN-001 (topology drift predicts divergence) should correlate with subsequent replay or telemetry warnings within a 5-10 cycle window. If XCOR codes fire without corresponding downstream effects, the cross-correlation thresholds may need recalibration.

**Practical Guidance for P5 Auditors:**
When reviewing a P5 evidence pack with all tiles active: (1) Start with the topology/bundle `joint_status`—if ALIGNED, structural health is nominal and other tiles are primary diagnostic sources. (2) If `joint_status` is TENSION, check the `conflict_codes` and `structural_notes` in the P4 calibration summary to understand which subsystem is under stress. (3) Cross-reference topology mode transitions (visible in `mode_history`) with replay safety events and telemetry drift alerts to establish temporal causation. (4) The `cross_system_consistency` boolean is the canonical indicator of end-to-end provenance health; if false, prioritize bundle chain investigation over topology metric analysis. The topology/bundle tile is designed to be a structural early-warning system—its primary value in P5 is detecting geometric anomalies before they manifest as downstream governance failures.

### 9.3 P5 Validation Scenarios

The following four scenarios define expected topology/bundle tile states for validation testing. Each scenario specifies the expected tile configuration to verify correct signal propagation.

#### Scenario 1: MOCK Baseline (Current P4)
# SPEC-ONLY

**Context:** P4 shadow mode with synthetic telemetry and mock bundle generation.

**Expected Tile State:**
```
status_light: YELLOW (intermittent GREEN)
topology_stability: STABLE or DRIFTING
bundle_stability: ATTENTION (occasional VALID)
joint_status: TENSION (frequent) or ALIGNED (intermittent)
cross_system_consistency: false (>40% of cycles)
conflict_codes: [XCOR-WARN-001, XCOR-WARN-002] (common)
```

**Characteristic Signals:**
- XCOR-WARN-002 fires frequently due to mock clock skew
- bundle_stability oscillates due to synthetic trace coverage gaps
- topology_mode may show DRIFT during stress test windows
- cross_system_consistency is unreliable indicator (mock-to-mock alignment)

**Validation Criteria:**
- [ ] XCOR-WARN-* codes present in >30% of cycles
- [ ] bundle_stability shows ATTENTION in >20% of cycles
- [ ] No XCOR-CRIT-001 (triple fault) unless stress test active

---

#### Scenario 2: HEALTHY P5 Run
# SPEC-ONLY

**Context:** P5 with real telemetry, mature slice, nominal execution.

**Expected Tile State:**
```
status_light: GREEN
topology_stability: STABLE
bundle_stability: VALID
joint_status: ALIGNED
cross_system_consistency: true
conflict_codes: [] (empty or [XCOR-OK-001] only)
```

**Characteristic Signals:**
- topology_mode remains STABLE for >95% of cycles
- bundle_stability holds VALID continuously
- No XCOR-WARN-* codes (or <5% of cycles)
- cross_system_consistency is true throughout

**Validation Criteria:**
- [ ] status_light GREEN for >90% of run
- [ ] conflict_codes empty for >95% of cycles
- [ ] cross_system_consistency true for entire run
- [ ] No topology mode transitions (STABLE throughout)

---

#### Scenario 3: MISMATCH (Topology Stable, Bundle Broken)
# SPEC-ONLY

**Context:** Topology metrics are healthy but bundle provenance chain is compromised.

**Expected Tile State:**
```
status_light: RED
topology_stability: STABLE
bundle_stability: BROKEN
joint_status: DIVERGENT
cross_system_consistency: false
conflict_codes: [BNDL-CRIT-001, XCOR-CRIT-001]
```

**Characteristic Signals:**
- topology_mode remains STABLE (geometry intact)
- bundle_stability shows BROKEN (hash mismatch or manifest gap)
- BNDL-CRIT-001 or BNDL-CRIT-002 active
- XCOR-CRIT-001 may fire (triple fault: topology ok + bundle broken + divergence)
- joint_status is DIVERGENT (topology and bundle disagree)

**Validation Criteria:**
- [ ] topology_stability remains STABLE (not affected by bundle failure)
- [ ] bundle_stability is BROKEN (not ATTENTION)
- [ ] joint_status is DIVERGENT (not TENSION)
- [ ] structural_notes contain bundle-specific diagnostic

**Triage Path:** Bundle chain investigation → manifest audit → trace coverage check

---

#### Scenario 4: XCOR Anomaly (Timing/Clock Skew)
# SPEC-ONLY

**Context:** Real telemetry with clock synchronization issues between subsystems.

**Expected Tile State:**
```
status_light: YELLOW
topology_stability: STABLE or DRIFTING
bundle_stability: VALID or ATTENTION
joint_status: TENSION
cross_system_consistency: false
conflict_codes: [XCOR-WARN-002] (temporal mismatch)
```

**Characteristic Signals:**
- XCOR-WARN-002 fires persistently (temporal mismatch)
- topology_mode may show spurious DRIFT (delayed metrics)
- bundle_stability may oscillate (timing-dependent hash windows)
- cross_system_consistency is false due to alignment failures
- No XCOR-CRIT-001 (not a triple fault, just timing)

**Validation Criteria:**
- [ ] XCOR-WARN-002 present in >50% of cycles
- [ ] No BNDL-CRIT-* codes (bundle itself is valid)
- [ ] No TOPO-CRIT-* codes (topology itself is valid)
- [ ] joint_status is TENSION (not DIVERGENT)

**Triage Path:** Clock sync investigation → RTTS latency check → timestamp alignment audit

---

### 9.4 P5 Auditor Runbook (10-Step)

This runbook provides a systematic approach for auditing topology/bundle tiles when P5 real telemetry is active alongside replay, telemetry, and RTTS systems.

#### Step 1: Establish Context
- **Inspect:** `run_context.phase` (should be "P5"), `run_context.runner_type`, `run_context.slice_name`
- **Action:** Confirm this is a P5 real-telemetry run, not P3/P4 synthetic

#### Step 2: Check Joint Status
- **Inspect:** `joint_status` field in topology_bundle tile
- **Interpretation:**
  - `ALIGNED` → Structural health nominal, proceed to other tiles
  - `TENSION` → Subsystem stress detected, continue to Step 3
  - `DIVERGENT` → Critical structural disagreement, escalate immediately

#### Step 3: Evaluate Status Light
- **Inspect:** `status_light` (GREEN/YELLOW/RED)
- **Interpretation:**
  - `GREEN` → All subsystems nominal
  - `YELLOW` → Warning condition, investigate conflict_codes
  - `RED` → Critical condition, identify which subsystem (topology or bundle)

#### Step 4: Identify Active Conflict Codes
- **Inspect:** `conflict_codes` array
- **Classification:**
  - `TOPO-*` codes → Topology subsystem issue
  - `BNDL-*` codes → Bundle subsystem issue
  - `XCOR-*` codes → Cross-correlation/timing issue
- **Action:** Note the highest severity code (OK < WARN < CRIT)

#### Step 5: Determine Issue Domain
- **Decision Tree:**
  ```
  IF conflict_codes contains TOPO-CRIT-* → Issue is TOPOLOGY
  ELSE IF conflict_codes contains BNDL-CRIT-* → Issue is BUNDLE
  ELSE IF conflict_codes contains XCOR-* only → Issue is EXTERNAL (timing/sync)
  ELSE IF topology_stability != STABLE → Issue is TOPOLOGY
  ELSE IF bundle_stability != VALID → Issue is BUNDLE
  ELSE → Issue is UNKNOWN (investigate cross-tile correlation)
  ```

#### Step 6: Cross-Reference with Replay Tile
- **Inspect:** replay_governance tile `status`
- **Correlation:**
  - Topology TENSION + Replay WARN → Bundle gaps causing hash mismatches
  - Topology GREEN + Replay RED → Reproducibility issue, not structural
  - Topology RED + Replay GREEN → Geometry shift, proofs still valid

#### Step 7: Cross-Reference with Telemetry Tile
- **Inspect:** telemetry_governance tile `status`
- **Correlation:**
  - Topology DRIFT + Telemetry high latency → Drift may be metric delay artifact
  - Topology STABLE + Telemetry WARN → Metric drift, geometry intact
  - Topology CRIT + Telemetry OK → Genuine structural anomaly

#### Step 8: Check Cross-System Consistency
- **Inspect:** `cross_system_consistency` boolean
- **Interpretation:**
  - `true` → End-to-end provenance verified
  - `false` → Provenance chain broken; prioritize bundle investigation over topology
- **Action:** If false, examine `structural_notes` in P4 calibration for specifics

#### Step 9: Review Mode History (if available)
- **Inspect:** `mode_history` array in director_panel
- **Action:** Identify when topology mode transitions occurred
- **Correlation:** Compare transition timestamps with replay/telemetry events
- **Pattern:** STABLE → DRIFT should correlate with downstream warnings within 5-10 cycles

#### Step 10: Document Findings and Hypothesis
- **Record:**
  - Primary issue domain (TOPOLOGY / BUNDLE / EXTERNAL)
  - Active conflict codes and their interpretation
  - Cross-tile correlation findings
  - Recommended triage path
- **Hypothesis Format:**
  ```
  P5_HYPOTHESIS: [TOPOLOGY|BUNDLE|EXTERNAL|UNKNOWN]
  CONFIDENCE: [HIGH|MEDIUM|LOW]
  EVIDENCE: [list of supporting signals]
  NEXT_ACTION: [specific investigation step]
  ```

---

### 9.5 Code-Facing Hook Plan

The following function signatures are proposed for P5 real-telemetry validation. These are **# REAL-READY** specifications—implementation is authorized once P5 execution begins.

#### Primary P5 Summary Builder

```python
# REAL-READY

def build_p5_topology_reality_summary(
    topology_tile: Dict[str, Any],
    bundle_tile: Dict[str, Any],
    replay_tile: Optional[Dict[str, Any]] = None,
    telemetry_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build unified P5 topology/reality summary from all active tiles.

    STATUS: # REAL-READY — Implement when P5 execution authorized

    Aggregates topology/bundle tile with replay and telemetry tiles to produce
    a unified P5 validation summary suitable for auditor review and automated
    hypothesis generation.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        topology_tile: Topology bundle console tile from build_topology_bundle_console_tile()
        bundle_tile: Bundle snapshot from joint_view (for raw bundle_status)
        replay_tile: Optional replay governance tile
        telemetry_tile: Optional telemetry governance tile

    Returns:
        P5 reality summary dict with:
        - joint_status: str (ALIGNED, TENSION, DIVERGENT)
        - cross_system_consistency: bool
        - xcor_codes: List[str] (XCOR-* codes only)
        - p5_hypothesis: Dict with:
            - domain: str (TOPOLOGY, BUNDLE, EXTERNAL, NOMINAL)
            - confidence: str (HIGH, MEDIUM, LOW)
            - evidence: List[str]
            - recommended_action: str
        - cross_tile_correlation: Dict with:
            - replay_alignment: str (ALIGNED, DIVERGENT, UNKNOWN)
            - telemetry_alignment: str (ALIGNED, DIVERGENT, UNKNOWN)
            - correlation_notes: List[str]
        - scenario_match: str (HEALTHY, MOCK_BASELINE, MISMATCH, XCOR_ANOMALY, UNKNOWN)

    Example:
        >>> summary = build_p5_topology_reality_summary(
        ...     topology_tile=topo_tile,
        ...     bundle_tile=bundle_snap,
        ...     replay_tile=replay_gov,
        ...     telemetry_tile=telem_gov,
        ... )
        >>> summary["p5_hypothesis"]["domain"]
        'NOMINAL'
    """
    pass  # Implementation deferred until P5 authorization
```

#### Scenario Matcher

```python
# REAL-READY

def match_p5_validation_scenario(
    topology_tile: Dict[str, Any],
    consistency_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Match current tile state to P5 validation scenarios.

    STATUS: # REAL-READY — Implement when P5 execution authorized

    Compares tile fields against the four canonical P5 validation scenarios
    (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY) and returns the best match
    with confidence score.

    Args:
        topology_tile: Topology bundle console tile
        consistency_result: Cross-system consistency result

    Returns:
        Scenario match result with:
        - scenario: str (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY, UNKNOWN)
        - confidence: float (0.0-1.0)
        - matching_criteria: List[str] (which criteria matched)
        - divergent_criteria: List[str] (which criteria did not match)
    """
    pass  # Implementation deferred until P5 authorization
```

#### Auditor Report Generator

```python
# REAL-READY

def generate_p5_auditor_report(
    p5_summary: Dict[str, Any],
    run_id: str,
    slice_name: str,
) -> Dict[str, Any]:
    """
    Generate structured auditor report from P5 summary.

    STATUS: # REAL-READY — Implement when P5 execution authorized

    Produces a report following the 10-step auditor runbook, with each step's
    findings pre-populated from the P5 summary.

    Args:
        p5_summary: Output from build_p5_topology_reality_summary()
        run_id: Run identifier
        slice_name: Curriculum slice name

    Returns:
        Auditor report dict with:
        - run_context: Dict (run_id, slice_name, timestamp)
        - runbook_steps: List[Dict] (10 steps with findings)
        - final_hypothesis: Dict (domain, confidence, evidence, action)
        - escalation_required: bool
        - escalation_reason: Optional[str]
    """
    pass  # Implementation deferred until P5 authorization
```

---

### 9.6 P5 Smoke-Test Readiness Checklist

Use this checklist to verify readiness before running topology/bundle P5 validation against real telemetry.

#### Infrastructure Readiness

- [ ] **RTTS Active:** Real-Time Telemetry Stream is live and producing data
- [ ] **Clock Sync Verified:** All subsystems have synchronized timestamps (NTP or equivalent)
- [ ] **Bundle Pipeline Live:** Real bundle generation is active (not mock)
- [ ] **TDA Adapter Connected:** TDA metrics are flowing from real execution

#### Tile Integration Readiness

- [ ] **topology_bundle tile attached:** `build_global_health_surface()` includes topology_bundle
- [ ] **replay_governance tile attached:** Replay tile is present in global health
- [ ] **telemetry_governance tile attached:** Telemetry tile is present in global health
- [ ] **Evidence chain configured:** `attach_topology_bundle_to_evidence()` is wired

#### P3/P4 Baseline Established

- [ ] **P3 synthetic run completed:** At least one full P3 run with topology/bundle logging
- [ ] **P4 mock run completed:** At least one full P4 run with mock telemetry
- [ ] **MOCK_BASELINE scenario validated:** Tile output matches Scenario 1 expectations
- [ ] **XCOR-WARN-002 baseline recorded:** Frequency of clock skew warnings in mock documented

#### Validation Scenario Preparation

- [ ] **HEALTHY scenario test case ready:** Known-good slice identified for HEALTHY validation
- [ ] **MISMATCH scenario test case ready:** Intentional bundle corruption scenario prepared
- [ ] **XCOR_ANOMALY scenario test case ready:** Clock skew injection mechanism available
- [ ] **Scenario matching criteria documented:** Validation criteria from Section 9.3 available

#### Auditor Tooling Readiness

- [ ] **Runbook accessible:** 10-step auditor runbook (Section 9.4) available to team
- [ ] **Hypothesis template ready:** P5_HYPOTHESIS format documented
- [ ] **Cross-tile correlation guide available:** Interpretation guidance accessible
- [ ] **Escalation path defined:** Who to contact for DIVERGENT findings

#### Monitoring and Alerting

- [ ] **status_light monitoring:** Dashboard shows topology_bundle status_light
- [ ] **conflict_codes alerting:** Alerts configured for XCOR-CRIT-001 (triple fault)
- [ ] **cross_system_consistency tracking:** Metric tracked over time
- [ ] **XCOR-WARN-* rate tracking:** Alert if XCOR-WARN rate exceeds P4 baseline by >50%

#### Sign-Off

- [ ] **Infrastructure owner sign-off:** _______________
- [ ] **Topology/bundle owner sign-off:** _______________
- [ ] **Auditor lead sign-off:** _______________
- [ ] **P5 execution authorization obtained:** _______________

---

## 10. P5 Reality Adapter Wiring Plan

### 10.1 Implementation Summary

The P5 Topology Reality Adapter has been implemented in `backend/health/p5_topology_reality_adapter.py` with the following components:

| Function | Purpose | Status |
|----------|---------|--------|
| `extract_topology_reality_metrics()` | Topology reality extraction | ✅ IMPLEMENTED |
| `validate_bundle_stability()` | Bundle stability validator | ✅ IMPLEMENTED |
| `detect_xcor_anomaly()` | XCOR anomaly detector (real telemetry mode) | ✅ IMPLEMENTED |
| `run_p5_smoke_validation()` | 3-case smoke validator | ✅ IMPLEMENTED |
| `match_p5_validation_scenario()` | Scenario matcher (REAL-READY signature) | ✅ IMPLEMENTED |
| `build_p5_topology_reality_summary()` | P5 reality summary builder | ✅ IMPLEMENTED |
| `generate_p5_auditor_report()` | 10-step auditor report generator | ✅ IMPLEMENTED |

### 10.2 Harness Integration Wiring

To integrate the P5 Reality Adapter into the First Light harness:

#### Step 1: Import the Adapter

```python
from backend.health.p5_topology_reality_adapter import (
    extract_topology_reality_metrics,
    validate_bundle_stability,
    detect_xcor_anomaly,
    run_p5_smoke_validation,
    build_p5_topology_reality_summary,
    generate_p5_auditor_report,
)
```

#### Step 2: Wire into First Light Evidence Chain

```python
# In first_light harness (after building topology_bundle_console_tile):

# 1. Extract reality metrics
topology_metrics = extract_topology_reality_metrics(
    topology_tile=console_tile,
    joint_view=joint_view,  # Optional
)

# 2. Validate bundle stability
bundle_validation = validate_bundle_stability(
    topology_tile=console_tile,
    joint_view=joint_view,
    consistency_result=consistency_result,
)

# 3. Detect XCOR anomalies
xcor_detection = detect_xcor_anomaly(
    topology_tile=console_tile,
    topology_metrics=topology_metrics,
    bundle_validation=bundle_validation,
)

# 4. Run smoke validation
smoke_result = run_p5_smoke_validation(
    topology_tile=console_tile,
    consistency_result=consistency_result,
    joint_view=joint_view,
)

# 5. Build P5 summary (with optional cross-tile correlation)
p5_summary = build_p5_topology_reality_summary(
    topology_tile=console_tile,
    bundle_tile=bundle_snapshot,
    replay_tile=replay_governance_tile,      # Optional
    telemetry_tile=telemetry_governance_tile,  # Optional
)

# 6. Generate auditor report
auditor_report = generate_p5_auditor_report(
    p5_summary=p5_summary,
    run_id=run_context.run_id,
    slice_name=run_context.slice_name,
)
```

#### Step 3: Attach to Evidence Pack

```python
from backend.health.topology_bundle_adapter import attach_topology_bundle_to_evidence

# Attach P5 reality data to evidence
evidence["p5_topology_reality"] = {
    "topology_metrics": topology_metrics,
    "bundle_validation": bundle_validation,
    "xcor_detection": xcor_detection,
    "smoke_validation": smoke_result,
    "p5_summary": p5_summary,
    "auditor_report": auditor_report,
}
```

### 10.3 Test Coverage

Unit tests are in `tests/health/test_p5_topology_reality_adapter.py`:

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestExtractTopologyRealityMetrics` | 4 | Extraction, derivation, bounds detection |
| `TestValidateBundleStability` | 4 | Validation, broken detection, derivation |
| `TestDetectXcorAnomaly` | 5 | Anomaly types, triple fault, divergent state |
| `TestRunP5SmokeValidation` | 8 | 3-case scenarios, override, diagnostics |
| `TestMatchP5ValidationScenario` | 3 | Scenario matching, output format |
| `TestBuildP5TopologyRealitySummary` | 6 | Summary building, cross-tile correlation |
| `TestGenerateP5AuditorReport` | 6 | Report structure, 10-step runbook, escalation |
| `TestShadowModeInvariants` | 5 | SHADOW MODE contract verification |
| `TestP5ScenariosConstant` | 3 | Scenario definitions |
| **Total** | **44** | **All PASS** |

### 10.4 SHADOW MODE Compliance

All functions in the P5 Reality Adapter maintain SHADOW MODE contract:

- ✅ All functions are read-only (aside from dict construction)
- ✅ No enforcement logic present
- ✅ No abort/gate/block fields in outputs
- ✅ `shadow_mode_invariant_ok` always returns `True`
- ✅ All outputs are JSON-serializable for evidence logging

### 10.5 Scenario Validation Matrix

| Scenario | Expected `status_light` | Expected `topology_stability` | Expected `bundle_stability` | Test Coverage |
|----------|------------------------|------------------------------|----------------------------|---------------|
| MOCK_BASELINE | YELLOW | STABLE/DRIFTING | ATTENTION | ✅ |
| HEALTHY | GREEN | STABLE | VALID | ✅ |
| MISMATCH | RED | STABLE | BROKEN | ✅ |
| XCOR_ANOMALY | YELLOW | STABLE/DRIFTING | VALID/ATTENTION | ✅ |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **β₀** | 0th Betti number: count of connected components |
| **β₁** | 1st Betti number: count of 1-dimensional holes (cycles) |
| **Persistence Diagram** | TDA summary of topological features across scales |
| **Bottleneck Distance** | Metric for comparing persistence diagrams |
| **Ω** | Safe control region (USLA formalism) |
| **Bundle** | Provenance package with manifest, traces, hashes |
| **SHADOW MODE** | Observation-only, no governance enforcement |

---

*Document Version: 1.2.0*
*Last Updated: 2025-12-11*
*Status: Design Specification (P5 Reality Adapter Implemented)*
