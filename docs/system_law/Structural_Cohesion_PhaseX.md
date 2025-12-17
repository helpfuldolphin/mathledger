# Structural Cohesion Phase X: Cross-Layer Structural Governance

---

> **PHASE X STRUCTURAL LAW — DESIGN SPECIFICATION**
>
> This document defines the structural cohesion requirements binding DAG, Topology, and HT layers
> into P3/P4 doctrine as structural safety invariants.
>
> **Status**: Design Freeze
> **Version**: 1.0.0
> **Date**: 2025-12-10

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Structural Layers](#2-structural-layers)
3. [Structural Invariants for Admissible P4 Runs](#3-structural-invariants-for-admissible-p4-runs)
4. [Divergence Severity Semantics](#4-divergence-severity-semantics)
5. [Structural Plots for Whitepaper](#5-structural-plots-for-whitepaper)
6. [Integration Points](#6-integration-points)
7. [Schema References](#7-schema-references)
8. [TODO Anchors](#8-todo-anchors)
9. [P5 RTTS Cross-Reference](#9-p5-rtts-cross-reference)

---

## 1. Executive Summary

Structural Cohesion Phase X establishes the formal binding between three architectural layers:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **DAG** | `backend/dag/` | Proof lineage, acyclicity, node/edge invariants |
| **Topology** | `backend/topology/` | USLA state space, shadow observation, divergence analysis |
| **HT** | `backend/ht/` | Hypothesis tracking, truth anchor validation |

These layers must maintain **structural consistency** for any P4 run to be considered admissible.
Violations produce divergence signals with classified severity that feed into governance analysis.

### Core Principle

**Structural integrity precedes behavioral analysis.** A P4 shadow experiment cannot produce
trustworthy divergence metrics if the underlying structural invariants are violated. This
document specifies:

1. Which invariants must hold for P4 admissibility
2. How violations map to divergence severity levels
3. What structural plots must be generated for the whitepaper

---

## 2. Structural Layers

### 2.1 DAG Layer (`backend/dag/`)

The DAG layer maintains proof lineage and derivation structure.

**Key Components:**
- `proof_dag.py`: Core DAG operations
- `invariant_guard.py`: Invariant checking and enforcement
- `schema_validator.py`: Schema validation for DAG structures

**Structural Properties:**
```
DAG Structure
│
├── Acyclicity
│   └── No node is reachable from itself via edges
│
├── Node Integrity
│   ├── Required fields present (id, type, etc.)
│   └── Node data satisfies slice-specific constraints
│
├── Edge Integrity
│   ├── Both endpoints exist
│   └── Edge types are valid for source/target node types
│
└── Slice Constraints
    ├── Max depth per slice
    ├── Allowed node kinds per slice
    └── Branching factor limits
```

### 2.2 Topology Layer (`backend/topology/`)

The Topology layer manages USLA state evolution and shadow observation.

**Key Components:**
- `usla_integration.py`: USLA state machine
- `divergence_monitor.py`: Real-time divergence tracking
- `first_light/`: P3/P4 shadow experiment infrastructure

**Structural Properties:**
```
Topology Structure
│
├── State Space Bounds
│   ├── H ∈ [0, 1]
│   ├── ρ ∈ [0, 1]
│   ├── τ ∈ [0.16, 0.24] (Goldilocks zone)
│   └── β ∈ [0, 1]
│
├── Safe Region (Ω)
│   └── Defined by (H, ρ, τ, β) constraints
│
├── Transition Integrity
│   └── State transitions follow USLA dynamics
│
└── Observation Consistency
    ├── Real observations match expected structure
    └── Twin predictions are well-formed
```

### 2.3 HT Layer (`backend/ht/`)

The HT (Hypothesis Tracking) layer manages truth anchor validation.

**Key Components:**
- Currently stub: `__init__.py`

**Structural Properties:**
```
HT Structure
│
├── Hypothesis Registry
│   └── Tracked hypotheses with verification status
│
├── Truth Anchors
│   ├── Lean verification results
│   └── Cryptographic attestations
│
└── Derivation Lineage
    └── Links to DAG proof paths
```

---

## 3. Structural Invariants for Admissible P4 Runs

### 3.1 Invariant Table

| ID | Invariant | Layer | Check | Violation Impact |
|----|-----------|-------|-------|------------------|
| **SI-001** | DAG Acyclicity | DAG | `check_dag_invariants()` | P4 run INADMISSIBLE |
| **SI-002** | Node Required Fields | DAG | `verify_node_invariant()` | CONFLICT |
| **SI-003** | Slice Depth Bounds | DAG | `evaluate_dag_invariants()` | TENSION |
| **SI-004** | Slice Node Kind Constraints | DAG | `evaluate_dag_invariants()` | TENSION |
| **SI-005** | State Space Bounds | Topology | State validation | CONFLICT |
| **SI-006** | Safe Region Definition | Topology | Ω membership check | TENSION |
| **SI-007** | Observation Structure | Topology | Schema validation | CONFLICT |
| **SI-008** | Twin Prediction Structure | Topology | Schema validation | CONFLICT |
| **SI-009** | Divergence Log Structure | Topology | Schema validation | CONFLICT |
| **SI-010** | Truth Anchor Integrity | HT | Attestation verification | P4 run INADMISSIBLE |

### 3.2 Admissibility Gate

A P4 run is **ADMISSIBLE** if and only if:

```
ADMISSIBLE(P4_run) ⟺
    ∀ inv ∈ {SI-001, SI-010}: CHECK(inv) = PASS
    ∧
    COUNT({inv | inv ∈ {SI-002..SI-009} ∧ CHECK(inv) = FAIL}) ≤ TOLERANCE_THRESHOLD
```

**Default Tolerance Threshold**: 0 (strict mode)

### 3.3 Pre-Run Structural Check

Before any P4 run, the following structural check must pass:

```python
def structural_admissibility_check(
    dag: ProofDag,
    topology_state: Dict[str, Any],
) -> StructuralAdmissibilityResult:
    """
    Check structural invariants for P4 admissibility.

    Returns:
        StructuralAdmissibilityResult with admissible flag and violations
    """
    violations = []

    # SI-001: DAG Acyclicity (CRITICAL)
    dag_result = check_dag_invariants(dag.nodes, list(dag.edges))
    if not dag_result.valid:
        violations.append(StructuralViolation(
            invariant_id="SI-001",
            severity="CONFLICT",
            message="DAG acyclicity violation",
            details=dag_result.violations,
        ))

    # SI-003, SI-004: Slice constraints
    slice_result = evaluate_dag_invariants(dag)
    if slice_result["status"] != "OK":
        for v in slice_result["violated_invariants"]:
            violations.append(StructuralViolation(
                invariant_id=f"SI-00{3 if v['invariant'] == 'max_depth' else 4}",
                severity="TENSION",
                message=v["message"],
                details=v,
            ))

    # SI-005: State space bounds
    if not _validate_state_bounds(topology_state):
        violations.append(StructuralViolation(
            invariant_id="SI-005",
            severity="CONFLICT",
            message="State space bounds violation",
            details=topology_state,
        ))

    admissible = all(
        v.severity != "CONFLICT" or v.invariant_id not in {"SI-001", "SI-010"}
        for v in violations
    )

    return StructuralAdmissibilityResult(
        admissible=admissible,
        violations=violations,
        checked_at=datetime.utcnow().isoformat(),
    )
```

---

## 4. Divergence Severity Semantics

### 4.1 Severity Levels

Structural violations map to three severity levels that inform governance decisions:

| Severity | Code | Semantics | Governance Impact |
|----------|------|-----------|-------------------|
| **CONFLICT** | `C` | Structural integrity compromised; data unreliable | P4 run INADMISSIBLE or results INVALIDATED |
| **TENSION** | `T` | Structural constraint exceeded; results degraded | Results flagged with confidence reduction |
| **CONSISTENT** | `S` | All structural invariants satisfied | Full confidence in results |

### 4.2 Divergence Classification Matrix

```
                    │ DAG Layer │ Topology Layer │ HT Layer │
────────────────────┼───────────┼────────────────┼──────────┤
CONFLICT (C)        │ Cycles    │ Out-of-bounds  │ Invalid  │
                    │ detected  │ state values   │ anchor   │
────────────────────┼───────────┼────────────────┼──────────┤
TENSION (T)         │ Depth     │ Prolonged Ω    │ Pending  │
                    │ exceeded  │ exit           │ verify   │
────────────────────┼───────────┼────────────────┼──────────┤
CONSISTENT (S)      │ All       │ Normal state   │ All      │
                    │ invariant │ evolution      │ anchors  │
                    │ pass      │                │ valid    │
────────────────────┴───────────┴────────────────┴──────────┘
```

### 4.3 Severity Propagation Rules

When multiple layers report different severities, apply the **maximum severity rule**:

```
COMBINED_SEVERITY = max(DAG_severity, Topology_severity, HT_severity)

where: CONFLICT > TENSION > CONSISTENT
```

### 4.4 Integration with P4 Divergence Analysis

The `DivergenceAnalyzer` (see `backend/topology/first_light/divergence_analyzer.py`)
must incorporate structural severity into its classification:

```python
# TODO: CLAUDE G - Structural invariants feeding into divergence analyzer
# Integration point: DivergenceAnalyzer._classify_severity() should
# incorporate structural layer signals from StructuralGovernanceSignal
```

---

## 5. Structural Plots for Whitepaper

### 5.1 Required Visualizations

The following structural plots must be generated for whitepaper inclusion:

| Plot ID | Title | X-Axis | Y-Axis | Description |
|---------|-------|--------|--------|-------------|
| **SP-001** | Structural Cohesion Over Time | Cycle | Cohesion Score | Combined structural integrity metric |
| **SP-002** | Layer Divergence Heat Map | Cycle | Layer | Heat map showing divergence severity per layer |
| **SP-003** | DAG Depth Distribution | Depth | Count | Histogram of proof depths across slices |
| **SP-004** | Topology State Trajectory | Time | State Variables | H, ρ, τ, β evolution over cycles |
| **SP-005** | Structural Violation Timeline | Cycle | Severity | Timeline of structural violations with severity |
| **SP-006** | Cross-Layer Correlation | Layer 1 Metric | Layer 2 Metric | Scatter plot showing layer correlations |

### 5.2 Structural Cohesion Score Calculation

The Structural Cohesion Score (SCS) is computed as:

```
SCS(t) = w_DAG × DAG_score(t) + w_Topo × Topo_score(t) + w_HT × HT_score(t)

where:
- w_DAG = 0.4 (DAG layer weight)
- w_Topo = 0.4 (Topology layer weight)
- w_HT = 0.2 (HT layer weight)

Layer scores:
- DAG_score = 1.0 - (violations / total_checks)
- Topo_score = 1.0 - (out_of_bounds_cycles / total_cycles)
- HT_score = 1.0 - (unverified_anchors / total_anchors)
```

### 5.3 Plot Data Schema

Structural plot data should be exported in the following format:

```json
{
  "plot_id": "SP-001",
  "title": "Structural Cohesion Over Time",
  "run_id": "p4_run_20251210_abc123",
  "data": {
    "x_values": [1, 2, 3, ...],
    "y_values": [0.98, 0.97, 0.99, ...],
    "x_label": "Cycle",
    "y_label": "Cohesion Score"
  },
  "annotations": [
    {"cycle": 42, "label": "SI-003 violation", "severity": "TENSION"}
  ],
  "generated_at": "2025-12-10T12:00:00Z"
}
```

---

## 6. Integration Points

### 6.1 DAG → Topology Integration

```
backend/dag/invariant_guard.py
        │
        ▼ check_dag_invariants() result
        │
backend/topology/first_light/divergence_analyzer.py
        │
        ▼ Structural signals inform divergence classification
        │
Divergence Log (JSONL)
```

### 6.2 Topology → HT Integration

```
backend/topology/first_light/runner_p4.py
        │
        ▼ Observations with structural metadata
        │
backend/ht/ (future)
        │
        ▼ Truth anchor validation
        │
Attestation Record
```

### 6.3 Cross-Layer Governance Signal Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Structural Governance Signal Flow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐                        │
│   │  DAG      │     │ Topology  │     │    HT     │                        │
│   │  Layer    │     │  Layer    │     │   Layer   │                        │
│   └─────┬─────┘     └─────┬─────┘     └─────┬─────┘                        │
│         │                 │                 │                               │
│         ▼                 ▼                 ▼                               │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                  StructuralGovernanceSignal                      │      │
│   │                                                                  │      │
│   │  {                                                               │      │
│   │    "dag_status": "CONSISTENT" | "TENSION" | "CONFLICT",         │      │
│   │    "topology_status": "CONSISTENT" | "TENSION" | "CONFLICT",    │      │
│   │    "ht_status": "CONSISTENT" | "TENSION" | "CONFLICT",          │      │
│   │    "combined_severity": "CONSISTENT" | "TENSION" | "CONFLICT",  │      │
│   │    "violations": [...],                                          │      │
│   │    "cohesion_score": 0.95                                        │      │
│   │  }                                                               │      │
│   └───────────────────────────────┬─────────────────────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      DivergenceAnalyzer                          │      │
│   │                                                                  │      │
│   │  - Incorporates structural signals into severity classification  │      │
│   │  - Adjusts confidence based on cohesion score                   │      │
│   │  - Flags results when CONFLICT detected                         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Schema References

### 7.1 Structural Governance Signal Schema

See: `docs/system_law/schemas/structural/structural_governance_signal.schema.json`

### 7.2 Structural Console Tile Schema

See: `docs/system_law/schemas/structural/structural_console_tile.schema.json`

### 7.3 Related Schemas

| Schema | Path | Purpose |
|--------|------|---------|
| P4 Divergence Log | `docs/system_law/schemas/phase_x_p4/p4_divergence_log.schema.json` | Divergence record format |
| Attestation Snapshot | `schemas/attestation_snapshot.schema.json` | Cryptographic attestation |
| Ledger Snapshot | `schemas/ledger_snapshot.schema.json` | Ledger state snapshot |

---

## 8. TODO Anchors

The following integration points require implementation:

### 8.1 DAG Layer TODOs

```python
# File: backend/dag/invariant_guard.py
# TODO: CLAUDE G - Add structural governance signal emission
# The check_dag_invariants() function should emit a StructuralGovernanceSignal
# when violations are detected. This signal feeds into the DivergenceAnalyzer.
```

### 8.2 Topology Layer TODOs

```python
# File: backend/topology/first_light/divergence_analyzer.py
# TODO: CLAUDE G - Structural invariants feeding into divergence analyzer
# The _classify_severity() method should incorporate structural layer signals:
#   1. Accept StructuralGovernanceSignal as optional parameter
#   2. Escalate severity when structural CONFLICT detected
#   3. Adjust confidence metrics based on cohesion_score
```

```python
# File: backend/topology/first_light/runner_p4.py
# TODO: CLAUDE G - Pre-run structural admissibility check
# Before starting P4 cycles, call structural_admissibility_check()
# and abort if admissible=False with reason logged.
```

### 8.3 HT Layer TODOs

```python
# File: backend/ht/__init__.py
# TODO: CLAUDE G - Implement HT structural layer
# Required components:
#   1. HypothesisRegistry class
#   2. TruthAnchorValidator class
#   3. Integration with DAG lineage
```

### 8.4 Console/Visualization TODOs

```python
# File: (new) scripts/generate_structural_plots.py
# TODO: CLAUDE G - Implement structural plot generation
# Generate SP-001 through SP-006 plots for whitepaper inclusion.
```

---

## Appendix A: Structural Cohesion Checklist

Pre-P4 Run Checklist:

- [ ] DAG acyclicity verified (SI-001)
- [ ] All nodes have required fields (SI-002)
- [ ] Slice depth within bounds (SI-003)
- [ ] Slice node kinds valid (SI-004)
- [ ] Topology state within bounds (SI-005)
- [ ] Safe region properly defined (SI-006)
- [ ] Observation schemas validated (SI-007, SI-008, SI-009)
- [ ] Truth anchors verified (SI-010)

---

## Appendix B: Severity Decision Tree

```
                        ┌─────────────────┐
                        │ Check Invariant │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │ SI-001 or    │         │ Other        │
            │ SI-010?      │         │ Invariant    │
            └──────┬───────┘         └──────┬───────┘
                   │                        │
           ┌───────┴───────┐        ┌───────┴───────┐
           │               │        │               │
           ▼               ▼        ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ FAIL       │  │ PASS       │  │ FAIL       │  │ PASS       │
    │ → CONFLICT │  │ → Continue │  │ → TENSION  │  │ → CONSISTENT│
    │ → ABORT    │  │            │  │ → CONTINUE │  │            │
    └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

---

## 9. P5 RTTS Cross-Reference

### 9.1 Structural Invariants and RTTS STRUCTURAL_BREAK Pattern

The P5 Real Telemetry Topology Specification (RTTS) defines a `STRUCTURAL_BREAK` divergence
pattern (see `Phase_X_P5_Implementation_Blueprint.md` Section 2.2.1). This pattern indicates
a sudden, persistent divergence suggesting a regime change in the real system.

**Relationship to Structural Invariants:**

| Structural Invariant | RTTS Pattern Link | Interpretation |
|---------------------|-------------------|----------------|
| **SI-001** (DAG Acyclicity) | STRUCTURAL_BREAK | A cycle introduced into the proof DAG represents a fundamental structural violation. If detected during a P5 run, the STRUCTURAL_BREAK pattern MUST be classified with `structural_cause: "SI-001_CYCLE"` |
| **SI-010** (Truth Anchor Integrity) | STRUCTURAL_BREAK | Failed truth anchor verification indicates the derivation foundation has been compromised. Pattern MUST be classified with `structural_cause: "SI-010_ANCHOR_FAIL"` |

**Detection Flow:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 SI → STRUCTURAL_BREAK Detection Flow                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  emit_structural_signal()                                               │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐                                                   │
│  │ SI-001 CONFLICT? │───Yes───┐                                         │
│  └────────┬─────────┘         │                                         │
│           │No                  │                                         │
│           ▼                    │                                         │
│  ┌──────────────────┐         │                                         │
│  │ SI-010 CONFLICT? │───Yes───┼──────────────────────────────┐         │
│  └────────┬─────────┘         │                              │         │
│           │No                  │                              ▼         │
│           ▼                    ▼                   ┌──────────────────┐ │
│   (Continue P4/P5    ┌────────────────────┐       │ DivergencePattern│ │
│    observation)      │ structural_conflict │       │ Classifier       │ │
│                      │ = True              │       └────────┬─────────┘ │
│                      └────────┬───────────┘                │           │
│                               │                             │           │
│                               ▼                             ▼           │
│                      ┌────────────────────────────────────────────┐    │
│                      │     Pattern = STRUCTURAL_BREAK              │    │
│                      │     structural_cause = "SI-001" | "SI-010"  │    │
│                      │     severity = CRITICAL                     │    │
│                      └────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Integration Point:**

The `DivergencePatternClassifier` (see `backend/topology/first_light/divergence_patterns.py`)
should accept an optional `structural_signal` parameter:

```python
def classify(
    self,
    real_series: List[float],
    twin_series: List[float],
    structural_signal: Optional[Dict[str, Any]] = None,  # From emit_structural_signal()
) -> PatternClassificationResult:
    """
    Classify divergence pattern with structural context.

    If structural_signal.admissible == False (SI-001 or SI-010 violated):
        - Return DivergencePattern.STRUCTURAL_BREAK
        - Set structural_cause in result metadata
        - Severity automatically CRITICAL
    """
```

### 9.2 Joint View: Structural Governance + P5 Divergence Patterns

**Figure Description: Unified Director Panel (Text Specification)**

The following describes the layout for a unified director panel view that combines
structural governance status with P5 divergence pattern analysis.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    DIRECTOR PANEL: Structural + P5 Unified View                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │     STRUCTURAL COHESION TILE        │  │     P5 DIVERGENCE PATTERN TILE  │  │
│  │                                     │  │                                  │  │
│  │  Overall: [HEALTHY|DEGRADED|CRIT]   │  │  Current Pattern: [PATTERN_NAME]│  │
│  │  SCS: ██████████░░░░ 85%            │  │  Confidence: 0.92               │  │
│  │                                     │  │                                  │  │
│  │  Layers:                            │  │  Decomposition:                  │  │
│  │    DAG      [●] CONSISTENT          │  │    Δ_bias:      ████░░ 0.03     │  │
│  │    Topology [◐] TENSION             │  │    Δ_variance:  ██░░░░ 0.01     │  │
│  │    HT       [●] CONSISTENT          │  │    Δ_timing:    █░░░░░ 0.005    │  │
│  │                                     │  │    Δ_structural:░░░░░░ 0.001    │  │
│  │  Active Violations: 1               │  │                                  │  │
│  │    SI-006: Omega exit 45 cycles     │  │  Severity: [INFO]               │  │
│  └─────────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    CROSS-LAYER CORRELATION STRIP                         │   │
│  │                                                                          │   │
│  │  SCS ─────●─────●─────●─────◐─────◐─────●─────────────────────  Cycle   │   │
│  │  Δp  ─────○─────○─────○─────◑─────◑─────○─────────────────────  500     │   │
│  │                         ↑                                                │   │
│  │                    Correlation point:                                    │   │
│  │                    SCS drop → Δp spike                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    STRUCTURAL-PATTERN BINDING STATUS                     │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │   │
│  │  │ SI-001 (DAG Acyclic)  │ PASS │ → No STRUCTURAL_BREAK from DAG    │  │   │
│  │  │ SI-010 (Truth Anchor) │ PASS │ → No STRUCTURAL_BREAK from HT     │  │   │
│  │  └───────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                          │   │
│  │  Escalation Advisory:                                                    │   │
│  │    Current divergence (INFO) would remain INFO                           │   │
│  │    No structural escalation active                                       │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  SHADOW MODE BANNER                                                      │   │
│  │  ═══════════════════════════════════════════════════════════════════════ │   │
│  │  All displays are OBSERVATIONAL ONLY. No enforcement actions taken.      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Panel Components:**

| Component | Data Source | Update Frequency |
|-----------|-------------|------------------|
| Structural Cohesion Tile | `build_structural_cohesion_tile()` | Per cycle |
| P5 Divergence Pattern Tile | `DivergencePatternClassifier.classify()` | Per window (50 cycles) |
| Cross-Layer Correlation Strip | Derived from SCS + Δp time series | Per cycle |
| Structural-Pattern Binding Status | SI-001/SI-010 check + escalation advisory | Per cycle |
| Shadow Mode Banner | Static | Always visible |

**Interpretation Guide:**

1. **Normal Operation**: Both tiles show green/CONSISTENT/NONE. Correlation strip shows
   independent variations within tolerance.

2. **Structural Tension without P5 Impact**: Structural tile shows TENSION (e.g., SI-006
   omega exit), but P5 tile shows NONE or DRIFT. Indicates structural pressure not yet
   affecting twin divergence.

3. **P5 Pattern without Structural Cause**: P5 tile shows PHASE_LAG or NOISE_AMPLIFICATION,
   but structural tile is HEALTHY. Twin model needs calibration, not structural issue.

4. **Coupled Degradation**: Both tiles show elevated status. Correlation strip shows
   synchronized drops. Investigate shared root cause.

5. **STRUCTURAL_BREAK Trigger**: If SI-001 or SI-010 fails, the P5 pattern MUST show
   STRUCTURAL_BREAK with corresponding `structural_cause`. This is the strongest signal
   and overrides other pattern classifications.

---

## 10. Structural Cohesion Register for Calibration Experiments

### 10.1 Overview

The **Structural Cohesion Register** aggregates structural cohesion data across multiple calibration experiments (CAL-EXP-1, CAL-EXP-2, etc.) to identify patterns and misalignments. This register provides auditors with a consolidated view of structural integrity across calibration runs.

**Purpose**: External reviewers can use this register to understand:
- Where structural cohesion remains sound across calibration experiments
- Where structural wobble occurs (misalignments)
- Patterns in lattice, lean shadow, and coherence band combinations
- Which experiments require attention due to structural issues

### 10.2 Per-Experiment Annex Capture

Each calibration experiment emits a **Structural Cohesion Annex** that captures the structural state at experiment completion:

```python
from backend.health.atlas_governance_adapter import (
    build_first_light_structural_cohesion_annex,
    emit_cal_exp_structural_cohesion_annex,
)

# Build annex from experiment tiles
annex = build_first_light_structural_cohesion_annex(
    atlas_tile=atlas_governance_tile,
    structure_tile=lean_shadow_tile,  # Optional
    coherence_tile=coherence_tile,     # Optional
)

# Emit and persist to disk
emitted = emit_cal_exp_structural_cohesion_annex(
    cal_id="CAL-EXP-1",
    annex=annex,
    output_dir="calibration/",
)
```

**Annex Schema:**
```json
{
  "schema_version": "1.0.0",
  "cal_id": "CAL-EXP-1",
  "lattice_band": "COHERENT" | "PARTIAL" | "MISALIGNED",
  "transition_status": "OK" | "ATTENTION" | "BLOCK",
  "lean_shadow_status": "OK" | "WARN" | "BLOCK" | null,
  "coherence_band": "COHERENT" | "PARTIAL" | "MISALIGNED" | null
}
```

**Persistence**: Each annex is written to `calibration/structural_cohesion_annex_{cal_id}.json` for audit trail.

### 10.3 Register Aggregation

The **Structural Cohesion Register** aggregates multiple annexes into a unified view:

```python
from backend.health.atlas_governance_adapter import (
    build_structural_cohesion_register,
)

# Load emitted annexes (from disk or in-memory)
annexes = [
    load_annex("CAL-EXP-1"),
    load_annex("CAL-EXP-2"),
    load_annex("CAL-EXP-3"),
]

# Build register
register = build_structural_cohesion_register(annexes)
```

**Register Schema:**
```json
{
  "schema_version": "1.0.0",
  "total_experiments": 3,
  "band_combinations": {
    "COHERENT×COHERENT": 2,
    "PARTIAL×PARTIAL": 1
  },
  "experiments_with_misaligned_structure": ["CAL-EXP-2"],
  "lattice_band_distribution": {
    "COHERENT": 2,
    "PARTIAL": 1
  },
  "transition_status_distribution": {
    "OK": 2,
    "ATTENTION": 1
  },
  "lean_shadow_status_distribution": {
    "OK": 2,
    "WARN": 1
  },
  "coherence_band_distribution": {
    "COHERENT": 2,
    "PARTIAL": 1
  }
}
```

### 10.4 Misaligned Structure Classification

An experiment is classified as **misaligned** if any of the following conditions hold:

1. **Lattice Misalignment**: `lattice_band == "MISALIGNED"`
2. **Lean Shadow Block**: `lean_shadow_status == "BLOCK"`
3. **Coherence Misalignment**: `coherence_band == "MISALIGNED"`

The register maintains a sorted list of `experiments_with_misaligned_structure` for deterministic reporting.

### 10.5 Evidence Integration

The register is attached to evidence packs for audit and compliance:

```python
from backend.health.atlas_governance_adapter import (
    attach_structural_cohesion_register_to_evidence,
)

# Attach register to evidence
enriched_evidence = attach_structural_cohesion_register_to_evidence(
    evidence=evidence_pack,
    register=structural_cohesion_register,
)
```

**Evidence Location**: `evidence["governance"]["structural_cohesion_register"]`

### 10.6 Auditor Reading Guide

**For External Auditors:**

1. **Check Total Experiments**: Verify `total_experiments` matches expected calibration run count.

2. **Review Misaligned Experiments**: 
   - Examine `experiments_with_misaligned_structure` list
   - For each misaligned experiment, review the corresponding annex file:
     - `calibration/structural_cohesion_annex_{cal_id}.json`
   - Identify root cause: lattice, lean shadow, or coherence issue

3. **Analyze Band Combinations**:
   - `band_combinations` shows lattice × coherence pairings
   - Look for patterns: Are most experiments `COHERENT×COHERENT`?
   - Identify outliers: Are there experiments with `MISALIGNED×COHERENT` or `COHERENT×MISALIGNED`?

4. **Review Distributions**:
   - `lattice_band_distribution`: Overall lattice convergence health
   - `transition_status_distribution`: Phase transition readiness
   - `lean_shadow_status_distribution`: Structural integrity (if available)
   - `coherence_band_distribution`: Topological alignment (if available)

5. **Interpretation**:
   - **Healthy Register**: Most experiments show `COHERENT` bands, `OK` transition status, few/no misaligned experiments
   - **Degraded Register**: Mixed bands, `ATTENTION` transition status, some misaligned experiments
   - **Critical Register**: Many `MISALIGNED` bands, `BLOCK` transition status, multiple misaligned experiments

**Example Audit Workflow:**

```
1. Load register from evidence pack:
   register = evidence["governance"]["structural_cohesion_register"]

2. Check misaligned experiments:
   if len(register["experiments_with_misaligned_structure"]) > 0:
       for cal_id in register["experiments_with_misaligned_structure"]:
           annex = load_annex(cal_id)
           # Investigate root cause
           if annex["lattice_band"] == "MISALIGNED":
               # Atlas convergence issue
           if annex["lean_shadow_status"] == "BLOCK":
               # Lean shadow structural integrity issue
           if annex["coherence_band"] == "MISALIGNED":
               # Topological coherence issue

3. Review band combinations:
   # Are there unexpected combinations?
   for combo, count in register["band_combinations"].items():
       if "MISALIGNED" in combo:
           # Flag for investigation

4. Assess overall health:
   coherent_ratio = (
       register["lattice_band_distribution"].get("COHERENT", 0) /
       register["total_experiments"]
   )
   if coherent_ratio < 0.8:
       # Flag: Less than 80% of experiments show coherent structure
```

### 10.7 SHADOW MODE Contract

**Critical Constraint**: The Structural Cohesion Register is **observational only**:

- The register does NOT influence control flow
- It does NOT gate calibration experiments
- It does NOT block system operations
- It is purely for audit and compliance reporting

All register functions are non-mutating and fail gracefully if annexes are missing or malformed.

### 10.8 GGFL Adapter

The Structural Cohesion Register provides a GGFL (Global Governance Fusion Layer) adapter for cross-subsystem alignment views:

**Function**: `structural_cohesion_register_for_alignment_view(signal_or_register)`

**Output Format**:
- `signal_type`: "SIG-STR" (constant identifier)
- `status`: "ok" | "warn" (warn if misaligned_count > 0)
- `conflict`: False (constant, structural cohesion is advisory only)
- `weight_hint`: "LOW" (constant, low-weight advisory signal)
- `drivers`: List[str] (max 1: DRIVER_MISALIGNED_EXPERIMENTS_PRESENT if misalignments exist)
- `summary`: str (single neutral sentence describing structural cohesion state)

**Usage**:
```python
from backend.health.atlas_governance_adapter import (
    structural_cohesion_register_for_alignment_view,
)

# From status signal
signal = status["signals"]["structural_cohesion_register"]
view = structural_cohesion_register_for_alignment_view(signal)

# Or from full register
register = build_structural_cohesion_register(annexes)
view = structural_cohesion_register_for_alignment_view(register)
```

**SHADOW MODE CONTRACT**: The GGFL adapter is advisory only. It does not enforce or gate any operations. The `SIG-STR` signal provides observational context for cross-subsystem alignment analysis but never triggers conflict or blocking behavior.

### 10.9 Status Integration

The Structural Cohesion Register is integrated into `first_light_status.json` with extraction provenance:

**Signal Location**: `signals["structural_cohesion_register"]`

**Signal Fields**:
- `total_experiments`: int
- `experiments_with_misaligned_structure_count`: int
- `top_misaligned`: List[str] (up to 5 cal_ids)
- `extraction_source`: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"

**Extraction Order**:
1. **MANIFEST**: Load from `manifest["governance"]["structural_cohesion_register"]` first
2. **EVIDENCE_JSON**: Fallback to `evidence["governance"]["structural_cohesion_register"]` if not in manifest
3. **MISSING**: If register not found in either location (safe, no error)

**Warning Generation**:
- Single advisory warning if `misaligned_count > 0`
- Warning lists at most 3 cal_ids (warning hygiene)
- Warning format: "Structural cohesion register: {count} experiment(s) with misaligned structure (out of {total} total). Top misalignments: {top_list}"

### 10.10 Implementation Location

**Functions**:
- `backend/health/atlas_governance_adapter.py`:
  - `emit_cal_exp_structural_cohesion_annex()`
  - `build_structural_cohesion_register()`
  - `attach_structural_cohesion_register_to_evidence()`
  - `extract_structural_cohesion_register_signal()`
  - `attach_structural_cohesion_register_signal_to_evidence()`
  - `structural_cohesion_register_for_alignment_view()` (GGFL adapter)

**Status Integration**:
- `scripts/generate_first_light_status.py`: Structural Cohesion Register signal extraction with provenance

**Tests**:
- `tests/health/test_structural_cohesion_register.py`
- `tests/health/test_structural_cohesion_register_ggfl.py`
- `tests/scripts/test_generate_first_light_status_structural_cohesion.py`

**Documentation**:
- This section (Section 10)

---

*Document Version: 1.2.0*
*Last Updated: 2025-12-12*
*Status: Design Freeze + P5 Cross-Reference + CAL-EXP Register
