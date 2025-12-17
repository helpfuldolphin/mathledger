# NCI Phase X Specification: Narrative Consistency Index Integration

---

> **PHASE X — NCI LAYER — FORMALIZATION DOCUMENT**
>
> This document defines the consistency laws and integration contracts
> for the Narrative Consistency Index (NCI) within Phase X architecture.
>
> **STATUS**: Specification Only (No Implementation Authorization)

---

**Document Version:** 1.2.0
**Status:** Specification
**Date:** 2025-12-11
**Author:** CLAUDE J — Narrative Consistency Layer

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [NCI → Telemetry Consistency Laws](#2-nci--telemetry-consistency-laws)
3. [NCI → Slice Identity Consistency Laws](#3-nci--slice-identity-consistency-laws)
4. [Integration Architecture](#4-integration-architecture)
5. [Data Contracts](#5-data-contracts)
6. [Governance Signal Specification](#6-governance-signal-specification)
7. [Director Panel Specification](#7-director-panel-specification)
8. [Schema Definitions](#8-schema-definitions)
9. [Invariants](#9-invariants)

---

## 1. Executive Summary

The Narrative Consistency Index (NCI) provides a quantitative measure of documentation coherence across the MathLedger system. This specification formalizes NCI's role within Phase X architecture, establishing:

1. **Telemetry Consistency Laws** — Rules ensuring NCI metrics align with telemetry event schemas
2. **Slice Identity Consistency Laws** — Rules ensuring documentation terminology matches slice definitions
3. **Governance Signal Contract** — Interface for NCI to contribute to global health decisions
4. **Director Panel Contract** — Interface for NCI status display in director dashboards

### 1.1 NCI Role in Phase X

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase X Health Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                  │
│   │  Telemetry  │   │   Slice     │   │   USLA      │                  │
│   │   Metrics   │   │  Identity   │   │   State     │                  │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                  │
│          │                 │                 │                          │
│          ▼                 ▼                 ▼                          │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │              Narrative Consistency Index (NCI)               │      │
│   │                                                              │      │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │      │
│   │  │ Terminology│  │   Phase    │  │  Uplift    │            │      │
│   │  │ Alignment  │  │ Discipline │  │ Avoidance  │            │      │
│   │  └────────────┘  └────────────┘  └────────────┘            │      │
│   │                                                              │      │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │      │
│   │  │ Structural │  │    SLO     │  │   Drift    │            │      │
│   │  │ Coherence  │  │ Evaluation │  │ Detection  │            │      │
│   │  └────────────┘  └────────────┘  └────────────┘            │      │
│   └────────────────────────────┬────────────────────────────────┘      │
│                                │                                        │
│                    ┌───────────┴───────────┐                           │
│                    ▼                       ▼                            │
│         ┌─────────────────┐     ┌─────────────────┐                    │
│         │  Director Panel │     │ Governance Sig  │                    │
│         │   (Dashboard)   │     │ (Health Fusion) │                    │
│         └─────────────────┘     └─────────────────┘                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. NCI → Telemetry Consistency Laws

These laws ensure that documentation terminology remains consistent with telemetry event schemas.

### 2.1 Law TCL-001: Event Name Alignment

**Statement:** Every telemetry event name referenced in documentation MUST match an event defined in the telemetry schema snapshot.

**Formal Definition:**
```
∀ doc ∈ Documentation:
  ∀ event_ref ∈ extract_event_references(doc):
    event_ref ∈ telemetry_schema.events.keys()
```

**Enforcement:**
- NCI scanner extracts telemetry event references from documentation
- Cross-references against `telemetry_schema_snapshot.schema.json`
- Violations reduce terminology alignment score

### 2.2 Law TCL-002: Field Name Consistency

**Statement:** Telemetry field names referenced in documentation MUST use canonical naming from schema definitions.

**Canonical Field Names:**
| Canonical | Variants (Violations) |
|-----------|----------------------|
| `H` | `h`, `health`, `H_t`, `Ht` |
| `rho` | `ρ`, `rsi`, `RSI`, `R_t` |
| `tau` | `τ`, `threshold`, `T_t` |
| `beta` | `β`, `block_rate`, `B_t` |
| `in_omega` | `in_Ω`, `omega_region`, `safe_region` |

**Enforcement:**
- Documentation scanner identifies field references
- Maps to canonical names via variant table
- Variant usage flagged but not blocking

### 2.3 Law TCL-003: Schema Version Reference

**Statement:** Documentation referencing telemetry schemas MUST cite a specific schema version.

**Valid Patterns:**
```markdown
<!-- VALID -->
Per telemetry_schema v1.0.0, the H field represents...
Schema: telemetry_schema_snapshot.schema.json (v1.0.0)

<!-- INVALID -->
The H field represents...  <!-- No version reference -->
```

**Enforcement:**
- Structural coherence score includes schema version checks
- Missing version references reduce structural score by 0.1

### 2.4 Law TCL-004: Telemetry Drift Synchronization

**Statement:** When telemetry schema changes, NCI MUST flag affected documentation for review within one commit cycle.

**Mechanism:**
1. Telemetry schema changes detected via git diff
2. NCI drift detector identifies documentation referencing changed fields
3. `silent_drift_files` list populated for unchanged-but-affected docs
4. Governance signal includes `telemetry_drift_detected: true`

---

## 3. NCI → Slice Identity Consistency Laws

These laws ensure documentation accurately describes slice configurations and behavior.

### 3.1 Law SIC-001: Slice Name Canonicalization

**Statement:** Slice names in documentation MUST match canonical definitions from slice registry.

**Canonical Slice Names:**
| Canonical | Variants (Violations) |
|-----------|----------------------|
| `arithmetic_simple` | `arithmetic-simple`, `ArithmeticSimple`, `simple_arithmetic` |
| `propositional_tautology` | `prop_taut`, `PropTautology`, `PL_tautology` |
| `group_theory` | `group-theory`, `GroupTheory`, `groups` |

**Enforcement:**
- NCI terminology scanner includes slice name patterns
- Violations flagged in `terminology.violations`
- Score impact: 0.6 weight (per CANONICAL_TERMS)

### 3.2 Law SIC-002: Slice Parameter Accuracy

**Statement:** Documentation describing slice parameters MUST accurately reflect current configuration.

**Required Parameters:**
```yaml
slice_definition:
  name: string              # Canonical slice name
  depth_max: integer        # Maximum derivation depth
  atom_max: integer         # Maximum atom count
  theory_id: integer        # Theory system ID
```

**Enforcement:**
- Parameter references extracted via regex
- Cross-referenced against `config/*.yaml` files
- Mismatches flagged in drift analysis

### 3.3 Law SIC-003: Slice Phase Mapping

**Statement:** Documentation MUST correctly map slices to their corresponding development phases.

**Phase Mappings:**
| Slice | Authorized Phase |
|-------|-----------------|
| `arithmetic_simple` | Phase II+ |
| `propositional_tautology` | Phase II+ |
| `group_theory` | Phase III+ |
| `linear_arithmetic` | Phase IV+ |

**Enforcement:**
- Phase discipline scorer includes slice-phase mappings
- Unauthorized phase references flagged
- Impact: Phase discipline ratio reduction

### 3.4 Law SIC-004: Slice Capability Claims

**Statement:** Documentation MUST NOT claim slice capabilities beyond authorized verification scope.

**Prohibited Patterns:**
```regex
# Violations
{slice_name}\s+(achieves|demonstrates|proves)\s+\w+
{slice_name}\s+success\s+rate\s+of\s+\d+%
{slice_name}\s+shows\s+\w+\s+uplift
```

**Allowed Patterns:**
```regex
# Acceptable
{slice_name}\s+experiment\s+\w+
{slice_name}\s+test\s+configuration
measure\s+{slice_name}\s+\w+
```

**Enforcement:**
- Uplift avoidance scorer extended for slice-specific claims
- Violations flagged with context
- Score impact: 0.30 weight

---

## 4. Integration Architecture

### 4.1 NCI Data Flow in Phase X

```
Documentation Files
        │
        ▼
┌──────────────────────┐
│ NarrativeConsistency │
│      Indexer         │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌────────┐
│ Per-Doc│   │ Global │
│ Metrics│   │  NCI   │
└───┬────┘   └───┬────┘
    │            │
    ▼            ▼
┌────────────────────────┐
│   NCI Area View        │
│   (terminology, phase, │
│    uplift, structure)  │
└──────────┬─────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────────┐ ┌────────────┐
│  SLO       │ │  Insight   │
│ Evaluation │ │  Summary   │
└─────┬──────┘ └─────┬──────┘
      │              │
      └──────┬───────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌────────────┐ ┌────────────┐
│ Director   │ │ Governance │
│  Panel     │ │   Signal   │
└────────────┘ └────────────┘
```

### 4.2 Component Responsibilities

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `NarrativeConsistencyIndexer` | Documentation files | `DocumentMetrics[]` | Per-document analysis |
| `build_nci_area_view` | `DocumentMetrics[]` | Area breakdown | Dimensional analysis |
| `evaluate_nci_slo` | Area view | SLO result | Threshold evaluation |
| `build_nci_insight_summary` | Snapshots | Summary | Dashboard data |
| `build_nci_director_panel` | Summary, SLO | Panel | Director display |
| `build_nci_governance_signal` | Panel, SLO | Signal | Health fusion |

---

## 5. Data Contracts

### 5.1 NCI Snapshot Contract

```python
@dataclass
class NCISnapshot:
    """Immutable snapshot of NCI state at a point in time."""

    timestamp: str                    # ISO 8601
    commit_hash: str                  # Git commit SHA
    global_nci: float                 # [0.0, 1.0]

    # Dimensional scores
    terminology_alignment: float      # [0.0, 1.0]
    phase_discipline: float           # [0.0, 1.0]
    uplift_avoidance: float           # [0.0, 1.0]
    structural_coherence: float       # [0.0, 1.0]

    # Category breakdown
    category_scores: Dict[str, float] # {category: nci_score}

    # Drift indicators
    drift_detected: bool
    silent_drift_files: List[str]

    # Telemetry consistency
    telemetry_drift_detected: bool
    telemetry_affected_docs: List[str]
```

### 5.2 SLO Thresholds Contract

```python
NCI_SLO_THRESHOLDS = {
    "global_nci_warn": 0.75,          # Global NCI below this triggers WARN
    "global_nci_breach": 0.60,        # Global NCI below this triggers BREACH
    "area_nci_warn": 0.70,            # Per-area NCI below this triggers WARN
    "structural_min": 0.60,           # Structural NCI minimum
    "terminology_min": 0.80,          # Terminology alignment minimum
    "violation_count_breach": 3,      # Violations above this count = BREACH
}
```

---

## 6. Governance Signal Specification

### 6.1 Signal Purpose

The NCI Governance Signal provides narrative health status to the global health fusion layer. It does NOT make governance decisions—it contributes information.

### 6.2 Signal Schema

See `nci_governance_signal.schema.json` for full specification.

**Key Fields:**
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-12-10T12:00:00Z",
  "source": "nci",

  "health_contribution": {
    "status": "OK" | "WARN" | "BREACH",
    "global_nci": 0.85,
    "confidence": 0.95
  },

  "telemetry_consistency": {
    "aligned": true,
    "drift_detected": false,
    "affected_docs_count": 0
  },

  "slice_consistency": {
    "aligned": true,
    "violation_count": 0
  },

  "dimensional_breakdown": {
    "terminology": 0.90,
    "phase": 0.88,
    "uplift": 1.0,
    "structure": 0.75
  },

  "recommendations": []
}
```

### 6.3 Signal Generation Function

```python
def build_nci_governance_signal(
    director_panel: dict,
    slo_result: dict,
    telemetry_drift: dict | None = None,
    slice_violations: list | None = None,
) -> dict:
    """
    Build governance signal for health fusion layer.

    IMPORTANT: This signal is INFORMATIONAL only.
    It does NOT make governance decisions.

    Args:
        director_panel: Output from build_nci_director_panel()
        slo_result: Output from evaluate_nci_slo()
        telemetry_drift: Optional telemetry drift report
        slice_violations: Optional slice consistency violations

    Returns:
        Governance signal dict conforming to schema
    """
```

---

## 7. Director Panel Specification

### 7.1 Panel Purpose

The Director Panel provides a high-level narrative consistency status for director dashboards. It uses neutral, non-judgmental language.

### 7.2 Panel Schema

See `nci_director_panel.schema.json` for full specification.

**Key Fields:**
```json
{
  "schema_version": "1.0.0",
  "timestamp": "2025-12-10T12:00:00Z",

  "status_light": "green" | "yellow" | "red",
  "global_nci": 0.85,
  "dominant_area": "terminology" | "phase" | "uplift" | "structure" | "none",

  "headline": "Narrative consistency within target. Global NCI: 0.85.",

  "metrics_summary": {
    "terminology_alignment": 0.90,
    "phase_discipline": 0.88,
    "uplift_avoidance": 1.0,
    "structural_coherence": 0.75
  },

  "slo_status": {
    "status": "OK",
    "violation_count": 0
  },

  "trend": "STABLE" | "IMPROVING" | "DEGRADING"
}
```

### 7.3 Status Light Mapping

| Global NCI | SLO Status | Status Light |
|------------|------------|--------------|
| ≥ 0.80 | OK | `green` |
| ≥ 0.70 | OK/WARN | `yellow` |
| < 0.70 | WARN/BREACH | `red` |
| Any | BREACH | `red` |

### 7.4 Headline Generation

Headlines use neutral, descriptive language:

| Condition | Headline Pattern |
|-----------|-----------------|
| SLO OK, NCI ≥ 0.80 | "Narrative consistency within target. Global NCI: {nci}." |
| SLO WARN | "Narrative consistency requires attention. Primary focus area: {area}." |
| SLO BREACH | "Narrative consistency SLO breach detected. {count} area(s) require attention." |

---

## 8. Schema Definitions

### 8.1 Schema File Locations

```
docs/system_law/schemas/nci/
├── nci_director_panel.schema.json
└── nci_governance_signal.schema.json
```

### 8.2 Schema Cross-References

| Schema | References |
|--------|------------|
| `nci_director_panel` | `nci_governance_signal`, `telemetry_schema_snapshot` |
| `nci_governance_signal` | `nci_director_panel`, `first_light_stability_report` |

---

## 9. Invariants

### 9.1 NCI Layer Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| NCI-INV-001 | Global NCI ∈ [0.0, 1.0] | Bounded computation |
| NCI-INV-002 | Dimensional scores ∈ [0.0, 1.0] | Per-dimension bounds |
| NCI-INV-003 | Status light derived from NCI + SLO | Deterministic mapping |
| NCI-INV-004 | Headlines use neutral language | Pattern validation |
| NCI-INV-005 | No governance decisions from NCI | Signal is informational only |
| NCI-INV-006 | Telemetry drift detected within 1 commit | Drift scanner in CI |
| NCI-INV-007 | Slice names use canonical forms | Terminology alignment |

### 9.2 Integration Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| INT-INV-001 | Director panel generated before governance signal | Dependency order |
| INT-INV-002 | SLO evaluated before panel generation | Dependency order |
| INT-INV-003 | Schemas versioned for forward compatibility | `schema_version` field |

---

## 10. NCI and Real Telemetry

This section clarifies how NCI consistency checks behave when the system transitions from synthetic (P3) or mock (P4) telemetry to real runner telemetry (P5+).

### 10.1 Telemetry Source Context

The NCI operates in the same manner regardless of telemetry source. However, the *meaning* of its signals shifts based on context:

| Telemetry Source | NCI Role | Interpretation |
|------------------|----------|----------------|
| **Synthetic (P3)** | Validate documentation against specification | Violations indicate spec/doc mismatch |
| **Mock (P4)** | Validate documentation against mock schema | Violations indicate mock/doc mismatch |
| **Real (P5+)** | Validate documentation against live telemetry schema | Violations indicate production drift |

### 10.2 TCL Behavior Under Real Telemetry

When `telemetry_source == "real"`:

**TCL-001 (Event Name Alignment):**
- References to telemetry events in documentation are validated against the *actual* event stream schema
- Stale documentation referencing deprecated events will trigger violations
- New events not yet documented will appear as "undocumented event" observations (advisory, not violations)

**TCL-002 (Field Name Consistency):**
- Canonical field names (H, rho, tau, beta, in_omega) remain invariant
- Documentation using variant names will be flagged regardless of telemetry source
- Real telemetry does not change what is "canonical" — the spec defines canonical names

**TCL-003 (Schema Version Reference):**
- Documentation must reference the schema version matching the live telemetry source
- Version drift (doc references v1.0, telemetry is v1.1) triggers a WARNING
- This becomes more critical under real telemetry as schema evolution occurs in production

**TCL-004 (Telemetry Drift Synchronization):**
- Under real telemetry, drift detection is continuous rather than commit-triggered
- Live schema changes from the runner are detected and flagged within the configured staleness window
- Documentation updates may lag real telemetry changes; NCI flags this lag

### 10.3 SIC Behavior Under Real Telemetry

When `telemetry_source == "real"`:

**SIC-001 through SIC-004:**
- Slice identity consistency laws operate identically under real telemetry
- The slice registry is the source of truth, not the telemetry source
- Real telemetry may reveal slice configurations not yet documented (advisory observation)

### 10.4 Confidence Adjustments

The NCI governance signal includes a `confidence` field. Under real telemetry, confidence may be adjusted:

| Condition | Confidence Adjustment |
|-----------|----------------------|
| Schema version matches documentation | +0.10 |
| Telemetry schema recently changed (< 24h) | -0.15 (doc may lag) |
| All TCL checks pass | +0.05 |
| Real telemetry provides richer field set | +0.05 (more validation surface) |

### 10.5 Operational Implications

**During P5 transition:**
- NCI violations detected under real telemetry should be prioritized for documentation updates
- TCL-004 drift detection becomes operationally significant
- Consider establishing a documentation SLO for responding to telemetry schema changes

**Steady-state real telemetry:**
- NCI operates as a continuous documentation health monitor
- Integrate NCI into CI/CD pipeline for documentation PRs
- Use NCI governance signal as input to release readiness checks

---

## 11. NCI P5 Operational Modes

This section defines the operational modes for NCI under P5 real telemetry, including check behaviors, confidence computation, and warning semantics.

### 11.1 Mode Definitions

NCI operates in one of three modes, determined by available data sources:

| Mode | Telemetry Source | Slice Registry | Description |
|------|------------------|----------------|-------------|
| **DOC_ONLY** | None | None | Documentation-only analysis; no external validation |
| **TELEMETRY_CHECKED** | Available | None | Documentation validated against live telemetry schema |
| **FULLY_BOUND** | Available | Available | Full validation against telemetry AND slice configs |

**Mode Selection Logic:**
```
IF telemetry_schema IS NULL AND slice_registry IS NULL:
    mode = DOC_ONLY
ELIF telemetry_schema IS NOT NULL AND slice_registry IS NULL:
    mode = TELEMETRY_CHECKED
ELIF telemetry_schema IS NOT NULL AND slice_registry IS NOT NULL:
    mode = FULLY_BOUND
ELSE:
    mode = DOC_ONLY  # slice_registry without telemetry is invalid
```

### 11.2 DOC_ONLY Mode

**Purpose:** Baseline documentation health assessment without external validation.

**TCL Checks:**
| Law | Behavior |
|-----|----------|
| TCL-001 | SKIPPED (no schema to validate against) |
| TCL-002 | ACTIVE (canonical field names are spec-defined) |
| TCL-003 | SKIPPED (no live schema version to compare) |
| TCL-004 | SKIPPED (no telemetry source for drift detection) |

**SIC Checks:**
| Law | Behavior |
|-----|----------|
| SIC-001 | ACTIVE (canonical slice names are spec-defined) |
| SIC-002 | SKIPPED (no registry to validate parameters) |
| SIC-003 | SKIPPED (no registry for phase mapping) |
| SIC-004 | ACTIVE (capability claim patterns are spec-defined) |

**Confidence Computation:**
```python
def compute_confidence_doc_only(panel: dict) -> float:
    base = 0.70  # Lower base due to limited validation

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    coverage = len([v for v in metrics.values() if v is not None]) / 4
    base += 0.10 * coverage

    # Penalty for missing external validation
    base -= 0.10  # No telemetry validation
    base -= 0.05  # No slice registry validation

    return max(0.50, min(1.0, base))  # Floor: 0.50, Ceiling: 1.0
```

**Warning Format:**
```json
{
  "mode": "DOC_ONLY",
  "warning_type": "TCL-002",
  "severity": "medium",
  "message": "Non-canonical field name 'Ht' found in docs/api.md:42",
  "remediation": "Replace 'Ht' with canonical 'H'",
  "validation_context": "documentation_only"
}
```

### 11.3 TELEMETRY_CHECKED Mode

**Purpose:** Documentation validated against live telemetry schema.

**TCL Checks:**
| Law | Behavior |
|-----|----------|
| TCL-001 | ACTIVE (validate event names against live schema) |
| TCL-002 | ACTIVE (canonical field names) |
| TCL-003 | ACTIVE (validate schema version references) |
| TCL-004 | ACTIVE (detect drift from live schema) |

**SIC Checks:**
| Law | Behavior |
|-----|----------|
| SIC-001 | ACTIVE (canonical slice names) |
| SIC-002 | SKIPPED (no registry) |
| SIC-003 | SKIPPED (no registry) |
| SIC-004 | ACTIVE (capability claim patterns) |

**Confidence Computation:**
```python
def compute_confidence_telemetry_checked(
    panel: dict,
    tcl_result: dict,
) -> float:
    base = 0.80  # Higher base with telemetry validation

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    coverage = len([v for v in metrics.values() if v is not None]) / 4
    base += 0.05 * coverage

    # TCL alignment bonus
    if tcl_result.get("aligned", False):
        base += 0.10
    else:
        # Penalty proportional to violation count
        violations = len(tcl_result.get("violations", []))
        base -= min(0.15, violations * 0.03)

    # Schema freshness
    schema_age_hours = tcl_result.get("schema_age_hours", 0)
    if schema_age_hours < 24:
        base -= 0.05  # Recent schema change, docs may lag

    # Penalty for missing slice validation
    base -= 0.05

    return max(0.50, min(1.0, base))
```

**Warning Format:**
```json
{
  "mode": "TELEMETRY_CHECKED",
  "warning_type": "TCL-004",
  "severity": "high",
  "message": "Telemetry schema drift detected: field 'convergence_class' added in v1.2.0",
  "remediation": "Update docs/telemetry.md to document 'convergence_class' field",
  "validation_context": "live_telemetry",
  "schema_version": "1.2.0",
  "doc_references_version": "1.1.0"
}
```

### 11.4 FULLY_BOUND Mode

**Purpose:** Complete validation against telemetry AND slice configurations.

**TCL Checks:**
| Law | Behavior |
|-----|----------|
| TCL-001 | ACTIVE |
| TCL-002 | ACTIVE |
| TCL-003 | ACTIVE |
| TCL-004 | ACTIVE |

**SIC Checks:**
| Law | Behavior |
|-----|----------|
| SIC-001 | ACTIVE |
| SIC-002 | ACTIVE (validate parameters against registry) |
| SIC-003 | ACTIVE (validate slice-phase mapping) |
| SIC-004 | ACTIVE |

**Confidence Computation:**
```python
def compute_confidence_fully_bound(
    panel: dict,
    tcl_result: dict,
    sic_result: dict,
) -> float:
    base = 0.85  # Highest base with full validation

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    coverage = len([v for v in metrics.values() if v is not None]) / 4
    base += 0.05 * coverage

    # TCL alignment
    if tcl_result.get("aligned", False):
        base += 0.05
    else:
        violations = len(tcl_result.get("violations", []))
        base -= min(0.10, violations * 0.02)

    # SIC alignment
    if sic_result.get("aligned", False):
        base += 0.05
    else:
        violations = len(sic_result.get("violations", []))
        base -= min(0.10, violations * 0.02)

    # Schema freshness penalty
    schema_age_hours = tcl_result.get("schema_age_hours", 0)
    if schema_age_hours < 24:
        base -= 0.05

    # Registry freshness bonus (if registry recently validated)
    registry_validated = sic_result.get("registry_validated", False)
    if registry_validated:
        base += 0.03

    return max(0.50, min(1.0, base))
```

**Warning Format:**
```json
{
  "mode": "FULLY_BOUND",
  "warning_type": "SIC-002",
  "severity": "high",
  "message": "Slice 'arithmetic_simple' parameter mismatch: doc says depth_max=5, registry shows depth_max=4",
  "remediation": "Update docs/slices/arithmetic_simple.md with correct depth_max",
  "validation_context": "live_telemetry_and_registry",
  "slice_name": "arithmetic_simple",
  "expected": {"depth_max": 4},
  "documented": {"depth_max": 5}
}
```

### 11.5 Mode Thresholds

**SLO Thresholds by Mode:**

| Threshold | DOC_ONLY | TELEMETRY_CHECKED | FULLY_BOUND |
|-----------|----------|-------------------|-------------|
| `global_nci_warn` | 0.70 | 0.75 | 0.80 |
| `global_nci_breach` | 0.55 | 0.60 | 0.65 |
| `area_nci_warn` | 0.65 | 0.70 | 0.75 |
| `terminology_min` | 0.75 | 0.80 | 0.85 |
| `violation_count_breach` | 5 | 3 | 2 |

**Rationale:** Higher validation coverage enables stricter thresholds. FULLY_BOUND mode has the most validation surface, so deviations are more significant.

### 11.6 P5 Evaluation Function (SPEC-ONLY)

```python
# SPEC-ONLY: This function signature is specified but not implemented
# Implementation requires P5 activation authorization

def evaluate_nci_p5(
    panel: dict[str, Any],
    telemetry: dict[str, Any] | None,
    slice_configs: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evaluate NCI under P5 operational mode.

    SPEC-ONLY: Not implemented. Requires P5 activation.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Output does NOT influence governance decisions
    - All warnings are advisory only

    Args:
        panel: NCI director panel (from build_nci_director_panel)
        telemetry: Live telemetry schema, or None for DOC_ONLY mode
            {
                "schema_version": "1.2.0",
                "events": {...},
                "fields": {...},
                "schema_age_hours": 12,
            }
        slice_configs: Slice registry, or None for DOC_ONLY/TELEMETRY_CHECKED
            {
                "slices": {
                    "arithmetic_simple": {"depth_max": 4, "atom_max": 4, ...},
                    ...
                },
                "registry_validated": True,
            }

    Returns:
        P5 evaluation result:
        {
            "schema_version": "1.0.0",
            "mode": "DOC_ONLY" | "TELEMETRY_CHECKED" | "FULLY_BOUND",
            "global_nci": float,
            "confidence": float,
            "tcl_result": {
                "aligned": bool,
                "checks_run": ["TCL-001", "TCL-002", ...],
                "checks_skipped": [...],
                "violations": [...],
            },
            "sic_result": {
                "aligned": bool,
                "checks_run": ["SIC-001", ...],
                "checks_skipped": [...],
                "violations": [...],
            },
            "warnings": [
                {"mode": "...", "warning_type": "...", "severity": "...", ...},
            ],
            "slo_evaluation": {
                "thresholds_used": {...},
                "status": "OK" | "WARN" | "BREACH",
            },
            "operational_notes": [
                "Telemetry schema v1.2.0 validated",
                "Slice registry contains 3 active slices",
            ],
        }

    Raises:
        NotImplementedError: Always (SPEC-ONLY, not implemented)
    """
    raise NotImplementedError(
        "evaluate_nci_p5 is SPEC-ONLY. "
        "Implementation requires P5 activation authorization."
    )
```

### 11.7 Smoke-Test Readiness Checklist

Before P5 NCI activation, verify the following:

**Infrastructure Readiness:**

| Check | Command | Expected |
|-------|---------|----------|
| NCI adapter imports | `python -c "from backend.health import build_nci_director_panel"` | No error |
| TCL checker works | `python -c "from backend.health import check_telemetry_consistency"` | No error |
| SIC checker works | `python -c "from backend.health import check_slice_consistency"` | No error |
| Schema files exist | `ls docs/system_law/schemas/nci/*.json` | 2 files |

**Functional Readiness:**

| Check | Verification | Pass Criteria |
|-------|--------------|---------------|
| DOC_ONLY mode produces valid panel | Run `build_nci_director_panel` with mock data | `status_light` in {green, yellow, red} |
| TCL-002 detects variants | Feed doc with "Ht" to `check_telemetry_consistency` | Returns `aligned=False` |
| SIC-001 detects variants | Feed doc with "ArithmeticSimple" to `check_slice_consistency` | Returns violations list |
| JSON serialization | `json.dumps(panel)` | No TypeError |
| Determinism | Run twice with same input | Identical output (except timestamp) |

**Integration Readiness:**

| Check | Verification | Pass Criteria |
|-------|--------------|---------------|
| Global health attachment | `attach_nci_tile_to_global_health(gh, tile)` | Returns dict with `nci` key |
| Evidence attachment | `attach_nci_to_evidence(ev, signal)` | Returns dict with `governance.nci` |
| P3 summary extraction | `build_nci_summary_for_p3(panel)` | Contains `global_nci_score` |

**Documentation Readiness:**

| Check | File | Pass Criteria |
|-------|------|---------------|
| NCI spec complete | `docs/system_law/NCI_PhaseX_Spec.md` | Sections 1-11 present |
| GGFL mapping documented | `docs/system_law/Global_Governance_Fusion_PhaseX.md` | Section 10 present |
| FAQ entry exists | `docs/system_law/First_Light_External_FAQ.md` | NCI section present |

**P5 Activation Gates:**

| Gate | Condition | Status |
|------|-----------|--------|
| P5-NCI-001 | All smoke tests pass | Pending |
| P5-NCI-002 | Real telemetry schema available | Pending |
| P5-NCI-003 | Slice registry populated | Pending |
| P5-NCI-004 | GGFL SIG-NAR adapter ready | Pending |
| P5-NCI-005 | Explicit P5 activation authorization | Pending |

---

## Appendix A: Function Signatures

### A.1 Core NCI Functions

```python
def build_nci_area_view(
    index: NarrativeIndex,
    max_files: int = 10,
) -> dict[str, Any]

def evaluate_nci_slo(
    area_view: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]

def build_nci_insight_summary(
    snapshots: Sequence[dict[str, Any]],
) -> dict[str, Any]

def build_nci_director_panel(
    insight_summary: dict[str, Any],
    priority_view: dict[str, Any],
    slo_result: dict[str, Any],
) -> dict[str, Any]
```

### A.2 New Functions (Specified Here)

```python
def build_nci_governance_signal(
    director_panel: dict[str, Any],
    slo_result: dict[str, Any],
    telemetry_drift: dict[str, Any] | None = None,
    slice_violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]

def check_telemetry_consistency(
    docs: list[Path],
    telemetry_schema: dict[str, Any],
) -> dict[str, Any]

def check_slice_consistency(
    docs: list[Path],
    slice_registry: dict[str, Any],
) -> list[dict[str, Any]]
```

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-10 | Initial NCI Phase X specification |
| 1.1.0 | 2025-12-11 | Added Section 10: NCI and Real Telemetry |
| 1.2.0 | 2025-12-11 | Added Section 11: NCI P5 Operational Modes (DOC_ONLY, TELEMETRY_CHECKED, FULLY_BOUND) |

---

*Document Version: 1.2.0*
*Last Updated: 2025-12-11*
*Status: Specification Only*
