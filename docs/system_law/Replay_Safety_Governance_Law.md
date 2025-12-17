# Replay Safety Governance Law

---

> **SYSTEM LAW DOCUMENT — REPLAY SAFETY GOVERNANCE**
>
> This document defines the formal semantics, mapping rules, and integration
> points for the Replay Safety Governance Signal subsystem.
>
> **Status**: Reference
> **Version**: 1.0.0
> **Date**: 2025-12-10
> **Author**: CLAUDE C (Consolidation Layer)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Safety x Radar Alignment Semantics](#2-safety-x-radar-alignment-semantics)
3. [BLOCK/WARN/OK Mapping Rules](#3-blockwarnok-mapping-rules)
4. [P4 Divergence Interpretation](#4-p4-divergence-interpretation)
   - 4.5 [Divergence Companion — P5 Co-Interpretation Guide](#45-divergence-companion--p5-co-interpretation-guide)
5. [Evidence Pack Integration](#5-evidence-pack-integration)
6. [Integration Points](#6-integration-points)
7. [Schema Reference](#7-schema-reference)

---

## 1. Executive Summary

The Replay Safety Governance Signal provides a unified signal that fuses two independent evaluation streams:

1. **Safety Evaluation** — Determinism verification from replay safety checks
2. **Radar Evaluation** — Governance drift detection from the governance radar

This fusion produces a single, normalized governance signal suitable for:
- Promotion/rejection decisions
- Evidence pack attestation
- Director panel visualization
- Phase X divergence correlation

### Key Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| RS-01 | **Consensus Required** | Both Safety and Radar must agree for OK status |
| RS-02 | **Divergence Escalation** | DIVERGENT alignment always produces BLOCK |
| RS-03 | **Conservative Stance** | Any BLOCK from either side propagates to final signal |
| RS-04 | **Reason Traceability** | All reasons prefixed with source tag ([Safety], [Radar], [CONFLICT]) |

---

## 2. Safety x Radar Alignment Semantics

The Replay Safety system evaluates two orthogonal dimensions:

### 2.1 Safety Dimension

Evaluates deterministic replay fidelity:
- Hash consistency between runs
- Trace data integrity
- Policy update safety
- Promotion eligibility

**Safety Outputs:**
- `status`: OK | WARN | BLOCK
- `safe_for_policy_update`: boolean
- `safe_for_promotion`: boolean
- `reasons`: list of diagnostic strings

### 2.2 Radar Dimension

Evaluates governance posture alignment:
- Drift detection from baseline
- Tier skew analysis
- External governance signal correlation
- Long-horizon stability trends

**Radar Outputs:**
- `status`: OK | WARN | BLOCK
- `governance_alignment`: ALIGNED | TENSION | DIVERGENT
- `conflict`: boolean
- `reasons`: list of diagnostic strings

### 2.3 Alignment States

The combination of Safety and Radar produces three alignment states:

| Alignment | Condition | Meaning |
|-----------|-----------|---------|
| **ALIGNED** | Safety and Radar agree on status | Consensus on system state |
| **TENSION** | Minor disagreement (e.g., OK vs WARN) | Review recommended but not blocking |
| **DIVERGENT** | Major disagreement (e.g., OK vs BLOCK) | Conflict requires manual review |

```
Safety Status     Radar Status      Alignment
─────────────     ────────────      ─────────
OK                OK                ALIGNED
OK                WARN              TENSION
WARN              WARN              ALIGNED
OK                BLOCK             DIVERGENT
BLOCK             OK                DIVERGENT
BLOCK             BLOCK             ALIGNED
WARN              BLOCK             TENSION*
BLOCK             WARN              TENSION*

* TENSION when one is BLOCK and other is WARN, but overall signal is BLOCK
```

---

## 3. BLOCK/WARN/OK Mapping Rules

The `to_governance_signal_for_replay_safety()` function implements the following mapping rules:

### 3.1 BLOCK Conditions

The final signal is **BLOCK** if ANY of these conditions hold:

| Condition | Rationale |
|-----------|-----------|
| `safety_status == BLOCK` | Safety violation detected |
| `radar_status == BLOCK` | Governance drift critical |
| `alignment == DIVERGENT` | Irreconcilable disagreement |
| `conflict == True` | Explicit conflict flag set |

**Code Reference:**
```python
if safety_status == PromotionStatus.BLOCK or radar_status == PromotionStatus.BLOCK:
    status = PromotionStatus.BLOCK
elif safety_eval.get("alignment") == GovernanceAlignment.DIVERGENT.value:
    status = PromotionStatus.BLOCK
elif radar_view.get("conflict") == True:
    status = PromotionStatus.BLOCK
```

### 3.2 WARN Conditions

The final signal is **WARN** if (not BLOCK) AND ANY of these conditions hold:

| Condition | Rationale |
|-----------|-----------|
| `safety_status == WARN` | Safety concern detected |
| `radar_status == WARN` | Governance drift warning |
| `alignment == TENSION` | Minor disagreement |

### 3.3 OK Conditions

The final signal is **OK** only if ALL of these conditions hold:

| Condition | Rationale |
|-----------|-----------|
| `safety_status == OK` | No safety violations |
| `radar_status == OK` | No governance drift |
| `alignment == ALIGNED` | Consensus achieved |
| `conflict == False` | No conflict flag |

### 3.4 Truth Table

```
Safety  Radar   Alignment   Conflict    Final Signal
──────  ─────   ─────────   ────────    ────────────
OK      OK      ALIGNED     False       OK
OK      OK      ALIGNED     True        BLOCK
OK      WARN    TENSION     False       WARN
OK      WARN    TENSION     True        BLOCK
OK      BLOCK   DIVERGENT   True        BLOCK
WARN    OK      TENSION     False       WARN
WARN    WARN    ALIGNED     False       WARN
WARN    BLOCK   TENSION     False       BLOCK
BLOCK   OK      DIVERGENT   True        BLOCK
BLOCK   WARN    TENSION     False       BLOCK
BLOCK   BLOCK   ALIGNED     False       BLOCK
```

---

## 4. P4 Divergence Interpretation

### 4.1 Replay Safety's Role in P4

In Phase X P4, Replay Safety provides an independent verification channel for detecting divergence between real runner execution and shadow twin predictions.

| P4 Component | Replay Safety Role |
|--------------|-------------------|
| Real Cycle Observer | Validates trace hash consistency |
| Twin Runner | Provides prediction baseline for replay |
| Divergence Analyzer | Correlates with replay safety violations |

### 4.2 Divergence Severity Mapping

Replay Safety contributes to P4 divergence classification:

| Replay Safety Signal | P4 Divergence Severity |
|---------------------|----------------------|
| OK | NONE or MINOR |
| WARN | MODERATE |
| BLOCK | SEVERE |

### 4.3 Integration with Twin Trajectory

When Replay Safety detects a violation:

1. **Hash Mismatch** — Indicates non-deterministic replay
   - Maps to `success_diverged` in P4 DivergenceSnapshot
   - Severity: SEVERE

2. **Config Drift** — Indicates configuration inconsistency
   - Maps to `state_diverged` in P4 DivergenceSnapshot
   - Severity: MODERATE

3. **Alignment Conflict** — Safety and Radar disagree
   - Maps to `governance_diverged` (custom P4 extension)
   - Severity: MODERATE to SEVERE based on gap

### 4.4 Evidence Chain

```
Replay Safety Violation
         │
         ▼
┌─────────────────────────┐
│ GovernanceSignal        │
│ - status: BLOCK         │
│ - reasons: [...]        │
│ - governance_status     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ P4 DivergenceSnapshot   │
│ - divergence_severity   │
│ - replay_safety_flag    │
│ - action: LOGGED_ONLY   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Evidence Pack           │
│ - governance.replay     │
│ - governance_status     │
└─────────────────────────┘
```

### 4.5 Divergence Companion — P5 Co-Interpretation Guide

This subsection provides guidance on co-interpreting Replay Safety signals with the P5 divergence taxonomy. When analyzing system behavior, replay safety status should be read alongside topology mode, budget stability, and divergence severity to correctly attribute root causes.

#### 4.5.1 P5 Divergence Taxonomy Reference

The P5 real-telemetry regime uses the following divergence classifications:

| Category | Classifications | Source |
|----------|-----------------|--------|
| **Divergence Severity** | NONE, INFO, WARN, CRITICAL | Phase_X_Divergence_Metric.md |
| **Divergence Type** | STATE, OUTCOME, COMBINED | divergence_analyzer.py |
| **Topology Mode** | STABLE, DRIFT, TURBULENT, CRITICAL | Topology_Bundle_PhaseX_Requirements.md |
| **Budget Stability** | STABLE, DRIFTING, VOLATILE | Budget_PhaseX_Doctrine.md |

These categories combine to form a **divergence context** that modulates how replay safety signals should be interpreted.

#### 4.5.2 Replay Safety × Topology Mode Matrix

| Replay Safety | Topology Mode | Interpretation | Root Cause Hypothesis |
|---------------|---------------|----------------|----------------------|
| **OK** | STABLE | Nominal operation | — |
| **OK** | DRIFT | Topology shifting but replay deterministic | Topology/model drift, not replay bug |
| **OK** | TURBULENT | Major topology change, replay still valid | Structural regime shift |
| **OK** | CRITICAL | Topology violated but replay consistent | Topology invariant failure (not replay) |
| **WARN** | STABLE | Replay concern in stable conditions | Potential replay non-determinism |
| **WARN** | DRIFT | Both systems showing strain | Correlated instability—investigate jointly |
| **BLOCK** | STABLE | Replay failure in stable topology | **Replay bug** — prioritize investigation |
| **BLOCK** | CRITICAL | Both systems failing | Cascading failure—check root cause chain |

#### 4.5.3 Worked Examples

**Example 1: Replay OK, Topology TURBULENT (STRUCTURAL_BREAK scenario)**

```
Replay Safety Signal:
  status: OK
  governance_alignment: ALIGNED
  conflict: false
  reasons: ["[Safety] All checks passed"]

Topology Bundle Signal:
  mode: TURBULENT
  persistence_drift: 0.18
  betti_status: β₀=1, β₁=2

P4 Divergence Log:
  divergence_type: STATE
  divergence_pct: 0.12
  severity: WARN
```

**Interpretation**: The replay verification passed (hashes match, determinism confirmed), but the topology is experiencing significant structural change. The P4 divergence is a STATE divergence (H/ρ mismatch, not outcome mismatch), which correlates with topology turbulence.

**Diagnosis**: This is a **model/topology issue**, not a replay bug. The twin model's assumptions about the H-ρ manifold may be outdated. Replay safety being OK confirms the real runner is behaving deterministically—the divergence stems from model miscalibration or genuine regime shift (STRUCTURAL_BREAK).

**Action**: Log for topology recalibration analysis. No replay investigation needed.

---

**Example 2: Replay BLOCK, Topology STABLE, Budget STABLE (replay determinism failure)**

```
Replay Safety Signal:
  status: BLOCK
  governance_alignment: ALIGNED
  conflict: false
  reasons: ["[Safety] Hash mismatch detected"]

Topology Bundle Signal:
  mode: STABLE
  persistence_drift: 0.02
  betti_status: β₀=1, β₁=0

Budget Signal:
  stability_class: STABLE
  health_score: 92
```

**Interpretation**: Replay verification failed with a hash mismatch, yet topology and budget are both nominal. This indicates the real runner produced different outputs on replay despite identical inputs.

**Diagnosis**: This is a **replay non-determinism bug**. With topology and budget stable, there's no environmental confound. The hash mismatch points to:
- Non-deterministic code path
- Floating-point instability
- External dependency variation (time, random, network)

**Action**: Prioritize replay investigation. Check for non-deterministic operations in the execution path.

---

**Example 3: Replay WARN, Topology DRIFT, Budget DRIFTING (PHASE_LAG scenario)**

```
Replay Safety Signal:
  status: WARN
  governance_alignment: TENSION
  conflict: false
  reasons: ["[Safety] Minor hash variance", "[Radar] Approaching drift threshold"]

Topology Bundle Signal:
  mode: DRIFT
  persistence_drift: 0.08

Budget Signal:
  stability_class: DRIFTING
  health_score: 74
  drift_trajectory: [0.02, 0.04, 0.06, 0.08]
```

**Interpretation**: Multiple systems are showing simultaneous strain. Replay has minor variance, topology is drifting, and budget is unstable. The TENSION alignment indicates Safety and Radar see different severity.

**Diagnosis**: This is a **correlated multi-factor instability** (PHASE_LAG). The budget drift may be causing increased compute variance, which affects both replay determinism and topology calculations. Root cause attribution is ambiguous—the systems are out of phase but not structurally broken.

**Action**: Apply P4 budget severity multiplier (0.7 for DRIFTING). Log as `budget_confounded: true`. Monitor for escalation but don't attribute to single subsystem.

#### 4.5.4 Root Cause Attribution Priority

When multiple signals are non-nominal, use this attribution priority:

```
1. Identity Signal (cryptographic)     → Security-critical, highest priority
2. Structure Signal (DAG coherence)    → Data integrity issue
3. Replay Safety Signal                → Determinism verification
4. Topology Signal                     → Model/manifold health
5. Budget Signal                       → Resource constraints
6. Metrics Signal                      → Performance indicators
```

**Attribution Rule**: If a higher-priority signal is BLOCK/CRITICAL, attribute root cause to that subsystem first. Lower-priority signals in degraded state may be **consequential** rather than **causal**.

#### 4.5.5 Divergence Type Correlation

| P5 Divergence Type | Replay Safety Implication | Co-Interpretation |
|--------------------|--------------------------|-------------------|
| **STATE** (H/ρ mismatch) | Replay OK likely | Model calibration issue, not replay |
| **OUTCOME** (success mismatch) | Replay WARN/BLOCK possible | May indicate replay path divergence |
| **COMBINED** (both) | Replay status varies | Full investigation needed |

**Key Insight**: STATE divergence with Replay OK strongly suggests the issue is in the **model layer** (topology, twin calibration) rather than the **execution layer** (replay determinism).

---

## 5. Evidence Pack Integration

### 5.1 Evidence Pack Structure

When Replay Safety contributes to an evidence pack:

```json
{
  "evidence_type": "replay_safety",
  "timestamp": "2025-12-10T00:00:00Z",
  "governance": {
    "replay_safety": {
      "governance_status": "OK|WARN|BLOCK",
      "governance_alignment": "ALIGNED|TENSION|DIVERGENT",
      "safety_status": "OK|WARN|BLOCK",
      "radar_status": "OK|WARN|BLOCK",
      "conflict": false,
      "reasons": [
        "[Safety] All checks passed",
        "[Radar] No drift detected"
      ]
    }
  },
  "replay_safety_ok": true,
  "confidence_score": 0.95
}
```

### 5.2 Harmonization Rules

When `summarize_replay_safety_for_evidence()` is called:

1. If `governance_signal` provided:
   - Extract `governance_status` from signal
   - Include in evidence pack top-level

2. If `governance_view` provided:
   - Extract `governance_alignment`
   - Include in evidence pack

3. Backward compatibility:
   - If neither provided, omit governance fields
   - Other fields (`replay_safety_ok`, `confidence_score`) still populated

---

## 6. Integration Points

### 6.1 Global Alignment View

<!-- TODO: CLAUDE I — Connect replay_safety governance signal to global alignment view -->
<!-- Integration point: experiments/u2/replay_safety.py::to_governance_signal_for_replay_safety -->
<!-- Target: backend/analytics/global_alignment_view.py (to be created) -->

The Replay Safety governance signal should be consumed by the global alignment view to provide:
- Unified governance dashboard
- Cross-subsystem alignment visualization
- Trend analysis across replay safety, PRNG governance, and other signals

**Planned Integration:**
```python
# backend/analytics/global_alignment_view.py
def build_global_alignment_view(
    replay_safety_signal: Dict[str, Any],
    prng_governance_tile: Dict[str, Any],
    # ... other governance signals
) -> Dict[str, Any]:
    """Build unified global alignment view."""
    # TODO: Implement global alignment synthesis
    pass
```

### 6.2 Phase X Director Panel

<!-- TODO: CLAUDE I — Connect replay_safety governance signal to Phase X director panel -->
<!-- Integration point: experiments/u2/replay_safety.py::build_replay_safety_director_panel -->
<!-- Target: backend/topology/director_panel.py (Phase X P4/P5) -->

The Replay Safety director panel provides:
- Real-time conflict visualization
- Manual review queue for DIVERGENT cases
- Historical trend display

**Planned Integration:**
```python
# backend/topology/director_panel.py
def add_replay_safety_panel(
    director_state: Dict[str, Any],
    replay_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """Add replay safety section to director panel."""
    # TODO: Integrate with Phase X director panel architecture
    pass
```

### 6.3 USLA Integration Bridge

The Replay Safety signal should connect to USLA governance layer:

**Planned Integration:**
```python
# backend/usla/governance_bridge.py
def bridge_replay_safety_to_usla(
    replay_signal: Dict[str, Any],
    usla_state: "USLAIntegration",
) -> None:
    """
    Bridge replay safety signal to USLA governance.

    SHADOW MODE: Observation only, no modification to USLA state.
    """
    # TODO: Implement USLA bridge (Phase X P5+)
    pass
```

---

## 7. Schema Reference

The formal JSON Schema for the Replay Safety Governance Signal is defined in:

```
docs/system_law/schemas/replay_safety_governance_signal.schema.json
```

### 7.1 Schema Summary

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | Schema version (1.0.0) |
| `signal_type` | string | Yes | Always "replay_safety" |
| `status` | enum | Yes | OK, WARN, BLOCK |
| `governance_status` | enum | Yes | OK, WARN, BLOCK |
| `governance_alignment` | enum | Yes | ALIGNED, TENSION, DIVERGENT |
| `safety_status` | enum | Yes | OK, WARN, BLOCK |
| `radar_status` | enum | No | OK, WARN, BLOCK |
| `conflict` | boolean | Yes | Conflict flag |
| `reasons` | array | Yes | Prefixed reason strings |
| `safe_for_policy_update` | boolean | Yes | Policy update eligibility |
| `safe_for_promotion` | boolean | Yes | Promotion eligibility |

### 7.2 Validation

All Replay Safety governance signals MUST validate against the schema before:
- Writing to evidence pack
- Sending to director panel
- Recording in governance history

---

## Appendix A: Reason Prefix Convention

All reasons in the governance signal MUST be prefixed:

| Prefix | Source | Example |
|--------|--------|---------|
| `[Safety]` | Safety evaluation | `[Safety] Hash mismatch detected` |
| `[Radar]` | Radar evaluation | `[Radar] Drift threshold exceeded` |
| `[CONFLICT]` | Alignment conflict | `[CONFLICT] Safety and Radar diverge - manual review required` |

**Deduplication Rule:** If a reason is already prefixed, do not double-prefix.

---

## Appendix B: Implementation Reference

**Source Files:**
- `experiments/u2/replay_safety.py` — Core implementation
- `tests/test_replay_safety_governance_signal.py` — Test suite

**Key Functions:**
- `to_governance_signal_for_replay_safety()` — Signal fusion
- `build_replay_safety_governance_view()` — Governance view builder
- `build_replay_safety_director_panel()` — Director panel builder
- `summarize_replay_safety_for_evidence()` — Evidence pack summarizer

---

## Appendix C: P5 Real Replay Expectations

> **Status**: SPECIFICATION DRAFT
> **Scope**: Defines stability contracts for P5 real-telemetry replay signals
> **Author**: CLAUDE A (Replay Governance Layer)

### C.1 Overview

Phase P5 introduces **real-telemetry replay** — the transition from synthetic (P3) and shadow-mode (P4) validation to actual production replay signals. This appendix specifies which fields must remain stable across the P3→P5 transition and which are permitted to vary.

### C.2 Field Stability Contract

#### C.2.1 MUST BE STABLE (Invariant Across P3/P4/P5)

These fields MUST have identical semantics and value ranges across all phases:

| Field | Type | Stability Requirement |
|-------|------|----------------------|
| `status` | enum(ok,warn,block) | **INVARIANT** — Same collapse rules apply |
| `governance_alignment` | enum(aligned,tension,divergent) | **INVARIANT** — Same alignment semantics |
| `conflict` | boolean | **INVARIANT** — Same conflict detection logic |
| `safe_for_policy_update` | boolean | **INVARIANT** — Same eligibility rules |
| `safe_for_promotion` | boolean | **INVARIANT** — Same eligibility rules |
| `signal_type` | string | **INVARIANT** — Always "replay_safety" |
| `schema_version` | string | **INVARIANT** — Same schema contract |

**Rationale**: External auditors and governance systems must be able to compare signals across phases without semantic drift. The collapse rules (Section 3) are phase-invariant.

#### C.2.2 ALLOWED TO VARY (Phase-Dependent)

These fields may exhibit different value distributions between phases:

| Field | P3 Mock | P4 Shadow | P5 Real | Variance Rationale |
|-------|---------|-----------|---------|-------------------|
| `determinism_rate` | 0.95-1.0 (high) | 0.85-1.0 (moderate) | 0.70-1.0 (real variance) | Real systems have more variance |
| `critical_incident_rate` | 0.0-0.1 (low) | 0.0-0.2 (moderate) | 0.0-0.5 (real incidents) | Real incidents occur |
| `hot_fingerprints_count` | 0-2 (minimal) | 0-5 (moderate) | 0-20 (full range) | Real drift detection active |
| `reasons` (count) | 0-3 | 0-5 | 0-10 | More diagnostic detail in P5 |

**Band Definitions for `determinism_rate`**:

| Phase | GREEN Band | YELLOW Band | RED Band |
|-------|------------|-------------|----------|
| P3 | >= 0.95 | 0.85-0.95 | < 0.85 |
| P4 | >= 0.90 | 0.75-0.90 | < 0.75 |
| P5 | >= 0.85 | 0.70-0.85 | < 0.70 |

**Note**: The relaxed bands in P5 reflect real-world variance, not reduced safety standards. The collapse rules remain unchanged.

#### C.2.3 P5-ONLY FIELDS (Optional Extensions)

These fields MAY appear only in P5 signals and are ignored in P3/P4:

| Field | Type | P5 Meaning |
|-------|------|------------|
| `telemetry_source` | string | "real" (vs "synthetic" or "shadow") |
| `production_run_id` | string | Unique identifier for production run |
| `replay_latency_ms` | number | Wall-clock latency of replay verification |
| `external_correlation_id` | string | Link to external monitoring systems |

### C.3 Summary Format Stability

#### C.3.1 First Light Status Summary

The `first_light_status` report MUST include replay safety with this stable structure:

```json
{
  "replay_safety": {
    "status": "ok|warn|block",              // INVARIANT
    "determinism_rate": 0.0-1.0,            // PHASE-DEPENDENT BANDS
    "critical_incident_rate": 0.0-1.0,      // PHASE-DEPENDENT
    "hot_fingerprints_count": 0-N,          // PHASE-DEPENDENT
    "governance_alignment": "aligned|..."   // INVARIANT
  }
}
```

#### C.3.2 First Light Alignment View

The `first_light_alignment` report MUST include replay safety with this stable structure:

```json
{
  "replay": {
    "status": "ok|warn|block",              // INVARIANT
    "alignment": "aligned|tension|divergent", // INVARIANT
    "conflict": true|false,                 // INVARIANT
    "top_reasons": ["..."]                  // CONTENT VARIES, FORMAT STABLE
  }
}
```

### C.4 Auditor Verification Points

When comparing P3 mock vs P5 real signals, auditors SHOULD verify:

| Checkpoint | P3 Expectation | P5 Expectation | Delta Allowed |
|------------|----------------|----------------|---------------|
| Status collapse rules | Per Section 3 | Per Section 3 | NONE |
| Alignment semantics | Per Section 2 | Per Section 2 | NONE |
| Reason prefixes | `[Safety]`, `[Radar]`, `[CONFLICT]` | Same | NONE |
| Determinism rate | >= 0.95 typical | >= 0.85 typical | Up to 0.10 |
| Incident rate | Near zero | Non-zero acceptable | Expected |

### C.5 Transition Ceremony

When promoting from P4 Shadow to P5 Real:

1. **Pre-Flight Check**: Verify P4 shadow signals achieved >= 95% alignment with P3 baselines
2. **Canary Period**: Run P5 real in parallel with P4 shadow for N cycles
3. **Divergence Threshold**: If P5 vs P4 divergence exceeds 10% on status field, halt and investigate
4. **Sign-Off**: Manual approval required before P5 signals are authoritative

---

## Appendix D: Replay Safety for External Auditors

> **Audience**: External verifiers, compliance reviewers, third-party auditors
> **Scope**: How to interpret replay safety signals in First Light reports
> **SHADOW MODE**: All observations are advisory; no real-time gating occurs

### D.1 Purpose

This appendix explains how external auditors should interpret the `replay_safety` signal that appears in:
- `first_light_status` reports
- `first_light_alignment` views
- Evidence packs for whitepaper attestation

### D.2 What is Replay Safety?

Replay Safety verifies that system behavior is **deterministic** — that identical inputs produce identical outputs across multiple replay executions. This is foundational for:

- **Auditability**: If we can't replay, we can't verify
- **Governance**: Non-deterministic systems can't be governed reliably
- **Evidence**: Whitepaper claims require reproducible results

### D.3 Where to Find Replay Safety in First Light

#### D.3.1 In `first_light_status`

Look for the `replay_governance` section:

```json
{
  "first_light_status": {
    "replay_governance": {
      "status": "ok",
      "determinism_rate": 0.97,
      "critical_incident_rate": 0.02,
      "hot_fingerprints_count": 1,
      "governance_alignment": "aligned"
    }
  }
}
```

**Reading Guide**:
| Field | What It Means | Good Value | Concern Threshold |
|-------|---------------|------------|-------------------|
| `status` | Overall health | "ok" | "warn" or "block" |
| `determinism_rate` | Fraction of replays that matched | >= 0.85 | < 0.70 |
| `critical_incident_rate` | Fraction of runs with critical issues | < 0.10 | >= 0.20 |
| `hot_fingerprints_count` | Number of drift indicators detected | 0-3 | > 10 |
| `governance_alignment` | Agreement between safety and radar | "aligned" | "divergent" |

#### D.3.2 In `first_light_alignment`

Look for the `replay` entry in the alignment view:

```json
{
  "first_light_alignment": {
    "signals": {
      "replay": {
        "status": "ok",
        "alignment": "aligned",
        "conflict": false,
        "top_reasons": []
      }
    }
  }
}
```

**Reading Guide**:
| Field | What It Means | Good Value | Concern Value |
|-------|---------------|------------|---------------|
| `status` | Collapsed status | "ok" | "block" |
| `alignment` | Safety-radar agreement | "aligned" | "divergent" |
| `conflict` | Explicit conflict flag | false | true |
| `top_reasons` | Human-readable issues | Empty list | Non-empty list |

### D.4 Interpreting Status Values

#### D.4.1 OK (Green)

**Meaning**: Replay verification passed. System behavior is deterministic within acceptable bounds.

**Auditor Action**: No action required. This signal supports governance claims.

#### D.4.2 WARN (Yellow)

**Meaning**: Minor concerns detected. System is likely deterministic but with caveats.

**Auditor Action**: Review the `top_reasons` list. Common causes:
- Transient timing variations
- Minor configuration drift
- Edge cases in test coverage

**Question to Ask**: "Is the underlying cause understood and documented?"

#### D.4.3 BLOCK (Red)

**Meaning**: Significant concern. Replay verification failed or safety-radar conflict detected.

**Auditor Action**: This is a material finding. Steps:
1. Review `top_reasons` for root cause
2. Check if `conflict` is true (safety-radar disagreement)
3. Request incident report from system operators
4. Do NOT rely on governance claims until resolved

**Question to Ask**: "What was the resolution? Was the underlying issue fixed?"

### D.5 Understanding Governance Alignment

The `governance_alignment` field indicates agreement between two independent evaluators:

| Alignment | Meaning | Auditor Interpretation |
|-----------|---------|----------------------|
| **aligned** | Safety and Radar agree | Consensus — high confidence |
| **tension** | Minor disagreement | Review recommended — moderate confidence |
| **divergent** | Major disagreement | Conflict — requires investigation |

**Why This Matters**: Independent agreement increases confidence. If Safety says "OK" and Radar says "OK", that's stronger than either alone. If they disagree, something may be wrong.

### D.6 Reason Prefixes

Reasons in the `top_reasons` list are prefixed to indicate their source:

| Prefix | Source | Example |
|--------|--------|---------|
| (none) | General | "Hash mismatch detected" |
| `[CONFLICT]` | Safety-Radar disagreement | "[CONFLICT] Safety and Radar alignment is DIVERGENT" |

**Note**: In the GGFL alignment view, the `[Replay]` prefix is stripped for readability. The underlying signal retains full prefixes.

### D.7 SHADOW MODE Disclaimer

**All replay safety signals operate in SHADOW MODE:**

- Signals are **advisory only**
- No real-time gating or blocking occurs
- Control flow is never influenced by these signals
- This is purely for **observability and evidence collection**

**Implication for Auditors**: These signals document what *would have happened* if the system were gated. They do not prove that the system *was* gated.

### D.8 Evidence Chain

When replay safety appears in whitepaper evidence:

```
First Light Run
      │
      ▼
┌─────────────────────┐
│ Replay Verification │
│ (determinism check) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Governance Signal   │
│ (fused status)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ First Light Status  │
│ (replay_governance) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evidence Pack       │
│ (for whitepaper)    │
└─────────────────────┘
```

### D.9 Questions Auditors Should Ask

1. **Coverage**: "What fraction of system behavior is covered by replay verification?"
2. **Consistency**: "Are replay results consistent across multiple runs?"
3. **Resolution**: "When WARN or BLOCK occurred, what was the resolution?"
4. **Independence**: "Are Safety and Radar truly independent evaluators?"
5. **SHADOW Mode**: "When will this transition from advisory to enforced?"

### D.10 Summary

Replay Safety provides assurance that MathLedger's behavior is reproducible and auditable. External auditors should:

1. Look for `replay_governance` in First Light status
2. Check that `status` is "ok" and `alignment` is "aligned"
3. Investigate any "warn" or "block" signals
4. Remember that SHADOW mode means advisory-only
5. Use these signals as evidence of governance posture, not enforcement

---

*Document Version: 1.2.0*
*Last Updated: 2025-12-11*
*Status: Reference*
*Change Log:*
- *1.2.0: Added Section 4.5 Divergence Companion for P5 co-interpretation*
- *1.1.0: Added Appendix C (P5 Real Replay Expectations) and D (External Auditors)*
- *1.0.0: Initial specification*
