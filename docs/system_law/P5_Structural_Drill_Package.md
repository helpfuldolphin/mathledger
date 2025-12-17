# P5 Structural Drill Package: STRUCTURAL_BREAK Event Simulation

---

> **STRATCOM: CLAUDE G — P5 STRUCTURAL DRILL**
>
> This document specifies synthetic scenarios, escalation advisory prototypes, and operator
> runbook fragments for exercising the STRUCTURAL_BREAK detection pipeline.
>
> **Status**: Drill Package Ready
> **Version**: 1.0.0
> **Date**: 2025-12-11
> **Prerequisites**: Structural_Cohesion_PhaseX.md v1.1.0, Phase_X_P5_Implementation_Blueprint.md

---

## Table of Contents

1. [STRUCTURAL_BREAK Scenario Script](#1-structural_break-scenario-script)
2. [Escalation Advisory Prototypes](#2-escalation-advisory-prototypes)
3. [Operator Runbook Fragment](#3-operator-runbook-fragment)
4. [Smoke-Test Readiness Checklist](#4-smoke-test-readiness-checklist)

---

## 1. STRUCTURAL_BREAK Scenario Script

### 1.1 Scenario Overview

This synthetic scenario exercises the full STRUCTURAL_BREAK detection pipeline:

| Phase | Cycle Range | Event | Expected Result |
|-------|-------------|-------|-----------------|
| **Baseline** | 1-100 | Normal operation | Pattern=NONE, SCS=1.0 |
| **Tension Onset** | 101-150 | SI-006 omega exit | Pattern=DRIFT, SCS=0.88 |
| **Structural Break** | 151 | SI-001 cycle injected | Pattern=STRUCTURAL_BREAK, SCS=0.0 |
| **Escalation Active** | 152-200 | Continued operation | Severity=CRITICAL, streak=1→49 |
| **Recovery (Simulated)** | 201+ | DAG repaired | Pattern=RECOVERY, SCS→1.0 |

### 1.2 Synthetic Telemetry Injection Points

```python
# SPEC-ONLY: Scenario injection configuration
STRUCTURAL_BREAK_SCENARIO = {
    "scenario_id": "DRILL-SB-001",
    "description": "SI-001 Cycle Injection → STRUCTURAL_BREAK",
    "shadow_mode": True,  # MANDATORY

    "phases": [
        {
            "name": "baseline",
            "cycle_range": [1, 100],
            "dag_state": {
                "is_acyclic": True,
                "node_count": 150,
                "violations": []
            },
            "topology_state": {
                "H_current": 0.12,
                "rho_current": 0.85,
                "omega_status": "INSIDE"
            },
            "ht_state": {
                "anchors_total": 10,
                "anchors_verified": 10
            },
            "expected": {
                "pattern": "NONE",
                "cohesion_score": 1.0,
                "severity": "INFO"
            }
        },
        {
            "name": "tension_onset",
            "cycle_range": [101, 150],
            "dag_state": {
                "is_acyclic": True,
                "node_count": 200,
                "violations": []
            },
            "topology_state": {
                "H_current": 0.18,
                "rho_current": 0.72,
                "omega_status": "OUTSIDE",
                "omega_exit_cycles": 45
            },
            "ht_state": {
                "anchors_total": 10,
                "anchors_verified": 10
            },
            "expected": {
                "pattern": "DRIFT",
                "cohesion_score": 0.88,
                "severity": "WARN",
                "violations": ["SI-006"]
            }
        },
        {
            "name": "structural_break",
            "cycle_range": [151, 151],
            "dag_state": {
                "is_acyclic": False,  # CYCLE INJECTED
                "node_count": 201,
                "cycle_nodes": ["node_42", "node_87", "node_42"],
                "violations": ["SI-001"]
            },
            "topology_state": {
                "H_current": 0.22,
                "rho_current": 0.65,
                "omega_status": "OUTSIDE"
            },
            "ht_state": {
                "anchors_total": 10,
                "anchors_verified": 10
            },
            "expected": {
                "pattern": "STRUCTURAL_BREAK",
                "structural_cause": "SI-001_CYCLE",
                "cohesion_score": 0.0,
                "severity": "CRITICAL",
                "admissible": False
            }
        },
        {
            "name": "escalation_active",
            "cycle_range": [152, 200],
            "dag_state": {
                "is_acyclic": False,  # Cycle persists
                "violations": ["SI-001"]
            },
            "expected": {
                "pattern": "STRUCTURAL_BREAK",
                "severity": "CRITICAL",
                "streak_min": 1,
                "streak_max": 49
            }
        }
    ]
}
```

### 1.3 Expected Signal Flow

```
Cycle 151: SI-001 Violation Detected
    │
    ▼
emit_structural_signal(dag={is_acyclic: False, ...})
    │
    ├── dag_status = "CONFLICT"
    ├── combined_severity = "CONFLICT"
    ├── cohesion_score = 0.0  (blocking invariant)
    ├── admissible = False
    └── violations = [{invariant: "SI-001", severity: "CONFLICT", blocking: True}]
    │
    ▼
DivergencePatternClassifier.classify(structural_signal=signal)
    │
    ├── Detects admissible=False
    ├── Returns pattern = STRUCTURAL_BREAK
    ├── structural_cause = "SI-001_CYCLE"
    └── severity = CRITICAL (auto-escalated)
    │
    ▼
build_escalation_advisory(severity="INFO", structural_signal=signal)
    │
    ├── would_escalate = True
    ├── original_severity = "INFO"
    ├── escalated_severity = "CRITICAL"
    └── reason = "SI-001 blocking invariant violated"
    │
    ▼
Evidence Attachment + Logging (SHADOW MODE)
```

### 1.4 Replay/Topology/Structure Field Behavior

| Field | Cycle 100 (Baseline) | Cycle 151 (Break) | Cycle 175 (Streak) |
|-------|---------------------|-------------------|-------------------|
| `replay.cycle` | 100 | 151 | 175 |
| `replay.telemetry_valid` | True | True | True |
| `topology.pattern` | "NONE" | "STRUCTURAL_BREAK" | "STRUCTURAL_BREAK" |
| `topology.pattern_confidence` | 0.95 | 1.0 | 1.0 |
| `structure.dag_status` | "CONSISTENT" | "CONFLICT" | "CONFLICT" |
| `structure.cohesion_score` | 1.0 | 0.0 | 0.0 |
| `structure.admissible` | True | False | False |
| `structure.violations` | [] | ["SI-001"] | ["SI-001"] |
| `divergence.severity` | "INFO" | "CRITICAL" | "CRITICAL" |
| `divergence.severity_escalated` | False | True | True |
| `divergence.original_severity` | "INFO" | "INFO" | "INFO" |
| `escalation.streak` | 0 | 1 | 25 |

---

## 2. Escalation Advisory Prototypes

### 2.1 Single Break Event

```json
// SPEC-ONLY: Escalation advisory for single STRUCTURAL_BREAK
{
    "advisory_id": "esc_adv_sb_single_001",
    "advisory_type": "SHADOW_ADVISORY",
    "timestamp": "2025-12-11T14:30:00.151Z",
    "cycle": 151,

    "escalation": {
        "would_escalate": true,
        "original_severity": "INFO",
        "escalated_severity": "CRITICAL",
        "reason": "CONFLICT-class structural signal detected",
        "escalation_rule": "CONFLICT → any severity becomes CRITICAL"
    },

    "structural_context": {
        "combined_severity": "CONFLICT",
        "cohesion_score": 0.0,
        "cohesion_degraded": true,
        "blocking_invariants_violated": ["SI-001"],
        "non_blocking_violations": []
    },

    "pattern_context": {
        "detected_pattern": "STRUCTURAL_BREAK",
        "structural_cause": "SI-001_CYCLE",
        "pattern_confidence": 1.0,
        "supersedes_pattern": "DRIFT"
    },

    "streak_info": {
        "current_streak": 1,
        "streak_start_cycle": 151,
        "is_new_break": true,
        "is_repeated_break": false
    },

    "operator_guidance": {
        "priority": "P0",
        "action_class": "INVESTIGATE_IMMEDIATELY",
        "summary": "DAG cycle detected. Proof lineage compromised."
    },

    "shadow_mode_notice": "This advisory is OBSERVATIONAL ONLY. No enforcement action taken."
}
```

### 2.2 Repeated Break (Streak >= 2)

```json
// SPEC-ONLY: Escalation advisory for repeated STRUCTURAL_BREAK (streak >= 2)
{
    "advisory_id": "esc_adv_sb_streak_025",
    "advisory_type": "SHADOW_ADVISORY",
    "timestamp": "2025-12-11T14:30:00.175Z",
    "cycle": 175,

    "escalation": {
        "would_escalate": true,
        "original_severity": "INFO",
        "escalated_severity": "CRITICAL",
        "reason": "CONFLICT-class structural signal persists",
        "escalation_rule": "CONFLICT → any severity becomes CRITICAL",
        "sustained_escalation": true
    },

    "structural_context": {
        "combined_severity": "CONFLICT",
        "cohesion_score": 0.0,
        "cohesion_degraded": true,
        "blocking_invariants_violated": ["SI-001"],
        "non_blocking_violations": [],
        "degradation_duration_cycles": 25
    },

    "pattern_context": {
        "detected_pattern": "STRUCTURAL_BREAK",
        "structural_cause": "SI-001_CYCLE",
        "pattern_confidence": 1.0,
        "pattern_stable": true
    },

    "streak_info": {
        "current_streak": 25,
        "streak_start_cycle": 151,
        "is_new_break": false,
        "is_repeated_break": true,
        "streak_severity": "CRITICAL",
        "streak_escalation_threshold": 2,
        "streak_exceeded_threshold": true
    },

    "operator_guidance": {
        "priority": "P0",
        "action_class": "SUSTAINED_BREAK",
        "summary": "STRUCTURAL_BREAK persists for 25 cycles. Requires manual DAG repair.",
        "escalation_note": "Streak >= 2 triggers sustained break protocol"
    },

    "correlation": {
        "structural_to_divergence": 1.0,
        "comment": "Divergence directly caused by structural failure"
    },

    "shadow_mode_notice": "This advisory is OBSERVATIONAL ONLY. No enforcement action taken."
}
```

---

## 3. Operator Runbook Fragment

### STRUCTURAL_BREAK + Cohesion < 0.8 Response Procedure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RUNBOOK: STRUCTURAL_BREAK with Cohesion Degradation                        │
│  Trigger: Pattern=STRUCTURAL_BREAK AND SCS < 0.8                            │
│  Priority: P0 — Immediate Response Required                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. VERIFY the alert is not a false positive:                               │
│     → Check `structure.violations[]` for specific invariant (SI-001/SI-010) │
│     → Confirm `structure.admissible == False`                               │
│                                                                             │
│  2. IDENTIFY the structural cause:                                          │
│     → SI-001 (DAG Cycle): Run `dag_verify --check-acyclicity`               │
│     → SI-010 (Anchor Fail): Run `ht_verify --check-anchors`                 │
│                                                                             │
│  3. HALT new derivations (manual hold):                                     │
│     → Do NOT feed new proofs into compromised DAG                           │
│     → Shadow mode continues observation automatically                       │
│                                                                             │
│  4. ISOLATE affected subgraph:                                              │
│     → Identify cycle nodes from `dag_state.cycle_nodes[]`                   │
│     → Mark subgraph as quarantined in ledger                                │
│                                                                             │
│  5. REPAIR structural violation:                                            │
│     → For SI-001: Remove or re-derive conflicting edges                     │
│     → For SI-010: Re-verify anchor proofs from known-good state             │
│                                                                             │
│  6. VALIDATE repair:                                                        │
│     → Run `emit_structural_signal()` on repaired state                      │
│     → Confirm `admissible == True` and `cohesion_score >= 0.95`             │
│                                                                             │
│  7. RESUME normal operation:                                                │
│     → Clear manual hold                                                     │
│     → Monitor next 100 cycles for recurrence                                │
│                                                                             │
│  8. POST-INCIDENT: Log timeline in incident tracker:                        │
│     → Break detection cycle, repair cycle, total duration                   │
│     → Root cause analysis (how did cycle/anchor fail enter system?)         │
│                                                                             │
│  9. ESCALATE if streak > 50 OR repair fails:                                │
│     → Contact system architect                                              │
│     → Consider full DAG rebuild from last known-good snapshot               │
│                                                                             │
│  10. SHADOW MODE REMINDER:                                                  │
│      → All actions are manual; system will not auto-abort                   │
│      → Continue monitoring; STRUCTURAL_BREAK logged but not enforced        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Smoke-Test Readiness Checklist

Before exercising the STRUCTURAL_BREAK pipeline in a P5 drill, verify the following:

### 4.1 Infrastructure Readiness

- [ ] **Database Available**: PostgreSQL running, `mathledger` database accessible
- [ ] **Redis Available**: Redis running, queue responsive
- [ ] **Test Isolation**: Drill runs against test/shadow database, not production
- [ ] **Logging Configured**: JSONL divergence logs directed to drill output directory

### 4.2 Module Availability

- [ ] **`emit_structural_signal()`**: Function importable from `backend.dag.invariant_guard`
- [ ] **`build_structural_cohesion_tile()`**: Function returns valid tile dict
- [ ] **`build_escalation_advisory()`**: Function returns valid advisory dict
- [ ] **`apply_structural_severity_escalation()`**: Function modifies DivergenceSnapshot correctly
- [ ] **`DivergencePatternClassifier`**: Can accept `structural_signal` parameter (stub OK for drill)

### 4.3 Test Data Preparation

- [ ] **Baseline DAG**: Clean acyclic DAG with 150+ nodes loaded
- [ ] **Injection Script**: Cycle injection script ready (creates SI-001 violation)
- [ ] **Anchor Corruption Script**: Anchor invalidation script ready (creates SI-010 violation)
- [ ] **Rollback Script**: Can restore clean state after drill

### 4.4 Scenario Execution Verification

- [ ] **Phase 1 (Baseline)**: Confirm SCS=1.0, Pattern=NONE for cycles 1-100
- [ ] **Phase 2 (Tension)**: Confirm SCS drops, Pattern=DRIFT for cycles 101-150
- [ ] **Phase 3 (Break)**: Confirm Pattern=STRUCTURAL_BREAK at cycle 151
- [ ] **Phase 4 (Streak)**: Confirm streak increments, severity=CRITICAL persists
- [ ] **Phase 5 (Recovery)**: Confirm SCS recovers after repair

### 4.5 Output Validation

- [ ] **Divergence Log**: Contains STRUCTURAL_BREAK entries with `structural_cause`
- [ ] **Escalation Advisory Log**: Contains advisory JSON for single and streak events
- [ ] **Console Tile**: Shows CRITICAL status with violation breakdown
- [ ] **Evidence Attachment**: `evidence["governance"]["structure"]` populated correctly

### 4.6 Shadow Mode Invariants

- [ ] **No Aborts**: System continues running despite STRUCTURAL_BREAK
- [ ] **No Enforcement**: No automated quarantine or halt triggered
- [ ] **Logging Only**: All actions are observational/advisory
- [ ] **Original State Preserved**: Structural signal does not mutate DAG/HT state

### 4.7 Regression Guards

- [ ] **Existing Tests Pass**: `pytest tests/dag/test_structural_governance.py` — 41 tests pass
- [ ] **No New Failures**: Full test suite shows no regressions
- [ ] **Schema Compliance**: All output JSON validates against respective schemas

---

## Appendix A: Quick Reference Commands

```bash
# Verify structural governance module
python -c "from backend.dag.invariant_guard import emit_structural_signal; print('OK')"

# Run structural governance tests
pytest tests/dag/test_structural_governance.py -v

# Validate escalation advisory schema (manual)
python -c "
from backend.dag.invariant_guard import build_escalation_advisory
adv = build_escalation_advisory('INFO', {'combined_severity': 'CONFLICT', 'cohesion_score': 0.0})
print('would_escalate:', adv['would_escalate'])
print('escalated_severity:', adv['escalated_severity'])
"

# Check SHADOW MODE compliance
grep -r "SHADOW" backend/dag/invariant_guard.py | head -5
```

---

**CLAUDE G: P5 Structural Drill Ready.**

---

*Document Version: 1.0.0*
*Classification: Drill Package — SHADOW MODE*
*Last Updated: 2025-12-11*
